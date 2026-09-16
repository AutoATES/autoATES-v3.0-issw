"""
driverStack.py - Build an aligned, multi-scenario driver stack for ATES (v3.0).

Every layer is co-registered to the DEM grid (identical shape / transform / crs)
so that any downstream classifier - the v2.10-style rules classifier, a fuzzy
classifier, or an ML model - can consume the layers as a pixel-aligned feature
stack without worrying about extent or resolution mismatches.

Design principles
-----------------
* **Alignment is guaranteed here, once.** com4FlowPy's merged output extent
  depends on where avalanches actually reached, so the frequent and extreme
  rasters can have different extents/origins. Each input is reprojected onto the
  DEM grid; cells outside a runout layer's extent are filled with 0 ("not
  reached"), never NODATA, so fusing scenarios never blanks terrain.
* **Magnitude consistency.** zDelta is scaled to max_z (0-100) and flux is a
  physical contributing area (m^2); both are resolution- and scenario-invariant,
  so the same classification thresholds apply to every scenario. The travel-angle
  term in alpha_zdelta is normalised by a FIXED reference angle (not the data
  max) to preserve that invariance.

The stack is scenario-keyed: terrain/forest layers are shared, while
``alpha_zdelta``, ``flux_area``, ``reach_mask`` and ``runout_prox`` are produced
per scenario (e.g. ``frequent`` and ``extreme``).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from osgeo import gdal
import scipy.ndimage

logger = logging.getLogger(__name__)

NODATA = -9999
# Fixed reference angle (deg) for normalising the flow-path travel angle into a
# 0-100 score. Using a constant (rather than the per-run data max) keeps
# alpha_zdelta consistent across scenarios and DEM resolutions.
FP_REF_ANGLE = 90.0


@dataclass
class ScenarioLayers:
    """Per-scenario runout-derived driver layers, all on the DEM grid."""
    alpha_zdelta: np.ndarray      # 0-100, consistent reach/energy score (int16)
    flux_area: np.ndarray         # contributing area (m^2) where fp >= cutoff (float32)
    fp_travel_angle: np.ndarray   # flow-path travel angle (deg), 0 where unreached (float32)
    reach_mask: np.ndarray        # 1 where avalanche reached, else 0 (int16)
    runout_prox: np.ndarray       # geographic distance (m) to nearest reached cell (float32)
    pra: np.ndarray               # 1 in release area, else 0 (int16)
    forest_interaction: np.ndarray = None  # com4FlowPy forestInteraction, 0 where none (float32)
    pra_continuous: np.ndarray = None      # PRA susceptibility 0-1, 0 where none (float32)


@dataclass
class DriverStack:
    """Aligned ATES feature stack. Terrain/forest shared, runout per scenario."""
    profile: dict
    transform: object
    cell_area: float              # m^2 per cell
    slope: np.ndarray             # degrees (int16, NODATA preserved)
    slope_smooth: np.ndarray      # win_size-smoothed slope (float32)
    forest_pct: np.ndarray        # canopy cover % (int16, NODATA preserved)
    valid_mask: np.ndarray        # True where DEM (terrain) is valid
    scenarios: dict = field(default_factory=dict)  # name -> ScenarioLayers


# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------
def _ref_profile(dem_path: Path) -> dict:
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="float32", nodata=NODATA)
    return profile


def _align(src_path: Path, ref: dict, fill: float,
           resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Reproject/pad a raster onto the reference (DEM) grid.

    Cells with no source coverage are set to ``fill`` (0 for runout layers so
    unreached terrain reads as "not reached", NODATA for terrain layers).
    """
    dst = np.full((ref["height"], ref["width"]), fill, dtype="float32")
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype("float32")
        src_nodata = src.nodata
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            dst_nodata=fill,
            resampling=resampling,
        )
    return dst


# ---------------------------------------------------------------------------
# Terrain layers (scenario-independent)
# ---------------------------------------------------------------------------
def _build_terrain(dem_path: Path, ates_dir: Path, win_size: int):
    slope_tif = ates_dir / "_slope_raw.tif"
    gdal.DEMProcessing(str(slope_tif), str(dem_path), "slope")
    with rasterio.open(slope_tif) as src:
        slope = src.read(1)
        s_nodata = src.nodata if src.nodata is not None else NODATA
    slope = np.where(slope == s_nodata, NODATA, slope).astype(np.int16)
    valid = slope != NODATA
    slope_nd = np.where(slope < 0, 0, slope).astype(np.float32)
    slope_smooth = scipy.ndimage.uniform_filter(slope_nd, size=win_size, mode="nearest")
    return slope, slope_smooth, valid


# ---------------------------------------------------------------------------
# Per-scenario runout layers
# ---------------------------------------------------------------------------
def _proximity_from_mask(reach_mask: np.ndarray, ref: dict, ates_dir: Path,
                         tag: str) -> np.ndarray:
    """GDAL geographic proximity from a binary reach mask."""
    mask_path = ates_dir / f"_reach_mask_{tag}.tif"
    p = ref.copy(); p.update(dtype="int16", nodata=NODATA)
    with rasterio.open(mask_path, "w", **p) as dst:
        dst.write(reach_mask.astype("int16"), 1)

    prox_path = ates_dir / f"_runout_prox_{tag}.tif"
    ds = gdal.Open(str(mask_path))
    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(str(prox_path), ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(float(NODATA))
    gdal.ComputeProximity(ds.GetRasterBand(1), band, ["DISTUNITS=GEO"])
    band = None; out_ds = None; ds = None
    with rasterio.open(prox_path) as src:
        return src.read(1).astype("float32")


def _build_scenario(name: str, runout: dict, pra: dict, ref: dict,
                    cell_area: float, ates_dir: Path, params: dict) -> ScenarioLayers:
    max_z = params["max_z"]
    flux_cutoff = params["flux_cutoff"]

    # zDelta -> 0-100 reach/energy score (resolution/scenario consistent)
    zdelta = _align(runout["zDelta"], ref, fill=0.0)
    zdelta_scale = np.clip(zdelta / max_z * 100.0, 0, 100)

    # routFluxSum -> physical contributing area (m^2)
    routflux = _align(runout["routFluxSum"], ref, fill=0.0)
    contrib_area = routflux * cell_area

    # fpTravelAngleMax -> reach mask + fixed-normalised angle score
    fp = _align(runout["fpTravelAngleMax"], ref, fill=0.0)
    reach_mask = (fp > 0).astype(np.int16)
    fp_scale = np.clip(fp / FP_REF_ANGLE * 100.0, 0, 100)

    # alpha_zdelta = mean of the two consistent 0-100 scores
    alpha_zdelta = ((zdelta_scale + fp_scale) / 2.0).astype(np.int16)

    # flux_area: contributing area only where travel angle clears the cutoff
    flux_area = np.where(fp < flux_cutoff, 0.0, contrib_area).astype(np.float32)

    runout_prox = _proximity_from_mask(reach_mask, ref, ates_dir, name)

    # PRA release-area mask (1/0) aligned to grid
    pra_arr = _align(pra["sieve"], ref, fill=0.0)
    pra_mask = (pra_arr >= 0.5).astype(np.int16)

    # Optional: com4FlowPy forestInteraction layer (forested terrain the flow
    # passed through - a signal for forest terrain traps / low-class separation)
    forest_int = None
    if runout.get("forestInteraction") and Path(runout["forestInteraction"]).exists():
        forest_int = _align(runout["forestInteraction"], ref, fill=0.0).astype(np.float32)

    # Optional: continuous PRA susceptibility (slope+forest+wind baked in) - the
    # base layer for the PRA-base ATES mode.
    pra_cont = None
    if pra.get("continuous") and Path(pra["continuous"]).exists():
        pra_cont = _align(pra["continuous"], ref, fill=0.0).astype(np.float32)

    return ScenarioLayers(
        alpha_zdelta=alpha_zdelta,
        flux_area=flux_area,
        fp_travel_angle=fp.astype(np.float32),
        reach_mask=reach_mask,
        runout_prox=runout_prox,
        pra=pra_mask,
        forest_interaction=forest_int,
        pra_continuous=pra_cont,
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------
def build_driver_stack(dem_path: Path, forest_path: Path,
                       runout_by_scenario: dict, pra_by_scenario: dict,
                       ates_dir: Path, params: dict) -> DriverStack:
    """Assemble the aligned multi-scenario driver stack.

    Parameters
    ----------
    dem_path : Path
        Clipped DEM defining the reference grid.
    forest_path : Path
        Canopy-cover % raster.
    runout_by_scenario : dict
        ``{scenario: {"zDelta": p, "routFluxSum": p, "fpTravelAngleMax": p}}``.
    pra_by_scenario : dict
        ``{scenario: {"sieve": p}}`` release-area rasters per scenario.
    params : dict
        ATES parameters (needs max_z, flux_cutoff, win_size).
    """
    ates_dir = Path(ates_dir)
    ates_dir.mkdir(parents=True, exist_ok=True)
    ref = _ref_profile(dem_path)
    with rasterio.open(dem_path) as src:
        transform = src.transform
    cell_area = abs(transform.a) * abs(transform.e)

    slope, slope_smooth, valid = _build_terrain(dem_path, ates_dir, params["win_size"])
    forest_f = _align(forest_path, ref, fill=NODATA, resampling=Resampling.bilinear)
    forest_pct = np.where(forest_f == NODATA, NODATA, np.rint(forest_f)).astype(np.int16)

    stack = DriverStack(
        profile=ref, transform=transform, cell_area=cell_area,
        slope=slope, slope_smooth=slope_smooth, forest_pct=forest_pct,
        valid_mask=valid,
    )

    for name, runout in runout_by_scenario.items():
        pra = pra_by_scenario.get(name, {})
        if "sieve" not in pra:
            logger.warning("No PRA sieve for scenario '%s'; using zero release mask.", name)
            pra = {"sieve": runout["zDelta"]}  # aligned then thresholded ~ harmless 0/1
        stack.scenarios[name] = _build_scenario(
            name, runout, pra, ref, cell_area, ates_dir, params)
        logger.info("Driver stack: built scenario '%s' (reach cells=%d)",
                    name, int(stack.scenarios[name].reach_mask.sum()))

    return stack
