"""
atesCore.py - ATES terrain classification for autoATES v3.0 (multi-scenario).

The default model is dev8 (ates/dev8.py): split votes, then reach floors, then a
class-0 ceiling. `ATES.model = legacy` still runs `_classify` below.

Consumes the aligned, multi-scenario driver stack (driverStack.build_driver_stack)
and produces an ATES 0-4 classification. Restructured from the v2.10 port:

  * Driver layers are pre-aligned to the DEM grid in driverStack, so the pixel
    arithmetic here never has to reproject and "not reached" reads as 0, not
    NODATA (fixes the NODATA-union blanking).
  * Multi-scenario fusion: the EXTREME scenario defines the outer envelope
    (class 1 reach / proximity), the FREQUENT scenario drives the inner, higher
    classes (2/3) - keeping the extreme footprint from over-classifying terrain.
  * NEW overhead -> class 4: cells whose FREQUENT flow-path travel angle
    (fpTravelAngleMax) is >= class4_alpha (deg) sit directly below start zones
    and are classed Extreme. Plus the v2.10 smoothed-slope>sat34 rule.
  * The v2.10 forest/PRA lookup tables (_LUT_1/_LUT_2) are retained.
  * Generalization no longer mean-filters ordinal classes (which erased the
    sparse 3/4 classes); island removal is sieve-only.

Open / tunable hooks (flagged inline):
  * forest-for-extreme: dense forest still demotes class 4 via _LUT_1/_LUT_2.
  * forestInteraction: available per scenario in the stack but not yet wired
    into the 0/1/2 separation - see _classify TODO.
"""
from pathlib import Path
import logging
import configparser

import numpy as np
import rasterio

from autoates.comAutoATES.ates.driverStack import build_driver_stack, NODATA, _align
from autoates.comAutoATES.ates.atesPostProcess import postprocess_ates, get_postprocess_params
from autoates.comAutoATES.ates.dev8 import classify as dev8_classify

logger = logging.getLogger(__name__)


def _write(arr: np.ndarray, profile: dict, path: Path, dtype: str = "int16") -> Path:
    p = profile.copy()
    p.update(dtype=dtype, count=1, nodata=NODATA)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def _read(path: Path, dtype="int16") -> np.ndarray:
    with rasterio.open(path) as src:
        nodata = src.nodata if src.nodata is not None else NODATA
        arr = src.read(1)
    return np.where(arr == nodata, NODATA, arr).astype(dtype)


# ---------------------------------------------------------------------------
# v2.10 reclassification lookup tables (merge1 + forest + pra -> ATES class)
# ---------------------------------------------------------------------------
_LUT_1 = {
    0: 0, 10: 0, 11: 1, 12: 2, 13: 3, 14: 4,
    20: 0, 21: 1, 22: 2, 23: 2, 24: 3,
    30: 0, 31: 1, 32: 1, 33: 2, 34: 3,
    40: 0, 41: 1, 42: 1, 43: 1, 44: 2,
    110: 1, 111: 1, 112: 2, 113: 3, 114: 4,
    120: 1, 121: 1, 122: 2, 123: 2, 124: 3,
    130: 1, 131: 1, 132: 2, 133: 2, 134: 3,
    140: 1, 141: 1, 142: 2, 143: 2, 144: 2,
}
# second pass: class-4 demotion in forest (1004 -> 3)
_LUT_2 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    1000: 0, 1001: 1, 1002: 2, 1003: 3, 1004: 3,
}


def _apply_lut(arr: np.ndarray, lut: dict) -> np.ndarray:
    out = np.full_like(arr, NODATA)
    for src_val, dst_val in lut.items():
        out[arr == src_val] = dst_val
    out[arr == NODATA] = NODATA
    return out.astype(np.int16)


# ---------------------------------------------------------------------------
# Classification (multi-scenario fusion)
# ---------------------------------------------------------------------------
def _classify(stack, params, ates_dir, wind_deposits_path=None) -> Path:
    p = params
    valid = stack.valid_mask
    freq = stack.scenarios.get("frequent")
    ext = stack.scenarios.get("extreme")
    # Single-scenario fallback: use whichever exists for both roles.
    if ext is None:
        ext = freq
    if freq is None:
        freq = ext
    if freq is None:
        raise ValueError("ATES classify: no scenarios in driver stack")

    shape = stack.slope.shape

    # --- shared exposure drivers (scenario fusion: extreme=envelope, freq=inner) ---
    az = np.zeros(shape, dtype=np.int16)
    az[ext.alpha_zdelta >= p["az1"]] = 1
    az[freq.alpha_zdelta >= p["az2"]] = 2
    az[freq.alpha_zdelta >= p["az3"]] = 3

    rf = np.zeros(shape, dtype=np.int16)
    rf[ext.flux_area > p["rf1"]] = 1
    rf[freq.flux_area >= p["rf2"]] = 2
    rf[freq.flux_area >= p["rf3"]] = 3

    prox = np.zeros(shape, dtype=np.int16)
    prox[ext.runout_prox < p["class0_prox"]] = 1

    # overhead -> class 4 from FREQUENT travel angle (directly below start zones)
    overhead4 = freq.fp_travel_angle >= p["class4_alpha"]

    wind = None
    if wind_deposits_path and Path(wind_deposits_path).exists():
        w = _read(wind_deposits_path)
        wind = np.where(w == NODATA, 0, w)

    # NOTE: forestInteraction is intentionally NOT used in the rules classifier
    # (diagnostic: ~69% redundant with flux/runout, only ~0.3% marginal cells).
    # It stays aligned in the stack as a feature for the future ML/fuzzy approaches.

    base_layer = p.get("base_layer", "slope")
    if base_layer == "pra" and freq.pra_continuous is not None and ext.pra_continuous is not None:
        arr2 = _classify_pra_base(stack, freq, ext, az, rf, prox, overhead4, wind, p)
    else:
        if base_layer == "pra":
            logger.warning("base_layer=pra but no continuous PRA in stack; using slope base.")
            base_layer = "slope"
        arr2 = _classify_slope_base(stack, freq, ext, az, rf, prox, overhead4, wind, p)

    arr2[~valid] = NODATA
    merge2_path = _write(arr2, stack.profile, ates_dir / "merge_2.tif")
    logger.info("ATES (%s base) merge_2 classes: %s", base_layer, np.unique(arr2[valid]))
    return merge2_path


def _classify_slope_base(stack, freq, ext, az, rf, prox, overhead4, wind, p):
    """v2.10-style: slope SAT base (0-4) + forest/PRA lookup tables."""
    shape = stack.slope.shape
    slope = stack.slope
    sc = np.zeros(shape, dtype=np.int16)
    sc[(slope > p["sat01"]) & (slope <= p["sat12"])] = 1
    sc[(slope > p["sat12"]) & (slope <= p["sat23"])] = 2
    sc[slope > p["sat23"]] = 3
    sc[stack.slope_smooth > p["sat34"]] = 4

    merge1 = np.maximum.reduce([sc, az, rf, prox])
    if wind is not None:
        merge1 = np.maximum(merge1, wind)
    merge1[overhead4] = 4

    # forest reclass: open=10 sparse=20 mixed=30 dense=40
    forest = stack.forest_pct
    fr = np.full(shape, 10, dtype=np.int16)  # default/NODATA forest -> open(10)
    fr[(forest > p["tree1"]) & (forest <= p["tree2"])] = 20
    fr[(forest > p["tree2"]) & (forest <= p["tree3"])] = 30
    fr[forest > p["tree3"]] = 40

    pra_union = (freq.pra == 1) | (ext.pra == 1)
    pr = np.where(pra_union, 100, 0).astype(np.int16)

    arr = _apply_lut((merge1 + fr + pr).astype(np.int16), _LUT_1)
    fmask = np.where((fr == 20) | (fr == 30) | (fr == 40), 1000, 0).astype(np.int16)
    arr2 = _apply_lut(arr + fmask, _LUT_2)
    # preserve overhead class-4 through the forest demotion
    arr2[overhead4] = 4
    return arr2


def _classify_pra_base(stack, freq, ext, az, rf, prox, overhead4, wind, p):
    """PRA-base: continuous PRA defines classes 0-3 (slope+forest+wind already
    baked in, which suppresses DSM canopy-edge artifacts). Slope owns class 4
    (PRA would wrongly suppress steep wind-scoured/rocky terrain). Forest is NOT
    re-applied to 0-3 (it is already in PRA) - it only demotes slope-driven 4."""
    shape = stack.slope.shape
    base = np.zeros(shape, dtype=np.int16)
    base[ext.pra_continuous >= p["pra1"]] = 1    # extreme envelope
    base[freq.pra_continuous >= p["pra2"]] = 2   # frequent inner
    base[freq.pra_continuous >= p["pra3"]] = 3

    merge = np.maximum.reduce([base, az, rf, prox])
    if wind is not None:
        merge = np.maximum(merge, wind)

    # class 4: slope is the defining factor for extreme terrain, plus overhead.
    slope_4 = stack.slope_smooth > p["sat34"]
    merge[slope_4 | overhead4] = 4

    # forest-for-extreme: demote slope-only class-4 in forest -> 3; preserve
    # overhead class-4 (only place forest is applied in PRA base).
    demote = (merge == 4) & (stack.forest_pct > p["tree1"]) & slope_4 & ~overhead4
    merge[demote] = 3
    return merge.astype(np.int16)


# ---------------------------------------------------------------------------
# dev8 (default). Legacy PRA-base rules stay in _classify.
# ---------------------------------------------------------------------------
def _scenario_token(path: Path):
    name = Path(path).name
    for scen in ("frequent", "extreme"):
        if name.startswith(f"pra_{scen}_"):
            return scen
    return None


def _sibling(pra: dict, filename: str):
    """A file next to the scenario's continuous or sieve raster."""
    for key in ("continuous", "sieve"):
        raw = pra.get(key)
        if not raw:
            continue
        candidate = Path(raw).parent / filename
        if candidate.exists():
            return candidate
    return None


def _rasterize_pra_size(pra: dict, stack):
    """Burn each start-zone polygon's area_ha onto the DEM grid.

    Returns None when the polygon file is missing. The size vote is then
    skipped rather than invented.
    """
    token = None
    for key in ("continuous", "sieve"):
        raw = pra.get(key)
        if raw:
            token = _scenario_token(raw)
            if token:
                break
    if token is None:
        logger.info("dev8: PRA paths have no scenario name; size vote skipped")
        return None
    gpkg = _sibling(pra, f"pra_{token}_poly_v1.gpkg")
    if gpkg is None:
        logger.info("dev8: no pra_%s_poly_v1.gpkg; size vote skipped", token)
        return None
    try:
        import geopandas as gpd
        from rasterio.crs import CRS
        from rasterio.features import rasterize
    except ImportError as exc:
        logger.warning("dev8: cannot rasterize %s (%s); size vote skipped", gpkg.name, exc)
        return None
    gdf = gpd.read_file(gpkg)
    if gdf.empty or "area_ha" not in gdf.columns:
        logger.warning("dev8: %s has no area_ha; size vote skipped", gpkg.name)
        return None
    dst_crs = stack.profile.get("crs")
    if gdf.crs is not None and dst_crs is not None:
        dst = CRS.from_user_input(dst_crs)
        if CRS.from_user_input(gdf.crs) != dst:
            gdf = gdf.to_crs(dst)
    shapes = []
    for geom, val in zip(gdf.geometry, gdf["area_ha"]):
        if geom is None or geom.is_empty:
            continue
        shapes.append((geom, 0.0 if val is None or val != val else float(val)))
    height, width = stack.slope.shape
    burned = rasterize(
        shapes, out_shape=(height, width), transform=stack.transform,
        fill=0.0, dtype="float32")
    logger.info("dev8: %s size vote, %.1f ha of start zone on the grid",
                token, float((burned > 0).sum() * stack.cell_area / 10000.0))
    return burned


def _distance_layer(arr: np.ndarray) -> np.ndarray:
    """Metres. Negative and nodata read as far away, not as 'inside the hazard'."""
    out = np.asarray(arr, dtype="float32")
    bad = ~np.isfinite(out) | (out < 0) | (out == NODATA)
    out = out.copy()
    out[bad] = np.float32(1.0e4)
    return out


def _classify_dev8(stack, ates_dir, pra_by_scenario) -> Path:
    freq = stack.scenarios.get("frequent")
    ext = stack.scenarios.get("extreme")
    if ext is None or freq is None:
        logger.warning("dev8: only one scenario in the stack; both roles use it")
    if ext is None:
        ext = freq
    if freq is None:
        freq = ext
    if freq is None:
        raise ValueError("ATES dev8: no scenarios in driver stack")
    for role, layers in (("frequent", freq), ("extreme", ext)):
        if layers.zdelta is None or layers.rout_flux_area is None:
            raise RuntimeError(
                f"ATES dev8: scenario '{role}' is missing ungated zdelta or rout_flux_area")

    valid = stack.valid_mask
    canopy = np.where(stack.forest_pct == NODATA, 0, stack.forest_pct).astype(np.float32)
    slope = np.where(valid, stack.slope.astype(np.float32), 0.0)
    if valid.any() and float(np.nanmax(canopy[valid])) > 100.5:
        logger.warning(
            "ATES dev8: canopy exceeds 100 (max %.1f). It should already be percent.",
            float(np.nanmax(canopy[valid])))

    def _pra(layers):
        if layers.pra_continuous is None:
            return np.zeros(slope.shape, dtype=np.float32)
        return layers.pra_continuous.astype(np.float32)

    model_layers = {
        "E_pra": _pra(ext),
        "F_pra": _pra(freq),
        "E_zdelta": ext.zdelta.astype(np.float32),
        "F_zdelta": freq.zdelta.astype(np.float32),
        "E_travel_angle": ext.fp_travel_angle.astype(np.float32),
        "F_travel_angle": freq.fp_travel_angle.astype(np.float32),
        "E_rout_flux_area": ext.rout_flux_area.astype(np.float32),
        "F_rout_flux_area": freq.rout_flux_area.astype(np.float32),
        "E_runout_prox": _distance_layer(ext.runout_prox),
    }
    for scen, tag in (("frequent", "F"), ("extreme", "E")):
        pra = pra_by_scenario.get(scen, {})
        size = _rasterize_pra_size(pra, stack)
        if size is not None:
            model_layers[f"{tag}_pra_size"] = size
    extreme_pra = pra_by_scenario.get("extreme", {})
    prox_path = _sibling(extreme_pra, "pra_extreme_pra_proximity.tif")
    if prox_path is None:
        logger.info("dev8: no extreme PRA proximity raster; that class-0 factor is omitted")
    else:
        aligned = _align(prox_path, stack.profile, fill=1.0e4,
                         resampling=rasterio.warp.Resampling.nearest)
        model_layers["E_pra_proximity"] = _distance_layer(aligned)

    cell = float(np.sqrt(stack.cell_area))
    classes = dev8_classify(model_layers, valid, slope, canopy, cell)
    # classes is int8. A where() against that dtype wraps -9999 to -15.
    out = np.full(classes.shape, NODATA, dtype=np.int16)
    keep = classes >= 0
    out[keep] = classes[keep]
    merge2_path = _write(out, stack.profile, ates_dir / "merge_2.tif")
    logger.info("ATES dev8 merge_2 classes: %s (cell %.2f m)", np.unique(out[valid]), cell)
    return merge2_path


# ---------------------------------------------------------------------------
# Params + public entry point
# ---------------------------------------------------------------------------
def _get_params(config: configparser.ConfigParser) -> dict:
    s = "ATES"
    g = lambda k, f: config.getfloat(s, k, fallback=f)
    gi = lambda k, f: config.getint(s, k, fallback=f)
    params = {
        "sat01": g("sat01", 15), "sat12": g("sat12", 25),
        "sat23": g("sat23", 32), "sat34": g("sat34", 45),
        "win_size": gi("win_size", 5),
        "az1": g("az1", 1), "az2": g("az2", 30), "az3": g("az3", 75),
        "rf1": g("rf1", 0), "rf2": g("rf2", 500), "rf3": g("rf3", 3000),
        "tree1": g("tree1", 20), "tree2": g("tree2", 60), "tree3": g("tree3", 85),
        "isl_size": g("isl_size", 2500),
        "isl_size_01": g("isl_size_01", 10000), "isl_size_12": g("isl_size_12", 2500),
        "isl_size_23": g("isl_size_23", 10000), "isl_size_34": g("isl_size_34", 10000),
        "class0_prox": g("class0_prox", 100),
        # overhead -> class 4 (frequent flow-path travel angle, deg)
        "class4_alpha": g("class4_alpha", 60),
        # base-layer mode + PRA-base thresholds (extreme PRA->1, frequent->2/3)
        "base_layer": config.get(s, "base_layer", fallback="slope").strip().lower(),
        "pra1": g("pra1", 0.05), "pra2": g("pra2", 0.25), "pra3": g("pra3", 0.5),
        # driver derivation
        "max_z": config.getfloat("FlowPy", "max_z", fallback=270.0),
        "flux_cutoff": g("flux_cutoff", 25),
    }
    params.update(get_postprocess_params(config))
    return params


def run_ates_classification(
    dem_path: Path,
    forest_path: Path,
    runout_by_scenario: dict,
    pra_by_scenario: dict,
    output_dir: Path,
    config: configparser.ConfigParser,
    aoi_path: Path = None,
    wind_deposits_path: Path = None,
) -> Path:
    """Run the multi-scenario ATES classification.

    Parameters
    ----------
    runout_by_scenario : dict
        ``{scenario: {"zDelta": p, "routFluxSum": p, "fpTravelAngleMax": p,
        ["forestInteraction": p]}}`` for "frequent" and/or "extreme".
    pra_by_scenario : dict
        ``{scenario: {"sieve": p}}`` release-area rasters per scenario.
    """
    logger.info("Starting ATES classification (v3.0 multi-scenario)...")
    ates_dir = Path(output_dir)
    ates_dir.mkdir(parents=True, exist_ok=True)

    required = ("zDelta", "routFluxSum", "fpTravelAngleMax")
    for scen, rr in runout_by_scenario.items():
        missing = [k for k in required if not rr.get(k) or not Path(rr[k]).exists()]
        if missing:
            raise FileNotFoundError(
                f"ATES scenario '{scen}' missing runout rasters {missing}")

    params = _get_params(config)
    stack = build_driver_stack(dem_path, forest_path, runout_by_scenario,
                               pra_by_scenario, ates_dir, params)

    model = config.get("ATES", "model", fallback="dev8").strip().lower()
    logger.info("ATES model: %s", model)
    if model == "legacy":
        merge2_path = _classify(stack, params, ates_dir, wind_deposits_path)
    elif model == "dev8":
        merge2_path = _classify_dev8(stack, ates_dir, pra_by_scenario)
    else:
        raise ValueError(f"ATES.model must be 'dev8' or 'legacy', got {model!r}")

    # Shared post-processing for all classifiers: MMU cleanup + polygons + colour.
    results = postprocess_ates(merge2_path, ates_dir, params, aoi_path=aoi_path)

    final_path = Path(results["classification"])
    logger.info(f"✅ ATES classification complete → {final_path.name}")
    return final_path
