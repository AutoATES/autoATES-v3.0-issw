"""
atesPostProcess.py - Shared post-processing for ALL ATES classifiers (v3.0).

Any classifier (rules / fuzzy / ML) emits a raw integer ATES class raster
(0-4 + NODATA). This module turns that into the shareable products, IDENTICALLY
for every approach so the three are directly comparable:

  1. clean     : per-class minimum-mapping-unit (MMU) removal of small polygons.
                 Sub-MMU connected components of each class are dissolved into
                 the nearest surviving terrain. Removes e.g. DSM canopy-edge
                 artifacts (small class-1 circles inside class-0 along roads).
  2. vectorize : polygonize the cleaned classes to a GeoPackage (one feature per
                 contiguous patch, with class + area attributes).
  3. colorize  : gdaldem color-relief RGB raster with the ATES palette.

This is the output-side counterpart of driverStack (the shared input stack).
"""
from pathlib import Path
import logging

import numpy as np
import rasterio
from rasterio import features
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)
NODATA = -9999

# ATES class colours now live in out1/palettes/ates_class.txt (official Revised
# Canadian ATES palette), shared with the Mapbox export. See _colorize().

# 8-connectivity for connected-component labelling
_STRUCT = np.ones((3, 3), dtype=bool)


def _read_class(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nd = src.nodata if src.nodata is not None else NODATA
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    arr = np.where(arr == nd, NODATA, arr).astype(np.int16)
    cell_area = abs(transform.a) * abs(transform.e)
    return arr, profile, transform, crs, cell_area


def _remove_small_polygons(cls: np.ndarray, valid: np.ndarray,
                           mmu_cells: dict) -> np.ndarray:
    """Dissolve sub-MMU connected components of each class into nearest survivor.

    mmu_cells maps class value -> minimum component size in cells. Components
    smaller than their class MMU are marked for removal, then every removed cell
    takes the value of its nearest surviving cell (a class-1 blob in a class-0
    sea therefore becomes class-0).
    """
    out = cls.copy()
    to_fill = np.zeros(cls.shape, dtype=bool)
    for c, mmu in mmu_cells.items():
        if mmu <= 1:
            continue
        mask = out == c
        if not mask.any():
            continue
        lbl, n = ndi.label(mask, structure=_STRUCT)
        if n == 0:
            continue
        sizes = np.bincount(lbl.ravel())
        small = np.where(sizes < mmu)[0]
        small = small[small != 0]  # drop background label 0
        if small.size:
            to_fill |= np.isin(lbl, small)

    if to_fill.any():
        survivors = valid & ~to_fill
        # nearest-survivor indices for every cell
        idx = ndi.distance_transform_edt(~survivors, return_distances=False,
                                         return_indices=True)
        out = out[tuple(idx)]
        logger.info("MMU cleanup: dissolved %d small-polygon cells", int(to_fill.sum()))
    out[~valid] = NODATA
    return out


def _write_int16(arr, profile, path, transform=None, crs=None):
    p = profile.copy()
    p.update(driver="GTiff", count=1, dtype="int16", nodata=NODATA)
    if transform is not None:
        p.update(transform=transform)
    if crs is not None:
        p.update(crs=crs)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype("int16"), 1)
    return path


def _clip_to_aoi(arr, profile, transform, crs, aoi_path):
    import geopandas as gpd
    from rasterio.mask import mask as rio_mask
    tmp = profile.copy()
    tmp.update(driver="GTiff", count=1, dtype="int16", nodata=NODATA,
               transform=transform, crs=crs)
    mem = rasterio.io.MemoryFile()
    with mem.open(**tmp) as ds:
        ds.write(arr.astype("int16"), 1)
        gdf = gpd.read_file(aoi_path)
        if gdf.crs != ds.crs:
            gdf = gdf.to_crs(ds.crs)
        clipped, ctransform = rio_mask(ds, gdf.geometry.values, crop=True, nodata=NODATA)
    out_profile = tmp.copy()
    out_profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=ctransform)
    return clipped[0].astype(np.int16), out_profile, ctransform


def _vectorize(cls, transform, crs, out_path):
    import geopandas as gpd
    from shapely.geometry import shape
    cell_area = abs(transform.a) * abs(transform.e)
    recs = []
    mask = cls != NODATA
    for geom, val in features.shapes(cls.astype(np.int16), mask=mask,
                                     transform=transform, connectivity=8):
        recs.append({"ates_class": int(val), "geometry": shape(geom)})
    gdf = gpd.GeoDataFrame(recs, crs=crs)
    gdf["area_m2"] = gdf.geometry.area
    gdf.to_file(out_path, driver="GPKG")
    return out_path, len(gdf)


def _colorize(cleaned_tif, out_tif):
    """RGBA colour raster in the official Canadian ATES palette.

    Delegates to the shared out1.colorize module so the ATES class palette lives
    in exactly one place (out1/palettes/ates_class.txt) and matches the Mapbox
    export. Categorical -> exact colour match, non-avalanche transparent.
    """
    from autoates.comAutoATES.out1.colorize import colorize_layer
    return colorize_layer(cleaned_tif, out_tif, "ates_class", categorical=True)


def get_postprocess_params(config) -> dict:
    """Per-class MMU (m^2) from [ATES]; low classes cleaned harder than 3/4."""
    g = lambda k, f: config.getfloat("ATES", k, fallback=f)
    return {
        "mmu_class0": g("mmu_class0", 10000),
        "mmu_class1": g("mmu_class1", 10000),
        "mmu_class2": g("mmu_class2", 5000),
        "mmu_class3": g("mmu_class3", 5000),
        "mmu_class4": g("mmu_class4", 2500),
    }


def postprocess_ates(class_raster_path: Path, output_dir: Path, params: dict,
                     aoi_path: Path = None, basename: str = "ATES_classification") -> dict:
    """Clean, vectorize and colorize a raw ATES class raster.

    Works for any classifier's output. Returns paths to the cleaned raster,
    polygon layer and RGB colour raster.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cls, profile, transform, crs, cell_area = _read_class(class_raster_path)
    valid = cls != NODATA

    mmu_cells = {c: max(1, int(round(params.get(f"mmu_class{c}", 0) / cell_area)))
                 for c in range(5)}
    logger.info("MMU (cells) per class: %s", mmu_cells)
    cleaned = _remove_small_polygons(cls, valid, mmu_cells)

    if aoi_path and Path(aoi_path).exists():
        cleaned, profile, transform = _clip_to_aoi(cleaned, profile, transform, crs, aoi_path)

    cleaned_tif = _write_int16(cleaned, profile, output_dir / f"{basename}.tif",
                               transform=transform, crs=crs)
    poly_path, n_poly = _vectorize(cleaned, transform, crs, output_dir / f"{basename}.gpkg")
    color_tif = _colorize(cleaned_tif, output_dir / f"{basename}_rgb.tif")

    logger.info("Post-process complete → %s (%d polygons), %s, %s",
                Path(cleaned_tif).name, n_poly, Path(poly_path).name, Path(color_tif).name)
    return {"classification": cleaned_tif, "polygons": poly_path, "color": color_tif}
