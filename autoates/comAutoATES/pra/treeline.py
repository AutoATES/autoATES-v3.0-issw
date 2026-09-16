"""
treeline.py
Regional treeline-elevation raster for autoATES v3.0.

Treeline is not a fixed elevation — it varies with snow climate, glacial
history, ecosystem and terrain. We estimate it per AvCan forecast subregion
(the natural snow-climate unit) from the DEM + forest canopy: treeline is a high
percentile (default 95th) of the elevations of *forested* pixels in the
subregion — the consistent upper limit of forest.

Two important points:
  * Start-zone polygons are largely treeless (avalanche activity), so treeline
    must be characterised from the forested terrain *around* them. Aggregating
    over a whole subregion does exactly that.
  * A large regional sample makes a high percentile both accurate (tracks the
    forest limit, not valley forest) and stable (no small-window noise).

The result is a full-resolution treeline-elevation raster (metres, constant
within each subregion), useful for large-polygon subdivision and potentially
other v3.0 steps.
"""
from pathlib import Path
import logging
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
import geopandas as gpd

logger = logging.getLogger(__name__)

NODATA = -9999.0

# Mountain-range -> coarse group, for the small-sample treeline fallback.
RANGE_GROUP = {
    "South Coast": "Coast", "North Coast": "Coast", "East Coast": "Coast",
    "Selkirks": "Columbia", "Monashees": "Columbia", "Cariboos": "Columbia",
    "Purcells": "Columbia",
    "Central Rockies": "Rockies", "South Rockies": "Rockies", "North Rockies": "Rockies",
}


def _estimate_treeline(forested_elev, valid_elev, method, percentile,
                       frac_threshold, elev_bin_m, min_forest_px):
    """Treeline elevation from one sample, or NaN if too little forest.

    'percentile'     : `percentile`-th percentile of forested-pixel elevations.
    'fraction_limit' : highest elevation bin whose forest fraction (forested /
                       all valid pixels) is >= `frac_threshold` — the forest edge.
    """
    if forested_elev.size < min_forest_px:
        return np.nan
    if method == "percentile":
        return float(np.percentile(forested_elev, percentile))
    # fraction_limit
    lo, hi = valid_elev.min(), valid_elev.max()
    edges = np.arange(lo, hi + elev_bin_m, elev_bin_m)
    if len(edges) < 2:
        return float(np.percentile(forested_elev, percentile))
    ctr = 0.5 * (edges[:-1] + edges[1:])
    tot, _ = np.histogram(valid_elev, edges)
    fo, _ = np.histogram(forested_elev, edges)
    frac = np.divide(fo, tot, out=np.zeros(len(tot), float), where=tot > 0)
    ok = np.where((frac >= frac_threshold) & (tot > 50))[0]
    return float(ctr[ok[-1]]) if len(ok) else np.nan


def compute_aoi_treeline(dem_path: Path, forest_path: Path, output_path: Path,
                         forest_threshold: float = 30.0, method: str = "percentile",
                         percentile: float = 95.0, frac_threshold: float = 0.3,
                         elev_bin_m: float = 50.0, min_forest_px: int = 500) -> Path:
    """Treeline from the input DEM + forest over the whole AOI (no zoning layer) —
    the portable default that works anywhere, inside or outside Canada. Writes a
    constant treeline raster; if the AOI has too little forest (e.g. all-alpine),
    the raster is all-NoData and splitting falls back to benches + size cap."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        profile = src.profile
        dem_nodata = src.nodata
        shp = (src.height, src.width)
    with rasterio.open(forest_path) as src:
        forest = src.read(1).astype(np.float64)
        forest_nodata = src.nodata

    valid = np.isfinite(dem)
    if dem_nodata is not None:
        valid &= (dem != dem_nodata)
    fnd = forest_nodata if forest_nodata is not None else NODATA
    forested = valid & (forest != fnd) & (forest >= forest_threshold)
    tl = _estimate_treeline(dem[forested], dem[valid], method, percentile,
                            frac_threshold, elev_bin_m, min_forest_px)

    out = np.where(valid & np.isfinite(tl), np.float32(tl), NODATA).astype(np.float32) \
        if np.isfinite(tl) else np.full(shp, NODATA, np.float32)
    profile.update({'dtype': 'float32', 'nodata': NODATA,
                    'tiled': 'YES', 'blockxsize': 256, 'blockysize': 256})
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(out, 1)
    logger.info(f"AOI treeline saved -> {output_path} "
                f"({'%.0f m' % tl if np.isfinite(tl) else 'insufficient forest -> NoData'})")
    return output_path


def compute_subregion_treeline(dem_path: Path, forest_path: Path,
                               subregions_path: Path, output_path: Path,
                               forest_threshold: float = 30.0,
                               method: str = "percentile", percentile: float = 95.0,
                               frac_threshold: float = 0.3, elev_bin_m: float = 50.0,
                               min_forest_px: int = 500) -> Path:
    """Write a treeline raster with one value per AvCan forecast subregion.

    forest_threshold : canopy-cover %% at/above which a pixel counts as forest.
    method           : 'percentile' (default, p95) or 'fraction_limit'.
    min_forest_px    : subregions with fewer forested pixels get NoData (their
                       oversized zones fall back to bench / size-cap splitting).
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        profile = src.profile
        transform = src.transform
        shp = (src.height, src.width)
        dem_nodata = src.nodata
        crs = src.crs
    with rasterio.open(forest_path) as src:
        forest = src.read(1).astype(np.float64)
        forest_nodata = src.nodata

    valid = np.isfinite(dem)
    if dem_nodata is not None:
        valid &= (dem != dem_nodata)
    fnd = forest_nodata if forest_nodata is not None else NODATA
    forested = valid & (forest != fnd) & (forest >= forest_threshold)

    sub = gpd.read_file(subregions_path).to_crs(crs)
    out = np.full(shp, NODATA, dtype=np.float32)
    n_done = 0
    for _, reg in sub.iterrows():
        if reg.geometry is None or reg.geometry.is_empty:
            continue
        inreg = ~geometry_mask([reg.geometry], out_shape=shp, transform=transform, invert=False)
        if not (inreg & valid).any():
            continue  # subregion does not overlap the DEM
        f_elev = dem[forested & inreg]
        v_elev = dem[valid & inreg]
        tl = _estimate_treeline(f_elev, v_elev, method, percentile,
                                frac_threshold, elev_bin_m, min_forest_px)
        if np.isfinite(tl):
            out[inreg & valid] = tl
            n_done += 1
            logger.info(f"  subregion '{reg.get('polygon_na', '?')}': treeline {tl:.0f} m "
                        f"({f_elev.size:,} forested px)")

    profile.update({'dtype': 'float32', 'nodata': NODATA,
                    'tiled': 'YES', 'blockxsize': 256, 'blockysize': 256})
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(out, 1)
    logger.info(f"Subregion treeline saved -> {output_path} "
                f"({n_done} subregions, method={method})")
    return output_path


def build_treeline_raster(dem_path: Path, subregions_path: Path, table_csv: Path,
                          output_path: Path, min_reliable_kpx: float = 150.0,
                          name_field: str = "polygon_na", range_field: str = "mountain_r") -> Path:
    """Rasterise a per-subregion treeline onto a site DEM grid from a precomputed
    national table (see treeline_by_subregion.csv / compute_subregion_treeline),
    applying a small-sample fallback: a subregion with < ``min_reliable_kpx`` k
    forested pixels inherits its mountain-range median treeline, then its coarse
    range-group (Coast/Columbia/Rockies) median. Keeps treeline estimation (done
    once, nationally, over full subregion extents) separate from per-site use."""
    import pandas as pd
    df = pd.read_csv(table_csv)
    reliable = df[df["n_forest_kpx"] >= min_reliable_kpx]
    range_med = reliable.groupby("range")["treeline"].median().to_dict()
    group_med = reliable.groupby("group")["treeline"].median().to_dict()
    lut = {r["name"]: r for _, r in df.iterrows()}

    with rasterio.open(dem_path) as src:
        shp = (src.height, src.width)
        transform = src.transform
        crs = src.crs
        profile = src.profile
        dem = src.read(1)
        dem_nodata = src.nodata

    sub = gpd.read_file(subregions_path).to_crs(crs)
    shapes, notes = [], {"subregion": 0, "range": 0, "group": 0, "none": 0}
    for _, reg in sub.iterrows():
        if reg.geometry is None or reg.geometry.is_empty:
            continue
        rng = reg.get(range_field)
        grp = RANGE_GROUP.get(rng)
        row = lut.get(reg.get(name_field))
        if row is not None and row["n_forest_kpx"] >= min_reliable_kpx:
            val, note = float(row["treeline"]), "subregion"
        elif rng in range_med:
            val, note = float(range_med[rng]), "range"
        elif grp in group_med:
            val, note = float(group_med[grp]), "group"
        else:
            notes["none"] += 1
            continue
        notes[note] += 1
        shapes.append((reg.geometry, val))

    out = rasterize(shapes, out_shape=shp, transform=transform, fill=NODATA,
                    dtype="float32") if shapes else np.full(shp, NODATA, np.float32)
    valid = np.isfinite(dem)
    if dem_nodata is not None:
        valid &= (dem != dem_nodata)
    out = np.where(valid, out, NODATA).astype(np.float32)

    profile.update({'dtype': 'float32', 'nodata': NODATA,
                    'tiled': 'YES', 'blockxsize': 256, 'blockysize': 256})
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(out, 1)
    logger.info(f"Treeline raster (with fallback) saved -> {output_path} "
                f"[subregion={notes['subregion']}, range-fallback={notes['range']}, "
                f"group-fallback={notes['group']}]")
    return output_path
