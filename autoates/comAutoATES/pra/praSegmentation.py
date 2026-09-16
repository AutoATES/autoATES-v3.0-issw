"""
PRA Polygon Segmentation Module for autoATES v3.0
Automatic versioning + full config support
"""

import configparser
from pathlib import Path
import logging
import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.segmentation import watershed
from skimage.measure import label, regionprops_table
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
import pandas as pd

logger = logging.getLogger(__name__)


def get_next_version(output_dir: Path, scenario: str) -> int:
    """Automatically find next version number (v1, v2, ...)"""
    pattern = f"pra_{scenario}_poly_v*.gpkg"
    existing = list(output_dir.glob(pattern))
    if not existing:
        return 1
    versions = []
    for f in existing:
        try:
            v = int(f.stem.split('_v')[-1])
            versions.append(v)
        except ValueError:
            continue
    return max(versions) + 1 if versions else 1


def get_segmentation_params(scenario: str, config: configparser.ConfigParser):
    section = 'PRA_Segmentation'
    base = scenario.lower()
    return {
        "min_distance": config.getint(section, f"{base}_min_distance", fallback=2 if base == "frequent" else 5),
        "pra_maxima_min_distance": config.getint(section, f"{base}_pra_maxima_min_distance", fallback=2 if base == "frequent" else 5),
        "pra_scale_factor": config.getfloat(section, f"{base}_pra_scale_factor", fallback=1.0),
        "merge_threshold": config.getfloat(section, f"{base}_merge_threshold", fallback=5000 if base == "frequent" else 15000),
        "max_polygon_size_m2": config.getfloat(section, f"{base}_max_polygon_size_m2", fallback=250000 if base == "frequent" else 500000),
        # Large-polygon subdivision is terrain-following (a watershed on the
        # PRA/curvature/aspect/elevation cost surface) with soft ridges at bench
        # (slope-minima) and treeline elevations. max_polygon_size_m2 bounds
        # sub-zone size; split_bench_prominence (deg) is the min slope-dip depth
        # counted as a bench; split_use_treeline toggles the treeline ridge.
        "split_bench_prominence": config.getfloat(section, f"{base}_split_bench_prominence", fallback=1.5),
        "split_use_treeline": config.getboolean(section, "split_use_treeline", fallback=True),
        # Trigger: 'size' = only oversized zones; 'terrain' = only zones crossing a
        # bench/treeline; 'both' = either. enforce_size_cap tiles oversized zones to
        # the cap (set False to only ever split on bench/treeline — no hard cap).
        "split_trigger": config.get(section, "split_trigger", fallback="both"),
        # Default: no hard size cap — split only where a bench/treeline exists, so
        # broad well-connected paths stay whole (conservative for the extreme
        # scenario). Set True to also tile oversized zones to max_polygon_size_m2.
        "split_enforce_size_cap": config.getboolean(section, "split_enforce_size_cap", fallback=False),
        "split_min_stratify_area_m2": config.getfloat(
            section, f"{base}_split_min_stratify_area_m2",
            fallback=(20000 if base == "frequent" else 60000)),
        "enable_morphological_cleanup": config.getboolean(section, "enable_morphological_cleanup", fallback=True),
        "opening_iterations": config.getint(section, "opening_iterations", fallback=1),
        "closing_iterations": config.getint(section, "closing_iterations", fallback=1),
        "weights": {
            "pra": config.getfloat(section, f"{base}_pra_weight", fallback=5.0),
            "curvature": config.getfloat(section, f"{base}_curvature_weight", fallback=2.5 if base == "frequent" else 1.0),
            "aspect": config.getfloat(section, f"{base}_aspect_weight", fallback=1.0 if base == "frequent" else 2.0),
            "forest": config.getfloat(section, f"{base}_forest_weight", fallback=1.0),
            "elevation": config.getfloat(section, f"{base}_elevation_weight", fallback=1.0 if base == "frequent" else 2.5),
        }
    }


def _grid_markers(cmask: np.ndarray, step: int):
    """Seed a regular grid of watershed markers (spacing ``step`` px) on the True
    pixels of ``cmask``. Returns (labelled marker array, count)."""
    markers = np.zeros(cmask.shape, dtype=np.int32)
    ys, xs = np.nonzero(cmask)
    if len(ys) == 0:
        return markers, 0
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    idx = 1
    for r in range(r0, r1 + 1, step):
        for c in range(c0, c1 + 1, step):
            if cmask[r, c]:
                markers[r, c] = idx
                idx += 1
    return markers, idx - 1


def _split_elevations(cmask, dem_l, slope_l, treeline_val, bench_prominence, bin_m):
    """Elevations at which to encourage a split within one oversized zone:
      * benches — local minima of mean-slope-vs-elevation (flatter bands between
        steeper faces), found with ``bench_prominence`` (deg) as the depth cutoff;
      * treeline — the local per-subregion treeline, if it falls inside the zone.
    Returns a de-duplicated, sorted list of elevations (m)."""
    ev = dem_l[cmask]
    if ev.size == 0:
        return []
    emin, emax = float(ev.min()), float(ev.max())
    elevs = []
    if slope_l is not None and (emax - emin) > 3 * bin_m:
        edges = np.arange(emin, emax + bin_m, bin_m)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        prof = binned_statistic(ev, slope_l[cmask], "mean", bins=edges).statistic
        prof = np.nan_to_num(prof, nan=np.nanmean(prof))
        smooth = ndi.gaussian_filter1d(prof, 1.5)
        mins, _ = find_peaks(-smooth, prominence=bench_prominence)
        elevs += [float(ctr[m]) for m in mins]
    if treeline_val is not None and np.isfinite(treeline_val) \
            and emin + bin_m < treeline_val < emax - bin_m:
        elevs.append(float(treeline_val))
    elevs.sort()
    merged = []
    for e in elevs:
        if not merged or e - merged[-1] > bin_m:
            merged.append(e)
    return merged


def _resegment_cost(cmask, gradient_l, dem_l, split_elevs, ridge_weight, sigma_m):
    """Custom cost surface for re-segmenting one oversized zone: the primary
    terrain gradient (curvature/aspect/PRA/elevation) normalised 0..1 over the
    zone, plus a soft Gaussian *ridge* at each split elevation (benches/treeline).
    The ridge makes watershed boundaries prefer those elevations, but the terrain
    term lets the boundary bend to the actual bench/gully near that elevation — so
    cuts stratify by elevation without being flat contour lines."""
    g = gradient_l
    gvals = g[cmask]
    gmin = float(gvals.min())
    grange = float(gvals.max()) - gmin
    cost = (g - gmin) / (grange + 1e-9)
    for e in split_elevs:
        cost = cost + ridge_weight * np.exp(-0.5 * ((dem_l - e) / sigma_m) ** 2)
    return cost.astype(np.float32)


def _stratify_watershed(cmask, cost, dem_l, split_elevs, sigma_m):
    """Split a zone at its bench/treeline elevations WITHOUT enforcing the size
    cap: seed each elevation stratum (interior, away from split elevations by
    ``sigma_m``) as one marker region and let the watershed assign the buffer
    bands along the terrain-following cost ridges. Yields ~(#split_elevs + 1)
    sub-zones. Returns an int label array (0 outside), or the whole zone if it
    could not seed ≥2 strata."""
    if not split_elevs:
        return cmask.astype(np.int32)
    strata = np.digitize(dem_l, split_elevs)  # 0..n by elevation
    near = np.zeros(cmask.shape, dtype=bool)
    for e in split_elevs:
        near |= np.abs(dem_l - e) < sigma_m  # leave a buffer band unseeded
    markers = np.where(cmask & ~near, strata + 1, 0).astype(np.int32)
    if len(np.unique(markers[markers > 0])) < 2:
        return cmask.astype(np.int32)
    return watershed(cost, markers=markers, mask=cmask)


def _densify_watershed(cmask, cost, saliency_l, max_px):
    """Split a zone and bound every sub-zone to ``max_px``: seed PRA-core maxima
    plus grid back-fill in gaps, tightening the spacing until no basin exceeds the
    cap. Boundaries follow ``cost`` (terrain + any bench/treeline ridges)."""
    step0 = max(2, int(np.sqrt(max(max_px, 1.0))))
    labels = cmask.astype(np.uint8)
    best = None
    for shrink in (1.0, 0.7, 0.5, 0.35, 0.25):
        step = max(2, int(step0 * shrink))
        peaks = peak_local_max(saliency_l, min_distance=step, labels=labels)
        markers = np.zeros(cmask.shape, dtype=np.int32)
        for i, (r, c) in enumerate(peaks, start=1):
            markers[r, c] = i
        if markers.any():
            far = ndi.distance_transform_edt(markers == 0) > step
        else:
            far = cmask
        gm, _ = _grid_markers(cmask & far, step)
        nxt = int(markers.max())
        for g in np.unique(gm[gm > 0]):
            nxt += 1
            markers[gm == g] = nxt
        if int(markers.max()) < 2:
            gm2, _ = _grid_markers(cmask, max(2, step // 2))
            for g in np.unique(gm2[gm2 > 0]):
                nxt += 1
                markers[gm2 == g] = nxt
            if int(markers.max()) < 2:
                return cmask.astype(np.int32)
        seg = watershed(cost, markers=markers, mask=cmask)
        best = seg
        counts = np.bincount(seg[cmask].ravel())
        if len(counts) <= 1 or counts[1:].max() <= max_px:
            break
    return best


def _resegment_terrain(cmask, gradient_l, saliency_l, dem_l, split_elevs, max_px,
                       bin_m=30.0, ridge_weight=3.0, enforce_cap=True):
    """Re-segment one zone with a terrain-following watershed. Boundaries are
    attracted to the ``split_elevs`` (bench/treeline) via soft Gaussian cost
    ridges but bend along the terrain, so cuts land on real benches not flat
    contours. If ``enforce_cap`` the zone is also tiled so every sub-zone ≤
    ``max_px`` (size backstop); otherwise it is only stratified at split_elevs."""
    cost = _resegment_cost(cmask, gradient_l, dem_l, split_elevs,
                           ridge_weight, sigma_m=1.3 * bin_m)
    if enforce_cap:
        return _densify_watershed(cmask, cost, saliency_l, max_px)
    return _stratify_watershed(cmask, cost, dem_l, split_elevs, sigma_m=1.3 * bin_m)


def subdivide_large_labels(segments: np.ndarray, gradient: np.ndarray, saliency: np.ndarray,
                           dem: np.ndarray, transform, max_size_m2: float,
                           slope: np.ndarray = None, treeline: np.ndarray = None,
                           bench_prominence: float = 1.5, use_treeline: bool = True,
                           trigger: str = "both", enforce_size_cap: bool = True,
                           min_stratify_area_m2: float = 0.0) -> np.ndarray:
    """Subdivide start-zone labels with a terrain-following watershed that
    stratifies by elevation at bench/treeline breaks.

    A label is re-segmented when:
      * ``trigger`` in {'size','both'} and its area exceeds ``max_size_m2`` (the
        oversized case — tiled to the cap when ``enforce_size_cap``), and/or
      * ``trigger`` in {'terrain','both'} and it crosses a bench or the treeline
        (area ≥ ``min_stratify_area_m2``) — split at those elevations regardless
        of size. This de-emphasises the hard size cap: zones with a real
        bench/treeline are split naturally; a smooth oversized face without either
        is only tiled if ``enforce_size_cap``.

    ``slope`` / ``treeline`` (optional full-grid rasters) supply the bench and
    treeline signals. Slivers are absorbed downstream by cap-aware ``merge_small``.
    """
    pixel_area = abs(transform.a * transform.e)
    if pixel_area <= 0:
        return segments
    have_cap = max_size_m2 is not None and max_size_m2 > 0
    max_px = (max_size_m2 / pixel_area) if have_cap else float("inf")
    min_stratify_px = (min_stratify_area_m2 / pixel_area) if min_stratify_area_m2 else 0.0
    want_size = trigger in ("size", "both") and have_cap
    want_terrain = trigger in ("terrain", "both")

    out = segments.astype(np.int32, copy=True)
    next_label = int(out.max()) + 1

    objs = ndi.find_objects(out)
    n_size = n_terrain = 0
    for lid in range(1, len(objs) + 1):
        sl = objs[lid - 1]
        if sl is None:
            continue
        sub_seg = out[sl]
        mask = sub_seg == lid
        area_px = int(mask.sum())
        oversized = have_cap and area_px > max_px

        slope_l = slope[sl] if slope is not None else None
        treeline_val = None
        if use_treeline and treeline is not None:
            tlz = treeline[sl][mask]
            tlz = tlz[tlz > -1000]
            if tlz.size:
                treeline_val = float(np.median(tlz))

        # Bench/treeline elevations for this zone (needed for either trigger, so
        # the size-triggered path also gets the ridges).
        split_elevs = _split_elevations(mask, dem[sl], slope_l, treeline_val,
                                        bench_prominence, bin_m=30.0)
        terrain_hit = (want_terrain and area_px >= min_stratify_px and len(split_elevs) > 0)
        size_hit = want_size and oversized
        if not (size_hit or terrain_hit):
            continue

        # Enforce the cap only when a genuinely oversized zone needs bounding.
        enforce = enforce_size_cap and oversized
        seg = _resegment_terrain(mask, gradient[sl], saliency[sl], dem[sl],
                                 split_elevs, max_px, enforce_cap=enforce)
        seg_ids = np.unique(seg[seg > 0])
        if len(seg_ids) < 2:
            continue  # could not split; leave label as-is

        for j, cid in enumerate(seg_ids):
            piece = seg == cid
            if j == 0:
                sub_seg[piece] = lid
            else:
                sub_seg[piece] = next_label
                next_label += 1
        if size_hit:
            n_size += 1
        else:
            n_terrain += 1

    if n_size or n_terrain:
        logger.info(f"Subdivided {n_size} oversized + {n_terrain} bench/treeline-crossing "
                    f"start zone(s) via terrain-following watershed "
                    f"(trigger={trigger}, enforce_cap={enforce_size_cap})")
    return out


def _shared_border(a, b):
    """Length of the shared boundary between two touching polygons (0 if none)."""
    try:
        inter = a.boundary.intersection(b.boundary)
        return inter.length
    except Exception:
        return 0.0


def merge_small(gdf_in: gpd.GeoDataFrame, thresh: float, max_size_m2: float = None):
    """Absorb small polygons that border larger terrain, in two phases:

    1. **Coalesce** touching sub-``thresh`` polygons into their connected clusters
       (a chain of small slivers becomes one polygon).
    2. **Attach** any still-sub-``thresh`` cluster to the touching neighbour it
       shares the longest border with (spatially the zone it most belongs to),
       rather than merely the largest one. When ``max_size_m2`` is given a merge
       that would exceed the cap is skipped.

    A small polygon that borders a larger polygon is dissolved into it; a small
    polygon that touches nothing (a genuine isolated PRA island, surrounded by
    non-PRA terrain) is kept as-is — it is a real, if small, start zone.
    """
    if len(gdf_in) == 0:
        return gdf_in
    orig_crs = gdf_in.crs
    g = gdf_in.reset_index(drop=True).copy()
    other_cols = [c for c in g.columns if c not in ("geometry", "area_m2")]

    # --- Phase 1: coalesce connected clusters of small polygons (union-find) ---
    small = g[g['area_m2'] < thresh].reset_index(drop=True)
    large = g[g['area_m2'] >= thresh].reset_index(drop=True)
    if len(small) == 0:
        return g.set_crs(orig_crs)

    parent = list(range(len(small)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sidx = small.sindex
    sgeom = small.geometry.values
    for i in range(len(small)):
        for j in sidx.query(sgeom[i], predicate="touches"):
            if j > i:
                parent[find(i)] = find(int(j))

    clusters = {}
    for i in range(len(small)):
        clusters.setdefault(find(i), []).append(i)

    rows = []
    for members in clusters.values():
        geom = unary_union([sgeom[m] for m in members]) if len(members) > 1 else sgeom[members[0]]
        rep = small.iloc[members].sort_values('area_m2').iloc[-1]  # attrs from largest member
        row = {c: rep[c] for c in other_cols}
        row['geometry'] = geom
        row['area_m2'] = geom.area
        rows.append(row)
    clustered = gpd.GeoDataFrame(rows, crs=orig_crs)

    # --- Phase 2: attach still-small clusters to longest-shared-border neighbour ---
    large = large.reset_index(drop=True)
    lidx = large.sindex if len(large) else None
    leftover = []
    for _, s in clustered.iterrows():
        if s['area_m2'] >= thresh or lidx is None:
            leftover.append(s)
            continue
        cand = [int(j) for j in lidx.query(s.geometry, predicate="touches")]
        if max_size_m2 and max_size_m2 > 0:
            cand = [j for j in cand if large.at[j, 'area_m2'] + s['area_m2'] <= max_size_m2]
        if not cand:
            leftover.append(s)
            continue
        best = max(cand, key=lambda j: _shared_border(s.geometry, large.at[j, 'geometry']))
        merged = unary_union([large.at[best, 'geometry'], s.geometry])
        large.at[best, 'geometry'] = merged
        large.at[best, 'area_m2'] = merged.area

    parts = [large] + ([gpd.GeoDataFrame(leftover, crs=orig_crs)] if leftover else [])
    final = pd.concat(parts, ignore_index=True)
    final = gpd.GeoDataFrame(final, crs=orig_crs)
    final['area_m2'] = final.geometry.area.round(2)
    return final


_COMPASS8 = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])


def add_zonal_metrics(gdf: gpd.GeoDataFrame, continuous, dem, slope, aspect, treeline,
                      transform, windshelter=None, ruggedness=None) -> gpd.GeoDataFrame:
    """Recompute per-polygon zonal statistics from the source rasters on the FINAL
    polygons (after subdivision + merge), so the fields are exact regardless of
    merge bookkeeping. Adds start-zone metrics useful for output maps:
      area_ha, pra_mean, pra_max, release_potential (pra_mean x area_ha),
      elev_min/max/mean, relief_m, slope_mean/max, aspect_deg + aspect_class,
      wind_mean (if windshelter given), rugg_mean (if ruggedness given), and —
      where treeline is available — treeline_m, above_tl_frac and a band label
      (alpine / treeline / below-treeline)."""
    if len(gdf) == 0:
        return gdf
    gdf = gdf.reset_index(drop=True)
    n = len(gdf)
    lab = rasterize(((g, i + 1) for i, g in enumerate(gdf.geometry)),
                    out_shape=continuous.shape, transform=transform, fill=0, dtype="int32")

    def _stats(values, valid):
        m = valid & (lab > 0)
        ids = lab[m]
        cnt = np.bincount(ids, minlength=n + 1)[1:]
        cs = np.where(cnt == 0, 1, cnt)
        mean = np.bincount(ids, weights=values[m], minlength=n + 1)[1:] / cs
        return mean, cnt
    idx = np.arange(1, n + 1)

    gdf["area_m2"] = gdf.geometry.area.round(1)
    gdf["area_ha"] = (gdf["area_m2"] / 1e4).round(3)

    cont_valid = np.isfinite(continuous) & (continuous > 0)
    pra_mean, _ = _stats(continuous, cont_valid)
    pra_max = ndi.maximum(np.where(cont_valid, continuous, -np.inf), lab, idx)
    gdf["pra_mean"] = np.round(pra_mean, 4)
    gdf["pra_max"] = np.round(np.where(np.isfinite(pra_max), pra_max, 0.0), 4)

    dvalid = np.isfinite(dem) & (dem > -1000)
    emean, _ = _stats(dem, dvalid)
    emin = ndi.minimum(np.where(dvalid, dem, np.inf), lab, idx)
    emax = ndi.maximum(np.where(dvalid, dem, -np.inf), lab, idx)
    gdf["elev_min"] = np.round(np.where(np.isfinite(emin), emin, 0), 1)
    gdf["elev_max"] = np.round(np.where(np.isfinite(emax), emax, 0), 1)
    gdf["elev_mean"] = np.round(emean, 1)
    gdf["relief_m"] = (gdf["elev_max"] - gdf["elev_min"]).round(1)

    if slope is not None:
        svalid = np.isfinite(slope) & (slope > -1000)
        smean, _ = _stats(slope, svalid)
        smax = ndi.maximum(np.where(svalid, slope, -np.inf), lab, idx)
        gdf["slope_mean"] = np.round(smean, 1)
        gdf["slope_max"] = np.round(np.where(np.isfinite(smax), smax, 0), 1)

    # Circular-mean aspect (deg) + compass class from the same DEM
    sin_a = np.sin(np.deg2rad(aspect)); cos_a = np.cos(np.deg2rad(aspect))
    ms, _ = _stats(sin_a, dvalid); mc, _ = _stats(cos_a, dvalid)
    adeg = np.mod(np.degrees(np.arctan2(ms, mc)), 360.0)
    gdf["aspect_deg"] = np.round(adeg, 1)
    gdf["aspect_class"] = _COMPASS8[(((adeg + 22.5) % 360) // 45).astype(int)]

    # Size-weighted release potential (handy for prioritising on maps)
    gdf["release_potential"] = (gdf["pra_mean"] * gdf["area_ha"]).round(3)

    if windshelter is not None:
        wmean, _ = _stats(windshelter, np.isfinite(windshelter))
        gdf["wind_mean"] = np.round(wmean, 3)
    if ruggedness is not None:
        rmean, _ = _stats(ruggedness, np.isfinite(ruggedness))
        gdf["rugg_mean"] = np.round(rmean, 4)

    if treeline is not None:
        tvalid = np.isfinite(treeline) & (treeline > -1000)
        tmean, tcnt = _stats(treeline, tvalid)
        above, _ = _stats((dem > treeline).astype(np.float32), dvalid & tvalid)
        has_tl = tcnt > 0
        gdf["treeline_m"] = np.where(has_tl, np.round(tmean, 0), np.nan)
        gdf["above_tl_frac"] = np.where(has_tl, np.round(above, 2), np.nan)
        band = np.where(above >= 0.7, "alpine",
                        np.where(above <= 0.3, "below_treeline", "treeline"))
        gdf["band"] = np.where(has_tl, band, "unknown")
    return gdf


def run_pra_segmentation(continuous_path: Path, binary_sieve_path: Path,
                        dem_path: Path, output_dir: Path, scenario: str,
                        config: configparser.ConfigParser,
                        slope_path: Path = None, treeline_path: Path = None,
                        windshelter_path: Path = None, ruggedness_path: Path = None) -> Path:
    """Main segmentation function with automatic versioning.

    ``slope_path`` / ``treeline_path`` (optional) drive the bench + treeline
    elevation stratification of oversized start zones. ``windshelter_path`` /
    ``ruggedness_path`` (optional) only add zonal-metric fields to the output.
    """

    version = get_next_version(output_dir, scenario)
    logger.info(f"Starting optimized PRA segmentation for {scenario.upper()} - Version v{version}")

    params = get_segmentation_params(scenario, config)

    # Load rasters
    with rasterio.open(binary_sieve_path) as src:
        binary = src.read(1) > 0.5
        transform = src.transform
        crs = src.crs

    with rasterio.open(continuous_path) as src:
        continuous = src.read(1).astype(np.float32)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)

    slope_arr = None
    if slope_path is not None and Path(slope_path).exists():
        with rasterio.open(slope_path) as src:
            slope_arr = src.read(1).astype(np.float32)
    treeline_arr = None
    if treeline_path is not None and Path(treeline_path).exists():
        with rasterio.open(treeline_path) as src:
            treeline_arr = src.read(1).astype(np.float32)

    def _opt_read(p):
        if p is not None and Path(p).exists():
            with rasterio.open(p) as src:
                return src.read(1).astype(np.float32)
        return None
    windshelter_arr = _opt_read(windshelter_path)
    ruggedness_arr = _opt_read(ruggedness_path)

    hard_mask = binary & np.isfinite(continuous)
    logger.info(f"Processing {hard_mask.sum():,} valid pixels")

    # Normalization and derivatives
    cont_norm = np.clip((continuous / np.nanmax(continuous)) * params["pra_scale_factor"], 0, 1)

    dy, dx = np.gradient(dem)
    aspect = np.mod(np.arctan2(-dy, dx) * (180 / np.pi), 360)
    dyy = np.gradient(dy, axis=0)
    dxx = np.gradient(dx, axis=1)
    dxy = np.gradient(dy, axis=1)
    curvature = -(dxx * dy**2 - 2*dxy*dx*dy + dyy*dx**2) / (dx**2 + dy**2 + 1e-8)**1.5
    curvature = np.clip(curvature, -5, 5)
    curv_norm = (curvature - curvature.min()) / (np.ptp(curvature) + 1e-8)
    elev_norm = (dem - dem.min()) / (np.ptp(dem) + 1e-8)

    # Gradient
    gradient = (1.0 - cont_norm) * params["weights"]["pra"]
    gradient += np.abs(curv_norm) * params["weights"]["curvature"]
    gradient += np.abs(np.sin(np.deg2rad(aspect))) * params["weights"]["aspect"]
    gradient += elev_norm * params["weights"]["elevation"]
    gradient *= hard_mask.astype(float)

    # Markers
    high_pra_mask = (cont_norm > 0.40) & hard_mask
    peaks_pra = peak_local_max(ndi.distance_transform_edt(high_pra_mask),
                               min_distance=params["pra_maxima_min_distance"], labels=high_pra_mask)
    peaks_curv = peak_local_max(np.abs(curvature), min_distance=params["min_distance"], labels=hard_mask)

    marker_labels = np.zeros(hard_mask.shape, dtype=int)
    idx = 1
    for pts in [peaks_pra, peaks_curv]:
        for r, c in pts:
            if 0 <= r < marker_labels.shape[0] and 0 <= c < marker_labels.shape[1]:
                if marker_labels[r, c] == 0:
                    marker_labels[r, c] = idx
                    idx += 1

    segments = watershed(gradient, markers=label(marker_labels), mask=hard_mask)

    # Morphological cleanup
    if params["enable_morphological_cleanup"]:
        logger.info("Performing fast morphological cleanup...")
        binary_segments = (segments > 0).astype(np.uint8)
        binary_clean = ndi.binary_opening(binary_segments, iterations=params["opening_iterations"])
        binary_clean = ndi.binary_closing(binary_clean, iterations=params["closing_iterations"])
        segments = np.where(binary_clean, segments, 0)

    # Re-attach any binary PRA pixels not assigned to a segment (e.g. removed by
    # the morphological opening) as their own labels, in raster space and BEFORE
    # subdivision — so the elevation/aspect cuts and the size cap apply to them
    # too. (Previously these were re-added as uncapped polygons AFTER
    # vectorization, which let large uncovered blobs bypass the size cap.)
    uncovered = binary & (segments == 0)
    if uncovered.sum() > 0:
        logger.info(f"Re-attaching {uncovered.sum():,} uncovered PRA pixels as labels")
        extra_lbl, _ = ndi.label(uncovered)
        segments = np.where(uncovered, extra_lbl + int(segments.max()), segments)

    # Subdivide oversized start zones in label space BEFORE vectorization, so the
    # sub-zones polygonize in the same shapes() pass. Cuts follow the terrain cost
    # surface (gradient); PRA continuous (cont_norm) seeds the basin cores.
    segments = subdivide_large_labels(
        segments, gradient, cont_norm, dem, transform,
        max_size_m2=params["max_polygon_size_m2"],
        slope=slope_arr, treeline=treeline_arr,
        bench_prominence=params["split_bench_prominence"],
        use_treeline=params["split_use_treeline"],
        trigger=params["split_trigger"],
        enforce_size_cap=params["split_enforce_size_cap"],
        min_stratify_area_m2=params["split_min_stratify_area_m2"],
    )

    # Vectorize: build a label -> mean-intensity lookup, then polygonize the whole
    # label raster in ONE pass. The previous version looped per label with a
    # full-array `segments == lid` scan AND a per-label DataFrame filter inside the
    # loop — O(num_segments x cells) plus O(num_segments^2) — which stalled for
    # hours on large domains (e.g. ~4 h Columbia_Kootenays, ~21 h Columbia_Revelstoke).
    props = regionprops_table(segments, intensity_image=continuous, properties=('label', 'area', 'mean_intensity'))
    df = pd.DataFrame(props)
    good = df[df['area'] >= 0].copy()
    mean_by_label = dict(zip(good['label'].astype(int), good['mean_intensity']))

    tagged = []
    for geom, lid in shapes(segments.astype(np.int32), mask=(segments > 0), transform=transform):
        lid = int(lid)
        if lid in mean_by_label:
            tagged.append((lid, shape(geom), mean_by_label[lid]))
    # Preserve the original per-label ordering (ascending label; stable within a
    # label = raster order) so downstream merge/split tie-breaking matches the
    # previous implementation exactly.
    tagged.sort(key=lambda t: t[0])
    polygons = [g for _, g, _ in tagged]
    means = [m for _, _, m in tagged]

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf['pra_mean'] = means
    gdf['area_m2'] = gdf.geometry.area.round(2)

    # Coverage is guaranteed above: all binary PRA pixels were folded into the
    # label raster before vectorization, so there is no post-hoc backfill here.

    # Coalesce small polygons / isolated pockets. Only cap-limit the merge when the
    # size cap is actually being enforced (otherwise merging is unconstrained).
    merge_cap = params["max_polygon_size_m2"] if params["split_enforce_size_cap"] else None
    final_gdf = merge_small(gdf, params["merge_threshold"], max_size_m2=merge_cap)

    # Final per-polygon zonal statistics (exact, recomputed from the rasters) for
    # output maps: area, PRA likelihood, elevation/relief, slope, aspect, treeline band.
    final_gdf = add_zonal_metrics(final_gdf, continuous, dem, slope_arr, aspect,
                                  treeline_arr, transform,
                                  windshelter=windshelter_arr, ruggedness=ruggedness_arr)

    # Save with version
    gpkg_path = output_dir / f"pra_{scenario}_poly_v{version}.gpkg"
    final_gdf.to_file(gpkg_path, driver="GPKG")
   
    logger.info(f"✅ Segmentation completed → {gpkg_path.name} ({len(final_gdf)} polygons)")
    return gpkg_path