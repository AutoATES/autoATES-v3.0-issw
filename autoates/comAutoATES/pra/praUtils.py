"""
praUtils.py
Stable version tuned to match original PRA outputs as closely as possible.
"""

from pathlib import Path
import logging
import numpy as np
import rasterio
from osgeo import gdal
from osgeo_utils.gdal_sieve import gdal_sieve
from numba import njit, prange

from autoates.comAutoATES.pra.fuzzy import (
    cauchy_membership_function,
    cauchy_membership_asymmetric,
    fuzzy_AND,
)

logger = logging.getLogger(__name__)


def calculate_slope(dem_path: Path, output_path: Path):
    logger.info("Calculating slope...")
    dem = gdal.Open(str(dem_path))
    gdal.DEMProcessing(str(output_path), dem, "slope", computeEdges=True)
    logger.info(f"Slope saved → {output_path}")


def calculate_aspect(dem_path: Path, output_path: Path):
    """Aspect raster via gdal.DEMProcessing (needed for ruggedness)."""
    logger.info("Calculating aspect...")
    dem = gdal.Open(str(dem_path))
    gdal.DEMProcessing(str(output_path), dem, "aspect", computeEdges=True)
    logger.info(f"Aspect saved → {output_path}")


def compute_ruggedness(slope_path: Path, aspect_path: Path, output_path: Path):
    """Vector-dispersion ruggedness from slope + aspect (RA jzheren method).

    Each pixel's surface-normal unit vector is summed over a 3x3 window; the
    resultant length R (0..9) gives ruggedness = 1 - R/9 (0 = planar, →1 = rough).
    Tuned for higher-resolution (≈5 m) DEM input.
    """
    logger.info("Computing ruggedness...")
    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float64)
        profile = src.profile
    with rasterio.open(aspect_path) as src:
        aspect = src.read(1).astype(np.float64)

    slope_rad = slope * np.pi / 180.0
    aspect_rad = aspect * np.pi / 180.0
    xy = np.sin(slope_rad)
    z_raster = np.cos(slope_rad)
    x_raster = np.sin(aspect_rad) * xy
    y_raster = np.cos(aspect_rad) * xy

    def _win_sum(arr):
        view = np.lib.stride_tricks.as_strided(
            arr, shape=(3, 3, arr.shape[0] - 2, arr.shape[1] - 2),
            strides=arr.strides * 2)
        return view.sum(axis=(0, 1))

    resultant = np.sqrt(_win_sum(x_raster) ** 2 + _win_sum(y_raster) ** 2 + _win_sum(z_raster) ** 2)
    ruggedness = 1 - (resultant / 9.0)
    ruggedness = np.pad(ruggedness, (1, 1), "constant", constant_values=(0, 0)).astype(np.float32)

    # nodata=-9999, NOT 0. Ruggedness is 1 - |resultant|/9 over a 3x3 vector-dispersion
    # window, so 0 means "perfectly co-planar normals", i.e. smooth terrain - the most
    # common value in flat ground and a real measurement, not a gap. Declaring 0 as nodata
    # made every smooth cell read as missing downstream. The only genuinely undefined cells
    # are the 1-px border, which the pad above fills with 0 as well; that is a known and
    # accepted approximation, not a reason to mislabel the interior.
    profile.update({'dtype': 'float32', 'nodata': -9999.0,
                    'tiled': 'YES', 'blockxsize': 256, 'blockysize': 256})
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(ruggedness, 1)
    logger.info(f"Ruggedness saved → {output_path}")


def compute_windshelter(dem_path: Path, output_path: Path, params: dict):
    logger.info("Computing windshelter...")

    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(np.float64)
        profile = src.profile
        cell_size = abs(profile['transform'][0])

    target_m = params.get('wind_radius_m', 60)
    radius = max(2, int(round(target_m / cell_size)))

    wind_start = params['wind_direction'] - params['wind_tolerance'] + 270
    wind_end   = params['wind_direction'] + params['wind_tolerance'] + 270

    dist, mask = windshelter_prep(radius, wind_start, wind_end, cell_size)

    windshelter = compute_windshelter_numba(
        elev, dist, mask, radius, params['wind_probability'], 
        profile.get('nodata', -9999)
    )

    windshelter = np.nan_to_num(windshelter, nan=0.0)   # Use 0 like original

    # NoData is -9999, NOT 0. Zero is a MEANINGFUL windshelter value: the metric is
    # the p-quantile of arctan((neighbour - centre)/distance) over the search sector,
    # so locally planar terrain of ANY gradient yields exactly 0 by symmetry -- "no net
    # sheltering or exposure", with positive = deposition and negative = scouring. On the
    # 21 m ski-area grids that is ~6% of cells (and ~4% of the mapped start-zone mask),
    # sitting as a sharp spike at exactly 0.0 with almost nothing within +-0.001 of it.
    # Declaring nodata=0.0 therefore mislabelled real, well-determined terrain as missing.
    # The actual sentinel is the `radius`-wide border frame, which compute_windshelter_numba
    # initialises to the DEM's nodata and never writes.
    profile.update({
        'dtype': 'float32',
        'nodata': -9999.0,
        'tiled': 'YES',
        'blockxsize': 256,
        'blockysize': 256
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(windshelter.astype(np.float32), 1)

    logger.info(f"Windshelter saved → {output_path}")


def windshelter_prep(radius, direction, tolerance, cellsize):
    x_size = y_size = 2 * radius + 1
    x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
    center = (radius, radius)
    dist = np.sqrt((x_arr - center[0])**2 + (y_arr - center[1])**2) * cellsize
    mask = sector_mask(dist.shape, center, radius, (direction, tolerance))
    mask[radius, radius] = True
    return dist, mask


def sector_mask(shape, centre, radius, angle_range):
    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)
    if tmax < tmin:
        tmax += 2 * np.pi
    r2 = (x - cx)**2 + (y - cy)**2
    theta = np.arctan2(x - cx, y - cy) - tmin
    theta %= (2 * np.pi)
    return (r2 <= radius**2) & (theta <= (tmax - tmin))


@njit(parallel=True, cache=True)
def compute_windshelter_numba(elev, dist, mask, radius, prob, nodata):
    rows, cols = elev.shape
    ws = 2 * radius + 1
    result = np.full((rows, cols), nodata, dtype=np.float64)

    for i in prange(radius, rows - radius):
        for j in range(radius, cols - radius):
            win = elev[i-radius:i+radius+1, j-radius:j+radius+1]
            data = win * mask.astype(np.float64)

            for di in range(ws):
                for dj in range(ws):
                    if data[di, dj] == nodata or data[di, dj] == 0:
                        data[di, dj] = np.nan
            data[radius, radius] = np.nan

            center = win[radius, radius]
            diffs = (data - center) / dist
            tan_vals = np.arctan(diffs)

            valid = [tan_vals[di, dj] for di in range(ws) for dj in range(ws) 
                    if not np.isnan(tan_vals[di, dj])]

            if len(valid) == 0:
                result[i, j] = 0.0
            else:
                valid = np.sort(np.array(valid))
                p = prob * (len(valid) - 1)
                ii = int(np.floor(p))
                frac = p - ii
                if ii + 1 < len(valid):
                    result[i, j] = valid[ii] + frac * (valid[ii + 1] - valid[ii])
                else:
                    result[i, j] = valid[ii]
    
    return result


def apply_fuzzy_logic(slope_path, windshelter_path, forest_path,
                     pra_cont_path, pra_bin_path, params: dict,
                     ruggedness_path=None):
    """Combine slope / windshelter / forest (and optional ruggedness) Cauchy
    memberships with the fuzzy-AND (OWA) aggregation.

    Membership + NoData handling follow the canonical RA (jzheren) core:
      * NoData is NOT mapped to 0 before the membership transform. A NoData
        forest pixel yields a near-zero forest membership (treated as fully
        excluded), not a membership of 1 (treated as open terrain).
      * The forest membership has its non-positive values reset to 1, matching
        the original model (negative Cauchy values arise from odd 2b powers
        with a negative forest_c offset).
      * Ruggedness, when enabled, is a 4th membership; pixels with ruggedness
        > 0.01 are forced to 0 (excluded).
    """
    logger.info("Applying fuzzy logic PRA combination...")

    with rasterio.open(slope_path) as src:
        slope = src.read(1)
        profile = src.profile

    with rasterio.open(windshelter_path) as src:
        windshelter = src.read(1)

    with rasterio.open(forest_path) as src:
        forest = src.read(1)
        forest_nodata = src.nodata

    # Forest NoData = open terrain (no canopy), NOT "unknown -> exclude". Canopy
    # products such as the Sen2 rasters store -9999 over alpine/rock (no trees to
    # map) rather than 0, and that is exactly the prime start-zone terrain. Feeding
    # NoData straight into the Cauchy would give forestC ≈ 0 and mask it out, so we
    # map NoData -> 0 (open) here, giving forestC = 1. (Reverts one aspect of the
    # RA-canonical NoData handling that collapsed PRA on such rasters.)
    nd = forest_nodata if forest_nodata is not None else -9999
    forest = np.where(forest == nd, 0, forest)

    # Slope membership: symmetric Cauchy by default. If any of the optional
    # two-sided keys are present the asymmetric form is used instead, letting the
    # low-angle shoulder and the cliff cutoff be tuned independently (a symmetric
    # Cauchy cannot reach 28-32 deg without also reaching 58-62 deg).
    a_low = params.get('slope_a_low') or params['slope_a']
    b_low = params.get('slope_b_low') or params['slope_b']
    a_high, b_high = params.get('slope_a_high'), params.get('slope_b_high')
    if a_high is None and b_high is None and \
            params.get('slope_a_low') is None and params.get('slope_b_low') is None:
        slopeC = cauchy_membership_function(slope, [params['slope_a'], params['slope_b'], params['slope_c']])
    else:
        slopeC = cauchy_membership_asymmetric(slope, a_low, b_low, params['slope_c'], a_high, b_high)

    # Optional hard release window, as in autoATES v2.0 (AutoATES.py:391-392, which
    # clipped the slope membership to 25-55 deg on top of the Cauchy). The clip was
    # dropped in the RA-canonical core that v3.0 inherited, so v3.0 effective windows
    # run wider than the classic release band. Off by default; set to restore it.
    clip_lo, clip_hi = params.get('slope_clip_lo'), params.get('slope_clip_hi')
    if clip_lo is not None:
        slopeC = np.where(slope < clip_lo, 0.0, slopeC)
    if clip_hi is not None:
        slopeC = np.where(slope > clip_hi, 0.0, slopeC)
    windC = cauchy_membership_function(windshelter, [params['wind_a'], params['wind_b'], params['wind_c']])
    forestC = cauchy_membership_function(forest, [params['forest_a'], params['forest_b'], params['forest_c']])
    forestC[forestC <= 0] = 1
    forestC = forestC.astype(np.float32)

    cmv_list = [slopeC, windC, forestC]

    if ruggedness_path is not None:
        with rasterio.open(ruggedness_path) as src:
            ruggedness = src.read(1)
        ruggC = cauchy_membership_function(
            ruggedness, [params['rugg_a'], params['rugg_b'], params['rugg_c']]
        )
        ruggC[ruggedness > 0.01] = 0
        cmv_list.append(ruggC)

    pra = fuzzy_AND(cmv_list)
    pra[pra <= 0] = 0
    pra = pra.astype(np.float32)

    # nodata=-9999, NOT 0, despite the original doing otherwise. `pra[pra <= 0] = 0` two
    # lines up makes 0 the CLAMPED FLOOR of the membership - "this cell is not a release
    # area" - which is the most informative value the layer carries and covers most of the
    # map. Declaring it nodata made "definitely not a start zone" indistinguishable from
    # "unknown", and any consumer honouring the header silently dropped all non-release
    # terrain. Same class of error as the forest-NoData bug, opposite direction.
    profile.update({
        'dtype': 'float32',
        'nodata': -9999.0,
        'tiled': 'YES',
        'blockxsize': 256,
        'blockysize': 256
    })
    with rasterio.open(pra_cont_path, 'w', **profile) as dst:
        dst.write(pra, 1)

    # Binary
    pra_bin = (pra >= params['pra_threshold']).astype(np.int16)
    profile.update({'dtype': 'int16', 'nodata': -9999})
    with rasterio.open(pra_bin_path, 'w', **profile) as dst:
        dst.write(pra_bin, 1)

    logger.info("PRA continuous + binary saved.")


# Keep the rest of the functions (calculate_proximity, sieve_pra, scale_forest) as they were in the previous stable version
def calculate_proximity(pra_bin_path: Path, output_path: Path):
    logger.info("Calculating PRA proximity...")
    ds = gdal.Open(str(pra_bin_path))
    if ds is None:
        raise FileNotFoundError(f"Could not open {pra_bin_path}")
    band = ds.GetRasterBand(1)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(str(output_path), ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Int16)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    gdal.ComputeProximity(band, out_band, ['DISTUNITS=GEO'])
    out_band = None
    out_ds = None
    ds = None
    logger.info(f"PRA proximity saved → {output_path}")


def sieve_pra(input_bin_path: Path, output_path: Path, threshold_m2: float):
    logger.info(f"Applying sieve filter (threshold = {threshold_m2} m²)")
    with rasterio.open(input_bin_path) as src:
        pixel_size_x = abs(src.transform[0])
        pixel_size_y = abs(src.transform[4])
        profile = src.profile

    num_cells = int(round(threshold_m2 / (pixel_size_x * pixel_size_y)))
    if num_cells < 1:
        num_cells = 1

    temp_sieve = output_path.parent / f"{output_path.stem}_temp.tif"
    gdal_sieve(src_filename=str(input_bin_path), dst_filename=str(temp_sieve), threshold=num_cells, connectedness=4)

    with rasterio.open(input_bin_path) as src:
        pra_original = src.read(1)
    with rasterio.open(temp_sieve) as src:
        pra_sieved = src.read(1)

    pra_filter = pra_original - pra_sieved
    pra_filter[pra_filter < 0] = 0
    pra_final = pra_original - pra_filter
    pra_final = pra_final.astype(np.int16)

    profile.update({'dtype': 'int16', 'nodata': -9999, 'tiled': 'YES', 'blockxsize': 256, 'blockysize': 256})
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pra_final, 1)

    temp_sieve.unlink(missing_ok=True)
    logger.info(f"Sieve filter completed → {output_path}")


def scale_forest(forest_path: Path, output_path: Path):
    logger.info("Scaling forest canopy...")
    with rasterio.open(forest_path) as src:
        forest = src.read(1)
        profile = src.profile

    forest_scale = np.clip(forest / 100.0, 0.0, 1.0).astype(np.float32)

    # nodata=-9999, NOT 0. 0 = OPEN GROUND, the single most important value here: it is what
    # alpine terrain above treeline reads as, and Sen2 canopy NoData is itself alpine (see
    # the forest-NoData fix). Declaring 0 as nodata therefore deleted exactly the terrain
    # that carries the avalanche signal - measured cost when a consumer honoured it: PRA's
    # macro AUC at ref1->2+ moved 0.624 -> 0.483 and flipped sign in the Rockies.
    profile.update({
        'dtype': 'float32',
        'nodata': -9999.0,
        'tiled': 'YES',
        'blockxsize': 256,
        'blockysize': 256
    })
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(forest_scale, 1)

    logger.info(f"Forest scale saved → {output_path}")