"""
praCore.py
Main Potential Release Area (PRA) module for autoATES v3.
Generates rasters + polygons for Frequent and Extreme scenarios.
"""

from pathlib import Path
import logging
import configparser
from autoates.comAutoATES.pra.praUtils import (
    calculate_slope,
    calculate_aspect,
    compute_ruggedness,
    compute_windshelter,
    apply_fuzzy_logic,
    calculate_proximity,
    sieve_pra,
    scale_forest
)

# Per-forest-type Cauchy [a, b, c] presets (RA jzheren defaults), used as a
# fallback when explicit forest_a/b/c are not provided in the config.
FOREST_PRESETS = {
    "stems":     [350, 2.5, -150],
    "pcc":       [350, 2.5, -150],
    "no_forest": [350, 2.5, -150],
    "bav":       [20, 3.5, -10],
    "ch":        [7, 1.5, 0],
    "sen2cc":    [50, 1.5, 0],
}
from autoates.comAutoATES.pra.praSegmentation import run_pra_segmentation
from autoates.comAutoATES.pra.treeline import (
    compute_aoi_treeline, compute_subregion_treeline, build_treeline_raster,
)

logger = logging.getLogger(__name__)


def _ensure_treeline(dem_path: Path, forest_path: Path, pra_dir: Path,
                     config: configparser.ConfigParser):
    """Build (once, cached) the treeline-elevation raster used to stratify large
    start-zone splits. autoATES is NOT tied to precomputed Canadian treeline —
    treeline is always derivable from the input DEM + forest. Resolution order:
      1. disabled -> None (splits use benches + size cap only);
      2. national per-subregion table (Canada, with range/group fallback), if
         both subregions_path and treeline_table_csv are configured;
      3. per-subregion from local rasters, if only a zoning layer is configured;
      4. whole-AOI from the input DEM+forest — the portable default, works
         anywhere in/outside Canada with no external data.
    Returns the treeline raster path, or None on disable/failure."""
    if not config.getboolean('PRA', 'enable_treeline', fallback=True):
        return None
    treeline_path = pra_dir / "pra_treeline.tif"
    if treeline_path.exists():
        return treeline_path  # scenario-independent; reuse across scenarios
    subregions = config.get('PRA', 'subregions_path', fallback=None)
    table_csv = config.get('PRA', 'treeline_table_csv', fallback=None)
    min_kpx = config.getfloat('PRA', 'treeline_min_reliable_kpx', fallback=150.0)
    try:
        if subregions and Path(subregions).exists():
            if table_csv and Path(table_csv).exists():
                return build_treeline_raster(dem_path, Path(subregions), Path(table_csv),
                                             treeline_path, min_reliable_kpx=min_kpx)
            return compute_subregion_treeline(dem_path, forest_path, Path(subregions),
                                              treeline_path, method="percentile", percentile=95.0)
        # No zoning layer -> derive treeline from the input rasters (portable).
        return compute_aoi_treeline(dem_path, forest_path, treeline_path,
                                    method="percentile", percentile=95.0)
    except Exception as e:
        logger.warning(f"Treeline computation failed ({e}); splits will use benches + size cap only")
        return None


def run_pra(dem_path: Path,
            forest_canopy_path: Path,
            output_dir: Path,
            scenario: str = "frequent",
            config: configparser.ConfigParser = None) -> dict:

    if config is None:
        raise ValueError("Config is required")

    logger.info(f"Starting PRA calculation - Scenario: {scenario.upper()}")

    # === Flat folder structure (no nesting) ===
    pra_dir = output_dir
    pra_dir.mkdir(parents=True, exist_ok=True)

    params = get_scenario_params(scenario, config)
    basename = params['basename']

    # All output paths now go directly into pra_dir (no extra "PRA" folder)
    slope_path          = pra_dir / f"{basename}_slope.tif"
    windshelter_path    = pra_dir / f"{basename}_windshelter.tif"
    pra_cont_path       = pra_dir / f"{basename}_pra_continuous.tif"
    pra_bin_path        = pra_dir / f"{basename}_pra_binary.tif"
    pra_sieve_path      = pra_dir / f"{basename}_pra_sieve.tif"
    pra_prox_path       = pra_dir / f"{basename}_pra_proximity.tif"
    forest_scale_path   = pra_dir / f"{basename}_forest_scale.tif"
    aspect_path         = pra_dir / f"{basename}_aspect.tif"
    ruggedness_path     = pra_dir / f"{basename}_ruggedness.tif"

    calculate_slope(dem_path, slope_path)
    compute_windshelter(dem_path, windshelter_path, params)

    # Optional ruggedness membership (RA core). Default OFF. Only consider for
    # ~5 m or finer DEMs; uncalibrated after the Connaught Creek 45-55 deg clip.
    rugg_arg = None
    if params.get('enable_ruggedness'):
        calculate_aspect(dem_path, aspect_path)
        compute_ruggedness(slope_path, aspect_path, ruggedness_path)
        rugg_arg = ruggedness_path

    apply_fuzzy_logic(
        slope_path=slope_path,
        windshelter_path=windshelter_path,
        forest_path=forest_canopy_path,
        pra_cont_path=pra_cont_path,
        pra_bin_path=pra_bin_path,
        params=params,
        ruggedness_path=rugg_arg,
    )
    calculate_proximity(pra_bin_path, pra_prox_path)
    sieve_pra(pra_bin_path, pra_sieve_path, params['sieve_filter_m2'])
    scale_forest(forest_canopy_path, forest_scale_path)

    # Treeline raster for elevation-stratified large-polygon splitting (cached,
    # scenario-independent). None if not configured -> benches + size cap only.
    treeline_path = _ensure_treeline(dem_path, forest_canopy_path, pra_dir, config)

    # === 2. Polygon Segmentation ===
    logger.info(f"Running polygon segmentation for {scenario} scenario...")
    gpkg_path = run_pra_segmentation(
        continuous_path=pra_cont_path,
        binary_sieve_path=pra_sieve_path,
        dem_path=dem_path,
        output_dir=pra_dir,           # Important: pass pra_dir here
        scenario=scenario,
        config=config,
        slope_path=slope_path,
        treeline_path=treeline_path,
        windshelter_path=windshelter_path,
        ruggedness_path=(ruggedness_path if params.get('enable_ruggedness') else None),
    )

    logger.info(f"✅ {scenario.upper()} PRA completed (rasters + polygons).")

    return {
        "scenario": scenario,
        "pra_continuous": pra_cont_path,
        "pra_sieve": pra_sieve_path,
        "pra_proximity": pra_prox_path,
        "forest_scale": forest_scale_path,
        "pra_polygons": gpkg_path,
        "pra_dir": pra_dir
    }


def get_scenario_params(scenario: str, config: configparser.ConfigParser) -> dict:
    """Extract parameters from config.

    Forest membership a/b/c fall back to the per-forest-type preset when not
    given explicitly. Ruggedness is an optional 4th membership (off by default).
    """
    config_prefix = "frequent" if scenario.lower() == "frequent" else "extreme"
    forest_type = config.get('PRA', 'forest_type', fallback='sen2cc')
    f_a, f_b, f_c = FOREST_PRESETS.get(forest_type, FOREST_PRESETS['sen2cc'])
    return {
        "basename": f"pra_{scenario.lower()}",
        "forest_type": forest_type,
        "pra_threshold": config.getfloat('PRA', f'{config_prefix}_pra_threshold'),
        "slope_a": config.getfloat('PRA', f'{config_prefix}_slope_a'),
        "slope_b": config.getfloat('PRA', f'{config_prefix}_slope_b'),
        "slope_c": config.getfloat('PRA', f'{config_prefix}_slope_c'),
        # Optional two-sided slope membership. Leave unset for the symmetric
        # Cauchy (default). Setting *_a_high / *_b_high decouples the low-angle
        # shoulder from the cliff cutoff; *_a_low / *_b_low override slope_a/b on
        # the low side only.
        "slope_a_low": config.getfloat('PRA', f'{config_prefix}_slope_a_low', fallback=None),
        "slope_b_low": config.getfloat('PRA', f'{config_prefix}_slope_b_low', fallback=None),
        "slope_a_high": config.getfloat('PRA', f'{config_prefix}_slope_a_high', fallback=None),
        "slope_b_high": config.getfloat('PRA', f'{config_prefix}_slope_b_high', fallback=None),
        # Optional hard release window (v2.0 clipped the slope membership to 25-55
        # deg; the RA-canonical core v3.0 inherited dropped it). Off by default.
        "slope_clip_lo": config.getfloat('PRA', f'{config_prefix}_slope_clip_lo', fallback=None),
        "slope_clip_hi": config.getfloat('PRA', f'{config_prefix}_slope_clip_hi', fallback=None),
        "forest_a": config.getfloat('PRA', f'{config_prefix}_forest_a', fallback=f_a),
        "forest_b": config.getfloat('PRA', f'{config_prefix}_forest_b', fallback=f_b),
        "forest_c": config.getfloat('PRA', f'{config_prefix}_forest_c', fallback=f_c),
        "wind_a": config.getfloat('PRA', 'wind_a', fallback=1.0),
        "wind_b": config.getfloat('PRA', 'wind_b', fallback=1.0),
        "wind_c": config.getfloat('PRA', 'wind_c', fallback=1.0),
        "wind_direction": config.getfloat('PRA', 'wind_direction', fallback=0),
        "wind_tolerance": config.getfloat('PRA', 'wind_tolerance', fallback=180),
        "wind_probability": config.getfloat('PRA', 'wind_probability', fallback=0.5),
        "wind_radius_m": config.getfloat('PRA', 'wind_radius_m', fallback=60),
        "sieve_filter_m2": config.getfloat('PRA', 'sieve_filter_m2', fallback=1000),
        # Optional ruggedness membership (RA core)
        "enable_ruggedness": config.getboolean('PRA', 'enable_ruggedness', fallback=False),
        "rugg_a": config.getfloat('PRA', 'rugg_a', fallback=0.01),
        "rugg_b": config.getfloat('PRA', 'rugg_b', fallback=5.0),
        "rugg_c": config.getfloat('PRA', 'rugg_c', fallback=-0.007),
    }