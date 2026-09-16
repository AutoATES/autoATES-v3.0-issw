"""
comAutoATES.py - Main workflow orchestrator for autoATES v3.0
Stable base from Friday + integrated runout and ATES
"""
from pathlib import Path
import logging
import configparser
import time
from datetime import datetime

from autoates.comAutoATES.in1.in1AutoATES import align_and_clip_rasters
from autoates.comAutoATES.pra.praCore import run_pra
from autoates.comAutoATES.runout.runoutCore import run_flowpy
from autoates.comAutoATES.ates.atesCore import run_ates_classification
from autoates.comAutoATES.utils.runManifest import (write_run_manifest,
                                                    finalize_run_manifest)

logger = logging.getLogger(__name__)


def load_config(config_path: Path = None) -> configparser.ConfigParser:
    if config_path is None:
        config_path = Path(__file__).parent / "autoATESCfg.ini"
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    cfg.read(config_path)
    return cfg


_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'


def _setup_run_logging(log_file: Path) -> list:
    """Bind a per-run log file, and return the handlers we installed.

    Deliberately NOT logging.basicConfig(): basicConfig is a silent no-op once the
    root logger has any handler, so a second runAutoATES() in the same process kept
    writing into the FIRST run's log file. runSitesBatch.py runs several sites per
    process and had to reset the root handlers from outside to work around it - which
    means every other caller was still exposed. Owning the handler lifecycle here
    makes all callers safe, and removing our handlers afterwards stops a long batch
    accumulating one handler per site (which would duplicate every line N times).
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(_LOG_FORMAT)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    installed = [fh]
    # Only add a console handler if nothing is already writing to the console,
    # so an interactive caller with its own handler does not get doubled output.
    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        installed.append(sh)
    for h in installed:
        root.addHandler(h)
    return installed


def _teardown_run_logging(handlers: list) -> None:
    root = logging.getLogger()
    for h in handlers:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def runAutoATES(config: configparser.ConfigParser = None, config_path: Path = None):
    """Main entry point.

    config_path is recorded in the run manifest. Pass it whenever a config object is
    built elsewhere (runSitesBatch.py mutates a loaded config in memory, so the config
    a site actually ran with never existed as a file - the manifest's config copy is
    then the only record of it).
    """
    if config is None:
        if config_path is None:
            config_path = Path(__file__).parent / "autoATESCfg.ini"
        config = load_config(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['General'].get('working_dir', './Outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"autoATES_run_{timestamp}.log"
    log_handlers = _setup_run_logging(log_file)
    manifest_path = None
    t0 = time.perf_counter()
    try:
        # Provenance FIRST, before any computation: a run that crashes or is killed
        # still leaves a complete record of what it was trying to do.
        manifest_path = write_run_manifest(output_dir, config,
                                           config_path=config_path,
                                           timestamp=timestamp)
        logger.info(f"Run manifest : {manifest_path.name}")

        result = _runAutoATES(config, timestamp, output_dir, log_file)

        finalize_run_manifest(manifest_path, status="completed",
                              wall_clock_min=round((time.perf_counter() - t0) / 60, 2),
                              log_file=log_file.name,
                              step_times_min=result.pop("_step_times", {}),
                              outputs=result.pop("_outputs", {}))
        return result
    except BaseException as exc:
        # Includes KeyboardInterrupt: a run killed 60 minutes in is exactly the case
        # where knowing what it was running matters most.
        if manifest_path is not None:
            finalize_run_manifest(manifest_path, status="failed",
                                  wall_clock_min=round((time.perf_counter() - t0) / 60, 2),
                                  log_file=log_file.name,
                                  error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _teardown_run_logging(log_handlers)


def _runAutoATES(config, timestamp, output_dir, log_file):
    logger.info("=" * 80)
    logger.info(f"autoATES v3.0 STARTED - {timestamp}")
    logger.info("=" * 80)
    logger.info(f"Project : {config['General'].get('project_name', 'Unnamed')}")
    logger.info(f"Output dir : {output_dir}")

    step_times = {}

    pra_dir = output_dir / "PRA"
    flowpy_dir = output_dir / "FlowPy"
    ates_dir = output_dir / "ATES"

    for d in (pra_dir, flowpy_dir, ates_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Preprocessing (from Friday version - this is important)
    logger.info("-" * 80)
    logger.info("Step 1: Preprocessing input rasters...")
    _t = time.perf_counter()

    target_res_x = config.get('General', 'target_resolution_x', fallback='0')
    target_res_y = config.get('General', 'target_resolution_y', fallback='0')
    target_resolution = None if target_res_x == '0' and target_res_y == '0' else (float(target_res_x), float(target_res_y))

    processed = align_and_clip_rasters(
        dem_path=Path(config['Input']['dem_path']),
        forest_canopy_path=Path(config['Input']['forest_canopy_path']),
        output_dir=pra_dir,
        aoi_path=Path(config['General'].get('aoi_path')) if config['General'].get('aoi_path') else None,
        target_resolution=target_resolution,
        output_basename=config['General'].get('output_basename', 'autoATES')
    )

    dem_path = processed[0][0]
    forest_path = processed[0][1]
    step_times["preprocess"] = round((time.perf_counter() - _t) / 60, 2)
    logger.info(f"Step 1 completed - using DEM: {dem_path.name}")

    # 2. PRA
    pra_results = {}
    if config.getboolean('Processing', 'run_pra', fallback=True):
        logger.info("-" * 80)
        logger.info("Step 2: Running Potential Release Area mapping...")
        _t = time.perf_counter()

        for scen in ['frequent', 'extreme']:
            if config.getboolean('PRA', f'run_{scen}', fallback=True):
                logger.info(f"→ Computing {scen.capitalize()} PRA")
                pra_results[scen] = run_pra(
                    dem_path=dem_path,
                    forest_canopy_path=forest_path,
                    output_dir=pra_dir,
                    scenario=scen,
                    config=config
                )
        step_times["pra"] = round((time.perf_counter() - _t) / 60, 2)
        logger.info("Step 2 completed (%.1f min)", step_times["pra"])

    # 3. Runout
    runout_results = {}
    if config.getboolean('Processing', 'run_runout', fallback=True):
        logger.info("-" * 80)
        logger.info("Step 3: Running avalanche runout simulation...")

        for scen in ['frequent', 'extreme']:
            release_path = pra_dir / f"pra_{scen}_pra_sieve.tif"
            if not release_path.exists():
                logger.warning(f"Skipping {scen} runout - no PRA sieve found")
                continue

            forest_scale_path = pra_results.get(scen, {}).get("forest_scale") or (pra_dir / f"pra_{scen}_forest_scale.tif")

            logger.info(f"→ Running {scen} runout")
            t0 = time.perf_counter()
            runout_results[scen] = run_flowpy(
                dem_path=dem_path,
                release_path=release_path,
                scenario=scen,
                config=config,
                output_dir=flowpy_dir,
                forest_path=forest_scale_path
            )
            elapsed = time.perf_counter() - t0
            step_times[f"runout_{scen}"] = round(elapsed / 60, 2)
            logger.info(f"⏱  {scen} runout wall-clock: {elapsed:.1f} s ({elapsed/60:.1f} min)")
        logger.info("Step 3 completed")

    # 4. ATES
    ates_raster = None
    if config.getboolean('Processing', 'run_ates', fallback=True):
        logger.info("-" * 80)
        logger.info("Step 4: Running ATES classification...")
        _t = time.perf_counter()

        # dem_path / forest_path are the CLIPPED rasters from Step 1
        # (forest_path is canopy %, which ATES needs - not the 0-1 forest_scale).
        clipped_dem = dem_path
        clipped_forest = forest_path

        # Build the per-scenario runout + PRA inputs for the multi-scenario
        # classifier. Extreme defines the envelope; frequent drives inner classes.
        runout_by_scenario = {}
        pra_by_scenario = {}
        for scen in ("frequent", "extreme"):
            rr = runout_results.get(scen, {})
            if not all(rr.get(k) for k in ("zDelta", "routFluxSum", "fpTravelAngleMax")):
                continue
            runout_by_scenario[scen] = {
                "zDelta": rr.get("zDelta"),
                "routFluxSum": rr.get("routFluxSum"),
                "fpTravelAngleMax": rr.get("fpTravelAngleMax"),
                "forestInteraction": rr.get("forestInteraction"),
            }
            pra_by_scenario[scen] = {
                "sieve": pra_dir / f"pra_{scen}_pra_sieve.tif",
                "continuous": pra_dir / f"pra_{scen}_pra_continuous.tif",
            }

        if not runout_by_scenario:
            logger.warning("Skipping ATES - no scenario produced zDelta / "
                           "routFluxSum / fpTravelAngleMax.")
        else:
            aoi = config['General'].get('aoi_path')
            ates_raster = run_ates_classification(
                dem_path=clipped_dem,
                forest_path=clipped_forest,
                runout_by_scenario=runout_by_scenario,
                pra_by_scenario=pra_by_scenario,
                output_dir=ates_dir,
                config=config,
                aoi_path=Path(aoi) if aoi else None,
            )
            step_times["ates"] = round((time.perf_counter() - _t) / 60, 2)
            logger.info("Step 4 completed (%.1f min)", step_times["ates"])

    # 5. Colorize outputs for Mapbox (RGBA GeoTIFFs, one per product/scenario)
    if config.getboolean('Processing', 'run_colorize', fallback=True):
        logger.info("-" * 80)
        logger.info("Step 5: Colorizing outputs for web visualisation...")
        from autoates.comAutoATES.out1.colorize import colorize_outputs
        try:
            rgb = colorize_outputs(
                output_dir,
                pra_dir=pra_dir,
                flowpy_results=runout_results,
                forest_path=forest_path,
                ates_raster=ates_raster,
            )
            logger.info("Step 5 completed - %d RGB layers in %s", len(rgb), output_dir / "rgb")
        except Exception as exc:
            logger.error("Step 5 (colorize) failed: %s", exc, exc_info=True)

    logger.info("=" * 80)
    logger.info("✅ autoATES v3.0 workflow completed successfully!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Full log: {log_file.name}")
    logger.info("=" * 80)

    # _step_times / _outputs are consumed by runAutoATES() into the run manifest, so
    # a finished directory records not just its parameters but what it produced.
    return {"status": "success",
            "_step_times": step_times,
            "_outputs": {
                "dem_clipped": str(dem_path),
                "forest_clipped": str(forest_path),
                "ates_raster": str(ates_raster) if ates_raster else None,
                "pra": {s: {k: str(v) for k, v in r.items()}
                        for s, r in pra_results.items()},
                "runout": {s: {k: str(v) for k, v in r.items()}
                           for s, r in runout_results.items()},
            }}


if __name__ == "__main__":
    runAutoATES()