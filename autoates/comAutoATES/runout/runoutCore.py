"""
runoutCore.py - In-process com4FlowPy runner for autoATES v3.0

Calls AvaFrame's com4FlowPy directly through its public Python API
(getModuleConfig + com4FlowPyMain), mirroring the `useCustomPaths` branch of
avaframe/runCom4FlowPy.py. This avoids:
  * any hardcoded path to an AvaFrame clone (avaframe is imported as a package),
  * the subprocess call to runCom4FlowPy.py (whose --cfg arg is ignored), and
  * mutating AvaFrame's own local_com4FlowPyCfg.ini.

The config is built in memory from the module defaults (onlyDefault=True), so a
stale local_com4FlowPyCfg.ini left in the AvaFrame install cannot affect a run.
"""
from pathlib import Path
import logging
import configparser
import shutil
import psutil

from avaframe.com4FlowPy import com4FlowPy
from avaframe.in3Utils import cfgUtils
from avaframe.runCom4FlowPy import checkOutputFilesFormat

logger = logging.getLogger(__name__)

# Default com4FlowPy outputs. The ATES step (v2.10 logic) needs zDelta,
# routFluxSum and fpTravelAngleMax; cellCounts is kept for QA.
DEFAULT_OUTPUT_FILES = "zDelta|routFluxSum|fpTravelAngleMax|cellCounts"


def _build_flowpy_config(dem_path: Path, release_path: Path, forest_path: Path,
                         scenario: str, work_dir: Path,
                         config: configparser.ConfigParser) -> configparser.ConfigParser:
    """Build a com4FlowPy ConfigParser from autoATES [FlowPy] settings.

    Starts from com4FlowPy's own defaults (onlyDefault=True ignores any
    local_com4FlowPyCfg.ini on disk) and overrides only the keys we care about.
    """
    cfg = cfgUtils.getModuleConfig(com4FlowPy, onlyDefault=True, toPrint=False)

    scenario_key = "frequent" if scenario.lower() == "frequent" else "extreme"
    default_alpha = 28.0 if scenario_key == "frequent" else 18.0

    fp = "FlowPy"  # section in autoATESCfg.ini
    gen = cfg["GENERAL"]
    gen["alpha"] = str(config.getfloat(fp, f"{scenario_key}_alpha", fallback=default_alpha))
    gen["exp"] = str(config.getfloat(fp, "exponent", fallback=8.0))
    gen["flux_threshold"] = str(config.getfloat(fp, "flux_threshold", fallback=0.003))
    gen["max_z"] = str(config.getfloat(fp, "max_z", fallback=270.0))
    gen["forest"] = "True"
    gen["forestModule"] = config.get(fp, "forest_module", fallback="forestDetrainment")
    gen["forestInteraction"] = str(config.getboolean(fp, "forest_interaction", fallback=True))
    gen["maxAddedFrictionFor"] = str(config.getfloat(fp, "max_added_friction", fallback=52.0))
    gen["minAddedFrictionFor"] = str(config.getfloat(fp, "min_added_friction", fallback=5.0))
    gen["velThForFriction"] = str(config.getfloat(fp, "vel_th_for_friction", fallback=270.0))
    gen["maxDetrainmentFor"] = str(config.getfloat(fp, "max_detrainment", fallback=0.003))
    gen["minDetrainmentFor"] = str(config.getfloat(fp, "min_detrainment", fallback=0.00001))
    gen["velThForDetrain"] = str(config.getfloat(fp, "vel_th_for_detrainment", fallback=270.0))
    gen["forestFrictionLayerType"] = config.get(fp, "f_fr_layer_type", fallback="absolute")
    gen["skipForestDist"] = str(config.getfloat(fp, "skip_forest_dist", fallback=0.0))
    gen["tileSize"] = str(config.getint(fp, "tile_size", fallback=15000))
    gen["tileOverlap"] = str(config.getint(fp, "tile_overlap", fallback=5000))
    # compute engine: "numba" (JIT kernel, default) or "python" (reference).
    # The numba engine is bit-identical to python at 21.3 m (verified 2026-08-28 on
    # Chickadee: 0 differing px on zdelta/routFluxSum/fpTravelAngleMax/cellCounts,
    # except 1 px at 9.5e-06 on extreme routFluxSum) and ~20x faster end-to-end.
    # com4FlowPy falls back to python by itself if numba is not installed.
    gen["engine"] = config.get(fp, "engine", fallback="numba")

    paths = cfg["PATHS"]
    paths["useCustomPaths"] = "True"
    # auto-delete the per-run tiling temp folder after outputs are written
    # (intermediate tiles only; the final com4_*.tif rasters are kept).
    paths["deleteTempFolder"] = "True"
    paths["workDir"] = str(work_dir)
    paths["demPath"] = str(dem_path)
    paths["releasePath"] = str(release_path)
    paths["forestPath"] = str(forest_path)
    paths["outputFiles"] = checkOutputFilesFormat(
        config.get(fp, "output_files", fallback=DEFAULT_OUTPUT_FILES)
    )

    return cfg


def run_flowpy(dem_path: Path, release_path: Path, scenario: str,
               config: configparser.ConfigParser, output_dir: Path,
               forest_path: Path = None) -> dict:
    """Run com4FlowPy for one scenario in-process and return output raster paths.

    Returns a dict mapping output name (e.g. "zDelta") to the produced .tif Path,
    plus status/scenario/res_dir metadata.
    """
    logger.info(f"→ Starting {scenario.upper()} runout simulation (in-process com4FlowPy)")

    if forest_path is None:
        forest_path = dem_path.parent / f"pra_{scenario}_forest_scale.tif"

    # Each scenario gets its own work dir under the FlowPy output folder.
    work_dir = output_dir / scenario
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_flowpy_config(dem_path, release_path, forest_path,
                               scenario, work_dir, config)

    # uid is derived from the full config; identical params -> identical uid.
    uid = cfgUtils.cfgHash(cfg)
    res_dir = work_dir / f"res_{uid}"
    # Clean a stale results folder so com4FlowPy does not abort on "already exists".
    if res_dir.exists():
        logger.info(f"Removing stale results folder {res_dir.name}")
        shutil.rmtree(res_dir, ignore_errors=True)
    temp_dir = res_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Persist the config the uid was hashed FROM. Without this the directory name is
    # a hash of something that exists nowhere: the four v3.0 benchmark-site stacks on
    # disk have res_<uid> folders whose parameters cannot be recovered at all. Written
    # before the simulation so a killed run still explains itself.
    cfg_dump = res_dir / f"com4FlowPyCfg_used_{uid}.ini"
    try:
        with open(cfg_dump, "w") as fh:
            fh.write(f"; resolved com4FlowPy config for scenario '{scenario}'\n")
            fh.write(f"; uid = cfgHash(this config) = {uid}\n")
            fh.write(f"; dem={dem_path}\n; release={release_path}\n")
            fh.write(f"; forest={forest_path}\n")
            cfg.write(fh)
        logger.info(f"{scenario}: com4FlowPy config recorded -> {cfg_dump.name}")
    except OSError as e:
        logger.warning(f"could not write {cfg_dump.name}: {e}")

    timeString = config.get("FlowPy", "_run_timestamp", fallback="run")

    cfgMain = cfgUtils.getGeneralConfig()
    cfgSetup = cfg["GENERAL"]
    # Cap at physical cores: hyperthreads give no speedup for the compute-bound BFS
    # and would ~double peak RAM. min() preserves any lower user-configured nCPU.
    _physical = psutil.cpu_count(logical=False) or 1
    cfgSetup["cpuCount"] = str(min(cfgUtils.getNumberOfProcesses(cfgMain, 9999), _physical))

    paths = cfg["PATHS"]
    cfgPath = {
        "workDir": work_dir,
        "outDir": res_dir,
        "resDir": res_dir,
        "tempDir": temp_dir,
        "demPath": Path(dem_path),
        "releasePath": Path(release_path),
        # Optional path keys: read defensively so the integration works across
        # AvaFrame versions whose default com4FlowPyCfg.ini may omit some of them
        # (e.g. master has no 'relIdPath'; it is a branch-only addition).
        "relIdPath": Path(paths.get("relIdPath") or ""),
        "infraPath": Path(paths.get("infraPath") or ""),
        "forestPath": Path(forest_path),
        "varUmaxPath": Path(paths.get("varUmaxPath") or ""),
        "varAlphaPath": Path(paths.get("varAlphaPath") or ""),
        "varExponentPath": Path(paths.get("varExponentPath") or ""),
        "deleteTemp": paths["deleteTempFolder"],
        "outputFileFormat": paths["outputFileFormat"],
        "outputFiles": paths["outputFiles"],
        "outputNoDataValue": paths.getfloat("outputNoDataValue"),
        "useCompression": paths.getboolean("useCompression"),
        "customDirs": "True",
        "uid": uid,
        "timeString": timeString,
    }

    logger.info(f"com4FlowPy: alpha={cfgSetup['alpha']}, exp={cfgSetup['exp']}, "
                f"tileSize={cfgSetup['tileSize']}, outputs={cfgPath['outputFiles']}")

    try:
        com4FlowPy.com4FlowPyMain(cfgPath, cfgSetup)
    except Exception as exc:
        logger.error(f"❌ {scenario} runout failed: {exc}", exc_info=True)
        return {"status": "failed", "scenario": scenario}

    # Collect produced rasters. com4FlowPy writes "com4_{uid}_{ts}_<name>.tif",
    # but the <name> token's casing/separators can vary, so match leniently.
    all_tifs = list(res_dir.glob("com4_*.tif"))
    outputs = {}
    for name in cfgPath["outputFiles"].split("|"):
        token = name.lower().replace("_", "")
        matches = [p for p in all_tifs
                   if p.stem.lower().replace("_", "").endswith(token)]
        if matches:
            outputs[name] = sorted(matches)[-1]

    # forestInteraction is auto-produced when forestInteraction=True but is not
    # part of outputFiles; surface it so ATES can use it as a driver layer.
    fi = [p for p in all_tifs if p.stem.lower().replace("_", "").endswith("forestinteraction")]
    if fi:
        outputs["forestInteraction"] = sorted(fi)[-1]

    if outputs:
        logger.info(f"✅ {scenario.upper()} runout completed → "
                    + ", ".join(f"{k}={v.name}" for k, v in outputs.items()))
    else:
        logger.warning(f"{scenario} runout finished but no output rasters were found in {res_dir}")

    return {"status": "success", "scenario": scenario, "res_dir": res_dir, **outputs}
