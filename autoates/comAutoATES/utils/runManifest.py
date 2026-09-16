"""runManifest.py - make every autoATES run reproducible from its own output dir.

THE PROBLEM THIS FIXES
----------------------
Until now a finished run recorded nothing about how it was produced. The four v3.0
benchmark-site stacks on disk (Chickadee 2026-06-10, Kootenays 06-26, Revelstoke 06-27,
NorthRockies 07-14) cannot be attributed to a parameter set at all: the log echoes
AvaFrame's ini paths but never the PRA or ATES parameters, no config was copied beside
the outputs, and com4FlowPy's `res_<uid>` directory is named after a hash of a config
that was never written down. So the uid identifies a configuration that no longer
exists anywhere. That makes those stacks unusable as a controlled comparison, which is
expensive: they are 275 km2 of finished computation.

Three things have to be captured, and the config file alone is none of them:

1. THE EFFECTIVE PARAMETERS, not the file. Every getter in the codebase has a code
   fallback (`config.getfloat(..., fallback=28.0)`), so behaviour is determined by the
   file AND the current source. A key absent from the ini still has a value. We
   therefore resolve parameters through the very functions the pipeline uses -
   get_scenario_params, get_segmentation_params, _get_params - so what is recorded is
   what actually ran, fallbacks included.
2. THE CODE VERSION. Same ini + different commit = different map. Records the autoates
   and AvaFrame commits, and whether either tree was dirty (a dirty tree means the run
   is NOT reproducible from the commit alone - it is flagged, not hidden).
3. THE INPUTS. DEM and canopy rasters get revised in place; a path is not an identity.
   Records size, mtime and a content hash.

Written at the START of a run, so a crashed or killed run still leaves its manifest,
then finalised on completion with timings and outputs.
"""
from __future__ import annotations

import configparser
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1
# Full-hash cap. Above this a head+tail fingerprint is recorded instead, labelled as
# such - an honest partial identity beats a 10-minute stall on a 30 GB provincial DEM.
_FULL_HASH_MAX_BYTES = 2 * 1024 ** 3
_FINGERPRINT_CHUNK = 8 * 1024 ** 2


def _run_git(repo: Path, *args) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_provenance(repo: Path) -> dict:
    """Commit, branch and dirty state of a working tree.

    `dirty` is the field that matters: a run made from uncommitted edits cannot be
    reproduced from its commit, so it is recorded explicitly rather than implied.
    """
    repo = Path(repo)
    sha = _run_git(repo, "rev-parse", "HEAD")
    if sha is None:
        return {"repo": str(repo), "available": False}
    status = _run_git(repo, "status", "--porcelain")
    return {"repo": str(repo), "available": True, "commit": sha,
            "branch": _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "describe": _run_git(repo, "describe", "--always", "--dirty", "--tags"),
            "dirty": bool(status), "dirty_files": len(status.splitlines()) if status else 0}


def _avaframe_repo() -> Path | None:
    try:
        import avaframe
        return Path(avaframe.__file__).resolve().parent.parent
    except Exception:
        return None


def package_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "rasterio", "geopandas", "shapely", "skimage",
                "numba", "pandas", "avaframe", "gdal"):
        try:
            m = __import__("osgeo.gdal" if mod == "gdal" else mod,
                           fromlist=["__version__"])
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    return out


def file_provenance(path) -> dict:
    """Size, mtime and content hash of one input raster."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    rec = {"path": str(p), "exists": True, "bytes": st.st_size,
           "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    h = hashlib.sha256()
    try:
        with open(p, "rb") as fh:
            if st.st_size <= _FULL_HASH_MAX_BYTES:
                for blk in iter(lambda: fh.read(1024 ** 2), b""):
                    h.update(blk)
                rec["sha256"] = h.hexdigest()
            else:
                h.update(fh.read(_FINGERPRINT_CHUNK))
                fh.seek(-min(_FINGERPRINT_CHUNK, st.st_size), os.SEEK_END)
                h.update(fh.read(_FINGERPRINT_CHUNK))
                rec["sha256_partial"] = h.hexdigest()
                rec["hash_note"] = ("head+tail fingerprint only - file exceeds "
                                    f"{_FULL_HASH_MAX_BYTES // 1024 ** 3} GiB")
    except OSError as e:
        rec["hash_error"] = str(e)
    return rec


def effective_params(config: configparser.ConfigParser) -> dict:
    """Resolve parameters THROUGH THE PIPELINE'S OWN GETTERS, so code fallbacks are
    captured. Imported locally to keep this module a leaf (no import cycle)."""
    from autoates.comAutoATES.pra.praCore import get_scenario_params
    from autoates.comAutoATES.pra.praSegmentation import get_segmentation_params
    from autoates.comAutoATES.ates.atesCore import _get_params as ates_params

    out: dict = {"pra": {}, "pra_segmentation": {}, "ates": {}, "flowpy": {}}
    for scen in ("frequent", "extreme"):
        try:
            out["pra"][scen] = get_scenario_params(scen, config)
        except Exception as e:                     # a required key may be absent
            out["pra"][scen] = {"error": f"{type(e).__name__}: {e}"}
        try:
            out["pra_segmentation"][scen] = get_segmentation_params(scen, config)
        except Exception as e:
            out["pra_segmentation"][scen] = {"error": f"{type(e).__name__}: {e}"}
    try:
        out["ates"] = ates_params(config)
    except Exception as e:
        out["ates"] = {"error": f"{type(e).__name__}: {e}"}
    # FlowPy's authoritative record is the resolved com4FlowPy cfg that runoutCore
    # dumps beside each res_<uid> directory (the uid IS its hash). Only the autoATES
    # side of the settings is echoed here.
    if config.has_section("FlowPy"):
        out["flowpy"] = dict(config["FlowPy"])
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_run_manifest(output_dir, config: configparser.ConfigParser,
                       config_path=None, timestamp: str | None = None,
                       input_paths=None, log_effective=True) -> Path:
    """Write `run_<ts>_config.ini` + `run_<ts>_manifest.json` into output_dir.

    Call at the start of a run: a killed run then still explains itself.
    Returns the manifest path, for finalize_run_manifest().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. the config as parsed (interpolation + inline comments already resolved)
    cfg_copy = output_dir / f"run_{ts}_config.ini"
    with open(cfg_copy, "w") as fh:
        fh.write(f"; autoATES resolved config, written {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"; source: {config_path or 'unknown (config object passed in)'}\n")
        fh.write("; NOTE: keys absent here still have values - code fallbacks are\n")
        fh.write(";       recorded under effective_params in the sibling manifest.json\n")
        config.write(fh)

    autoates_repo = Path(__file__).resolve().parents[3]
    params = effective_params(config)

    if input_paths is None:
        input_paths = [config.get("Input", k, fallback=None)
                       for k in ("dem_path", "forest_canopy_path")]
        aoi = config.get("General", "aoi_path", fallback=None)
        if aoi:
            input_paths.append(aoi)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run_timestamp": ts,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "status": "started",
        "project_name": config.get("General", "project_name", fallback=None),
        "output_dir": str(output_dir),
        "config_source": str(config_path) if config_path else None,
        "config_copy": cfg_copy.name,
        "environment": {
            "host": socket.gethostname(), "user": os.environ.get("USER"),
            "platform": platform.platform(), "cwd": os.getcwd(),
            "argv": sys.argv, "cpu_count": os.cpu_count(),
        },
        "code": {"autoates": git_provenance(autoates_repo),
                 "avaframe": git_provenance(_avaframe_repo()) if _avaframe_repo()
                 else {"available": False}},
        "packages": package_versions(),
        "inputs": [file_provenance(p) for p in input_paths if p],
        "effective_params": params,
    }

    path = output_dir / f"run_{ts}_manifest.json"
    with open(path, "w") as fh:
        json.dump(_jsonable(manifest), fh, indent=2, sort_keys=False)

    if log_effective:
        _log_effective(manifest)
    return path


def _log_effective(manifest: dict) -> None:
    """Echo the run's identity into the log, so the log alone is self-describing."""
    code = manifest["code"]
    for name in ("autoates", "avaframe"):
        c = code.get(name, {})
        if c.get("available"):
            flag = "  *** DIRTY TREE - not reproducible from this commit ***" \
                if c.get("dirty") else ""
            logger.info("code %-9s %s (%s)%s", name, (c.get("commit") or "")[:10],
                        c.get("branch"), flag)
        else:
            logger.warning("code %-9s NOT a git checkout - version unrecorded", name)
    for f in manifest["inputs"]:
        if f.get("exists"):
            logger.info("input %s  %.1f MB  %s  %s", Path(f["path"]).name,
                        f["bytes"] / 1e6, f["mtime"],
                        (f.get("sha256") or f.get("sha256_partial", ""))[:12])
        else:
            logger.error("input MISSING: %s", f["path"])
    p = manifest["effective_params"]
    for scen, d in p.get("pra", {}).items():
        if "error" in d:
            logger.error("PRA %s params unresolved: %s", scen, d["error"])
            continue
        logger.info("PRA %-8s thr=%s slope(a=%s b=%s c=%s) forest(a=%s b=%s c=%s) "
                    "wind(a=%s b=%s c=%s) sieve=%s", scen, d.get("pra_threshold"),
                    d.get("slope_a"), d.get("slope_b"), d.get("slope_c"),
                    d.get("forest_a"), d.get("forest_b"), d.get("forest_c"),
                    d.get("wind_a"), d.get("wind_b"), d.get("wind_c"),
                    d.get("sieve_filter_m2"))
    a = p.get("ates", {})
    if "error" not in a:
        logger.info("ATES base=%s pra1/2/3=%s/%s/%s sat=%s/%s/%s/%s tree=%s/%s/%s",
                    a.get("base_layer"), a.get("pra1"), a.get("pra2"), a.get("pra3"),
                    a.get("sat01"), a.get("sat12"), a.get("sat23"), a.get("sat34"),
                    a.get("tree1"), a.get("tree2"), a.get("tree3"))
    fp = p.get("flowpy", {})
    if fp:
        logger.info("FlowPy alpha freq=%s extreme=%s exp=%s engine=%s",
                    fp.get("frequent_alpha"), fp.get("extreme_alpha"),
                    fp.get("exponent"), fp.get("engine", "python (default)"))


def finalize_run_manifest(manifest_path, status="completed", **updates) -> None:
    """Record outcome, timings and produced outputs. Never raises - a manifest
    failure must not fail a finished 70-minute run."""
    try:
        path = Path(manifest_path)
        m = json.loads(path.read_text())
        m["status"] = status
        m["finished_at"] = datetime.now().isoformat(timespec="seconds")
        m.update(_jsonable(updates))
        path.write_text(json.dumps(m, indent=2, sort_keys=False))
    except Exception as e:
        logger.warning("could not finalise run manifest %s: %s", manifest_path, e)
