"""
colorize.py - Mapbox-ready RGB(A) rasters for AutoATES v3.0 outputs.

Every model layer (ATES class, PRA, runout, slope, forest) is turned into a
4-band RGBA GeoTIFF via ``gdaldem color-relief`` driven by a palette file in
``palettes/``. Palettes are the single source of truth for colour + transparency
so they can be tuned without touching code, and the same file can be reused in
QGIS. Outputs are internally tiled with overviews and DEFLATE compression, ready
for ``gdal2tiles`` / ``rio pmtiles`` and direct Mapbox upload.

Palette formats (auto-detected by _normalize_palette):
  * native gdaldem stops:   value,R,G,B[,A]   (one colour per value)
  * QGIS "Generated Color Map Export File"    (headers + label column dropped)
  * ramp spec (``# @ramp ...``): a handful of anchor stops that get expanded into
    a dense stop list by interpolating in **LAB** colour space along a **log** or
    linear value domain. This lets gdaldem's linear RGB interpolation faithfully
    reproduce the AvCan designer's LAB/log ramps (slope, PRA, route-flux area).

Two colour-selection modes:
  * categorical (ATES class, PRA binary) -> exact_color_entry (no interpolation)
  * continuous  (PRA prob, slope, alpha, flux, canopy) -> linear interpolation

Continuous layers can be smoothed in place (2x2 bilinear/box, NoData-aware)
before colouring so the web map reads less blocky. Categorical layers are never
smoothed (keeps class boundaries crisp).

Public API:
  colorize_layer(src, out, palette, categorical=False, smooth=False)  -> Path
  colorize_outputs(...)                                               -> dict
  colorize_run_dir(run_dir)                                           -> dict
"""
from pathlib import Path
import math
import logging

import numpy as np
from osgeo import gdal

logger = logging.getLogger(__name__)
gdal.UseExceptions()

PALETTE_DIR = Path(__file__).parent / "palettes"

# GeoTIFF creation options: web-friendly, tiled, compressed.
_CREATION_OPTIONS = ["COMPRESS=DEFLATE", "TILED=YES", "PREDICTOR=1", "ZLEVEL=6"]

# Baseline opacity for baked overlays. 0.65 (~166/255) matches the AvCan viewer
# default so baked RGBA reads the same as the browser-coloured prototype.
DEFAULT_ALPHA = 166

# Reference map of the palettes shipped in palettes/ and whether each layer is
# categorical (exact colour match) or continuous (interpolated). runout_alpha is
# scenario-specific (see _scenario_palette); routflux_area is derived from
# routFluxSum (see _derive_flux_area).
LAYER_PALETTES = {
    "ates_class":              ("ates_class.txt",              True),
    "slope_angle":             ("slope_angle.txt",             False),
    "forest_canopy":           ("forest_canopy.txt",           False),
    "pra_continuous_frequent": ("pra_continuous_frequent.txt", False),
    "pra_continuous_extreme":  ("pra_continuous_extreme.txt",  False),
    "pra_binary_frequent":     ("pra_binary_frequent.txt",     True),   # teal
    "pra_binary_extreme":      ("pra_binary_extreme.txt",      True),    # pink
    "runout_alpha_frequent":   ("runout_alpha_frequent.txt",   False),
    "runout_alpha_extreme":    ("runout_alpha_extreme.txt",    False),
    "runout_alpha":            ("runout_alpha.txt",            False),   # generic fallback
    "zdelta":                  ("zdelta.txt",                  False),
    "routflux_area":           ("routflux_area.txt",           False),
}


# --------------------------------------------------------------------------- #
# Colour maths: sRGB <-> CIE-LAB, matched to the browser prototype's lerp so a
# baked ramp lands on the same path as the in-browser one.
# --------------------------------------------------------------------------- #
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883


def _hex_to_rgb(tok: str):
    s = tok.strip().lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _srgb_to_lin(c):
    c = c / 255.0
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def _rgb_to_lab(rgb):
    r, g, b = (_srgb_to_lin(v) for v in rgb)
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / _XN
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / _YN
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / _ZN
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def _lab_to_rgb(lab):
    L, a, b = lab
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    inv = lambda t: t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
    x, y, z = inv(fx) * _XN, inv(fy) * _YN, inv(fz) * _ZN
    r = x * 3.2406 - y * 1.5372 - z * 0.4986
    g = -x * 0.9689 + y * 1.8758 + z * 0.0415
    bb = x * 0.0557 - y * 0.2040 + z * 1.0570
    def lin_to_srgb(c):
        c = min(1.0, max(0.0, c))
        return 1.055 * c ** (1 / 2.4) - 0.055 if c > 0.0031308 else 12.92 * c
    return tuple(int(round(lin_to_srgb(v) * 255)) for v in (r, g, bb))


def _lerp(a, b, t):
    return a + (b - a) * t


def _mix(c0, c1, t, lab):
    if lab:
        la, lb = _rgb_to_lab(c0), _rgb_to_lab(c1)
        return _lab_to_rgb([_lerp(la[i], lb[i], t) for i in range(3)])
    return tuple(int(round(_lerp(c0[i], c1[i], t))) for i in range(3))


def _expand_ramp(anchors, *, scale="linear", lab=False, per_seg=14):
    """Expand few (value, rgb) anchors into a dense (value, rgb) stop list.

    Colour is interpolated in LAB (``lab=True``) or straight RGB; values are
    spaced linearly or geometrically (``scale='log'``) between adjacent anchors.
    """
    dense = []
    for (v0, c0), (v1, c1) in zip(anchors, anchors[1:]):
        for k in range(per_seg):
            t = k / per_seg
            val = math.exp(_lerp(math.log(v0), math.log(v1), t)) if scale == "log" \
                else _lerp(v0, v1, t)
            dense.append((val, _mix(c0, c1, t, lab)))
    dense.append((anchors[-1][0], anchors[-1][1]))
    return dense


def _parse_ramp(lines):
    """Return (params, anchors) for a ``# @ramp`` palette, or None.

    Directive:  ``# @ramp scale=log lab=1 per_seg=16 floor=100 alpha=166``
    Anchors:    ``value #rrggbb``  or  ``value R G B``  (one per line)
    """
    params, anchors, is_ramp = {}, [], False
    for raw in lines:
        line = raw.strip()
        low = line.lower().lstrip("# ").strip()
        if low.startswith("@ramp"):
            is_ramp = True
            for kv in low[len("@ramp"):].split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    params[k] = v
            continue
        if not line or line.startswith("#"):
            continue
        toks = [t for t in line.replace(",", " ").split() if t]
        if not toks:
            continue
        try:
            val = float(toks[0])
        except ValueError:
            continue
        if toks[1].startswith("#"):
            rgb = _hex_to_rgb(toks[1])
        else:
            rgb = (int(float(toks[1])), int(float(toks[2])), int(float(toks[3])))
        anchors.append((val, rgb))
    if not is_ramp:
        return None
    return params, anchors


def _normalize_palette(palette_path: Path, out_clr: Path, alpha_override=None) -> Path:
    """Rewrite any supported palette into a gdaldem-ready color file.

    Supported inputs:
      * ramp spec (``# @ramp ...``) -> expanded LAB/log dense stops
      * native gdaldem:  value,R,G,B[,A]   (comma or space)
      * QGIS "Generated Color Map Export File" (headers + label column dropped)

    A transparent NODATA ('nv') entry is appended if absent. alpha_override, if
    given, replaces every data entry's alpha (NODATA stays fully transparent).
    """
    lines = palette_path.read_text().splitlines()
    ramp = _parse_ramp(lines)

    if ramp is not None:
        params, anchors = ramp
        anchors.sort(key=lambda a: a[0])
        scale = params.get("scale", "linear")
        lab = params.get("lab", "0") not in ("0", "false", "False")
        per_seg = int(params.get("per_seg", 14))
        alpha = int(alpha_override if alpha_override is not None
                    else params.get("alpha", DEFAULT_ALPHA))
        floor = params.get("floor")
        dense = _expand_ramp(anchors, scale=scale, lab=lab, per_seg=per_seg)
        entries = []
        if floor is not None:
            # Everything below the floor is hidden (not-reached / below cutoff).
            floor = float(floor)
            r0, g0, b0 = anchors[0][1]
            entries.append(("0", r0, g0, b0, 0))
            entries.append((f"{floor * 0.999:.6g}", r0, g0, b0, 0))
        for val, (r, g, b) in dense:
            entries.append((f"{val:.6g}", r, g, b, alpha))
        entries.append(("nv", 0, 0, 0, 0))
        out_clr.parent.mkdir(parents=True, exist_ok=True)
        out_clr.write_text("\n".join(f"{v} {r} {g} {b} {a}" for v, r, g, b, a in entries) + "\n")
        return out_clr

    # ---- legacy / static stop list (native gdaldem or QGIS export) ----
    entries = []
    has_nv = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.upper().startswith("INTERPOLATION"):
            continue
        toks = [t for t in line.replace(",", " ").split() if t]
        if len(toks) < 4:
            continue
        val = toks[0]
        try:
            r, g, b = int(float(toks[1])), int(float(toks[2])), int(float(toks[3]))
        except ValueError:
            continue
        is_nv = val.lower() == "nv"
        if is_nv:
            has_nv = True
            a = 0
        else:
            a = int(float(toks[4])) if len(toks) >= 5 else 255
            if alpha_override is not None:
                a = int(alpha_override)
        entries.append((val, r, g, b, a))
    if not has_nv:
        entries.append(("nv", 0, 0, 0, 0))

    out_clr.parent.mkdir(parents=True, exist_ok=True)
    out_clr.write_text("\n".join(f"{v} {r} {g} {b} {a}" for v, r, g, b, a in entries) + "\n")
    return out_clr


def _shift(m, di, dj):
    """m shifted up by di / left by dj, zero-filled (di,dj in {0,1})."""
    out = np.zeros_like(m)
    h, w = m.shape
    out[:h - di, :w - dj] = m[di:, dj:]
    return out


def _smooth_raster(src_path, out_path) -> Path:
    """NoData-aware 2x2 bilinear/box smoothing of a continuous raster.

    Each cell becomes the mean of itself and its right/down/down-right neighbours,
    ignoring NoData in the window. Keeps grid size, geotransform and NoData; a
    cell with no valid neighbours stays NoData. Categorical layers must NOT be
    passed here (it would blur class edges).
    """
    src_path, out_path = Path(src_path), Path(out_path)
    ds = gdal.Open(str(src_path))
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype("float64")
    nodata = band.GetNoDataValue()
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= (arr != nodata)

    a = np.where(valid, arr, 0.0)
    w = valid.astype("float64")
    acc = np.zeros_like(a)
    cnt = np.zeros_like(w)
    for di in (0, 1):
        for dj in (0, 1):
            acc += _shift(a, di, dj)
            cnt += _shift(w, di, dj)
    fill = float(nodata) if nodata is not None else -9999.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, acc / np.maximum(cnt, 1), fill)
    out = np.where(valid, out, fill).astype("float32")

    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(str(out_path), ds.RasterXSize, ds.RasterYSize, 1,
                     gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "TILED=YES"])
    dst.SetGeoTransform(ds.GetGeoTransform())
    dst.SetProjection(ds.GetProjection())
    ob = dst.GetRasterBand(1)
    ob.WriteArray(out)
    ob.SetNoDataValue(fill)
    dst = None
    ds = None
    return out_path


def colorize_layer(src_path, out_path, palette, categorical: bool = False,
                   add_overviews: bool = True, alpha=None, smooth: bool = False) -> Path:
    """Colorize one single-band raster into an RGBA GeoTIFF using a palette file.

    categorical=True  -> exact colour match (class layers); never smoothed.
    categorical=False -> linear interpolation between palette stops (continuous).
    smooth=True       -> 2x2 bilinear/box pre-smooth (continuous only).
    palette accepts ramp specs, native gdaldem or QGIS-exported color files.
    alpha, if given (0-255), overrides every data entry's opacity.
    """
    src_path = Path(src_path)
    out_path = Path(out_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source raster not found: {src_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    palette_src = _resolve_palette(palette)
    # Normalize (ramp expand / QGIS export / alpha override) into a sidecar .clr.
    clr = _normalize_palette(palette_src, out_path.with_suffix(".clr"), alpha_override=alpha)

    dem_src = src_path
    tmp_smooth = None
    if smooth and not categorical:
        tmp_smooth = out_path.with_name(f".{out_path.stem}_smooth.tif")
        _smooth_raster(src_path, tmp_smooth)
        dem_src = tmp_smooth

    try:
        opts = gdal.DEMProcessingOptions(
            colorFilename=str(clr),
            addAlpha=True,
            colorSelection="exact_color_entry" if categorical else None,
            format="GTiff",
            creationOptions=_CREATION_OPTIONS,
        )
        gdal.DEMProcessing(str(out_path), str(dem_src), "color-relief", options=opts)
    finally:
        if tmp_smooth is not None and tmp_smooth.exists():
            tmp_smooth.unlink()

    if add_overviews:
        ds = gdal.Open(str(out_path), gdal.GA_Update)
        # Nearest keeps categorical edges crisp; average is smoother for continuous.
        ds.BuildOverviews("NEAREST" if categorical else "AVERAGE", [2, 4, 8, 16])
        ds = None

    logger.info("Colorized %s -> %s (%s%s, palette=%s)", src_path.name, out_path.name,
                "categorical" if categorical else "continuous",
                ", smoothed" if (smooth and not categorical) else "", palette_src.name)
    return out_path


def _resolve_palette(palette) -> Path:
    """Accept a palette name ('pra_continuous'), a bare filename, or a full path."""
    p = Path(palette)
    if p.exists():
        return p
    cand = PALETTE_DIR / palette
    if cand.exists():
        return cand
    cand = PALETTE_DIR / f"{palette}.txt"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Palette not found: {palette} (looked in {PALETTE_DIR})")


def _derive_flux_area(routfluxsum_path, out_path) -> Path:
    """routFluxArea = routFluxSum * cell_area (m^2), resolution-independent.

    com4FlowPy only writes routFluxSum (a per-cell routing sum whose magnitude
    scales with cell count / resolution). Multiplying by the cell area gives the
    physical contributing area, exactly as the ATES driver stack does. NODATA is
    preserved.
    """
    routfluxsum_path, out_path = Path(routfluxsum_path), Path(out_path)
    ds = gdal.Open(str(routfluxsum_path))
    gt = ds.GetGeoTransform()
    cell_area = abs(gt[1]) * abs(gt[5])
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype("float32")
    nodata = band.GetNoDataValue()
    area = arr * cell_area
    if nodata is not None:
        area[arr == nodata] = nodata
    out = gdal.GetDriverByName("GTiff").Create(
        str(out_path), ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES"])
    out.SetGeoTransform(gt)
    out.SetProjection(ds.GetProjection())
    ob = out.GetRasterBand(1)
    ob.WriteArray(area)
    if nodata is not None:
        ob.SetNoDataValue(float(nodata))
    out = None
    ds = None
    return out_path


def _try(name, src, out_dir, palette, categorical, results, alpha=None, smooth=False):
    """Colorize if src exists; skip-and-log otherwise. Never raises."""
    if src is None:
        return
    src = Path(src)
    if not src.exists():
        logger.warning("colorize: skipping %s - source missing (%s)", name, src)
        return
    try:
        out = colorize_layer(src, Path(out_dir) / f"{name}_rgb.tif", palette,
                             categorical, alpha=alpha, smooth=smooth)
        results[name] = out
    except Exception as exc:  # keep colorizing the rest of the layers
        logger.error("colorize: failed on %s (%s): %s", name, src, exc)


def _scenario_palette(base: str, scen: str) -> str:
    """Scenario-specific palette name, falling back to the generic one.

    e.g. base='runout_alpha', scen='extreme' -> 'runout_alpha_extreme' if that
    file exists, else 'runout_alpha'. Used for the layers whose colour/anchors
    differ by scenario: runout alpha (min-angle-matched Spectral), PRA continuous
    (per-scenario floor) and PRA binary (distinct scenario colour).
    """
    cand = f"{base}_{scen}"
    return cand if (PALETTE_DIR / f"{cand}.txt").exists() else base


def colorize_outputs(output_dir, *, pra_dir=None, flowpy_results=None,
                     forest_path=None, ates_raster=None, slope_path=None,
                     scenarios=("frequent", "extreme"), smooth: bool = True) -> dict:
    """Colorize the full AutoATES v3.0 product set into ``output_dir/rgb/``.

    Parameters mirror what runAutoATES already has in hand:
      pra_dir        : the PRA folder (holds pra_<scen>_pra_continuous/sieve.tif)
      flowpy_results : {scenario: {"zDelta":Path, "routFluxSum":Path,
                        "fpTravelAngleMax":Path, ...}} as returned by run_flowpy
      forest_path    : clipped canopy-% raster (0-100)
      ates_raster    : cleaned ATES_classification.tif
      slope_path     : slope-angle raster (deg), e.g. ATES/_slope_raw.tif
      smooth         : 2x2 pre-smooth continuous layers (categorical never smoothed)

    Missing inputs are skipped with a warning; returns {layer_name: Path}.
    """
    output_dir = Path(output_dir)
    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # Single-instance layers
    _try("ates_class", ates_raster, rgb_dir, "ates_class", True, results)
    _try("slope_angle", slope_path, rgb_dir, "slope_angle", False, results, smooth=smooth)
    _try("forest_canopy", forest_path, rgb_dir, "forest_canopy", False, results, smooth=smooth)

    # Per-scenario layers
    flowpy_results = flowpy_results or {}
    for scen in scenarios:
        if pra_dir is not None:
            pra_dir = Path(pra_dir)
            _try(f"pra_continuous_{scen}", pra_dir / f"pra_{scen}_pra_continuous.tif",
                 rgb_dir, _scenario_palette("pra_continuous", scen), False, results,
                 smooth=smooth)
            _try(f"pra_binary_{scen}", pra_dir / f"pra_{scen}_pra_sieve.tif",
                 rgb_dir, _scenario_palette("pra_binary", scen), True, results)

        rr = flowpy_results.get(scen, {})
        _try(f"runout_alpha_{scen}", rr.get("fpTravelAngleMax"),
             rgb_dir, _scenario_palette("runout_alpha", scen), False, results,
             smooth=smooth)
        _try(f"zdelta_{scen}", rr.get("zDelta"),
             rgb_dir, "zdelta", False, results, smooth=smooth)
        # Route flux AREA (resolution-independent): derive from routFluxSum, then
        # colorize. com4FlowPy does not emit an area layer directly.
        rfs = rr.get("routFluxSum")
        if rfs and Path(rfs).exists():
            try:
                area_tif = Path(rfs).with_name(f"{Path(rfs).stem}_area.tif")
                _derive_flux_area(rfs, area_tif)
                _try(f"routflux_area_{scen}", area_tif, rgb_dir,
                     "routflux_area", False, results, smooth=smooth)
            except Exception as exc:
                logger.error("colorize: routflux_area_%s failed: %s", scen, exc)

    logger.info("colorize_outputs: produced %d RGB layers in %s", len(results), rgb_dir)
    return results


def _find_flowpy(flowpy_dir: Path, scen: str, token: str):
    """Locate a com4FlowPy output tif for a scenario by lenient name token."""
    scen_dir = flowpy_dir / scen
    if not scen_dir.exists():
        return None
    key = token.lower().replace("_", "")
    matches = [p for p in scen_dir.rglob("com4_*.tif")
               if p.stem.lower().replace("_", "").endswith(key)]
    return sorted(matches)[-1] if matches else None


def colorize_run_dir(run_dir, scenarios=("frequent", "extreme"), smooth: bool = True) -> dict:
    """Colorize an existing AutoATES output directory after the fact.

    Discovers rasters by their on-disk layout (PRA/, FlowPy/<scen>/res_*/, ATES/)
    so it works on a completed run without the in-memory result dicts.
    """
    run_dir = Path(run_dir)
    pra_dir = run_dir / "PRA"
    flowpy_dir = run_dir / "FlowPy"
    ates_dir = run_dir / "ATES"

    ates_raster = next((p for p in sorted(ates_dir.glob("*classification*.tif"))
                        if not p.stem.endswith("_rgb")), None) \
        if ates_dir.exists() else None
    # slope raster written to ATES/ during runout prep
    slope_path = (ates_dir / "_slope_raw.tif") if ates_dir.exists() else None
    slope_path = slope_path if (slope_path and slope_path.exists()) else None
    # canopy-% raster produced by preprocessing lands in PRA/ as *_FOREST_CANOPY.tif
    forest_path = next(iter(sorted(pra_dir.glob("*FOREST_CANOPY*.tif"))), None) \
        if pra_dir.exists() else None

    flowpy_results = {}
    for scen in scenarios:
        flowpy_results[scen] = {
            "fpTravelAngleMax": _find_flowpy(flowpy_dir, scen, "fptravelanglemax"),
            "zDelta": _find_flowpy(flowpy_dir, scen, "zdelta"),
            "routFluxSum": _find_flowpy(flowpy_dir, scen, "routfluxsum"),
        }

    return colorize_outputs(run_dir, pra_dir=pra_dir if pra_dir.exists() else None,
                            flowpy_results=flowpy_results, forest_path=forest_path,
                            ates_raster=ates_raster, slope_path=slope_path,
                            scenarios=scenarios, smooth=smooth)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Colorize AutoATES v3.0 outputs for Mapbox.")
    ap.add_argument("run_dir", help="AutoATES output directory (contains PRA/, FlowPy/, ATES/)")
    ap.add_argument("--scenarios", nargs="+", default=["frequent", "extreme"])
    ap.add_argument("--no-smooth", action="store_true", help="disable 2x2 continuous smoothing")
    args = ap.parse_args()
    produced = colorize_run_dir(args.run_dir, scenarios=tuple(args.scenarios),
                                smooth=not args.no_smooth)
    print(f"\nProduced {len(produced)} RGB layers:")
    for name, path in sorted(produced.items()):
        print(f"  {name:24s} -> {path}")
