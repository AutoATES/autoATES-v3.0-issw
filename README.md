# autoATES v3.0 — ISSW 2026 snapshot

Public library snapshot of **autoATES v3.0** for the ISSW 2026 workshop
in Whistler. Default settings are the ISSW / western Canada working
version presented in the papers.

This is a snapshot of the library, not the development repository.
Calibration sweeps, site batches, and unpublished experiments stay
private while that tree is still being tested. See [SOURCE.md](SOURCE.md).

Hands-on notebooks and the Connaught Creek data live in
[issw2026-autoates-workshop](https://github.com/AutoATES/issw2026-autoates-workshop).

## What this code does

DEM + forest canopy → typical and infrequent potential release area
(PRA) → com4FlowPy runout (α 30° / 18°) → ATES classes 0–4.

Typical (config name `frequent`) is the tighter scenario. Infrequent
(`extreme`) is more inclusive, including start zones below 30°. Keep
both. Ruggedness is **off** on ~30 m surfaces.

## Run

You need Python 3.11, rasterio/GDAL, and [AvaFrame](https://github.com/OpenNHM/AvaFrame)
on `PYTHONPATH` (com4FlowPy). The workshop `SETUP.md` installs that stack.

```bash
export PYTHONPATH=/path/to/autoATES-v3.0-issw:/path/to/AvaFrame:$PYTHONPATH
cp autoates/comAutoATES/autoATESCfg.ini my_area.ini
# edit the three paths in my_area.ini  (AOI, DEM, canopy percent)
python run_autoates.py my_area.ini
```

`forest_type = sen2cc` means the forest raster is **canopy cover 0–100**,
not a binary mask.

## What to change for a local area

Change these. Leave the rest unless you have a mapped start-zone or ATES
reference to test against.

| Config key | What it is |
|---|---|
| `[General] aoi_path` | Polygon of the area you want classified |
| `[General] working_dir` | Where rasters are written |
| `[Input] dem_path` | Elevation. ISSW used ALOS AW3D30 (~30 m surface) |
| `[Input] forest_canopy_path` | Canopy percent 0–100, same idea as the Sentinel-2 product |

Optional, still “local data” rather than “new model”:

| Config key | When |
|---|---|
| `[General] target_resolution_*` | `0` keeps the DEM grid. Set only if you are resampling on purpose. |
| `[FlowPy] engine` | `numba` if installed (much faster); otherwise `python` |
| `[FlowPy] tile_size` | Raise if a large domain runs out of memory |

## What to leave as the ISSW defaults

These values are the working model in the papers. Changing them so the
map looks more like a local drawing is a different parameterization, and
it needs a validation set.

- `[PRA]` Cauchy parameters and typical/infrequent thresholds
- `[PRA] enable_ruggedness` — leave `False` on ~30 m ALOS. It is
  uncalibrated at that resolution and clipped conventional start zone
  on 5 m lidar at Connaught Creek.
- `[FlowPy] frequent_alpha` / `extreme_alpha` (30° / 18°)
- `[ATES]` floors and class thresholds

If your snow climate or forest is unlike western Canada, note that on
the map and keep these defaults until you have evidence to change them.
The design idea we want people to take home is that regional calibration
lives in the two scenarios, not in retuned class breaks.

## Citation

Sykes, J., Knies, D., Haegeli, P., Anthony-Malone, K., & Statham, G.
(2026). AutoATES v3.0: Automated ATES mapping at scale across western
Canada. *ISSW*, Whistler.

Maps produced with this snapshot are a research/tutorial product, not an
operational ATES map.

## License

MIT. See [LICENSE](LICENSE).
