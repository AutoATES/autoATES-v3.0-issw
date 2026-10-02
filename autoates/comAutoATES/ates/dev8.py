"""dev8.py - the shipped ATES classifier (split votes, then reach floors).

This is the production model described in the private tree's
ATES_CLASSIFIER_SPEC.md at the snapshot commit. Cuts are the 2026-08-19
dev8 values. The 2026-08-27 no-ruggedness recut was not adopted.

`atesCore._classify` remains available as ATES.model = legacy. The pipeline
default is this module.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from autoates.comAutoATES.atesValidation.forestGaps import gap_metrics

# Cell size the area cuts were calibrated on. Runtime cuts scale by
# (cell / CUT_REF_CELL)^2 so "about one cell of flux" means the same thing
# at 5 m and at 21 m.
CUT_REF_CELL = 21.34
CUT_REF_CELL_AREA = CUT_REF_CELL ** 2

FLOOR3_FLUX_AREA = 500.0          # m^2 at the calibration grid
FLOOR4_SLOPE = 45.0
FLOOR4_CANOPY = 20.0              # percent

# (role_low, cut1, cut2, role_high, cut3). Low steps read the extreme
# scenario. The top step reads the typical scenario.
SPLIT_VOTES = {
    "pra":   ("E_pra",            0.0, 0.0087, "F_pra",            0.1613),
    "size":  ("E_pra_size",       0.0, 0.0136, "F_pra_size",       0.3086),
    "alpha": ("E_travel_angle",   0.0, 24.005, "F_travel_angle",   34.270),
    "flux":  ("E_rout_flux_area", 0.0, 10.299, "F_rout_flux_area", 375.18),
}
AREA_VOTES = ("flux",)

COARSE_CUTS = (8.029, 17.933, 31.419)
COARSE_WEIGHT = 1.0
COARSE_WINDOW_M = 300.0
COARSE_FLOORS = ((40.0, 3), (45.0, 4))

GAP_CUTS = (0.0, 54.9, 100.0)
GAP_WEIGHT = 2.0
GAP_CONTEXT_PCT = 30.0

CLASS0_T = 0.60
CLASS0_SCALE = dict(coarse=14.0, slope=20.0, runout_prox=200.0, pra_prox=500.0, forest=60.0)

# dev8 minimum mapping unit, m^2. Class 3 and 4 are smaller than the legacy ini.
MMU_M2 = {0: 10000.0, 1: 10000.0, 2: 5000.0, 3: 5000.0, 4: 2500.0}


def _win_cells(window_m: float, cell: float) -> int:
    """Odd cell count spanning `window_m` metres, minimum 3."""
    return max(3, int(round(window_m / cell)) | 1)


def _scale_area_cuts(cuts, cell: float):
    factor = (cell * cell) / CUT_REF_CELL_AREA
    return tuple(c * factor for c in cuts)


def _spoke(layers, role):
    return np.nan_to_num(layers[role], nan=0.0) > 0.0


def _split_vote(layers, spec, use_lo=True):
    """0-3. Extreme owns the two low steps. Typical owns the Complex step."""
    role_lo, c1, c2, role_hi, c3 = spec
    hi = np.nan_to_num(layers[role_hi], nan=0.0)
    if use_lo and role_lo in layers:
        lo = np.nan_to_num(layers[role_lo], nan=0.0)
    else:
        lo = np.zeros(hi.shape, dtype=hi.dtype)
    vote = np.zeros(lo.shape, dtype="int8")
    vote[lo > c1] = 1
    vote[lo > c2] = 2
    vote[hi > c3] = 3
    return vote


def _ordinal(values, cuts):
    out = np.zeros(np.shape(values), dtype="int8")
    data = np.nan_to_num(values, nan=0.0)
    for cut in cuts:
        out += (data > cut).astype("int8")
    return out


def _coarse_slope(slope, cell):
    filled = np.nan_to_num(np.asarray(slope, "float32"), nan=0.0)
    return ndimage.uniform_filter(filled, _win_cells(COARSE_WINDOW_M, cell))


def _class0_score(slope, canopy_pct, coarse, runout_prox, pra_prox):
    scale = CLASS0_SCALE
    factors = [
        np.clip((scale["coarse"] - np.nan_to_num(coarse, nan=0.0)) / scale["coarse"], 0, 1),
        np.clip((scale["slope"] - np.nan_to_num(slope, nan=0.0)) / scale["slope"], 0, 1),
        np.clip(np.nan_to_num(canopy_pct, nan=0.0) / scale["forest"], 0, 1),
    ]
    if runout_prox is not None:
        distance = np.nan_to_num(runout_prox, nan=1e4)
        factors.append(np.clip(distance / scale["runout_prox"], 0, 1))
    if pra_prox is not None:
        distance = np.nan_to_num(pra_prox, nan=1e4)
        factors.append(np.clip(distance / scale["pra_prox"], 0, 1))
    return np.mean(factors, axis=0)


def _floors(layers, slope, canopy_pct, valid, cell):
    """Minimum class. Returns the floor and the Complex flux threshold in m^2."""
    floor = np.zeros(valid.shape, dtype="int8")
    extreme_reach = np.nan_to_num(layers["E_zdelta"], nan=0.0) > 0.0
    typical_reach = np.nan_to_num(layers["F_zdelta"], nan=0.0) > 0.0
    floor[extreme_reach] = np.maximum(floor[extreme_reach], 1)
    floor[typical_reach] = np.maximum(floor[typical_reach], 2)

    flux_cut = _scale_area_cuts((FLOOR3_FLUX_AREA,), cell)[0]
    frequent_flux = np.nan_to_num(layers["F_rout_flux_area"], nan=0.0) >= flux_cut
    floor[frequent_flux] = np.maximum(floor[frequent_flux], 3)

    steep_open = (np.nan_to_num(slope, nan=0.0) > FLOOR4_SLOPE) & (
        np.nan_to_num(canopy_pct, nan=0.0) <= FLOOR4_CANOPY)
    floor[steep_open] = np.maximum(floor[steep_open], 4)
    floor[~valid] = 0
    return floor, flux_cut


def classify(layers, valid, slope, canopy_pct, cell):
    """Shipped dev8 over one grid.

    Parameters
    ----------
    layers : dict
        Role arrays on the DEM grid. Required: E_pra, F_pra, E_zdelta,
        F_zdelta, E_travel_angle, F_travel_angle, E_rout_flux_area,
        F_rout_flux_area. Optional: E_pra_size, F_pra_size, E_runout_prox,
        E_pra_proximity.
    valid : bool array
    slope : degrees
    canopy_pct : 0-100
    cell : metres

    Returns
    -------
    int8 array, classes 0-4 on valid cells and -1 outside.
    """
    valid = np.asarray(valid, dtype=bool)
    slope = np.nan_to_num(np.asarray(slope, "float32"), nan=0.0)
    canopy_pct = np.nan_to_num(np.asarray(canopy_pct, "float32"), nan=0.0)
    work = dict(layers)
    work["coarse_slope"] = _coarse_slope(slope, cell)
    _, gap_area = gap_metrics(canopy_pct, cell, valid=valid)
    work["gap_area_ha"] = gap_area

    num = np.zeros(valid.shape, dtype="float64")
    den = np.zeros(valid.shape, dtype="float64")

    for name, spec in SPLIT_VOTES.items():
        if spec[3] not in work:
            continue
        role_lo, c1, c2, role_hi, c3 = spec
        if name in AREA_VOTES:
            c1, c2, c3 = _scale_area_cuts((c1, c2, c3), cell)
        voted = (role_lo, c1, c2, role_hi, c3)
        use_lo = spec[0] in work
        spoke = _spoke(work, spec[3])
        if use_lo:
            spoke = spoke | _spoke(work, spec[0])
        mask = valid & spoke
        values = _split_vote(work, voted, use_lo=use_lo).astype("float64")
        num[mask] += values[mask]
        den[mask] += 1.0

    coarse_vote = _ordinal(work["coarse_slope"], COARSE_CUTS).astype("float64")
    num[valid] += COARSE_WEIGHT * coarse_vote[valid]
    den[valid] += COARSE_WEIGHT

    neighbourhood = ndimage.uniform_filter(canopy_pct, _win_cells(COARSE_WINDOW_M, cell))
    gap_mask = valid & (neighbourhood >= GAP_CONTEXT_PCT)
    gap_vote = _ordinal(work["gap_area_ha"], GAP_CUTS).astype("float64")
    num[gap_mask] += GAP_WEIGHT * gap_vote[gap_mask]
    den[gap_mask] += GAP_WEIGHT

    tier_b = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    classes = np.clip(np.rint(tier_b), 0, 4).astype("int8")

    floor, _flux_cut = _floors(work, slope, canopy_pct, valid, cell)
    classes = np.maximum(classes, floor).astype("int8")

    coarse = work["coarse_slope"]
    for degrees, klass in COARSE_FLOORS:
        raise_it = valid & (coarse >= degrees) & (classes < klass)
        classes[raise_it] = klass

    score = _class0_score(
        slope, canopy_pct, coarse,
        work.get("E_runout_prox"), work.get("E_pra_proximity"))
    classes[valid & (score >= CLASS0_T)] = 0
    classes[~valid] = -1
    return classes
