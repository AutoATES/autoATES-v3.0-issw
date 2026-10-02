"""Lock the two dev8 rules that separated the workshop map from the AvCan map.

Run from anywhere:

    python tests/test_dev8.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoates.comAutoATES.ates.dev8 import classify  # noqa: E402


def _layers(shape, **overrides):
    layers = {
        "E_pra": np.zeros(shape, dtype="float32"),
        "F_pra": np.zeros(shape, dtype="float32"),
        "E_zdelta": np.zeros(shape, dtype="float32"),
        "F_zdelta": np.zeros(shape, dtype="float32"),
        "E_travel_angle": np.zeros(shape, dtype="float32"),
        "F_travel_angle": np.zeros(shape, dtype="float32"),
        "E_rout_flux_area": np.zeros(shape, dtype="float32"),
        "F_rout_flux_area": np.zeros(shape, dtype="float32"),
    }
    layers.update(overrides)
    return layers


def _grid(cell, slope=10.0, canopy=0.0, n=8):
    """Constant terrain. Canopy 0 keeps the gap vote silent (alpine abstention).

    Slope 10 votes coarse class 1, stays under the 40 degree coarse floor, and
    keeps the class-0 score under 0.60 when canopy is 0.
    """
    shape = (n, n)
    valid = np.ones(shape, dtype=bool)
    slope_a = np.full(shape, slope, dtype="float32")
    canopy_a = np.full(shape, canopy, dtype="float32")
    return shape, valid, slope_a, canopy_a


def _center(classes):
    return int(classes[classes.shape[0] // 2, classes.shape[1] // 2])


def test_typical_reach_is_at_least_challenging():
    shape, valid, slope, canopy = _grid(21.34)
    layers = _layers(shape, F_zdelta=np.full(shape, 10.0, dtype="float32"))
    got = _center(classify(layers, valid, slope, canopy, 21.34))
    assert got == 2, got


def test_complex_flux_floor_scales_with_cell_area():
    # At the calibration cell the floor is 500 m^2. The flux vote's Complex
    # step is 375 m^2, so a 600 m^2 cell is Complex only because the floor
    # lifts the rounded vote.
    shape, valid, slope, canopy = _grid(21.34)
    quiet = _layers(shape, F_zdelta=np.full(shape, 10.0, dtype="float32"))
    assert _center(classify(quiet, valid, slope, canopy, 21.34)) == 2

    loud = _layers(
        shape,
        F_zdelta=np.full(shape, 10.0, dtype="float32"),
        F_rout_flux_area=np.full(shape, 600.0, dtype="float32"),
    )
    assert _center(classify(loud, valid, slope, canopy, 21.34)) == 3, _center(
        classify(loud, valid, slope, canopy, 21.34))

    # At 5 m the same physical floor is about 27 m^2. 10 m^2 stays Challenging.
    # 30 m^2 is Complex. An unscaled 500 m^2 floor would leave 30 m^2 at class 2.
    shape, valid, slope, canopy = _grid(5.0)
    small = _layers(
        shape,
        F_zdelta=np.full(shape, 10.0, dtype="float32"),
        F_rout_flux_area=np.full(shape, 10.0, dtype="float32"),
    )
    assert _center(classify(small, valid, slope, canopy, 5.0)) == 2, _center(
        classify(small, valid, slope, canopy, 5.0))
    enough = _layers(
        shape,
        F_zdelta=np.full(shape, 10.0, dtype="float32"),
        F_rout_flux_area=np.full(shape, 30.0, dtype="float32"),
    )
    assert _center(classify(enough, valid, slope, canopy, 5.0)) == 3, _center(
        classify(enough, valid, slope, canopy, 5.0))


def test_extreme_reach_is_at_least_simple():
    # Slope 0 and proximity 0: the class-0 score stays under 0.60, and the
    # coarse vote is 0, so class 1 can only come from the extreme-reach floor.
    shape, valid, slope, canopy = _grid(21.34, slope=0.0)
    layers = _layers(
        shape,
        E_zdelta=np.full(shape, 10.0, dtype="float32"),
        E_runout_prox=np.zeros(shape, dtype="float32"),
        E_pra_proximity=np.zeros(shape, dtype="float32"),
    )
    got = _center(classify(layers, valid, slope, canopy, 21.34))
    assert got == 1, got


def test_class0_ceiling_overrides_a_reach_floor():
    # Flat, dense, and far from both the extreme runout and the extreme
    # start zones. The score clears 0.60 and puts the cell back to class 0
    # after the typical-reach floor has raised it.
    shape, valid, slope, canopy = _grid(21.34, slope=0.0, canopy=100.0)
    layers = _layers(
        shape,
        F_zdelta=np.full(shape, 10.0, dtype="float32"),
        E_runout_prox=np.full(shape, 500.0, dtype="float32"),
        E_pra_proximity=np.full(shape, 500.0, dtype="float32"),
    )
    got = _center(classify(layers, valid, slope, canopy, 21.34))
    assert got == 0, got


def test_open_cliff_is_extreme_without_the_coarse_floor():
    # One steep open cell in a gentle neighbourhood. The 300 m mean stays
    # near 10 degrees, so class 4 is the local slope-and-canopy floor.
    n = 31
    shape, valid, slope, canopy = _grid(21.34, slope=10.0, canopy=10.0, n=n)
    mid = n // 2
    slope = slope.copy()
    slope[mid, mid] = 50.0
    layers = _layers(shape)
    classes = classify(layers, valid, slope, canopy, 21.34)
    assert int(classes[mid, mid]) == 4, int(classes[mid, mid])
    assert int(classes[0, 0]) != 4


def main():
    test_typical_reach_is_at_least_challenging()
    test_complex_flux_floor_scales_with_cell_area()
    test_extreme_reach_is_at_least_simple()
    test_class0_ceiling_overrides_a_reach_floor()
    test_open_cliff_is_extreme_without_the_coarse_floor()
    print("dev8 synthetic tests passed")


if __name__ == "__main__":
    main()
