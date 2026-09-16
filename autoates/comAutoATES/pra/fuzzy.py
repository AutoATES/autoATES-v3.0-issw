"""
fuzzy.py
Cauchy membership function + fuzzy-AND (OWA) aggregation.

Vendored from the RA (jzheren) PRA implementation, which is the canonical core
for autoATES v3.0. Kept as a standalone module so the membership math is shared
verbatim between the PRA core and any reconciliation/equivalence tests.
"""
import numpy as np


def cauchy_membership_function(x, *args):
    """Cauchy curve:  1 / (1 + ((x - c) / a) ** (2b)).

    Accepts either a single [a, b, c] sequence or three positional args.
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        a, b, c = args[0]
    elif len(args) == 3:
        a, b, c = args
    else:
        raise ValueError("Expects a list [a, b, c] or three args a, b, c.")
    return 1 / (1 + ((x - c) / a) ** (2 * b))


def cauchy_membership_asymmetric(x, a_low, b_low, c, a_high=None, b_high=None):
    """Two-sided Cauchy curve: independent width/exponent below and above ``c``.

    The symmetric Cauchy cannot admit low-angle terrain without also admitting
    the mirror-image steep terrain (widening to reach 28-32 deg necessarily
    reaches 58-62 deg). This variant uses ``a_low``/``b_low`` for x < c and
    ``a_high``/``b_high`` for x >= c, so the low-angle shoulder and the cliff
    cutoff can be tuned independently.

    ``a_high``/``b_high`` default to the low-side values, in which case the
    result is bit-identical to :func:`cauchy_membership_function` -- the
    symmetric path (and the RA-canonical core equivalence) is unaffected.

    Keeps the signed ``(x - c)`` offset of the original expression so that
    equal-width parameters reproduce it exactly rather than to within a ULP.
    As in the original, ``b`` must be an integer: the exponent ``2b`` has to be
    even for the negative (below-centre) offsets to stay real.
    """
    a_high = a_low if a_high is None else a_high
    b_high = b_low if b_high is None else b_high
    below = 1 / (1 + ((x - c) / a_low) ** (2 * b_low))
    above = 1 / (1 + ((x - c) / a_high) ** (2 * b_high))
    return np.where(x < c, below, above)


def fuzzy_AND(rasters: list):
    """Ordered-weighted fuzzy AND used by the original PRA model.

        (1 - min) * min  +  min * mean(rasters)

    Generalises over N membership rasters (3 without ruggedness, 4 with).
    """
    min_vals = np.minimum.reduce(rasters)
    return (1 - min_vals) * min_vals + min_vals * np.sum(rasters, axis=0) / len(rasters)
