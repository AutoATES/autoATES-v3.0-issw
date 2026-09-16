"""forestGaps.py - how big is the opening, not just how much canopy is there.

WHY (JM, 2026-08-10). Canopy fraction cannot express the thing that actually controls release
in forested terrain: **dense continuous forest and heavily gladed forest have the same canopy
percentage and completely different release potential.** A single canopy number therefore
forces the classifier into a bad choice - a flat ceiling that says "all Simple" or "all
Challenging" - which is exactly the artifact the dense-forest ceiling produced.

Gap DIMENSION is the standard control in forest-avalanche guidance (the Swiss NaiS forest
management rules state gap thresholds in metres along the fall line, not in canopy percent),
and the mechanism is physical: a gap has to be long enough for a slab to form and for a
fracture to propagate. So this module measures the opening, not the cover.

TWO METRICS, because the literature and the classifier want different things:

  gap_width_m   twice the distance from a cell to the nearest canopy, i.e. the diameter of
                the largest circle that fits in the opening at that point. Closest to the
                published thresholds, which are lengths.
  gap_area_ha   area of the connected open component the cell belongs to. This is JM's
                "potential avalanche size" proxy - a bigger clearing can produce a bigger
                avalanche, independent of how wide it is at any one point.

They are NOT redundant: a long narrow cut-block has large area and small width; a small round
clearing has the reverse.

THE ALPINE TRAP, and it must be handled by the caller. Above treeline "gap" is unbounded, so
both metrics saturate and an entire alpine basin reads as one enormous opening - which would
make this layer a treeline detector rather than a forest-structure measure. Two defences are
provided and at least one should always be used:
  * `max_area_ha` / `max_width_m` clamp the metrics, so alpine terrain sits at the ceiling
    rather than at an arbitrary huge value.
  * `enclosed_only=True` zeroes cells in components that touch the raster edge, on the
    reasoning that a true forest gap is bounded by forest on all sides within the AOI.
The cleanest defence is to read these layers jointly with elevation relative to treeline,
which autoATES already computes per AvCan subregion - not done here, because this module is
deliberately DEM-free.

RESOLUTION. Both metrics are in metres / hectares and are computed from the cell size, so
they are resolution-fair in the same sense as the MMU: a 40 m gap is a 40 m gap at 5 m and at
27 m. The CAVEAT is detectability, not units - a 27 m grid cannot resolve a gap narrower than
about two cells, so `gap_width_m` has a floor of ~2*cell and small gaps are simply invisible.
Report that rather than pretending the layer means the same thing at every resolution.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# Canopy percent below which a cell counts as OPEN for gap purposes. Deliberately NOT
# `open_max` (10%) from Table 6's forest-density bands: that threshold answers "is this cell
# unforested", and this one answers "would a slab find continuous snow here", which tolerates
# scattered stems. 30% is a starting value and is a parameter, not a published number.
OPEN_MAX_PCT = 30.0

_STRUCT8 = np.ones((3, 3), dtype=bool)


def gap_metrics(canopy_pct, cell, valid=None, open_max_pct=OPEN_MAX_PCT,
                max_width_m=500.0, max_area_ha=100.0, enclosed_only=False):
    """(gap_width_m, gap_area_ha), both zero where the cell is forested.

    `canopy_pct` is 0-100. `cell` is the pixel size in metres. Cells outside `valid` are
    treated as forest, so the AOI edge does not merge separate openings.
    """
    cp = np.nan_to_num(np.asarray(canopy_pct, "float32"), nan=100.0)
    openm = cp <= open_max_pct
    if valid is not None:
        openm &= valid

    # WIDTH. distance_transform_edt measures distance to the nearest ZERO, so on the open
    # mask it gives distance-to-canopy. Doubling turns a radius into a diameter, which is the
    # quantity the gap-size literature states.
    dist = ndimage.distance_transform_edt(openm, sampling=(cell, cell))
    width = np.where(openm, 2.0 * dist, 0.0).astype("float32")

    # AREA of the connected component, 8-connected because a diagonal string of openings is
    # one gap for snow purposes.
    lbl, n = ndimage.label(openm, structure=_STRUCT8)
    if n:
        counts = np.bincount(lbl.ravel())
        area_ha = (counts * cell * cell / 10000.0).astype("float32")
        area_ha[0] = 0.0                       # background label
        if enclosed_only:
            # A component touching the AOI edge is not demonstrably bounded by forest, so its
            # size is a lower bound at best. Zeroing is the conservative reading.
            edge = set(np.unique(np.concatenate([lbl[0, :], lbl[-1, :],
                                                 lbl[:, 0], lbl[:, -1]])))
            edge.discard(0)
            for e in edge:
                area_ha[e] = 0.0
        area = area_ha[lbl].astype("float32")
    else:
        area = np.zeros(cp.shape, dtype="float32")

    if max_width_m:
        width = np.minimum(width, max_width_m)
    if max_area_ha:
        area = np.minimum(area, max_area_ha)
    return width, area
