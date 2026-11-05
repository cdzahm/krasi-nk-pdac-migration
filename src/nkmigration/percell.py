"""One row per tracked cell, with every per-cell measure.

A measure is reported for a cell only if the cell meets that measure's
inclusion rule; otherwise the value is left empty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import states


@dataclass(frozen=True)
class Measure:
    name: str
    label: str
    count_col: str | None   # column the minimum applies to
    minimum: float


def _m(name, label, count_col="n_frames", minimum=3):
    return Measure(name, label, count_col, minimum)


_SHAPE = "n_shape_frames"
MEASURES = [
    _m("speed_um_per_min", "speed (µm/min)"),
    _m("arrest_coefficient", "arrest coefficient"),
    _m("net_displacement_um", "net displacement (µm)"),
    _m("path_length_um", "path length (µm)"),
    _m("confinement_ratio", "confinement ratio"),
    _m("turning_angle_deg", "turning angle (°)"),
    _m("axis_motion_angle_deg", "axis-motion angle (°)", "n_axis_motion_steps"),
    _m("axis_motion_frac_aligned", "fraction of steps along the long axis", "n_axis_motion_steps"),
    _m("area_um2_median", "cell area (µm²)", _SHAPE),
    _m("aspect_ratio_median", "aspect ratio", _SHAPE),
    _m("aspect_ratio_var", "aspect ratio variance", _SHAPE),
    _m("eccentricity_median", "eccentricity", _SHAPE),
    _m("equivalent_diameter_um_median", "equivalent diameter (µm)", _SHAPE),
    _m("solidity_median", "solidity", _SHAPE),
    _m("circularity_median", "circularity", _SHAPE),
    _m("elongation_mean", "mean aspect ratio", _SHAPE),
    _m("elongation_cv", "aspect ratio CV", _SHAPE),
    _m("state_transitions", "state transitions"),
    _m("state_transition_rate", "transition rate (per frame)"),
    _m("mean_dwell_frames", "mean dwell (frames)"),
    *[_m(f"orientation_acf_lag{k}", f"orientation autocorrelation, lag {k}", "n_frames", k + 1)
      for k in range(1, 11)],
    *[Measure(f"{s}_occupancy", f"{s.replace('_', ' ')} occupancy", "stated_frames",
              states.MIN_OCCUPANCY_FRAMES) for s in states.STATES.values()],
    Measure("motility_coefficient", "motility coefficient (µm²/min)", None, 0),
    Measure("speed_3d_prwm", "PRWM speed (µm/min)", None, 0),
]
BY_NAME = {m.name: m for m in MEASURES}


def apply_inclusion(cells: pd.DataFrame) -> pd.DataFrame:
    out = cells.copy()
    for m in MEASURES:
        if m.count_col is None:
            continue
        out[m.name] = out[m.name].where(out[m.count_col] >= m.minimum)
    # No episodes means the track was never given a state, not zero transitions.
    out.loc[out["stated_frames"].isna(), "state_transitions"] = np.nan
    out["motility_coefficient"] = out["motility_coefficient"].where(
        out["motility_coefficient"] > 0)
    return out
