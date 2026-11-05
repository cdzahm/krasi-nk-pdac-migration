"""Per-track shape and motion measures.

Motion uses XY displacements between consecutive frames. Shape uses only
frames where the cell does not touch the image border. A step shorter than
one pixel counts as arrested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACF_LAGS = range(1, 11)


def _axis_motion_angle(orient: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Angle between the long axis (mod 180) and the step direction, in [0, 90] deg."""
    d = np.degrees(np.abs(np.arctan2(dy, dx) - orient)) % 180.0
    return np.where(d > 90.0, 180.0 - d, d)


def _median(v) -> float:
    return float(np.median(v)) if len(v) else np.nan


def track_measures(g: pd.DataFrame, pixel_um: float, frame_min: float) -> dict:
    """All per-track measures for one track, sorted by t."""
    t = g["t"].to_numpy()
    y, x = g["y_um"].to_numpy(float), g["x_um"].to_numpy(float)
    shape_ok = g["use_for_shape"].to_numpy(bool)
    out = {"n_frames": len(g), "n_shape_frames": int(shape_ok.sum())}

    ar = g.loc[shape_ok, "aspect_ratio"].to_numpy(float)
    out["aspect_ratio_median"] = _median(ar)
    out["aspect_ratio_var"] = float(np.var(ar, ddof=1)) if len(ar) > 1 else np.nan
    for col in ("eccentricity", "area_um2", "equivalent_diameter_um", "solidity", "circularity"):
        out[f"{col}_median"] = _median(g.loc[shape_ok, col].to_numpy(float))
    out["elongation_mean"] = float(np.mean(ar)) if len(ar) else np.nan
    ar_fit = ar[ar < 100]  # a degenerate ellipse fit can give ~1e7
    out["elongation_cv"] =(float(np.std(ar_fit, ddof=1) / np.mean(ar_fit))
                            if len(ar_fit) >= 2 and np.mean(ar_fit) > 0 else np.nan)

    dy, dx = np.diff(y), np.diff(x)
    single = np.diff(t) == 1
    step = np.hypot(dy, dx)
    steps = step[single]
    out["speed_um_per_min"] = _median(steps) / frame_min if len(steps) else np.nan
    out["arrest_coefficient"] = float((steps < pixel_um).mean()) if len(steps) else np.nan
    net = float(np.hypot(y[-1] - y[0], x[-1] - x[0]))
    path = float(steps.sum())
    out["net_displacement_um"] = net
    out["path_length_um"] = path
    out["confinement_ratio"] = net / path if path > 0 else np.nan

    idx = np.where(single)[0]
    consec = idx[:-1][np.diff(idx) == 1]
    if len(consec):
        turn = np.angle(np.exp(1j * (np.arctan2(dy[consec + 1], dx[consec + 1])
                                     - np.arctan2(dy[consec], dx[consec]))))
        out["turning_angle_deg"] = float(np.median(np.degrees(np.abs(turn))))
    else:
        out["turning_angle_deg"] = np.nan

    orient = g["orientation"].to_numpy(float)
    moving = single & (step >= pixel_um) & shape_ok[:-1] & shape_ok[1:]
    ang = _axis_motion_angle(orient[:-1][moving], dy[moving], dx[moving])
    out["n_axis_motion_steps"] = int(moving.sum())
    out["axis_motion_angle_deg"] = _median(ang)
    out["axis_motion_frac_aligned"] = float((ang < 30).mean()) if len(ang) else np.nan

    by_t = {int(tt): o for tt, o, ok in zip(t, orient, shape_ok) if ok}
    for lag in ACF_LAGS:
        vals = [np.cos(2.0 * (by_t[tt + lag] - by_t[tt])) for tt in map(int, t)
                if tt in by_t and tt + lag in by_t]
        out[f"orientation_acf_lag{lag}"] = float(np.mean(vals)) if vals else np.nan
    return out


def per_track(tracks: pd.DataFrame, pixel_um: float, frame_min: float) -> pd.DataFrame:
    rows = []
    for tid, g in tracks.sort_values(["track_id", "t"]).groupby("track_id"):
        rows.append({"track_id": int(tid), **track_measures(g, pixel_um, frame_min)})
    return pd.DataFrame(rows)
