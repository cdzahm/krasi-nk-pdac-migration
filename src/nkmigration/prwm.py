"""Persistent random walk model (PRWM) fit, separately along x, y and z.

For each axis the mean squared displacement by overlapping intervals is fit to

    MSD(tau) = 2 S^2 P [tau - P (1 - exp(-tau / P))] + 2 sigma^2

with S the migration speed, P the persistence time and sigma the positional
noise. The axis motility coefficient is mu = S^2 P, and a cell's motility
coefficient is mu_x + mu_y + mu_z.

Bounds: S <= 20 um/min, sigma <= 100 um, and P <= the track's duration. A
cell that moves straight for its whole track only constrains P from below,
so P is capped at the duration rather than left free to run to the bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

MIN_PAIRS = 4   # displacement pairs a lag needs to be used
MIN_LAGS = 4    # lags in the fit, counting tau = 0
S_MAX, SIGMA_MAX = 20.0, 100.0


def msd_model(tau, S, P, sigma):
    return 2.0 * S ** 2 * P * (tau - P * (1.0 - np.exp(-tau / P))) + 2.0 * sigma ** 2


def fit_axis(pos: np.ndarray, frame_min: float) -> dict | None:
    """Fit one axis of one track. Returns None if the track is too short."""
    n = len(pos)
    lags = np.arange(0, n - MIN_PAIRS + 1)
    if len(lags) < MIN_LAGS:
        return None
    msd = np.zeros(n)
    for lag in range(1, n):
        msd[lag] = np.mean((pos[lag:] - pos[:-lag]) ** 2)
    p_max = (n - 1) * frame_min
    try:
        (S, P, sigma), _ = curve_fit(msd_model, lags * frame_min, msd[lags],
                                     p0=[1.0, min(10.0, 0.5 * p_max), 1.0],
                                     bounds=([0.0, 0.0, 0.0], [S_MAX, p_max, SIGMA_MAX]),
                                     maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    return {"S": float(S), "P": float(P), "sigma": float(sigma), "mu": float(S ** 2 * P)}


def fit_tracks(tracks: pd.DataFrame, frame_min: float) -> pd.DataFrame:
    """One row per track fitted on all three axes. A track with a frame that
    has no Z position is not fit, since the lags would no longer be in frames."""
    rows = []
    df = tracks.sort_values(["track_id", "t"])
    for tid, g in df.groupby("track_id"):
        if g["z_um"].isna().any() or (np.diff(g["t"].to_numpy()) != 1).any():
            continue
        fits = {ax: fit_axis(g[col].to_numpy(float), frame_min)
                for ax, col in (("x", "x_um"), ("y", "y_um"), ("z", "z_um"))}
        if any(f is None for f in fits.values()):
            continue
        row = {"track_id": int(tid), "n_fit_frames": len(g),
               "motility_coefficient": sum(f["mu"] for f in fits.values()),
               "speed_3d_prwm": float(np.sqrt(sum(f["S"] ** 2 for f in fits.values())))}
        for ax, f in fits.items():
            row.update({f"S_{ax}": f["S"], f"P_{ax}": f["P"], f"sigma_{ax}": f["sigma"],
                        f"mu_{ax}": f["mu"]})
        rows.append(row)
    return pd.DataFrame(rows)
