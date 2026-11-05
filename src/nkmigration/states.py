"""Morphodynamic states: round/elongated x slow/fast, per cell-timepoint.

Aspect ratio and instantaneous XY speed are each split at their median,
computed separately for each experiment (one donor with one tumor line, or
the IL-15 titration). A frame needs a valid shape and a step from the
previous frame to get a state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATES = {0: "round_slow", 1: "round_fast", 2: "elongated_slow", 3: "elongated_fast"}
MIN_OCCUPANCY_FRAMES = 10


def frame_speeds(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add the step length from the previous frame (um/frame) to every row."""
    df = tracks.sort_values(["recording_id", "track_id", "t"]).copy()
    g = df.groupby(["recording_id", "track_id"])
    step = np.hypot(g["y_um"].diff(), g["x_um"].diff())
    df["step_um"] = np.where(g["t"].diff() == 1, step, np.nan)
    return df


def state_cutoffs(frames: pd.DataFrame) -> pd.DataFrame:
    """Median aspect ratio and step length per experiment."""
    ok = frames["use_for_shape"] & frames["step_um"].notna() & (frames["aspect_ratio"] < 100)
    return (frames.loc[ok].groupby("experiment")
            .agg(aspect_ratio_cut=("aspect_ratio", "median"),
                 step_cut_um=("step_um", "median")).reset_index())


def assign_states(frames: pd.DataFrame, cuts: pd.DataFrame) -> pd.DataFrame:
    df = frames.merge(cuts, on="experiment", how="left")
    ok = df["use_for_shape"] & df["step_um"].notna() & df["aspect_ratio_cut"].notna()
    df["state"] = np.nan
    df.loc[ok, "state"] = (2 * (df.loc[ok, "aspect_ratio"] >= df.loc[ok, "aspect_ratio_cut"])
                           + (df.loc[ok, "step_um"] >= df.loc[ok, "step_cut_um"]))
    return df


def episodes(states: pd.DataFrame) -> pd.DataFrame:
    """Runs of the same state within a track. A run also ends at a frame
    with no state."""
    rows = []
    stated = states.dropna(subset=["state"])
    for (rid, tid), g in stated.groupby(["recording_id", "track_id"], sort=False):
        g = g.sort_values("t")
        st, t = g["state"].to_numpy(int), g["t"].to_numpy(int)
        start = 0
        for i in range(1, len(st) + 1):
            if i == len(st) or st[i] != st[start] or t[i] != t[i - 1] + 1:
                rows.append((rid, tid, st[start], t[start], t[i - 1] - t[start] + 1))
                start = i
    return pd.DataFrame(rows, columns=["recording_id", "track_id", "state", "t_start", "duration"])


def per_cell_states(ep: pd.DataFrame) -> pd.DataFrame:
    """Occupancy of each state and the state-transition measures, per track."""
    key = ["recording_id", "track_id"]
    g = ep.groupby(key)
    out = g.agg(n_episodes=("state", "size"), stated_frames=("duration", "sum"),
                mean_dwell_frames=("duration", "mean")).reset_index()
    out["state_transitions"] = out["n_episodes"] - 1
    out["state_transition_rate"] = np.where(out["stated_frames"] > 1,
                                            out["state_transitions"] / (out["stated_frames"] - 1),
                                            np.nan)
    occ = (ep.groupby(key + ["state"])["duration"].sum().unstack("state", fill_value=0)
           .reindex(columns=list(STATES), fill_value=0))
    occ = occ.div(occ.sum(axis=1), axis=0)
    occ.columns = [f"{STATES[s]}_occupancy" for s in occ.columns]
    return out.drop(columns="n_episodes").merge(occ.reset_index(), on=key)
