"""Frame-to-frame linking of detections into tracks."""
from __future__ import annotations

import pandas as pd
from laptrack import LapTrack

MAX_DISPLACEMENT_UM = 15.0
Z_WEIGHT = 1.0


def _sq_dist(a, b, **_):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (Z_WEIGHT * (a[2] - b[2])) ** 2


def link(detections: pd.DataFrame) -> pd.DataFrame:
    """Linear-assignment linking on (y, x, z), at most 15 um per frame,
    with no gap closing, splitting or merging.

    Detections without a Z position are placed at their slab's center.
    Returns the detections with a `track_id` column.
    """
    df = detections.copy()
    df["z_link"] = df["z_um"].fillna(df["slab_center_z_um"])
    df["t_coord"] = df["t"]
    lt = LapTrack(metric=_sq_dist, cutoff=MAX_DISPLACEMENT_UM ** 2,
                  gap_closing_metric="sqeuclidean", gap_closing_cutoff=False,
                  gap_closing_max_frame_count=0,
                  splitting_cutoff=False, merging_cutoff=False)
    tracked, splits, merges = lt.predict_dataframe(
        df, coordinate_cols=["y_um", "x_um", "z_link", "t_coord"], frame_col="t")
    if not (splits.empty and merges.empty):
        raise RuntimeError("splitting and merging must be off")
    tracked = tracked.drop(columns=["z_link", "t_coord"])
    keep = [c for c in tracked.columns if c not in ("frame", "index", "tree_id")]
    return tracked[keep].sort_values(["track_id", "t"]).reset_index(drop=True)
