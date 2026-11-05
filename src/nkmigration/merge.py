"""Merge objects seen in overlapping slabs into one detection per cell, and
give each detection a Z position."""
from __future__ import annotations

import numpy as np
import pandas as pd

IOU_MIN = 0.3
BBOX = ["bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col"]


class _UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _pair_ious(rows_a: pd.DataFrame, rows_b: pd.DataFrame, lab_a: np.ndarray,
               lab_b: np.ndarray) -> list[tuple[int, int, float]]:
    """Mask IoU for every bbox-overlapping pair between two adjacent slabs."""
    ba, bb = rows_a[BBOX].to_numpy(), rows_b[BBOX].to_numpy()
    apart = ((ba[:, None, 0] >= bb[None, :, 2]) | (bb[None, :, 0] >= ba[:, None, 2])
             | (ba[:, None, 1] >= bb[None, :, 3]) | (bb[None, :, 1] >= ba[:, None, 3]))
    ia, ib = rows_a.index.to_numpy(), rows_b.index.to_numpy()
    la, lb = rows_a["label"].to_numpy(), rows_b["label"].to_numpy()
    out = []
    for a, b in np.argwhere(~apart):
        r0, c0 = min(ba[a, 0], bb[b, 0]), min(ba[a, 1], bb[b, 1])
        r1, c1 = max(ba[a, 2], bb[b, 2]), max(ba[a, 3], bb[b, 3])
        ma = lab_a[r0:r1, c0:c1] == la[a]
        mb = lab_b[r0:r1, c0:c1] == lb[b]
        inter = np.logical_and(ma, mb).sum()
        if inter:
            out.append((ia[a], ib[b], float(inter) / float(np.logical_or(ma, mb).sum())))
    return out


def _greedy_matches(pairs, iou_min: float = IOU_MIN):
    """One-to-one matches, highest IoU first."""
    used, matched = set(), []
    for a, b, iou in sorted(pairs, key=lambda p: -p[2]):
        if iou < iou_min or a in used or b in used:
            continue
        used.update((a, b))
        matched.append((a, b))
    return matched


def merge_slabs(objects: pd.DataFrame, labels: list[np.ndarray]) -> pd.DataFrame:
    """Chain matched objects across adjacent slabs and keep, per chain, the
    object with the largest area x mean intensity.

    Returns the kept objects, in the order of the input table, with
    `group_id` and `slabs_seen`.
    """
    n_slabs = len(labels)
    keep, group_of, slabs_of = [], {}, {}
    gid = 0
    for t, at_t in objects.groupby("t", sort=True):
        by_slab = {s: at_t[at_t["slab"] == s] for s in range(n_slabs)}
        uf = _UnionFind(list(at_t.index))
        for s in range(n_slabs - 1):
            a, b = by_slab[s], by_slab[s + 1]
            if len(a) and len(b):
                for i, j in _greedy_matches(_pair_ious(a, b, labels[s][t], labels[s + 1][t])):
                    uf.union(i, j)
        chains: dict[int, list[int]] = {}
        for i in at_t.index:
            chains.setdefault(uf.find(i), []).append(i)
        for members in chains.values():
            score = (objects.loc[members, "area_px"].to_numpy(float)
                     * objects.loc[members, "mean_intensity"].to_numpy(float))
            winner = members[int(np.argmax(score))]
            keep.append(winner)
            group_of[winner] = gid
            slabs_of[winner] = ",".join(str(int(objects.at[m, "slab"]))
                                        for m in sorted(members, key=lambda m: objects.at[m, "slab"]))
            gid += 1
    out = objects.loc[sorted(keep)].copy()
    out["group_id"] = [group_of[i] for i in out.index]
    out["slabs_seen"] = [slabs_of[i] for i in out.index]
    return out.reset_index(drop=True)


def z_position(detections: pd.DataFrame, labels: list[np.ndarray], nk: np.ndarray,
               z_step_um: float) -> np.ndarray:
    """Intensity-weighted Z centroid (um) of the NK signal under each
    detection's 2D mask, taken through the full stack depth."""
    z_idx = np.arange(nk.shape[1], dtype=np.float32)
    out = np.full(len(detections), np.nan)
    for t, at_t in detections.groupby("t"):
        vol = nk[int(t)].astype(np.float32)
        for k, row in zip(at_t.index, at_t.itertuples(index=False)):
            m = labels[int(row.slab)][int(t)] == int(row.label)
            if not m.any():
                continue
            prof = vol[:, m].sum(axis=1)
            total = float(prof.sum())
            if total > 0:
                out[k] = float((prof * z_idx).sum() / total) * z_step_um
    return out


def to_detections(kept: pd.DataFrame, pixel_um: float, z_um: np.ndarray,
                  z_step_um: float) -> pd.DataFrame:
    """Physical units, the Z position, and the shape-measurement flag."""
    d = kept
    return pd.DataFrame({
        "t": d["t"].astype(int), "slab": d["slab"].astype(int), "label": d["label"].astype(int),
        "group_id": d["group_id"].astype(int),
        "x_um": d["centroid_x_px"] * pixel_um,
        "y_um": d["centroid_y_px"] * pixel_um,
        "z_um": z_um,
        "slab_center_z_um": (d["z_lo"] + d["z_hi"]) / 2.0 * z_step_um,
        "area_um2": d["area_px"] * pixel_um ** 2,
        "aspect_ratio": d["aspect_ratio"],
        "eccentricity": d["eccentricity"],
        "orientation": d["orientation"],
        "equivalent_diameter_um": d["equivalent_diameter_px"] * pixel_um,
        "solidity": d["solidity"],
        "circularity": d["circularity"],
        "use_for_shape": ~d["touches_border"].astype(bool),
    })
