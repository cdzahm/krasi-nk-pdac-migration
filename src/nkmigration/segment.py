"""NK-cell segmentation on overlapping Z slabs.

Z sampling (5 um) is coarse next to an NK cell, so each time point is cut
into 7-plane slabs at a 4-plane stride, and the maximum-intensity projection
of each slab is segmented in 2D with Cellpose-SAM. A cell seen in two
overlapping slabs is merged later (merge.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

SLAB_SIZE = 7
SLAB_STRIDE = 4
FLOW_THRESHOLD = 0.4
MIN_SIZE_PX = 5
MODEL = "cpsam_v2"

PROPERTIES = ("label", "area", "centroid", "orientation", "axis_major_length",
              "axis_minor_length", "eccentricity", "equivalent_diameter_area", "bbox",
              "solidity", "perimeter", "intensity_mean")


def slab_starts(size_z: int, size: int = SLAB_SIZE, stride: int = SLAB_STRIDE) -> list[int]:
    """Start planes of the slabs: every `stride` planes, plus one ending on the
    top plane if the strided slabs stop short of it."""
    starts = list(range(0, size_z - size + 1, stride))
    last = size_z - size
    if last >= 0 and (not starts or starts[-1] != last):
        starts.append(last)
    return starts


def load_model(gpu: bool = True):
    from cellpose import models
    return models.CellposeModel(gpu=gpu, pretrained_model=MODEL)


def _objects(masks: np.ndarray, proj: np.ndarray) -> pd.DataFrame:
    tbl = pd.DataFrame(regionprops_table(masks, intensity_image=proj, properties=PROPERTIES))
    h, w = masks.shape
    return pd.DataFrame({
        "label": tbl["label"].astype(int),
        "area_px": tbl["area"].astype(float),
        "centroid_y_px": tbl["centroid-0"],
        "centroid_x_px": tbl["centroid-1"],
        "orientation": tbl["orientation"],
        "major_axis_px": tbl["axis_major_length"],
        "minor_axis_px": tbl["axis_minor_length"],
        "aspect_ratio": tbl["axis_major_length"] / np.clip(tbl["axis_minor_length"], 1e-6, None),
        "eccentricity": tbl["eccentricity"],
        "equivalent_diameter_px": tbl["equivalent_diameter_area"],
        "solidity": tbl["solidity"],
        "circularity": 4 * np.pi * tbl["area"] / np.clip(tbl["perimeter"], 1e-6, None) ** 2,
        "mean_intensity": tbl["intensity_mean"],
        "bbox_min_row": tbl["bbox-0"].astype(int),
        "bbox_min_col": tbl["bbox-1"].astype(int),
        "bbox_max_row": tbl["bbox-2"].astype(int),
        "bbox_max_col": tbl["bbox-3"].astype(int),
        "touches_border": ((tbl["bbox-0"] == 0) | (tbl["bbox-1"] == 0)
                           | (tbl["bbox-2"] == h) | (tbl["bbox-3"] == w)),
    })


def segment_stack(nk: np.ndarray, model, first_z_planes: int | None = None,
                  flow_threshold: float = FLOW_THRESHOLD):
    """Segment every slab of every time point.

    Returns the object table (one row per object per slab, pixel units) and
    one (T, Y, X) label array per slab. `first_z_planes` keeps only the
    bottom planes, for recordings whose upper planes are unusable.
    """
    if first_z_planes is not None:
        nk = nk[:, :first_z_planes]
    starts = slab_starts(nk.shape[1])
    tables, labels = [], [[] for _ in starts]
    for t in range(nk.shape[0]):
        for s, z0 in enumerate(starts):
            proj = nk[t, z0:z0 + SLAB_SIZE].max(axis=0)
            masks, _, _ = model.eval(proj, diameter=None, do_3D=False, min_size=MIN_SIZE_PX,
                                     flow_threshold=flow_threshold,
                                     normalize={"norm3D": True})
            labels[s].append(masks.astype(np.uint16))
            if masks.max() > 0:
                obj = _objects(masks, proj)
                obj.insert(0, "t", t)
                obj.insert(1, "slab", s)
                obj.insert(2, "z_lo", z0)
                obj.insert(3, "z_hi", z0 + SLAB_SIZE - 1)
                tables.append(obj)
    objects = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return objects, [np.stack(lab) for lab in labels], starts
