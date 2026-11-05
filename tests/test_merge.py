import numpy as np
import pandas as pd

from nkmigration import merge


def _obj(t, slab, label, box, area, intensity):
    r0, c0, r1, c1 = box
    return dict(t=t, slab=slab, label=label, z_lo=4 * slab, z_hi=4 * slab + 6, area_px=area,
                mean_intensity=intensity, bbox_min_row=r0, bbox_min_col=c0,
                bbox_max_row=r1, bbox_max_col=c1)


def test_same_cell_in_two_slabs_is_merged():
    lab0, lab1 = np.zeros((1, 20, 20), np.uint16), np.zeros((1, 20, 20), np.uint16)
    lab0[0, 2:8, 2:8] = 1
    lab1[0, 3:8, 2:8] = 1          # IoU 30/36
    lab1[0, 12:16, 12:16] = 2      # a different cell
    objects = pd.DataFrame([_obj(0, 0, 1, (2, 2, 8, 8), 36, 10.0),
                            _obj(0, 1, 1, (3, 2, 8, 8), 30, 20.0),
                            _obj(0, 1, 2, (12, 12, 16, 16), 16, 5.0)])
    kept = merge.merge_slabs(objects, [lab0, lab1])
    assert len(kept) == 2
    winner = kept[kept["label"] == 1].iloc[0]
    assert winner["slab"] == 1          # 30 * 20 beats 36 * 10
    assert winner["slabs_seen"] == "0,1"


def test_low_overlap_is_not_merged():
    lab0, lab1 = np.zeros((1, 20, 20), np.uint16), np.zeros((1, 20, 20), np.uint16)
    lab0[0, 0:10, 0:10] = 1
    lab1[0, 7:17, 7:17] = 1          # IoU 9/191
    objects = pd.DataFrame([_obj(0, 0, 1, (0, 0, 10, 10), 100, 1.0),
                            _obj(0, 1, 1, (7, 7, 17, 17), 100, 1.0)])
    assert len(merge.merge_slabs(objects, [lab0, lab1])) == 2


def test_z_position_is_intensity_weighted():
    nk = np.zeros((1, 5, 4, 4), np.float32)
    nk[0, 1, 1:3, 1:3] = 1.0
    nk[0, 3, 1:3, 1:3] = 3.0
    lab = np.zeros((1, 4, 4), np.uint16)
    lab[0, 1:3, 1:3] = 1
    det = pd.DataFrame({"t": [0], "slab": [0], "label": [1]})
    z = merge.z_position(det, [lab], nk, z_step_um=5.0)
    assert np.isclose(z[0], 2.5 * 5.0)
