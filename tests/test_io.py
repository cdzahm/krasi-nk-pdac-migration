import numpy as np
import pytest
import tifffile

from nkmigration.io import Calibration, read_calibration, save_stack


def test_calibration_round_trip(tmp_path):
    cal = Calibration(voxel_um=(5.0, 1.25, 2.5), frame_interval_s=60.0)
    save_stack(tmp_path / "s.tif", np.zeros((2, 3, 4, 5), np.uint16), cal)
    back = read_calibration(tmp_path / "s.tif")
    assert np.allclose(back.voxel_um, cal.voxel_um) and back.frame_interval_s == 60.0


def test_calibration_must_be_in_microns(tmp_path):
    tifffile.imwrite(tmp_path / "nm.tif", np.zeros((2, 3, 4, 5), np.uint16), imagej=True,
                     resolution=(1.0, 1.0), metadata={"axes": "TZYX", "unit": "nm", "spacing": 5.0,
                                                      "finterval": 60.0})
    with pytest.raises(ValueError):
        read_calibration(tmp_path / "nm.tif")
