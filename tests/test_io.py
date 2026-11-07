import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from nkmigration.io import Calibration, prairieview_voxel_um, read_calibration, save_stack


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


REPO = Path(__file__).resolve().parents[1]
PV_XML = '''<PVScan>
    <PVStateValue key="micronsPerPixel">
      <IndexedValue index="XAxis" value="1.1640625" />
      <IndexedValue index="YAxis" value="1.1640625" />
      <IndexedValue index="ZAxis" value="5" />
    </PVStateValue>
</PVScan>'''


def test_prairieview_voxel_size(tmp_path):
    xml = tmp_path / "TSeries.xml"
    xml.write_text(PV_XML)
    assert prairieview_voxel_um(xml) == (5.0, 1.1640625, 1.1640625)


def test_add_calibration_script(tmp_path):
    (tmp_path / "TSeries.xml").write_text(PV_XML)
    tifffile.imwrite(tmp_path / "raw.tif", np.zeros((2, 3, 4, 5), np.uint16), imagej=True,
                     metadata={"axes": "TZYX"})
    script = REPO / "scripts" / "00_add_calibration.py"
    args = [str(tmp_path / "raw.tif"), str(tmp_path / "cal.tif"),
            "--xml", str(tmp_path / "TSeries.xml"), "--interval", "60"]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    subprocess.run([sys.executable, str(script), *args], check=True, env=env)
    cal = read_calibration(tmp_path / "cal.tif")
    assert np.allclose(cal.voxel_um, (5.0, 1.1640625, 1.1640625)) and cal.frame_interval_s == 60.0
