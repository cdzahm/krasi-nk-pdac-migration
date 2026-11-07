"""Reading and writing calibrated TZYX image stacks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

MICRON = {"um", "µm", "μm", "\\u00B5m", "micron", "microns"}


@dataclass(frozen=True)
class Calibration:
    voxel_um: tuple[float, float, float]  # (z, y, x)
    frame_interval_s: float

    @property
    def z_step_um(self) -> float:
        return self.voxel_um[0]

    @property
    def pixel_um(self) -> float:
        _, y, x = self.voxel_um
        if abs(y - x) > 1e-3:
            raise ValueError(f"non-square pixels: {y} x {x} um")
        return y

    @property
    def frame_interval_min(self) -> float:
        return self.frame_interval_s / 60.0


def read_calibration(path: str | Path) -> Calibration:
    """Voxel size and frame interval from the ImageJ metadata of a stack.

    Z step and frame interval are in the ImageDescription; XY pixel size is
    in the TIFF resolution tags.
    """
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        desc = page.tags["ImageDescription"].value
        fields = dict(line.split("=", 1) for line in desc.splitlines() if "=" in line)
        if fields.get("unit", "").strip() not in MICRON:
            raise ValueError(f"{path}: calibration unit {fields.get('unit')!r}, expected um "
                             "(see scripts/00_add_calibration.py)")
        x_res = page.tags["XResolution"].value
        y_res = page.tags["YResolution"].value
    px_x = 1.0 / (x_res[0] / x_res[1])
    px_y = 1.0 / (y_res[0] / y_res[1])
    return Calibration(voxel_um=(float(fields["spacing"]), px_y, px_x),
                       frame_interval_s=float(fields["finterval"]))


def load_stack(path: str | Path) -> tuple[np.ndarray, Calibration]:
    """Return a (T, Z, Y, X) array and its calibration."""
    data = tifffile.imread(path)
    if data.ndim != 4:
        raise ValueError(f"{path}: expected a TZYX stack, got shape {data.shape}")
    return data, read_calibration(path)


def save_stack(path: str | Path, data: np.ndarray, cal: Calibration) -> None:
    """Write a TZYX stack with its calibration as ImageJ metadata."""
    vz, vy, vx = cal.voxel_um
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, data, imagej=True, resolution=(1.0 / vx, 1.0 / vy),
                     metadata={"axes": "TZYX", "unit": "um", "spacing": vz,
                               "finterval": cal.frame_interval_s})


def prairieview_voxel_um(xml_path: str | Path) -> tuple[float, float, float]:
    """(z, y, x) voxel size in um from a Bruker PrairieView acquisition XML."""
    text = Path(xml_path).read_text(encoding="utf-8")
    block = re.search(r'<PVStateValue key="micronsPerPixel">(.*?)</PVStateValue>',
                      text, re.DOTALL).group(1)
    um = {axis: float(v) for axis, v in
          re.findall(r'index="(XAxis|YAxis|ZAxis)" value="([\d.]+)"', block)}
    return um["ZAxis"], um["YAxis"], um["XAxis"]
