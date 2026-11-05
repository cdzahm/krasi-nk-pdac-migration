"""Step 0 (only for stacks without calibration): write voxel size and frame
interval into a TZYX stack's metadata.

Every later step reads calibration from the stack itself. Stacks written by
PPP Stack Organizer already carry it; this is for stacks that do not, with
the voxel size taken from the Bruker PrairieView acquisition XML.

    python scripts/00_add_calibration.py in.tif out.tif --xml TSeries.xml --interval 60
"""
from __future__ import annotations

import argparse

import tifffile

from nkmigration.io import Calibration, prairieview_voxel_um, save_stack


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--xml", required=True, help="PrairieView acquisition XML")
    p.add_argument("--interval", type=float, required=True, help="frame interval (s)")
    args = p.parse_args()
    data = tifffile.imread(args.src)
    if data.ndim != 4:
        raise SystemExit(f"expected a TZYX stack, got shape {data.shape}")
    save_stack(args.dst, data, Calibration(prairieview_voxel_um(args.xml), args.interval))


if __name__ == "__main__":
    main()
