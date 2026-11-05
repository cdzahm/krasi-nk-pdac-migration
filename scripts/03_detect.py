"""Step 3: merge objects across slabs into detections and add each one's Z position."""
from __future__ import annotations

import json

import pandas as pd
import tifffile

from nkmigration import cli, merge
from nkmigration.io import load_stack


def main():
    args = cli.parser(__doc__).parse_args()
    layout, recs = cli.setup(args)
    for rec in recs:
        seg = layout.segmentation(rec.recording_id)
        n_slabs = len(json.loads((seg / "slabs.json").read_text())["slab_starts"])
        labels = [tifffile.imread(seg / f"labels_slab{s}.tif") for s in range(n_slabs)]
        objects = pd.read_csv(seg / "objects.csv")
        nk, cal = load_stack(layout.corrected(rec.recording_id))
        kept = merge.merge_slabs(objects, labels)
        z_um = merge.z_position(kept, labels, nk, cal.z_step_um)
        det = merge.to_detections(kept, cal.pixel_um, z_um, cal.z_step_um)
        layout.detections(rec.recording_id).parent.mkdir(parents=True, exist_ok=True)
        det.to_csv(layout.detections(rec.recording_id), index=False)
        layout.calibration(rec.recording_id).write_text(json.dumps(
            {"pixel_um": cal.pixel_um, "z_step_um": cal.z_step_um,
             "frame_interval_s": cal.frame_interval_s}, indent=1))
        print(f"{rec.recording_id}: {len(objects):,} objects -> {len(det):,} detections")


if __name__ == "__main__":
    main()
