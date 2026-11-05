"""Step 2: segment NK cells on overlapping Z slabs (GPU recommended)."""
from __future__ import annotations

import json

import numpy as np
import tifffile

from nkmigration import cli, segment
from nkmigration.io import load_stack


def main():
    p = cli.parser(__doc__)
    p.add_argument("--cpu", action="store_true", help="run Cellpose on the CPU")
    args = p.parse_args()
    layout, recs = cli.setup(args)
    model = segment.load_model(gpu=not args.cpu)
    for rec in recs:
        nk, cal = load_stack(layout.corrected(rec.recording_id))
        objects, labels, starts = segment.segment_stack(nk, model, rec.segment_first_z_planes)
        out = layout.segmentation(rec.recording_id)
        out.mkdir(parents=True, exist_ok=True)
        objects.to_csv(out / "objects.csv", index=False)
        for s, lab in enumerate(labels):
            tifffile.imwrite(out / f"labels_slab{s}.tif", lab.astype(np.uint16),
                             imagej=True, metadata={"axes": "TYX"})
        (out / "slabs.json").write_text(json.dumps(
            {"slab_starts": starts, "slab_size": segment.SLAB_SIZE,
             "planes_used": int(nk.shape[1] if rec.segment_first_z_planes is None
                                else rec.segment_first_z_planes),
             "flow_threshold": segment.FLOW_THRESHOLD, "model": segment.MODEL}, indent=1))
        print(f"{rec.recording_id}: {len(objects):,} objects in {len(starts)} slabs")


if __name__ == "__main__":
    main()
