"""Step 4: link detections into tracks."""
from __future__ import annotations

import pandas as pd

from nkmigration import cli, track


def main():
    args = cli.parser(__doc__).parse_args()
    layout, recs = cli.setup(args)
    for rec in recs:
        tracks = track.link(pd.read_csv(layout.detections(rec.recording_id)))
        layout.tracks(rec.recording_id).parent.mkdir(parents=True, exist_ok=True)
        tracks.to_csv(layout.tracks(rec.recording_id), index=False)
        n = tracks.groupby("track_id").size()
        print(f"{rec.recording_id}: {len(n):,} tracks, median {n.median():.0f} frames")


if __name__ == "__main__":
    main()
