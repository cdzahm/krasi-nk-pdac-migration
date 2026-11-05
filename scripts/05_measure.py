"""Step 5: per-cell measures for every recording.

Writes measures/per_cell.csv (one row per tracked cell), the PRWM fits, and
the per-experiment state cutoffs.
"""
from __future__ import annotations

import json

import pandas as pd

from nkmigration import cli, features, percell, prwm, states


def main():
    args = cli.parser(__doc__, per_recording=False).parse_args()
    layout, recs = cli.setup(args)
    out = layout.measures
    out.mkdir(parents=True, exist_ok=True)

    tracks, cells, fits = [], [], []
    for rec in recs:
        cal = json.loads(layout.calibration(rec.recording_id).read_text())
        frame_min = cal["frame_interval_s"] / 60.0
        tr = pd.read_csv(layout.tracks(rec.recording_id))
        meta = {"recording_id": rec.recording_id, "experiment": rec.experiment,
                "cell_line": rec.cell_line, "donor": rec.donor, "arm": rec.arm}
        tracks.append(tr.assign(recording_id=rec.recording_id, experiment=rec.experiment))
        cells.append(features.per_track(tr, cal["pixel_um"], frame_min).assign(**meta))
        f = prwm.fit_tracks(tr, frame_min).assign(recording_id=rec.recording_id)
        fits.append(f)
        print(f"{rec.recording_id}: {tr['track_id'].nunique():,} tracks, {len(f):,} PRWM fits")

    frames = states.frame_speeds(pd.concat(tracks, ignore_index=True))
    cuts = states.state_cutoffs(frames)
    ep = states.episodes(states.assign_states(frames, cuts))
    fits = pd.concat(fits, ignore_index=True)

    key = ["recording_id", "track_id"]
    table = (pd.concat(cells, ignore_index=True)
             .merge(states.per_cell_states(ep), on=key, how="left")
             .merge(fits[key + ["motility_coefficient", "speed_3d_prwm"]], on=key, how="left"))
    table = percell.apply_inclusion(table)
    front = ["recording_id", "experiment", "cell_line", "donor", "arm", "track_id"]
    table = table[front + [c for c in table.columns if c not in front]]

    table.to_csv(out / "per_cell.csv", index=False)
    fits.to_csv(out / "prwm_fits.csv", index=False)
    cuts.to_csv(out / "state_cutoffs.csv", index=False)
    print(f"{len(table):,} cells -> {out / 'per_cell.csv'}")


if __name__ == "__main__":
    main()
