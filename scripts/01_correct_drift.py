"""Step 1: measure stage drift and write drift-corrected NK and tumor stacks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nkmigration import cli, drift
from nkmigration.io import load_stack, save_stack


def registration_input(rec, raw: dict[str, np.ndarray]) -> np.ndarray:
    if rec.drift_channel == "collagen":
        return raw["collagen"]
    if rec.drift_channel == "collagen_enhanced":
        return drift.enhance_collagen(raw["collagen"])
    if rec.drift_channel == "tumor_minus_nk":
        return drift.tumor_minus_nk(raw["tumor"], raw["nk"])
    raise ValueError(f"{rec.recording_id}: unknown drift channel {rec.drift_channel!r}")


def correct(rec, layout) -> None:
    raw, cal = {}, None
    for ch in ("nk", "tumor", "collagen"):
        path = layout.raw(rec.recording_id, ch)
        if path.exists():
            raw[ch], cal = load_stack(path)
            raw[ch] = raw[ch][rec.drop_first_frames:]
    if rec.drift_method == "none":
        shifts = pd.DataFrame({"frame": np.arange(raw["nk"].shape[0]),
                               "dz_um": 0.0, "dy_um": 0.0, "dx_um": 0.0})
    else:
        shifts = drift.measure_drift(rec.drift_method, registration_input(rec, raw), cal.voxel_um)
    layout.shifts(rec.recording_id).parent.mkdir(parents=True, exist_ok=True)
    shifts.to_csv(layout.shifts(rec.recording_id), index=False)
    for ch in ("nk", "tumor"):
        if ch in raw:
            save_stack(layout.corrected(rec.recording_id, ch),
                       drift.apply_shifts(raw[ch], shifts, cal.voxel_um), cal)
    print(f"{rec.recording_id}: {rec.drift_method}, max |shift| "
          f"{shifts[['dz_um', 'dy_um', 'dx_um']].abs().max().max():.1f} um")


def main():
    args = cli.parser(__doc__).parse_args()
    layout, recs = cli.setup(args)
    for rec in recs:
        correct(rec, layout)


if __name__ == "__main__":
    main()
