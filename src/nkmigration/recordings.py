"""The recording table and where each step reads and writes its files."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RECORDINGS_CSV = REPO / "config" / "recordings.csv"

# Raw stacks as written by PPP Stack Organizer, one folder per recording.
CHANNEL_FILES = {"nk": "nk.tif", "tumor": "tumor.tif", "collagen": "collagen.tif"}


@dataclass(frozen=True)
class Recording:
    recording_id: str
    experiment: str
    cell_line: str
    donor: str
    stimulus: str
    il15_ng: float | None
    kras_inhibitor: bool
    drop_first_frames: int
    segment_first_z_planes: int | None
    drift_method: str
    drift_channel: str | None

    @property
    def arm(self) -> str:
        """Treatment arm label used in panels, e.g. 'TriKE+KRASi'."""
        if self.experiment.startswith("Titration"):
            return f"{self.il15_ng:g} ng"
        base = {"none": "no IL-15", "IL15": "IL-15", "TriKE": "TriKE"}[self.stimulus]
        if not self.kras_inhibitor:
            return base
        return "KRASi" if self.stimulus == "IL15" else f"{base}+KRASi"


def load_recordings(path: str | Path = RECORDINGS_CSV) -> list[Recording]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = []
    for r in df.itertuples(index=False):
        out.append(Recording(
            recording_id=r.recording_id, experiment=r.experiment, cell_line=r.cell_line,
            donor=r.donor, stimulus=r.stimulus,
            il15_ng=float(r.il15_ng) if r.il15_ng else None,
            kras_inhibitor=r.kras_inhibitor == "yes",
            drop_first_frames=int(r.drop_first_frames or 0),
            segment_first_z_planes=int(r.segment_first_z_planes) if r.segment_first_z_planes else None,
            drift_method=r.drift_method, drift_channel=r.drift_channel or None,
        ))
    return out


def get_recording(recording_id: str) -> Recording:
    for rec in load_recordings():
        if rec.recording_id == recording_id:
            return rec
    raise KeyError(f"no recording {recording_id!r} in {RECORDINGS_CSV}")


@dataclass(frozen=True)
class Layout:
    """Folder layout under the data and output roots."""
    data_root: Path
    out_root: Path

    @classmethod
    def from_env(cls, data_root: str | None = None, out_root: str | None = None) -> "Layout":
        data = data_root or os.environ.get("NKMIG_DATA")
        out = out_root or os.environ.get("NKMIG_OUT")
        if not data or not out:
            raise SystemExit("set --data-root/--out-root or NKMIG_DATA/NKMIG_OUT")
        return cls(Path(data), Path(out))

    def raw(self, rec_id: str, channel: str) -> Path:
        return self.data_root / rec_id / CHANNEL_FILES[channel]

    def corrected(self, rec_id: str, channel: str = "nk") -> Path:
        return self.out_root / "drift" / rec_id / CHANNEL_FILES[channel]

    def shifts(self, rec_id: str) -> Path:
        return self.out_root / "drift" / rec_id / "shifts.csv"

    def segmentation(self, rec_id: str) -> Path:
        return self.out_root / "segmentation" / rec_id

    def detections(self, rec_id: str) -> Path:
        return self.out_root / "detections" / f"{rec_id}.csv"

    def calibration(self, rec_id: str) -> Path:
        return self.out_root / "detections" / f"{rec_id}.calibration.json"

    def tracks(self, rec_id: str) -> Path:
        return self.out_root / "tracks" / f"{rec_id}.csv"

    @property
    def measures(self) -> Path:
        return self.out_root / "measures"

    @property
    def panels(self) -> Path:
        return self.out_root / "panels"

    @property
    def screens(self) -> Path:
        return self.out_root / "screens"
