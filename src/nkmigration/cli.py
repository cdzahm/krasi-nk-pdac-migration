"""Shared command-line options for the scripts."""
from __future__ import annotations

import argparse

from .recordings import Layout, load_recordings


def parser(description: str, per_recording: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", help="raw stacks, one folder per recording (or $NKMIG_DATA)")
    p.add_argument("--out-root", help="where outputs are written (or $NKMIG_OUT)")
    if per_recording:
        p.add_argument("recordings", nargs="*",
                       help="recording IDs from config/recordings.csv (default: all)")
    return p


def setup(args) -> tuple[Layout, list]:
    layout = Layout.from_env(args.data_root, args.out_root)
    recs = load_recordings()
    wanted = getattr(args, "recordings", None)
    if wanted:
        unknown = set(wanted) - {r.recording_id for r in recs}
        if unknown:
            raise SystemExit(f"unknown recording(s): {', '.join(sorted(unknown))}")
        recs = [r for r in recs if r.recording_id in wanted]
    return layout, recs
