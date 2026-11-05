"""Step 7: the redundancy and donor-consistency screens over every candidate
per-cell measure (see nkmigration/screens.py)."""
from __future__ import annotations

import pandas as pd

from nkmigration import cli, percell, screens


def main():
    p = cli.parser(__doc__, per_recording=False)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    layout, _ = cli.setup(args)
    cells = pd.read_csv(layout.measures / "per_cell.csv")
    out = layout.screens
    out.mkdir(parents=True, exist_ok=True)

    measures = [m.name for m in percell.MEASURES]
    eff = screens.effects(cells, measures, workers=args.workers)
    eff.to_csv(out / "effects.csv", index=False)
    groups = screens.redundancy(eff)
    groups.to_csv(out / "redundancy_groups.csv", index=False)
    cons = screens.consistency(eff)
    cons.to_csv(out / "donor_consistency.csv", index=False)

    print(f"{len(measures)} measures, {eff[['comparison', 'cell_line', 'donor']].drop_duplicates().shape[0]} units, "
          f"{groups['group'].nunique()} groups at |rho| >= {screens.RHO_MIN}")
    agree = cons[cons["same_sign_all_donors"]]
    for comp, g in agree.groupby("comparison"):
        print(f"{comp}: same sign in every donor for {len(g)} measures, "
              f"intervals exclude zero for {int(g['intervals_exclude_zero'].sum())}")


if __name__ == "__main__":
    main()
