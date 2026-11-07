"""Step 6: draw the published panels from measures/per_cell.csv.

Each panel is written as a vector PDF (editable in Illustrator), and every
comparison drawn goes into panels/panel_stats.csv.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nkmigration import cli, panels, plotting


def main():
    p = cli.parser(__doc__, per_recording=False)
    p.add_argument("--only", nargs="*", help="panel names to draw (default: all)")
    args = p.parse_args()
    layout, _ = cli.setup(args)
    cells = pd.read_csv(layout.measures / "per_cell.csv")
    layout.panels.mkdir(parents=True, exist_ok=True)

    rows = []
    for panel in panels.PANELS:
        if args.only and panel.name not in args.only:
            continue
        sub, donors = panels.panel_cells(cells, panel)
        values = [sub.loc[sub["arm"] == a, panel.measure].to_numpy(float) for a in panel.arms]
        is_motility = panel.measure == "motility_coefficient"
        fig, res = plotting.dot_panel(
            panel.arms, values, panels.YLABEL[panel.measure], panel.title, key=panel.name,
            ceiling=panels.MOTILITY_CEILING if is_motility else None,
            bounded01=(panel.measure.endswith("_occupancy")
                       or panel.measure == "arrest_coefficient"),
            connect_medians=panel.arms == panels.DOSES)
        # tight box: two-line axis labels would otherwise be clipped at the left edge
        fig.savefig(layout.panels / f"{panel.name}.pdf", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        for pr in res["pairs"]:
            a, b = values[pr["i"]], values[pr["j"]]
            rows.append({"panel": panel.name, "measure": panel.measure,
                         "donors": ",".join(donors), "test": res["test"],
                         "omnibus_p": res["omnibus_p"],
                         "group_1": panel.arms[pr["i"]], "group_2": panel.arms[pr["j"]],
                         "n_1": len(a), "n_2": len(b),
                         "median_1": float(np.median(a)), "median_2": float(np.median(b)),
                         "p": pr["p"], "p_adjusted": pr["p_adj"], "shown": pr["shown"],
                         "above_ceiling": res["above_ceiling"] or ""})
        print(f"{panel.name}: " + ", ".join(f"{a} n={len(v):,}" for a, v in zip(panel.arms, values)))
    pd.DataFrame(rows).to_csv(layout.panels / "panel_stats.csv", index=False)


if __name__ == "__main__":
    main()
