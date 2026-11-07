"""The published panels.

Cells are pooled by cell line across the donors that have every arm in a
comparison. Single-donor and titration panels use one experiment.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MOTILITY_CEILING = 300.0

YLABEL = {
    "motility_coefficient": "motility coefficient\n(µm²/min)",
    "speed_um_per_min": "speed (µm/min)",
    "arrest_coefficient": "arrest coefficient",
    "area_um2_median": "cell area (µm²)",
    "elongated_fast_occupancy": "elongated fast\noccupancy",
    "round_slow_occupancy": "round slow\noccupancy",
}

KRASI = ["IL-15", "KRASi"]
TRIKE = ["IL-15", "TriKE"]
COMBINED = ["IL-15", "TriKE", "TriKE+KRASi"]
NO_IL15 = ["no IL-15", "IL-15"]
DOSES = ["0 ng/mL", "1 ng/mL", "10 ng/mL"]


@dataclass(frozen=True)
class Panel:
    name: str
    measure: str
    arms: list[str]
    cell_line: str | None = None      # pool this cell line's donors
    experiment: str | None = None     # or use one experiment

    @property
    def title(self) -> str:
        if self.experiment and self.experiment.startswith("Titration"):
            return "IL-15 titration"
        if self.experiment:
            return f"{self.cell_line}, one donor"
        return f"{self.cell_line}, donors pooled"


def _p(name, measure, arms, **kw):
    return Panel(name, measure, arms, **kw)


PANELS = [
    _p("Fig3B_motility", "motility_coefficient", KRASI, cell_line="MIA-PaCa-2"),
    _p("Fig3B_speed", "speed_um_per_min", KRASI, cell_line="MIA-PaCa-2"),
    _p("Fig3D_motility", "motility_coefficient", KRASI, cell_line="HPAF-II"),
    _p("Fig3E_arrest", "arrest_coefficient", KRASI, cell_line="MIA-PaCa-2"),
    _p("Fig3E_elongated_fast", "elongated_fast_occupancy", KRASI, cell_line="MIA-PaCa-2"),
    _p("Fig3E_round_slow", "round_slow_occupancy", KRASI, cell_line="MIA-PaCa-2"),
    _p("Fig3E_area", "area_um2_median", KRASI, cell_line="MIA-PaCa-2"),
    _p("Fig3F_arrest", "arrest_coefficient", KRASI, cell_line="HPAF-II"),
    _p("Fig4G_motility", "motility_coefficient", TRIKE, cell_line="MIA-PaCa-2"),
    _p("Fig4H_motility", "motility_coefficient", COMBINED, cell_line="HPAF-II"),
    _p("Fig4J_elongated_fast", "elongated_fast_occupancy", COMBINED, cell_line="HPAF-II"),
    _p("FigS3B_motility", "motility_coefficient", DOSES, experiment="Titration_A"),
    _p("FigS3B_speed", "speed_um_per_min", DOSES, experiment="Titration_A"),
    _p("FigS3C_arrest", "arrest_coefficient", DOSES, experiment="Titration_A"),
    _p("FigS3C_elongated_fast", "elongated_fast_occupancy", DOSES, experiment="Titration_A"),
    _p("FigS3C_round_slow", "round_slow_occupancy", DOSES, experiment="Titration_A"),
    _p("FigS3C_area", "area_um2_median", DOSES, experiment="Titration_A"),
    _p("FigS3D_motility", "motility_coefficient", NO_IL15, experiment="MIAPaCa2_A",
       cell_line="MIA-PaCa-2"),
    _p("FigS3E_arrest", "arrest_coefficient", NO_IL15, experiment="MIAPaCa2_A",
       cell_line="MIA-PaCa-2"),
    _p("FigS3E_elongated_fast", "elongated_fast_occupancy", NO_IL15, experiment="MIAPaCa2_A",
       cell_line="MIA-PaCa-2"),
    _p("FigS3E_round_slow", "round_slow_occupancy", NO_IL15, experiment="MIAPaCa2_A",
       cell_line="MIA-PaCa-2"),
    _p("FigS3E_area", "area_um2_median", NO_IL15, experiment="MIAPaCa2_A",
       cell_line="MIA-PaCa-2"),
    _p("FigS3F_elongated_fast", "elongated_fast_occupancy", KRASI, cell_line="HPAF-II"),
    _p("FigS3F_round_slow", "round_slow_occupancy", KRASI, cell_line="HPAF-II"),
    _p("FigS5G_motility", "motility_coefficient", COMBINED, experiment="HPAC_A",
       cell_line="HPAC"),
]


def donors_with_all_arms(cells: pd.DataFrame, cell_line: str, arms: list[str],
                         measure: str) -> list[str]:
    """Donors of a cell line that have cells in every arm compared."""
    sub = cells[(cells["cell_line"] == cell_line) & cells[measure].notna()]
    have = sub.groupby("donor")["arm"].agg(set)
    return sorted(d for d, a in have.items() if set(arms) <= a)


def panel_cells(cells: pd.DataFrame, panel: Panel) -> tuple[pd.DataFrame, list[str]]:
    """The cells drawn in a panel, and the donors they come from."""
    if panel.experiment:
        sub = cells[cells["experiment"] == panel.experiment]
        donors = sorted(sub["donor"].unique())
    else:
        donors = donors_with_all_arms(cells, panel.cell_line, panel.arms, panel.measure)
        sub = cells[(cells["cell_line"] == panel.cell_line) & cells["donor"].isin(donors)]
    sub = sub[sub["arm"].isin(panel.arms) & sub[panel.measure].notna()]
    return sub, donors
