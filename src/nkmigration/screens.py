"""Screens used to choose the reported measures from the candidates.

Each unit is one donor in one comparison (KRASi, TriKE or TriKE+KRASi, each
against IL-15), and each measure's effect in a unit is Cliff's delta over
cells.

1. Redundancy: measures whose effects correlate across units at
   |Spearman rho| >= 0.8 are grouped by average-linkage clustering, and one
   representative is suggested per group.
2. Donor consistency: whether a measure's effect has the same sign in every
   donor, and whether every donor's 95% bootstrap interval excludes zero.

Both screens guided the choice of measures; they did not make it.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from . import percell, stats

COMPARISONS = {"KRASi": ("IL-15", "KRASi"), "TriKE": ("IL-15", "TriKE"),
               "TriKE+KRASi": ("IL-15", "TriKE+KRASi")}
CELL_LINES = ["MIA-PaCa-2", "HPAF-II", "HPAC"]
RHO_MIN = 0.8
MIN_UNITS = 10
NEGLIGIBLE = 0.147   # |delta| below this is negligible
SEED = 20260918

# Measures that grow with track length rather than with motility.
LENGTH_DEPENDENT = {"net_displacement_um", "path_length_um", "state_transitions"}
KEEP_AS_OWN = "motility_coefficient"


def units(cells: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(comparison, cell line, donor) for every donor with both arms."""
    out = []
    for comp, (ref, treat) in COMPARISONS.items():
        for cl in CELL_LINES:
            sub = cells[cells["cell_line"] == cl]
            for donor, g in sub.groupby("donor"):
                if {ref, treat} <= set(g["arm"]):
                    out.append((comp, cl, donor))
    return out


def _effect(args):
    measure, comp, cl, donor, ref, treat, seed = args
    rng = np.random.default_rng(seed)
    lo, hi = stats.bootstrap_ci(ref, treat, rng)
    return {"measure": measure, "comparison": comp, "cell_line": cl, "donor": donor,
            "delta": stats.cliffs_delta(ref, treat), "ci_low": lo, "ci_high": hi,
            "n_ref": int(np.isfinite(ref).sum()), "n_treat": int(np.isfinite(treat).sum())}


def effects(cells: pd.DataFrame, measures: list[str], workers: int = 4) -> pd.DataFrame:
    """Cliff's delta and its bootstrap interval for every measure and unit."""
    tasks = []
    for m in measures:
        for comp, cl, donor in units(cells):
            ref_arm, treat_arm = COMPARISONS[comp]
            sub = cells[(cells["cell_line"] == cl) & (cells["donor"] == donor)]
            ref = sub.loc[sub["arm"] == ref_arm, m].to_numpy(float)
            treat = sub.loc[sub["arm"] == treat_arm, m].to_numpy(float)
            tasks.append((m, comp, cl, donor, ref, treat, SEED + len(tasks)))
    with Pool(workers) as pool:
        return pd.DataFrame(pool.map(_effect, tasks, chunksize=4))


def redundancy(eff: pd.DataFrame) -> pd.DataFrame:
    """Group measures by the correlation of their effects across units."""
    mat = eff.pivot_table(index=["comparison", "cell_line", "donor"], columns="measure",
                          values="delta")
    names = list(mat.columns)
    rho = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = mat.iloc[:, i].notna() & mat.iloc[:, j].notna()
            r = (spearmanr(mat.iloc[:, i][both], mat.iloc[:, j][both]).statistic
                 if both.sum() >= MIN_UNITS else np.nan)
            rho[i, j] = rho[j, i] = r
    dist = 1.0 - np.abs(rho)
    dist[np.isnan(dist)] = 1.0     # no evidence of correlation: don't merge
    np.fill_diagonal(dist, 0.0)
    groups = fcluster(linkage(squareform((dist + dist.T) / 2, checks=False), "average"),
                      t=1.0 - RHO_MIN, criterion="distance")
    out = pd.DataFrame({"measure": names, "group": groups,
                        "n_units": mat.notna().sum().reindex(names).to_numpy()})
    out["representative"] = False
    for _, g in out.groupby("group"):
        members = list(g["measure"])
        if KEEP_AS_OWN in members:
            out.loc[out["measure"] == KEEP_AS_OWN, "representative"] = True
            members.remove(KEEP_AS_OWN)
        if members:
            best = min(members, key=lambda m: _rank(m, int(out.loc[out.measure == m, "n_units"].iloc[0])))
            out.loc[out["measure"] == best, "representative"] = True
    return out.sort_values(["group", "measure"]).reset_index(drop=True)


def _rank(measure: str, n_units: int):
    """Prefer physical units, then measures that do not scale with track
    length, then wider coverage."""
    label = percell.BY_NAME[measure].label
    physical = any(u in label for u in ("µm", "°", "frames"))
    return (not physical, measure in LENGTH_DEPENDENT, -n_units)


def consistency(eff: pd.DataFrame) -> pd.DataFrame:
    """Per measure and comparison: does every donor agree in sign, and does
    every donor's interval exclude zero on that side?"""
    rows = []
    for (m, comp), g in eff.groupby(["measure", "comparison"], sort=False):
        d = g["delta"].to_numpy(float)
        complete = np.isfinite(d).all()
        same_sign = bool(complete and (np.all(d > 0) or np.all(d < 0)))
        up = same_sign and d[0] > 0
        ci_side = bool(same_sign and ((g["ci_low"] > 0).all() if up else (g["ci_high"] < 0).all()))
        rows.append({"measure": m, "comparison": comp, "n_donors": len(g),
                     "n_up": int((d > 0).sum()), "n_down": int((d < 0).sum()),
                     "median_delta": float(np.nanmedian(d)),
                     "same_sign_all_donors": same_sign,
                     "intervals_exclude_zero": ci_side,
                     "all_non_negligible": bool(ci_side and np.all(np.abs(d) >= NEGLIGIBLE)),
                     "direction": ("up" if up else "down") if same_sign else ""})
    return pd.DataFrame(rows)
