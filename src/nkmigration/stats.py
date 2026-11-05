"""Group comparisons and effect sizes. The cell is the unit of analysis."""
from __future__ import annotations

import numpy as np
from scipy.stats import kruskal, mannwhitneyu, norm, rankdata

ALPHA = 0.05


def stars(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return ""
    for cut, s in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (ALPHA, "*")):
        if p < cut:
            return s
    return "ns"


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p values, in input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def _dunn(groups: list[np.ndarray], pairs: list[tuple[int, int]]) -> list[float]:
    """Dunn's test on the pooled ranks, with the tie correction."""
    pooled = np.concatenate(groups)
    ranks = rankdata(pooled)
    n = len(pooled)
    _, ties = np.unique(pooled, return_counts=True)
    var = n * (n + 1) / 12.0 - float(np.sum(ties ** 3 - ties)) / (12.0 * (n - 1))
    bounds = np.cumsum([0] + [len(g) for g in groups])
    mean_rank = [ranks[bounds[k]:bounds[k + 1]].mean() for k in range(len(groups))]
    out = []
    for a, b in pairs:
        se = np.sqrt(var * (1.0 / len(groups[a]) + 1.0 / len(groups[b])))
        z = (mean_rank[a] - mean_rank[b]) / se if se > 0 else 0.0
        out.append(float(2.0 * norm.sf(abs(z))))
    return out


def compare(values: list[np.ndarray]) -> dict:
    """Two groups: two-sided Mann-Whitney U. Three or more: Kruskal-Wallis,
    then Dunn's test with Holm correction. Pairs are shown as not
    significant when the Kruskal-Wallis p is 0.05 or more."""
    ok = [i for i, v in enumerate(values) if len(v) >= 2]
    if len(ok) < 2:
        return {"test": None, "omnibus_p": None, "pairs": []}
    if len(ok) == 2:
        a, b = ok
        p = float(mannwhitneyu(values[a], values[b], alternative="two-sided").pvalue)
        return {"test": "Mann-Whitney U", "omnibus_p": None,
                "pairs": [{"i": a, "j": b, "p": p, "p_adj": p, "shown": stars(p)}]}
    groups = [values[i] for i in ok]
    p_omni = float(kruskal(*groups).pvalue)
    local = [(x, y) for x in range(len(ok)) for y in range(x + 1, len(ok))]
    raw = _dunn(groups, local)
    adj = holm(raw)
    gated = p_omni >= ALPHA
    return {"test": "Kruskal-Wallis + Dunn (Holm)", "omnibus_p": p_omni,
            "pairs": [{"i": ok[x], "j": ok[y], "p": r, "p_adj": q,
                       "shown": "ns" if gated else stars(q)}
                      for (x, y), r, q in zip(local, raw, adj)]}


def cliffs_delta(ref: np.ndarray, treat: np.ndarray) -> float:
    """P(treat > ref) - P(treat < ref)."""
    ref, treat = _finite(ref), _finite(treat)
    if len(ref) < 2 or len(treat) < 2:
        return np.nan
    u = mannwhitneyu(treat, ref, alternative="two-sided").statistic
    return float(2.0 * u / (len(ref) * len(treat)) - 1.0)


def bootstrap_ci(ref: np.ndarray, treat: np.ndarray, rng: np.random.Generator,
                 n_boot: int = 1000) -> tuple[float, float]:
    """95% percentile interval for Cliff's delta, resampling cells."""
    ref, treat = _finite(ref), _finite(treat)
    if len(ref) < 2 or len(treat) < 2:
        return np.nan, np.nan
    boot = [cliffs_delta(ref[rng.integers(0, len(ref), len(ref))],
                         treat[rng.integers(0, len(treat), len(treat))]) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _finite(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v[np.isfinite(v)]
