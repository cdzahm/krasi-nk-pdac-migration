import numpy as np

from nkmigration import prwm


def _prw_track(S, P, n, dt=1.0, rng=None):
    """Positions of a 1D persistent random walk (Ornstein-Uhlenbeck velocity)."""
    a = np.exp(-dt / P)
    v = np.empty(n)
    v[0] = rng.normal(0, S)
    for i in range(1, n):
        v[i] = a * v[i - 1] + rng.normal(0, S * np.sqrt(1 - a * a))
    return np.concatenate([[0.0], np.cumsum(v[:-1] * dt)])


def test_model_limits():
    tau = np.array([0.0, 1e-3, 1e4])
    msd = prwm.msd_model(tau, 2.0, 5.0, 0.0)
    assert msd[0] == 0.0
    assert np.isclose(msd[2] / tau[2], 2 * 2.0 ** 2 * 5.0, rtol=1e-2)  # diffusive limit


def test_recovers_motility_coefficient():
    rng = np.random.default_rng(1)
    S, P = 2.0, 4.0
    mus = [prwm.fit_axis(_prw_track(S, P, 120, rng=rng), 1.0)["mu"] for _ in range(40)]
    assert abs(np.median(mus) / (S ** 2 * P) - 1) < 0.35


def test_persistence_capped_at_duration():
    fit = prwm.fit_axis(np.arange(10) * 3.0, 1.0)  # straight line at 3 um/min
    assert np.isclose(fit["P"], 9.0, rtol=1e-3)


def test_short_tracks_not_fit():
    assert prwm.fit_axis(np.arange(6, dtype=float), 1.0) is None
    assert prwm.fit_axis(np.arange(7, dtype=float), 1.0) is not None
