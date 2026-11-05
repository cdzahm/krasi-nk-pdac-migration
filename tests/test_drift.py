import numpy as np
from scipy.ndimage import gaussian_filter, shift

from nkmigration import drift

VOXEL = (2.0, 1.0, 1.0)  # (z, y, x) um


def _blobs(seed=0, shape=(16, 64, 64), n=80):
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, np.float32)
    for _ in range(n):
        vol[rng.integers(3, shape[0] - 3), rng.integers(8, shape[1] - 8),
            rng.integers(8, shape[2] - 8)] = 1000
    return gaussian_filter(vol, (1.0, 2.0, 2.0))


def _moving(shifts_um):
    """Stack whose frame t is the base volume moved by -shifts_um[t], in
    whole voxels so the truth is exact."""
    base = _blobs()
    return np.stack([shift(base, [-s / v for s, v in zip(d, VOXEL)], order=0)
                     for d in shifts_um])


def test_reference_frame_recovers_shift():
    truth = [(0, 0, 0), (2, 3, -2), (4, -1, 1)]
    out = drift.register_to_reference(_moving(truth), VOXEL, ref=0)
    got = out[["dz_um", "dy_um", "dx_um"]].to_numpy()
    assert np.allclose(got, truth, atol=0.3)


def test_windowed_recovers_trajectory():
    truth = [(0, 0, 0), (0, 1, 1), (0, 2, 2), (2, 3, 2), (2, 4, 1)]
    out = drift.register_windowed(_moving(truth), VOXEL, window=2)
    assert np.allclose(out[["dz_um", "dy_um", "dx_um"]].to_numpy(), truth, atol=0.35)


def test_windowed_z_only_moves_z():
    truth = [(0, 0, 0), (2, 0, 0), (4, 0, 0), (4, 0, 0)]
    out = drift.register_windowed_z(_moving(truth), VOXEL[0], window=2)
    assert np.allclose(out["dz_um"], [t[0] for t in truth], atol=0.5)
    assert (out["dy_um"] == 0).all() and (out["dx_um"] == 0).all()


def test_apply_shifts_undoes_drift():
    truth = [(0, 0, 0), (2, 3, -2)]
    stack = _moving(truth).astype(np.uint16)
    out = drift.register_to_reference(stack, VOXEL, ref=0)
    fixed = drift.apply_shifts(stack, out, VOXEL).astype(float)
    before = np.abs(stack[1].astype(float) - stack[0]).mean()
    after = np.abs(fixed[1] - fixed[0]).mean()
    assert after < 0.2 * before
