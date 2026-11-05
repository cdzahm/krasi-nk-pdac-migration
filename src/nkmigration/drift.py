"""Rigid stage-drift correction.

Drift is a translation estimated by phase cross-correlation (10x subpixel
upsampling) on isotropically resampled volumes. Three ways of building the
per-frame shifts are used, chosen per recording in config/recordings.csv:

- reference_frame: every frame registered to one reference, the frame that
  differs least from its neighbors.
- windowed: every pair of frames within +/-3 frames registered directly, and
  the pairwise shifts combined by least squares into one trajectory.
- windowed_z: the same, on 1D Z-intensity profiles, so only Z is corrected.
  Pairs are weighted by 1/residual so a few poorly matched frames do not
  offset everything after them.

The registration input can be the collagen channel, collagen after
background subtraction and CLAHE, or the tumor channel with NK-cell signal
masked out. The shifts found are always applied to the unedited stacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, zoom
from scipy.ndimage import shift as ndi_shift
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation
from skimage.restoration import rolling_ball

WINDOW = 3
UPSAMPLE = 10

# Collagen enhancement: rolling-ball background (~58 um at 1.16 um/px) then
# CLAHE over ~37 um tiles.
BALL_RADIUS_PX = 50
CLAHE_KERNEL_PX = 32
CLAHE_CLIP = 0.03

# NK bleed-through halo removed from the tumor channel, in voxels (Z, Y, X).
NK_DILATION_ZYX = (1, 2, 2)


# Registration inputs

def enhance_collagen(stack: np.ndarray) -> np.ndarray:
    """Per-plane rolling-ball background subtraction, then CLAHE."""
    out = np.empty(stack.shape, dtype=np.float64)
    for t in range(stack.shape[0]):
        for z in range(stack.shape[1]):
            plane = stack[t, z].astype(np.float64)
            bg_sub = np.clip(plane - rolling_ball(plane, radius=BALL_RADIUS_PX), 0, None)
            lo, hi = bg_sub.min(), bg_sub.max()
            norm = (bg_sub - lo) / (hi - lo) if hi > lo else np.zeros_like(bg_sub)
            out[t, z] = equalize_adapthist(norm, clip_limit=CLAHE_CLIP,
                                           kernel_size=CLAHE_KERNEL_PX)
    return out


def _nk_mask(frame: np.ndarray) -> np.ndarray:
    mask = frame > threshold_otsu(frame)
    for axis, n in enumerate(NK_DILATION_ZYX):
        if n:
            structure = np.zeros((3, 3, 3), dtype=bool)
            idx = [1, 1, 1]
            idx[axis] = slice(0, 3)
            structure[tuple(idx)] = True
            mask = binary_dilation(mask, structure=structure, iterations=n)
    return mask


def tumor_minus_nk(tumor: np.ndarray, nk: np.ndarray) -> np.ndarray:
    """Tumor channel with Otsu-thresholded, dilated NK signal set to zero."""
    out = tumor.copy()
    for t in range(tumor.shape[0]):
        out[t] = np.where(_nk_mask(np.asarray(nk[t])), 0, out[t])
    return out


# Shift estimation

def _mse(a, b) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def _isotropic(volume: np.ndarray, voxel_um) -> np.ndarray:
    target = min(voxel_um)
    return zoom(volume.astype(np.float32), tuple(v / target for v in voxel_um), order=1)


def _zscore(a: np.ndarray) -> np.ndarray:
    return ((a - a.mean()) / (a.std() + 1e-8)).astype(np.float64)


def _pair_shift(a_iso: np.ndarray, b_iso: np.ndarray, iso_um: float):
    """Shift (um) that registers b onto a, and the residual after shifting."""
    shift_px, _, _ = phase_cross_correlation(_zscore(a_iso), _zscore(b_iso),
                                             upsample_factor=UPSAMPLE)
    residual = _mse(a_iso, ndi_shift(b_iso, shift_px, order=1))
    return tuple(float(s * iso_um) for s in shift_px), residual


def select_reference_frame(stack: np.ndarray) -> int:
    """Frame with the smallest mean squared difference to its neighbors,
    measured on maximum-intensity projections."""
    proj = [stack[t].max(axis=0) for t in range(stack.shape[0])]
    step = [_mse(proj[t], proj[t + 1]) for t in range(len(proj) - 1)]
    energy = [np.mean(([step[t - 1]] if t > 0 else []) + ([step[t]] if t < len(step) else []))
              for t in range(len(proj))]
    return int(np.argmin(energy))


def register_to_reference(stack: np.ndarray, voxel_um, ref: int) -> pd.DataFrame:
    iso_um = min(voxel_um)
    ref_iso = _isotropic(stack[ref], voxel_um)
    rows = []
    for t in range(stack.shape[0]):
        if t == ref:
            rows.append((t, 0.0, 0.0, 0.0, 0.0))
            continue
        (dz, dy, dx), res = _pair_shift(ref_iso, _isotropic(stack[t], voxel_um), iso_um)
        rows.append((t, dz, dy, dx, res))
    return pd.DataFrame(rows, columns=["frame", "dz_um", "dy_um", "dx_um", "residual"])


def _combine(edges: list[tuple[int, int, float]], n_t: int, weights=None) -> np.ndarray:
    """Least-squares trajectory from pairwise shifts, with frame 0 fixed at 0."""
    A = np.zeros((len(edges), n_t - 1))
    b = np.zeros(len(edges))
    for e, (i, j, d) in enumerate(edges):
        if j:
            A[e, j - 1] += 1.0
        if i:
            A[e, i - 1] -= 1.0
        b[e] = d
    if weights is not None:
        w = np.sqrt(weights)
        A, b = A * w[:, None], b * w
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.concatenate([[0.0], sol])


def register_windowed(stack: np.ndarray, voxel_um, window: int = WINDOW) -> pd.DataFrame:
    """Register all frame pairs within `window` and combine by least squares."""
    n_t = stack.shape[0]
    iso_um = min(voxel_um)
    cache: dict[int, np.ndarray] = {}

    def iso(t):
        if t not in cache:
            cache[t] = _isotropic(stack[t], voxel_um)
        return cache[t]

    pairs = []
    for i in range(n_t):
        for j in range(i + 1, min(i + window, n_t - 1) + 1):
            (dz, dy, dx), _ = _pair_shift(iso(i), iso(j), iso_um)
            pairs.append((i, j, dz, dy, dx))
        cache.pop(i, None)
    cols = [_combine([(i, j, p[k]) for i, j, *p in pairs], n_t) for k in range(3)]
    return pd.DataFrame({"frame": np.arange(n_t), "dz_um": cols[0],
                         "dy_um": cols[1], "dx_um": cols[2]})


def register_windowed_z(stack: np.ndarray, z_step_um: float, window: int = WINDOW) -> pd.DataFrame:
    """Z-only windowed registration on per-plane mean-intensity profiles.

    Plain cross-correlation is used here, not phase correlation: whitening a
    smooth 1D profile buries the true peak in amplified noise.
    """
    n_t = stack.shape[0]
    prof = [stack[t].astype(np.float64).mean(axis=(1, 2)) for t in range(n_t)]
    edges, residuals = [], []
    for i in range(n_t):
        for j in range(i + 1, min(i + window, n_t - 1) + 1):
            a = (prof[i] - prof[i].mean()) / (prof[i].std() + 1e-8)
            b = (prof[j] - prof[j].mean()) / (prof[j].std() + 1e-8)
            shift_px, _, _ = phase_cross_correlation(a, b, upsample_factor=UPSAMPLE,
                                                     normalization=None)
            dz_px = float(shift_px[0])
            edges.append((i, j, dz_px * z_step_um))
            residuals.append(_mse(prof[i], ndi_shift(prof[j], (dz_px,), order=1)))
    dz = _combine(edges, n_t, weights=1.0 / (np.asarray(residuals) + 1e-8))
    return pd.DataFrame({"frame": np.arange(n_t), "dz_um": dz, "dy_um": 0.0, "dx_um": 0.0})


def measure_drift(method: str, reg_input: np.ndarray | None, voxel_um) -> pd.DataFrame:
    if method == "reference_frame":
        ref = select_reference_frame(reg_input)
        shifts = register_to_reference(reg_input, voxel_um, ref)
        shifts["reference_frame"] = ref
        return shifts
    if method == "windowed":
        return register_windowed(reg_input, voxel_um)
    if method == "windowed_z":
        return register_windowed_z(reg_input, voxel_um[0])
    raise ValueError(f"unknown drift method {method!r}")


def apply_shifts(stack: np.ndarray, shifts: pd.DataFrame, voxel_um) -> np.ndarray:
    """Shift every frame by linear interpolation, in the stack's own dtype."""
    vz, vy, vx = voxel_um
    out = np.empty_like(stack)
    for row in shifts.itertuples(index=False):
        t = int(row.frame)
        moved = ndi_shift(stack[t].astype(np.float32),
                          (row.dz_um / vz, row.dy_um / vy, row.dx_um / vx), order=1)
        out[t] = np.clip(moved, 0, None).astype(stack.dtype)
    return out
