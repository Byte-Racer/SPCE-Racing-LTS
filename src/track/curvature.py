"""
Curvature computation for track paths.

Provides signed curvature κ = 1/R (1/m) at each path point using the
standard 3-point tangent-cross-product formula, with optional Gaussian
smoothing and closed-loop handling.

Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute cumulative arc-length distance along a path.

    Args:
        x: X coordinates in **metres**.
        y: Y coordinates in **metres**.

    Returns:
        1-D array of cumulative distance values, shape ``(N,)`` in **metres**.
        First element is always 0.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    seg_lengths = np.hypot(dx, dy)
    return np.concatenate(([0.0], np.cumsum(seg_lengths)))


def compute_curvature(
    x: np.ndarray,
    y: np.ndarray,
    loop: bool = True,
    sigma: float = 2.0,
) -> np.ndarray:
    """Compute signed curvature κ at every path point.

    Uses the 3-point tangent-vector cross-product formula::

        v1 = P[i] - P[i-1],  v2 = P[i+1] - P[i]
        κ[i] = 2 × cross(v1, v2) / (|v1| × |v2| × |v1 + v2|)

    The numerator's sign encodes direction:
      *  positive  → left-hand bend (counter-clockwise)
      *  negative  → right-hand bend (clockwise)

    After raw computation the curvature is smoothed with a
    ``scipy.ndimage.gaussian_filter1d`` so noisy GPS / cone data does not
    produce spiky curvature profiles.

    Args:
        x:     X coordinates in **metres**, shape ``(N,)``.
        y:     Y coordinates in **metres**, shape ``(N,)``.
        loop:  If ``True``, the path is treated as a closed loop and the
               end points wrap around correctly.  If ``False``, curvature
               at the endpoints is copied from the adjacent interior point.
        sigma: Standard deviation for the Gaussian smoothing kernel in
               **samples**.  Larger values produce smoother curvature.
               Set to 0 to disable smoothing.

    Returns:
        1-D array of signed curvature values in **1/m**, shape ``(N,)``.
    """
    n = len(x)
    if n < 3:
        logger.warning("compute_curvature: fewer than 3 points — returning zeros")
        return np.zeros(n)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if loop:
        # Wrap endpoints — index i-1 and i+1 are valid for all i
        x_ext = np.concatenate(([x[-1]], x, [x[0]]))
        y_ext = np.concatenate(([y[-1]], y, [y[0]]))
    else:
        # Clamp: duplicate first/last points
        x_ext = np.concatenate(([x[0]], x, [x[-1]]))
        y_ext = np.concatenate(([y[0]], y, [y[-1]]))

    # Tangent vectors between consecutive points (length N)
    v1x = x_ext[1:-1] - x_ext[:-2]   # P[i] - P[i-1]
    v1y = y_ext[1:-1] - y_ext[:-2]
    v2x = x_ext[2:] - x_ext[1:-1]    # P[i+1] - P[i]
    v2y = y_ext[2:] - y_ext[1:-1]

    # 2-D cross product (scalar): v1 × v2 = v1x*v2y - v1y*v2x
    cross = v1x * v2y - v1y * v2x

    # Magnitudes
    mag1 = np.hypot(v1x, v1y)
    mag2 = np.hypot(v2x, v2y)

    # |v1 + v2|
    chord_x = v1x + v2x
    chord_y = v1y + v2y
    mag_sum = np.hypot(chord_x, chord_y)

    # κ = 2 × cross / (|v1| × |v2| × |v1+v2|)
    denom = mag1 * mag2 * mag_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 1e-12, 2.0 * cross / denom, 0.0)

    # Gaussian smoothing
    if sigma > 0:
        if loop:
            # Tile the signal so the filter sees a continuous loop
            kappa = _smooth_loop(kappa, sigma)
        else:
            kappa = gaussian_filter1d(kappa, sigma=sigma)

    logger.debug(
        "curvature: n=%d, |κ|_max=%.4f 1/m, σ=%.1f",
        n,
        float(np.max(np.abs(kappa))),
        sigma,
    )
    return kappa


def identify_corners(
    distance: np.ndarray,
    curvature: np.ndarray,
    min_curvature: float = 0.05,
    min_length: float = 2.0,
) -> List[Tuple[float, float, float]]:
    """Identify corner segments from the curvature profile.

    A corner is defined as a contiguous region where
    ``|κ| >= min_curvature`` and the total arc length is at least
    ``min_length`` metres.

    Args:
        distance:      Cumulative distance array in **metres**, shape ``(N,)``.
        curvature:     Signed curvature array in **1/m**, shape ``(N,)``.
        min_curvature: Absolute curvature threshold in **1/m** that separates
                       corners from straights (default 0.05 ↔ R = 20 m).
        min_length:    Minimum corner arc length in **metres** to report.

    Returns:
        List of ``(start_dist, end_dist, apex_radius)`` tuples where:
          * ``start_dist`` — distance at corner entry in **metres**.
          * ``end_dist``   — distance at corner exit in **metres**.
          * ``apex_radius`` — radius at tightest point in **metres** (always > 0).
    """
    in_corner = np.abs(curvature) >= min_curvature
    corners: List[Tuple[float, float, float]] = []

    i = 0
    n = len(distance)
    while i < n:
        if in_corner[i]:
            j = i
            while j < n and in_corner[j]:
                j += 1
            # Segment is distance[i] … distance[j-1]
            seg_len = distance[min(j - 1, n - 1)] - distance[i]
            if seg_len >= min_length:
                apex_kappa = float(np.max(np.abs(curvature[i:j])))
                apex_radius = 1.0 / apex_kappa if apex_kappa > 1e-9 else np.inf
                corners.append(
                    (float(distance[i]), float(distance[min(j - 1, n - 1)]), apex_radius)
                )
            i = j
        else:
            i += 1

    logger.debug("identify_corners: found %d corners", len(corners))
    return corners


def resample_path(
    x: np.ndarray,
    y: np.ndarray,
    spacing: float = 0.5,
    loop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a path to uniform arc-length spacing.

    Args:
        x:       X coordinates in **metres**.
        y:       Y coordinates in **metres**.
        spacing: Target arc-length spacing in **metres** between resampled
                 points (default 0.5 m).
        loop:    If ``True``, do not include the duplicate closing point.

    Returns:
        ``(x_new, y_new)`` resampled coordinate arrays.
    """
    dist = compute_distance(x, y)
    total = dist[-1]

    n_pts = max(3, int(np.round(total / spacing)) + 1)
    dist_new = np.linspace(0.0, total, n_pts)

    x_new = np.interp(dist_new, dist, x)
    y_new = np.interp(dist_new, dist, y)

    if loop:
        # Drop the duplicate point at the very end (coincides with start)
        x_new = x_new[:-1]
        y_new = y_new[:-1]

    return x_new, y_new


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _smooth_loop(kappa: np.ndarray, sigma: float) -> np.ndarray:
    """Apply circular Gaussian smoothing for a closed-loop curvature array."""
    n = len(kappa)
    # Tile 3 copies, filter the middle, take the middle section
    tiled = np.tile(kappa, 3)
    smoothed = gaussian_filter1d(tiled, sigma=sigma)
    return smoothed[n : 2 * n]
