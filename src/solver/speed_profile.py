"""
Minimum-Speed Profile Generator (Step 1 of QSS algorithm).

For each point on the track computes the maximum speed the vehicle can carry
through that curvature without exceeding its lateral acceleration limit.
This forms the *lower bound* on the speed profile — the car must never travel
faster than this through a corner.

Straights (κ ≈ 0) receive ``top_speed`` as their corner limit.

The lateral acceleration limit itself increases with speed (downforce), so
we use fixed-point iteration to find the self-consistent solution.

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.track.track_representation import Track
    from src.vehicle.vehicle_model import VehicleDynamics

logger = logging.getLogger(__name__)

# Curvature magnitude below which a segment is treated as a straight
_KAPPA_THRESHOLD = 1e-4   # 1/m  →  radius > 10 000 m is "straight"


class SpeedProfileGenerator:
    """Generate the corner-speed (minimum-speed) profile for a track.

    The QSS assumption: the car is always at the lateral adhesion limit in
    corners and at top speed on straights.  This gives the strictest lower
    bound on achievable speed; forward and backward integration passes then
    respect this bound while adding acceleration and braking zones.

    Args:
        vehicle:        Fully initialised ``VehicleDynamics`` instance.
        top_speed:      Maximum vehicle speed in **m/s**.
        speed_floor:    Minimum allowed speed in **m/s** (avoids div/0).
        max_iterations: Fixed-point iteration count for corner speed (≤ 10).
    """

    def __init__(
        self,
        vehicle: "VehicleDynamics",
        top_speed: float,
        speed_floor: float = 0.5,
        max_iterations: int = 5,
    ) -> None:
        self.vehicle = vehicle
        self.top_speed = top_speed
        self.speed_floor = speed_floor
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------ #
    #  Public interface                                                   #
    # ------------------------------------------------------------------ #

    def generate(self, track: "Track") -> np.ndarray:
        """Compute the corner-speed profile for *track*.

        Algorithm (per track point *i*):

        1. If ``|κ[i]| < threshold``: ``v[i] = top_speed`` (straight).
        2. Otherwise iterate:
           * ``v_guess = 15 m/s`` (initial)
           * ``ay_max = vehicle.max_lateral_acceleration(v_guess)``
           * ``v_new  = sqrt(ay_max / |κ[i]|)``
           * Repeat until convergence or ``max_iterations`` reached.

        Args:
            track: Discretised ``Track`` object.

        Returns:
            1-D ``np.ndarray`` of shape ``(N,)`` — corner speed limit in
            **m/s** at each track point.
        """
        kappa = np.abs(track.curvature)   # |κ|  [1/m]
        n = len(kappa)
        v_corner = np.full(n, self.top_speed, dtype=float)

        corner_mask = kappa >= _KAPPA_THRESHOLD
        n_corners = int(np.sum(corner_mask))
        logger.debug(
            "SpeedProfile: %d total points, %d corner points", n, n_corners
        )

        # Vectorise where possible: compute for all corner points together
        # using fixed-point iteration over all at once.
        kappa_corners = kappa[corner_mask]

        v_iter = np.full(n_corners, 15.0)  # initial guess 15 m/s

        for it in range(self.max_iterations):
            # Vectorised ay_max: call vehicle method per unique speed bucket
            # (vehicle method doesn't accept arrays, so iterate)
            ay_arr = np.array([
                self.vehicle.max_lateral_acceleration(float(v))
                for v in v_iter
            ])

            v_new = np.sqrt(np.maximum(ay_arr / kappa_corners, 0.0))
            v_new = np.clip(v_new, self.speed_floor, self.top_speed)

            delta = np.max(np.abs(v_new - v_iter))
            v_iter = v_new

            if delta < 0.01:   # converged (< 1 cm/s change)
                logger.debug("Corner speed converged in %d iterations", it + 1)
                break

        v_corner[corner_mask] = v_iter

        logger.debug(
            "Corner speed profile: min=%.2f m/s, mean=%.2f m/s",
            float(np.min(v_corner)),
            float(np.mean(v_corner)),
        )
        return v_corner
