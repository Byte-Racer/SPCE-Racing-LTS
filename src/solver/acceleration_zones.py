"""
Acceleration and Braking Integration Passes (Steps 2–4 of QSS).

Given the corner-speed profile (minimum speeds), this module performs:

* **Forward pass** — accelerates from each minimum-speed point toward the
  next, limited by motor/battery force.
* **Backward pass** — decelerates into each minimum-speed point,
  limited by braking force.  Run *twice* so that the wrap-around at
  lap start/end is handled correctly.
* **Final profile** — element-wise minimum of forward and backward results.

The output is used directly by :class:`~src.solver.qss_solver.QSSSolver`
to compute lap time and energy consumption.

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from src.track.track_representation import Track
    from src.vehicle.vehicle_model import VehicleDynamics

logger = logging.getLogger(__name__)


class AccelerationZones:
    """Forward and backward integration over the QSS speed profile.

    Args:
        vehicle:     Fully initialised ``VehicleDynamics`` instance.
        speed_floor: Minimum speed in **m/s** (avoids division by zero).
        mode:        ``'peak'`` or ``'continuous'`` power mode.
    """

    def __init__(
        self,
        vehicle: "VehicleDynamics",
        speed_floor: float = 0.5,
        mode: str = "peak",
    ) -> None:
        self.vehicle = vehicle
        self.speed_floor = speed_floor
        self.mode = mode

    # ------------------------------------------------------------------ #
    #  Public interface                                                   #
    # ------------------------------------------------------------------ #

    def integrate(
        self,
        track: "Track",
        v_corner: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run forward + backward passes and return the final speed profile.

        Args:
            track:    Discretised ``Track`` object.
            v_corner: Corner-speed limits from
                      :class:`~src.solver.speed_profile.SpeedProfileGenerator`,
                      shape ``(N,)`` in **m/s**.

        Returns:
            Tuple ``(v_final, ax_profile, limiting_factors)`` where:

            * ``v_final`` — speed at each track point in **m/s**, shape ``(N,)``
            * ``ax_profile`` — longitudinal acceleration in **m/s²**, shape
              ``(N-1,)``.  Positive = accelerating, negative = braking.
            * ``limiting_factors`` — list of strings length ``N-1`` describing
              which limit was active for each segment.
        """
        ds = np.diff(track.distance)   # segment lengths  [N-1]
        n = len(v_corner)

        # ── Forward pass ─────────────────────────────────────────────────
        v_fwd = self._forward_pass(v_corner, ds)

        # ── Backward pass (×2 for closed-loop wrap) ──────────────────────
        v_bwd = self._backward_pass(v_fwd, v_corner, ds)
        v_bwd = self._backward_pass(v_bwd, v_corner, ds)

        # ── Element-wise min = achievable profile ─────────────────────────
        v_final = np.minimum(v_fwd, v_bwd)
        v_final = np.clip(v_final, self.speed_floor, float(self.vehicle.top_speed))

        # ── Derive per-segment acceleration and limiting factor ───────────
        ax_profile, limiting_factors = self._compute_ax_profile(v_final, ds)

        logger.debug(
            "Integration done: v_min=%.2f m/s, v_max=%.2f m/s",
            float(np.min(v_final)),
            float(np.max(v_final)),
        )
        return v_final, ax_profile, limiting_factors

    # ------------------------------------------------------------------ #
    #  Forward pass                                                       #
    # ------------------------------------------------------------------ #

    def _forward_pass(self, v_corner: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """Accelerate forward from the start, respecting corner limits.

        For each segment ``i → i+1``:
            ``v_try = sqrt(v[i]² + 2 × ax_max × ds[i])``
            ``v[i+1] = min(v_try, v_corner[i+1])``

        The maximum longitudinal acceleration ``ax_max`` is evaluated at the
        *entry* speed ``v[i]`` and uses ``VehicleDynamics.max_longitudinal_acceleration``.

        Returns:
            Speed array in **m/s**, shape ``(N,)``.
        """
        n = len(v_corner)
        v = v_corner.copy()

        for i in range(n - 1):
            vi = float(v[i])
            speed = max(vi, self.speed_floor)
            ax = self.vehicle.max_longitudinal_acceleration(speed, throttle=100.0)
            v_try = float(np.sqrt(max(vi ** 2 + 2.0 * ax * float(ds[i]), 0.0)))
            v[i + 1] = min(v_try, float(v_corner[i + 1]))

        return v

    # ------------------------------------------------------------------ #
    #  Backward pass                                                      #
    # ------------------------------------------------------------------ #

    def _backward_pass(
        self, v_fwd: np.ndarray, v_corner: np.ndarray, ds: np.ndarray
    ) -> np.ndarray:
        """Decelerate backward from each corner minimum, respecting limits.

        For each segment ``i ← i+1`` (iterating in reverse):
            ``v_try = sqrt(v[i+1]² + 2 × ax_brake × ds[i])``
            ``v[i]  = min(v[i], v_try)``

        ``ax_brake`` is evaluated at the *exit* speed ``v[i+1]``; this is
        conservative and numerically stable.

        Args:
            v_fwd:    Forward-pass result to limit against.
            v_corner: Corner-speed limits (lower bound).
            ds:       Segment lengths in **m**.

        Returns:
            Speed array in **m/s**, shape ``(N,)``.
        """
        v = v_fwd.copy()
        n = len(v)

        for i in range(n - 2, -1, -1):
            v_exit = float(max(v[i + 1], self.speed_floor))
            ax_brake = self.vehicle.max_braking_deceleration(v_exit)
            v_try = float(np.sqrt(max(v_exit ** 2 + 2.0 * ax_brake * float(ds[i]), 0.0)))
            v[i] = min(float(v[i]), v_try)
            # Always respect the corner speed floor
            v[i] = max(v[i], float(v_corner[i]))

        return v

    # ------------------------------------------------------------------ #
    #  Acceleration profile derivation                                    #
    # ------------------------------------------------------------------ #

    def _compute_ax_profile(
        self, v: np.ndarray, ds: np.ndarray
    ) -> Tuple[np.ndarray, list]:
        """Derive per-segment longitudinal acceleration from the speed profile.

        Uses kinematics: ``ax = (v[i+1]² - v[i]²) / (2 × ds[i])``.

        Also classifies each segment as ``'accel'``, ``'braking'``, or
        ``'cruise'`` for telemetry.

        Args:
            v:  Final speed profile, shape ``(N,)``.
            ds: Segment lengths, shape ``(N-1,)``.

        Returns:
            ``(ax_array, labels)`` — shapes ``(N-1,)`` and list of ``N-1`` str.
        """
        v_sq_diff = v[1:] ** 2 - v[:-1] ** 2   # [N-1]
        ax = v_sq_diff / (2.0 * np.maximum(ds, 1e-6))

        labels: list = []
        for val in ax:
            if val > 0.1:
                labels.append("accel")
            elif val < -0.1:
                labels.append("braking")
            else:
                labels.append("cruise")

        return ax, labels
