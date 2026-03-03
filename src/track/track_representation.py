"""
Core Track representation for the QSS lap time simulator.

A ``Track`` object is an immutable snapshot of a discretised path: it holds
the (x, y) coordinates, cumulative distance, signed curvature, and a list
of identified corner segments.

Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Track:
    """Immutable, discretised representation of a racing track.

    All 1-D arrays must have the same length ``N`` (equal to
    ``segment_count``).

    Args:
        x:          X coordinates along the path in **metres**.
        y:          Y coordinates along the path in **metres**.
        distance:   Cumulative arc-length distance in **metres**.
                    ``distance[0] == 0.0``, ``distance[-1] == total_length``.
        curvature:  Signed curvature κ in **1/m** at each path point.
                    Positive → left turn, negative → right turn.
        corners:    List of ``(start_dist, end_dist, apex_radius)`` tuples
                    describing identified corner segments.  Distances are in
                    **metres**; radius is in **metres**.

    Attributes:
        total_length:   Total path length in **metres**.
        segment_count:  Number of discretised path points.
    """

    x: np.ndarray
    y: np.ndarray
    distance: np.ndarray
    curvature: np.ndarray
    corners: List[Tuple[float, float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Post-init validation                                               #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        n = len(self.x)
        if len(self.y) != n or len(self.distance) != n or len(self.curvature) != n:
            raise ValueError(
                "x, y, distance, and curvature arrays must all have the same length. "
                f"Got sizes: x={len(self.x)}, y={len(self.y)}, "
                f"distance={len(self.distance)}, curvature={len(self.curvature)}"
            )
        if n < 2:
            raise ValueError("A Track must have at least 2 points.")
        logger.info(
            "Track created: %d points, total_length=%.1f m, %d corners",
            n,
            float(self.distance[-1]),
            len(self.corners),
        )

    # ------------------------------------------------------------------ #
    #  Computed properties                                                #
    # ------------------------------------------------------------------ #

    @property
    def total_length(self) -> float:
        """Total track length in **metres**."""
        return float(self.distance[-1])

    @property
    def segment_count(self) -> int:
        """Number of discretised path points."""
        return len(self.x)

    # ------------------------------------------------------------------ #
    #  Convenience                                                        #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"Track(segment_count={self.segment_count}, "
            f"total_length={self.total_length:.1f} m, "
            f"corners={len(self.corners)})"
        )

    def interpolate_curvature(self, s: float) -> float:
        """Linearly interpolate curvature at an arbitrary distance *s*.

        Args:
            s: Arc-length position in **metres**.

        Returns:
            Interpolated curvature in **1/m**.
        """
        return float(np.interp(s, self.distance, self.curvature))

    def interpolate_position(self, s: float) -> Tuple[float, float]:
        """Linearly interpolate (x, y) at an arbitrary distance *s*.

        Args:
            s: Arc-length position in **metres**.

        Returns:
            ``(x, y)`` in **metres**.
        """
        xi = float(np.interp(s, self.distance, self.x))
        yi = float(np.interp(s, self.distance, self.y))
        return xi, yi
