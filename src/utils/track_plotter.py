"""
Track Plotter — x-y map and curvature profile visualisation.

Renders the racing line as a colour-mapped :class:`~matplotlib.collections.LineCollection`
(curvature or speed), plus a curvature-vs-distance profile with secondary
radius axis.

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar

from src.solver.lap_results import LapResult
from src.track.track_representation import Track

logger = logging.getLogger(__name__)


class TrackPlotter:
    """Visualise track geometry and speed-on-track data.

    All ``plot_*`` methods return a :class:`matplotlib.figure.Figure` and
    optionally save to *save_path* at 300 dpi.

    Args:
        track: Immutable :class:`~src.track.track_representation.Track` object.

    Example::

        from src.utils.track_plotter import TrackPlotter
        plotter = TrackPlotter(track)
        fig = plotter.plot_track_map(save_path="results/track_map.png")
    """

    def __init__(self, track: Track) -> None:
        self.track = track
        logger.info(
            "TrackPlotter created — %d pts, %.1f m",
            track.segment_count,
            track.total_length,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _save(self, fig: plt.Figure, save_path: Optional[str]) -> None:
        """Save *fig* to *save_path* at 300 dpi if a path is given."""
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Figure saved → %s", save_path)

    @staticmethod
    def _build_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Build (N-1, 2, 2) segment array for LineCollection."""
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        return np.concatenate([pts[:-1], pts[1:]], axis=1)

    # ------------------------------------------------------------------ #
    #  Track map coloured by curvature                                    #
    # ------------------------------------------------------------------ #

    def plot_track_map(
            self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """x-y track map coloured by curvature magnitude.

        Colour map: ``RdYlGn_r`` (red = high curvature, green = straight).
        Includes start/finish marker and aspect-equal axes.

        Args:
            save_path: Optional path; saved at 300 dpi.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        t = self.track
        x, y = t.x, t.y
        kappa = np.abs(t.curvature)

        segments = self._build_segments(x, y)
        # One colour value per segment: average of the two end-point values
        seg_kappa = (kappa[:-1] + kappa[1:]) / 2.0

        fig, ax = plt.subplots(figsize=(9, 9))

        norm = plt.Normalize(vmin=0.0, vmax=float(np.percentile(kappa, 95)) or 0.1)
        lc = LineCollection(segments, cmap="RdYlGn_r", norm=norm,
                            linewidth=2.5, zorder=3)
        lc.set_array(seg_kappa)
        ax.add_collection(lc)

        cbar: Colorbar = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label("κ (1/m)", fontsize=11)

        # Start/finish marker
        ax.plot(x[0], y[0], "^", color="lime", ms=10, zorder=5,
                label="Start / Finish", markeredgecolor="black", markeredgewidth=0.8)
        ax.legend(fontsize=9, loc="upper right")

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_title("Track Map — Curvature", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    # ------------------------------------------------------------------ #
    #  Track map coloured by speed                                        #
    # ------------------------------------------------------------------ #

    def plot_track_with_speed(
            self,
            lap_result: LapResult,
            save_path: Optional[str] = None,
    ) -> plt.Figure:
        """x-y track map coloured by vehicle speed from a lap result.

        The speed profile is interpolated from *lap_result* arc-length
        positions onto track coordinate positions.
        Colour map: ``RdYlGn`` (red = slow, green = fast).

        Args:
            lap_result: Completed lap telemetry from the solver.
            save_path:  Optional path; saved at 300 dpi.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        t = self.track
        x, y = t.x, t.y

        # Interpolate speed (km/h) at every track point
        speed_kmh_at_pts = np.interp(
            t.distance,
            lap_result.distance,
            lap_result.speed_profile * 3.6,
        )

        segments = self._build_segments(x, y)
        seg_speed = (speed_kmh_at_pts[:-1] + speed_kmh_at_pts[1:]) / 2.0

        fig, ax = plt.subplots(figsize=(9, 9))

        norm = plt.Normalize(vmin=float(np.min(seg_speed)),
                             vmax=float(np.max(seg_speed)))
        lc = LineCollection(segments, cmap="RdYlGn", norm=norm,
                            linewidth=2.5, zorder=3)
        lc.set_array(seg_speed)
        ax.add_collection(lc)

        cbar: Colorbar = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label("Speed (km/h)", fontsize=11)

        # Start/finish marker
        ax.plot(x[0], y[0], "^", color="lime", ms=10, zorder=5,
                label="Start / Finish", markeredgecolor="black", markeredgewidth=0.8)
        ax.legend(fontsize=9, loc="upper right")

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_title("Track Map — Speed", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    # ------------------------------------------------------------------ #
    #  Curvature profile with secondary radius axis                       #
    # ------------------------------------------------------------------ #

    def plot_curvature_profile(
            self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Curvature (1/m) vs distance with secondary radius (m) axis.

        The secondary y-axis is clipped at 1 000 m (effectively infinite
        on straights) so it remains readable.

        Args:
            save_path: Optional path; saved at 300 dpi.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        t = self.track
        dist = t.distance
        kappa = t.curvature

        fig, ax1 = plt.subplots(figsize=(13, 5))

        ax1.fill_between(dist, 0, kappa, alpha=0.30, color="#1f77b4")
        ax1.plot(dist, kappa, color="#1f77b4", lw=1.5, zorder=3)
        ax1.axhline(0, color="k", ls="--", lw=0.6)

        ax1.set_xlabel("Distance (m)", fontsize=12)
        ax1.set_ylabel("Curvature κ (1/m)", fontsize=12)
        ax1.set_title("Curvature Profile", fontsize=13, fontweight="bold")
        ax1.set_xlim([0, dist[-1]])
        ax1.grid(True, alpha=0.3)

        # Secondary y-axis: radius R = 1/κ (clipped at 1 000 m)
        ax2 = ax1.twinx()
        _R_MAX = 1_000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = np.where(np.abs(kappa) > 1e-6, 1.0 / np.abs(kappa), _R_MAX)
        radius = np.clip(radius, 0.0, _R_MAX)
        ax2.plot(dist, radius, color="#ff7f0e", lw=1.0,
                 alpha=0.5, ls=":", label="R (m)")
        ax2.set_ylabel("Radius R (m)", fontsize=12, color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        ax2.set_ylim([0, _R_MAX])
        ax2.legend(fontsize=9, loc="upper right")

        plt.tight_layout()
        self._save(fig, save_path)
        return fig
