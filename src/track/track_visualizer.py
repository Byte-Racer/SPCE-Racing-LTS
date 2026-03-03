"""
Track visualisation: map view with curvature colour overlay and
curvature-vs-distance profile plot.

Matches the style conventions of ``src/utils/ggv_diagram.py``.

Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from src.track.track_representation import Track

logger = logging.getLogger(__name__)


class TrackVisualizer:
    """Generate static visualisation plots for a :class:`Track`.

    Typical usage::

        from src.track import Track, TrackVisualizer
        viz = TrackVisualizer(track)
        viz.plot_track_map(save_path="results/track_map.png")
        viz.plot_curvature_profile(save_path="results/curvature.png")

    Args:
        track: The :class:`Track` to visualise.
    """

    def __init__(self, track: Track) -> None:
        self.track = track
        logger.info(
            "TrackVisualizer: %d pts, length=%.1f m",
            track.segment_count,
            track.total_length,
        )

    # ------------------------------------------------------------------ #
    #  Plot 1 — Track map with curvature colour overlay                  #
    # ------------------------------------------------------------------ #

    def plot_track_map(
        self,
        ax: Optional[plt.Axes] = None,
        cmap: str = "RdYlGn_r",
        linewidth: float = 3.0,
        max_kappa: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Draw the track in 2-D coloured by |κ|.

        The colour map runs from **green** (straight/low curvature) to
        **red** (tight corner/high curvature), using a ``RdYlGn_r``
        (reversed red-yellow-green) colormap by default.

        Args:
            ax:         Existing ``Axes``; a new figure is created if *None*.
            cmap:       Matplotlib colormap name.
            linewidth:  Path line width in points.
            max_kappa:  Upper bound of the curvature colour scale in **1/m**.
                        Defaults to the 95th percentile of ``|κ|``.
            save_path:  If provided, save the figure to this path.

        Returns:
            ``(figure, axes)`` tuple.
        """
        track = self.track
        kappa_abs = np.abs(track.curvature)

        if max_kappa is None:
            max_kappa = float(np.percentile(kappa_abs, 95)) or 0.2
            max_kappa = max(max_kappa, 1e-3)

        norm = Normalize(vmin=0.0, vmax=max_kappa)
        cmap_fn = cm.get_cmap(cmap)

        # Build a LineCollection of 1-segment lines, each coloured separately
        x, y = track.x, track.y
        points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Close the loop segment
        closing = np.array([[[x[-1], y[-1]], [x[0], y[0]]]])
        segments = np.concatenate([segments, closing], axis=0)
        kappa_seg = np.concatenate([kappa_abs[:-1], [kappa_abs[-1]]])

        lc = LineCollection(segments, cmap=cmap_fn, norm=norm, linewidth=linewidth, zorder=2)
        lc.set_array(kappa_seg)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.get_figure()

        ax.add_collection(lc)

        # Mark start / finish
        ax.plot(x[0], y[0], "ko", markersize=8, zorder=5, label="Start / Finish")

        # Annotate apex of each corner
        for (s_start, s_end, radius) in track.corners:
            s_apex = (s_start + s_end) / 2.0
            xi, yi = track.interpolate_position(s_apex)
            ax.plot(xi, yi, "bx", markersize=6, zorder=4)

        # Formatting
        padding = max(  # dynamic padding so the map is not clipped
            (x.max() - x.min()) * 0.07,
            (y.max() - y.min()) * 0.07,
            1.0,
        )
        ax.set_xlim(x.min() - padding, x.max() + padding)
        ax.set_ylim(y.min() - padding, y.max() + padding)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, zorder=0)
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_title(
            f"Track Map  |  Length = {track.total_length:.0f} m  "
            f"|  {len(track.corners)} corners",
            fontsize=13,
            fontweight="bold",
        )

        cbar = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("|κ|  (1/m)", fontsize=11)

        # Legend label for start/finish
        ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()

        if save_path:
            _save(fig, save_path)

        return fig, ax

    # ------------------------------------------------------------------ #
    #  Plot 2 — Curvature vs distance                                    #
    # ------------------------------------------------------------------ #

    def plot_curvature_profile(
        self,
        ax: Optional[plt.Axes] = None,
        signed: bool = True,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the curvature profile against cumulative distance.

        Args:
            ax:        Existing ``Axes``; a new figure is created if *None*.
            signed:    If ``True`` (default) plots signed κ; if ``False``
                       plots ``|κ|`` only.
            save_path: If provided, save the figure to this path.

        Returns:
            ``(figure, axes)`` tuple.
        """
        track = self.track
        dist = track.distance / 1000.0  # convert to km for readability if long
        unit = "km" if track.total_length > 500 else "m"
        if unit == "m":
            dist = track.distance

        kappa = track.curvature if signed else np.abs(track.curvature)
        label_k = "Signed κ (1/m)" if signed else "|κ|  (1/m)"

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.get_figure()

        ax.plot(dist, kappa, color="#1f77b4", lw=1.5, label=label_k, zorder=3)

        if signed:
            ax.axhline(0, color="k", lw=0.6, ls="--", zorder=2)
            # Fill positive (left) and negative (right) separately
            ax.fill_between(dist, 0, kappa, where=(kappa > 0),
                            alpha=0.25, color="green", label="Left turn")
            ax.fill_between(dist, 0, kappa, where=(kappa < 0),
                            alpha=0.25, color="red", label="Right turn")

        # Shade corner regions
        for idx, (s0, s1, _) in enumerate(track.corners):
            d0 = s0 / 1000.0 if unit == "km" else s0
            d1 = s1 / 1000.0 if unit == "km" else s1
            ax.axvspan(d0, d1, alpha=0.08, color="orange" if idx % 2 == 0 else "purple")

        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f"Distance ({unit})", fontsize=12)
        ax.set_ylabel(label_k, fontsize=12)
        ax.set_title(
            "Curvature Profile",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim([dist[0], dist[-1]])
        plt.tight_layout()

        if save_path:
            _save(fig, save_path)

        return fig, ax

    # ------------------------------------------------------------------ #
    #  Combined report                                                    #
    # ------------------------------------------------------------------ #

    def generate_track_report(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Generate a 2-panel track analysis figure.

        Top panel: track map coloured by curvature.
        Bottom panel: curvature vs distance profile.

        Args:
            save_path: If provided, save the combined figure to this path.

        Returns:
            The :class:`matplotlib.figure.Figure` object.
        """
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 14),
            gridspec_kw={"height_ratios": [2.5, 1]},
        )

        self.plot_track_map(ax=axes[0])
        self.plot_curvature_profile(ax=axes[1])

        plt.tight_layout()

        if save_path:
            _save(fig, save_path)

        return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, save_path: str) -> None:
    """Save *fig* and create parent directories as needed."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    logger.info("Track plot saved → %s", save_path)
