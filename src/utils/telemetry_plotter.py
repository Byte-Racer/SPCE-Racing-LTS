"""
Telemetry Plotter for a single simulated lap.

Produces a 5-panel comprehensive figure (speed, longitudinal acceleration,
battery power, motor temperature, limiting-factor band) all sharing the
distance axis, plus individual single-panel helpers.

No regenerative braking is modelled in this simulator.

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from src.solver.lap_results import LapResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limiting-factor colour map
# ---------------------------------------------------------------------------
_FACTOR_COLOURS: Dict[str, str] = {
    "torque_curve":     "#1f77b4",   # blue
    "voltage":          "#9467bd",   # purple
    "thermal":          "#d62728",   # red
    "traction":         "#2ca02c",   # green
    "battery":          "#ff7f0e",   # orange
    "driver_request":   "#7f7f7f",   # grey
}
_FACTOR_DEFAULT_COLOUR = "#7f7f7f"


def _compute_battery_power_kw(lap: LapResult) -> np.ndarray:
    """Derive per-segment battery power draw from SOC changes.

    Since no regenerative braking exists, power is always non-negative
    (draw from battery only).

    Args:
        lap: Completed :class:`~src.solver.lap_results.LapResult`.

    Returns:
        1-D array of length ``N-1`` containing battery power in **kW**.
        Returns zeros if SOC data is insufficient.
    """
    n = len(lap.distance)
    if n < 2 or len(lap.soc_profile) < 2:
        return np.zeros(max(n - 1, 0))

    soc_drop = -np.diff(lap.soc_profile)          # positive = discharging
    total_drop = lap.soc_profile[0] - lap.soc_profile[-1]

    if total_drop > 1e-9:
        energy_per_seg_wh = soc_drop / total_drop * lap.energy_consumed
    else:
        energy_per_seg_wh = np.zeros(n - 1)

    v_avg = (lap.speed_profile[:-1] + lap.speed_profile[1:]) / 2.0
    v_avg = np.maximum(v_avg, 0.1)                # avoid division by zero
    ds = np.diff(lap.distance)
    dt_s = ds / v_avg                             # seconds per segment

    power_kw = energy_per_seg_wh * 3_600.0 / dt_s / 1_000.0
    return np.maximum(power_kw, 0.0)              # no regen


class TelemetryPlotter:
    """Visualise per-lap telemetry from a QSS simulation.

    All ``plot_*`` methods return a :class:`matplotlib.figure.Figure` and
    optionally save to *save_path* at 300 dpi.

    Args:
        lap_result: Completed lap telemetry container.

    Example::

        from src.utils.telemetry_plotter import TelemetryPlotter
        plotter = TelemetryPlotter(lap_result)
        fig = plotter.plot_comprehensive_telemetry(save_path="results/telemetry.png")
    """

    def __init__(self, lap_result: LapResult) -> None:
        self.lap = lap_result
        logger.info(
            "TelemetryPlotter created — lap_time=%.3f s, %d track points",
            lap_result.lap_time,
            len(lap_result.distance),
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

    def _seg_dist(self) -> np.ndarray:
        """Midpoint distances for segment-length arrays (N-1)."""
        return (self.lap.distance[:-1] + self.lap.distance[1:]) / 2.0

    # ------------------------------------------------------------------ #
    #  Individual panels                                                  #
    # ------------------------------------------------------------------ #

    def plot_speed_trace(self, save_path: Optional[str] = None) -> plt.Figure:
        """Speed (km/h) vs distance with fill, max/min/avg annotations.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        dist = self.lap.distance
        v_kmh = self.lap.speed_profile * 3.6

        ax.plot(dist, v_kmh, color="#1f77b4", lw=1.8, zorder=3)
        ax.fill_between(dist, 0, v_kmh, alpha=0.18, color="#1f77b4")

        avg_kmh = self.lap.avg_speed * 3.6
        ax.axhline(avg_kmh, color="#ff7f0e", ls="--", lw=1.2,
                   label=f"Avg {avg_kmh:.1f} km/h")

        i_max = int(np.argmax(v_kmh))
        i_min = int(np.argmin(v_kmh))
        ax.annotate(f"Max {v_kmh[i_max]:.1f}",
                    xy=(dist[i_max], v_kmh[i_max]),
                    xytext=(dist[i_max], v_kmh[i_max] + 3),
                    fontsize=8, color="navy", ha="center")
        ax.annotate(f"Min {v_kmh[i_min]:.1f}",
                    xy=(dist[i_min], v_kmh[i_min]),
                    xytext=(dist[i_min], v_kmh[i_min] + 3),
                    fontsize=8, color="darkred", ha="center")

        ax.set_xlabel("Distance (m)", fontsize=12)
        ax.set_ylabel("Speed (km/h)", fontsize=12)
        ax.set_title("Speed Trace", fontsize=13, fontweight="bold")
        ax.set_xlim([0, dist[-1]])
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_acceleration_trace(self, save_path: Optional[str] = None) -> plt.Figure:
        """Longitudinal acceleration (g) vs distance with pos/neg fill.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        if len(self.lap.ax_profile):
            dist_seg = self._seg_dist()
            ax_g = self.lap.ax_profile / 9.81
            ax.plot(dist_seg, ax_g, color="#555555", lw=1.2, zorder=3)
            ax.fill_between(dist_seg, 0, ax_g,
                            where=ax_g >= 0, alpha=0.35, color="#2ca02c",
                            label="Acceleration")
            ax.fill_between(dist_seg, 0, ax_g,
                            where=ax_g < 0, alpha=0.35, color="#d62728",
                            label="Braking")
            ax.axhline(0, color="k", ls="--", lw=0.6)

        ax.set_xlabel("Distance (m)", fontsize=12)
        ax.set_ylabel("Longitudinal Accel (g)", fontsize=12)
        ax.set_title("Acceleration Trace", fontsize=13, fontweight="bold")
        ax.set_xlim([0, self.lap.distance[-1]])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_energy_trace(self, save_path: Optional[str] = None) -> plt.Figure:
        """Battery power draw (kW) vs distance.  No regen component.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        if len(self.lap.distance) > 1:
            dist_seg = self._seg_dist()
            power_kw = _compute_battery_power_kw(self.lap)
            ax.plot(dist_seg, power_kw, color="#ff7f0e", lw=1.8, zorder=3)
            ax.fill_between(dist_seg, 0, power_kw, alpha=0.20, color="#ff7f0e")

        ax.set_xlabel("Distance (m)", fontsize=12)
        ax.set_ylabel("Battery Power (kW)", fontsize=12)
        ax.set_title("Battery Power Draw", fontsize=13, fontweight="bold")
        ax.set_xlim([0, self.lap.distance[-1]])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_motor_thermal(self, save_path: Optional[str] = None) -> plt.Figure:
        """Motor temperature (°C) vs distance with derating limit lines.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        if len(self.lap.motor_temp_profile):
            ax.plot(self.lap.distance, self.lap.motor_temp_profile,
                    color="#d62728", lw=1.8, zorder=3)

        ax.axhline(90,  color="#ff7f0e", ls="--", lw=1.2, label="Derate  90 °C")
        ax.axhline(100, color="#d62728",  ls="--", lw=1.2, label="Cont.  100 °C")
        ax.axhline(120, color="#7f0000",  ls="--", lw=1.2, label="Peak   120 °C")

        if self.lap.thermal_derating_occurred:
            ax.annotate("⚠ Derating occurred",
                        xy=(0.97, 0.95), xycoords="axes fraction",
                        ha="right", va="top", fontsize=10,
                        color="#d62728",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="lightyellow", ec="#d62728", lw=1))

        ax.set_xlabel("Distance (m)", fontsize=12)
        ax.set_ylabel("Motor Temperature (°C)", fontsize=12)
        ax.set_title("Motor Thermal State", fontsize=13, fontweight="bold")
        ax.set_xlim([0, self.lap.distance[-1]])
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_limiting_factors(self, save_path: Optional[str] = None) -> plt.Figure:
        """Colour-coded limiting-factor band vs distance.

        Colours: torque_curve=blue, voltage=purple, thermal=red,
        traction=green, battery=orange, driver_request=grey.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(12, 2))

        factors = self.lap.limiting_factor_profile
        if factors and len(self.lap.distance) > 1:
            d_starts = self.lap.distance[:-1]
            d_ends   = self.lap.distance[1:]
            for d0, d1, factor in zip(d_starts, d_ends, factors):
                colour = _FACTOR_COLOURS.get(factor, _FACTOR_DEFAULT_COLOUR)
                ax.axvspan(d0, d1, ymin=0, ymax=1, color=colour, alpha=0.80)

        # Legend patches
        legend_handles = [
            mpatches.Patch(color=c, label=lbl)
            for lbl, c in _FACTOR_COLOURS.items()
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=7, ncol=3, framealpha=0.85)

        ax.set_xlabel("Distance (m)", fontsize=12)
        ax.set_yticks([])
        ax.set_title("Limiting Factor", fontsize=13, fontweight="bold")
        ax.set_xlim([0, self.lap.distance[-1]])
        ax.grid(False)
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    # ------------------------------------------------------------------ #
    #  Comprehensive 5-panel figure                                       #
    # ------------------------------------------------------------------ #

    def plot_comprehensive_telemetry(
            self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """5-panel telemetry figure sharing the distance x-axis.

        Panels
        ------
        1. Speed (km/h) — blue fill, max/min/avg markers
        2. Longitudinal acceleration (g) — green/red fill
        3. Battery power draw (kW) — orange, no regen
        4. Motor temperature (°C) — red, dashed limit lines
        5. Limiting-factor band — colour-coded by factor type

        Args:
            save_path: Optional path; saved at 300 dpi.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        lap = self.lap
        avg_kmh  = lap.avg_speed * 3.6
        energy   = lap.energy_consumed

        sup_title = (
            f"Lap Telemetry — {lap.lap_time:.2f} s  |  "
            f"Avg: {avg_kmh:.1f} km/h  |  "
            f"Energy: {energy:.0f} Wh"
        )

        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(sup_title, fontsize=13, fontweight="bold", y=0.995)

        gs = gridspec.GridSpec(
            5, 1,
            hspace=0.05,
            height_ratios=[3, 2, 2, 2, 1],
        )

        dist = lap.distance
        v_kmh = lap.speed_profile * 3.6

        # ── Panel 1: Speed ─────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(dist, v_kmh, color="#1f77b4", lw=1.8, zorder=3)
        ax1.fill_between(dist, 0, v_kmh, alpha=0.18, color="#1f77b4")
        avg_kmh_val = avg_kmh
        ax1.axhline(avg_kmh_val, color="#ff7f0e", ls="--", lw=1.2,
                    label=f"Avg {avg_kmh_val:.1f} km/h")
        i_max = int(np.argmax(v_kmh))
        i_min = int(np.argmin(v_kmh))
        ax1.annotate(f"Max {v_kmh[i_max]:.1f}",
                     xy=(dist[i_max], v_kmh[i_max]),
                     xytext=(dist[i_max], v_kmh[i_max] + 3),
                     fontsize=7, color="navy", ha="center", zorder=5)
        ax1.annotate(f"Min {v_kmh[i_min]:.1f}",
                     xy=(dist[i_min], v_kmh[i_min]),
                     xytext=(dist[i_min], v_kmh[i_min] + 3),
                     fontsize=7, color="darkred", ha="center", zorder=5)
        ax1.set_ylabel("Speed (km/h)", fontsize=12)
        ax1.set_ylim(bottom=0)
        ax1.legend(fontsize=9, loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelbottom=False)

        # ── Panel 2: Longitudinal acceleration ────────────────────────
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        if len(lap.ax_profile):
            dist_seg = self._seg_dist()
            ax_g = lap.ax_profile / 9.81
            ax2.plot(dist_seg, ax_g, color="#555555", lw=1.0, zorder=3)
            ax2.fill_between(dist_seg, 0, ax_g,
                             where=ax_g >= 0, alpha=0.35, color="#2ca02c",
                             label="Accel")
            ax2.fill_between(dist_seg, 0, ax_g,
                             where=ax_g < 0, alpha=0.35, color="#d62728",
                             label="Brake")
            ax2.axhline(0, color="k", ls="--", lw=0.6)
        ax2.set_ylabel("Long. Accel (g)", fontsize=12)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelbottom=False)

        # ── Panel 3: Battery power ─────────────────────────────────────
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        if len(dist) > 1:
            dist_seg = self._seg_dist()
            power_kw = _compute_battery_power_kw(lap)
            ax3.plot(dist_seg, power_kw, color="#ff7f0e", lw=1.5, zorder=3)
            ax3.fill_between(dist_seg, 0, power_kw,
                             alpha=0.20, color="#ff7f0e")
        ax3.set_ylabel("Battery Power (kW)", fontsize=12)
        ax3.set_ylim(bottom=0)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelbottom=False)

        # ── Panel 4: Motor temperature ─────────────────────────────────
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        if len(lap.motor_temp_profile):
            ax4.plot(dist, lap.motor_temp_profile,
                     color="#d62728", lw=1.5, zorder=3)
        ax4.axhline(90,  color="#ff7f0e", ls="--", lw=1.0,
                    label="90 °C derate")
        ax4.axhline(100, color="#d62728",  ls="--", lw=1.0,
                    label="100 °C cont.")
        ax4.axhline(120, color="#7f0000",  ls="--", lw=1.0,
                    label="120 °C peak")
        if lap.thermal_derating_occurred:
            ax4.annotate("⚠ Derating occurred",
                         xy=(0.97, 0.92), xycoords="axes fraction",
                         ha="right", va="top", fontsize=9,
                         color="#d62728",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   fc="lightyellow", ec="#d62728", lw=1))
        ax4.set_ylabel("Motor Temp (°C)", fontsize=12)
        ax4.legend(fontsize=8, loc="upper left")
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelbottom=False)

        # ── Panel 5: Limiting factor band ─────────────────────────────
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        factors = lap.limiting_factor_profile
        if factors and len(dist) > 1:
            d_starts = dist[:-1]
            d_ends   = dist[1:]
            for d0, d1, factor in zip(d_starts, d_ends, factors):
                colour = _FACTOR_COLOURS.get(factor, _FACTOR_DEFAULT_COLOUR)
                ax5.axvspan(d0, d1, ymin=0, ymax=1,
                            color=colour, alpha=0.80)
        legend_handles = [
            mpatches.Patch(color=c, label=lbl)
            for lbl, c in _FACTOR_COLOURS.items()
        ]
        ax5.legend(handles=legend_handles, loc="upper right",
                   fontsize=6, ncol=3, framealpha=0.85)
        ax5.set_yticks([])
        ax5.set_xlabel("Distance (m)", fontsize=12)
        ax5.set_ylabel("Limit", fontsize=10)
        ax5.grid(False)

        ax1.set_xlim([0, dist[-1]])

        self._save(fig, save_path)
        logger.info("Comprehensive telemetry figure generated.")
        return fig
