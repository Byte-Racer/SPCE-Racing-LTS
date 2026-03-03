"""
Endurance Plotter — multi-lap simulation visualisation.

Provides a 2×2 comprehensive overview plus individual progression charts
for lap times, energy budget, thermal state, and lap comparison overlays.

No regenerative braking is modelled in this simulator.

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.solver.lap_results import EnduranceResult, LapResult

logger = logging.getLogger(__name__)


class EndurancePlotter:
    """Visualise aggregated endurance run results.

    All ``plot_*`` methods return a :class:`matplotlib.figure.Figure` and
    optionally save to *save_path* at 300 dpi.

    Args:
        endurance_result: Completed endurance simulation container.

    Example::

        from src.utils.endurance_plotter import EndurancePlotter
        plotter = EndurancePlotter(endurance_result)
        fig = plotter.plot_comprehensive_endurance(save_path="results/endurance.png")
    """

    def __init__(self, endurance_result: EnduranceResult) -> None:
        self.result = endurance_result
        self._laps = endurance_result.laps
        self._n = len(self._laps)
        logger.info(
            "EndurancePlotter created — %d laps, total_time=%.1f s",
            self._n,
            endurance_result.total_time,
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

    def _lap_numbers(self) -> np.ndarray:
        """1-based lap number array."""
        return np.arange(1, self._n + 1)

    def _lap_times(self) -> np.ndarray:
        return np.array([lap.lap_time for lap in self._laps])

    def _final_soc_pct(self) -> np.ndarray:
        return np.array([lap.final_soc * 100.0 for lap in self._laps])

    def _final_motor_temp(self) -> np.ndarray:
        return np.array([lap.final_motor_temp for lap in self._laps])

    def _energy_kwh(self) -> np.ndarray:
        return np.array([lap.energy_consumed / 1_000.0 for lap in self._laps])

    def _derating_mask(self) -> np.ndarray:
        return np.array([lap.thermal_derating_occurred for lap in self._laps])

    # ------------------------------------------------------------------ #
    #  Individual plots                                                   #
    # ------------------------------------------------------------------ #

    def plot_lap_time_progression(
            self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Lap time bar chart with average line and derating highlights.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        laps = self._lap_numbers()
        times = self._lap_times()
        derating = self._derating_mask()

        colours = [
            "#ff7f0e" if d else "#1f77b4"
            for d in derating
        ]
        ax.bar(laps, times, color=colours, width=0.7, zorder=3)
        ax.axhline(self.result.avg_lap_time, color="#d62728", ls="--",
                   lw=1.5, label=f"Avg {self.result.avg_lap_time:.2f} s")

        if np.any(derating):
            import matplotlib.patches as mpatches
            orange_patch = mpatches.Patch(color="#ff7f0e",
                                          label="Thermal derating")
            blue_patch   = mpatches.Patch(color="#1f77b4",
                                          label="Normal lap")
            ax.legend(handles=[blue_patch, orange_patch,
                                plt.Line2D([], [], color="#d62728",
                                           ls="--", label=f"Avg {self.result.avg_lap_time:.2f} s")],
                      fontsize=9)
        else:
            ax.legend(fontsize=9)

        ax.set_xlabel("Lap Number", fontsize=12)
        ax.set_ylabel("Lap Time (s)", fontsize=12)
        ax.set_title("Lap Time Progression", fontsize=13, fontweight="bold")
        ax.set_xlim([0.3, self._n + 0.7])
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_energy_budget(
            self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Energy consumed per lap (kW·h) bar chart.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        laps = self._lap_numbers()
        energy = self._energy_kwh()

        ax.bar(laps, energy, color="#1f77b4", width=0.7, zorder=3)
        ax.axhline(float(np.mean(energy)), color="#ff7f0e", ls="--",
                   lw=1.5, label=f"Avg {np.mean(energy):.3f} kW·h")

        ax.set_xlabel("Lap Number", fontsize=12)
        ax.set_ylabel("Energy (kW·h)", fontsize=12)
        ax.set_title("Energy per Lap", fontsize=13, fontweight="bold")
        ax.set_xlim([0.3, self._n + 0.7])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_thermal_progression(
            self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Motor temperature at end of each lap with derating limit lines.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        laps = self._lap_numbers()
        temps = self._final_motor_temp()

        ax.plot(laps, temps, color="#d62728", lw=2.0, marker="o",
                ms=4, zorder=3)
        ax.axhline(90,  color="#ff7f0e", ls="--", lw=1.2,
                   label="Derate  90 °C")
        ax.axhline(100, color="#d62728",  ls="--", lw=1.2,
                   label="Cont.  100 °C")
        ax.axhline(120, color="#7f0000",  ls="--", lw=1.2,
                   label="Peak   120 °C")

        ax.set_xlabel("Lap Number", fontsize=12)
        ax.set_ylabel("Motor Temperature (°C)", fontsize=12)
        ax.set_title("Motor Thermal Progression", fontsize=13, fontweight="bold")
        ax.set_xlim([0.5, self._n + 0.5])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_lap_comparison(
            self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Overlay speed traces from first, middle, and last lap.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        idx_first  = 0
        idx_mid    = max(0, (self._n - 1) // 2)
        idx_last   = max(0, self._n - 1)

        configs = [
            (idx_first, "#1f77b4", "Lap 1 (first)"),
            (idx_mid,   "#ff7f0e", f"Lap {idx_mid + 1} (middle)"),
            (idx_last,  "#d62728", f"Lap {idx_last + 1} (last)"),
        ]

        plotted: set = set()
        for idx, colour, label in configs:
            if idx in plotted:
                continue
            plotted.add(idx)
            lap: LapResult = self._laps[idx]
            v_kmh = lap.speed_profile * 3.6
            ax.plot(lap.distance, v_kmh, color=colour, lw=1.6,
                    alpha=0.85, label=label)

        ax.set_xlabel("Distance (m)", fontsize=12)
        ax.set_ylabel("Speed (km/h)", fontsize=12)
        ax.set_title("Lap Speed Comparison (First / Middle / Last)",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, save_path)
        return fig

    # ------------------------------------------------------------------ #
    #  Comprehensive 2×2 figure                                          #
    # ------------------------------------------------------------------ #

    def plot_comprehensive_endurance(
            self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """2×2 figure summarising the full endurance run.

        Panels
        ------
        1. Lap time bar chart — derating laps highlighted orange
        2. SOC (%) vs lap — red dashed 20 % "Critical" threshold
        3. Motor temperature (°C) vs lap — derating limit lines
        4. Energy per lap (kW·h) bar chart

        Args:
            save_path: Optional path; saved at 300 dpi.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        r = self.result
        sup_title = (
            f"Endurance — {self._n} laps  |  "
            f"{r.total_time / 60:.1f} min  |  "
            f"Final SOC: {r.final_soc * 100:.1f} %"
        )

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(sup_title, fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

        laps    = self._lap_numbers()
        times   = self._lap_times()
        soc     = self._final_soc_pct()
        temps   = self._final_motor_temp()
        energy  = self._energy_kwh()
        derate  = self._derating_mask()

        # ── 1. Lap time bar chart ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        colours = ["#ff7f0e" if d else "#1f77b4" for d in derate]
        ax1.bar(laps, times, color=colours, width=0.7, zorder=3)
        ax1.axhline(r.avg_lap_time, color="#d62728", ls="--", lw=1.5,
                    label=f"Avg {r.avg_lap_time:.2f} s")
        ax1.set_xlabel("Lap Number", fontsize=12)
        ax1.set_ylabel("Lap Time (s)", fontsize=12)
        ax1.set_title("Lap Time Progression", fontsize=13, fontweight="bold")
        ax1.set_xlim([0.3, self._n + 0.7])
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis="y")

        # ── 2. SOC vs lap ──────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(laps, soc, color="#2ca02c", lw=2.0, marker="o", ms=4, zorder=3)
        ax2.axhline(20.0, color="#d62728", ls="--", lw=1.5,
                    label="Critical 20 %")
        ax2.fill_between(laps, 20.0, soc,
                         where=soc >= 20.0, alpha=0.12, color="#2ca02c")
        ax2.set_xlabel("Lap Number", fontsize=12)
        ax2.set_ylabel("SOC (%)", fontsize=12)
        ax2.set_title("State of Charge", fontsize=13, fontweight="bold")
        ax2.set_xlim([0.5, self._n + 0.5])
        ax2.set_ylim([0, 105])
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ── 3. Motor temperature ───────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(laps, temps, color="#d62728", lw=2.0, marker="o",
                 ms=4, zorder=3)
        ax3.axhline(90,  color="#ff7f0e", ls="--", lw=1.2,
                    label="Derate 90 °C")
        ax3.axhline(100, color="#d62728",  ls="--", lw=1.2,
                    label="Cont. 100 °C")
        ax3.axhline(120, color="#7f0000",  ls="--", lw=1.2,
                    label="Peak 120 °C")
        ax3.set_xlabel("Lap Number", fontsize=12)
        ax3.set_ylabel("Motor Temperature (°C)", fontsize=12)
        ax3.set_title("Motor Thermal Progression", fontsize=13, fontweight="bold")
        ax3.set_xlim([0.5, self._n + 0.5])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ── 4. Energy per lap ──────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.bar(laps, energy, color="#1f77b4", width=0.7, zorder=3)
        ax4.axhline(float(np.mean(energy)), color="#ff7f0e", ls="--",
                    lw=1.5, label=f"Avg {np.mean(energy):.3f} kW·h")
        ax4.set_xlabel("Lap Number", fontsize=12)
        ax4.set_ylabel("Energy (kW·h)", fontsize=12)
        ax4.set_title("Energy per Lap", fontsize=13, fontweight="bold")
        ax4.set_xlim([0.3, self._n + 0.7])
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis="y")

        self._save(fig, save_path)
        logger.info("Comprehensive endurance figure generated.")
        return fig
