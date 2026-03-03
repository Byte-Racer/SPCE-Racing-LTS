"""
Lap and Endurance Result Containers for the QSS Solver.

``LapResult`` stores all per-lap telemetry arrays and scalar summaries.
``EnduranceResult`` aggregates multiple ``LapResult`` objects and computes
cross-lap statistics (best/worst lap, degradation from thermal effects).

Both classes support:
* ``to_dataframe()`` — export to :class:`pandas.DataFrame` for CSV.
* ``summary_string()`` — human-readable printed summary.
* ``LapResult.plot_telemetry()`` — 4-subplot matplotlib figure.

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LapResult:
    """Complete telemetry and summary for a single simulated lap.

    All array attributes are of the same length as ``track.distance`` (``N``),
    except ``ax_profile`` and ``limiting_factor_profile`` which have length
    ``N-1``.

    Scalar summaries
    ----------------
    lap_time             : float — total lap time in **s**
    avg_speed            : float — mean speed in **m/s**
    max_speed            : float — peak speed in **m/s**
    min_speed            : float — minimum speed in **m/s**
    energy_consumed      : float — gross energy from battery in **Wh**
    net_energy           : float — same (no regen recovery) in **Wh**
    final_soc            : float — SOC at lap end (0–1)
    final_motor_temp     : float — motor winding temperature at lap end in **°C**
    thermal_derating_occurred : bool
    energy_critical      : bool — SOC floor was breached

    Array telemetry (all in SI units)
    -----------------------------------
    speed_profile          : (N,)   m/s
    distance               : (N,)   m
    ax_profile             : (N-1,) m/s²
    limiting_factor_profile: (N-1,) list[str]
    soc_profile            : (N,)   fraction 0–1
    motor_temp_profile     : (N,)   °C
    """

    # ── Scalar summaries ─────────────────────────────────────────────────
    lap_time: float = 0.0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    min_speed: float = 0.0
    energy_consumed: float = 0.0   # Wh
    net_energy: float = 0.0        # Wh (same as consumed — no regen)
    final_soc: float = 1.0
    final_motor_temp: float = 25.0
    thermal_derating_occurred: bool = False
    energy_critical: bool = False

    # ── Array telemetry ──────────────────────────────────────────────────
    speed_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    distance: np.ndarray = field(default_factory=lambda: np.array([]))
    ax_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    limiting_factor_profile: List[str] = field(default_factory=list)
    soc_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    motor_temp_profile: np.ndarray = field(default_factory=lambda: np.array([]))

    # ------------------------------------------------------------------ #
    #  DataFrame export                                                   #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Export telemetry arrays to a :class:`pandas.DataFrame`.

        The DataFrame has one row per track point.  Segment-length arrays
        (``ax_profile``, ``limiting_factor_profile``) are right-padded with
        ``NaN`` / ``''`` so they align with the point arrays.

        Returns:
            DataFrame with columns: ``distance``, ``speed``, ``ax``,
            ``limiting_factor``, ``soc``, ``motor_temp``.
        """
        n = len(self.distance)
        ax_padded = np.append(self.ax_profile, np.nan) if len(self.ax_profile) else np.full(n, np.nan)
        lf_padded = self.limiting_factor_profile + [""] if len(self.limiting_factor_profile) else [""] * n

        return pd.DataFrame({
            "distance_m": self.distance,
            "speed_ms": self.speed_profile,
            "speed_kmh": self.speed_profile * 3.6,
            "ax_ms2": ax_padded[:n],
            "ax_g": ax_padded[:n] / 9.81,
            "limiting_factor": lf_padded[:n],
            "soc_pct": self.soc_profile * 100.0,
            "motor_temp_c": self.motor_temp_profile,
        })

    # ------------------------------------------------------------------ #
    #  Human-readable summary                                             #
    # ------------------------------------------------------------------ #

    def summary_string(self) -> str:
        """Return a formatted text summary of the lap.

        Returns:
            Multi-line string ready to ``print()``.
        """
        sep = "=" * 56
        lines = [
            sep,
            "  QSS LAP RESULT — SPCE Racing",
            sep,
            f"  Lap time          : {self.lap_time:>8.3f} s",
            f"  Avg speed         : {self.avg_speed:>8.2f} m/s  "
            f"({self.avg_speed * 3.6:.1f} km/h)",
            f"  Max speed         : {self.max_speed:>8.2f} m/s  "
            f"({self.max_speed * 3.6:.1f} km/h)",
            f"  Min speed         : {self.min_speed:>8.2f} m/s  "
            f"({self.min_speed * 3.6:.1f} km/h)",
            sep,
            f"  Energy consumed   : {self.energy_consumed:>8.1f} Wh",
            f"  Final SOC         : {self.final_soc * 100:>8.1f} %",
            f"  Energy critical   : {str(self.energy_critical):>8}",
            sep,
            f"  Final motor temp  : {self.final_motor_temp:>8.1f} °C",
            f"  Thermal derating  : {str(self.thermal_derating_occurred):>8}",
            sep,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Telemetry plot                                                     #
    # ------------------------------------------------------------------ #

    def plot_telemetry(self, save_path: Optional[str] = None):
        """Generate a 4-panel telemetry figure.

        Panels
        ------
        1. Speed vs distance
        2. Longitudinal acceleration vs distance
        3. Motor temperature vs distance
        4. SOC vs distance

        Args:
            save_path: If given, save the figure to this file path.

        Returns:
            ``matplotlib.figure.Figure``
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("QSS Lap Telemetry — SPCE Racing", fontsize=15, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

        dist_km = self.distance / 1000.0
        speed_kmh = self.speed_profile * 3.6

        # ── Panel 1: Speed ────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(dist_km, speed_kmh, color="#1f77b4", lw=1.5)
        ax1.fill_between(dist_km, 0, speed_kmh, alpha=0.15, color="#1f77b4")
        ax1.set_xlabel("Distance (km)")
        ax1.set_ylabel("Speed (km/h)")
        ax1.set_title("Speed Profile")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, dist_km[-1]])

        # ── Panel 2: Longitudinal Acceleration ───────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.ax_profile):
            # Align segment values to mid-segment distances
            dist_seg = (self.distance[:-1] + self.distance[1:]) / 2.0
            ax_g = self.ax_profile / 9.81
            ax2.plot(dist_seg / 1000.0, ax_g, color="#d62728", lw=1.2)
            ax2.axhline(0, color="k", ls="--", lw=0.6)
            ax2.fill_between(dist_seg / 1000.0, 0, ax_g,
                             where=ax_g >= 0, alpha=0.2, color="green",
                             label="Accel")
            ax2.fill_between(dist_seg / 1000.0, 0, ax_g,
                             where=ax_g < 0, alpha=0.2, color="red",
                             label="Brake")
        ax2.set_xlabel("Distance (km)")
        ax2.set_ylabel("Longitudinal Accel (g)")
        ax2.set_title("Acceleration Profile")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        # ── Panel 3: Motor Temperature ────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        if len(self.motor_temp_profile):
            ax3.plot(dist_km, self.motor_temp_profile, color="#ff7f0e", lw=1.5)
            ax3.axhline(90, color="orange", ls="--", lw=1, label="Derate start 90°C")
            ax3.axhline(100, color="red", ls="--", lw=1, label="Continuous limit 100°C")
            ax3.axhline(120, color="darkred", ls="--", lw=1, label="Peak limit 120°C")
            ax3.legend(fontsize=8)
        ax3.set_xlabel("Distance (km)")
        ax3.set_ylabel("Motor Temp (°C)")
        ax3.set_title("Motor Temperature")
        ax3.grid(True, alpha=0.3)

        # ── Panel 4: SOC ──────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        if len(self.soc_profile):
            soc_pct = self.soc_profile * 100.0
            ax4.plot(dist_km, soc_pct, color="#2ca02c", lw=1.5)
            ax4.fill_between(dist_km, 20.0, soc_pct,
                             where=soc_pct >= 20.0, alpha=0.15, color="green")
            ax4.axhline(20, color="red", ls="--", lw=1, label="SOC floor 20%")
            ax4.legend(fontsize=9)
        ax4.set_xlabel("Distance (km)")
        ax4.set_ylabel("State of Charge (%)")
        ax4.set_title("Battery SOC")
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 105])

        if save_path:
            import pathlib
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            logger.info("Telemetry figure saved → %s", save_path)

        return fig


@dataclass
class EnduranceResult:
    """Aggregated results for a multi-lap endurance simulation.

    Args:
        laps: Ordered list of ``LapResult`` objects (one per lap).

    Attributes (auto-computed from *laps*)
    ----------------------------------------
    total_time       : float — sum of all lap times (s)
    total_energy     : float — sum of all lap energy consumed (Wh)
    final_soc        : float — SOC at end of the last lap (0–1)
    avg_lap_time     : float — mean lap time (s)
    best_lap_time    : float — fastest lap (s)
    worst_lap_time   : float — slowest lap (s)  [thermal derating visible]
    lap_time_degradation: float — worst − best (s)
    """

    laps: List[LapResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.laps:
            self._compute_stats()

    def _compute_stats(self) -> None:
        times = [lap.lap_time for lap in self.laps]
        self.total_time: float = float(np.sum(times))
        self.total_energy: float = float(np.sum(lap.energy_consumed for lap in self.laps))
        self.final_soc: float = self.laps[-1].final_soc
        self.avg_lap_time: float = float(np.mean(times))
        self.best_lap_time: float = float(np.min(times))
        self.worst_lap_time: float = float(np.max(times))
        self.lap_time_degradation: float = self.worst_lap_time - self.best_lap_time

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #

    def summary_string(self) -> str:
        """Return a formatted endurance summary."""
        if not self.laps:
            return "EnduranceResult: no laps simulated."
        sep = "=" * 56
        lines = [
            sep,
            "  QSS ENDURANCE RESULT — SPCE Racing",
            sep,
            f"  Laps simulated    : {len(self.laps):>8}",
            f"  Total time        : {self.total_time:>8.1f} s  "
            f"({self.total_time / 60:.1f} min)",
            f"  Avg lap time      : {self.avg_lap_time:>8.3f} s",
            f"  Best lap          : {self.best_lap_time:>8.3f} s  "
            f"(Lap {self._best_lap_index() + 1})",
            f"  Worst lap         : {self.worst_lap_time:>8.3f} s  "
            f"(Lap {self._worst_lap_index() + 1})",
            f"  Degradation       : {self.lap_time_degradation:>8.3f} s  "
            "(thermal effect)",
            sep,
            f"  Total energy      : {self.total_energy:>8.1f} Wh",
            f"  Final SOC         : {self.final_soc * 100:>8.1f} %",
            sep,
        ]
        return "\n".join(lines)

    def _best_lap_index(self) -> int:
        return int(np.argmin([lap.lap_time for lap in self.laps]))

    def _worst_lap_index(self) -> int:
        return int(np.argmax([lap.lap_time for lap in self.laps]))
