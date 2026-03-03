"""
Energy Tracker for QSS Lap Time Solver.

Tracks State-of-Charge (SOC), open-circuit voltage, voltage sag under load,
and cumulative energy budgets on a per-segment basis throughout a simulated
lap or endurance run.

No regenerative braking — all braking energy is dissipated as heat.

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SOC → cell-voltage look-up table (from vehicle_params.yaml)
# Multiply by 135 for pack voltage.
# ---------------------------------------------------------------------------
_SOC_POINTS_FRAC = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                               0.6, 0.7, 0.8, 0.9, 1.0])
_CELL_VOLTAGE_POINTS = np.array([2.50, 3.10, 3.35, 3.45, 3.50, 3.55,
                                  3.60, 3.65, 3.75, 3.90, 4.15])

_SOC_TO_CELL_VOLTAGE = interp1d(
    _SOC_POINTS_FRAC,
    _CELL_VOLTAGE_POINTS,
    kind="linear",
    bounds_error=False,
    fill_value=(_CELL_VOLTAGE_POINTS[0], _CELL_VOLTAGE_POINTS[-1]),
)

# Battery pack series count (135 cells in series)
_SERIES_COUNT = 135


class EnergyTracker:
    """Per-segment energy accounting for the QSS solver.

    Tracks:
    * State-of-Charge (SOC) from 0 to 1
    * Open-circuit pack voltage (from SOC–voltage curve)
    * Terminal voltage under load (voltage sag)
    * Cumulative energy consumed from the battery (Wh)
    * ``energy_critical`` flag (SOC floor reached)

    All energy inputs and outputs are in **Wh**; power in **Watts**;
    time in **seconds**.

    Args:
        battery_energy_total_wh: Total battery energy in **Wh** (e.g. 8750).
        battery_resistance_ohm:  Pack internal resistance in **Ω** (e.g. 0.506).
        soc_floor:               SOC fraction below which ``energy_critical``
                                 is set and power is limited (default 0.20).
        peak_power_w:            Peak battery discharge power in **W** (97 000).
        continuous_power_w:      Continuous discharge power in **W** (80 000).
        soc_peak_boost_min:      Minimum SOC to allow peak power (default 0.50).
        initial_soc:             Starting SOC fraction (default 1.0).
    """

    def __init__(
        self,
        battery_energy_total_wh: float,
        battery_resistance_ohm: float,
        soc_floor: float = 0.20,
        peak_power_w: float = 97_000.0,
        continuous_power_w: float = 80_000.0,
        soc_peak_boost_min: float = 0.50,
        initial_soc: float = 1.0,
    ) -> None:
        self.battery_energy_total_wh = battery_energy_total_wh
        self.battery_resistance_ohm = battery_resistance_ohm
        self.soc_floor = soc_floor
        self.peak_power_w = peak_power_w
        self.continuous_power_w = continuous_power_w
        self.soc_peak_boost_min = soc_peak_boost_min

        # ── State variables ──────────────────────────────────────────────
        self.soc: float = float(np.clip(initial_soc, 0.0, 1.0))
        self.energy_consumed_wh: float = 0.0   # cumulative draw from battery
        self.energy_critical: bool = False

        # History arrays (filled by step())
        self._soc_history: List[float] = [self.soc]
        self._voltage_history: List[float] = [self.open_circuit_voltage]

        logger.debug(
            "EnergyTracker init: SOC=%.1f%%, V_oc=%.1f V",
            self.soc * 100,
            self.open_circuit_voltage,
        )

    # ------------------------------------------------------------------ #
    #  Read-only computed properties                                      #
    # ------------------------------------------------------------------ #

    @property
    def open_circuit_voltage(self) -> float:
        """Pack open-circuit voltage in **V** at current SOC."""
        cell_v = float(_SOC_TO_CELL_VOLTAGE(self.soc))
        return cell_v * _SERIES_COUNT

    def terminal_voltage(self, battery_power_w: float) -> float:
        """Pack terminal voltage in **V** under a given power draw.

        Uses Thevenin equivalent: ``V_t = V_oc − I × R_pack``
        where ``I = P / V_oc``.

        Args:
            battery_power_w: Instantaneous battery power in **W**
                             (positive = discharge).

        Returns:
            Terminal voltage in **V** (clamped to minimum safe value 337.5 V).
        """
        v_oc = self.open_circuit_voltage
        if v_oc <= 0 or battery_power_w <= 0:
            return v_oc
        current = battery_power_w / v_oc        # A
        v_terminal = v_oc - current * self.battery_resistance_ohm
        return float(max(v_terminal, 337.5))    # min pack voltage

    # ------------------------------------------------------------------ #
    #  Power limit enforcement                                            #
    # ------------------------------------------------------------------ #

    def max_power_allowed(self, mode: str = "peak") -> float:
        """Return the currently-allowed battery discharge power in **W**.

        If SOC is below ``soc_peak_boost_min`` or ``energy_critical`` is set,
        returns ``continuous_power_w`` regardless of *mode*.

        Args:
            mode: ``'peak'`` or ``'continuous'``.

        Returns:
            Maximum allowable power draw from battery in **W**.
        """
        if self.energy_critical or self.soc < self.soc_peak_boost_min:
            return self.continuous_power_w
        if mode == "peak":
            return self.peak_power_w
        return self.continuous_power_w

    # ------------------------------------------------------------------ #
    #  Segment step                                                       #
    # ------------------------------------------------------------------ #

    def step(
        self,
        battery_power_w: float,
        dt: float,
        mode: str = "peak",
    ) -> Dict[str, float]:
        """Advance energy state by one time segment.

        Args:
            battery_power_w: Gross battery power demand in **W**
                             (before power-limit clamping; positive = discharge).
            dt:              Segment duration in **seconds**.
            mode:            ``'peak'`` or ``'continuous'`` — sets power cap.

        Returns:
            Dictionary with keys:

            * ``battery_power_clamped`` — actual power drawn (W, after limits)
            * ``energy_segment_wh``     — energy this segment (Wh)
            * ``soc``                   — SOC after this segment (fraction)
            * ``v_oc``                  — open-circuit voltage (V)
            * ``v_terminal``            — terminal voltage under load (V)
            * ``energy_critical``       — bool flag
        """
        # Clamp power to current limit
        power_limit = self.max_power_allowed(mode)
        actual_power = float(np.clip(battery_power_w, 0.0, power_limit))

        # Energy this segment (Wh)
        dE_wh = actual_power * dt / 3600.0

        # Accumulate and update SOC
        self.energy_consumed_wh += dE_wh
        soc_delta = dE_wh / self.battery_energy_total_wh
        self.soc = float(np.clip(self.soc - soc_delta, 0.0, 1.0))

        # Check floor
        if self.soc <= self.soc_floor and not self.energy_critical:
            self.energy_critical = True
            logger.warning(
                "⚠ Energy critical — SOC reached %.1f%% (floor %.1f%%)",
                self.soc * 100,
                self.soc_floor * 100,
            )

        v_oc = self.open_circuit_voltage
        v_t = self.terminal_voltage(actual_power)

        # Store history
        self._soc_history.append(self.soc)
        self._voltage_history.append(v_oc)

        return {
            "battery_power_clamped": actual_power,
            "energy_segment_wh": dE_wh,
            "soc": self.soc,
            "v_oc": v_oc,
            "v_terminal": v_t,
            "energy_critical": self.energy_critical,
        }

    # ------------------------------------------------------------------ #
    #  State persistence (for multi-lap)                                  #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict[str, float]:
        """Return a dict of current state for cross-lap persistence."""
        return {
            "soc": self.soc,
            "energy_consumed_wh": self.energy_consumed_wh,
            "energy_critical": float(self.energy_critical),
        }

    def restore(self, snap: Dict[str, float]) -> None:
        """Restore state from a snapshot dict."""
        self.soc = float(snap["soc"])
        self.energy_consumed_wh = float(snap["energy_consumed_wh"])
        self.energy_critical = bool(snap["energy_critical"])
        self._soc_history = [self.soc]
        self._voltage_history = [self.open_circuit_voltage]

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #

    def reset(self, initial_soc: float = 1.0) -> None:
        """Hard-reset tracker to a fresh state."""
        self.soc = float(np.clip(initial_soc, 0.0, 1.0))
        self.energy_consumed_wh = 0.0
        self.energy_critical = False
        self._soc_history = [self.soc]
        self._voltage_history = [self.open_circuit_voltage]
