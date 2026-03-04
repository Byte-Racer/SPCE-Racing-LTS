"""
Quasi-Steady-State (QSS) Lap Time Solver — Main Entry Point.

The QSS method assumes the vehicle is continuously at the limit of its
performance envelope at every track point.  Transient dynamics are
*not* modelled — only position-based integration.

Algorithm (per lap)
-------------------
1. **Corner-speed profile** — :class:`~src.solver.speed_profile.SpeedProfileGenerator`
   computes the maximum speed achievable in each corner from the lateral
   acceleration limit.
2. **Forward pass** — accelerate from every minimum-speed point toward the
   next, limited by motor force.
3. **Backward pass × 2** — brake into every minimum-speed point; doubled to
   handle the closed loop wrap.
4. **Final profile** — element-wise minimum of the two passes.
5. **Lap time** — ``Σ ds / v_avg`` over all segments.
6. **Energy accounting** — per-segment battery power, SOC, voltage sag.
7. **Thermal update** — motor temperature updated via
   ``motor.update_thermal_state()`` each segment.

Multi-lap (endurance)
---------------------
SOC and motor thermal state carry over between laps.  Battery voltage is
re-derived from SOC each lap.  Progress is logged every 10%.

Usage::

    from src.vehicle.vehicle_model import VehicleDynamics
    from src.solver.qss_solver import QSSSolver

    vehicle = VehicleDynamics("config/vehicle_params.yaml")
    solver  = QSSSolver(vehicle, "config/solver_config.yaml")
    result  = solver.solve_autocross(track)
    print(result.summary_string())

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from src.solver.acceleration_zones import AccelerationZones
from src.solver.energy_tracker import EnergyTracker
from src.solver.lap_results import EnduranceResult, LapResult
from src.solver.speed_profile import SpeedProfileGenerator

if TYPE_CHECKING:
    from src.track.track_representation import Track
    from src.vehicle.vehicle_model import VehicleDynamics

logger = logging.getLogger(__name__)

G = 9.81  # m/s²


class QSSSolver:
    """Quasi-Steady-State lap time solver for SPCE Racing FSAE EV.

    Args:
        vehicle:     Fully initialised :class:`~src.vehicle.vehicle_model.VehicleDynamics`.
        config_path: Path to ``config/solver_config.yaml``.
    """

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #

    def __init__(self, vehicle: "VehicleDynamics", config_path: str) -> None:
        self.vehicle = vehicle

        # ── Load solver config ────────────────────────────────────────
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Solver config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        sol = cfg["solver"]
        self.default_mode: str = sol.get("mode", "peak")
        self.n_laps_endurance: int = sol.get("n_laps_endurance", 22)
        self.initial_soc: float = sol.get("initial_soc", 1.0)
        self.speed_floor: float = sol.get("speed_floor", 0.5)
        self.max_iterations: int = sol.get("max_iterations", 5)
        self.ds_resample: float = sol.get("ds_resample", 0.5)
        self.soc_floor: float = sol.get("soc_floor", 0.20)
        self.soc_peak_boost_min: float = sol.get("soc_peak_boost_min", 0.50)

        logger.info(
            "QSSSolver initialised: mode=%s, soc_floor=%.0f%%, ds=%.2f m",
            self.default_mode,
            self.soc_floor * 100,
            self.ds_resample,
        )

    # ------------------------------------------------------------------ #
    #  Public high-level API                                              #
    # ------------------------------------------------------------------ #

    def solve_lap(
        self,
        track: "Track",
        initial_soc: float = 1.0,
        mode: str = "peak",
    ) -> LapResult:
        """Simulate a single lap on *track*.

        Args:
            track:       Discretised :class:`~src.track.track_representation.Track`.
            initial_soc: Starting SOC (0 – 1).
            mode:        ``'peak'`` (97 kW) or ``'continuous'`` (80 kW).

        Returns:
            :class:`~src.solver.lap_results.LapResult` with full telemetry.
        """
        self.vehicle.motor_model.motor_temp = (
            self.vehicle.motor_model.motor_ambient_temp
        )
        return self._run_lap(track, initial_soc=initial_soc, mode=mode)

    def solve_autocross(self, track: "Track") -> LapResult:
        """Single lap, peak power, fresh battery and motor (autocross event).

        Args:
            track: Discretised track (~800 m, mix of straights & corners).

        Returns:
            :class:`~src.solver.lap_results.LapResult`.
        """
        logger.info("Solving autocross lap — peak mode, fresh state")
        return self.solve_lap(track, initial_soc=1.0, mode="peak")

    def solve_endurance(
        self,
        track: "Track",
        n_laps: int = 0,
        initial_soc: float = 1.0,
    ) -> EnduranceResult:
        """Simulate a multi-lap endurance run.

        State carried across laps:
        * Battery SOC (and derived pack voltage)
        * Motor winding temperature (via ``motor_model.motor_temp``)

        Progress is logged every 10 %.

        Args:
            track:       Discretised :class:`~src.track.track_representation.Track`.
            n_laps:      Number of laps; defaults to ``n_laps_endurance`` from config.
            initial_soc: Starting SOC for lap 1.

        Returns:
            :class:`~src.solver.lap_results.EnduranceResult`.
        """
        if n_laps <= 0:
            n_laps = self.n_laps_endurance

        logger.info(
            "Starting endurance simulation: %d laps, SOC=%.0f%%",
            n_laps,
            initial_soc * 100,
        )

        # Reset motor thermal state at race start
        self.vehicle.motor_model.motor_temp = (
            self.vehicle.motor_model.motor_ambient_temp
        )

        laps: list[LapResult] = []
        soc = float(initial_soc)
        log_interval = max(1, n_laps // 10)

        for lap_num in range(1, n_laps + 1):
            if (lap_num - 1) % log_interval == 0:
                logger.info(
                    "  Lap %d / %d  (SOC=%.1f%%, motor=%.1f°C)",
                    lap_num,
                    n_laps,
                    soc * 100,
                    self.vehicle.motor_model.motor_temp,
                )

            result = self._run_lap(
                track,
                initial_soc=soc,
                mode="continuous",
            )
            laps.append(result)
            soc = result.final_soc   # carry SOC forward

            # Motor temp carries forward automatically via motor_model state

            if result.energy_critical:
                logger.warning(
                    "  Lap %d: energy critical (SOC=%.1f%%) — stopping endurance.",
                    lap_num,
                    soc * 100,
                )
                break

        logger.info(
            "Endurance complete: %d laps, total time=%.1f s, final SOC=%.1f%%",
            len(laps),
            sum(l.lap_time for l in laps),
            soc * 100,
        )
        return EnduranceResult(laps=laps)

    def get_speed_profile(self, track: "Track") -> np.ndarray:
        """Return the achievable speed profile without full energy solve.

        Useful for quick track visualisation.

        Args:
            track: Discretised :class:`~src.track.track_representation.Track`.

        Returns:
            1-D array of speed in **m/s**, shape ``(N,)``.
        """
        v_corner = SpeedProfileGenerator(
            self.vehicle,
            top_speed=self.vehicle.top_speed,
            speed_floor=self.speed_floor,
            max_iterations=self.max_iterations,
        ).generate(track)

        zones = AccelerationZones(
            self.vehicle,
            speed_floor=self.speed_floor,
            mode=self.default_mode,
        )
        v_final, _, _ = zones.integrate(track, v_corner)
        return v_final

    # ------------------------------------------------------------------ #
    #  Core private solver                                                #
    # ------------------------------------------------------------------ #

    def _run_lap(
        self,
        track: "Track",
        initial_soc: float,
        mode: str,
    ) -> LapResult:
        """Run one complete lap and return a populated :class:`LapResult`.

        Steps
        -----
        1. Build corner-speed profile.
        2. Run forward + backward integration.
        3. Walk segments: accumulate dt, battery power, energy, SOC, motor temp.
        4. Assemble and return :class:`LapResult`.

        Args:
            track:       Discretised track.
            initial_soc: Starting SOC for this lap.
            mode:        Power mode.

        Returns:
            Populated :class:`LapResult`.
        """
        # ── 1. Corner speed profile ───────────────────────────────────
        v_corner = SpeedProfileGenerator(
            self.vehicle,
            top_speed=self.vehicle.top_speed,
            speed_floor=self.speed_floor,
            max_iterations=self.max_iterations,
        ).generate(track)

        # ── 2. Forward + backward integration ────────────────────────
        zones = AccelerationZones(
            self.vehicle,
            speed_floor=self.speed_floor,
            mode=mode,
        )
        v_final, ax_profile, limiting_factors = zones.integrate(track, v_corner)

        # ── 3. Segment-level time, energy, thermal walk ───────────────
        energy_tracker = EnergyTracker(
            battery_energy_total_wh=self.vehicle.battery_continuous_power
            * 1e-3
            / (80.0 / 8750.0),   # derive from params
            battery_resistance_ohm=self.vehicle.battery_resistance,
            soc_floor=self.soc_floor,
            peak_power_w=self.vehicle.battery_peak_power,
            continuous_power_w=self.vehicle.battery_continuous_power,
            soc_peak_boost_min=self.soc_peak_boost_min,
            initial_soc=initial_soc,
        )
        # Directly set the correct total energy from the vehicle object
        # battery_energy_total_wh = 8750 Wh (from config: 8.75 kWh)
        energy_tracker.battery_energy_total_wh = self._battery_total_energy_wh()

        ds = np.diff(track.distance)  # segment lengths [N-1]
        n_seg = len(ds)

        lap_time = 0.0
        energy_consumed_wh = 0.0
        thermal_derating_occurred = False

        # Per-point telemetry (length N — track points)
        soc_arr = np.zeros(len(v_final))
        motor_temp_arr = np.zeros(len(v_final))

        soc_arr[0] = energy_tracker.soc
        motor_temp_arr[0] = self.vehicle.motor_model.motor_temp

        for i in range(n_seg):
            vi = float(v_final[i])
            vi1 = float(v_final[i + 1])
            v_avg = max((vi + vi1) / 2.0, self.speed_floor)
            seg_len = float(ds[i])
            dt = seg_len / v_avg

            lap_time += dt

            # Battery power for this segment
            ax = float(ax_profile[i])
            bat_power = self._segment_battery_power(v_avg, ax)

            # Energy tracker step
            step = energy_tracker.step(bat_power, dt, mode=mode)
            energy_consumed_wh += step["energy_segment_wh"]

            # Motor thermal update
            motor_state = self.vehicle.motor_model.get_available_torque(
                throttle_percent=100.0 if ax >= 0 else 0.0,
                rpm=self._speed_to_rpm(v_avg),
                voltage_dc=step["v_oc"],
            )

            # Compute motor-only heat dissipation.
            # bat_power includes losses from both the motor and the drivetrain.
            # Motor electrical input = bat_power (battery → motor, neglecting
            # battery I²R for thermal purposes).
            # Motor mechanical output = bat_power * motor_efficiency.
            # Motor heat = bat_power * (1 - motor_efficiency).
            # BUT the drivetrain loss (7 %) is downstream of the motor and does
            # NOT heat the motor windings.  The correct motor heat is:
            #   P_mech_at_wheels = bat_power * eta_motor * eta_drivetrain
            #   P_motor_shaft    = bat_power * eta_motor
            #   P_motor_heat     = bat_power - P_motor_shaft
            #                    = bat_power * (1 - eta_motor)
            # However, bat_power was derived as P_mech_wheels / (eta_motor *
            # eta_dt), so P_motor_shaft = P_mech_wheels / eta_dt.
            # The motor electrical input is P_motor_shaft / eta_motor
            #   = P_mech_wheels / (eta_dt * eta_motor) = bat_power.
            # So motor heat = bat_power - bat_power * eta_motor
            #               = bat_power * (1 - eta_motor).
            # This is correct, BUT bat_power is now clamped to battery peak.
            # The original bug was that bat_power was unclamped (up to 368 kW).
            eta_motor = motor_state["efficiency"]
            power_loss_w = bat_power * (1.0 - eta_motor)

            self.vehicle.motor_model.update_thermal_state(
                power_loss=max(power_loss_w, 0.0),
                dt=dt,
                airflow_speed=v_avg,
            )

            if motor_state["thermal_factor"] < 0.99:
                thermal_derating_occurred = True

            soc_arr[i + 1] = step["soc"]
            motor_temp_arr[i + 1] = self.vehicle.motor_model.motor_temp

        # ── 4. Assemble LapResult ─────────────────────────────────────
        return LapResult(
            lap_time=lap_time,
            avg_speed=float(np.mean(v_final)),
            max_speed=float(np.max(v_final)),
            min_speed=float(np.min(v_final)),
            energy_consumed=energy_consumed_wh,
            net_energy=energy_consumed_wh,   # no regen
            final_soc=energy_tracker.soc,
            final_motor_temp=float(self.vehicle.motor_model.motor_temp),
            thermal_derating_occurred=thermal_derating_occurred,
            energy_critical=energy_tracker.energy_critical,
            speed_profile=v_final,
            distance=track.distance,
            ax_profile=ax_profile,
            limiting_factor_profile=limiting_factors,
            soc_profile=soc_arr,
            motor_temp_profile=motor_temp_arr,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _segment_battery_power(self, v_avg: float, ax: float) -> float:
        """Estimate battery power demand for a speed/acceleration segment.

        During acceleration: derives traction force, applies efficiency chain.
        During braking / cruise: only drag + rolling resistance (no battery draw
        from mechanical braking since regen is disabled).

        Args:
            v_avg: Average segment speed in **m/s**.
            ax:    Segment longitudinal acceleration in **m/s²**.

        Returns:
            Battery power demanded in **W** (≥ 0).
        """
        aero = self.vehicle.calculate_aero_forces(v_avg)
        drag = aero["drag"]
        f_rr = self.vehicle.rolling_resistance_coeff * self.vehicle.total_weight

        if ax >= 0:
            # Traction required
            f_traction = self.vehicle.mass * ax + drag + f_rr
            f_traction = max(f_traction, 0.0)
            p_mech = f_traction * v_avg
            p_bat = p_mech / (
                self.vehicle.motor_efficiency * self.vehicle.drivetrain_efficiency
            )
            # Clamp to battery pack power limit — the pack physically cannot
            # deliver more than its peak rating.
            p_bat = min(p_bat, self.vehicle.battery_peak_power)
        else:
            # Braking — only drag + roll assistance (no battery draw)
            # Net force from drag/rr helps braking; we model zero battery draw.
            p_bat = 0.0

        return float(p_bat)

    def _speed_to_rpm(self, speed: float) -> float:
        """Convert vehicle speed (m/s) to motor RPM."""
        wheel_rpm = (speed / self.vehicle.wheel_radius) * (60.0 / (2.0 * np.pi))
        return float(min(wheel_rpm * self.vehicle.gear_ratio, self.vehicle.max_rpm))

    def _battery_total_energy_wh(self) -> float:
        """Return total battery energy in Wh from vehicle config."""
        # 8.75 kWh total. Derive from continuous power and known pack spec.
        # vehicle_params.yaml: energy.total = 8.75 kWh
        # We read it from a known derived constant. Battery pack total is
        # capacity_ah * voltage_nom / 1000 = 18 * 486 / 1000 ~ 8748 Wh
        try:
            return float(self.vehicle.battery_continuous_power / 1000.0 / 80.0 * 8750.0)
        except (AttributeError, ZeroDivisionError):
            return 8750.0   # fallback: 8.75 kWh in Wh
