"""
Vehicle Dynamics Model.

This module is the physics engine of the lap time simulator. It calculates all forces, accelerations, and performance limits that determine how fast the vehicle can go at any point on any track.

Physics models included:
  - Motor torque and power delivery (EMRAX 228 MV LC)
  - Battery power and energy limits (Molicel P45B 135s4p)
  - Tire grip — simplified Pacejka "Magic Formula"
  - Aerodynamic forces — downforce and drag
  - Weight transfer — longitudinal (accel/brake) and lateral (cornering)
  - Combined tire loading — friction circle constraint
  - Mechanical braking (all four wheels, tire-limited)

Author: Arceus
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from scipy.interpolate import interp1d

from src.vehicle.motor_inverter_model import MotorInverterModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G = 9.81  # m/s² — gravitational acceleration


class VehicleDynamics:
    """Complete vehicle physics model for an FSAE Electric Vehicle.

    All parameters are loaded once from a YAML configuration file and stored
    as instance attributes for fast repeated access during simulation.

    Typical usage::

        vehicle = VehicleDynamics("config/vehicle_params.yaml")
        ax = vehicle.max_longitudinal_acceleration(speed=15.0)
        ay = vehicle.max_lateral_acceleration(speed=15.0)
    """

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #

    def __init__(self, config_path: str) -> None:
        """Load vehicle parameters from YAML and pre-compute derived values.

        Args:
            config_path: Path to ``vehicle_params.yaml``.

        Raises:
            FileNotFoundError: If *config_path* does not exist.
            KeyError: If a required YAML key is missing.
            AssertionError: If a parameter fails validation.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        # ── Mass properties ──────────────────────────────────────────────
        mass_cfg = config["mass"]
        self.mass: float = mass_cfg["total"]  # kg
        self.weight_dist_front: float = mass_cfg["weight_distribution"]["front"]
        self.weight_dist_rear: float = mass_cfg["weight_distribution"]["rear"]
        self.cg_height: float = mass_cfg["center_of_gravity_height"]  # m
        self.inertia_yaw: float = mass_cfg["moment_of_inertia_yaw"]  # kg·m²

        # ── Chassis geometry ─────────────────────────────────────────────
        geom = config["geometry"]
        self.wheelbase: float = geom["wheelbase"]  # m
        self.track_width_front: float = geom["track_width"]["front"]  # m
        self.track_width_rear: float = geom["track_width"]["rear"]  # m

        # ── Derived static loads ─────────────────────────────────────────
        self.total_weight: float = self.mass * G  # N
        self.weight_front: float = self.total_weight * self.weight_dist_front  # N
        self.weight_rear: float = self.total_weight * self.weight_dist_rear  # N

        # ── Motor specifications ─────────────────────────────────────────
        motor = config["motor"]
        self.motor_count: int = motor["count"]
        self.peak_torque: float = motor["peak_torque"]  # Nm
        self.continuous_torque: float = motor["continuous_torque"]  # Nm
        self.peak_power: float = motor["peak_power"] * 1000  # YAML is kW → W
        self.continuous_power: float = motor["continuous_power"] * 1000  # kW → W
        self.max_rpm: float = motor["max_rpm"]
        self.motor_efficiency: float = motor["peak_efficiency"]  # 0–1
        self.kt_constant: float = motor["kt_constant"]  # Nm/A
        self.peak_current: float = motor["peak_current"]  # A

        # ── Drivetrain ───────────────────────────────────────────────────
        dt = config["drivetrain"]
        self.gear_ratio: float = dt["gear_ratio"]  # motor:wheel
        self.drivetrain_efficiency: float = dt["final_drive_efficiency"]
        self.wheel_radius: float = dt["wheel_radius"]  # m

        # Pre-compute peak wheel torque (used frequently)
        self.wheel_torque_peak: float = (
                self.peak_torque
                * self.motor_count
                * self.gear_ratio
                * self.drivetrain_efficiency
        )  # Nm at wheel

        # ── Battery pack ─────────────────────────────────────────────────
        bat = config["battery_pack"]
        self.battery_voltage_nom: float = bat["voltage"]["nominal"]  # V
        self.battery_continuous_current: float = bat["current_limits"]["continuous"]  # A
        self.battery_peak_current: float = bat["current_limits"]["peak"]  # A
        self.battery_continuous_power: float = bat["power_limits"]["continuous"] * 1000  # kW → W
        self.battery_peak_power: float = bat["power_limits"]["peak"] * 1000  # kW → W
        self.battery_resistance: float = bat["resistance"]["pack_total"]  # Ω

        # ── Tires ────────────────────────────────────────────────────────
        tires = config["tires"]
        self.tire_peak_mu: float = tires["friction"]["peak_mu"]
        self.tire_braking_mu: float = tires["friction"].get("braking_mu", self.tire_peak_mu)
        self.tire_nominal_mu: float = tires["friction"]["nominal_mu"]
        self.tire_load_sensitivity: float = tires["friction"]["load_sensitivity"]
        self.pacejka_B: float = tires["pacejka"]["B"]  # stiffness
        self.pacejka_C: float = tires["pacejka"]["C"]  # shape
        self.rolling_resistance_coeff: float = tires["rolling_resistance"]

        # ── Aerodynamics ─────────────────────────────────────────────────
        aero = config["aero"]
        self.frontal_area: float = aero["frontal_area"]  # m²
        self.drag_coeff: float = aero["drag_coefficient"]  # Cd
        self.lift_coeff_front: float = aero["lift_coefficients"]["front"]  # negative
        self.lift_coeff_rear: float = aero["lift_coefficients"]["rear"]  # negative
        self.air_density: float = aero["air_density"]  # kg/m³
        self.aero_reference_area: float = aero["reference_area"]  # m²

        # Regenerative braking has been removed from this vehicle model.

        # ── Performance limits ───────────────────────────────────────────
        limits = config["limits"]
        self.top_speed: float = limits["top_speed"] / 3.6  # km/h → m/s

        # ── Motor torque curve (interpolation objects) ───────────────────
        self._peak_torque_curve: Optional[interp1d] = None
        self._continuous_torque_curve: Optional[interp1d] = None
        self._build_torque_curves(config["motor_torque_curve"])

        # ── Electro-thermal model ────────────────────────────────────────
        self.motor_model = MotorInverterModel(config["motor"], config["thermal"])

        # ── Validate everything ──────────────────────────────────────────
        self._validate_parameters()

        logger.info(
            "Vehicle model initialised: %.0f kg, %.0f kW peak, %.0f kW continuous",
            self.mass,
            self.peak_power / 1000,
            self.battery_continuous_power / 1000,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_torque_curves(self, curve_cfg: dict) -> None:
        """Create ``interp1d`` objects for motor torque vs RPM.

        Electric motors have two operating regions:

        1. **Constant-torque** (0 → base RPM): current-limited.
        2. **Constant-power** (base RPM → max RPM): voltage-limited,
           torque falls as *T = P × 60 / (2π × RPM)*.

        We sample both regions and build a linear interpolation for fast
        look-up during simulation.
        """
        rpm_pts: list[float] = []
        peak_pts: list[float] = []
        cont_pts: list[float] = []

        for region in curve_cfg["regions"]:
            rpm_lo, rpm_hi = region["rpm_range"]

            if region["torque_type"] == "constant":
                rpm_pts.extend([float(rpm_lo), float(rpm_hi)])
                peak_pts.extend([region["peak_torque"]] * 2)
                cont_pts.extend([region["continuous_torque"]] * 2)

            elif region["torque_type"] == "power_limited":
                # Sample at 10 points for smooth interpolation
                rpms = np.linspace(rpm_lo, rpm_hi, 10)
                for rpm in rpms:
                    # T = P × 60 / (2π × RPM)   [power in W]
                    t_peak = (self.peak_power * 60.0) / (2.0 * np.pi * rpm)
                    t_cont = (self.continuous_power * 60.0) / (2.0 * np.pi * rpm)
                    rpm_pts.append(float(rpm))
                    peak_pts.append(float(t_peak))
                    cont_pts.append(float(t_cont))

        self._peak_torque_curve = interp1d(
            rpm_pts, peak_pts, kind="linear", fill_value="extrapolate"
        )
        self._continuous_torque_curve = interp1d(
            rpm_pts, cont_pts, kind="linear", fill_value="extrapolate"
        )
        logger.debug(
            "Torque curves built: %d sample points, RPM range [%.0f, %.0f]",
            len(rpm_pts), min(rpm_pts), max(rpm_pts),
        )

    def _validate_parameters(self) -> None:
        """Run sanity checks on every loaded parameter."""
        assert self.mass > 0, "Mass must be positive"
        assert abs(self.weight_dist_front + self.weight_dist_rear - 1.0) < 0.01, (
            "Weight distribution must sum to 1.0, "
            f"got {self.weight_dist_front + self.weight_dist_rear}"
        )
        assert self.wheelbase > 0, "Wheelbase must be positive"
        assert self.gear_ratio > 0, "Gear ratio must be positive"
        assert self.wheel_radius > 0, "Wheel radius must be positive"
        assert self.battery_peak_power >= self.battery_continuous_power, (
            "Battery peak power must be ≥ continuous power"
        )
        assert 0.5 < self.tire_peak_mu < 2.5, (
            f"Tire μ = {self.tire_peak_mu} seems unrealistic (expected 0.5–2.5)"
        )
        logger.debug("All parameter validations passed")

    # ------------------------------------------------------------------ #
    #  Speed ↔ RPM conversion helpers                                     #
    # ------------------------------------------------------------------ #

    def _speed_to_motor_rpm(self, speed: float) -> float:
        """Convert vehicle speed (m/s) to motor RPM.

        Kinematic chain::

            wheel_angular_vel = speed / wheel_radius          [rad/s]
            wheel_rpm         = wheel_angular_vel × 60 / 2π   [RPM]
            motor_rpm         = wheel_rpm × gear_ratio         [RPM]
        """
        wheel_rpm = (speed / self.wheel_radius) * (60.0 / (2.0 * np.pi))
        return min(wheel_rpm * self.gear_ratio, self.max_rpm)

    def _motor_rpm_to_speed(self, motor_rpm: float) -> float:
        """Convert motor RPM to vehicle speed (m/s)."""
        wheel_rpm = motor_rpm / self.gear_ratio
        return wheel_rpm * self.wheel_radius * (2.0 * np.pi / 60.0)

    # ------------------------------------------------------------------ #
    #  Core physics methods                                               #
    # ------------------------------------------------------------------ #

    def get_motor_torque(self, speed: float, mode: str = "peak") -> float:
        """Return available motor torque at *speed*.

        Args:
            speed: Vehicle speed in **m/s**.
            mode: ``'peak'`` (2-min rating) or ``'continuous'``.

        Returns:
            Total motor-shaft torque in **Nm** (all motors combined).
        """
        # Guard against near-zero speed (avoids numerical noise)
        motor_rpm = self._speed_to_motor_rpm(max(speed, 0.01))

        if mode == "peak":
            torque_per_motor = float(self._peak_torque_curve(motor_rpm))
        else:
            torque_per_motor = float(self._continuous_torque_curve(motor_rpm))

        # Clamp: can never exceed rated peak (extrapolation guard)
        if mode == "peak":
            torque_per_motor = min(torque_per_motor, self.peak_torque)
        else:
            torque_per_motor = min(torque_per_motor, self.continuous_torque)

        return torque_per_motor * self.motor_count

    def get_motor_power(self, speed: float, mode: str = "peak") -> float:
        """Return available motor mechanical power at *speed*.

        .. math::
            P = T \\times \\omega = T \\times \\frac{2\\pi \\times \\text{RPM}}{60}

        Args:
            speed: Vehicle speed in **m/s**.
            mode: ``'peak'`` or ``'continuous'``.

        Returns:
            Mechanical power in **Watts**.
        """
        torque = self.get_motor_torque(speed, mode)
        motor_rpm = self._speed_to_motor_rpm(max(speed, 0.01))
        omega = motor_rpm * 2.0 * np.pi / 60.0  # rad/s
        return float(torque * omega)

    def calculate_aero_forces(self, speed: float) -> Dict[str, float]:
        """Calculate aerodynamic downforce and drag.

        All aero forces scale with the **square** of speed:

        .. math::
            F = \\tfrac{1}{2} \\rho \\, C \\, A \\, v^2

        At 30 m/s (108 km/h) downforce can nearly double the effective
        tyre load — a huge grip advantage, but drag limits top speed.

        Args:
            speed: Vehicle speed in **m/s**.

        Returns:
            Dictionary with keys ``downforce_front``, ``downforce_rear``,
            ``total_downforce``, and ``drag`` — all in **Newtons**.
        """
        # Dynamic pressure q = ½ρv²
        q = 0.5 * self.air_density * speed ** 2

        # Lift coefficients are negative in the config (negative = downforce).
        # We take the absolute value so that the returned forces are positive-down.
        df_front = q * abs(self.lift_coeff_front) * self.aero_reference_area
        df_rear = q * abs(self.lift_coeff_rear) * self.aero_reference_area

        drag = q * self.drag_coeff * self.frontal_area

        return {
            "downforce_front": float(df_front),
            "downforce_rear": float(df_rear),
            "total_downforce": float(df_front + df_rear),
            "drag": float(drag),
        }

    def calculate_weight_transfer(
            self, ax: float, ay: float
    ) -> Dict[str, float]:
        """Calculate tyre normal loads under combined accel / cornering.

        **Longitudinal weight transfer** (braking/acceleration)::

            ΔFz_long = m × ax × h_cg / wheelbase

        **Lateral weight transfer** per axle::

            ΔFz_lat = m × ay × h_cg × weight_ratio / track_width

        Args:
            ax: Longitudinal acceleration in **m/s²**
                (positive = accelerating forward).
            ay: Lateral acceleration in **m/s²**
                (positive = turning left → load goes to the right).

        Returns:
            Normal forces (N) on each corner and per-axle totals.
        """
        # ── Static loads ─────────────────────────────────────────────
        static_f = self.weight_front  # total front axle, N
        static_r = self.weight_rear  # total rear axle, N

        # ── Longitudinal transfer ────────────────────────────────────
        delta_long = (self.mass * ax * self.cg_height) / self.wheelbase
        front_total = static_f - delta_long  # accel unloads front
        rear_total = static_r + delta_long  # accel loads rear

        # ── Lateral transfer (per axle) ──────────────────────────────
        delta_lat_f = (
                              self.mass * ay * self.cg_height * self.weight_dist_front
                      ) / self.track_width_front
        delta_lat_r = (
                              self.mass * ay * self.cg_height * self.weight_dist_rear
                      ) / self.track_width_rear

        fl = (front_total / 2.0) - delta_lat_f
        fr = (front_total / 2.0) + delta_lat_f
        rl = (rear_total / 2.0) - delta_lat_r
        rr = (rear_total / 2.0) + delta_lat_r

        # Wheel can't pull the ground — clamp to zero
        fl = max(fl, 0.0)
        fr = max(fr, 0.0)
        rl = max(rl, 0.0)
        rr = max(rr, 0.0)

        return {
            "front_left": float(fl),
            "front_right": float(fr),
            "rear_left": float(rl),
            "rear_right": float(rr),
            "front_total": float(fl + fr),
            "rear_total": float(rl + rr),
        }

    def calculate_tire_force_capacity(
            self, normal_load: float, slip_angle: float = 0.0, is_braking: bool = False
    ) -> Dict[str, float]:
        """Return the force budget of a single tyre.

        Uses the **friction-circle** concept and a simplified
        **Pacejka Magic Formula** for lateral force:

        .. math::
            F_y = D \\sin\\bigl(C \\arctan(B \\alpha)\\bigr)

        where *D = μ_eff × F_z*.

        Load sensitivity models the real-world effect that μ *decreases*
        slightly with increasing load (tyre saturates).

        Args:
            normal_load: Vertical force on the tyre, **N**.
            slip_angle: Tyre slip angle in **degrees** (default 0).

        Returns:
            Dictionary with ``max_total_force``, ``max_longitudinal``,
            ``lateral_force``, and ``mu_effective``.
        """
        if normal_load <= 0:
            return {
                "max_total_force": 0.0,
                "max_longitudinal": 0.0,
                "lateral_force": 0.0,
                "mu_effective": 0.0,
            }

        # ── Effective μ with load sensitivity ────────────────────────
        # Reference load = static quarter-car weight
        # Sensitivity is per kN of load difference (YAML value is −0.02/kN)
        ref_load = self.total_weight / 4.0
        base_mu = self.tire_braking_mu if is_braking else self.tire_peak_mu
        mu_eff = base_mu + self.tire_load_sensitivity * (
                (normal_load - ref_load) / 1000.0
        )
        mu_eff = float(np.clip(mu_eff, 0.3, 2.0))

        # ── Friction circle radius ───────────────────────────────────
        max_total = mu_eff * normal_load  # N

        # ── Pacejka lateral force ────────────────────────────────────
        alpha_rad = np.deg2rad(slip_angle)
        B = self.pacejka_B
        C = self.pacejka_C
        D = max_total  # peak = μ × Fz
        lat_force = float(D * np.sin(C * np.arctan(B * alpha_rad)))

        # ── Remaining longitudinal budget (friction circle) ──────────
        # Fx² + Fy² ≤ (μ Fz)²  →  Fx_max = √(max² − Fy²)
        max_long = float(np.sqrt(max(0.0, max_total ** 2 - lat_force ** 2)))

        return {
            "max_total_force": float(max_total),
            "max_longitudinal": max_long,
            "lateral_force": lat_force,
            "mu_effective": mu_eff,
        }

    # ------------------------------------------------------------------ #
    #  Acceleration-limit methods                                         #
    # ------------------------------------------------------------------ #

    def max_lateral_acceleration(self, speed: float) -> float:
        """Maximum lateral (cornering) acceleration at *speed*.

        Process:
        1. Add aero downforce to static tyre loads.
        2. Evaluate Pacejka lateral force at peak slip angle (4°).
        3. Sum all four tyres → *a_y = ΣF_y / m*.

        At low speed the limit is pure tyre grip (~1.5 g).
        At high speed aero downforce pushes the limit well above 2 g.

        Args:
            speed: Vehicle speed in **m/s**.

        Returns:
            Maximum lateral acceleration in **m/s²**.
        """
        aero = self.calculate_aero_forces(speed)

        # Per-tyre normal loads (static + aero, no cornering yet)
        fz_f = self.weight_front / 2.0 + aero["downforce_front"] / 2.0
        fz_r = self.weight_rear / 2.0 + aero["downforce_rear"] / 2.0

        # Lateral capacity at peak slip angle (≈ 4° for racing slicks)
        front = self.calculate_tire_force_capacity(fz_f, slip_angle=4.0)
        rear = self.calculate_tire_force_capacity(fz_r, slip_angle=4.0)

        # Two front + two rear tyres
        total_lat = 2.0 * front["lateral_force"] + 2.0 * rear["lateral_force"]

        return float(total_lat / self.mass)

    def max_cornering_speed(self, radius: float) -> float:
        """Maximum speed through a corner of given *radius*.

        For circular motion: *a_y = v² / r*, so *v = √(a_y_max × r)*.

        Because downforce (and therefore grip) increases with speed, we
        iterate to find the self-consistent solution.

        Args:
            radius: Corner radius in **metres**.

        Returns:
            Maximum cornering speed in **m/s**.
        """
        if radius <= 0:
            return 0.0

        # Start with a guess using static grip (no aero)
        ay_static = self.tire_peak_mu * G
        v = np.sqrt(ay_static * radius)

        # Iterate: recalculate ay_max at current speed, recompute v
        for _ in range(10):
            ay = self.max_lateral_acceleration(v)
            v_new = np.sqrt(ay * radius)
            if abs(v_new - v) < 0.01:
                break
            v = v_new

        # Never exceed top speed
        return float(min(v, self.top_speed))

    def max_longitudinal_acceleration(
            self, speed: float, throttle: float = 100.0, dt: float = 0.01
    ) -> float:
        """Maximum forward acceleration at *speed*.

        The result is the **minimum** of four independent limits:

        1. **Motor torque** → force at contact patch.
        2. **Motor power** → *F = P / v* (dominates at high speed).
        3. **Battery power** → limited by pack current / thermal rating.
        4. **Rear-tyre traction** → grip on driven axle only.

        Drag is subtracted from the net available force.

        Args:
            speed: Vehicle speed in **m/s**.
            throttle: Throttle input (0-100%).
            dt: Time since last call (for thermal update).

        Returns:
            Maximum longitudinal acceleration in **m/s²** (≥ 0).
        """
        # ── 1. Motor & Drivetrain limit ────────────────────────────────────
        # Get dynamic properties from electro-thermal model
        motor_state = self.motor_model.get_wheel_torque_realtime(
            throttle=throttle,
            vehicle_speed=speed,
            battery_voltage=self.battery_voltage_nom,  # Or a dynamically calculated voltage
            gear_ratio=self.gear_ratio,
            wheel_radius=self.wheel_radius,
            dt=dt
        )
        motor_force = motor_state["wheel_force"]

        # ── 2. Battery power limit ───────────────────────────────────
        bat_p = self.battery_peak_power
        # Mechanical power available after all efficiency losses
        bat_force = (
            (bat_p * motor_state["motor_efficiency"] * self.drivetrain_efficiency / speed)
            if speed > 1.0
            else 1e6
        )

        # ── 3. Rear-tyre traction limit ──────────────────────────────
        # Estimate weight transfer at ~1 g forward acceleration
        loads = self.calculate_weight_transfer(ax=1.0 * G, ay=0.0)
        aero = self.calculate_aero_forces(speed)
        # Add aero downforce to rear axle
        rear_load_per_tire = (
                loads["rear_total"] / 2.0 + aero["downforce_rear"] / 2.0
        )
        rear_tire = self.calculate_tire_force_capacity(rear_load_per_tire)
        traction_force = 2.0 * rear_tire["max_longitudinal"]  # N

        # ── Net force (subtract drag) ────────────────────────────────
        available = min(motor_force, bat_force, traction_force)
        net = available - aero["drag"]

        # a = F / m, floored at zero (can't "decelerate" via throttle)
        return float(max(net / self.mass, 0.0))

    def max_braking_deceleration(self, speed: float) -> float:
        """Maximum braking deceleration at *speed* (returned as positive value).

        Mechanical brakes only — all four wheels, tire-grip limited.
        Regenerative braking has been removed from this model.

        Args:
            speed: Vehicle speed in **m/s**.

        Returns:
            Maximum braking deceleration in **m/s²** (positive number).
        """
        # ── Weight transfer under heavy braking (~1.5 g estimate) ────
        loads = self.calculate_weight_transfer(ax=-1.5 * G, ay=0.0)
        aero = self.calculate_aero_forces(speed)

        # ── Mechanical tyre braking ──────────────────────────────────
        fz_f = loads["front_total"] / 2.0 + aero["downforce_front"] / 2.0
        fz_r = loads["rear_total"] / 2.0 + aero["downforce_rear"] / 2.0

        brake_front = 2.0 * self.calculate_tire_force_capacity(fz_f, is_braking=True)["max_longitudinal"]
        brake_rear = 2.0 * self.calculate_tire_force_capacity(fz_r, is_braking=True)["max_longitudinal"]
        mech_force = brake_front + brake_rear

        return float(mech_force / self.mass)

    # ------------------------------------------------------------------ #
    #  Energy accounting                                                  #
    # ------------------------------------------------------------------ #

    def calculate_energy_consumption(
            self, speed: float, ax: float, distance: float
    ) -> Dict[str, float]:
        """Estimate energy consumed over a short track segment.

        Args:
            speed: Average speed over the segment in **m/s**.
            ax: Average longitudinal acceleration in **m/s²**.
            distance: Segment length in **metres**.

        Returns:
            Dictionary with ``energy_consumed`` (Wh), ``time_elapsed`` (s),
            and ``power_avg`` (W).
        """
        if speed <= 0:
            return {"energy_consumed": 0.0, "time_elapsed": 0.0, "power_avg": 0.0}

        dt = distance / speed  # s

        # Forces acting on the car
        f_accel = self.mass * ax  # N
        f_drag = self.calculate_aero_forces(speed)["drag"]  # N
        f_rr = self.rolling_resistance_coeff * self.total_weight  # N

        # Total mechanical force required at the wheels
        total_force = f_accel + f_drag + f_rr  # N
        mech_power = total_force * speed  # W

        # Map to battery power through efficiency chain
        # Regenerative braking has been removed — only motoring accounted for.
        if mech_power > 0:
            battery_power = mech_power / (
                    self.motor_efficiency * self.drivetrain_efficiency
            )
        else:
            battery_power = 0.0  # No regen — unpowered segment costs nothing

        energy_wh = battery_power * dt / 3600.0  # J → Wh

        return {
            "energy_consumed": float(energy_wh),
            "time_elapsed": float(dt),
            "power_avg": float(battery_power),
        }

    # ------------------------------------------------------------------ #
    #  Diagnostics / validation helper                                    #
    # ------------------------------------------------------------------ #

    def validate_at_speed(self, speed: float) -> None:
        """Print a full physics snapshot at the given speed.

        Useful for quick sanity-checking during development or when
        comparing against hand calculations.

        Args:
            speed: Test speed in **m/s**.
        """
        kmh = speed * 3.6
        print(f"\n{'=' * 60}")
        print(f"VEHICLE DYNAMICS AT {speed:.1f} m/s ({kmh:.1f} km/h)")
        print(f"{'=' * 60}")

        # Aero
        aero = self.calculate_aero_forces(speed)
        print(f"\nAERODYNAMICS:")
        print(f"  Downforce Front: {aero['downforce_front']:>8.0f} N")
        print(f"  Downforce Rear:  {aero['downforce_rear']:>8.0f} N")
        print(f"  Total Downforce: {aero['total_downforce']:>8.0f} N")
        print(f"  Drag:            {aero['drag']:>8.0f} N")

        # Motor
        t_peak = self.get_motor_torque(speed, "peak")
        p_peak = self.get_motor_power(speed, "peak")
        rpm = self._speed_to_motor_rpm(speed)
        print(f"\nMOTOR ({rpm:.0f} RPM):")
        print(f"  Torque (peak):   {t_peak:>8.1f} Nm")
        print(f"  Power  (peak):   {p_peak / 1000:>8.1f} kW")

        # Acceleration limits
        ax = self.max_longitudinal_acceleration(speed, throttle=100.0)
        ay = self.max_lateral_acceleration(speed)
        ab = self.max_braking_deceleration(speed)
        print(f"\nACCELERATION LIMITS:")
        print(f"  Forward:  {ax / G:>5.2f} g  ({ax:>6.1f} m/s²)")
        print(f"  Lateral:  {ay / G:>5.2f} g  ({ay:>6.1f} m/s²)")
        print(f"  Braking:  {ab / G:>5.2f} g  ({ab:>6.1f} m/s²)")

        # Energy at this speed (steady-state, ax=0, over 100 m)
        energy = self.calculate_energy_consumption(speed, ax=0.0, distance=100.0)
        print(f"\nENERGY (100 m steady-state):")
        print(f"  Power draw:  {energy['power_avg'] / 1000:>6.1f} kW")
        print(f"  Energy used: {energy['energy_consumed']:>6.2f} Wh")

if __name__ == '__main__':
    from pathlib import Path
    
    # Path to your config file (assuming you run it from the root of 'Custom LTS')
    config_file = Path(__file__).parent.parent.parent / "config" / "vehicle_params.yaml"
    
    try:
        # 1. Instantiate the physics model
        vehicle = VehicleDynamics(str(config_file))
        
        # 2. Print a full physics snapshot at 15 m/s (54 km/h)
        print("\n--- Running Diagnostic Test ---")
        vehicle.validate_at_speed(speed=15.0)
        
        # You can also test it at another speed, like top speed
        vehicle.validate_at_speed(speed=30.0)
        
    except FileNotFoundError:
        print(f"Error: Could not find config file at {config_file}")
        print("Make sure you are running the script from the correct directory.")
