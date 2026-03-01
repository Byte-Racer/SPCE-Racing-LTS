"""
Unit tests for VehicleDynamics.

Tests cover:
  - Initialisation & parameter loading
  - Edge cases (zero / negative / extreme speeds)
  - Physics sanity checks (energy conservation, force directions)
  - Individual subsystem calculations (aero, weight transfer, tires)
"""

import math
import os
import pytest

# Adjust the import path so tests can find src/
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vehicle.vehicle_model import VehicleDynamics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "vehicle_params.yaml"
)


@pytest.fixture(scope="module")
def vehicle() -> VehicleDynamics:
    """Load vehicle model once for the entire test module."""
    return VehicleDynamics(CONFIG_PATH)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInitialisation:
    """Verify that the model loads without error and stores sane values."""

    def test_loads_successfully(self, vehicle: VehicleDynamics):
        assert vehicle.mass > 0

    def test_weight_distribution_sums_to_one(self, vehicle: VehicleDynamics):
        total = vehicle.weight_dist_front + vehicle.weight_dist_rear
        assert abs(total - 1.0) < 0.01

    def test_power_conversion_kw_to_w(self, vehicle: VehicleDynamics):
        # YAML has peak_power: 62 (kW), should be stored as 62 000 W
        assert vehicle.peak_power == pytest.approx(62_000, rel=0.01)
        assert vehicle.continuous_power == pytest.approx(64_000, rel=0.01)
        assert vehicle.battery_peak_power == pytest.approx(97_000, rel=0.01)

    def test_top_speed_in_ms(self, vehicle: VehicleDynamics):
        # YAML has 120 km/h → 33.33 m/s
        assert vehicle.top_speed == pytest.approx(120 / 3.6, rel=0.01)

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            VehicleDynamics("nonexistent.yaml")


# ---------------------------------------------------------------------------
# Motor torque & power
# ---------------------------------------------------------------------------


class TestMotor:
    """Motor torque curve and power output."""

    def test_torque_at_zero_speed(self, vehicle: VehicleDynamics):
        """At standstill, motor should deliver peak torque."""
        t = vehicle.get_motor_torque(0.0, mode="peak")
        assert t == pytest.approx(vehicle.peak_torque, rel=0.05)

    def test_torque_at_low_speed(self, vehicle: VehicleDynamics):
        """In the constant-torque region torque should be flat."""
        # Both speeds must be below the base RPM of 4500.
        # At 5 m/s → motor RPM ≈ 2350, at 2 m/s → ~940  (both constant-torque).
        t2 = vehicle.get_motor_torque(2.0, mode="peak")
        t5 = vehicle.get_motor_torque(5.0, mode="peak")
        assert t2 == pytest.approx(t5, rel=0.10)

    def test_torque_decreases_at_high_speed(self, vehicle: VehicleDynamics):
        """In the power-limited region torque must drop."""
        t_low = vehicle.get_motor_torque(5.0, mode="peak")
        t_high = vehicle.get_motor_torque(30.0, mode="peak")
        assert t_high < t_low

    def test_continuous_less_than_peak(self, vehicle: VehicleDynamics):
        t_peak = vehicle.get_motor_torque(5.0, mode="peak")
        t_cont = vehicle.get_motor_torque(5.0, mode="continuous")
        assert t_cont <= t_peak

    def test_power_positive(self, vehicle: VehicleDynamics):
        p = vehicle.get_motor_power(10.0)
        assert p > 0

    def test_power_bounded_by_peak(self, vehicle: VehicleDynamics):
        """Mechanical power should never exceed rated peak."""
        for v in [5, 10, 15, 20, 25, 30]:
            p = vehicle.get_motor_power(float(v), mode="peak")
            # Allow 5 % tolerance for interpolation artefacts
            assert p <= vehicle.peak_power * 1.05


# ---------------------------------------------------------------------------
# Aerodynamics
# ---------------------------------------------------------------------------


class TestAero:
    def test_zero_speed_no_aero(self, vehicle: VehicleDynamics):
        aero = vehicle.calculate_aero_forces(0.0)
        assert aero["drag"] == 0.0
        assert aero["total_downforce"] == 0.0

    def test_drag_increases_with_speed(self, vehicle: VehicleDynamics):
        d10 = vehicle.calculate_aero_forces(10.0)["drag"]
        d20 = vehicle.calculate_aero_forces(20.0)["drag"]
        # Drag ∝ v²  →  d20 should be ~4× d10
        assert d20 > d10
        assert d20 == pytest.approx(d10 * 4, rel=0.05)

    def test_downforce_positive(self, vehicle: VehicleDynamics):
        aero = vehicle.calculate_aero_forces(20.0)
        assert aero["downforce_front"] > 0
        assert aero["downforce_rear"] > 0

    def test_rear_downforce_greater(self, vehicle: VehicleDynamics):
        """Rear Cl magnitude is larger → more rear downforce."""
        aero = vehicle.calculate_aero_forces(20.0)
        assert aero["downforce_rear"] > aero["downforce_front"]


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------


class TestWeightTransfer:
    def test_static_loads(self, vehicle: VehicleDynamics):
        """No acceleration → loads should equal static distribution."""
        loads = vehicle.calculate_weight_transfer(ax=0.0, ay=0.0)
        expected_front = vehicle.weight_front
        expected_rear = vehicle.weight_rear
        assert loads["front_total"] == pytest.approx(expected_front, rel=0.01)
        assert loads["rear_total"] == pytest.approx(expected_rear, rel=0.01)

    def test_accel_shifts_weight_rear(self, vehicle: VehicleDynamics):
        loads_static = vehicle.calculate_weight_transfer(0.0, 0.0)
        loads_accel = vehicle.calculate_weight_transfer(ax=5.0, ay=0.0)
        assert loads_accel["rear_total"] > loads_static["rear_total"]
        assert loads_accel["front_total"] < loads_static["front_total"]

    def test_braking_shifts_weight_front(self, vehicle: VehicleDynamics):
        loads_static = vehicle.calculate_weight_transfer(0.0, 0.0)
        loads_brake = vehicle.calculate_weight_transfer(ax=-5.0, ay=0.0)
        assert loads_brake["front_total"] > loads_static["front_total"]
        assert loads_brake["rear_total"] < loads_static["rear_total"]

    def test_no_negative_loads(self, vehicle: VehicleDynamics):
        """Even under extreme conditions, tyre loads should be ≥ 0."""
        loads = vehicle.calculate_weight_transfer(ax=30.0, ay=30.0)
        for key in ("front_left", "front_right", "rear_left", "rear_right"):
            assert loads[key] >= 0


# ---------------------------------------------------------------------------
# Tire model
# ---------------------------------------------------------------------------


class TestTires:
    def test_zero_load_zero_force(self, vehicle: VehicleDynamics):
        result = vehicle.calculate_tire_force_capacity(0.0)
        assert result["max_total_force"] == 0.0

    def test_force_increases_with_load(self, vehicle: VehicleDynamics):
        r500 = vehicle.calculate_tire_force_capacity(500.0)
        r1000 = vehicle.calculate_tire_force_capacity(1000.0)
        assert r1000["max_total_force"] > r500["max_total_force"]

    def test_friction_circle_constraint(self, vehicle: VehicleDynamics):
        """Fx² + Fy² should not exceed (μFz)²."""
        r = vehicle.calculate_tire_force_capacity(800.0, slip_angle=4.0)
        fx = r["max_longitudinal"]
        fy = r["lateral_force"]
        fmax = r["max_total_force"]
        assert (fx ** 2 + fy ** 2) <= fmax ** 2 * 1.01  # 1 % tolerance

    def test_load_sensitivity(self, vehicle: VehicleDynamics):
        """Higher load should have slightly lower μ."""
        r_light = vehicle.calculate_tire_force_capacity(500.0)
        r_heavy = vehicle.calculate_tire_force_capacity(1500.0)
        assert r_heavy["mu_effective"] < r_light["mu_effective"]


# ---------------------------------------------------------------------------
# Acceleration limits
# ---------------------------------------------------------------------------


class TestAccelerationLimits:
    def test_forward_accel_positive(self, vehicle: VehicleDynamics):
        ax = vehicle.max_longitudinal_acceleration(5.0)
        assert ax > 0

    def test_forward_accel_decreases_with_speed(self, vehicle: VehicleDynamics):
        """Higher speed → more drag, less available torque."""
        ax5 = vehicle.max_longitudinal_acceleration(5.0)
        ax25 = vehicle.max_longitudinal_acceleration(25.0)
        assert ax25 < ax5

    def test_lateral_accel_positive(self, vehicle: VehicleDynamics):
        ay = vehicle.max_lateral_acceleration(10.0)
        assert ay > 0

    def test_lateral_accel_reasonable(self, vehicle: VehicleDynamics):
        """Should be roughly 1.0–3.0 g for FSAE car at moderate speed."""
        ay = vehicle.max_lateral_acceleration(15.0)
        g_val = ay / 9.81
        assert 1.0 < g_val < 3.0

    def test_braking_decel_positive(self, vehicle: VehicleDynamics):
        ab = vehicle.max_braking_deceleration(15.0)
        assert ab > 0

    def test_braking_greater_without_regen_below_cutoff(
        self, vehicle: VehicleDynamics
    ):
        """Below regen cutoff speed, regen should make no difference."""
        speed = 1.0  # well below 2.78 m/s cutoff
        with_regen = vehicle.max_braking_deceleration(speed, use_regen=True)
        without_regen = vehicle.max_braking_deceleration(speed, use_regen=False)
        assert with_regen == pytest.approx(without_regen, rel=0.01)

    def test_cornering_speed_values(self, vehicle: VehicleDynamics):
        """Tighter corner → lower max speed."""
        v_tight = vehicle.max_cornering_speed(7.5)
        v_wide = vehicle.max_cornering_speed(15.0)
        assert v_wide > v_tight

    def test_cornering_speed_zero_radius(self, vehicle: VehicleDynamics):
        assert vehicle.max_cornering_speed(0.0) == 0.0


# ---------------------------------------------------------------------------
# Energy consumption
# ---------------------------------------------------------------------------


class TestEnergy:
    def test_steady_state_energy_positive(self, vehicle: VehicleDynamics):
        """Cruising at constant speed should consume positive energy."""
        e = vehicle.calculate_energy_consumption(speed=15.0, ax=0.0, distance=100.0)
        assert e["energy_consumed"] > 0

    def test_zero_speed_zero_energy(self, vehicle: VehicleDynamics):
        e = vehicle.calculate_energy_consumption(speed=0.0, ax=0.0, distance=100.0)
        assert e["energy_consumed"] == 0.0

    def test_higher_speed_more_energy(self, vehicle: VehicleDynamics):
        """Drag ∝ v² → higher speed should use more energy per metre."""
        e15 = vehicle.calculate_energy_consumption(speed=15.0, ax=0.0, distance=100.0)
        e25 = vehicle.calculate_energy_consumption(speed=25.0, ax=0.0, distance=100.0)
        assert e25["energy_consumed"] > e15["energy_consumed"]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_validate_at_speed_runs(self, vehicle: VehicleDynamics, capsys):
        """validate_at_speed should print without errors."""
        vehicle.validate_at_speed(15.0)
        captured = capsys.readouterr()
        assert "VEHICLE DYNAMICS" in captured.out
        assert "AERODYNAMICS" in captured.out
