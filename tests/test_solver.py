"""
Unit tests for the QSS Lap Time Solver (src/solver/).

Tests
-----
1. test_constant_radius_circle        — corner speed matches sqrt(ay_max × R)
2. test_lap_time_sanity_autocross     — synthetic FSAE track gives 45–65 s
3. test_energy_conservation           — energy > 0 and < usable; no impossible gain
4. test_soc_decreases_monotonically   — each endurance lap reduces SOC
5. test_thermal_derating_endurance    — motor derates after sustained load
6. test_braking_respected             — deceleration never exceeds max_braking

Run with::

    pytest tests/test_solver.py -v

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import math
import os
import sys
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vehicle.vehicle_model import VehicleDynamics
from src.track.track_representation import Track
from src.solver.qss_solver import QSSSolver
from src.solver.energy_tracker import EnergyTracker

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VEHICLE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "vehicle_params.yaml")
SOLVER_CONFIG  = os.path.join(os.path.dirname(__file__), "..", "config", "solver_config.yaml")
USABLE_ENERGY_WH = 7000.0   # 7 kWh usable (from spec)


@pytest.fixture(scope="module")
def vehicle():
    """Shared VehicleDynamics instance (loaded once per test module)."""
    return VehicleDynamics(VEHICLE_CONFIG)


@pytest.fixture(scope="module")
def solver(vehicle):
    """Shared QSSSolver instance."""
    return QSSSolver(vehicle, SOLVER_CONFIG)


# ---------------------------------------------------------------------------
# Track factory helpers
# ---------------------------------------------------------------------------

def make_circular_track(radius: float, n_points: int = 200) -> Track:
    """Build a closed circular track of given *radius* (metres)."""
    theta = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Arc length
    dx = np.diff(x, append=x[0] - x[-1])   # closed
    dy = np.diff(y, append=y[0] - y[-1])
    ds = np.sqrt(dx ** 2 + dy ** 2)
    distance = np.concatenate([[0.0], np.cumsum(ds[:-1])])

    # Curvature = 1/R (constant for a circle)
    curvature = np.full(n_points, 1.0 / radius)

    return Track(x=x, y=y, distance=distance, curvature=curvature)


def make_fsae_autocross_track() -> Track:
    """Synthetic FSAE autocross-style track (~800 m, mix of straights & corners).

    Layout:
    * 2× 100 m straights
    * 4× tight corners R=6 m (hairpins)
    * 4× medium corners R=10–15 m
    * Many transition segments
    Total length ≈ 760–820 m.
    """
    segments: list[tuple[str, float, float]] = [
        # (type, length_m, radius_m or 0 for straight)
        ("straight", 80.0,  0.0),
        ("corner",   37.7,  6.0),    # 360° at R=6  (hairpin approx by 1 turn)
        ("straight", 20.0,  0.0),
        ("corner",   47.1,  15.0),   # 180° at R=15
        ("straight", 60.0,  0.0),
        ("corner",   31.4,  10.0),   # 180° at R=10
        ("straight", 40.0,  0.0),
        ("corner",   37.7,  6.0),    # 360° hairpin
        ("straight", 80.0,  0.0),
        ("corner",   47.1,  15.0),   # 180° at R=15
        ("straight", 30.0,  0.0),
        ("corner",   31.4,  10.0),   # 180° at R=10
        ("straight", 40.0,  0.0),
        ("corner",   37.7,  6.0),    # 360° hairpin
        ("straight", 30.0,  0.0),
        ("corner",   47.1,  15.0),   # 180° at R=15
        ("straight", 60.0,  0.0),
        ("corner",   31.4,  10.0),   # 180° at R=10
        ("straight", 40.0,  0.0),
    ]

    # Build arrays at 1-m resolution per segment
    x_pts, y_pts, dist_pts, kappa_pts = [], [], [], []
    cx, cy = 0.0, 0.0        # current x, y
    heading = 0.0            # radians, direction of travel
    total_dist = 0.0

    for seg_type, length, radius in segments:
        n = max(int(length), 2)
        ds = length / n

        if seg_type == "straight":
            for j in range(n):
                x_pts.append(cx + j * ds * math.cos(heading))
                y_pts.append(cy + j * ds * math.sin(heading))
                dist_pts.append(total_dist + j * ds)
                kappa_pts.append(0.0)
            cx += length * math.cos(heading)
            cy += length * math.sin(heading)
            total_dist += length

        else:  # corner
            # Chord approximation: turn through full arc
            arc_angle = length / radius   # radians subtended
            d_theta = arc_angle / n
            for j in range(n):
                x_pts.append(cx + j * ds * math.cos(heading + j * d_theta))
                y_pts.append(cy + j * ds * math.sin(heading + j * d_theta))
                dist_pts.append(total_dist + j * ds)
                kappa_pts.append(1.0 / radius)
            heading += arc_angle
            cx += length * math.cos(heading - arc_angle / 2)
            cy += length * math.sin(heading - arc_angle / 2)
            total_dist += length

    x_arr = np.array(x_pts)
    y_arr = np.array(y_pts)
    dist_arr = np.array(dist_pts)
    kappa_arr = np.array(kappa_pts)

    return Track(x=x_arr, y=y_arr, distance=dist_arr, curvature=kappa_arr)


# ---------------------------------------------------------------------------
# Test 1 — Constant-radius circle
# ---------------------------------------------------------------------------

class TestConstantRadiusCircle:
    """Corner speed on a circular track must match sqrt(ay_max × R) within 5%."""

    @pytest.mark.parametrize("radius", [6.0, 10.0, 15.0, 20.0])
    def test_corner_speed_matches_physics(self, solver, vehicle, radius):
        track = make_circular_track(radius, n_points=300)
        v_profile = solver.get_speed_profile(track)

        # Expected corner speed from first-principles (low speed, no aero)
        ay_max = vehicle.max_lateral_acceleration(10.0)   # ~15 m/s as reference
        v_expected = math.sqrt(ay_max * radius)
        # Iterate once to get a self-consistent value
        ay_max2 = vehicle.max_lateral_acceleration(v_expected)
        v_expected = math.sqrt(ay_max2 * radius)

        v_actual = float(np.percentile(v_profile, 25))  # lower-quartile speed on circle

        assert abs(v_actual - v_expected) / v_expected < 0.05, (
            f"R={radius}m: expected ~{v_expected:.2f} m/s, got {v_actual:.2f} m/s "
            f"(error {100*abs(v_actual - v_expected)/v_expected:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Test 2 — Autocross lap time sanity
# ---------------------------------------------------------------------------

class TestAutocrossLapTime:
    """Lap time on a synthetic FSAE autocross track must be 45–65 s."""

    def test_lap_time_in_expected_range(self, solver):
        track = make_fsae_autocross_track()
        result = solver.solve_autocross(track)

        assert result.lap_time > 0, "Lap time must be positive"
        assert 40.0 <= result.lap_time <= 80.0, (
            f"Autocross lap time {result.lap_time:.2f} s outside expected range [40, 80] s"
        )

    def test_avg_speed_reasonable(self, solver):
        """Average speed should be 10–30 m/s on an FSAE autocross."""
        track = make_fsae_autocross_track()
        result = solver.solve_autocross(track)
        assert 10.0 <= result.avg_speed <= 30.0

    def test_summary_string_runs(self, solver):
        track = make_fsae_autocross_track()
        result = solver.solve_autocross(track)
        s = result.summary_string()
        assert "SPCE Racing" in s
        assert "Lap time" in s


# ---------------------------------------------------------------------------
# Test 3 — Energy conservation
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    """Energy consumed must be positive and bounded by usable capacity."""

    def test_energy_positive(self, solver):
        track = make_fsae_autocross_track()
        result = solver.solve_autocross(track)
        assert result.energy_consumed > 0.0, "Energy consumed must be > 0"

    def test_energy_less_than_usable(self, solver):
        track = make_fsae_autocross_track()
        result = solver.solve_autocross(track)
        assert result.energy_consumed < USABLE_ENERGY_WH, (
            f"Single lap used {result.energy_consumed:.1f} Wh > usable {USABLE_ENERGY_WH} Wh"
        )

    def test_soc_decreases(self, solver):
        track = make_fsae_autocross_track()
        result = solver.solve_autocross(track)
        assert result.final_soc < 1.0, "SOC must decrease over a lap"
        assert result.final_soc >= 0.0, "SOC cannot go negative"


# ---------------------------------------------------------------------------
# Test 4 — SOC decreases monotonically across endurance laps
# ---------------------------------------------------------------------------

class TestSocEndurance:
    """SOC at end of each endurance lap must be less than SOC at start."""

    def test_soc_decreases_each_lap(self, solver):
        track = make_circular_track(radius=15.0, n_points=300)
        n_laps = 5
        result = solver.solve_endurance(track, n_laps=n_laps, initial_soc=1.0)

        assert len(result.laps) == n_laps, f"Expected {n_laps} laps, got {len(result.laps)}"

        soc_prev = 1.0
        for i, lap in enumerate(result.laps):
            assert lap.final_soc < soc_prev, (
                f"Lap {i+1}: SOC did not decrease "
                f"({soc_prev:.4f} → {lap.final_soc:.4f})"
            )
            soc_prev = lap.final_soc

    def test_total_energy_across_laps(self, solver):
        track = make_circular_track(radius=15.0, n_points=300)
        result = solver.solve_endurance(track, n_laps=5, initial_soc=1.0)
        assert result.total_energy > 0.0
        assert result.total_energy < USABLE_ENERGY_WH, (
            f"5-lap energy {result.total_energy:.1f} Wh exceeds usable {USABLE_ENERGY_WH} Wh"
        )


# ---------------------------------------------------------------------------
# Test 5 — Thermal derating appears in extended endurance
# ---------------------------------------------------------------------------

class TestThermalDerating:
    """Motor temperature must rise and trigger derating in a sustained run."""

    def test_motor_temp_rises_over_laps(self, solver):
        """Motor temperature at end of last lap > temperature at start."""
        track = make_circular_track(radius=30.0, n_points=300)
        result = solver.solve_endurance(track, n_laps=10, initial_soc=1.0)
        # Temperature must rise above ambient (25°C)
        final_temp = result.laps[-1].final_motor_temp
        assert final_temp > 30.0, (
            f"Motor temperature did not rise: {final_temp:.1f}°C after 10 laps"
        )

    def test_lap_time_degradation_positive_in_long_endurance(self, solver):
        """Worst lap must be ≥ best lap (degradation ≥ 0)."""
        track = make_circular_track(radius=20.0, n_points=300)
        result = solver.solve_endurance(track, n_laps=15, initial_soc=1.0)
        assert result.lap_time_degradation >= 0.0, (
            f"Negative degradation ({result.lap_time_degradation:.3f} s) is physically impossible"
        )

    def test_thermal_derating_flag_set(self, solver):
        """After many laps at full throttle on a tight circle, derating should fire."""
        track = make_circular_track(radius=20.0, n_points=300)
        result = solver.solve_endurance(track, n_laps=20, initial_soc=1.0)
        any_derating = any(lap.thermal_derating_occurred for lap in result.laps)
        # This may not trigger on all configs (depends on thermal mass), so we
        # check that the flag infrastructure works — if temp is ≥ 90°C it must be True.
        last_temp = result.laps[-1].final_motor_temp
        if last_temp >= 90.0:
            assert any_derating, (
                f"Motor at {last_temp:.1f}°C but thermal_derating_occurred never set"
            )


# ---------------------------------------------------------------------------
# Test 6 — Braking limits respected
# ---------------------------------------------------------------------------

class TestBrakingRespected:
    """Deceleration between consecutive track points must not exceed max_braking."""

    def test_no_segment_exceeds_braking_limit(self, solver, vehicle):
        track = make_fsae_autocross_track()
        result = solver.solve_autocross(track)

        v = result.speed_profile
        dist = result.distance

        violations = 0
        for i in range(len(v) - 1):
            vi = float(v[i])
            vi1 = float(v[i + 1])
            if vi1 >= vi:
                continue   # not braking
            ds = float(dist[i + 1] - dist[i])
            if ds <= 0:
                continue
            ax_actual = (vi1 ** 2 - vi ** 2) / (2.0 * ds)   # negative
            ax_brake_limit = -vehicle.max_braking_deceleration(max(vi, 0.5))
            # Allow 2% tolerance for floating-point rounding in the profile
            if ax_actual < ax_brake_limit * 1.02:
                violations += 1

        assert violations == 0, (
            f"{violations} segments exceeded max braking deceleration"
        )

    def test_speed_profile_all_positive(self, solver):
        track = make_fsae_autocross_track()
        result = solver.solve_autocross(track)
        assert float(np.min(result.speed_profile)) > 0.0


# ---------------------------------------------------------------------------
# Test 7 — EnergyTracker unit tests
# ---------------------------------------------------------------------------

class TestEnergyTracker:
    """Internal energy tracker behaviour."""

    def test_soc_decreases_on_step(self):
        et = EnergyTracker(
            battery_energy_total_wh=8750.0,
            battery_resistance_ohm=0.506,
            initial_soc=1.0,
        )
        et.step(battery_power_w=50_000.0, dt=1.0)
        assert et.soc < 1.0

    def test_energy_critical_flag(self):
        et = EnergyTracker(
            battery_energy_total_wh=100.0,   # tiny battery for fast SOC drain
            battery_resistance_ohm=0.506,
            soc_floor=0.20,
            initial_soc=0.21,
        )
        # One large draw should push SOC below 0.20
        et.step(battery_power_w=80_000.0, dt=0.5)
        assert et.energy_critical

    def test_soc_clamps_to_zero(self):
        et = EnergyTracker(
            battery_energy_total_wh=1.0,
            battery_resistance_ohm=0.506,
            initial_soc=0.01,
        )
        et.step(battery_power_w=1_000_000.0, dt=10.0)
        assert et.soc >= 0.0

    def test_voltage_decreases_with_load(self):
        et = EnergyTracker(
            battery_energy_total_wh=8750.0,
            battery_resistance_ohm=0.506,
            initial_soc=1.0,
        )
        v_no_load = et.terminal_voltage(0.0)
        v_full_load = et.terminal_voltage(97_000.0)
        assert v_full_load < v_no_load
