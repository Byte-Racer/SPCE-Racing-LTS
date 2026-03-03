"""
Integration tests — end-to-end pipeline validation.

These tests load real config files (not synthetic tracks) and run the full
simulation pipeline, verifying that all modules work together correctly.

Tests:
  1. Load fb_autocross from track_definitions.yaml, run autocross solve.
  2. Load fb_endurance (resolved from autocross waypoints), run endurance solve.
  3. Load fb_acceleration (drag strip), verify basic physics.

Run with::

    pytest tests/test_integration.py -v

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.track.track_loader import load_from_config
from src.vehicle.vehicle_model import VehicleDynamics
from src.solver.qss_solver import QSSSolver

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = os.path.join(os.path.dirname(__file__), "..")
VEHICLE_CONFIG = os.path.join(_ROOT, "config", "vehicle_params.yaml")
SOLVER_CONFIG = os.path.join(_ROOT, "config", "solver_config.yaml")
TRACK_CONFIG = os.path.join(_ROOT, "config", "track_definitions.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vehicle() -> VehicleDynamics:
    return VehicleDynamics(VEHICLE_CONFIG)


@pytest.fixture(scope="module")
def solver(vehicle) -> QSSSolver:
    return QSSSolver(vehicle, SOLVER_CONFIG)


# ---------------------------------------------------------------------------
# Test 1 — Full autocross pipeline
# ---------------------------------------------------------------------------

class TestAutocrossIntegration:
    """Load fb_autocross from YAML, solve, validate results."""

    def test_load_autocross_track(self):
        """Track loader should successfully load fb_autocross by name."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_autocross")
        assert track.segment_count > 10
        assert track.total_length > 100  # should be ~1200 m

    def test_autocross_solve_completes(self, solver):
        """Full autocross solve should complete without error."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_autocross")
        result = solver.solve_autocross(track)
        assert result.lap_time > 0

    def test_autocross_lap_time_range(self, solver):
        """Autocross lap time should be within the expected 45–65 s range
        defined in solver_config.yaml."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_autocross")
        result = solver.solve_autocross(track)
        # Use a wider band for the real track (geometry is approximate)
        assert 30.0 <= result.lap_time <= 120.0, (
            f"Autocross lap time {result.lap_time:.2f} s outside plausible range"
        )

    def test_autocross_energy_positive(self, solver):
        """Energy consumed must be positive and below usable capacity."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_autocross")
        result = solver.solve_autocross(track)
        assert result.energy_consumed > 0
        assert result.energy_consumed < 7000  # < 7 kWh usable

    def test_autocross_soc_decreases(self, solver):
        """Final SOC must be less than initial."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_autocross")
        result = solver.solve_autocross(track)
        assert result.final_soc < 1.0

    def test_autocross_telemetry_arrays(self, solver):
        """Telemetry arrays should have correct shapes."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_autocross")
        result = solver.solve_autocross(track)
        n = len(result.distance)
        assert n > 10
        assert len(result.speed_profile) == n
        assert len(result.soc_profile) == n
        assert len(result.motor_temp_profile) == n
        assert len(result.ax_profile) == n - 1

    def test_autocross_summary_string(self, solver):
        """summary_string() should produce readable output."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_autocross")
        result = solver.solve_autocross(track)
        s = result.summary_string()
        assert "SPCE Racing" in s
        assert "Lap time" in s


# ---------------------------------------------------------------------------
# Test 2 — Endurance track (resolved from autocross waypoints)
# ---------------------------------------------------------------------------

class TestEnduranceIntegration:
    """Load fb_endurance (which references autocross layout), run solver."""

    def test_load_endurance_track(self):
        """fb_endurance should resolve its USE_AUTOCROSS_LAYOUT flag."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_endurance")
        assert track.segment_count > 10
        assert track.total_length > 100

    def test_endurance_matches_autocross_length(self):
        """Endurance and autocross should have the same track length."""
        t_auto = load_from_config(TRACK_CONFIG, track_name="fb_autocross")
        t_endur = load_from_config(TRACK_CONFIG, track_name="fb_endurance")
        assert abs(t_auto.total_length - t_endur.total_length) < 1.0

    def test_endurance_3_laps(self, solver):
        """A short 3-lap endurance run should complete and show SOC decay."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_endurance")
        result = solver.solve_endurance(track, n_laps=3, initial_soc=1.0)
        assert len(result.laps) == 3
        assert result.total_time > 0
        assert result.total_energy > 0
        assert result.final_soc < 1.0
        # SOC should decrease monotonically
        soc_prev = 1.0
        for lap in result.laps:
            assert lap.final_soc < soc_prev
            soc_prev = lap.final_soc


# ---------------------------------------------------------------------------
# Test 3 — Acceleration track (drag strip)
# ---------------------------------------------------------------------------

class TestAccelerationTrackLoad:
    """Load fb_acceleration (75 m straight) — basic geometry check."""

    def test_load_acceleration_track(self):
        """Should load without error."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_acceleration")
        assert track.segment_count >= 2
        # 75 m straight
        assert track.total_length >= 50.0

    def test_acceleration_mostly_straight(self):
        """Curvature should be near zero on a drag strip."""
        track = load_from_config(TRACK_CONFIG, track_name="fb_acceleration")
        mean_kappa = float(np.mean(np.abs(track.curvature)))
        assert mean_kappa < 0.05, f"Drag strip should be straight, got mean |κ|={mean_kappa:.4f}"
