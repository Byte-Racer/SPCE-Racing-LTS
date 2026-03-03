"""
tests/test_validation.py — Physics Validation Suite for the SPCE Racing LTS.

Verifies that the simulator produces physically correct and sensible results
against known analytical ground truths and sport-engineering benchmarks.

Tests
-----
1. ``TestCircularCurvature``  — Track module computes κ ≈ 1/R on a perfect circle.
2. ``TestCornerSpeedPhysics`` — QSS speed profile matches v = √(ay_max × R) within 5 %.
3. ``TestAutocrossSanity``    — Full fb_autocross run passes all engineering checks.
4. ``TestEnduranceCarryOver`` — 5-lap endurance shows correct SOC decay, energy sum,
                                 time sum, and motor heating.

Run with::

    pytest tests/test_validation.py -v

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — project root on sys.path so ``src.*`` imports resolve.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.track.track_loader import load_from_config, load_primitives
from src.track.track_representation import Track
from src.vehicle.vehicle_model import VehicleDynamics
from src.solver.qss_solver import QSSSolver
from src.solver.lap_results import LapResult, EnduranceResult

# ---------------------------------------------------------------------------
# Config paths (all relative to project root)
# ---------------------------------------------------------------------------
_ROOT = os.path.join(os.path.dirname(__file__), "..")
VEHICLE_CONFIG = os.path.join(_ROOT, "config", "vehicle_params.yaml")
SOLVER_CONFIG = os.path.join(_ROOT, "config", "solver_config.yaml")
TRACK_CONFIG = os.path.join(_ROOT, "config", "track_definitions.yaml")

# ---------------------------------------------------------------------------
# Shared module-scoped fixtures (constructed once, reused across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vehicle() -> VehicleDynamics:
    """Real VehicleDynamics loaded from vehicle_params.yaml."""
    return VehicleDynamics(VEHICLE_CONFIG)


@pytest.fixture(scope="module")
def solver(vehicle) -> QSSSolver:
    """Real QSSSolver built on top of the real vehicle model."""
    return QSSSolver(vehicle, SOLVER_CONFIG)


@pytest.fixture(scope="module")
def autocross_track() -> Track:
    """fb_autocross loaded from track_definitions.yaml (shared across tests)."""
    return load_from_config(TRACK_CONFIG, track_name="fb_autocross")


@pytest.fixture(scope="module")
def autocross_result(solver, autocross_track) -> LapResult:
    """Single autocross lap result (computed once per module)."""
    return solver.solve_autocross(autocross_track)


@pytest.fixture(scope="module")
def endurance_result(solver, autocross_track) -> EnduranceResult:
    """5-lap endurance result on the fb_autocross layout (computed once)."""
    return solver.solve_endurance(autocross_track, n_laps=5, initial_soc=1.0)


# ---------------------------------------------------------------------------
# Helper: build a perfect circle track via the primitive builder
# ---------------------------------------------------------------------------

def _build_circle_track(radius: float = 10.0) -> Track:
    """Use the production primitive builder to construct a closed circular track.

    The track is composed of a single 360° corner, letting the track module
    handle all smoothing and curvature computation — so we are testing the
    *real* pipeline, not a synthetic shortcut.

    Args:
        radius: Circle radius in metres.

    Returns:
        Fully initialised :class:`Track`.
    """
    primitives = [
        {"type": "corner", "radius": radius, "angle": 360, "direction": "left"},
    ]
    return load_primitives(primitives, smoothing_sigma=1.0, resample_spacing=0.25)


# ===========================================================================
# TEST 1 — Circular Track Curvature
# ===========================================================================

class TestCircularCurvature:
    """Track module must compute κ ≈ 1/R and total_length ≈ 2πR on a circle."""

    RADIUS = 10.0  # metres
    EXPECTED_KAPPA = 1.0 / RADIUS              # 0.1 /m
    EXPECTED_LENGTH = 2.0 * math.pi * RADIUS   # ≈ 62.83 m

    KAPPA_TOL = 0.05   # 5 % relative tolerance on curvature
    LENGTH_TOL = 0.01  # 1 % relative tolerance on total length

    @pytest.fixture(scope="class")
    def circle_track(self) -> Track:
        return _build_circle_track(self.RADIUS)

    def test_curvature_at_every_point(self, circle_track: Track) -> None:
        """κ at every point must be within 5 % of 1/R = 0.1 /m."""
        kappa = np.abs(circle_track.curvature)
        errors = np.abs(kappa - self.EXPECTED_KAPPA) / self.EXPECTED_KAPPA

        worst_idx = int(np.argmax(errors))
        worst_err = float(errors[worst_idx])
        worst_kappa = float(kappa[worst_idx])

        failing = int(np.sum(errors > self.KAPPA_TOL))
        total = len(kappa)

        print(
            f"\n[Curvature] expected κ={self.EXPECTED_KAPPA:.4f} /m  "
            f"| worst={worst_kappa:.4f} /m ({100*worst_err:.1f}% error)  "
            f"| {failing}/{total} points outside {100*self.KAPPA_TOL:.0f}% band"
        )

        assert failing == 0, (
            f"{failing}/{total} points exceeded κ tolerance.\n"
            f"  Expected : {self.EXPECTED_KAPPA:.4f} /m\n"
            f"  Worst    : {worst_kappa:.4f} /m at index {worst_idx} "
            f"(error {100*worst_err:.1f}%)"
        )

    def test_total_length(self, circle_track: Track) -> None:
        """Total track length must be within 1 % of 2πR = 62.83 m."""
        actual = circle_track.total_length
        rel_err = abs(actual - self.EXPECTED_LENGTH) / self.EXPECTED_LENGTH

        print(
            f"\n[Length]    expected={self.EXPECTED_LENGTH:.3f} m  "
            f"actual={actual:.3f} m  error={100*rel_err:.2f}%"
        )

        assert rel_err <= self.LENGTH_TOL, (
            f"Track length out of 1% tolerance.\n"
            f"  Expected : {self.EXPECTED_LENGTH:.3f} m\n"
            f"  Actual   : {actual:.3f} m  (error {100*rel_err:.2f}%)"
        )


# ===========================================================================
# TEST 2 — Corner Speed Physics on Circular Track
# ===========================================================================

class TestCornerSpeedPhysics:
    """QSS speed profile on a pure circle must match v = √(ay_max × R).

    The expected corner speed is computed using 5 fixed-point iterations to
    account for the speed-dependent aero downforce that raises ay_max at
    higher speeds.
    """

    RADIUS = 10.0          # metres — same circle as Test 1
    TOL = 0.05             # 5 % relative tolerance on speed

    @pytest.fixture(scope="class")
    def circle_track(self) -> Track:
        return _build_circle_track(self.RADIUS)

    def _expected_corner_speed(self, vehicle: VehicleDynamics) -> float:
        """Iterate 5 times to get a self-consistent v = √(ay_max(v) × R)."""
        v = math.sqrt(vehicle.max_lateral_acceleration(5.0) * self.RADIUS)
        for _ in range(4):
            ay = vehicle.max_lateral_acceleration(v)
            v = math.sqrt(ay * self.RADIUS)
        return v

    def test_speed_profile_within_5pct(
        self, solver: QSSSolver, vehicle: VehicleDynamics, circle_track: Track
    ) -> None:
        """Every point in the speed profile must be ≤ 5 % above the corner limit."""
        lap = solver.solve_autocross(circle_track)
        v_profile = lap.speed_profile
        v_expected = self._expected_corner_speed(vehicle)

        # On a pure circle every point should be near the corner speed.
        # Allow a small over-shoot margin for numerical integration artefacts.
        errors = (v_profile - v_expected) / v_expected
        bad_indices = np.where(errors > self.TOL)[0]
        max_err = float(np.max(errors)) if len(errors) else 0.0

        print(
            f"\n[CornerSpeed] expected v≈{v_expected:.3f} m/s  "
            f"| profile min={v_profile.min():.3f}  max={v_profile.max():.3f} m/s  "
            f"| worst over-shoot={100*max_err:.1f}%"
        )

        assert len(bad_indices) == 0, (
            f"{len(bad_indices)} points exceed corner speed by >{100*self.TOL:.0f}%.\n"
            f"  Expected  : {v_expected:.3f} m/s\n"
            f"  Max actual: {float(v_profile.max()):.3f} m/s "
            f"(over-shoot {100*max_err:.1f}%)\n"
            f"  First violation at profile index {int(bad_indices[0])}"
        )


# ===========================================================================
# TEST 3 — Autocross Sanity Check (fb_autocross)
# ===========================================================================

class TestAutocrossSanity:
    """Full fb_autocross lap must satisfy all engineering bounds.

    Each individual assertion produces a detailed failure message identifying
    exactly which value was out of range and by how much.
    """

    # Bounds
    LAP_TIME_MIN = 45.0      # s
    LAP_TIME_MAX = 65.0      # s
    ENERGY_MIN_WH = 0.0      # Wh  (strict: must use *some* energy)
    ENERGY_MAX_WH = 500.0    # Wh  (1 lap, no regen)
    MAX_SPEED_MIN_KMH = 60.0 # km/h
    MAX_SPEED_MAX_KMH = 110.0# km/h
    BRAKE_MARGIN = 0.05      # 5 % over-decel tolerance

    # ------------------------------------------------------------------
    # Lap time
    # ------------------------------------------------------------------

    def test_lap_time_in_range(self, autocross_result: LapResult) -> None:
        t = autocross_result.lap_time
        print(f"\n[LapTime]   {t:.3f} s  (expected [{self.LAP_TIME_MIN}, {self.LAP_TIME_MAX}] s)")
        assert self.LAP_TIME_MIN <= t <= self.LAP_TIME_MAX, (
            f"Lap time out of range.\n"
            f"  Expected : [{self.LAP_TIME_MIN}, {self.LAP_TIME_MAX}] s\n"
            f"  Actual   : {t:.3f} s  "
            f"({'too fast by' if t < self.LAP_TIME_MIN else 'too slow by'} "
            f"{abs(t - (self.LAP_TIME_MIN if t < self.LAP_TIME_MIN else self.LAP_TIME_MAX)):.2f} s)"
        )

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def test_energy_consumed_in_range(self, autocross_result: LapResult) -> None:
        e = autocross_result.energy_consumed
        print(f"\n[Energy]    {e:.2f} Wh  (expected ({self.ENERGY_MIN_WH}, {self.ENERGY_MAX_WH}] Wh)")
        assert e > self.ENERGY_MIN_WH, (
            f"Energy consumed must be > 0 Wh.  Got {e:.4f} Wh."
        )
        assert e < self.ENERGY_MAX_WH, (
            f"Energy consumed exceeds single-lap limit.\n"
            f"  Limit  : {self.ENERGY_MAX_WH} Wh\n"
            f"  Actual : {e:.2f} Wh  (excess {e - self.ENERGY_MAX_WH:.2f} Wh)"
        )

    # ------------------------------------------------------------------
    # Max speed
    # ------------------------------------------------------------------

    def test_max_speed_in_range(self, autocross_result: LapResult) -> None:
        v_max_ms = autocross_result.max_speed
        v_max_kmh = v_max_ms * 3.6
        print(
            f"\n[MaxSpeed]  {v_max_kmh:.1f} km/h  "
            f"(expected [{self.MAX_SPEED_MIN_KMH}, {self.MAX_SPEED_MAX_KMH}] km/h)"
        )
        assert self.MAX_SPEED_MIN_KMH <= v_max_kmh <= self.MAX_SPEED_MAX_KMH, (
            f"Max speed out of range.\n"
            f"  Expected : [{self.MAX_SPEED_MIN_KMH}, {self.MAX_SPEED_MAX_KMH}] km/h\n"
            f"  Actual   : {v_max_kmh:.1f} km/h"
        )

    # ------------------------------------------------------------------
    # No negative speeds
    # ------------------------------------------------------------------

    def test_no_negative_speeds(self, autocross_result: LapResult) -> None:
        v = autocross_result.speed_profile
        min_v = float(np.min(v))
        print(f"\n[MinSpeed]  {min_v:.4f} m/s  (must be ≥ 0)")
        assert min_v >= 0.0, (
            f"Negative speed encountered in speed_profile.\n"
            f"  Min speed : {min_v:.4f} m/s\n"
            f"  At index  : {int(np.argmin(v))}"
        )

    # ------------------------------------------------------------------
    # Profile length matches track
    # ------------------------------------------------------------------

    def test_profile_length_matches_track(
        self, autocross_result: LapResult, autocross_track: Track
    ) -> None:
        n_track = len(autocross_track.distance)
        n_speed = len(autocross_result.speed_profile)
        print(f"\n[Lengths]   track={n_track}  speed_profile={n_speed}")
        assert n_speed == n_track, (
            f"speed_profile length mismatch.\n"
            f"  track.distance  : {n_track} points\n"
            f"  speed_profile   : {n_speed} points"
        )

    # ------------------------------------------------------------------
    # No NaN / Inf in speed_profile or ax_profile
    # ------------------------------------------------------------------

    def test_no_nan_or_inf_speed_profile(self, autocross_result: LapResult) -> None:
        v = autocross_result.speed_profile
        n_nan = int(np.sum(np.isnan(v)))
        n_inf = int(np.sum(np.isinf(v)))
        print(f"\n[SpeedNaN]  NaN={n_nan}  Inf={n_inf}  (both must be 0)")
        assert n_nan == 0, f"speed_profile contains {n_nan} NaN value(s)."
        assert n_inf == 0, f"speed_profile contains {n_inf} Inf value(s)."

    def test_no_nan_or_inf_ax_profile(self, autocross_result: LapResult) -> None:
        ax = autocross_result.ax_profile
        n_nan = int(np.sum(np.isnan(ax)))
        n_inf = int(np.sum(np.isinf(ax)))
        print(f"\n[AxNaN]     NaN={n_nan}  Inf={n_inf}  (both must be 0)")
        assert n_nan == 0, f"ax_profile contains {n_nan} NaN value(s)."
        assert n_inf == 0, f"ax_profile contains {n_inf} Inf value(s)."

    # ------------------------------------------------------------------
    # Braking deceleration never exceeds vehicle limit by > 5 %
    # ------------------------------------------------------------------

    def test_braking_deceleration_respected(
        self,
        autocross_result: LapResult,
        vehicle: VehicleDynamics,
    ) -> None:
        """Deceleration between consecutive points must not exceed
        ``vehicle.max_braking_deceleration(v)`` by more than 5 %."""
        v = autocross_result.speed_profile
        dist = autocross_result.distance

        violations: list[tuple[int, float, float]] = []  # (index, ax_actual, ax_limit)

        for i in range(len(v) - 1):
            vi = float(v[i])
            vi1 = float(v[i + 1])
            if vi1 >= vi:
                continue  # accelerating or constant — not a braking event

            ds = float(dist[i + 1] - dist[i])
            if ds <= 1e-9:
                continue

            # Kinematic deceleration from speed change (negative value)
            ax_actual = (vi1 ** 2 - vi ** 2) / (2.0 * ds)

            # Vehicle physics limit (positive value → negate for comparison)
            v_ref = max(vi, 0.5)          # guard against zero speed
            ax_limit = -vehicle.max_braking_deceleration(v_ref)

            # Allow BRAKE_MARGIN over-run before flagging
            if ax_actual < ax_limit * (1.0 + self.BRAKE_MARGIN):
                violations.append((i, ax_actual, ax_limit))

        n_viol = len(violations)
        if n_viol:
            worst_i, worst_ax, worst_lim = min(violations, key=lambda t: t[1] - t[2])
            over_pct = 100.0 * abs(worst_ax - worst_lim) / abs(worst_lim)
        else:
            over_pct = 0.0

        print(
            f"\n[Braking]   violations={n_viol}  "
            f"worst over-run={over_pct:.1f}%  "
            f"(tolerance {100*self.BRAKE_MARGIN:.0f}%)"
        )

        assert n_viol == 0, (
            f"{n_viol} segments exceeded max braking deceleration by >{100*self.BRAKE_MARGIN:.0f}%.\n"
            f"  Worst segment index : {worst_i}\n"
            f"  Actual decel        : {abs(worst_ax):.3f} m/s²\n"
            f"  Vehicle limit       : {abs(worst_lim):.3f} m/s²\n"
            f"  Over-run            : {over_pct:.1f}%"
        )


# ===========================================================================
# TEST 4 — Endurance State Carry-Over (5 laps)
# ===========================================================================

class TestEnduranceCarryOver:
    """5-lap endurance run must show correct physical state carry-over.

    Checks:
    * SOC decreases each lap (battery is being discharged).
    * total_energy = Σ per-lap energy, within 1 %.
    * total_time   = Σ per-lap lap_time, within 0.1 s.
    * Motor temperature rises over the run (thermal accumulation).
    * No individual lap time is pathologically fast or slow.
    """

    LAP_TIME_MIN = 30.0   # s — absolute lower bound (physically impossible to be faster)
    LAP_TIME_MAX = 120.0  # s — absolute upper bound (solver would be broken if slower)
    ENERGY_SUM_TOL = 0.01         # 1 % relative tolerance
    TIME_SUM_TOL_S = 0.1          # 0.1 s absolute tolerance

    # ------------------------------------------------------------------
    # SOC decreases lap-to-lap
    # ------------------------------------------------------------------

    def test_soc_decreases_each_lap(self, endurance_result: EnduranceResult) -> None:
        laps = endurance_result.laps
        soc_prev = 1.0
        first_violation: dict | None = None

        for i, lap in enumerate(laps):
            soc_now = lap.final_soc
            print(f"  Lap {i+1:2d}: SOC {soc_prev:.4f} → {soc_now:.4f}")
            if soc_now >= soc_prev and first_violation is None:
                first_violation = {"lap": i + 1, "soc_prev": soc_prev, "soc_now": soc_now}
            soc_prev = soc_now

        assert first_violation is None, (
            f"SOC did not decrease on lap {first_violation['lap']}.\n"
            f"  SOC at start of lap : {first_violation['soc_prev']:.4f}\n"
            f"  SOC at end of lap   : {first_violation['soc_now']:.4f}"
        )

    # ------------------------------------------------------------------
    # Total energy = sum of per-lap energy (within 1 %)
    # ------------------------------------------------------------------

    def test_total_energy_matches_sum(self, endurance_result: EnduranceResult) -> None:
        laps = endurance_result.laps
        sum_energy = sum(lap.energy_consumed for lap in laps)
        reported = endurance_result.total_energy
        rel_err = abs(reported - sum_energy) / max(sum_energy, 1e-9)

        print(
            f"\n[Energy]  sum_per_lap={sum_energy:.2f} Wh  "
            f"total={reported:.2f} Wh  "
            f"rel_err={100*rel_err:.3f}%"
        )

        assert rel_err <= self.ENERGY_SUM_TOL, (
            f"total_energy deviates from per-lap sum by more than 1 %.\n"
            f"  Σ per-lap energy : {sum_energy:.4f} Wh\n"
            f"  total_energy     : {reported:.4f} Wh\n"
            f"  Relative error   : {100*rel_err:.3f}%"
        )

    # ------------------------------------------------------------------
    # Total time = sum of per-lap times (within 0.1 s)
    # ------------------------------------------------------------------

    def test_total_time_matches_sum(self, endurance_result: EnduranceResult) -> None:
        laps = endurance_result.laps
        sum_time = sum(lap.lap_time for lap in laps)
        reported = endurance_result.total_time
        abs_err = abs(reported - sum_time)

        print(
            f"\n[Time]    sum_per_lap={sum_time:.4f} s  "
            f"total={reported:.4f} s  "
            f"abs_err={abs_err:.4f} s"
        )

        assert abs_err <= self.TIME_SUM_TOL_S, (
            f"total_time deviates from per-lap sum by more than {self.TIME_SUM_TOL_S} s.\n"
            f"  Σ per-lap time : {sum_time:.4f} s\n"
            f"  total_time     : {reported:.4f} s\n"
            f"  Absolute error : {abs_err:.4f} s"
        )

    # ------------------------------------------------------------------
    # Motor temperature rises over the run
    # ------------------------------------------------------------------

    def test_motor_temperature_rises(self, endurance_result: EnduranceResult) -> None:
        """Winding temperature at end of lap 5 must exceed starting temperature.

        Lap 1 starts from ambient (25 °C).  After 5 laps at sustained load
        the motor must have accumulated some heat.
        """
        laps = endurance_result.laps
        AMBIENT_TEMP_C = 25.0  # °C — default starting temperature in vehicle_params.yaml
        final_temp = laps[-1].final_motor_temp
        first_temp = laps[0].final_motor_temp   # already warmer than ambient after lap 1

        # We only need to prove the run warmed up from ambient
        print(
            f"\n[MotorTemp] ambient={AMBIENT_TEMP_C:.1f}°C  "
            f"after lap 1={first_temp:.2f}°C  "
            f"after lap 5={final_temp:.2f}°C"
        )

        assert final_temp > AMBIENT_TEMP_C, (
            f"Motor temperature did not rise above ambient after 5 laps.\n"
            f"  Ambient           : {AMBIENT_TEMP_C:.1f} °C\n"
            f"  Temp after lap 5  : {final_temp:.2f} °C"
        )

    # ------------------------------------------------------------------
    # Per-lap sanity bounds
    # ------------------------------------------------------------------

    def test_no_lap_outside_time_bounds(self, endurance_result: EnduranceResult) -> None:
        """Every individual lap time must be within physically sensible bounds."""
        laps = endurance_result.laps
        bad: list[tuple[int, float]] = []

        for i, lap in enumerate(laps):
            t = lap.lap_time
            print(f"  Lap {i+1:2d}: {t:.3f} s", end="")
            if not (self.LAP_TIME_MIN <= t <= self.LAP_TIME_MAX):
                bad.append((i + 1, t))
                print(f"  ← OUT OF RANGE", end="")
            print()

        assert not bad, (
            "The following laps have times outside "
            f"[{self.LAP_TIME_MIN}, {self.LAP_TIME_MAX}] s:\n"
            + "\n".join(
                f"  Lap {lap_no}: {t:.3f} s  "
                f"(delta: {'too fast' if t < self.LAP_TIME_MIN else 'too slow'} "
                f"by {abs(t - (self.LAP_TIME_MIN if t < self.LAP_TIME_MIN else self.LAP_TIME_MAX)):.2f} s)"
                for lap_no, t in bad
            )
        )
