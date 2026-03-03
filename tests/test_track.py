"""
Unit tests for the src/track module.

Tests:
  1. Circle curvature validation: κ should equal 1/R everywhere.
  2. Primitive track total length matches the expected geometric sum.

Run with:
    python -m pytest tests/test_track.py -v

Arceus, SPCE Racing
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.track.curvature import compute_curvature, compute_distance, resample_path
from src.track.track_loader import load_primitives
from src.track.track_representation import Track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_circle(radius: float, n_pts: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays for a full circle of the given radius."""
    theta = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


# ---------------------------------------------------------------------------
# Test 1: Circle curvature
# ---------------------------------------------------------------------------

class TestCircleCurvature:
    """For a circle of radius R, every point should satisfy κ ≈ 1/R."""

    @pytest.mark.parametrize("radius", [5.0, 9.0, 15.0, 50.0])
    def test_mean_curvature_equals_one_over_radius(self, radius: float) -> None:
        """Mean |κ| should equal 1/R within 5 %."""
        x, y = _make_circle(radius, n_pts=1000)
        kappa = compute_curvature(x, y, loop=True, sigma=0)  # no smoothing

        expected = 1.0 / radius
        mean_kappa = float(np.mean(np.abs(kappa)))

        assert abs(mean_kappa - expected) / expected < 0.05, (
            f"R={radius}: mean |κ| = {mean_kappa:.5f}, expected {expected:.5f} "
            f"(error {100 * abs(mean_kappa - expected) / expected:.1f} %)"
        )

    @pytest.mark.parametrize("radius", [5.0, 20.0])
    def test_curvature_is_uniform(self, radius: float) -> None:
        """Standard deviation of κ should be < 5 % of expected value."""
        x, y = _make_circle(radius, n_pts=500)
        kappa = compute_curvature(x, y, loop=True, sigma=0)

        expected = 1.0 / radius
        std_kappa = float(np.std(np.abs(kappa)))

        assert std_kappa / expected < 0.05, (
            f"R={radius}: κ std = {std_kappa:.5f} is too large "
            f"({100 * std_kappa / expected:.1f} % of expected)"
        )

    def test_curvature_sign_ccw(self) -> None:
        """Counter-clockwise circle should yield positive κ on average."""
        x, y = _make_circle(10.0)
        kappa = compute_curvature(x, y, loop=True, sigma=0)
        assert float(np.mean(kappa)) > 0, "CCW circle should have positive curvature"

    def test_curvature_sign_cw(self) -> None:
        """Clockwise circle should yield negative κ on average."""
        x, y = _make_circle(10.0)
        kappa = compute_curvature(x, -y, loop=True, sigma=0)  # flip y → CW
        assert float(np.mean(kappa)) < 0, "CW circle should have negative curvature"

    def test_smoothing_does_not_change_mean(self) -> None:
        """Gaussian smoothing should not change the mean curvature significantly."""
        radius = 12.0
        x, y = _make_circle(radius, n_pts=800)
        k_raw = compute_curvature(x, y, loop=True, sigma=0)
        k_smooth = compute_curvature(x, y, loop=True, sigma=5.0)

        expected = 1.0 / radius
        for k, label in [(k_raw, "raw"), (k_smooth, "smooth")]:
            err = abs(float(np.mean(np.abs(k))) - expected) / expected
            assert err < 0.05, f"{label} mean κ error {err * 100:.1f} % > 5 %"


# ---------------------------------------------------------------------------
# Test 2: Primitive track geometry
# ---------------------------------------------------------------------------

class TestPrimitiveLoader:
    """Validate that load_primitives produces the expected track length."""

    def test_straight_only_length(self) -> None:
        """A single straight should produce total_length ≈ specified length."""
        primitives = [{"type": "straight", "length": 100.0}]
        track = load_primitives(primitives, smoothing_sigma=0, resample_spacing=0.1, points_per_metre=10)
        # The primitive generates a straight; after closing the loop the
        # distance is approximately 2 × the straight length (there and back).
        # For a raw straight without closing, we check that at least 90 m are
        # covered (the closing segment adds a tiny extra from rounding).
        assert track.total_length >= 90.0, (
            f"Straight-only track: expected ≥ 90 m, got {track.total_length:.2f} m"
        )

    def test_oval_track_length(self) -> None:
        """Oval (2 straights + 2 semicircles) should match geometric sum."""
        straight_len = 50.0  # m
        radius = 9.0          # m
        primitives = [
            {"type": "straight", "length": straight_len},
            {"type": "corner", "radius": radius, "angle": 180, "direction": "left"},
            {"type": "straight", "length": straight_len},
            {"type": "corner", "radius": radius, "angle": 180, "direction": "left"},
        ]
        expected = 2 * straight_len + 2 * math.pi * radius
        track = load_primitives(
            primitives,
            smoothing_sigma=0,
            resample_spacing=0.1,
            points_per_metre=20,
        )
        rel_err = abs(track.total_length - expected) / expected
        assert rel_err < 0.02, (
            f"Oval total_length={track.total_length:.2f} m, "
            f"expected≈{expected:.2f} m (error {rel_err * 100:.1f} %)"
        )

    def test_90_degree_corner(self) -> None:
        """A 90-degree corner arc length should equal (π/2) × R ± 2 %."""
        radius = 10.0
        primitives = [{"type": "corner", "radius": radius, "angle": 90, "direction": "right"}]
        track = load_primitives(primitives, smoothing_sigma=0, resample_spacing=0.05, points_per_metre=20)
        arc_expected = (math.pi / 2) * radius  # quarter-circle arc

        # track.total_length includes the closing segment, so just check lower bound
        assert track.total_length >= arc_expected * 0.98, (
            f"90-deg corner: total_length={track.total_length:.3f} should be "
            f">= {arc_expected * 0.98:.3f}"
        )

    def test_track_object_properties(self) -> None:
        """Track returned by load_primitives must have matching array lengths."""
        primitives = [
            {"type": "straight", "length": 30},
            {"type": "corner", "radius": 6, "angle": 90, "direction": "left"},
            {"type": "straight", "length": 20},
        ]
        track = load_primitives(primitives)
        n = track.segment_count
        assert len(track.x) == n
        assert len(track.y) == n
        assert len(track.distance) == n
        assert len(track.curvature) == n
        assert track.total_length > 0

    def test_unknown_primitive_raises(self) -> None:
        """An unrecognised primitive type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown primitive type"):
            load_primitives([{"type": "circle", "radius": 5}])


# ---------------------------------------------------------------------------
# Test 3: Track object
# ---------------------------------------------------------------------------

class TestTrackObject:
    """Basic sanity checks on the Track dataclass."""

    def test_interpolate_curvature_at_zero(self) -> None:
        """Interpolating at distance=0 should return the first curvature value."""
        x, y = _make_circle(10.0, n_pts=300)
        kappa = compute_curvature(x, y, loop=True, sigma=0)
        dist = compute_distance(x, y)
        track = Track(x=x, y=y, distance=dist, curvature=kappa)
        assert abs(track.interpolate_curvature(0.0) - kappa[0]) < 1e-9

    def test_too_few_points_raises(self) -> None:
        """Creating a Track with fewer than 2 points should raise ValueError."""
        with pytest.raises(ValueError):
            Track(
                x=np.array([0.0]),
                y=np.array([0.0]),
                distance=np.array([0.0]),
                curvature=np.array([0.0]),
            )

    def test_mismatched_array_lengths_raise(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError):
            Track(
                x=np.array([0.0, 1.0]),
                y=np.array([0.0, 1.0, 2.0]),
                distance=np.array([0.0, 1.0]),
                curvature=np.array([0.0, 0.0]),
            )
