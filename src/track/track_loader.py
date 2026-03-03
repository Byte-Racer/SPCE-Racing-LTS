"""
Track loading from CSV files and geometric primitives.

Supports two track-input formats:

1. **CSV** — columns ``(x, y)`` in metres, or ``(lat, lon)`` in decimal
   degrees (auto-detected; requires *pyproj* for UTM conversion).
2. **Primitives** — a list of dicts describing straights and corners by
   geometric parameters; the loader auto-generates (x, y) points and
   closes the loop.

The module also exposes a ``load_from_config`` entry-point that reads a
YAML block and dispatches to the appropriate back-end.

Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.track.curvature import (
    compute_curvature,
    compute_distance,
    identify_corners,
    resample_path,
)
from src.track.track_representation import Track

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry-points
# ---------------------------------------------------------------------------


def load_from_config(config_path: str, track_name: Optional[str] = None) -> Track:
    """Load a ``Track`` from a YAML config file.

    The YAML file should follow this schema::

        track_name: "FSAE_Autocross_2024"
        input_type: "csv"            # "csv" or "primitives"
        csv_path: "config/tracks/autocross.csv"
        smoothing_sigma: 2.0
        resample_spacing: 0.5        # metres between points

    Or for a primitives-based track::

        track_name: "test_oval"
        input_type: "primitives"
        smoothing_sigma: 1.5
        resample_spacing: 0.5
        primitives:
          - {type: straight, length: 50}
          - {type: corner, radius: 8, angle: 90, direction: left}

    The file may also use the multi-track ``tracks:`` list format that is
    present in the project's existing ``track_definitions.yaml``; in that
    case pass the desired ``track_name`` to select the entry.

    Args:
        config_path:  Path to the YAML file.
        track_name:   Name of the track to load when the file holds multiple
                      entries under a ``tracks:`` key.  Ignored for
                      single-track files.

    Returns:
        A fully initialised :class:`Track` object.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        KeyError:          If *track_name* is required but not found.
        ValueError:        If ``input_type`` is not recognised.
    """
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with cfg_file.open("r") as fh:
        raw = yaml.safe_load(fh)

    # ── Multi-track file (existing track_definitions.yaml format) ──────
    if "tracks" in raw:
        entries: List[Dict[str, Any]] = raw["tracks"]
        if track_name is None:
            cfg = entries[0]
            logger.info("load_from_config: using first track '%s'", cfg.get("name"))
        else:
            matches = [e for e in entries if e.get("name") == track_name]
            if not matches:
                available = [e.get("name") for e in entries]
                raise KeyError(
                    f"Track '{track_name}' not found. Available: {available}"
                )
            cfg = matches[0]

        # Convert waypoints list → CSV-style DataFrame and go through CSV path
        waypoints = cfg.get("waypoints", [])
        if isinstance(waypoints, str):
            # Special flag (e.g. "USE_AUTOCROSS_LAYOUT") — unsupported inline
            raise ValueError(
                f"Track '{track_name}' uses a redirect waypoints value "
                f"('{waypoints}'). Load the referenced track by name instead."
            )
        xs = [float(wp["x"]) for wp in waypoints]
        ys = [float(wp["y"]) for wp in waypoints]
        sigma = float(raw.get("smoothing_sigma", 2.0))
        spacing = float(raw.get("resample_spacing", 0.5))
        return _build_track(np.array(xs), np.array(ys), sigma=sigma, spacing=spacing, loop=True)

    # ── Single-track YAML ────────────────────────────────────────────────
    cfg = raw
    input_type: str = cfg.get("input_type", "csv").lower()
    sigma = float(cfg.get("smoothing_sigma", 2.0))
    spacing = float(cfg.get("resample_spacing", 0.5))

    if input_type == "csv":
        csv_path = cfg.get("csv_path")
        if not csv_path:
            raise ValueError("'csv_path' is required when input_type='csv'.")
        return load_csv(csv_path, smoothing_sigma=sigma, resample_spacing=spacing)

    elif input_type == "primitives":
        primitives = cfg.get("primitives", [])
        if not primitives:
            raise ValueError("'primitives' list is empty or missing.")
        return load_primitives(primitives, smoothing_sigma=sigma, resample_spacing=spacing)

    else:
        raise ValueError(f"Unknown input_type '{input_type}'. Use 'csv' or 'primitives'.")


def load_csv(
    csv_path: str,
    smoothing_sigma: float = 2.0,
    resample_spacing: float = 0.5,
    loop: bool = True,
) -> Track:
    """Load a ``Track`` from a CSV file of path coordinates.

    The file must contain either:
    * ``x, y`` columns — Cartesian coordinates in **metres**; or
    * ``lat, lon`` columns — decimal-degree GPS coordinates.  These are
      automatically converted to a local UTM frame (requires *pyproj*).

    Any additional columns (``width``, ``section``, ``radius``, …) are
    ignored.

    Args:
        csv_path:          Path to the CSV file.
        smoothing_sigma:   Gaussian smoothing sigma in **samples**.
        resample_spacing:  Target point spacing after resampling in **metres**.
        loop:              Treat the path as closed loop.

    Returns:
        A fully initialised :class:`Track`.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError:        If required columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(path)
    cols = {c.strip().lower() for c in df.columns}

    # ── GPS → UTM conversion ─────────────────────────────────────────────
    if "lat" in cols and "lon" in cols:
        logger.info("Detected lat/lon columns — converting to UTM (requires pyproj)")
        x_arr, y_arr = _latlon_to_utm(df["lat"].to_numpy(), df["lon"].to_numpy())
    elif "x" in cols and "y" in cols:
        df.columns = [c.strip().lower() for c in df.columns]
        x_arr = df["x"].to_numpy(dtype=float)
        y_arr = df["y"].to_numpy(dtype=float)
    else:
        raise ValueError(
            f"CSV '{csv_path}' must contain columns [x, y] or [lat, lon]. "
            f"Found: {list(df.columns)}"
        )

    return _build_track(x_arr, y_arr, sigma=smoothing_sigma, spacing=resample_spacing, loop=loop)


def load_primitives(
    primitives: List[Dict[str, Any]],
    smoothing_sigma: float = 1.5,
    resample_spacing: float = 0.5,
    points_per_metre: float = 4.0,
) -> Track:
    """Build a ``Track`` from a list of geometric primitive segments.

    Each element should be a dict with ``"type"`` set to either
    ``"straight"`` or ``"corner"``.

    **Straight**::

        {"type": "straight", "length": 50}  # length in metres

    **Corner**::

        {
          "type": "corner",
          "radius": 8,          # metres
          "angle": 90,          # degrees
          "direction": "left",  # "left" or "right"
        }

    The loader traces the vehicle heading through each primitive, closing
    the loop automatically (the last point is connected back to the first).

    Args:
        primitives:        Ordered list of primitive segment dicts.
        smoothing_sigma:   Gaussian smoothing sigma in **samples**.
        resample_spacing:  Target point spacing after resampling in **metres**.
        points_per_metre:  Density of intermediate points generated per metre
                           of arc length before resampling.

    Returns:
        A fully initialised :class:`Track`.

    Raises:
        ValueError: If a primitive ``"type"`` is not recognised.
    """
    xs: List[float] = [0.0]
    ys: List[float] = [0.0]
    heading = math.pi / 2.0  # start pointing in +Y direction (north)

    for seg in primitives:
        seg_type = str(seg.get("type", "")).lower()

        if seg_type == "straight":
            length = float(seg["length"])
            n_pts = max(2, int(math.ceil(length * points_per_metre)))
            for k in range(1, n_pts + 1):
                frac = k / n_pts
                xs.append(xs[-1] + math.cos(heading) * length * (1.0 / n_pts))
                ys.append(ys[-1] + math.sin(heading) * length * (1.0 / n_pts))

        elif seg_type == "corner":
            radius = float(seg["radius"])
            angle_deg = float(seg["angle"])
            direction = str(seg.get("direction", "left")).lower()

            # sign: +1 = CCW (left), -1 = CW (right)
            sign = 1.0 if direction == "left" else -1.0
            angle_rad = math.radians(angle_deg)
            n_pts = max(3, int(math.ceil(radius * angle_rad * points_per_metre)))

            # Centre of the arc — perpendicular to current heading
            cx = xs[-1] + sign * radius * math.cos(heading - math.pi / 2.0)
            cy = ys[-1] + sign * radius * math.sin(heading - math.pi / 2.0)

            # Angle from centre to start point
            theta0 = math.atan2(ys[-1] - cy, xs[-1] - cx)

            for k in range(1, n_pts + 1):
                frac = k / n_pts
                theta = theta0 + sign * angle_rad * frac
                xs.append(cx + radius * math.cos(theta))
                ys.append(cy + radius * math.sin(theta))

            # Update heading after the corner
            heading += sign * angle_rad

        else:
            raise ValueError(
                f"Unknown primitive type '{seg_type}'. "
                "Use 'straight' or 'corner'."
            )

    x_arr = np.array(xs)
    y_arr = np.array(ys)

    return _build_track(x_arr, y_arr, sigma=smoothing_sigma, spacing=resample_spacing, loop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_track(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
    spacing: float,
    loop: bool = True,
) -> Track:
    """Resample, compute curvature, identify corners, construct a ``Track``."""
    x_r, y_r = resample_path(x, y, spacing=spacing, loop=loop)
    dist = compute_distance(x_r, y_r)
    kappa = compute_curvature(x_r, y_r, loop=loop, sigma=sigma)
    corners = identify_corners(dist, kappa)

    return Track(
        x=x_r,
        y=y_r,
        distance=dist,
        curvature=kappa,
        corners=corners,
    )


def _latlon_to_utm(lat: np.ndarray, lon: np.ndarray):
    """Convert latitude/longitude arrays to a local UTM frame (metres).

    Requires **pyproj** (optional dependency).  The UTM zone is chosen
    automatically from the centroid of the supplied coordinates.

    Args:
        lat: Latitude values in decimal degrees.
        lon: Longitude values in decimal degrees.

    Returns:
        ``(easting, northing)`` arrays in **metres** with the first point
        as the local origin.

    Raises:
        ImportError: If *pyproj* is not installed.
    """
    try:
        from pyproj import Proj  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyproj is required for GPS (lat/lon) track loading. "
            "Install it with:  pip install pyproj"
        ) from exc

    mean_lon = float(np.mean(lon))
    zone = int((mean_lon + 180) / 6) + 1
    proj = Proj(proj="utm", zone=zone, ellps="WGS84", preserve_units=False)

    easting, northing = proj(lon, lat)
    easting = np.asarray(easting, dtype=float) - easting[0]
    northing = np.asarray(northing, dtype=float) - northing[0]
    return easting, northing
