"""
src/track — Track module for the SPCE Racing QSS Lap Time Simulator.

Public API::

    from src.track import Track, load_from_config, load_csv, load_primitives, TrackVisualizer

Arceus, SPCE Racing
"""

from src.track.track_representation import Track
from src.track.track_loader import load_from_config, load_csv, load_primitives
from src.track.track_visualizer import TrackVisualizer
from src.track.curvature import (
    compute_curvature,
    compute_distance,
    identify_corners,
    resample_path,
)

__all__ = [
    "Track",
    "load_from_config",
    "load_csv",
    "load_primitives",
    "TrackVisualizer",
    "compute_curvature",
    "compute_distance",
    "identify_corners",
    "resample_path",
]
