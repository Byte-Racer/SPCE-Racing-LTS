"""
src/solver — Quasi-Steady-State Lap Time Solver package.

Public API::

    from src.solver import QSSSolver, LapResult, EnduranceResult

Author: Arceus, SPCE Racing
"""

from src.solver.qss_solver import QSSSolver
from src.solver.lap_results import LapResult, EnduranceResult
from src.solver.energy_tracker import EnergyTracker
from src.solver.speed_profile import SpeedProfileGenerator
from src.solver.acceleration_zones import AccelerationZones

__all__ = [
    "QSSSolver",
    "LapResult",
    "EnduranceResult",
    "EnergyTracker",
    "SpeedProfileGenerator",
    "AccelerationZones",
]
