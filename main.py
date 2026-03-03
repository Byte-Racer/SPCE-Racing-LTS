"""
main.py — Integration Smoke Test for the SPCE Racing LTS Simulator.

Wires together every major subsystem in the correct order and verifies
that data flows through without errors.  No new physics logic is added
here; this file is pure orchestration.

Usage
-----
    python main.py            # full run (autocross + 5-lap endurance)
    python main.py --quick    # autocross only (skips endurance)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — configure before any project imports so sub-module loggers
# inherit this handler automatically.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root — everything is resolved relative to this file's directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Project imports (after sys.path is implicitly correct because we run from
# the project root).
# ---------------------------------------------------------------------------
from src.vehicle.vehicle_model import VehicleDynamics          # noqa: E402
from src.track.track_loader import load_from_config            # noqa: E402
from src.solver.qss_solver import QSSSolver                    # noqa: E402
from src.utils.report_generator import SimulationReportGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elapsed(t0: float) -> str:
    """Return a human-readable elapsed time string since *t0*."""
    return f"{time.perf_counter() - t0:.1f} s"


def _step(label: str) -> float:
    """Log the start of a major step and return the start timestamp."""
    logger.info("── %s …", label)
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(quick: bool = False) -> None:
    """Run the end-to-end smoke test.

    Args:
        quick: When *True*, skip the endurance simulation and exit after
               the autocross report.
    """

    # ── 1. Load vehicle dynamics ──────────────────────────────────────────
    t0 = _step("Loading VehicleDynamics")
    vehicle_cfg = PROJECT_ROOT / "config" / "vehicle_params.yaml"
    vehicle = VehicleDynamics(str(vehicle_cfg))
    logger.info("VehicleDynamics loaded in %s", _elapsed(t0))

    # ── 2. Load track (fb_autocross from track_definitions.yaml) ─────────
    t0 = _step("Loading track  [fb_autocross]")
    track_cfg = PROJECT_ROOT / "config" / "track_definitions.yaml"
    track = load_from_config(str(track_cfg), track_name="fb_autocross")
    logger.info(
        "Track loaded in %s  (%.0f m, %d points)",
        _elapsed(t0),
        track.distance[-1],
        len(track.distance),
    )

    # ── 3. Build solver ───────────────────────────────────────────────────
    solver_cfg = PROJECT_ROOT / "config" / "solver_config.yaml"
    solver = QSSSolver(vehicle, str(solver_cfg))

    # ── 4. Autocross solve + report ───────────────────────────────────────
    t0 = _step("Solving autocross lap")
    autocross_result = solver.solve_autocross(track)
    elapsed_ac = _elapsed(t0)
    logger.info("Autocross solved in %s", elapsed_ac)
    print(f"Autocross solved in {elapsed_ac}")

    t0 = _step("Generating autocross report")
    reporter = SimulationReportGenerator(
        vehicle=vehicle,
        track=track,
        results_dir=str(PROJECT_ROOT / "results"),
    )
    reporter.generate_autocross_report(autocross_result)
    reporter.print_console_summary(autocross_result)
    logger.info("Autocross report written in %s", _elapsed(t0))

    if quick:
        logger.info("--quick flag set — skipping endurance simulation.")
        print("✓ Smoke test complete")
        return

    # ── 5. Endurance solve (5 laps) + report ─────────────────────────────
    t0 = _step("Solving endurance  [5 laps, state carry-over]")
    endurance_result = solver.solve_endurance(track, n_laps=5)
    elapsed_en = _elapsed(t0)
    logger.info("Endurance solved in %s", elapsed_en)
    print(f"Endurance solved in {elapsed_en}")

    t0 = _step("Generating endurance report")
    reporter.generate_endurance_report(endurance_result)
    reporter.print_console_summary(endurance_result)
    logger.info("Endurance report written in %s", _elapsed(t0))

    print("✓ Smoke test complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SPCE Racing LTS — integration smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the endurance simulation (autocross only).",
    )
    args = parser.parse_args()

    try:
        main(quick=args.quick)
    except Exception:  # noqa: BLE001
        print("\n── SMOKE TEST FAILED ──────────────────────────────────────────", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
