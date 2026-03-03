"""
Simulation Report Generator — orchestrates all plots, CSVs, and Excel.

Creates a timestamped output subfolder (``results/YYYY-MM-DD_HH-MM/``) and
delegates to :class:`~src.utils.telemetry_plotter.TelemetryPlotter`,
:class:`~src.utils.endurance_plotter.EndurancePlotter`,
:class:`~src.utils.track_plotter.TrackPlotter`, and
:class:`~src.utils.data_exporter.DataExporter`.

No regenerative braking is modelled in this simulator.

Author: Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from src.solver.lap_results import EnduranceResult, LapResult
from src.utils.data_exporter import DataExporter
from src.utils.endurance_plotter import EndurancePlotter
from src.utils.telemetry_plotter import TelemetryPlotter
from src.utils.track_plotter import TrackPlotter

logger = logging.getLogger(__name__)


class SimulationReportGenerator:
    """Orchestrate full simulation reporting for SPCE Racing LTS.

    Creates a timestamped subfolder ``results/<YYYY-MM-DD_HH-MM>/``
    and writes all figures, CSVs, and an Excel workbook there.

    Args:
        vehicle:     Fully initialised ``VehicleDynamics`` instance.
        track:       Immutable ``Track`` object used in the simulation.
        results_dir: Base directory for all outputs (default ``"results/"``).

    Example::

        from src.utils.report_generator import SimulationReportGenerator
        gen = SimulationReportGenerator(vehicle, track, "results/")
        paths = gen.generate_autocross_report(lap_result)
        print(paths)
    """

    def __init__(
        self,
        vehicle,
        track,
        results_dir: str = "results/",
    ) -> None:
        self.vehicle = vehicle
        self.track = track

        # Create timestamped output subfolder
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.out_dir = Path(results_dir) / ts
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._exporter = DataExporter(str(self.out_dir))

        logger.info(
            "SimulationReportGenerator initialised → %s", self.out_dir
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _p(self, filename: str) -> str:
        """Absolute path string inside the output directory."""
        return str(self.out_dir / filename)

    # ------------------------------------------------------------------ #
    #  Autocross (single-lap) report                                     #
    # ------------------------------------------------------------------ #

    def generate_autocross_report(self, lap_result: LapResult) -> Dict[str, str]:
        """Generate the full autocross (single-lap) report package.

        Runs
        ----
        * Comprehensive 5-panel telemetry figure
        * Track speed map coloured by lap speed
        * Lap telemetry CSV
        * Excel workbook (Lap Telemetry + Lap Summary sheets)
        * Console summary box

        Args:
            lap_result: Completed single-lap simulation result.

        Returns:
            Dictionary mapping output type keys to saved file path strings.
        """
        logger.info("Generating autocross report …")
        paths: Dict[str, str] = {}

        tel_plotter  = TelemetryPlotter(lap_result)
        trk_plotter  = TrackPlotter(self.track)

        # ── Telemetry figures ─────────────────────────────────────────
        paths["comprehensive_telemetry"] = self._p("telemetry_comprehensive.png")
        tel_plotter.plot_comprehensive_telemetry(
            save_path=paths["comprehensive_telemetry"]
        )

        paths["speed_trace"] = self._p("speed_trace.png")
        tel_plotter.plot_speed_trace(save_path=paths["speed_trace"])

        paths["acceleration_trace"] = self._p("acceleration_trace.png")
        tel_plotter.plot_acceleration_trace(save_path=paths["acceleration_trace"])

        paths["energy_trace"] = self._p("energy_trace.png")
        tel_plotter.plot_energy_trace(save_path=paths["energy_trace"])

        paths["motor_thermal"] = self._p("motor_thermal.png")
        tel_plotter.plot_motor_thermal(save_path=paths["motor_thermal"])

        paths["limiting_factors"] = self._p("limiting_factors.png")
        tel_plotter.plot_limiting_factors(save_path=paths["limiting_factors"])

        # ── Track map with speed ──────────────────────────────────────
        paths["track_speed_map"] = self._p("track_speed_map.png")
        trk_plotter.plot_track_with_speed(
            lap_result, save_path=paths["track_speed_map"]
        )

        paths["track_map"] = self._p("track_map.png")
        trk_plotter.plot_track_map(save_path=paths["track_map"])

        paths["curvature_profile"] = self._p("curvature_profile.png")
        trk_plotter.plot_curvature_profile(save_path=paths["curvature_profile"])

        # ── Data files ────────────────────────────────────────────────
        paths["lap_csv"] = self._exporter.export_lap_csv(
            lap_result, "lap_telemetry.csv"
        )

        try:
            paths["excel_report"] = self._exporter.export_excel_report(
                filename="autocross_report.xlsx",
                lap_result=lap_result,
            )
        except ImportError as exc:
            logger.warning("Excel export skipped: %s", exc)
            paths["excel_report"] = "skipped — openpyxl not installed"

        # ── Console summary ───────────────────────────────────────────
        self.print_console_summary(lap_result)

        logger.info("Autocross report complete → %s", self.out_dir)
        return paths

    # ------------------------------------------------------------------ #
    #  Endurance (multi-lap) report                                      #
    # ------------------------------------------------------------------ #

    def generate_endurance_report(
            self, endurance_result: EnduranceResult
    ) -> Dict[str, str]:
        """Generate the full endurance (multi-lap) report package.

        Runs
        ----
        * Comprehensive 2×2 endurance figure
        * Individual progression charts (lap time, energy, thermal, comparison)
        * Telemetry figures for laps 1, middle, and last
        * Track speed maps for lap 1 and last lap
        * Endurance CSV and lap telemetry CSVs (lap 1 / last)
        * Excel workbook (all sheets)
        * Console summary box

        Args:
            endurance_result: Completed endurance simulation container.

        Returns:
            Dictionary mapping output type keys to saved file path strings.
        """
        logger.info("Generating endurance report …")
        paths: Dict[str, str] = {}
        laps = endurance_result.laps
        n = len(laps)

        end_plotter = EndurancePlotter(endurance_result)
        trk_plotter = TrackPlotter(self.track)

        # ── Endurance overview figures ────────────────────────────────
        paths["comprehensive_endurance"] = self._p("endurance_comprehensive.png")
        end_plotter.plot_comprehensive_endurance(
            save_path=paths["comprehensive_endurance"]
        )

        paths["lap_time_progression"] = self._p("lap_time_progression.png")
        end_plotter.plot_lap_time_progression(
            save_path=paths["lap_time_progression"]
        )

        paths["energy_budget"] = self._p("energy_budget.png")
        end_plotter.plot_energy_budget(save_path=paths["energy_budget"])

        paths["thermal_progression"] = self._p("thermal_progression.png")
        end_plotter.plot_thermal_progression(save_path=paths["thermal_progression"])

        paths["lap_comparison"] = self._p("lap_comparison.png")
        end_plotter.plot_lap_comparison(save_path=paths["lap_comparison"])

        # ── Telemetry for laps 1 / mid / last ────────────────────────
        idx_first = 0
        idx_mid   = max(0, (n - 1) // 2)
        idx_last  = max(0, n - 1)

        for key, idx in [
            ("lap1_telemetry",  idx_first),
            ("mid_telemetry",   idx_mid),
            ("last_telemetry",  idx_last),
        ]:
            lap = laps[idx]
            tel = TelemetryPlotter(lap)
            fname = f"telemetry_lap{idx + 1:02d}.png"
            paths[key] = self._p(fname)
            tel.plot_comprehensive_telemetry(save_path=paths[key])

        # ── Track speed maps for lap 1 and last ──────────────────────
        paths["lap1_track_speed"] = self._p("track_speed_lap01.png")
        trk_plotter.plot_track_with_speed(
            laps[idx_first], save_path=paths["lap1_track_speed"]
        )

        paths["last_track_speed"] = self._p(f"track_speed_lap{idx_last + 1:02d}.png")
        trk_plotter.plot_track_with_speed(
            laps[idx_last], save_path=paths["last_track_speed"]
        )

        # ── Data files ────────────────────────────────────────────────
        paths["endurance_csv"] = self._exporter.export_endurance_csv(
            endurance_result, "endurance_results.csv"
        )

        paths["lap1_csv"] = self._exporter.export_lap_csv(
            laps[idx_first], "lap01_telemetry.csv"
        )
        paths["last_csv"] = self._exporter.export_lap_csv(
            laps[idx_last], f"lap{idx_last + 1:02d}_telemetry.csv"
        )

        try:
            paths["excel_report"] = self._exporter.export_excel_report(
                filename="endurance_report.xlsx",
                lap_result=laps[idx_first],
                endurance_result=endurance_result,
            )
        except ImportError as exc:
            logger.warning("Excel export skipped: %s", exc)
            paths["excel_report"] = "skipped — openpyxl not installed"

        # ── Console summary ───────────────────────────────────────────
        self.print_console_summary(endurance_result)

        logger.info("Endurance report complete → %s", self.out_dir)
        return paths

    # ------------------------------------------------------------------ #
    #  Console summary                                                    #
    # ------------------------------------------------------------------ #

    def print_console_summary(
            self, result: Union[LapResult, EnduranceResult]
    ) -> None:
        """Print a formatted box summary to stdout.

        Handles both :class:`~src.solver.lap_results.LapResult` and
        :class:`~src.solver.lap_results.EnduranceResult`.

        Args:
            result: Completed simulation result of either type.
        """
        W = 38   # inner width

        def _top()    -> str: return "╔" + "═" * W + "╗"
        def _sep()    -> str: return "╠" + "═" * W + "╣"
        def _bot()    -> str: return "╚" + "═" * W + "╝"
        def _row(s: str) -> str:
            return "║  " + s.ljust(W - 2) + "║"

        lines = [_top(), _row("  SPCE RACING — LTS RESULTS"), _sep()]

        if isinstance(result, LapResult):
            lap = result
            avg_kmh   = lap.avg_speed * 3.6
            derate_str = "Yes" if lap.thermal_derating_occurred else "No"

            lines += [
                _row(f"Lap Time    :  {lap.lap_time:.2f} s"),
                _row(f"Avg Speed   :  {avg_kmh:.1f} km/h"),
                _row(f"Energy Used :  {lap.energy_consumed:.0f} Wh"),
                _row(f"Final SOC   :  {lap.final_soc * 100:.1f} %"),
                _row(f"Motor Temp  :  {lap.final_motor_temp:.1f} \u00b0C"),
                _row(f"Derating    :  {derate_str}"),
            ]

        elif isinstance(result, EnduranceResult):
            r = result
            n_derating = sum(
                1 for lap in r.laps if lap.thermal_derating_occurred
            )
            lines += [
                _row(f"Laps        :  {len(r.laps)}"),
                _row(f"Total Time  :  {r.total_time:.1f} s  "
                     f"({r.total_time / 60:.1f} min)"),
                _row(f"Avg Lap     :  {r.avg_lap_time:.3f} s"),
                _row(f"Best Lap    :  {r.best_lap_time:.3f} s"),
                _row(f"Worst Lap   :  {r.worst_lap_time:.3f} s"),
                _sep(),
                _row(f"Total Energy:  {r.total_energy:.0f} Wh"),
                _row(f"Final SOC   :  {r.final_soc * 100:.1f} %"),
                _row(f"Derate Laps :  {n_derating}"),
            ]

        else:
            lines.append(_row("(unknown result type)"))

        lines.append(_bot())
        print("\n".join(lines))
