"""
Test file for generating GGV diagrams - step 0/4 in the coding part of this project
(GGV) Performance Diagram Generator.

GGV diag - 3D performance map that shows the vehicle's acceleration capabilities across all speeds:
X-axis — Lateral acceleration *a_y* (cornering)
Y-axis— Longitudinal acceleration *a_x* (acceleration / braking)
Z-axis (or colour) — Velocity

At any operating point (V, a_y) the diagram answers: "What is the maximum longitudinal acceleration the car can achieve?"

Why it matters:
  1. Visualises the full performance envelope at every speed.
  2. Reveals the limiting factor (motor, battery, tyres, aero).
  3. Used by advanced lap-time simulators (e.g. OptimumLap).
  4. Enables setup optimisation (gear ratio, aero balance, weight dist.).

Arceus, SPCE Racing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

# Type alias for the vehicle dynamics model
from src.vehicle.vehicle_model import VehicleDynamics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------
G = 9.81  # m/s²


class GGVDiagramGenerator:
    """Compute and visualise g-g-V performance envelopes.

    Typical usage::

        from src.vehicle.vehicle_model import VehicleDynamics
        from src.utils.ggv_diagram import GGVDiagramGenerator

        vehicle = VehicleDynamics("config/vehicle_params.yaml")
        ggv = GGVDiagramGenerator(vehicle)
        ggv.compute_ggv_envelope()
        ggv.generate_comprehensive_report(save_path="results/ggv.png")

    Attributes:
        vehicle: The ``VehicleDynamics`` instance to evaluate.
        speed_points: 1-D array of test speeds (m/s).
        ay_points: 1-D array of lateral accelerations (m/s²).
        ggv_data: Computed envelope, shape ``(n_speeds, n_ay, 2)``.
            The last dimension stores ``[ax_forward, ax_brake]``.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
            self,
            vehicle: VehicleDynamics,
            speed_range: Tuple[float, float] = (0.5, 35.0),
            speed_resolution: int = 50,
    ) -> None:
        """Create a GGV generator around a vehicle model.

        Args:
            vehicle: Fully initialised ``VehicleDynamics`` instance.
            speed_range: ``(v_min, v_max)`` in **m/s**.
            speed_resolution: Number of speed sample points.
        """
        self.vehicle = vehicle

        # Speed array — start slightly above zero to avoid singularities
        self.speed_points: np.ndarray = np.linspace(
            max(speed_range[0], 0.5), speed_range[1], speed_resolution
        )

        # Lateral acceleration array (±3 g, denser near zero)
        self.ay_points: np.ndarray = np.concatenate(
            [
                np.linspace(-3.0 * G, -0.5 * G, 20),  # hard right
                np.linspace(-0.5 * G, 0.5 * G, 40),  # moderate
                np.linspace(0.5 * G, 3.0 * G, 20),  # hard left
            ]
        )

        # Will be populated by compute_ggv_envelope()
        self.ggv_data: Optional[np.ndarray] = None

        logger.info(
            "GGV generator initialised: %d speed pts × %d ay pts",
            len(self.speed_points),
            len(self.ay_points),
        )

    # ------------------------------------------------------------------ #
    #  Core computation                                                   #
    # ------------------------------------------------------------------ #

    def compute_ggv_envelope(self, mode: str = "peak") -> np.ndarray:
        """Compute the full GGV performance envelope.

        For every ``(speed, a_y)`` pair the method finds the maximum
        forward and braking longitudinal acceleration possible, accounting
        for tyre friction circles, motor / battery limits, and drag.

        Args:
            mode: ``'peak'`` or ``'continuous'`` motor / battery ratings.

        Returns:
            Array of shape ``(n_speeds, n_ay, 2)`` where the last axis
            is ``[ax_forward, ax_brake]`` in **m/s²**.
        """
        logger.info("Computing GGV envelope [%s mode] …", mode)

        n_spd = len(self.speed_points)
        n_ay = len(self.ay_points)
        envelope = np.zeros((n_spd, n_ay, 2))

        log_interval = max(1, n_spd // 10)

        for i, speed in enumerate(self.speed_points):
            if i % log_interval == 0:
                logger.info("  %3d / %d speeds  (%2.0f %%)", i, n_spd, 100.0 * i / n_spd)

            for j, ay in enumerate(self.ay_points):
                envelope[i, j, 0] = self._max_forward_accel(speed, ay, mode)
                envelope[i, j, 1] = self._max_braking_decel(speed, ay, mode)

        self.ggv_data = envelope
        logger.info("GGV envelope computed — shape %s", envelope.shape)
        return envelope

    # ------------------------------------------------------------------ #
    #  Internal physics — forward acceleration                            #
    # ------------------------------------------------------------------ #

    def _max_forward_accel(self, speed: float, ay: float, mode: str) -> float:
        """Max forward *a_x* at a given ``(speed, a_y)`` operating point.

        Steps
        -----
        1. Tyre normal loads  (static + aero + lateral weight transfer).
        2. Lateral force budget per tyre from ``F_y = m · a_y``.
        3. Remaining longitudinal budget via friction circle:
           ``F_x = √((μ F_z)² − F_y²)``.
        4. Motor-torque, motor-power, and battery-power limits → force.
        5. Subtract drag.  Return ``min(all limits) / m``.
        """
        veh = self.vehicle

        # ── 1. Tyre loads (weight transfer from cornering only) ──────
        loads = veh.calculate_weight_transfer(ax=0.0, ay=ay)
        aero = veh.calculate_aero_forces(speed)

        fz_f = (loads["front_total"] + aero["downforce_front"]) / 2.0  # per tyre
        fz_r = (loads["rear_total"] + aero["downforce_rear"]) / 2.0

        # ── 2. Required lateral force per tyre ───────────────────────
        fy_total = abs(veh.mass * ay)
        fy_f = fy_total * veh.weight_dist_front / 2.0  # per front tyre
        fy_r = fy_total * veh.weight_dist_rear / 2.0  # per rear tyre

        # ── 3. Remaining longitudinal grip (friction circle) ─────────
        cap_f = veh.calculate_tire_force_capacity(fz_f)
        cap_r = veh.calculate_tire_force_capacity(fz_r)

        fx_r_per_tyre = float(np.sqrt(max(0.0, cap_r["max_total_force"] ** 2 - fy_r ** 2)))
        traction_force = 2.0 * fx_r_per_tyre  # two rear (driven) tyres

        # ── 4a. Motor torque limit ───────────────────────────────────
        mt = veh.get_motor_torque(speed, mode)
        wt = mt * veh.gear_ratio * veh.drivetrain_efficiency
        motor_force = wt / veh.wheel_radius

        # ── 4b. Motor power limit (F = P / v) ───────────────────────
        mp = veh.get_motor_power(speed, mode)
        power_force = (mp / speed) if speed > 1.0 else 1e6

        # ── 4c. Battery power limit ──────────────────────────────────
        bp = veh.battery_peak_power if mode == "peak" else veh.battery_continuous_power
        bp_mech = bp * veh.motor_efficiency * veh.drivetrain_efficiency
        battery_force = (bp_mech / speed) if speed > 1.0 else 1e6

        # ── 5. Net force after drag ──────────────────────────────────
        available = min(traction_force, motor_force, power_force, battery_force)
        net = available - aero["drag"]

        return max(net / veh.mass, 0.0)

    # ------------------------------------------------------------------ #
    #  Internal physics — braking deceleration                            #
    # ------------------------------------------------------------------ #

    def _max_braking_decel(self, speed: float, ay: float, mode: str) -> float:
        """Max braking deceleration (positive number) at ``(speed, a_y)``.

        All four tyres contribute.  Regen is added on the rear axle but
        is conservatively reduced when cornering hard (> 1.5 g lateral).
        """
        veh = self.vehicle

        # Weight transfer estimate: moderate braking at −1 g
        loads = veh.calculate_weight_transfer(ax=-1.0 * G, ay=ay)
        aero = veh.calculate_aero_forces(speed)

        fz_f = (loads["front_total"] + aero["downforce_front"]) / 2.0
        fz_r = (loads["rear_total"] + aero["downforce_rear"]) / 2.0

        # Lateral force budget
        fy_total = abs(veh.mass * ay)
        fy_f = fy_total * veh.weight_dist_front / 2.0
        fy_r = fy_total * veh.weight_dist_rear / 2.0

        # Remaining braking grip (friction circle)
        cf = veh.calculate_tire_force_capacity(fz_f)
        cr = veh.calculate_tire_force_capacity(fz_r)

        fx_f = float(np.sqrt(max(0.0, cf["max_total_force"] ** 2 - fy_f ** 2)))
        fx_r = float(np.sqrt(max(0.0, cr["max_total_force"] ** 2 - fy_r ** 2)))

        mech_force = 2.0 * fx_f + 2.0 * fx_r  # all four tyres

        # Regen — reduce linearly when lateral-g exceeds 1.5 g
        regen_force = 0.0
        if veh.regen_enabled and speed > veh.regen_cutoff_speed:
            scale = float(np.clip(1.0 - abs(ay) / (1.5 * G), 0.0, 1.0))
            regen_wt = (
                    veh.regen_max_torque
                    * scale
                    * veh.motor_count
                    * veh.gear_ratio
                    * veh.drivetrain_efficiency
            )
            regen_f_torque = regen_wt / veh.wheel_radius
            regen_f_power = (veh.regen_max_power / speed) if speed > 1.0 else 1e6
            regen_force = min(regen_f_torque, regen_f_power)

        return (mech_force + regen_force) / veh.mass

    # ================================================================== #
    #  Plotting helpers                                                   #
    # ================================================================== #

    def _ensure_computed(self) -> None:
        """Raise if ``compute_ggv_envelope`` has not been called yet."""
        if self.ggv_data is None:
            raise RuntimeError(
                "Call compute_ggv_envelope() before generating plots."
            )

    # ------------------------------------------------------------------ #
    #  2-D g-g slice at a single speed                                    #
    # ------------------------------------------------------------------ #

    def plot_gg_diagram_at_speed(
            self, speed: float, ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot the classic g-g diagram (2-D slice at constant *V*).

        Args:
            speed: Target speed in **m/s**.
            ax: Existing ``Axes``; a new figure is created if *None*.

        Returns:
            The ``Axes`` that was drawn on.
        """
        self._ensure_computed()

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        # Closest speed index
        idx = int(np.argmin(np.abs(self.speed_points - speed)))
        v_actual = self.speed_points[idx]

        ay_g = self.ay_points / G
        ax_fwd_g = self.ggv_data[idx, :, 0] / G
        ax_brk_g = -self.ggv_data[idx, :, 1] / G  # negative for display

        # Acceleration envelope (upper)
        ax.plot(ay_g, ax_fwd_g, "b-", lw=2, label="Acceleration limit")
        ax.fill_between(ay_g, 0, ax_fwd_g, alpha=0.25, color="blue")

        # Braking envelope (lower)
        ax.plot(ay_g, ax_brk_g, "r-", lw=2, label="Braking limit")
        ax.fill_between(ay_g, 0, ax_brk_g, alpha=0.25, color="red")

        # Reference lines and grid
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Lateral Acceleration (g)", fontsize=12)
        ax.set_ylabel("Longitudinal Acceleration (g)", fontsize=12)
        ax.set_title(
            f"g-g Diagram at {v_actual:.1f} m/s ({v_actual * 3.6:.0f} km/h)",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim([-3, 3])
        ax.set_ylim([-2.5, 1.8])

        return ax

    # ------------------------------------------------------------------ #
    #  3-D GGV surface                                                    #
    # ------------------------------------------------------------------ #

    def plot_ggv_3d_surface(
            self, ax: Optional[Axes3D] = None  # type: ignore[assignment]
    ) -> Axes3D:
        """Full 3-D GGV surface: laterals vs speed vs longitudinals.

        Args:
            ax: Existing 3-D axes; creates a new figure if *None*.

        Returns:
            The 3-D ``Axes`` object.
        """
        self._ensure_computed()

        if ax is None:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")

        Ay, V = np.meshgrid(self.ay_points / G, self.speed_points * 3.6)
        Ax_fwd = self.ggv_data[:, :, 0] / G
        Ax_brk = -self.ggv_data[:, :, 1] / G

        ax.plot_surface(Ay, V, Ax_fwd, cmap="viridis", alpha=0.8)
        ax.plot_surface(Ay, V, Ax_brk, cmap="plasma", alpha=0.8)

        ax.set_xlabel("Lat. Accel (g)", fontsize=10)
        ax.set_ylabel("Speed (km/h)", fontsize=10)
        ax.set_zlabel("Long. Accel (g)", fontsize=10)
        ax.set_title(
            "GGV Diagram — Performance Envelope", fontsize=13, fontweight="bold"
        )

        return ax

    # ------------------------------------------------------------------ #
    #  2-D heatmap                                                        #
    # ------------------------------------------------------------------ #

    def plot_ggv_heatmap(
            self, accel_type: str = "forward"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """2-D heatmap of acceleration capacity vs speed and lateral g.

        Args:
            accel_type: ``'forward'`` or ``'braking'``.

        Returns:
            ``(figure, axes)`` tuple.
        """
        self._ensure_computed()

        fig, ax = plt.subplots(figsize=(12, 8))

        if accel_type == "forward":
            data = self.ggv_data[:, :, 0] / G
            title = "Max Forward Acceleration (g)"
            cmap = "YlOrRd"
        else:
            data = self.ggv_data[:, :, 1] / G
            title = "Max Braking Deceleration (g)"
            cmap = "YlGnBu"

        extent = [
            self.ay_points[0] / G,
            self.ay_points[-1] / G,
            self.speed_points[0] * 3.6,
            self.speed_points[-1] * 3.6,
        ]

        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap=cmap)

        # Contour overlay
        cs = ax.contour(
            self.ay_points / G,
            self.speed_points * 3.6,
            data,
            colors="black",
            linewidths=0.5,
            alpha=0.4,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f g")

        ax.set_xlabel("Lateral Acceleration (g)", fontsize=12)
        ax.set_ylabel("Vehicle Speed (km/h)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Acceleration (g)", fontsize=11)

        return fig, ax

    # ------------------------------------------------------------------ #
    #  Speed vs straight-line accel                                       #
    # ------------------------------------------------------------------ #

    def plot_speed_vs_max_accel(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot max straight-line acceleration & braking vs speed.

        Evaluates the GGV at *a_y = 0* to isolate pure longitudinal
        performance.

        Returns:
            ``(figure, axes)`` tuple.
        """
        self._ensure_computed()

        fig, ax = plt.subplots(figsize=(10, 6))

        ay0 = int(np.argmin(np.abs(self.ay_points)))
        fwd_g = self.ggv_data[:, ay0, 0] / G
        brk_g = self.ggv_data[:, ay0, 1] / G
        v_kmh = self.speed_points * 3.6

        ax.plot(v_kmh, fwd_g, "b-", lw=2, label="Max Acceleration")
        ax.plot(v_kmh, brk_g, "r-", lw=2, label="Max Braking")

        # Annotate peak acceleration
        peak = float(np.max(fwd_g))
        ax.annotate(
            f"Peak: {peak:.2f} g",
            xy=(v_kmh[np.argmax(fwd_g)], peak),
            xytext=(20, peak - 0.15),
            arrowprops=dict(arrowstyle="->", color="blue"),
            fontsize=10,
            color="blue",
        )

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
        ax.set_ylabel("Acceleration (g)", fontsize=12)
        ax.set_title(
            "Straight-line Acceleration vs Speed",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim([0, v_kmh[-1]])
        ax.set_ylim([0, max(peak, float(np.max(brk_g))) * 1.15])

        return fig, ax

    # ------------------------------------------------------------------ #
    #  Multi-speed g-g overlay                                            #
    # ------------------------------------------------------------------ #

    def plot_gg_multi_speed(
            self, speeds_kmh: Optional[List[float]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Overlay g-g diagrams at several speeds on a single plot.

        Args:
            speeds_kmh: List of speeds in km/h (default [20, 40, 60, 80, 100]).

        Returns:
            ``(figure, axes)`` tuple.
        """
        self._ensure_computed()

        if speeds_kmh is None:
            speeds_kmh = [20, 40, 60, 80, 100]

        fig, ax = plt.subplots(figsize=(9, 9))
        cmap_fn = cm.get_cmap("coolwarm", len(speeds_kmh))

        for k, v_kmh in enumerate(speeds_kmh):
            v_ms = v_kmh / 3.6
            idx = int(np.argmin(np.abs(self.speed_points - v_ms)))
            ay_g = self.ay_points / G
            fwd_g = self.ggv_data[idx, :, 0] / G
            brk_g = -self.ggv_data[idx, :, 1] / G

            colour = cmap_fn(k)
            ax.plot(ay_g, fwd_g, color=colour, lw=1.8, label=f"{v_kmh:.0f} km/h")
            ax.plot(ay_g, brk_g, color=colour, lw=1.8)

        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Lateral Acceleration (g)", fontsize=12)
        ax.set_ylabel("Longitudinal Acceleration (g)", fontsize=12)
        ax.set_title("g-g Envelope at Multiple Speeds", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim([-3, 3])
        ax.set_ylim([-2.5, 1.8])

        return fig, ax

    # ------------------------------------------------------------------ #
    #  Comprehensive report (6-panel figure)                              #
    # ------------------------------------------------------------------ #

    def generate_comprehensive_report(
            self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Generate a 6-panel GGV analysis figure.

        Panels
        ------
        1. g-g @ 20 km/h   2. g-g @ 60 km/h   3. g-g @ 100 km/h
        4. 3-D surface      5. Forward heatmap  6. Speed vs accel

        Args:
            save_path: If given, save figure to this path (PNG / PDF).

        Returns:
            The ``Figure`` object.
        """
        self._ensure_computed()

        fig = plt.figure(figsize=(20, 12))

        # ── Row 1: g-g slices at three speeds ────────────────────────
        for k, v_kmh in enumerate([20, 60, 100], start=1):
            ax_sub = fig.add_subplot(2, 3, k)
            self.plot_gg_diagram_at_speed(v_kmh / 3.6, ax=ax_sub)

        # ── Row 2 col 1: 3-D surface ────────────────────────────────
        ax4 = fig.add_subplot(2, 3, 4, projection="3d")
        self.plot_ggv_3d_surface(ax=ax4)

        # ── Row 2 col 2: forward-accel heatmap ──────────────────────
        ax5 = fig.add_subplot(2, 3, 5)
        data = self.ggv_data[:, :, 0] / G
        extent = [
            self.ay_points[0] / G,
            self.ay_points[-1] / G,
            self.speed_points[0] * 3.6,
            self.speed_points[-1] * 3.6,
        ]
        im = ax5.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="YlOrRd")
        ax5.set_xlabel("Lat. Accel (g)")
        ax5.set_ylabel("Speed (km/h)")
        ax5.set_title("Forward Accel Heatmap")
        fig.colorbar(im, ax=ax5, shrink=0.75, label="g")

        # ── Row 2 col 3: speed vs straight-line accel ────────────────
        ax6 = fig.add_subplot(2, 3, 6)
        ay0 = int(np.argmin(np.abs(self.ay_points)))
        ax6.plot(
            self.speed_points * 3.6,
            self.ggv_data[:, ay0, 0] / G,
            "b-", lw=2, label="Accel",
        )
        ax6.plot(
            self.speed_points * 3.6,
            self.ggv_data[:, ay0, 1] / G,
            "r-", lw=2, label="Braking",
        )
        ax6.grid(True, alpha=0.3)
        ax6.set_xlabel("Speed (km/h)")
        ax6.set_ylabel("Max Accel (g)")
        ax6.set_title("Straight-line Performance")
        ax6.legend(fontsize=9)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("GGV report saved → %s", save_path)

        return fig


# ====================================================================== #
#  Standalone entry-point                                                 #
# ====================================================================== #

def main() -> None:
    """Compute and display GGV analysis for the default vehicle config."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    config = "config/vehicle_params.yaml"
    if len(sys.argv) > 1:
        config = sys.argv[1]

    print(f"Loading vehicle → {config}")
    vehicle = VehicleDynamics(config)

    ggv = GGVDiagramGenerator(vehicle, speed_range=(0.5, 33.0), speed_resolution=40)

    print("Computing GGV envelope (this may take a minute) …")
    ggv.compute_ggv_envelope(mode="peak")

    print("Generating comprehensive report …")
    ggv.generate_comprehensive_report(save_path="results/ggv_analysis.png")

    # Additional standalone plots
    ggv.plot_gg_multi_speed()
    plt.savefig("results/gg_multi_speed.png", dpi=300, bbox_inches="tight")

    ggv.plot_speed_vs_max_accel()
    plt.savefig("results/accel_vs_speed.png", dpi=300, bbox_inches="tight")

    plt.show()
    print("✓ GGV analysis complete!")


if __name__ == "__main__":
    main()
