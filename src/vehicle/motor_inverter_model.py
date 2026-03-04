import logging
from typing import Dict

import numpy as np
from scipy.interpolate import RectBivariateSpline

logger = logging.getLogger(__name__)


class MotorInverterModel:
    """
    Detailed electro-thermal model of EMRAX 228 motor and inverter.

    Models:
    - Motor electromagnetic behavior (torque production)
    - Winding temperature dynamics (heating/cooling)
    - Thermal derating (power reduction at high temps)
    - Inverter switching losses
    - Field weakening control
    - Battery voltage effects

    This is a dynamic model - state evolves over time.
    Must be stepped forward using update() method.

    Attributes:
        motor_temp: Current motor winding temperature (°C)
        inverter_temp: Current inverter temperature (°C)
        field_weakening_active: Boolean flag
        available_torque: Current max torque accounting for all limits (Nm)
        actual_efficiency: Current motor+inverter efficiency (0-1)
    """

    def __init__(self, motor_config: dict, thermal_config: dict):
        """
        Initialize motor/inverter model from configuration.

        Args:
            motor_config: Motor parameters from vehicle_params.yaml
            thermal_config: Thermal model parameters (separate section)
        """
        self.logger = logging.getLogger(__name__)

        # === MOTOR ELECTRICAL PARAMETERS ===
        self.peak_torque = motor_config['peak_torque']  # Nm
        self.continuous_torque = motor_config['continuous_torque']  # Nm
        self.peak_power = motor_config['peak_power'] * 1000  # W
        self.continuous_power = motor_config['continuous_power'] * 1000  # W
        self.max_rpm = motor_config['max_rpm']
        self.motor_count = motor_config['count']

        # Electrical constants
        self.kt_constant = motor_config['kt_constant']  # Nm/A
        self.phase_resistance = motor_config['phase_resistance'] / 1000  # Convert mΩ to Ω
        self.kv_constant = motor_config['kv_constant_peak']  # RPM/V

        # === THERMAL PARAMETERS ===
        # Motor thermal model (simplified lumped capacitance)
        self.motor_thermal_mass = thermal_config['motor']['thermal_mass']  # J/K (heat capacity)
        self.motor_cooling_coeff = thermal_config['motor']['cooling_coefficient']  # W/K
        self.motor_ambient_temp = thermal_config['motor']['ambient_temp']  # °C

        # Temperature limits and derating
        self.motor_temp_continuous = motor_config.get('max_temperature_continuous', 100)  # °C (can run here indefinitely)
        self.motor_temp_peak = motor_config.get('max_temperature_peak', 120)  # °C (absolute limit, 2min max)
        self.motor_temp_derate_start = 90  # °C (start reducing power)
        self.motor_temp_derate_rate = 0.02  # 2% power loss per °C above derate_start

        # Inverter thermal model
        self.inverter_thermal_mass = thermal_config['inverter']['thermal_mass']  # J/K
        self.inverter_cooling_coeff = thermal_config['inverter']['cooling_coefficient']  # W/K
        self.inverter_ambient_temp = thermal_config['inverter']['ambient_temp']  # °C
        self.inverter_temp_limit = 85  # °C (IGBT junction temp limit)

        # === STATE VARIABLES ===
        # These evolve during simulation
        self.motor_temp = self.motor_ambient_temp  # °C
        self.inverter_temp = self.inverter_ambient_temp  # °C
        self.field_weakening_active = False

        # === EFFICIENCY MAPS ===
        # Create 2D efficiency map: efficiency(torque, rpm)
        self._create_efficiency_map()

        # === FIELD WEAKENING ===
        # RPM where constant torque region ends (voltage limit reached)
        # This depends on battery voltage
        self.base_speed_rpm = None  # Calculated dynamically

        self.logger.info(
            "✓ Motor-Inverter model initialized: "
            "Motor temp=%.1f°C, Inverter temp=%.1f°C",
            self.motor_temp, self.inverter_temp
        )

    def _create_efficiency_map(self):
        """
        Generate 2D efficiency map for EMRAX 228.

        Efficiency varies with operating point:
        - Low torque: Poor efficiency (~85-90%) due to core losses dominating
        - Medium torque, medium RPM: Peak efficiency (~94-96%)
        - High torque, low RPM: Good efficiency (~92-94%)
        - High torque, high RPM: Lower efficiency (~88-92%) due to iron losses

        Based on EMRAX 228 datasheet efficiency map (page 1 of PDF).

        Creates scipy.interpolate.RectBivariateSpline object for efficiency(torque_percent, rpm)
        """
        # Define grid points (torque as % of peak, RPM)
        torque_points = np.array([0, 25, 50, 75, 100, 125])  # % of peak torque
        rpm_points = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 6500])  # RPM

        # Efficiency values at each (torque, rpm) point
        # Based on EMRAX efficiency map from datasheet
        # Rows: torque %, Columns: RPM
        efficiency_data = np.array([
            # 0 RPM    1k     2k     3k     4k     5k     6k    6.5k
            [0.70, 0.80, 0.82, 0.83, 0.82, 0.80, 0.78, 0.75],  # 0% torque (friction/core losses)
            [0.85, 0.88, 0.90, 0.91, 0.90, 0.89, 0.87, 0.85],  # 25% torque
            [0.90, 0.92, 0.94, 0.95, 0.95, 0.94, 0.92, 0.90],  # 50% torque (sweet spot)
            [0.92, 0.94, 0.95, 0.96, 0.96, 0.95, 0.93, 0.91],  # 75% torque (peak efficiency region)
            [0.93, 0.94, 0.95, 0.96, 0.95, 0.94, 0.92, 0.90],  # 100% torque (peak)
            [0.91, 0.92, 0.93, 0.94, 0.93, 0.91, 0.89, 0.87],  # 125% torque (overload, reduced efficiency)
        ])

        # Create interpolation function
        self.efficiency_map = RectBivariateSpline(torque_points, rpm_points, efficiency_data)

        self.logger.debug("Motor efficiency map created (8x6 grid points)")

    def get_motor_efficiency(self, torque: float, rpm: float) -> float:
        """
        Look up motor efficiency at given operating point.

        Args:
            torque: Motor torque (Nm)
            rpm: Motor speed (RPM)

        Returns:
            Efficiency as fraction (0.85 to 0.96 typical)
        """
        # Convert torque to percentage of peak
        torque_percent = (torque / self.peak_torque) * 100

        # Clip to valid range
        torque_percent = np.clip(torque_percent, 0, 125)
        rpm = np.clip(rpm, 0, self.max_rpm)

        # Look up efficiency (returns array, extract scalar)
        efficiency = float(self.efficiency_map(torque_percent, rpm)[0, 0])

        return float(np.clip(efficiency, 0.7, 0.97))  # Sanity bounds

    def calculate_inverter_losses(self, current_rms: float, voltage_dc: float,
                                  switching_freq: float = 10000) -> Dict[str, float]:
        """
        Calculate inverter power losses (switching + conduction).

        Inverter converts DC from battery to 3-phase AC for motor.
        Losses occur in:
        1. Switching losses: Energy lost turning IGBTs on/off
        2. Conduction losses: I²R losses when IGBTs/diodes conduct

        Physics:
        - Switching loss per cycle: E_sw = 0.5 × V_dc × I × (t_rise + t_fall)
        - Switching power: P_sw = E_sw × f_switching
        - Conduction loss: P_cond = I² × R_on + V_f × I (IGBT + diode)

        Args:
            current_rms: Motor phase current RMS (A)
            voltage_dc: DC bus voltage (V)
            switching_freq: Inverter switching frequency (Hz, typically 8-15kHz)

        Returns:
            Dictionary with:
            - 'switching_loss': Switching losses (W)
            - 'conduction_loss': Conduction losses (W)
            - 'total_loss': Total inverter losses (W)
            - 'efficiency': Inverter efficiency (0-1)

        Notes:
        - Typical inverter efficiency: 95-98%
        - Losses increase with current and voltage
        - Higher switching frequency = more switching loss
        """
        # === IGBT PARAMETERS (typical for 600V, 300A IGBT module) ===
        # These would ideally come from actual inverter datasheet
        v_ce_sat = 2.0  # IGBT saturation voltage (V)
        v_f_diode = 1.5  # Diode forward voltage (V)
        r_on = 0.005  # On-state resistance (Ω)

        # Switching times (from IGBT datasheet)
        t_rise = 100e-9  # 100ns rise time
        t_fall = 150e-9  # 150ns fall time

        # === CONDUCTION LOSSES ===
        # Average current through IGBT (assuming sinusoidal, duty cycle ~0.5)
        i_avg = current_rms * (2 / np.pi)  # Average current

        # Loss in IGBT: P = V_ce × I_avg + I²_rms × R_on
        p_cond_igbt = v_ce_sat * i_avg + current_rms ** 2 * r_on

        # Loss in freewheeling diode (conducts during dead time)
        p_cond_diode = v_f_diode * i_avg * 0.3  # Diode conducts ~30% of time

        # Total conduction loss (3 phases, 2 switches per phase)
        p_conduction = (p_cond_igbt + p_cond_diode) * 6  # 6 switches total

        # === SWITCHING LOSSES ===
        # Energy lost per switching event
        e_on = 0.5 * voltage_dc * current_rms * t_rise
        e_off = 0.5 * voltage_dc * current_rms * t_fall
        e_switch = e_on + e_off

        # Power = Energy × Frequency × Number of switches
        # Each IGBT switches twice per cycle (on and off)
        p_switching = e_switch * switching_freq * 6  # 6 IGBTs

        # === TOTAL LOSSES ===
        p_total_loss = p_conduction + p_switching

        # === EFFICIENCY ===
        # Power into motor
        p_motor = np.sqrt(3) * voltage_dc * current_rms * 0.95  # Approximate motor power

        if p_motor > 0:
            efficiency = p_motor / (p_motor + p_total_loss)
        else:
            efficiency = 0.95  # Default

        return {
            'switching_loss': float(p_switching),
            'conduction_loss': float(p_conduction),
            'total_loss': float(p_total_loss),
            'efficiency': float(np.clip(efficiency, 0.85, 0.99))
        }

    def calculate_field_weakening(self, rpm: float, voltage_dc: float) -> Dict[str, float]:
        """
        Determine if field weakening is active and calculate available torque.

        FIELD WEAKENING EXPLAINED:
        Electric motors have a "base speed" where voltage limit is reached.
        Below base speed: Constant torque (limited by current)
        Above base speed: Constant power (limited by voltage)

        In constant power region, torque decreases: T = P / (RPM × 2π/60)

        This is achieved by "weakening" the magnetic field (reducing flux linkage)
        to allow higher speeds at the cost of lower torque.

        Physics:
        - Back-EMF voltage: V_bemf = Ke × RPM (Ke is voltage constant)
        - Controller needs: V_dc > V_bemf + I × R (overhead for control)
        - When V_bemf ≈ V_dc: can't increase current → torque limited by power

        Args:
            rpm: Motor speed (RPM)
            voltage_dc: DC bus voltage from battery (V)

        Returns:
            Dictionary with:
            - 'base_speed': Base speed at this voltage (RPM)
            - 'field_weakening': True if in FW region
            - 'torque_reduction_factor': Multiplier for available torque (0-1)
            - 'max_torque_fw': Maximum torque in field weakening (Nm)
        """
        # Voltage constant (back-EMF per RPM)
        # From EMRAX datasheet: Ke ≈ 0.61 Nm/A, and V/RPM relationship
        # Induced voltage: V_rms/RPM = 0.0479 V/RPM (from datasheet page 2)
        # Peak voltage: V_pk = V_rms × sqrt(2)
        # DC voltage needed: V_dc ≈ V_pk × sqrt(3) / modulation_index

        ke_line_to_line = 0.0479  # V_rms per RPM (from datasheet)
        modulation_index = 0.95  # PWM modulation depth (typical)
        voltage_overhead = 1.15  # Safety margin for control

        # Calculate base speed where voltage limit is reached
        # V_dc / voltage_overhead = ke × RPM × sqrt(3) / modulation_index
        base_speed = (voltage_dc / voltage_overhead) * modulation_index / (ke_line_to_line * np.sqrt(3))

        # Check if we're in field weakening region
        if rpm <= base_speed:
            # Constant torque region
            field_weakening = False
            torque_reduction = 1.0
            max_torque = self.peak_torque
        else:
            # Field weakening region - constant power
            field_weakening = True

            # Torque scales inversely with speed
            # T_fw = T_base × (base_speed / rpm)
            torque_reduction = base_speed / rpm

            # Also limited by absolute power limit
            # T_max = P_peak × 60 / (2π × RPM)
            power_limited_torque = (self.peak_power * 60) / (2 * np.pi * max(rpm, 1))

            max_torque = min(self.peak_torque * torque_reduction, power_limited_torque)

        return {
            'base_speed': float(base_speed),
            'field_weakening': field_weakening,
            'torque_reduction_factor': float(torque_reduction),
            'max_torque_fw': float(max_torque)
        }

    def calculate_thermal_derating(self) -> float:
        """
        Calculate power derating factor based on motor temperature.

        As motor heats up, must reduce power to prevent damage:
        - Below 90°C: No derating (100% power)
        - 90-100°C: Linear derating (2% per °C)
        - 100-120°C: Heavy derating (emergency, 5% per °C)
        - Above 120°C: Emergency shutdown (0% power)

        Returns:
            Power multiplier (0 to 1.0)
        """
        if self.motor_temp < self.motor_temp_derate_start:
            # Below derating threshold - full power
            return 1.0

        elif self.motor_temp < self.motor_temp_continuous:
            # Mild derating region (90-100°C)
            temp_above_threshold = self.motor_temp - self.motor_temp_derate_start
            derate = 1.0 - (self.motor_temp_derate_rate * temp_above_threshold)
            return max(0.5, derate)  # Never derate below 50%

        elif self.motor_temp < self.motor_temp_peak:
            # Heavy derating region (100-120°C)
            temp_above_continuous = self.motor_temp - self.motor_temp_continuous
            derate = 0.8 - (0.05 * temp_above_continuous)  # Start at 80%, lose 5% per °C
            return max(0.1, derate)  # Emergency minimum 10%

        else:
            # Thermal shutdown
            self.logger.warning(f"MOTOR THERMAL SHUTDOWN at {self.motor_temp:.1f}°C")
            return 0.0

    def get_available_torque(self, throttle_percent: float, rpm: float,
                             voltage_dc: float) -> Dict[str, float]:
        """
        Calculate actual available motor torque accounting for ALL limits.

        This is the main interface for the simulator.
        Returns torque that motor can actually produce right now.

        Considers:
        1. Throttle command (driver input)
        2. RPM-dependent torque limit (torque curve)
        3. Voltage-dependent limit (field weakening)
        4. Thermal derating (temperature protection)
        5. Peak vs continuous rating

        Args:
            throttle_percent: Driver throttle input (0-100%)
            rpm: Current motor speed (RPM)
            voltage_dc: Current battery voltage (V)

        Returns:
            Dictionary with:
            - 'requested_torque': What driver asked for (Nm)
            - 'available_torque': What motor can deliver (Nm)
            - 'limiting_factor': String describing what limits torque
            - 'field_weakening': Boolean
            - 'efficiency': Motor efficiency at this point
            - 'power_output': Mechanical power output (W)
        """
        # === STEP 1: CALCULATE BASE TORQUE LIMITS ===
        # Torque curve limit (from motor characteristics)
        if rpm < 4500:
            base_torque_limit = self.peak_torque  # Constant torque region
        else:
            # Power-limited region
            base_torque_limit = (self.peak_power * 60) / (2 * np.pi * max(rpm, 1))

        # === STEP 2: APPLY FIELD WEAKENING ===
        fw_result = self.calculate_field_weakening(rpm, voltage_dc)
        voltage_limited_torque = fw_result['max_torque_fw']

        # === STEP 3: APPLY THERMAL DERATING ===
        thermal_factor = self.calculate_thermal_derating()
        thermally_limited_torque = voltage_limited_torque * thermal_factor

        # === STEP 4: DRIVER REQUEST ===
        requested_torque = (throttle_percent / 100.0) * self.peak_torque

        # === STEP 5: FIND MINIMUM (MOST RESTRICTIVE) ===
        limits = {
            'torque_curve': base_torque_limit,
            'voltage': voltage_limited_torque,
            'thermal': thermally_limited_torque,
            'driver_request': requested_torque
        }

        available_torque = min(limits.values())

        # Identify which factor is limiting
        limiting_factor = min(limits, key=limits.get)

        # === STEP 6: CALCULATE EFFICIENCY AND POWER ===
        efficiency = self.get_motor_efficiency(available_torque, rpm)

        # Mechanical power output
        power_output = available_torque * (rpm * 2 * np.pi / 60)

        return {
            'requested_torque': float(requested_torque),
            'available_torque': float(available_torque),
            'limiting_factor': limiting_factor,
            'field_weakening': fw_result['field_weakening'],
            'efficiency': float(efficiency),
            'power_output': float(power_output),
            'thermal_factor': float(thermal_factor),
            'base_speed_rpm': fw_result['base_speed']
        }

    def update_thermal_state(self, power_loss: float, dt: float,
                             ambient_temp: float = None, airflow_speed: float = 0):
        """
        Update motor and inverter temperatures based on heat generation.

        Uses lumped capacitance thermal model:
        dT/dt = (Q_gen - Q_cool) / (m × c_p)

        Where:
        - Q_gen = Power losses (W)
        - Q_cool = h × A × (T - T_ambient) (convective cooling)
        - m × c_p = Thermal mass (J/K)

        Args:
            power_loss: Total power dissipated as heat (W)
            dt: Time step (seconds)
            ambient_temp: Ambient temperature (°C), overrides default if provided
            airflow_speed: Air speed past motor for cooling (m/s), increases cooling

        Updates:
            self.motor_temp (°C)
            self.inverter_temp (°C)
        """
        if ambient_temp is None:
            ambient_temp = self.motor_ambient_temp

        # === MOTOR HEATING ===
        # Power loss in motor = mechanical power / efficiency - mechanical power
        # Or equivalently: P_loss = P_electrical - P_mechanical
        # For now, use provided power_loss (from efficiency calculation)

        motor_heat_gen = power_loss  # W

        # Cooling power (liquid core + some forced convection from airflow on casing)
        # EMRAX is mostly internally liquid cooled. Surface air cooling at 15m/s adds ≈ 10% bonus.
        airflow_factor = 1.0 + (airflow_speed / 150.0)
        cooling_power = self.motor_cooling_coeff * airflow_factor * (self.motor_temp - ambient_temp)

        # Net heat rate
        q_net = motor_heat_gen - cooling_power

        # Temperature change: dT = Q × dt / (m × c_p)
        dT_motor = (q_net * dt) / self.motor_thermal_mass

        # Update temperature
        self.motor_temp += dT_motor

        # Can't go below ambient (unrealistic)
        self.motor_temp = max(self.motor_temp, ambient_temp)

        # === INVERTER HEATING ===
        # Inverter heats from its own losses (switching + conduction)
        # Estimate: inverter loss ≈ 3-5% of motor power
        inverter_heat_gen = power_loss * 0.2  # 20% of motor losses go to inverter

        inverter_cooling = self.inverter_cooling_coeff * (self.inverter_temp - ambient_temp)
        q_net_inv = inverter_heat_gen - inverter_cooling

        dT_inverter = (q_net_inv * dt) / self.inverter_thermal_mass
        self.inverter_temp += dT_inverter
        self.inverter_temp = max(self.inverter_temp, ambient_temp)

        # Thermal protection check
        if self.motor_temp > self.motor_temp_peak:
            self.logger.warning(f"⚠ Motor temperature critical: {self.motor_temp:.1f}°C")

        if self.inverter_temp > self.inverter_temp_limit:
            self.logger.warning(f"⚠ Inverter temperature critical: {self.inverter_temp:.1f}°C")

    def simulate_operation(self, throttle_percent: float, rpm: float,
                           voltage_dc: float, duration: float, dt: float = 0.1,
                           airflow_speed: float = 15.0) -> Dict:
        """
        Simulate motor operation over a time period.

        This is a dynamic simulation - state evolves over time.
        Use this for endurance event modeling where thermal effects matter.

        Args:
            throttle_percent: Throttle command (0-100%)
            rpm: Motor speed (RPM, assumed constant for this duration)
            voltage_dc: Battery voltage (V)
            duration: How long to run at these conditions (seconds)
            dt: Time step for thermal integration (seconds)

        Returns:
            Dictionary with:
            - 'avg_torque': Average torque over period (Nm)
            - 'avg_power': Average power output (W)
            - 'energy_consumed': Total energy from battery (Wh)
            - 'final_motor_temp': Motor temperature at end (°C)
            - 'thermal_derating_occurred': Boolean
            - 'time_history': Arrays of (time, torque, temp) for plotting
        """
        # Storage for history
        time_points = np.arange(0, duration, dt)
        torque_history = []
        temp_history = []
        power_history = []

        total_energy = 0  # Wh
        derating_flag = False

        for t in time_points:
            # Get current available torque
            result = self.get_available_torque(throttle_percent, rpm, voltage_dc)
            torque = result['available_torque']
            efficiency = result['efficiency']
            power_mech = result['power_output']

            # Check for derating
            if result['thermal_factor'] < 0.95:
                derating_flag = True

            # Electrical power consumption
            power_elec = power_mech / efficiency if efficiency > 0 else 0
            power_loss = power_elec - power_mech

            # Update thermal state
            self.update_thermal_state(power_loss, dt, airflow_speed=airflow_speed)

            # Accumulate energy
            energy_kwh = (power_elec * dt) / 3600  # Wh
            total_energy += energy_kwh

            # Store history
            torque_history.append(torque)
            temp_history.append(self.motor_temp)
            power_history.append(power_mech)

        return {
            'avg_torque': float(np.mean(torque_history)),
            'avg_power': float(np.mean(power_history)),
            'energy_consumed': float(total_energy),
            'final_motor_temp': float(self.motor_temp),
            'thermal_derating_occurred': derating_flag,
            'time_history': {
                'time': time_points.tolist(),
                'torque': torque_history,
                'temperature': temp_history,
                'power': power_history
            }
        }

    def get_wheel_torque_realtime(self, throttle: float, vehicle_speed: float,
                                  battery_voltage: float, gear_ratio: float,
                                  wheel_radius: float, dt: float = 0.01,
                                  update_thermal: bool = False) -> Dict:
        """
        Interface for real-time simulation - replaces simplified motor model.

        This method should be called by VehicleDynamics.max_longitudinal_acceleration()
        instead of using the simplified torque lookup.

        Args:
            throttle: Throttle input (0-100%)
            vehicle_speed: Vehicle speed (m/s)
            battery_voltage: Current battery voltage (V)
            gear_ratio: Final drive ratio
            wheel_radius: Wheel radius (m)
            dt: Time since last call (for thermal update)

        Returns:
            Dictionary with:
            - 'wheel_torque': Torque at wheels (Nm)
            - 'wheel_force': Force at ground (N)
            - 'motor_efficiency': Current efficiency
            - 'battery_power': Power draw from battery (W)
            - 'motor_temp': Current motor temperature
            - 'limiting_factor': What factor limited the torque calculation
        """
        # Convert vehicle speed to motor RPM
        wheel_rpm = (vehicle_speed / wheel_radius) * (60 / (2 * np.pi))
        motor_rpm = wheel_rpm * gear_ratio
        motor_rpm = min(motor_rpm, self.max_rpm)  # Clip to limit

        # Get available motor torque
        result = self.get_available_torque(throttle, motor_rpm, battery_voltage)
        motor_torque = result['available_torque'] * self.motor_count  # Total from all motors
        motor_efficiency = result['efficiency']

        # Convert to wheel torque
        wheel_torque = motor_torque * gear_ratio * 0.93  # Include drivetrain efficiency

        # Force at ground
        wheel_force = wheel_torque / wheel_radius

        # Battery power consumption
        power_mechanical = motor_torque * (motor_rpm * 2 * np.pi / 60)
        power_battery = power_mechanical / motor_efficiency if motor_efficiency > 0 else 0

        # Update thermal state
        power_loss = power_battery - power_mechanical
        if update_thermal:
            self.update_thermal_state(power_loss, dt, airflow_speed=vehicle_speed)

        return {
            'wheel_torque': float(wheel_torque),
            'wheel_force': float(wheel_force),
            'motor_efficiency': float(motor_efficiency),
            'battery_power': float(power_battery),
            'motor_temp': float(self.motor_temp),
            'limiting_factor': result['limiting_factor']
        }

    def plot_efficiency_map(self, save_path: str = None):
        """
        Visualize motor efficiency across operating range.

        Creates 2D heatmap: efficiency(torque, rpm)
        """
        import matplotlib.pyplot as plt

        # Create grid
        torque_range = np.linspace(0, self.peak_torque * 1.2, 50)
        rpm_range = np.linspace(0, self.max_rpm, 50)

        Torque, RPM = np.meshgrid(torque_range, rpm_range)
        Efficiency = np.zeros_like(Torque)

        # Calculate efficiency at each point
        for i in range(len(rpm_range)):
            for j in range(len(torque_range)):
                Efficiency[i, j] = self.get_motor_efficiency(Torque[i, j], RPM[i, j])

        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.contourf(RPM, Torque, Efficiency * 100, levels=15, cmap='RdYlGn')
        ax.contour(RPM, Torque, Efficiency * 100, levels=[90, 92, 94, 96],
                   colors='black', linewidths=0.5)

        ax.set_xlabel('Motor Speed (RPM)', fontsize=12)
        ax.set_ylabel('Motor Torque (Nm)', fontsize=12)
        ax.set_title('EMRAX 228 Efficiency Map', fontsize=14, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Efficiency (%)', fontsize=11)

        # Mark peak efficiency region
        ax.text(3000, self.peak_torque * 0.75, 'Peak Efficiency\n94-96%',
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    def plot_thermal_derating_curve(self, save_path: str = None):
        """
        Show how available power decreases with temperature.
        """
        import matplotlib.pyplot as plt

        temps = np.linspace(20, 130, 100)
        power_factors = []

        for temp in temps:
            self.motor_temp = temp
            factor = self.calculate_thermal_derating()
            power_factors.append(factor * 100)  # Convert to percentage

        # Reset to original temp
        self.motor_temp = self.motor_ambient_temp

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(temps, power_factors, 'r-', linewidth=2)
        ax.axvline(self.motor_temp_derate_start, color='orange', linestyle='--',
                   label='Derating starts')
        ax.axvline(self.motor_temp_continuous, color='red', linestyle='--',
                   label='Continuous limit')
        ax.axvline(self.motor_temp_peak, color='darkred', linestyle='--',
                   label='Peak limit')

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Motor Temperature (°C)', fontsize=12)
        ax.set_ylabel('Available Power (%)', fontsize=12)
        ax.set_title('Thermal Derating Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim([0, 105])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax


def example_endurance_simulation():
    """
    Simulate 20-minute endurance race showing thermal derating.
    """
    # Initialize motor model
    motor_config = {
        'peak_torque': 220,
        'continuous_torque': 130,
        'peak_power': 62,
        'continuous_power': 64,
        'max_rpm': 6500,
        'count': 1,
        'kt_constant': 0.61,
        'phase_resistance': 7.06,
        'kv_constant_peak': 8.68
    }
    
    thermal_config = {
        'motor': {
            'thermal_mass': 8100,  # J/K (13.5 kg motor mass * ~600 J/kg*K typical specific heat for copper+iron+housing)
            'cooling_coefficient': 85,  # W/K (Back-calc: P_loss ≈ 3720W, ΔT=45°C, h = 3720/45 ≈ 83 W/K, 85 used for margin)
            'ambient_temp': 25  # °C
        },
        'inverter': {
            'thermal_mass': 2000,  # J/K
            'cooling_coefficient': 40,  # W/K
            'ambient_temp': 30  # °C
        }
    }
    
    motor = MotorInverterModel(motor_config, thermal_config)
    
    # Simulate aggressive driving for 5 minutes
    print("Simulating 5 minutes at 90% throttle, 5000 RPM...")
    result = motor.simulate_operation(
        throttle_percent=90,
        rpm=5000,
        voltage_dc=486,
        duration=300,  # 5 minutes
        dt=1.0  # 1 second time steps
    )
    
    print(f"\\nResults:")
    print(f"  Average torque: {result['avg_torque']:.1f} Nm")
    print(f"  Average power: {result['avg_power']/1000:.1f} kW")
    print(f"  Energy consumed: {result['energy_consumed']:.2f} Wh")
    print(f"  Final motor temp: {result['final_motor_temp']:.1f}°C")
    print(f"  Thermal derating? {result['thermal_derating_occurred']}")

if __name__ == "__main__":
    example_endurance_simulation()
