import os
import sys
import matplotlib.pyplot as plt

sys.path.append("c:\\Users\\garim\\OneDrive\\Desktop\\Chaos\\Custom LTS")

from src.vehicle.motor_inverter_model import MotorInverterModel

def test_endurance_simulation():
    # Initialize motor model
    motor_config = {
        'peak_torque': 220,
        'continuous_torque': 130,
        'peak_power': 62, # This is defined as kw in the yaml config, so pass 62
        'continuous_power': 64, # Defined as kw
        'max_rpm': 6500,
        'count': 1,
        'kt_constant': 0.61,
        'phase_resistance': 7.06,
        'kv_constant_peak': 8.68,
        'max_temperature_continuous': 100,
        'max_temperature_peak': 120
    }
    
    thermal_config = {
        'motor': {
            'thermal_mass': 5000,
            'cooling_coefficient': 50,
            'ambient_temp': 25
        },
        'inverter': {
            'thermal_mass': 2000,
            'cooling_coefficient': 40,
            'ambient_temp': 30
        }
    }
    
    motor = MotorInverterModel(motor_config, thermal_config)
    
    print("Simulating 5 minutes at 90% throttle, 5000 RPM...")
    result = motor.simulate_operation(
        throttle_percent=90,
        rpm=5000,
        voltage_dc=486,
        duration=300,  # 5 minutes
        dt=1.0  # 1 second time steps
    )
    
    print(f"\nResults:")
    print(f"  Average torque: {result['avg_torque']:.1f} Nm")
    print(f"  Average power: {result['avg_power']/1000:.1f} kW")
    print(f"  Energy consumed: {result['energy_consumed']:.2f} Wh")
    print(f"  Final motor temp: {result['final_motor_temp']:.1f}°C")
    print(f"  Thermal derating? {result['thermal_derating_occurred']}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    time = result['time_history']['time']
    torque = result['time_history']['torque']
    temp = result['time_history']['temperature']
    
    ax1.plot(time, torque, 'b-', linewidth=2)
    ax1.set_ylabel('Torque (Nm)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Motor Performance During Endurance', fontsize=13, fontweight='bold')
    
    ax2.plot(time, temp, 'r-', linewidth=2)
    ax2.axhline(motor.motor_temp_derate_start, color='orange', linestyle='--', label='Derate start')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Temperature (°C)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('c:\\Users\\garim\\OneDrive\\Desktop\\Chaos\\Custom LTS\\endurance_thermal_simulation.png', dpi=300)
    print("Saved plot to endurance_thermal_simulation.png")

if __name__ == "__main__":
    test_endurance_simulation()
