# FSEV Lap Time Simulator
Forged into existence by Arceus (GP)
A Python based quasi-steady-state lap time simulator for our racing team's Formula Student Electric vehicle. 

Built to predict lap times for dynamic events, model energy consumption, and test vehicle parameter changes.

## Background

We got tired of inaccurate simulation results :/

This simulator gives our racing team a data-driven tool to:
- Predict lap times for Formula Bharat dynamic events (Autocross, Endurance, Acceleration)
- Understand vehicle performance (motor, battery, tires, aero)
- Test parameter changes (gear ratio, aero setup, motor limits) before build

Built around the EMRAX 228 MV LC motor and a Molicel P45B 135s4p battery pack.

## Features

- **QSS Solver** — forward/backward integration pass algorithm for speed profile generation
- **Electro-thermal motor model** — torque delivery, thermal derating, field weakening, inverter losses
- **GGV diagram generation** — full 3D performance envelope across all speeds
- **Energy tracking** — SOC progression, voltage sag, battery power limits per segment
- **Multi-lap endurance simulation** — thermal and energy state carried across laps
- **Full visualization suite** — plots, track maps, endurance progression charts
- **Data export** — CSV telemetry and Excel reports with per-lap breakdown

## Project Structure
```
Custom LTS/
├── config/
│   ├── vehicle_params.yaml      # Single file for all vehicle parameters
│   ├── track_definitions.yaml   # Track layouts and waypoints
│   └── solver_config.yaml       # Solver settings and event configuration
├── src/
│   ├── vehicle/
│   │   ├── vehicle_model.py         # Full vehicle dynamics (tires, aero, weight transfer)
│   │   └── motor_inverter_model.py  # EMRAX 228 electro-thermal model
│   ├── track/
│   │   ├── track_loader.py          # CSV and primitive-based track loading
│   │   ├── track_representation.py  # Core Track object
│   │   ├── curvature.py             # Curvature computation from path
│   │   └── track_visualizer.py      # Track map plotting
│   ├── solver/
│   │   ├── qss_solver.py            # Main QSS lap time solver
│   │   ├── speed_profile.py         # Min-speed and integration passes
│   │   ├── energy_tracker.py        # SOC and battery state tracking
│   │   └── lap_results.py           # Results containers
│   └── utils/
│       ├── ggv_diagram.py           # GGV performance envelope generator
│       ├── telemetry_plotter.py     # Single-lap visualization
│       ├── endurance_plotter.py     # Multi-lap visualization
│       ├── track_plotter.py         # Track map with data overlay
│       ├── data_exporter.py         # CSV and Excel export
│       └── report_generator.py      # Full report generation entry point
├── tests/                       # Unit and integration tests
├── results/                     # Simulation output (plots, CSVs, Excel)
├── main.py                      # Integration smoke test / entry point
└── requirements.txt
```

## Installation

**Requirements:** Python 3.10+

Clone the repo and install dependencies:
```bash
git clone https://github.com/Byte-Racer/FSEV-LTS.git
cd FSEV-LTS
pip install -r requirements.txt
```

## Usage

**Set PYTHONPATH before running** (required for module imports):
```powershell
# Windows (PowerShell)
cd "C:\path\to\Custom LTS"
$env:PYTHONPATH = "C:\path\to\Custom LTS"
```

**Run the full simulation :**
```bash
python main.py
```

**Run autocross only (faster, skips endurance):**
```bash
python main.py --quick
```

**Run GGV diagram only:**
```bash
python -m src.utils.ggv_diagram
```

**Run tests:**
```bash
python -m pytest tests/ -v
```

Results are saved to `results/{YYYY-MM-DD_HH-MM}/` with plots, CSVs, and an Excel report.

## Module Overview

| Module | Purpose |
|---|---|
| `src.vehicle` | Tire model, powertrain, aerodynamics, full vehicle dynamics |
| `src.track` | Track loading, curvature computation, segment discretisation |
| `src.solver` | QSS lap time optimisation, energy tracking, multi-lap state |
| `src.utils` | Visualization, telemetry plots, GGV diagrams, data export |

## Vehicle Configuration

All vehicle parameters live in `config/vehicle_params.yaml`. 
Key specs :

| Parameter              | Value                |
| ---------------------- | -------------------- |
| Motor                  | EMRAX 228 MV LC      |
| Peak torque            | 220 Nm               |
| Peak power             | 62 kW                |
| Battery                | Molicel P45B, 135s4p |
| Pack voltage (nominal) | 486 V                |
| Total mass             | 300 kg               |

## Roadmap

- [ ] Validate lap times against OptimumLap predictions  
- [ ] DAQ telemetry overlay (compare simulation vs real car data)  
## Author

**Garima (Arceus)** — Electric Powertrain Department
