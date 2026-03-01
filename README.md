# SPCE EV Lap Time Simulator

A Python-based lap time simulation tool for SPCE Racing, created by Arceus. 
- Project aim : Build an accurate EV lap time simulator to predict lap times in dynamic events, test parameters by giving varied inputs
- Progress : Configuration part done, coding ongoing

## To run GGV file:
cd "C:\Users\garim\OneDrive\Desktop\Chaos\Custom LTS"
$env:PYTHONPATH = "C:\Users\garim\OneDrive\Desktop\Chaos\Custom LTS"
& "C:\Users\garim\AppData\Local\Programs\Python\Python313\python.exe" -m src.utils.ggv_diagram

## Installation
If ever published into a github repo, add steps

## Project Structure

Custom LTS/

- src/
    - vehicle/ — Vehicle dynamics models (tyres, powertrain, aero, chassis)
    - track/ — Track representation and geometry processing
    - solver/ — Lap time optimisation algorithms
    - utils/ — Plotting, data export, and helper functions
- config/ — YAML configuration files for vehicle & track parameters
- tests/ — Unit and integration tests
- requirements.txt
- README.md

## Modules

| Module        | Purpose                                                          |
| ------------- | ---------------------------------------------------------------- |
| `src.vehicle` | Tyre model, powertrain, aerodynamics, and full vehicle dynamics  |
| `src.track`   | Track loading, curvature computation, and segment discretisation |
| `src.solver`  | Quasi-steady-state and transient lap time optimisation           |
| `src.utils`   | Result visualisation, telemetry plots, and CSV/Excel export      |

## Dependencies

- **NumPy** — numerical arrays and linear algebra
- **SciPy** — optimisation solvers and interpolation
- **Matplotlib** — plotting and visualisation
- **PyYAML** — YAML configuration parsing
- **Pandas** — tabular data handling and export
- **CasADi** _(optional)_ — symbolic framework for advanced NLP-based optimisation
