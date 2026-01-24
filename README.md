# orbit-propagator

## Usage
- `pip install -r requirements.txt` - installs all the requirements(libraries)
- `python main.py` – propagates the configured orbit, runs optional ground-station analysis, and opens the 3D visualisation.
- `python power_analysis.py` – propagates the orbit while computing solar-array power, reports key statistics, and plots the power-vs-time curve.

Tune `config.json` to adjust simulation, satellite, and `power_analysis` parameters (panel area, efficiency, losses, and body-frame normal).

## How to use

### Changing parameters

The parameters can be changed in the config.json file.

Simulation time and the step interval can be changed in the "simulation" section 

The initial conditions can be done in 2 ways:
1. Orbital elements- the altitude and inclination of orbit
2. State Vector- initial position and velocity
The initial quaternion and angular velocity in taken separately

The satellite properties and other parts are self understood.
