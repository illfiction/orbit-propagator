# orbit-propagator-3d

## Usage
- `python main.py` – propagates the configured orbit, runs optional ground-station analysis, and opens the 3D visualisation.
- `python power_analysis.py` – propagates the orbit while computing solar-array power, reports key statistics, and plots the power-vs-time curve.

Tune `config.json` to adjust simulation, satellite, and `power_analysis` parameters (panel area, efficiency, losses, and body-frame normal).