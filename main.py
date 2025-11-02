import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from orbit_propagator.OrbitPropagator import OrbitPropagator
from orbit_propagator.utils.initial_conditions_conversions import orbital_elements_to_state_vectors
from orbit_propagator.core.satellite import Satellite
from orbit_propagator.visualization.visualize import plot_orbit
from orbit_propagator.analysis.link_budget import time_over_ground_station
from orbit_propagator.analysis.groundstation import GroundStation

def run_simulation(config_path='config.json'):
    # Loading configs from config.json file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}. Check for syntax errors.")
        return

    orbit_propagator = OrbitPropagator(config)

    orbit_propagator.simulate()

    orbit_propagator.analyze()

    orbit_propagator.visualize()



if __name__ == '__main__':
    # --- Build a robust path to the config file ---
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join that directory path with the config file name
    config_file_path = os.path.join(script_dir, 'config.json')

    run_simulation(config_file_path)
