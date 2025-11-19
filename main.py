import json
import sys
import os

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from satellite.satellite import Satellite
from visualisation.visualize import plot_orbit
from analysis.link_budget import time_over_ground_station
from analysis.groundstation import GroundStation


def _load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def _simulate_orbit(satellite, time_span_seconds, dt):
    steps = int(time_span_seconds / dt)
    positions = []
    quaternions = []

    print(
        f"Running simulation for {time_span_seconds / 86400:.2f} days with a timestep of {dt}s..."
    )

    for step, current_time in enumerate(tqdm(range(steps), desc="Simulating Orbit")):
        t_seconds = current_time * dt
        satellite.update(t_seconds, dt)
        positions.append(satellite.position.copy())
        quaternions.append(satellite.quaternion.copy())

    print("  Simulation 100% complete.")
    return np.array(positions), np.array(quaternions)


def run_simulation(config_path="config.json"):
    try:
        config = _load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {config_path}. Check for syntax errors."
        )
        return

    sim_params = config["simulation"]
    satellite = Satellite(config["initial_conditions"], config["satellite_properties"])

    time_span_seconds = sim_params["time_span_days"] * 24 * 60 * 60
    dt = sim_params["dt_seconds"]

    positions, quaternions = _simulate_orbit(satellite, time_span_seconds, dt)

    analysis_params = config.get("analysis", {})
    ground_station_params = config.get("ground_station")

    if analysis_params.get("run_ground_station_analysis"):
        if ground_station_params:
            ground_station = GroundStation.from_config(ground_station_params)
            print(f"Running ground station analysis for: {ground_station.name}")
            time_over_ground_station(
                position_list=positions,
                quaternion_list=quaternions,
                dt=dt,
                ground_station=ground_station,
                analysis_params=analysis_params,
            )
        else:
            print(
                "Warning: Ground station analysis is enabled, but no ground station data found in config.json."
            )

    print("\nGenerating 3D orbit visualization...")
    plot_orbit(positions, quaternions)


if __name__ == "__main__":
    # --- Build a robust path to the config file ---
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join that directory path with the config file name
    config_file_path = os.path.join(script_dir, "config.json")

    run_simulation(config_file_path)
