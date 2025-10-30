import numpy as np
from tqdm import tqdm
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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

    sim_params = config['simulation']
    init_conds = config['initial_conditions']
    sat_properties = config['satellite_properties']
    analysis_params = config['analysis']
    ground_station_params = config.get('ground_station')        #used .get() here so if "ground_station does not exist program won't crash as it will return none if it does not get anything

    if init_conds['method'] == 'orbital_elements':
        print("Initializing state vectors from orbital elements...")
        initial_position, initial_velocity = orbital_elements_to_state_vectors(
            altitude_km=init_conds['altitude_km'],
            inclination_deg=init_conds['inclination_deg']
        )
        print(f"  -> Calculated Initial Position (km): {np.round(initial_position, 2)}")
        print(f"  -> Calculated Initial Velocity (km/s): {np.round(initial_velocity, 2)}")
    elif init_conds['method'] == 'state_vectors':
        print("Initializing state vectors directly from config.")
        initial_position = np.array(init_conds['position_km'])
        initial_velocity = np.array(init_conds['velocity_km_s'])
    else:
        # Raise an error if the method is not recognized
        raise ValueError(f"Invalid initial condition method specified in config: '{init_conds['method']}'")

    # Setting up initial values from config.json file
    initial_quaternion = np.array(init_conds['quaternion'])
    initial_angular_velocity = np.array(init_conds['angular_velocity_rad_s'])

    # Initializing satellite
    satellite = Satellite(initial_position, initial_velocity, initial_quaternion, initial_angular_velocity,sat_properties)

    # Setting up the simulation parameters
    time_span = sim_params['time_span_days'] * 24 * 60 * 60
    dt = sim_params['dt_seconds']
    steps = int(time_span / dt)

    print(f"Running simulation for {sim_params['time_span_days']} days with a timestep of {dt}s...")

    # List to store the position and quaternion values that will be calculated in the simulation
    position_list = []
    quaternion_list = []

    for t in tqdm(range(steps), desc="Simulating Orbit"):
        satellite.update(t, dt)
        position_list.append(satellite.position.copy())
        quaternion_list.append(satellite.quaternion.copy())

    print("  Simulation 100% complete.")

    # Converting the lists into np.arrays
    positions = np.array(position_list)
    quaternions = np.array(quaternion_list)

    if analysis_params['run_ground_station_analysis']:
        if ground_station_params:
            ground_station = GroundStation.from_config(ground_station_params)
            print(f"Running ground station analysis for: {ground_station.name}")
            time_over_ground_station(position_list=positions, quaternion_list=quaternions, dt=dt, ground_station=ground_station, analysis_params=analysis_params)
        else:
            print("Warning: Ground station analysis is enabled, but no ground station data found in config.json.")


    print("\nGenerating 3D orbit visualization...")
    plot_orbit(positions, quaternion_list)



if __name__ == '__main__':
    # --- Build a robust path to the config file ---
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join that directory path with the config file name
    config_file_path = os.path.join(script_dir, 'config.json')

    run_simulation(config_file_path)
