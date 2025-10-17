import numpy as np
import json
import os # Import the 'os' module
from satellite import Satellite
from visualize import plot_orbit
from link_budget import time_over_ground_station
from groundstation import GroundStation

def run_simulation(config_path='config.json'):
    # --- Load Configuration from JSON file ---
    with open(config_path, 'r') as f:
        config = json.load(f)

    sim_params = config['simulation']
    init_conds = config['initial_conditions']
    sat_properties = config['satellite_properties']
    analysis_params = config['analysis']
    # Load the new ground station section
    ground_station_params = config.get('ground_station')

    # --- Set Initial Conditions from Config ---
    initial_position = np.array(init_conds['position_km'])
    initial_velocity = np.array(init_conds['velocity_km_s'])
    initial_quaternion = np.array(init_conds['quaternion'])
    initial_angular_velocity = np.array(init_conds['angular_velocity_rad_s'])

    satellite = Satellite(initial_position, initial_velocity, initial_quaternion, initial_angular_velocity,sat_properties)

    # --- Set Simulation Time Parameters from Config ---
    time_span = sim_params['time_span_days'] * 24 * 60 * 60
    dt = sim_params['dt_seconds']
    steps = int(time_span / dt)

    print(f"Running simulation for {sim_params['time_span_days']} days with a timestep of {dt}s...")

    position_list = []
    quaternion_list = []
    # energy_list = []

    for t in range(steps):

        if t % (steps // 10) == 0:
            print(f"  Simulation is {int(t / steps * 100)}% complete...")


        satellite.update(t, dt)
        position_list.append(satellite.position.copy())
        quaternion_list.append(satellite.quaternion.copy())
        # v = np.linalg.norm(satellite.velocity)
        # r = np.linalg.norm(satellite.position)
        # print("angular velocity = ", satellite.angular_velocity)
        # energy = 0.5 * v ** 2 - EARTH_MU / r + 0.5 * np.trace(J)
        # energy_list.append(energy)

    print("  Simulation 100% complete.")

    states = np.array(position_list)

    if analysis_params['run_ground_station_analysis']:
        if ground_station_params:
            ground_station = GroundStation(ground_station_params['name'],ground_station_params['latitude_deg'],ground_station_params['longitude_deg'],ground_station_params['altitude_km'],ground_station_params['min_elevation_deg'])
            print(f"Running ground station analysis for: {ground_station.name}")
            time_over_ground_station(
                position_list=states,
                quaternion_list=np.array(quaternion_list),
                dt=dt,
                ground_station=ground_station,
                analysis_params=analysis_params
            )
        else:
            print("Warning: Ground station analysis is enabled, but no ground station data found in config.json.")


    print("\nGenerating 3D orbit visualization...")
    plot_orbit(states, quaternion_list)



if __name__ == '__main__':
    # --- Build a robust path to the config file ---
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join that directory path with the config file name
    config_file_path = os.path.join(script_dir, 'config.json')

    run_simulation(config_file_path)
