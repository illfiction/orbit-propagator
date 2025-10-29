import numpy as np
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from orbit_propagator.core.satellite import Satellite
from orbit_propagator.monte_carlo.carlo_gs_analysis import monte_carlo_time_over_ground_station
from orbit_propagator.analysis.groundstation import GroundStation



def run_monte_carlo_simulation(config_path='config.json'):
    
    # --- Load Configuration from JSON file ---
    with open(config_path, 'r') as f:
        config = json.load(f)

    sim_params = config['simulation']
    init_conds = config['initial_conditions']
    sat_properties = config['satellite_properties']
    analysis_params = config['analysis']
    # Load the new ground station section
    ground_station_params = config.get('ground_station')

    # --- Set Simulation Time Parameters from Config ---
    time_span = sim_params['time_span_days'] * 24 * 60 * 60
    dt = sim_params['dt_seconds']
    steps = int(time_span / dt)

    # --- Load the Angular Velocities from CSV file as a list of vectors ---
    angular_velocities = np.loadtxt('random_angular_velocities.csv', delimiter=',', skiprows=1)

    my_list = []

    print(f"Running simulation for {sim_params['time_span_days']} days with a timestep of {dt}s for the different values of angular velocities...")


    # --- Set Initial Conditions from Config and the Initial Angular Velocities List and Run the Loop ---
    initial_position = np.array(init_conds['position_km'])
    initial_velocity = np.array(init_conds['velocity_km_s'])
    initial_quaternion = np.array(init_conds['quaternion'])
    for idx, initial_angular_velocity in enumerate(angular_velocities):
            
        my_sat = Satellite(initial_position, initial_velocity, initial_quaternion, initial_angular_velocity,sat_properties)

        position_list = []
        quaternion_list = []

        for t in range(steps):

            my_sat.update(t, dt)
            position_list.append(my_sat.position.copy())
            quaternion_list.append(my_sat.quaternion.copy())

            if t % (steps // 10) == 0:
                print(f"\rSimulation is {int(t / steps * 100)}% complete for the {idx + 1}th angular velocity...", end = "", flush=True)


        if analysis_params['run_ground_station_analysis']:
            if ground_station_params:
                ground_station = GroundStation(ground_station_params['name'],ground_station_params['latitude_deg'],ground_station_params['longitude_deg'],ground_station_params['altitude_km'],ground_station_params['min_elevation_deg'])
                print(f"Running ground station analysis for: {ground_station.name} for the {idx + 1}th angular velocity...")

                my_tuple = time_over_ground_station(
                    position_list=np.array(position_list),
                    quaternion_list=np.array(quaternion_list),
                    dt=dt,
                    ground_station=ground_station,
                    analysis_params=analysis_params
                )

                my_list.append(my_tuple[0])
                
            else:
                print("Warning: Ground station analysis is enabled, but no ground station data found in config.json.")


    print("  Simulation 100% complete.")

    print("Here is the final list of average pointing intervals for different angular velocities:")
    print(my_list)
    print("The Average of the above list is:", np.mean(my_list) if my_list else 0)

    





if __name__ == '__main__':
    # --- Build a robust path to the config file ---
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join that directory path with the config file name
    config_file_path = os.path.join(script_dir, 'config.json')

    run_monte_carlo_simulation(config_file_path)
