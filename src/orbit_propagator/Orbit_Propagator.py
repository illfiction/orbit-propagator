import numpy as np
from tqdm import tqdm

from orbit_propagator.core.satellite import Satellite

class OrbitPropagator:
    def __init__(self, config):

        sim_params = config['simulation']
        init_conds = config['initial_conditions']
        sat_properties = config['satellite_properties']
        analysis_params = config['analysis']
        ground_station_params = config.get('ground_station')  # used .get() here so if "ground_station does not exist program won't crash as it will return none if it does not get anything

        # Initializing satellite
        self.satellite = Satellite(init_conds, sat_properties)

        # Setting up the simulation parameters
        self.time_span = sim_params['time_span_days'] * 24 * 60 * 60
        self.dt = sim_params['dt_seconds']
        self.steps = int(self.time_span / self.dt)

        print(f"Running simulation for {sim_params['time_span_days']} days with a timestep of {self.dt}s...")


    def simulate(self):

        # List to store the position and quaternion values that will be calculated in the simulation
        position_list = []
        quaternion_list = []

        for t in tqdm(range(self.steps), desc="Simulating Orbit"):
            self.satellite.update(t, self.dt)
            position_list.append(self.satellite.position.copy())
            quaternion_list.append(self.satellite.quaternion.copy())

        print("  Simulation 100% complete.")

        # Converting the lists into np.arrays
        positions = np.array(position_list)
        quaternions = np.array(quaternion_list)

        return positions, quaternions
