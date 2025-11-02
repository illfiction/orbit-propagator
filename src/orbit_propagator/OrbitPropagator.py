import numpy as np
from tqdm import tqdm

from orbit_propagator.core.satellite import Satellite
from orbit_propagator.visualization.visualize import plot_orbit

class OrbitPropagator:
    def __init__(self, config):

        sim_params = config['simulation']
        init_conds = config['initial_conditions']
        sat_properties = config['satellite_properties']
        self.analysis_params = config['analysis']
        self.ground_station_params = config.get('ground_station')  # used .get() here so if "ground_station does not exist program won't crash as it will return none if it does not get anything

        # Initializing satellite
        self.satellite = Satellite(init_conds, sat_properties)

        # Setting up the simulation parameters
        self.time_span = sim_params['time_span_days'] * 24 * 60 * 60
        self.dt = sim_params['dt_seconds']
        self.steps = int(self.time_span / self.dt)

        print(f"Running simulation for {sim_params['time_span_days']} days with a timestep of {self.dt}s...")

        # List to store the position and quaternion values that will be calculated in the simulation
        self.position_list = []
        self.quaternion_list = []


    def simulate(self):
        """
        Simulate the propagation of the orbit

        :return: void
        """

        for t in tqdm(range(self.steps), desc="Simulating Orbit"):
            self.satellite.update(t, self.dt)
            self.position_list.append(self.satellite.position.copy())
            self.quaternion_list.append(self.satellite.quaternion.copy())

        print("  Simulation 100% complete.")

        # Converting the lists into np.arrays
        self.positions = np.array(self.position_list)
        self.quaternions = np.array(self.quaternion_list)

    def analyze(self):
        """
        Analyze the propagation of the orbit

        :return: void
        """

        if self.analysis_params['run_ground_station_analysis']:
            if self.ground_station_params:
                ground_station = GroundStation.from_config(self.ground_station_params)
                print(f"Running ground station analysis for: {ground_station.name}")
                time_over_ground_station(position_list=self.positions, quaternion_list=self.quaternions, dt=self.dt,
                                         ground_station=ground_station, analysis_params=self.analysis_params)
            else:
                print("Warning: Ground station analysis is enabled, but no ground station data found in config.json.")

    def visualize(self):
        """
        Visualize the propagation of the orbit

        :return: void
        """

        print("\nGenerating 3D orbit visualization...")
        plot_orbit(self.positions, self.quaternion_list)
