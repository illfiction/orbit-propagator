import numpy as np
from orbit_propagator.core.ode_solvers import position_rk4_step, attitude_rk4_step

class Satellite:
    def __init__(self, initial_conditions, properties):

        # Setting up initial values from config.json file
        if initial_conditions['method'] == 'orbital_elements':
            print("Initializing state vectors from orbital elements...")
            initial_position, initial_velocity = orbital_elements_to_state_vectors(
                altitude_km=initial_conditions['altitude_km'],
                inclination_deg=initial_conditions['inclination_deg']
            )
            print(f"  -> Calculated Initial Position (km): {np.round(initial_position, 2)}")
            print(f"  -> Calculated Initial Velocity (km/s): {np.round(initial_velocity, 2)}")
        elif initial_conditions['method'] == 'state_vectors':
            print("Initializing state vectors directly from config.")
            initial_position = np.array(initial_conditions['position_km'])
            initial_velocity = np.array(initial_conditions['velocity_km_s'])
        else:
            # Raise an error if the method is not recognized
            raise ValueError(f"Invalid initial condition method specified in config: '{initial_conditions['method']}'")

        initial_quaternion = np.array(initial_conditions['quaternion'])
        initial_angular_velocity = np.array(initial_conditions['angular_velocity_rad_s'])

        self.translational = np.concatenate((initial_position, initial_velocity))   # 6D
        self.rotational    = np.concatenate((initial_quaternion, initial_angular_velocity))  # 7D

        self.state = np.concatenate((self.translational, self.rotational))

        self.J = np.array(properties['inertia_tensor'])
        self.J_inv = np.linalg.inv(self.J)
        self.dimensions = np.array(properties['dimensions_m'])
        self.c_srp = properties['srp_coefficient']

    def update(self, t, dt):
        self.translational = position_rk4_step(t, self, dt)
        self.rotational = attitude_rk4_step(t, self, dt)
        self.state = np.concatenate((self.translational, self.rotational))

    @property
    def position(self): return self.state[:3]

    @property
    def velocity(self): return self.state[3:6]

    @property
    def quaternion(self): return self.state[6:10]

    @property
    def angular_velocity(self): return self.state[10:13]
