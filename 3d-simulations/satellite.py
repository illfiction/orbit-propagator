import numpy as np
from ode_solvers import position_rk4_step, attitude_rk4_step

class Satellite:
    def __init__(self, position, velocity, quaternion, angular_velocity):
        self.state = np.concatenate((position, velocity, quaternion, angular_velocity))
        self.translational = np.concatenate((position, velocity))   # 6D
        self.rotational    = np.concatenate((quaternion, angular_velocity))  # 7D

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
