import numpy as np
from satellite import Satellite
from visualize import plot_orbit
from constants import EARTH_MU, J

def run_simulation():
    initial_position = np.array([-4579.5, -4097.12, 3296.86])
    initial_velocity = np.array([-1.866, -3.234, -6.571])
    initial_quaternion = np.array([0.58,0.58,0,0.5720139858])
    initial_angular_velocity = np.array([0, 0, 0.0])

    satellite = Satellite(initial_position, initial_velocity, initial_quaternion, initial_angular_velocity)

    time_span = 1 * 24 * 60 * 3.95  # 90 days
    dt = 1
    steps = int(time_span / dt)

    state_list = []
    quaternion_list = []
    # energy_list = []

    for t in range(steps):
        satellite.update(t, dt)
        state_list.append(satellite.position.copy())
        quaternion_list.append(satellite.quaternion.copy())
        v = np.linalg.norm(satellite.velocity)
        r = np.linalg.norm(satellite.position)
        print("angular velocity = ", satellite.angular_velocity)
        # energy = 0.5 * v ** 2 - EARTH_MU / r + 0.5 * np.trace(J)
        # energy_list.append(energy)

    states = np.array(state_list)
    plot_orbit(states, quaternion_list)


if __name__ == '__main__':
    run_simulation()
