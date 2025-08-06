import numpy as np
import matplotlib.pyplot as plt

mu = 1
dt = 0.001
Earth_Position = np.array([0, 0])

def two_body_ode(t, state):
    r = state[:2]
    a = -mu * r / np.linalg.norm(r) ** 3

    return np.concatenate((state[2:], a))


def rk4_step(f, t, y):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * k1 * dt)
    k3 = f(t + 0.5 * dt, y + k2 * dt)
    k4 = f(t + dt, y + k3 * dt)

    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class Satellite:
    def __init__(self,position,velocity):
        self.position = position
        self.velocity = velocity
        self.acceleration = 0

    # def get_distance(self,other_position):
    #     return np.linalg.norm(self.position - other_position)

    def update(self,t):
        current_state = np.concatenate((self.position, self.velocity))
        new_state = rk4_step(two_body_ode, t, current_state)
        self.position = new_state[:2]
        self.velocity = new_state[2:]


initial_pos = np.array([7.0, 0.0])
initial_vel = np.array([0.0, 0.35])


satellite = Satellite(initial_pos,initial_vel)

position_list = []
velocity_list = []

t = 0

while True:
    satellite.update(Earth_Position)
    position_list.append(satellite.position.copy())
    velocity_list.append(satellite.velocity.copy())
    t += dt
    if t > 1000:
        break

position_array = np.array(position_list)

x = position_array[:, 0]  # all rows, column 0
y = position_array[:, 1]  # all rows, column 1


plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Satellite position")
plt.show()