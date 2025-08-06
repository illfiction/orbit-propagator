import numpy as np
import matplotlib.pyplot as plt

dt = 0.00001
Mass_Earth = 1000
# Gravitational_const = 6.6743 * 10**(-11)
Gravitational_const = 1
Earth_Position = np.array([0, 0])

class Satellite:
    def __init__(self,position,velocity):
        self.position = position
        self.velocity = velocity

    def get_distance(self,position,other_position):
        distance = np.linalg.norm(position - other_position)
        return distance

    def update(self,position,velocity,acceleration):
        self.position = position + velocity * dt
        self.velocity = velocity + acceleration * dt


initial_pos = np.array([7,4])
initial_vel = np.array([-3,4])
# initial_pos = np.array(list(map(float, input("Enter initial position: ").split())))
# initial_vel = np.array(list(map(float, input("Enter initial velocity: ").split())))

satellite = Satellite(initial_pos,initial_vel)

position_list = []
velocity_list = []

t = 0

while True:
    distance = satellite.get_distance(satellite.position,Earth_Position)
    acceleration_magnitude = Gravitational_const * Mass_Earth / (distance**2)
    acceleration = - (acceleration_magnitude * satellite.position / distance)
    # print("acc",acceleration)
    # print(satellite.position)

    satellite.update(satellite.position,satellite.velocity,acceleration)
    position_list.append(satellite.position.copy())
    velocity_list.append(satellite.velocity.copy())
    t = t + dt
    if t > 10:
        break


position_array = np.array(position_list)

x = position_array[:, 0]  # all rows, column 0
y = position_array[:, 1]  # all rows, column 1


plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Satellite position")
plt.show()