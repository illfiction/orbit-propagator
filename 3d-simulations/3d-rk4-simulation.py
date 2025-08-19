import numpy as np
import plotly.graph_objects as go
from quaternions import Quaternion
from maths import *
import plotly.io as pio

J2 = 1.08262668e-3
earth_radius = 6378.0 # km
earth_mu = 3.9860043543609598E+05 # km^3 / s^2
J = np.diag([1.0, 1.2, 1.0])
J_inverse = np.linalg.inv(J)


def position_ode(t, state):
    r = state[:3]

    a_kepler = -earth_mu * r / np.linalg.norm(r) ** 3

    a_j2 = j2_accel(r)

    a = a_kepler + a_j2

    return np.concatenate((state[3:6], a))

def j2_accel(r_vec):
    x, y, z = r_vec
    r2 = x*x + y*y + z*z
    r  = np.sqrt(r2)
    if r == 0:
        return np.zeros(3)

    z2 = z*z
    r5 = r2 * r2 * r
    k  = 1.5 * J2 * earth_mu * (earth_radius**2) / r5
    f  = 5.0 * z2 / r2  # = 5 (z/r)^2

    ax = k * x * (f - 1.0)
    ay = k * y * (f - 1.0)
    az = k * z * (f - 3.0)
    return np.array([ax, ay, az])


def position_rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * k1 * dt)
    k3 = f(t + 0.5 * dt, y + 0.5 * k2 * dt)
    k4 = f(t + dt, y + k3 * dt)

    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def attitude_ode(t, state):
    q = state[:4]
    w = state[4:]

    q_dot = 0.5 * Omega(w).dot(q)
    # print(q_dot)

    # TODO: Torque is taken as zero so add a way to calculate torque
    # TODO: add J_inverse also

    Torque = np.array([0, 0, 0])


    w_dot = J_inverse.dot(Torque - np.cross(w,np.dot(J,w)))

    return np.concatenate([q_dot, w_dot])



def attitude_rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * k1 * dt)
    k3 = f(t + 0.5 * dt, y + 0.5 * k2 * dt)
    k4 = f(t + dt, y + k3 * dt)

    y_next = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    y_next[:4] /= np.linalg.norm(y_next[:4])  # Normalize quaternion
    return y_next


class Satellite:
    def __init__(self, position, velocity, quaternion, angular_velocity):
        self.state = np.concatenate((position, velocity, quaternion, angular_velocity))

    def update(self, t, dt):
        self.state[:6] = position_rk4_step(position_ode, t, self.state[:6],dt)
        self.state[6:] = attitude_rk4_step(attitude_ode, t, self.state[6:],dt)

    @property
    def position(self):
        return self.state[:3]

    @property
    def velocity(self):
        return self.state[3:6]

    @property
    def quaternion(self):
        return self.state[6:10]

    @property
    def angular_velocity(self):
        return self.state[10:13]


# ---- Simulation ----
initial_position = np.array([earth_radius + 450, 0, 0])
initial_velocity = np.array([0, ( earth_mu / initial_position[0]) ** 0.5, 3])
initial_quaternion = np.array([1, 0, 0, 0])
initial_angular_velocity = np.array([0, 0, 0.03])
satellite = Satellite(initial_position, initial_velocity, initial_quaternion, initial_angular_velocity)
time_span = 500000
dt = 1
steps = int(time_span / dt)
state_list = []
quaternion_list = []
energy_list = []


for t in range(steps):
    satellite.update(t, dt)
    state_list.append(satellite.position.copy())
    quaternion_list.append(satellite.quaternion.copy())
    v = np.linalg.norm(satellite.velocity)
    r = np.linalg.norm(satellite.position)
    energy = 0.5 * v ** 2 - earth_mu / r
    energy_list.append(energy)
    # print(t,energy)

states = np.array(state_list)
# Satellite orbit line
orbit_trace = go.Scatter3d(
    x=states[:, 0],
    y=states[:, 1],
    z=states[:, 2],
    mode='lines',
    line=dict(width=2, color='yellow'),
    name='Orbit'
)

# Earth sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = earth_radius * np.outer(np.cos(u), np.sin(v))
y = earth_radius * np.outer(np.sin(u), np.sin(v))
z = earth_radius * np.outer(np.ones_like(u), np.cos(v))

earth_trace = go.Surface(
    x=x, y=y, z=z,
    colorscale='Blues',
    opacity=0.6,
    showscale=False,
    name='Earth'
)

fig = go.Figure(data=[orbit_trace, earth_trace])
fig.update_layout(
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        aspectmode='data'
    ),
    title='Interactive Orbit Visualization'
)
# Scale for body axes
L = 500  # adjust based on orbit size
colors = ['red', 'green', 'blue']

# Add attitude representation for a few sample points (to avoid clutter)
step_interval = 2000  # every 1000 steps
for i in range(0, len(states), step_interval):
    pos = states[i]
    q = quaternion_list[i]
    q0, q1, q2, q3 = q

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ])

    # Body axes
    for j in range(3):
        axis_end = pos + L * R[:, j]
        fig.add_trace(go.Scatter3d(
            x=[pos[0], axis_end[0]],
            y=[pos[1], axis_end[1]],
            z=[pos[2], axis_end[2]],
            mode='lines',
            line=dict(color=colors[j], width=5),
            showlegend=False
        ))

pio.renderers.default = 'browser'

fig.show()

# 9.188666918285625