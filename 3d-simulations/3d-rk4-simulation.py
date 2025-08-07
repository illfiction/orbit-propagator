import numpy as np
import plotly.graph_objects as go
# import plotly.io as pio

earth_radius = 6378.0 # km
earth_mu = 3.9860043543609598E+05 # km^3 / s^2


def two_body_ode(t, state):
    r = state[:3]

    a = -earth_mu * r / np.linalg.norm(r) ** 3

    return np.concatenate((state[3:], a))

def rk4_step(f, t, y,dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * k1 * dt)
    k3 = f(t + 0.5 * dt, y + k2 * dt)
    k4 = f(t + dt, y + k3 * dt)

    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

class Satellite:
    def __init__(self, position, velocity):
        self.state = np.concatenate((position, velocity))

    def update(self, t, dt):
        self.state = rk4_step(two_body_ode, t, self.state,dt)

    @property
    def position(self):
        return self.state[:3]

    @property
    def velocity(self):
        return self.state[3:]


# ---- Simulation ----
initial_position = np.array([earth_radius + 450, 0, 0])
initial_velocity = np.array([0, ( earth_mu / initial_position[0] + 40) ** 0.5, 0])
satellite = Satellite(initial_position, initial_velocity)
time_span = 50000
dt = 0.1
steps = int(time_span / dt)
state_list = []
energy_list = []


for t in range(steps):
    satellite.update(t, dt)
    state_list.append(satellite.position.copy())
    v = np.linalg.norm(satellite.velocity)
    r = np.linalg.norm(satellite.position)
    energy = 0.5 * v ** 2 - earth_mu / r
    energy_list.append(energy)
    print(t,energy)

states = np.array(state_list)
# Satellite orbit line
orbit_trace = go.Scatter3d(
    x=states[:, 0],
    y=states[:, 1],
    z=states[:, 2],
    mode='lines',
    line=dict(width=2, color='red'),
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

# pio.renderers.default = 'browser'

fig.show()