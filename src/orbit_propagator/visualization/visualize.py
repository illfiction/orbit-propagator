import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from orbit_propagator.constants import EARTH_RADIUS

def plot_orbit(states, quaternion_list):
    orbit_trace = go.Scatter3d(
        x=states[:, 0], y=states[:, 1], z=states[:, 2],
        mode='lines',
        line=dict(width=2, color='yellow'),
        name='Orbit'
    )

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))

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

    L = 500
    colors = ['red', 'green', 'blue']
    step_interval = states.size // 5000

    for i in range(0, len(states), step_interval):
        pos = states[i]
        q = quaternion_list[i]
        q0, q1, q2, q3 = q

        R = np.array([
            [1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)],
            [2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1)],
            [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)]
        ])

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
