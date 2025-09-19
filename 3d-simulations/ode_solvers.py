import numpy as np
from j2_acceleration import j2_accel
from solar_radiation_force import solar_radiation_force
from solar_radiation_torque import solar_radiation_torque
from constants import EARTH_MU, J, J_INV
import sys, os

# go up one directory and into math_helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "math_helpers")))

from maths import Omega # now maths.py is available


def position_ode(t, state ,quaternion):
    r = state[:3]
    a_kepler = -EARTH_MU * r / np.linalg.norm(r) ** 3
    a_j2 = j2_accel(r)
    a_solar_radiation = solar_radiation_force(quaternion)

    a = a_kepler + a_j2 + a_solar_radiation
    return np.concatenate((state[3:6], a))

def attitude_ode(t, state, position):
    q = state[:4]
    w = state[4:]
    q_dot = 0.5 * Omega(w).dot(q)
    Torque = solar_radiation_torque(q)
    w_dot = J_INV.dot(Torque - np.cross(w, np.dot(J, w)))
    return np.concatenate([q_dot, w_dot])

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * k1 * dt)
    k3 = f(t + 0.5 * dt, y + 0.5 * k2 * dt)
    k4 = f(t + dt, y + k3 * dt)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def position_rk4_step(t, sat, dt):
    f = lambda t_, y_: position_ode(t_, y_, sat.rotational[:4])
    return rk4_step(f, t, sat.translational, dt)

def attitude_rk4_step(t, sat, dt):
    f = lambda t_, y_: attitude_ode(t_, y_, sat.translational[:3])
    y_next = rk4_step(f, t, sat.rotational, dt)
    y_next[:4] /= np.linalg.norm(y_next[:4])
    return y_next
