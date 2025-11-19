import numpy as np

from constants import EARTH_MU
from dynamics.atmospheric_drag import drag_acceleration
from dynamics.j2_acceleration import j2_accel
from dynamics.solar_radiation_force import solar_radiation_force
from maths.maths import Omega


def position_ode(t, state, sat):
    r = state[:3]
    a_kepler = -EARTH_MU * r / np.linalg.norm(r) ** 3
    a_j2 = j2_accel(r)
    a_solar_radiation = solar_radiation_force(t, r, sat)
    a_drag = drag_acceleration(r, state[3:6], sat)

    a = a_kepler + a_j2 + a_solar_radiation + a_drag
    return np.concatenate((state[3:6], a))


def attitude_ode(t, state, sat):
    q = state[:4]
    w = state[4:]
    q_dot = 0.5 * Omega(w).dot(q)
    Torque = np.zeros(3)
    w_dot = sat.J_inv.dot(Torque - np.cross(w, np.dot(sat.J, w)))
    return np.concatenate([q_dot, w_dot])


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * k1 * dt)
    k3 = f(t + 0.5 * dt, y + 0.5 * k2 * dt)
    k4 = f(t + dt, y + k3 * dt)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def position_rk4_step(t, sat, dt):
    f = lambda t_, y_: position_ode(t_, y_, sat)
    return rk4_step(f, t, sat.translational, dt)


def attitude_rk4_step(t, sat, dt):
    f = lambda t_, y_: attitude_ode(t_, y_, sat)
    y_next = rk4_step(f, t, sat.rotational, dt)
    y_next[:4] /= np.linalg.norm(y_next[:4])
    return y_next
