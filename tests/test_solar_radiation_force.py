import numpy as np
from types import SimpleNamespace

from constants import EARTH_RADIUS
from dynamics.solar_radiation_force import (
    solar_radiation_force,
    sun_direction_unit,
)


def make_satellite():
    return SimpleNamespace(
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        face_areas=np.array([0.01, 0.01, 0.01]),
        srp_specular=0.5,
        srp_diffuse=0.2,
        mass=10.0,
    )


def test_srp_zero_in_shadow():
    sat = make_satellite()
    sun_dir = sun_direction_unit(0.0)
    position = -sun_dir * (EARTH_RADIUS + 400.0)
    accel = solar_radiation_force(0.0, position, sat)
    assert np.allclose(accel, np.zeros(3))


def test_srp_nonzero_when_illuminated():
    sat = make_satellite()
    sun_dir = sun_direction_unit(0.0)
    position = sun_dir * (EARTH_RADIUS + 400.0)
    accel = solar_radiation_force(0.0, position, sat)
    assert np.linalg.norm(accel) > 0.0
    assert np.dot(accel, sun_dir) < 0.0
