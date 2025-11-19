import numpy as np
from types import SimpleNamespace

from constants import EARTH_RADIUS
from dynamics.atmospheric_drag import (
    atmospheric_density,
    drag_acceleration,
)


def make_satellite():
    return SimpleNamespace(
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        face_areas=np.array([0.01, 0.01, 0.01]),
        drag_coefficient=2.2,
        mass=10.0,
    )


def test_atmospheric_density_ground_level():
    density = atmospheric_density(np.array([6378.0, 0.0, 0.0]))
    assert density > 1.0


def test_atmospheric_density_high_altitude_zero():
    density = atmospheric_density(np.array([7000.0, 0.0, 0.0]))
    assert 0.0 < density < 1e-12


def test_drag_acceleration_opposes_velocity():
    sat = make_satellite()
    position = np.array([EARTH_RADIUS + 200.0, 0.0, 0.0])
    velocity = np.array([0.0, 7.5, 0.0])
    accel = drag_acceleration(position, velocity, sat)
    assert accel[1] < 0.0
