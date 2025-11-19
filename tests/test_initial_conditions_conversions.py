import numpy as np
from numpy.testing import assert_allclose

from constants import EARTH_MU, EARTH_RADIUS
from maths import initial_conditions_conversions as icc


def test_orbital_elements_to_state_vectors():
    altitude_km = 500.0
    inclination_deg = 60.0

    position, velocity = icc.orbital_elements_to_state_vectors(
        altitude_km=altitude_km, inclination_deg=inclination_deg
    )

    expected_radius = EARTH_RADIUS + altitude_km
    expected_speed = np.sqrt(EARTH_MU / expected_radius)

    assert_allclose(position, np.array([expected_radius, 0.0, 0.0]))

    inclination_rad = np.radians(inclination_deg)
    expected_velocity = np.array(
        [
            0.0,
            expected_speed * np.cos(inclination_rad),
            expected_speed * np.sin(inclination_rad),
        ]
    )
    assert_allclose(velocity, expected_velocity)
