import numpy as np
from constants import *


def orbital_elements_to_state_vectors(altitude_km: float, inclination_deg: float):

    orbit_radius_km = EARTH_RADIUS + altitude_km

    velocity_magnitude_km_s = np.sqrt(EARTH_MU / orbit_radius_km)

    inclination_radians = np.radians(inclination_deg)

    position = np.array([orbit_radius_km, 0, 0])

    velocity = np.array(
        [
            0,
            velocity_magnitude_km_s * np.cos(inclination_radians),
            velocity_magnitude_km_s * np.sin(inclination_radians),
        ]
    )

    return position, velocity
