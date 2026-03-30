import numpy as np
# from satellite.satellite import Satellite
from maths.earth_frame_conversions import *
from maths.quaternion import Quaternion
from constants import MAGNETIC_MOMENT_EARTH,R_EARTH

MAG_POLE_LAT = np.radians(-80.65)   # South
MAG_POLE_LON = np.radians(107.32)

def earth_magnetic_field(lat, lon, alt_km, time):
    """
    Return the earth magnetic field at a given position in space at a given time
    Magnetic field is in ECI Frame

    :param lat: latitude in radians
    :param lon:  longitude in radians
    :param alt_km: altitude in kilometers
    :param time: Datetime object
    :return: Earth magnetic field in ECI Frame in T
    """

    r_m = (R_EARTH + alt_km) * 1000.0

    coef = (10 ** (-7) * MAGNETIC_MOMENT_EARTH) / (r_m ** 3)

    # Dipole axis unit vector in ECEF (tilted magnetic pole)
    m_hat = np.array([
        np.cos(MAG_POLE_LAT) * np.cos(MAG_POLE_LON),
        np.cos(MAG_POLE_LAT) * np.sin(MAG_POLE_LON),
        np.sin(MAG_POLE_LAT)
    ])

    # Position unit vector in ECEF
    r_hat = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])

    # Dipole field formula: B = (μ₀/4π·r³)[3(m̂·r̂)r̂ - m̂]
    # coef already includes μ₀M/4π·r³
    B_ecef = coef * (3 * np.dot(m_hat, r_hat) * r_hat - m_hat)

    B_vector_eci = ecef_to_eci(B_ecef,time)

    return B_vector_eci
