import numpy as np

from constants import OMEGA_EARTH, R_EARTH
from maths.maths import rot_z


def ecef_to_eci(
    coordinate_vector_in_ecef: np.ndarray, t: float, phi: float = 0
) -> np.ndarray:
    """
    Transforms ECEF(Earth Centered Earth Fixed) coordinate system to ECI(Earth-Centered Inertial) coordinate system.

    :param coordinate_vector_in_ecef: coordinates(x,y,z) in ECEF(Earth-Centered Earth-Fixed)
    :param t: time
    :param phi: angle in radians
    :return: coordinates(x,y,z) in ECI(Earth-Centered Inertial)
    """

    # passive transformation matrix from ECEF to ECI frame!
    R_ecef_to_eci = rot_z(-1 * OMEGA_EARTH * t + phi)

    coordinate_vector_in_eci = R_ecef_to_eci @ coordinate_vector_in_ecef
    return coordinate_vector_in_eci


def geodetic_to_ecef(lat: float, lon: float, h: float = 0.0) -> np.ndarray:
    """
    Transforms Geodetic coordinate system to ECEF(Earth-Centered Earth-Fixed) coordinate system.

    :param lat: latitude
    :param lon: longitude
    :param h: height in meters
    :return: coordinates(x,y,z) in ECEF(Earth-Centered Earth-Fixed)
    """

    x = (R_EARTH + h) * np.cos(lat) * np.cos(lon)
    y = (R_EARTH + h) * np.cos(lat) * np.sin(lon)
    z = (R_EARTH + h) * np.sin(lat)
    return np.array([x, y, z])
