import numpy as np

from constants import OMEGA_EARTH, R_EARTH,JULIAN_DATE_J2000
from maths.maths import rot_z
from astropy.time import Time
import astropy.units as u


def get_theta_gst(time: Time) -> float:
    """
    :param time:
    :return: theta_gst in radians
    """

    T_UT1 = (time.tt.jd - JULIAN_DATE_J2000) / 36525.0

    # 3. Calculate GMST in Seconds (IAU-82 Standard)
    gmst_seconds = 67310.54841 + (876600 * 3600 + 8640184.812866) * T_UT1 + \
                   0.093104 * T_UT1 ** 2 - 6.2e-6 * T_UT1 ** 3

    # 4. Modulo 86400 (seconds in a day) to wrap around
    gmst_seconds %= 86400.0

    # 5. Convert to Radians (Omega_earth * seconds)
    theta_rad = gmst_seconds * (2 * np.pi / 86400.0)

    return theta_rad


def ecef_to_eci(
    coordinate_vector_in_ecef: np.ndarray, t: Time, phi: float = 0
) -> np.ndarray:
    """
    Transforms ECEF(Earth Centered Earth Fixed) coordinate system to ECI(Earth-Centered Inertial) coordinate system.

    :param coordinate_vector_in_ecef: coordinates(x,y,z) in ECEF(Earth-Centered Earth-Fixed)
    :param t: time
    :param phi: angle in radians
    :return: coordinates(x,y,z) in ECI(Earth-Centered Inertial)
    """

    # passive transformation matrix from ECEF to ECI frame!
    R_ecef_to_eci = rot_z(get_theta_gst(t) + phi)

    coordinate_vector_in_eci = R_ecef_to_eci @ coordinate_vector_in_ecef
    return coordinate_vector_in_eci

def eci_to_ecef(coordinate_vector_in_eci: np.ndarray,t: Time, phi: float = 0) -> np.ndarray:
    """
    Transforms ECI coordinate system to ECEF coordinate system.

    :param coordinate_vector_in_eci: coordinates(x,y,z) in ECI(Earth-Centered Inertial)
    :param t: time in seconds
    :param phi: angle in radians
    :return: coordinates(x,y,z) in ECEF
    """

    R_eci_to_ecef = rot_z(get_theta_gst(t) + phi)

    coordinate_vector_in_ecef = R_eci_to_ecef @ coordinate_vector_in_eci
    return coordinate_vector_in_ecef



def geodetic_to_ecef(lat: float, lon: float, h: float = 0.0) -> np.ndarray:
    """
    Transforms Geodetic coordinate system to ECEF(Earth-Centered Earth-Fixed) coordinate system.
    Geodetic is lat lon alt
    Assuming earth is a sphere and not elllipsoid

    :param lat: latitude in radians
    :param lon: longitude in radians
    :param h: height in meters
    :return: coordinates(x,y,z) in ECEF(Earth-Centered Earth-Fixed)
    """

    x = (R_EARTH + h) * np.cos(lat) * np.cos(lon)
    y = (R_EARTH + h) * np.cos(lat) * np.sin(lon)
    z = (R_EARTH + h) * np.sin(lat)
    return np.array([x, y, z])

def ecef_to_geodetic(coordinate_vector_in_ecef: np.ndarray) -> tuple[float, float, float]:

    print("coordinate vector in ecef",coordinate_vector_in_ecef)
    x, y, z = map(float, coordinate_vector_in_ecef)

    print("x y z", x, y, z)
    lon = np.arctan2(y, x)

    r: float = float(np.linalg.norm([x, y, z]))
    lat = np.arcsin(np.clip(z / r, -1.0, 1.0))

    h = r - R_EARTH

    print("lat lon r",lat, lon, r)

    return lat, lon, h

def eci_to_geodetic(coordinate_vector_in_eci: np.ndarray ,t: Time) -> tuple[float, float, float]:
    return ecef_to_geodetic(eci_to_ecef(coordinate_vector_in_eci, t))

