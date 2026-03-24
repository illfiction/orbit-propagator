import numpy as np
# from satellite.satellite import Satellite
from maths.earth_frame_conversions import *
from maths.quaternion import Quaternion
from constants import MAGNETIC_MOMENT_EARTH,R_EARTH

def earth_magnetic_field(lat, lon, alt_km, time):
    """
    Return the earth magnetic field at a given position in space at a given time
    Magnetic field is in ECI Frame

    :param lat: latitude in radians
    :param lon:  longitude in radians
    :param alt_km: altitude in kilometers
    :param time: Datetime object
    :return: Earth magnetic field in ECI Frame in nT
    """

    r_m = (R_EARTH + alt_km) * 1000.0

    coef = (10 ** (2) * MAGNETIC_MOMENT_EARTH) / (r_m ** 3)

    Bn_nT = coef * np.cos(lat)
    Bu_nT = -2 * coef * np.sin(lat)
    Be_nT = 0.0

    B_vector_enu = np.array([Be_nT, Bn_nT, Bu_nT])


    print("B_vector_ned = ", B_vector_enu)

    print(f"DEBUG - lat: {lat}, alt_km: {alt_km}, M: {MAGNETIC_MOMENT_EARTH}, R: {R_EARTH}")

    R = np.array([
        [-np.sin(lon), -np.cos(lon)*np.sin(lat), np.cos(lon)*np.cos(lat)],
        [np.cos(lon),-np.sin(lon)*np.sin(lat), np.sin(lon)*np.cos(lat)],
        [0.0, np.cos(lat),np.sin(lat)]
        ])

    B_vector_ecef = R @ B_vector_enu

    B_vector_eci = ecef_to_eci(B_vector_ecef,time)

    return B_vector_eci