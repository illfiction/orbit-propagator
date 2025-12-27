import numpy as np
import ppigrf
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
# from satellite.satellite import Satellite
from maths.earth_frame_conversions import *

def earth_magnetic_field(lat, lon, alt_km, time):
    """
    Return the earth magnetic field at a given position in space at a given time
    Magnetic field is in ECI Frame

    :param lat: latitude in degrees
    :param lon:  longitude in degrees
    :param alt_km: altitude in kilometers
    :param time: Datetime object
    :return: Earth magnetic field in ECI Frame in nT
    """

    Be, Bn, Bu= ppigrf.igrf(lat, lon, alt_km, time.utc.to_datetime()) ## nanoTesla nT

    B_vector_ned = np.array([Bn[0], Be[0], Bu[0]])


    R = np.array([
        [-np.sin(lon), -np.cos(lon)*np.sin(lat), np.cos(lon)*np.cos(lat)],
        [np.cos(lon),-np.sin(lon)*np.sin(lat), np.sin(lon)*np.cos(lat)],
        [0, np.cos(lat),np.sin(lat)]
        ])

    B_vector_ecef = R @ B_vector_ned

    B_vector_eci = ecef_to_eci(B_vector_ecef,time)

    return B_vector_eci

def earth_magnetic_field_from_sat(sat):
    """
    Return the earth magnetic field at a given satellite position
    Magnetic field is in ECI Frame and nT

    :param sat: Satellite
    :return: Earth magnetic field in ECI Frame in nT
    """

    lat,lon,alt_m = eci_to_geodetic(sat.position , sat.time)

    alt_km = alt_m / 1000.0

    return earth_magnetic_field(lat, lon, alt_km, sat.time)