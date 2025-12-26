import numpy as np
import ppigrf
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
# from satellite.satellite import Satellite
from maths.earth_frame_conversions import *

def earth_magnetic_field(lat, lon, alt_km, year):
    Be, Bn, Bu= ppigrf.igrf(lat, lon, alt_km, year)

    return Bn, Be, Bu

def earth_magnetic_field_from_sat(sat):

    lat,lon,alt_m = eci_to_geodetic(sat.position , sat.time)

    alt_km = alt_m / 1000.0

    return earth_magnetic_field(lat, lon, alt_km, sat.time.utc.to_datetime())