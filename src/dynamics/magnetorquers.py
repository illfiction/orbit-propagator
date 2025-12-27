import numpy as np
from satellite.satellite import Satellite
from dynamics.earth_magnetic_field import earth_magnetic_field_from_sat
from maths.quaternion import Quaternion

def magnetorquer_torque(satellite: Satellite):
    B_eci = earth_magnetic_field_from_sat(satellite)

    B_body_frame = satellite.Quaternion.rotate_vector(B_eci)

