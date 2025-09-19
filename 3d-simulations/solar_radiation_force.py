import numpy as np
from constants import J,J_INV,EARTH_RADIUS,C_SRP,SOLAR_PRESSURE,SATELLITE_DIMENSIONS
import sys, os

# go up one directory and into math_helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "math_helpers")))

from maths import *

def solar_pressure_torque(quaternion):
    attitude_matrix = maths.attitude_matrix_from_quaternion(quaternion)

    area_vectors = []

    r = []

    for rad in SATELLITE_DIMENSIONS:
        r.append(rad)
        r.append(-rad)

    for col in range(3):
        v = attitude_matrix[:,col]
        area_vectors.append(np.linalg.norm(v)/100)
        area_vectors.append(-np.linalg.norm(v)/100)

    sun_vector = [1,0,0] ##TODO: add sun vector calculation

    F_solar = np.zeros(3)

    for area_vector in area_vectors:
        F_solar += C_SRP * SOLAR_PRESSURE * np.dot(area_vector,sun_vector)

    return F_solar