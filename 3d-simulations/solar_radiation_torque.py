import numpy as np
from constants import EARTH_RADIUS, C_SRP, SOLAR_PRESSURE, SATELLITE_DIMENSIONS
import sys, os

# go up one directory and into math_helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "math_helpers")))

from maths import *


def solar_radiation_torque(quaternion):
    sun_vector = [0, 0, 1]  ##TODO: add sun vector calculation

    # TODO: need to check if satellite is eclipsed or not

    attitude_matrix = attitude_matrix_from_quaternion(quaternion)

    basis_vectors = []

    for col in range(3):
        v = attitude_matrix[:, col]
        basis_vectors.append(v / np.linalg.norm(v))


    r = []

    for i in range(3):
        r.append(basis_vectors[i]*SATELLITE_DIMENSIONS[i]/2)
        r.append(-basis_vectors[i]*SATELLITE_DIMENSIONS[i]/2)


    area_vectors = []

    for i in range(3):
        v = basis_vectors[i] * SATELLITE_DIMENSIONS[(i + 1) % 3] * SATELLITE_DIMENSIONS[(i + 2) % 3]
        area_vectors.append(v)
        area_vectors.append(-v)

    F_solar_list = []   #List of all the forces on the respective areas due to solar pressure will use when we do T = sigma(rxF)

    for i in range(6):
        cos_theta = np.dot(area_vectors[i],sun_vector)/np.linalg.norm(area_vectors[i])

        if cos_theta > 0:
            F_solar_list.append(-C_SRP * SOLAR_PRESSURE * cos_theta * area_vectors[i])
        else:
            F_solar_list.append(np.zeros(3))

    Torque = np.zeros(3)

    for i in range(6):
        Torque += np.cross(r[i], F_solar_list[i])


    print("Torque = ", Torque)
    return Torque