import numpy as np
from orbit_propagator.constants import  SOLAR_PRESSURE
from orbit_propagator.utils.maths import *


def solar_radiation_force(sat):
    sun_vector = [1, 0, 0]  ##TODO: add sun vector calculation

    quaternion = sat.quaternion

    # TODO: need to check if satellite is eclipsed or not

    attitude_matrix = attitude_matrix_from_quaternion(quaternion)

    basis_vectors = []

    for col in range(3):
        v = attitude_matrix[:, col]
        basis_vectors.append(v / np.linalg.norm(v))


    r = []

    for i in range(3):
        r.append(basis_vectors[i]*sat.dimensions[i]/2)
        r.append(-basis_vectors[i]*sat.dimensions[i]/2)


    area_vectors = []

    for i in range(3):
        v = basis_vectors[i] * sat.dimensions[(i + 1) % 3] * sat.dimensions[(i + 2) % 3]
        area_vectors.append(v)
        area_vectors.append(-v)

    F_solar = np.zeros(3)   #Net force being added up


    for i in range(6):
        cos_theta = np.dot(area_vectors[i], sun_vector) / np.linalg.norm(area_vectors[i])

        if cos_theta > 0:
            F_solar += -sat.c_srp * SOLAR_PRESSURE * cos_theta * area_vectors[i]
        else:
            F_solar += 0

    # print(F_solar)

    return F_solar