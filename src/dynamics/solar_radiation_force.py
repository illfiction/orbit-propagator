import numpy as np

from constants import (
    ASTRONOMICAL_UNIT_KM,
    EARTH_RADIUS,
    JULIAN_DATE_J2000,
    SOLAR_FLUX,
    SPEED_OF_LIGHT,
)
from maths.maths import attitude_matrix_from_quaternion

ASTRONOMICAL_UNIT_M = ASTRONOMICAL_UNIT_KM * 1_000.0


def sun_direction_unit(t: float) -> np.ndarray:
    julian_date = JULIAN_DATE_J2000 + t / 86_400.0
    T = (julian_date - JULIAN_DATE_J2000) / 36_525.0

    phi = np.deg2rad(280.460 + 36_000.771 * T)
    M = np.deg2rad(357.5277233 + 35_999.05034 * T)
    lambda_ecl = (
        phi
        + np.deg2rad(1.914666471) * np.sin(M)
        + np.deg2rad(0.019994643) * np.sin(2 * M)
    )
    epsilon = np.deg2rad(23.439291 - 0.0130042 * T)

    dir_vec = np.array(
        [
            np.cos(lambda_ecl),
            np.cos(epsilon) * np.sin(lambda_ecl),
            np.sin(epsilon) * np.sin(lambda_ecl),
        ]
    )
    return dir_vec / np.linalg.norm(dir_vec)


def solar_radiation_force(t: float, position_km: np.ndarray, sat) -> np.ndarray:
    """Compute per-face SRP acceleration (km/s^2)."""

    sun_dir_earth = sun_direction_unit(t)
    sat_to_sun = sun_dir_earth * ASTRONOMICAL_UNIT_KM - position_km
    sun_distance = np.linalg.norm(sat_to_sun)
    if sun_distance == 0.0:
        return np.zeros(3)

    sat_to_sun_unit = sat_to_sun / sun_distance

    r_sat = position_km
    r_norm = np.linalg.norm(r_sat)
    if r_norm == 0.0:
        return np.zeros(3)

    theta = np.arccos(np.clip(np.dot(r_sat / r_norm, sun_dir_earth), -1.0, 1.0))
    if theta > np.pi / 2 and r_norm * np.sin(theta) < EARTH_RADIUS:
        return np.zeros(3)

    pressure_nominal = SOLAR_FLUX / SPEED_OF_LIGHT  # N/m^2 at 1 AU
    pressure = pressure_nominal * (ASTRONOMICAL_UNIT_M / (sun_distance * 1_000.0)) ** 2

    attitude_matrix = attitude_matrix_from_quaternion(sat.quaternion)
    force = np.zeros(3)

    face_normals = []
    for i in range(3):
        normal = attitude_matrix[:, i]
        area = sat.face_areas[i]
        face_normals.append((normal, area))
        face_normals.append((-normal, area))

    for normal, area in face_normals:
        cos_theta = np.dot(normal, sat_to_sun_unit)
        if cos_theta <= 0.0:
            continue
        projected_area = area * cos_theta
        reflection_term = (
            2.0 * (sat.srp_diffuse / 3.0 + sat.srp_specular * cos_theta) * normal
            + (1.0 - sat.srp_specular) * sat_to_sun_unit
        )
        force += -pressure * projected_area * reflection_term

    acceleration_m = force / sat.mass
    return acceleration_m / 1_000.0
