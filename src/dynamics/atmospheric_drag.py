import numpy as np
from pathlib import Path

from constants import EARTH_RADIUS
from maths.maths import attitude_matrix_from_quaternion

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "atmosphere.csv"
_ATMOS_TABLE = np.genfromtxt(DATA_FILE, delimiter=",", names=True)
_ALTITUDE_BREAKPOINTS_M = _ATMOS_TABLE["altitude_m"]
_DENSITY_AT_BREAKPOINT = _ATMOS_TABLE["density_kg_m3"]
_SCALE_HEIGHT_M = _ATMOS_TABLE["scale_height_m"]

EARTH_RADIUS_M = EARTH_RADIUS * 1_000.0


def atmospheric_density(position_km: np.ndarray) -> float:
    """Compute atmospheric density (kg/m^3) based on tabulated exponential model."""

    altitude_m = np.linalg.norm(position_km) * 1_000.0 - EARTH_RADIUS_M
    if altitude_m < _ALTITUDE_BREAKPOINTS_M[0]:
        return float(_DENSITY_AT_BREAKPOINT[0])
    if altitude_m > _ALTITUDE_BREAKPOINTS_M[-1]:
        return 0.0

    idx = np.searchsorted(_ALTITUDE_BREAKPOINTS_M, altitude_m, side="right") - 1
    base_alt = _ALTITUDE_BREAKPOINTS_M[idx]
    rho0 = _DENSITY_AT_BREAKPOINT[idx]
    scale_height = _SCALE_HEIGHT_M[idx]
    return float(rho0 * np.exp(-(altitude_m - base_alt) / scale_height))


def drag_acceleration(
    position_km: np.ndarray, velocity_km_s: np.ndarray, sat
) -> np.ndarray:
    """Compute atmospheric drag acceleration (km/s^2)."""

    rho = atmospheric_density(position_km)
    if rho <= 0.0:
        return np.zeros(3)

    velocity_m_s = velocity_km_s * 1_000.0
    speed = np.linalg.norm(velocity_m_s)
    if speed == 0.0:
        return np.zeros(3)

    velocity_hat = velocity_m_s / speed

    attitude_matrix = attitude_matrix_from_quaternion(sat.quaternion)
    face_areas = sat.face_areas  # m^2

    projected_area = 0.0
    for i in range(3):
        normal = attitude_matrix[:, i]
        cos_theta = np.dot(normal, velocity_hat)
        projected_area += face_areas[i] * abs(cos_theta)

    drag_force = (
        -0.5 * rho * speed**2 * sat.drag_coefficient * projected_area * velocity_hat
    )
    acceleration_m = drag_force / sat.mass
    return acceleration_m / 1_000.0
