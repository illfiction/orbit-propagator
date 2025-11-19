import numpy as np
from typing import Dict, Tuple, Optional

from constants import ASTRONOMICAL_UNIT_KM, EARTH_RADIUS, SOLAR_FLUX
from dynamics.solar_radiation_force import sun_direction_unit
from maths.maths import attitude_matrix_from_quaternion

ASTRONOMICAL_UNIT_M = ASTRONOMICAL_UNIT_KM * 1_000.0


def _is_in_earth_eclipse(position_km: np.ndarray, sun_dir_earth: np.ndarray) -> bool:
    """Return True if the satellite lies in Earth's umbra."""

    r_norm = np.linalg.norm(position_km)
    if r_norm == 0.0:
        return False

    cos_theta = np.dot(position_km, sun_dir_earth) / r_norm
    if cos_theta >= 0.0:
        return False

    perpendicular_distance = r_norm * np.sqrt(max(0.0, 1.0 - cos_theta**2))
    return perpendicular_distance < EARTH_RADIUS


def compute_generated_power(
    time_seconds: float,
    satellite,
    power_config: Optional[Dict[str, float]] = None,
) -> Tuple[float, bool]:
    """Compute generated electrical power and eclipse state."""

    power_config = power_config or {}

    area_m2 = float(
        power_config.get("panel_area_m2", float(np.max(satellite.face_areas)))
    )
    efficiency = float(power_config.get("panel_efficiency", 0.28))
    system_losses = float(power_config.get("system_losses", 0.05))
    panel_normal_body = np.array(
        power_config.get("panel_normal_body_frame", [0.0, 0.0, 1.0]), dtype=float
    )

    if np.allclose(panel_normal_body, 0.0):
        panel_normal_body = np.array([0.0, 0.0, 1.0])
    panel_normal_body /= np.linalg.norm(panel_normal_body)

    sun_dir_earth = sun_direction_unit(time_seconds)
    position_km = np.asarray(satellite.position, dtype=float)

    in_eclipse = _is_in_earth_eclipse(position_km, sun_dir_earth)
    if in_eclipse:
        return 0.0, True

    sat_to_sun = sun_dir_earth * ASTRONOMICAL_UNIT_KM - position_km
    sun_distance_km = np.linalg.norm(sat_to_sun)
    if sun_distance_km == 0.0:
        return 0.0, False

    sat_to_sun_unit = sat_to_sun / sun_distance_km

    attitude_matrix = attitude_matrix_from_quaternion(satellite.quaternion)
    panel_normal_eci = attitude_matrix @ panel_normal_body
    panel_normal_eci /= np.linalg.norm(panel_normal_eci)

    incidence = max(0.0, np.dot(panel_normal_eci, sat_to_sun_unit))
    if incidence <= 0.0:
        return 0.0, False

    sun_distance_m = sun_distance_km * 1_000.0
    flux = SOLAR_FLUX * (ASTRONOMICAL_UNIT_M / sun_distance_m) ** 2

    raw_power = flux * area_m2 * incidence
    electrical_power = raw_power * efficiency * (1.0 - system_losses)
    return electrical_power, False
