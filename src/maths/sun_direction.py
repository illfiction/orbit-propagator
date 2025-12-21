from astropy.time import Time
import astropy.units as u
# from satellite.satellite import Satellite
import numpy as np
from constants import JULIAN_DATE_J2000


def sun_direction_unit(t: float, sat) -> np.ndarray:
    T = (sat.time.tt.jd - JULIAN_DATE_J2000) / 36_525.0

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