import numpy as np
from constants import J2, EARTH_RADIUS, EARTH_MU


def j2_accel(r_vec):
    x, y, z = r_vec
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    if r == 0:
        return np.zeros(3)

    z2 = z**2
    r5 = r2**2 * r
    k = 1.5 * J2 * EARTH_MU * (EARTH_RADIUS**2) / r5
    f = 5.0 * z2 / r2

    # implementing j2 formula

    ax = k * x * (f - 1.0)
    ay = k * y * (f - 1.0)
    az = k * z * (f - 3.0)
    return np.array([ax, ay, az])
