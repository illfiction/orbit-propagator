import numpy as np

J2 = 1.08262668e-3
EARTH_RADIUS = 6378.0  # km
EARTH_MU = 3.9860043543609598E+05  # km³/s²
J = np.diag([1.0, 1.2, 1.0])
J_INV = np.linalg.inv(J)
SATELLITE_DIMENSIONS = np.array([0.1, 0.1, 0.1]) # m (assuming 1U cubesat)
C_SRP = 1.8
SOLAR_FLUX = 1367 # W/m²
SOLAR_PRESSURE = 4.56 * 10**-6 # N/m²

