import numpy as np

J2 = 1.08262668e-3
EARTH_RADIUS = 6378.0  # km
EARTH_MU = 3.9860043543609598E+05  # km³/s²
J = np.diag([1.0, 1.2, 1.0])
J_INV = np.linalg.inv(J)
