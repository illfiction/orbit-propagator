import numpy as np
from dynamics.earth_magnetic_field import earth_magnetic_field
from maths.earth_frame_conversions import eci_to_geodetic
from maths.quaternion import Quaternion

def magnetorquer_torque(satellite, dt: float):
    Torque = np.zeros(3)

    if np.linalg.norm(satellite.angular_velocity) > 0.05:
        # print("sat position ", satellite.position)
        lat, lon, alt_m = eci_to_geodetic(satellite.position, satellite.time)

        alt_km = alt_m / 1000.0

        B_eci = earth_magnetic_field(lat, lon, alt_km, satellite.time)

        B_body_frame = satellite.Quaternion.rotate_vector(B_eci)

        B_dot = (B_body_frame - satellite.magnetic_field_history[-1]) / dt

        B_dot_norm = np.linalg.norm(B_dot)

        m_max = 0.00001

        m = -m_max * (B_dot / B_dot_norm)

        if B_dot_norm < 1e-12 or dt == 0.0:
            return np.zeros(3)  # No field change → no torque

        Torque = np.cross(m, B_body_frame)

        print(Torque)

        # Torque = np.zeros(3)

        satellite.time_list.append(satellite.time)
        satellite.magnetic_field_history.append(B_body_frame)

    return Torque