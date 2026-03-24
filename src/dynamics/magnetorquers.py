import numpy as np
from dynamics.earth_magnetic_field import earth_magnetic_field
from maths.earth_frame_conversions import eci_to_geodetic
from maths.quaternion import Quaternion

def magnetorquer_torque(satellite, dt: float):
    Torque = np.zeros(3)

    if (np.linalg.norm(satellite.angular_velocity) != 0.1):
        print("sat position ", satellite.position)
        lat, lon, alt_m = eci_to_geodetic(satellite.position, satellite.time)

        alt_km = alt_m / 1000.0

        B_eci = earth_magnetic_field(lat, lon, alt_km, satellite.time)

        B_body_frame = satellite.Quaternion.rotate_vector(B_eci)

        B_dot = (B_body_frame - satellite.magnetic_field_history[-1]) / dt

        m_max = 0.0000001

        Torque = m_max * B_dot / np.linalg.norm(B_dot)

        # Torque = np.zeros(3)

        satellite.time_list.append(satellite.time)
        satellite.magnetic_field_history.append(B_body_frame)

    return Torque