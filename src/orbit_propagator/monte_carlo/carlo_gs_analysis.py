import numpy as np
from orbit_propagator.constants import *
from orbit_propagator.utils.maths import attitude_matrix_from_quaternion,angle_between_vectors


# ----------------------------
# Rotation about z-axis (Active Transformation Matrix)
# ----------------------------
def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

# ----------------------------
# Convert geodetic lat, lon, h to ECEF (spherical Earth approx.)
# ----------------------------
def geodetic_to_ecef(lat, lon, h=0.0):
    x = (R_EARTH + h) * np.cos(lat) * np.cos(lon)
    y = (R_EARTH + h) * np.cos(lat) * np.sin(lon)
    z = (R_EARTH + h) * np.sin(lat)
    return np.array([x, y, z])

# ----------------------------
# Local "up" vector in ECEF
# ----------------------------
def normal_vector_ecef(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])

# ----------------------------
# Elevation angle calculation (sanity check done!)
# ----------------------------
def elevation_angle_deg(r_sat, r_gs, up_eci):

    # for the spherical approximation of Earth, up_eci points in the direction of the ground station position vector!
    # for the ellipsoidal Earth model, they would be different!
    rho = r_sat - r_gs
    rho_norm = np.linalg.norm(rho)
    up_norm = np.linalg.norm(up_eci)
    cos_angle = np.dot(rho, up_eci) / (rho_norm * up_norm)
    return 90 - np.degrees(np.arccos(cos_angle))


def ecef_to_eci(coordinate_vector_in_ecef, t, phi = 0):

    # the angle phi is an optional parameter to account for an initial rotation offset!
    # phi depends on the time of the launch of the satellite!

    # passive transformation matrix from ECEF to ECI frame!
    R_ecef_to_eci = rot_z( -1 * OMEGA_EARTH * t + phi)

    coordinate_vector_in_eci = R_ecef_to_eci @ coordinate_vector_in_ecef
    return coordinate_vector_in_eci


# ----------------------------
# Main function: compute downlink time per day and avg downlink time in without discontinuities!
# ----------------------------
def monte_carlo_time_over_ground_station(position_list, quaternion_list, dt, ground_station, analysis_params):

    
    # this is the vector along which the sattelite's antennas point!
    pointing_axis_body = np.array(analysis_params['pointing_axis_body_frame'])

    pointing_threshold_deg = analysis_params['pointing_angle_threshold_deg']

    in_pass = False
    pass_start = None
    pass_durations = {}
    pass_times = []

    # minimum elevation allowed for the satellite's data transmission for the ground station!
    elevation_threshold_deg = ground_station.min_elevation_deg

    pointing_error_angles = []
    pointing_times = []
    pointing_intervals = []
    total_on_target_time = 0


    for i, r_sat_eci in enumerate(position_list):
        
        # current time in seconds
        t = i * dt

        # compute ground station position in ECI, normal vector at the ground station position at time t!
        r_gs_ecef = geodetic_to_ecef(ground_station.lat_rad, ground_station.lon_rad_initial, ground_station.alt_km)
        up_ecef = normal_vector_ecef(ground_station.lat_rad, ground_station.lon_rad_initial)
        r_gs_eci = ecef_to_eci(r_gs_ecef, t)
        up_eci = ecef_to_eci(up_ecef, t)

        # compute elevation angle...
        elev = elevation_angle_deg(r_sat_eci, r_gs_eci, up_eci)

        if elev >= elevation_threshold_deg and not in_pass:
            # start of a new pass
            in_pass = True
            pass_start = t

        elif elev < elevation_threshold_deg and in_pass:
            # end of pass
            in_pass = False
            duration = t - pass_start
            day = int(pass_start // 86400)
            pass_durations[day] = pass_durations.get(day, 0) + duration

        if in_pass:
            # if in 
            q_sat = quaternion_list[i]
            attitude_matrix_body_to_eci = attitude_matrix_from_quaternion(q_sat)

            pointing_vector_eci = attitude_matrix_body_to_eci @ pointing_axis_body

            # line of sight vector
            los_vector_eci = r_gs_eci - r_sat_eci

            current_error_angle = angle_between_vectors(pointing_vector_eci, los_vector_eci)
            pointing_error_angles.append(current_error_angle)
            pass_times.append(t/3600)

            # check if satellite is pointing to the ground station
            if current_error_angle <= pointing_threshold_deg:
                total_on_target_time += dt
                pointing_times.append(t)


    if not pass_durations:
        print(f"No passes found over {ground_station.name} with elevation >= {elevation_threshold_deg}°.")
    else:
    
        interval = 0
        for j in range(1,len(pointing_times)):
            if pointing_times[j] - pointing_times[j-1] > dt:
                pointing_intervals.append(interval)
                interval = 0
            else:
                interval += dt


        return (np.mean(pointing_intervals), total_on_target_time)
        




# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example: dummy satellite states (circular orbit ~7000 km from Earth center)
    n_points = 50000
    dt = 10.0  # sec per step
    times = np.arange(0, n_points*dt, dt)
    r_mag = R_EARTH + 500.0  # 500 km altitude

    # simple equatorial circular orbit in ECI
    position_list = np.zeros((n_points, 3))
    for i, t in enumerate(times):
        theta = 2*np.pi * (t / 5400.0)  # ~90 min period
        position_list[i] = [r_mag*np.cos(theta), r_mag*np.sin(theta), 0]

    time_over_ground_station(position_list, dt)
