from tkinter.messagebox import RETRY

import numpy as np
import matplotlib.pyplot as plt
from constants import *
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "math_helpers")))
from maths import attitude_matrix_from_quaternion

# ----------------------------
# Rotation about z-axis
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
def geodetic_up(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])

# ----------------------------
# Elevation angle calculation
# ----------------------------
def elevation_angle(r_sat, r_gs, up_eci):
    rho = r_sat - r_gs
    rho_norm = np.linalg.norm(rho)
    up_norm = np.linalg.norm(up_eci)
    cos_angle = np.dot(rho, up_eci) / (rho_norm * up_norm)
    return 90 - np.degrees(np.arccos(cos_angle))

# ----------------------------
# Main function: compute downlink time per day
# ----------------------------

def angle_between_vectors(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def position_of_gs_eci(ground_station, t):

    longitude = ground_station.lon_rad_initial + OMEGA_EARTH * t
    r_gs_ecef = geodetic_to_ecef()



def time_over_ground_station(position_list, quaternion_list, dt, ground_station, analysis_params):

    # What is Analysis Params?
    # This is a dictionary containing various parameters for the analysis.
    # Where is this dictionary defined?
    # The dictionary is defined in the main simulation script, where the simulation parameters are set up.


    # this is the vector along which the sattelite's antennas point!
    pointing_axis_body = np.array(analysis_params['pointing_axis_body_frame'])

    # i don't understand what this is right now!
    pointing_threshold_deg = analysis_params['pointing_angle_threshold_deg']

    in_pass = False
    pass_start = None
    pass_durations = {}

    # elevation threshold for the satellite's data transmission for the ground station!
    elevation_threshold_deg = ground_station.min_elevation_deg

    pointing_angles = []
    pointing_times = []
    total_on_target_time = 0


    for i, r_sat_eci in enumerate(position_list):
        
        # current time in seconds
        t = i * dt

        # compute ground station position in ECI at time t!

        current_lon = ground_station.lon_rad_initial + OMEGA_EARTH * t
        r_gs_ecef = geodetic_to_ecef(ground_station.lat_rad, ground_station.lon_rad_initial, ground_station.alt_km)
        up_ecef = geodetic_up(ground_station.lat_rad, ground_station.lon_rad_initial)

        R_ecef_to_eci = rot_z( -1 * OMEGA_EARTH * t)
        r_gs_eci = R_ecef_to_eci @ r_gs_ecef
        up_eci = R_ecef_to_eci @ up_ecef

        elev = elevation_angle(r_sat_eci, r_gs_eci, up_eci)

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

            q_sat = quaternion_list[i]
            attitude_matrix_body_to_eci = attitude_matrix_from_quaternion(q_sat)

            pointing_vector_eci = attitude_matrix_body_to_eci @ pointing_axis_body

            # line of sight vector
            los_vector_eci = r_gs_eci - r_sat_eci

            angle = angle_between_vectors(pointing_vector_eci, los_vector_eci)
            pointing_angles.append(angle)
            pointing_times.append(t/3600)

            # check if satellite is pointing to the ground station
            if angle <= pointing_threshold_deg:
                total_on_target_time += dt

    # --- Reporting and Plotting ---
    if not pass_durations:
        print(f"No passes found over {ground_station.name} with elevation >= {elevation_threshold_deg}°.")
    else:
        avg_downlink_time = np.mean(list(pass_durations.values()))
        print(f"\n--- Ground Station Pass Analysis for {ground_station.name} ---")
        print(f"Average pass duration per day: {avg_downlink_time:.2f} seconds")
        print(
            f"Total time on-target (pointing error <= {pointing_threshold_deg}°): {total_on_target_time:.2f} seconds")

        # Plot pass durations
        days = sorted(pass_durations.keys())
        durations = [pass_durations[d] for d in days]
        plt.figure(figsize=(10, 5))
        plt.bar(days, durations, width=0.8)
        plt.xlabel("Day")
        plt.ylabel("Total Pass Duration (s)")
        plt.title(f"Daily Pass Duration over {ground_station.name} (Elevation ≥ {elevation_threshold_deg}°)")
        plt.grid(axis='y')
        plt.show()

    # Plot pointing angle over time
    if pointing_times:
        plt.figure(figsize=(12, 6))
        plt.plot(pointing_times, pointing_angles, label='Pointing Error Angle')
        plt.axhline(y=pointing_threshold_deg, color='r', linestyle='--',
                    label=f'On-Target Threshold ({pointing_threshold_deg}°)')
        plt.xlabel("Time (hours)")
        plt.ylabel("Pointing Angle (degrees)")
        plt.title(f"Satellite Pointing Error to {ground_station.name} During Passes")
        plt.legend()
        plt.grid(True)
        plt.show()

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

    downlink_time_per_day(position_list, dt)
