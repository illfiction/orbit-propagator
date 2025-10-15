import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Constants
# ----------------------------
OMEGA_EARTH = 7.2921159e-5  # rad/s
R_EARTH = 6378.137  # km (equatorial radius)

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
    cosang = np.dot(rho, up_eci) / (rho_norm * up_norm)
    return np.degrees(np.arcsin(cosang))

# ----------------------------
# Main function: compute downlink time per day
# ----------------------------
def time_over_ground_station(position_list, dt):
    # Ground station (example: BITS Pilani, India)
    lat = np.radians(28.3591)
    lon0 = np.radians(75.5882)
    h = 0.0

    in_pass = False
    pass_start = None
    pass_durations = {}

    for i, r_sat in enumerate(position_list[:, :3]):
        t = i * dt
        lon = lon0 + OMEGA_EARTH * t

        r_gs_ecef = geodetic_to_ecef(lat, lon, h)
        up_ecef = geodetic_up(lat, lon)

        R = rot_z(OMEGA_EARTH * t)
        r_gs_eci = R @ r_gs_ecef
        up_eci = R @ up_ecef

        elev = elevation_angle(r_sat, r_gs_eci, up_eci)

        if elev >= 60 and not in_pass:
            # start of a new pass
            in_pass = True
            pass_start = t
        elif elev < 60 and in_pass:
            # end of pass
            in_pass = False
            duration = t - pass_start
            day = int(pass_start // 86400)
            pass_durations[day] = pass_durations.get(day, 0) + duration

    # convert to sorted arrays
    days = sorted(pass_durations.keys())
    durations = [pass_durations[d] for d in days]

    # plot
    plt.figure(figsize=(8, 4))
    plt.bar(days, durations, width=0.8)
    plt.xlabel("Day")
    plt.ylabel("Downlink time (s)")
    plt.title("Daily Downlink Time (Elevation ≥ 60°)")
    plt.show()

    return pass_durations

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
