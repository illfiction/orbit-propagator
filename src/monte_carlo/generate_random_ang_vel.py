import numpy as np

def generate_random_unit_vector(number_of_vectors=1):
    """
    Generating a list of randomly oriented unit vectors in 3D space such that they are uniformly distributed over the surface of a unit sphere.
    1. Generate random polar angle (theta) uniformly from 0 to pi.
    2. Generate random azimuthal angle (phi) uniformly from 0 to 2pi.
    """
    theta_vals = np.random.uniform(0, np.pi, size=number_of_vectors)
    phi_vals = np.random.uniform(0, 2 * np.pi, size=number_of_vectors)

    # Convert spherical coordinates (theta, phi) to Cartesian coordinates (x, y, z)
    x = np.sin(theta_vals) * np.cos(phi_vals)
    y = np.sin(theta_vals) * np.sin(phi_vals)
    z = np.cos(theta_vals)

    return np.column_stack((x, y, z))

if __name__ == "__main__":
    
    n = 100
    random_unit_vectors = generate_random_unit_vector(n)
    angular_velocities_rad_s = random_unit_vectors * 24 * (np.pi / 180)

    # cutting to 3 decimal places
    angular_velocities_rad_s = np.round(angular_velocities_rad_s, 3)

    # save to file in csv format
    np.savetxt(
        "random_angular_velocities.csv",
        angular_velocities_rad_s,
        delimiter=",",
        header="wx,wy,wz",
        comments="",
        fmt="%.3f"   # <-- controls decimal places
    )

    print(f"Generated {n} random angular velocity vectors (rad/s) and saved to 'random_angular_velocities.csv'")


