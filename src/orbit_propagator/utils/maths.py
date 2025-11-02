import numpy as np


def cross_product_matrix(v: np.ndarray) -> np.ndarray:
    """
    Computes the skew-symmetric cross product matrix of a 3-D vector.


    :param v:  A 3 dimensional vector [x, y, z]
    :return: A 3x3 matrix that represents the cross product matrix of v([v]x)
    """
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ]
    )

def attitude_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Computes the attitude matrix of a quaternion.

    :param q: A quaternion [w, x, y, z]
    :return: A 3x3 matrix that represents the attitude matrix of q
    """
    return np.eye(3)*(q[3]**2 - np.dot(q[0:3],q[0:3])) - 2*q[3]*cross_product_matrix(q[0:3]) + 2*np.outer(q[0:3],q[0:3])

def quaternion_from_attitude_matrix(A: np.ndarray) -> np.ndarray:
    """
    Computes the quaternion from attitude matrix

    :param A: A 3x3 matrix that represents the attitude matrix
    :return: A quaternion [w, x, y, z] from attitude matrix
    """
    return quaternion_from_euler_axis_angle(*euler_axis_angle_from_attitude_matrix(A))

def Omega(w: np.ndarray) -> np.ndarray:
    """
    Creates the 4x4 matrix used in quaternion kinematics: dq/dt = 0.5 * Omega(w) * q.

    :param w: angular velocity [wx, wy, wz] in rad/s
    :return: a 4x4 matrix that represents the Omega matrix
    """
    wx, wy, wz = w

    # For quaternion = [q0, q1, q2, q3] (scalar-first).
    return np.array([
        [ 0,  -wx,  -wy,  -wz],
        [ wx,   0,   wz,  -wy],
        [ wy, -wz,    0,   wx],
        [ wz,  wy,  -wx,    0]
    ])

def angle_between_vectors(v1: np.ndarray, v2:np.ndarray) -> float:
    """
    Computes the angle between two vectors.

    :param v1: a 3 dimensional vector
    :param v2: a 3 dimensional vector
    :return: angle in radians
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def rot_z(theta: float) -> np.ndarray:
    """
    Computes the rotation matrix of a rotation about the z axis.

    :param theta: angle in radians
    :return: a 3x3 matrix that represents the rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def rot_x(theta: float) -> np.ndarray:
    """
    Computes the rotation matrix of a rotation about the x axis.

    :param theta: angle in radians
    :return: a 3x3 matrix that represents the rotation matrix
    """

    c, s = np.cos(theta), np.sin(theta)

    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rot_y(theta: float) -> np.ndarray:
    """
    Computes the rotation matrix of a rotation about the y axis.

    :param theta: angle in radians
    :return: a 3x3 matrix that represents the rotation matrix
    """

    c, s = np.cos(theta), np.sin(theta)

    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])