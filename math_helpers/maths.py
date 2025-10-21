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