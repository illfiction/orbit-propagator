import numpy as np


def cross_product_matrix(v):
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ]
    )

def attitude_matrix_from_quaternion(q):
    return np.eye(3)*(q[3]**2 - np.dot(q[0:3],q[0:3])) - 2*q[3]*cross_product_matrix(q[0:3]) + 2*np.outer(q[0:3],q[0:3])

def quaternion_from_attitude_matrix(A):
    return quaternion_from_euler_axis_angle(*euler_axis_angle_from_attitude_matrix(A))

def Omega(w):
    wx, wy, wz = w

    # For quaternion = [q0, q1, q2, q3] (scalar-first).
    return np.array([
        [ 0,  -wx,  -wy,  -wz],
        [ wx,   0,   wz,  -wy],
        [ wy, -wz,    0,   wx],
        [ wz,  wy,  -wx,    0]
    ])