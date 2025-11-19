import math

import numpy as np
from numpy.testing import assert_allclose

from maths import maths as m


def test_cross_product_matrix_matches_numpy():
    v = np.array([1.0, 2.0, -0.5])
    w = np.array([-4.0, 0.25, 3.0])
    assert_allclose(m.cross_product_matrix(v) @ w, np.cross(v, w))


def test_attitude_matrix_from_quaternion_identity():
    R = m.attitude_matrix_from_quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
    assert_allclose(R, np.eye(3))


def test_attitude_matrix_from_quaternion_rotation_matches_rot_z():
    angle = np.pi / 2
    q = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    R = m.attitude_matrix_from_quaternion(q)
    assert_allclose(R, m.rot_z(angle), atol=1e-9)


def test_quaternion_matrix_round_trip():
    angle = np.pi / 6
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    q = m.quaternion_from_euler_axis_angle(axis, angle)
    R = m.attitude_matrix_from_quaternion(q)
    axis_extracted, angle_extracted = m.euler_axis_angle_from_attitude_matrix(R)
    assert math.isclose(angle_extracted, angle, rel_tol=1e-9)
    assert_allclose(axis_extracted, axis)


def test_euler_axis_angle_from_attitude_matrix():
    angle = np.pi / 3
    R = m.rot_x(angle)
    axis, extracted_angle = m.euler_axis_angle_from_attitude_matrix(R)
    assert_allclose(axis, np.array([1.0, 0.0, 0.0]), atol=1e-8)
    assert math.isclose(extracted_angle, angle, rel_tol=1e-9)


def test_quaternion_from_euler_axis_angle():
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 4
    q = m.quaternion_from_euler_axis_angle(axis, angle)
    assert math.isclose(q[0], np.cos(angle / 2.0))
    assert_allclose(q[1:], axis * np.sin(angle / 2.0))


def test_quaternion_from_attitude_matrix_round_trip():
    angle = np.pi / 5
    R = m.rot_y(angle)
    q = m.quaternion_from_attitude_matrix(R)
    R_round_trip = m.attitude_matrix_from_quaternion(q)
    assert_allclose(R_round_trip, R, atol=1e-8)


def test_omega_matrix_structure():
    w = np.array([1.0, 2.0, -3.0])
    expected = np.array(
        [
            [0.0, -1.0, -2.0, 3.0],
            [1.0, 0.0, -3.0, -2.0],
            [2.0, 3.0, 0.0, 1.0],
            [-3.0, 2.0, -1.0, 0.0],
        ]
    )
    assert_allclose(m.Omega(w), expected)


def test_angle_between_vectors_returns_degrees():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.5, math.sqrt(3) / 2.0, 0.0])
    assert math.isclose(m.angle_between_vectors(v1, v2), 60.0, abs_tol=1e-9)


def test_rotation_matrices_are_orthogonal():
    angle = np.pi / 7
    for rot in (m.rot_x, m.rot_y, m.rot_z):
        R = rot(angle)
        assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert math.isclose(np.linalg.det(R), 1.0, rel_tol=1e-9)
