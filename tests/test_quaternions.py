import numpy as np
from numpy.testing import assert_allclose

from maths.quaternions import Quaternion


def test_quaternion_norm_and_normalize():
    q = Quaternion(0.0, 3.0, 0.0, 4.0)
    assert np.isclose(q.norm(), 5.0)
    q.normalize()
    assert np.isclose(q.norm(), 1.0)


def test_hamilton_product():
    q1 = Quaternion(0.0, 1.0, 0.0, 0.0)
    q2 = Quaternion(0.0, 0.0, 1.0, 0.0)
    product = q1 * q2
    assert_allclose(product.as_array(), np.array([0.0, 0.0, 0.0, 1.0]))


def test_inverse_returns_identity_when_multiplied():
    q = Quaternion(0.9238795, 0.0, 0.3826834, 0.0)  # 45 deg about y
    q_inv = q.inverse()
    identity = q * q_inv
    assert_allclose(identity.as_array(), np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


def test_rotate_vector_about_z_axis():
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2.0
    q = Quaternion(np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0)))
    rotated = q.rotate_vector(np.array([1.0, 0.0, 0.0]))
    assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_from_omega_builds_expected_rotation():
    omega = np.array([0.0, 0.0, 1.0])
    dt = np.pi
    q = Quaternion.from_omega(omega, dt)
    assert_allclose(q.as_array(), np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6)


def test_quaternion_inverse_round_trip():
    q = Quaternion(0.5, -0.5, 0.5, -0.5)
    rotated = q.rotate_vector(np.array([1.0, 2.0, 3.0]))
    recovered = q.inverse().rotate_vector(rotated)
    assert_allclose(recovered, np.array([1.0, 2.0, 3.0]), atol=1e-6)
