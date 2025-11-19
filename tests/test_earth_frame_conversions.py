import numpy as np
from numpy.testing import assert_allclose

from constants import OMEGA_EARTH, R_EARTH
from maths import earth_frame_conversions as efc


def test_geodetic_to_ecef_equator():
    vec = efc.geodetic_to_ecef(0.0, 0.0, 0.0)
    assert_allclose(vec, np.array([R_EARTH, 0.0, 0.0]))


def test_geodetic_to_ecef_pole():
    vec = efc.geodetic_to_ecef(np.pi / 2.0, 0.0, 0.0)
    assert_allclose(vec, np.array([0.0, 0.0, R_EARTH]), atol=1e-9)


def test_ecef_to_eci_no_rotation():
    vec = np.array([R_EARTH, 0.0, 0.0])
    result = efc.ecef_to_eci(vec, t=0.0)
    assert_allclose(result, vec)


def test_ecef_to_eci_quarter_rotation():
    vec = np.array([R_EARTH, 0.0, 0.0])
    t = (np.pi / 2.0) / OMEGA_EARTH
    result = efc.ecef_to_eci(vec, t=t)
    assert_allclose(result, np.array([0.0, -R_EARTH, 0.0]), atol=1e-6)


def test_ecef_to_eci_with_phi_offset():
    vec = np.array([0.0, R_EARTH, 0.0])
    phi = np.pi / 2.0
    result = efc.ecef_to_eci(vec, t=0.0, phi=phi)
    assert_allclose(result, np.array([-R_EARTH, 0.0, 0.0]), atol=1e-6)
