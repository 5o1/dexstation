"""Mount corrections must be pure relative z rotations, independent of quaternion scale."""
import numpy as np
import pytest

from dexstation.model.mount_geometry import j1_offset_deg
from dexstation.examples.ar5_g20.compose import rotate_about_base_z


@pytest.mark.parametrize('angle, expected', [
    (-179, -179), (0, 0), (179, 179), (-180, 180), (180, 180),
    (181, -179), (-181, 179), (360, 0), (539, 179),
])
def test_relative_z_rotation_and_wrap(angle, expected):
    base = np.array([.5, -.5, .5, .5])
    target = rotate_about_base_z(base, angle)
    assert j1_offset_deg(base, target) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize('tilt', [.011, 1., 180.])
def test_tilt_above_tolerance_is_rejected(tilt):
    half = np.radians(tilt) / 2
    with pytest.raises(ValueError, match='recalibrate'):
        j1_offset_deg([1, 0, 0, 0], [np.cos(half), np.sin(half), 0, 0])


def test_explicit_tilt_tolerance():
    half = np.radians(.005) / 2
    target = [np.cos(half), np.sin(half), 0, 0]
    assert j1_offset_deg([1, 0, 0, 0], target) == pytest.approx(0.)
    with pytest.raises(ValueError):
        j1_offset_deg([1, 0, 0, 0], target, tilt_tolerance_deg=.001)


def test_unnormalized_and_opposite_sign_quaternions():
    base = np.array([.5, -.5, .5, .5])
    target = np.array(rotate_about_base_z(base, -42))
    assert j1_offset_deg(3 * base, -7 * target) == pytest.approx(-42., abs=1e-10)


@pytest.mark.parametrize('invalid', [[0, 0, 0, 0], [1, 0, 0], [np.nan, 0, 0, 0], [np.inf, 0, 0, 0]])
def test_invalid_quaternion_is_rejected(invalid):
    with pytest.raises(ValueError):
        j1_offset_deg(invalid, [1, 0, 0, 0])
