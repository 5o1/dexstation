from helpers import bench_data
"""Moving the scene arm mounts to the measured real mounts must not move any teleop pose in the world.

The scene rotates each vendor mount about the arm's base z (J1) axis by arm_base_rotation_deg; teleop
was calibrated on reference_base_orientation_wxyz. teleop_calibration.json shifts the mapped teleop J1
by the difference, and the scene's home J1 absorbs it, so base * Rz(q1) is unchanged for every input.
"""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dexstation.inputs.arm_mapping import ArmMapper, load_teleop_calibration
from dexstation.model.spec import spec_from_dict
from dexstation.examples.ar5_g20.compose import rotate_about_base_z

ROOT = Path(__file__).parents[1]
OVERRIDES = json.loads((ROOT / 'src/dexstation/examples/ar5_g20/data/sources/overrides.json').read_text())
BASELINE = json.loads((ROOT / OVERRIDES['baseline']).read_text())
SCENE = bench_data()
CALIBRATION = ROOT / 'src/dexstation/examples/ar5_g20/data/calibration/teleop.json'
SPEC = spec_from_dict(SCENE)
SIDES = ('left', 'right')


def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rz(radians):
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_offsets_are_the_mount_change_for_this_scene():
    offsets = load_teleop_calibration(CALIBRATION, SPEC)
    for side in SIDES:
        assert offsets[side][0] == pytest.approx({'left': -5.90, 'right': 19.15}[side], abs=1e-6)
        assert offsets[side][1:] == [0] * 6


def test_renamed_scene_keeps_offsets_regression():
    # The v1 loader returned None here and silently removed both corrections.
    offsets = load_teleop_calibration(CALIBRATION, replace(SPEC, name='renamed-scene'))
    assert offsets is not None
    assert offsets == load_teleop_calibration(CALIBRATION, SPEC)


def test_reference_mounts_need_no_correction():
    scene = deepcopy(SCENE)
    for side, angle in [('left', -90), ('right', 105)]:
        scene[f'{side}_arm']['base_pose'][3:] = rotate_about_base_z(BASELINE['arms'][side]['base_pose'][3:], angle)
    offsets = load_teleop_calibration(CALIBRATION, spec_from_dict(scene))
    for side in SIDES:
        assert offsets[side] == pytest.approx([0.] * 7, abs=1e-6)




def test_tilted_scene_requires_recalibration():
    scene = deepcopy(SCENE)
    w, x, y, z = scene['left_arm']['base_pose'][3:]
    c, s = np.cos(np.radians(.5)), np.sin(np.radians(.5))
    scene['left_arm']['base_pose'][3:] = [w*c-x*s, x*c+w*s, y*c+z*s, z*c-y*s]
    with pytest.raises(ValueError, match='recalibrate'):
        load_teleop_calibration(CALIBRATION, spec_from_dict(scene))


def test_single_arm_has_explicit_zero_offsets_for_absent_arm():
    spec = replace(SPEC, right_arm=None, right_hand=None)
    offsets = load_teleop_calibration(CALIBRATION, spec)
    assert offsets['left'] == load_teleop_calibration(CALIBRATION, SPEC)['left']
    assert offsets['right'] == [0.] * 7
    mapper = ArmMapper({'left': spec.left_arm.home_q, 'right': [0.] * 7},
                       startup='none', joint_offset_deg=offsets)
    assert mapper.joint_offset['right'].tolist() == [0.] * 7


@pytest.mark.parametrize('side', SIDES)
def test_uncovered_arm_model_requires_recalibration(side):
    arm = replace(getattr(SPEC, side + '_arm'), base_link=f'AR5_5_07{side[0].upper()}_W4C4A2_base')
    spec = replace(SPEC, **{side + '_arm': arm})
    with pytest.raises(ValueError, match=f'calibration does not cover arm model {arm.base_link}; recalibrate'):
        load_teleop_calibration(CALIBRATION, spec)






def test_new_base_with_offset_j1_reproduces_the_teleop_calibrated_base():
    offsets = load_teleop_calibration(CALIBRATION, SPEC)
    for side in SIDES:
        # Keep the independent, unrounded historical world-pose reference.
        calibrated = rotation(rotate_about_base_z(BASELINE['arms'][side]['base_pose'][3:],
                                                 {'left': -90., 'right': 105.}[side]))
        scene = rotation(SCENE[f'{side}_arm']['base_pose'][3:])
        for q1 in np.radians([-150., -40., 0., 25., 160.]):
            np.testing.assert_allclose(scene @ rz(q1 + np.radians(offsets[side][0])), calibrated @ rz(q1), atol=1e-12)


@pytest.mark.parametrize('station', [False, True])
def test_mapper_shifts_only_j1_by_the_offset(station):
    offsets = load_teleop_calibration(CALIBRATION, SPEC)
    home = {s: [0.] * 7 for s in SIDES}
    math = None
    if station:
        import shutil
        if not shutil.which('c++'):
            pytest.skip('station math needs a C++ compiler')
        from dexstation.model.station_math import StationMath
        math = StationMath()
    plain = ArmMapper(home, startup='none', station_math=math)
    shifted = ArmMapper(home, startup='none', station_math=math, joint_offset_deg=offsets)
    rng = np.random.default_rng(3)
    for side in SIDES:
        for _ in range(10):
            raw = rng.uniform([-60, -60, -60, -40, -100, -40, -40], [60, 20, 60, 100, 100, 40, 40])
            a = np.degrees(plain.command(side, raw, 0.).positions)
            b = np.degrees(shifted.command(side, raw, 0.).positions)
            np.testing.assert_allclose(b - a, offsets[side], atol=1e-9)


def test_shifted_j1_still_respects_the_absolute_limits():
    offsets = load_teleop_calibration(CALIBRATION, SPEC)
    mapper = ArmMapper({s: [0.] * 7 for s in SIDES}, startup='none', joint_offset_deg=offsets)
    command = mapper.command('right', [175., 0, 0, 0, 0, 0, 0], 0.)   # right J1 = -90 + 175 = 85 -> +19.15
    assert np.degrees(command.positions[0]) == pytest.approx(85 + 19.15)
    command = mapper.command('right', [300., 0, 0, 0, 0, 0, 0], 0.)
    assert np.degrees(command.positions[0]) <= 175. + 1e-9 and command.clamp_flags['absolute'][0]


def test_j1_is_a_pure_base_z_rotation_in_the_scene_urdfs():
    from dexstation.config.paths import resolve_workspace_path
    for side in SIDES:
        path = Path(resolve_workspace_path(SCENE[f'{side}_arm']['urdf']))
        if not path.exists():
            pytest.skip('arm URDFs are not in the configured workspace')
        joint = next(j for j in ET.parse(path).getroot().iter('joint') if j.get('name') == SCENE[f'{side}_arm']['joint_names'][0])
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
        axis = [float(v) for v in joint.find('axis').get('xyz').split()]
        assert xyz[:2] == pytest.approx([0, 0], abs=1e-9) and rpy == pytest.approx([0, 0, 0], abs=1e-9)
        assert np.abs(axis) == pytest.approx([0, 0, 1])
