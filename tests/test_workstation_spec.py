from helpers import bench_data, bench_spec
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dexstation.model import spec




def test_station_spec_and_pose_composition(tmp_path):
    workcell = bench_spec()
    assert workcell.world_frame == "world"
    assert workcell.left_hand.mount_status in ("unverified", "assumed", "measured")
    assert workcell.left_arm.joint_names[-1].endswith("joint_7")
    np.testing.assert_allclose(np.linalg.det(spec.pose_to_matrix(workcell.left_arm.base_pose)[:3, :3]), 1.0)
    identity = spec.compose(workcell.left_arm.base_pose, spec.invert(workcell.left_arm.base_pose))
    np.testing.assert_allclose(identity, [0, 0, 0, 1, 0, 0, 0], atol=1e-12)
    tip = [0.1, -0.2, 0.3, 1, 0, 0, 0]
    expected = spec.compose(spec.compose(workcell.left_arm.base_pose, tip), workcell.left_hand.mount)
    np.testing.assert_allclose(spec.hand_root_in_world(workcell, tip), expected)
    output = tmp_path / "spec.json"
    spec.dump_spec(workcell, output)
    assert spec.load_spec(output) == workcell






def test_quaternion_normalization_and_rpy_convention():
    pose = spec.matrix_to_pose(spec.pose_to_matrix([1, 2, 3, -2, 0, 0, 0]))
    assert pose == (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    quat = spec.rpy_to_quat(0.2, -0.4, 0.9)
    matrix = spec.pose_to_matrix([0, 0, 0, *quat])
    np.testing.assert_allclose(matrix[:3, :3], spec.rpy_to_matrix((0.2, -0.4, 0.9)), atol=1e-12)


@pytest.mark.parametrize("mutate,message", [
    (lambda data: data.update(extra=1), "unknown keys"),
    (lambda data: data["left_arm"].update(home_q=[0] * 6), "exactly 7"),
    (lambda data: data["table"].update(pose=[0, 0, 0, 1, 0, 0, float("nan")]), "finite"),
    (lambda data: data["left_hand"].update(mount_status="guessed"), "mount_status"),
    (lambda data: data["gamepad"].update(collision="mesh"), "collision"),
])
def test_strict_validation(tmp_path, mutate, message):
    data = bench_data()
    mutate(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        spec.load_spec(path)


@pytest.mark.parametrize('explicit_null', [False, True])
def test_gamepad_is_optional_and_empty_scene_round_trips(tmp_path, explicit_null):
    data = bench_data()
    data.pop('gamepad')
    if explicit_null:
        data['gamepad'] = None
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(data))
    workcell = spec.load_spec(source)
    assert workcell.gamepad is None
    assert 'gamepad' not in spec.spec_to_dict(workcell)
    output = tmp_path / 'empty.json'
    spec.dump_spec(workcell, output)
    assert spec.load_spec(output) == workcell
