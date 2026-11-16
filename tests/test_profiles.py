"""WP2 composition contracts; no physics, hardware, services or asset loading."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from dexstation.contracts.backend import FakeBackend
from dexstation.contracts.profiles import EMPTY_TASK
from dexstation.profiles.compose import compose, load_task, load_workbench, profile_digest, referenced_files
from helpers import bench_directory

ROOT = Path(__file__).resolve().parents[1]
BENCH = bench_directory()


@pytest.fixture
def workbench():
    return load_workbench(BENCH)




@pytest.mark.parametrize('task_id', ['empty', 'xbox_gamepad_v3_3', 'tabletop_cube_grasp'])
def test_three_tasks_compose_and_feed_fake_backend(workbench, task_id):
    task = load_task(task_id)
    before = deepcopy(task)
    scene = compose(workbench, task, 'sapien', seed=23)
    assert task == before
    assert len(scene.objects) == len(task['objects'])
    assert len(scene.spec.fixtures) == len(workbench['geometry']['fixtures']) + sum(o.type == 'static' for o in scene.objects)
    assert (scene.spec.gamepad is not None) == (task_id == 'xbox_gamepad_v3_3')
    fake = FakeBackend()
    for obj in scene.objects:
        fake.set_object_state(obj.name, obj.initial_pose_world)
    assert fake.observe().object_poses == {o.name: list(o.initial_pose_world) for o in scene.objects}
    fake.close()
    assert scene.limits_path == BENCH / 'limits.json'
    assert scene.takeover_path == BENCH / 'takeover.json'


def test_empty_is_builtin_identity_and_default_hands_are_open(workbench, monkeypatch):
    import dexstation.profiles.compose as profiles
    original = profiles._read
    def read(path):
        assert 'tasks/empty' not in str(path)
        return original(path)
    monkeypatch.setattr(profiles, '_read', read)
    task = load_task('empty')
    assert task == EMPTY_TASK
    task['objects'].append({'caller': 'mutation'})
    assert load_task('empty') == EMPTY_TASK
    scene = compose(workbench, load_task('empty'), 'sapien', seed=1)
    assert not scene.channels and not scene.cells and not scene.actions and not scene.objects
    assert not any(scene.spec.left_hand.home_q) and not any(scene.spec.right_hand.home_q)
    assert len(scene.cameras) == 1 and scene.cameras[0].name == 'camera_3'


def test_static_fixture_collision_preserves_proxies_and_visuals(workbench):
    task = load_task('xbox_gamepad_v3_3')
    assert task['fixtures'] == []
    static = [o for o in task['objects'] if o['type'] == 'static']
    for obj in static:
        assert obj['collision']['environment'] is False and obj['collision']['objects'] is False
    assert not next(o for o in task['objects'] if o['name'] == 'gamepad')['collision']['objects']
    scene = compose(workbench, task, 'sapien', seed=1)
    assert all(o.collision == task['objects'][i]['collision'] for i, o in enumerate(scene.objects))


def test_randomization_is_seeded_in_frame_and_preserves_source(workbench):
    task = load_task('tabletop_cube_grasp')
    original = deepcopy(task)
    first = compose(workbench, task, 'sapien', seed=123)
    same = compose(workbench, task, 'sapien', seed=123)
    different = compose(workbench, task, 'sapien', seed=124)
    assert first.objects == same.objects
    assert first.identity['scene_sha256'] == same.identity['scene_sha256']
    assert first.objects != different.objects
    table_top = workbench['frames']['table_top']['pose']
    x, y, z = first.objects[0].initial_pose_world[:3]
    assert table_top[0] + .05 <= x <= table_top[0] + .15
    assert table_top[1] - .08 <= y <= table_top[1] + .08
    assert z == pytest.approx(table_top[2] + .025, abs=1e-15)
    assert first.channels[0].shape == (7,) and first.channels[1].shape == (2,)
    assert task == original
    no_seed = compose(workbench, task, 'sapien')
    replay = compose(workbench, task, 'sapien', seed=no_seed.frozen['seed'])
    assert no_seed.objects == replay.objects


@pytest.mark.parametrize('mutate,message', [
    (lambda t: t['workbenches'].clear(), 'minItems|array length'),
    (lambda t: t['workbenches'].__setitem__(0, 'other'), 'does not support'),
    (lambda t: t['objects'][0]['initial_pose'].update(frame='missing'), 'Unknown workbench frame'),
    (lambda t: t['objects'][0]['randomize'].update(xy_m=[[1, -1], [0, 1]]), 'randomization interval'),
    (lambda t: t['objects'][0]['initial_pose'].update(pose=[0, 0, 1, 0, 0, 0, 0]), 'quaternion must be non-zero'),
    (lambda t: t['objects'][0].update(name='station_column'), 'Duplicate scene object'),
    (lambda t: t['observations'][0].update(object='missing'), 'Unknown observation object'),
    (lambda t: t['actions']['reset'].update(objects=['missing']), 'unknown object'),
    (lambda t: t.update(cameras=[]), 'forbidden field'),
])
def test_non_geometric_conflicts_fail_before_backend(workbench, mutate, message):
    task = load_task('tabletop_cube_grasp')
    mutate(task)
    with pytest.raises(ValueError, match=message):
        compose(workbench, task, 'sapien', seed=1)


def test_unsupported_backend_and_missing_profile_references(workbench, tmp_path):
    with pytest.raises(ValueError, match='Unsupported online backend'):
        compose(workbench, load_task('empty'), 'mujoco')
    data = json.loads((ROOT / 'src/dexstation/task_plugins/xbox_gamepad_v3_3/data/task.json').read_text())
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='not found'):
        load_task(path)
    (tmp_path / 'fixtures.json').symlink_to(ROOT / 'src/dexstation/task_plugins/xbox_gamepad_v3_3/data/fixtures.json')
    with pytest.raises(ValueError, match='regular file'):
        load_task(path)


def test_frame_parent_composition_and_cycles(workbench):
    # In-memory edits must carry their new checksum; no filesystem changes.
    workbench['frames']['offset'] = dict(parent='table_top', pose=[.1, .2, .3, 1, 0, 0, 0])
    workbench['manifest']['sha256'] = profile_digest(workbench, BENCH)
    task = load_task('tabletop_cube_grasp')
    task['objects'][0].pop('randomize')
    task['objects'][0]['initial_pose'] = dict(frame='offset', pose=[.01, .02, .03, 1, 0, 0, 0])
    scene = compose(workbench, task, 'sapien', seed=1)
    np.testing.assert_allclose(scene.objects[0].initial_pose_world[:3],
                               np.array(workbench['frames']['table_top']['pose'][:3]) + [.11, .22, .33], atol=1e-15)
    workbench['frames']['offset']['parent'] = 'offset'
    workbench['manifest']['sha256'] = profile_digest(workbench, BENCH)
    with pytest.raises(ValueError, match='Frame cycle'):
        compose(workbench, task, 'sapien', seed=1)


def test_manifest_checksum_covers_references_and_wp5_placeholders(tmp_path):
    directory = tmp_path / 'bench'
    shutil.copytree(BENCH, directory)
    profile = load_workbench(directory)
    refs = referenced_files(profile, directory)
    for name in ('limits.json', 'takeover.json'):
        assert refs[name] == hashlib.sha256((directory / name).read_bytes()).hexdigest()
    assert refs['calibration/teleop.json'] == hashlib.sha256((directory / 'calibration/teleop.json').read_bytes()).hexdigest()
    before = profile_digest(profile, directory)
    # Valid JSON formatting changes are still byte changes in referenced files.
    with (directory / 'calibration/teleop.json').open('a') as stream:
        stream.write('\n')
    assert profile_digest(profile, directory) != before
    with pytest.raises(ValueError, match='sha256 mismatch'):
        load_workbench(directory)
    profile['manifest']['sha256'] = profile_digest(profile, directory)
    (directory / 'workbench.json').write_text(json.dumps(profile))
    load_workbench(directory)
    with (directory / 'limits.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='sha256 mismatch'):
        load_workbench(directory)


def test_task_reference_byte_changes_change_identity(workbench, tmp_path):
    for name in ('task.json', 'fixtures.json'):
        shutil.copyfile(ROOT / 'src/dexstation/task_plugins/xbox_gamepad_v3_3/data' / name, tmp_path / name)
    before = compose(workbench, load_task(tmp_path), 'sapien', seed=1)
    with (tmp_path / 'fixtures.json').open('a') as stream:
        stream.write('\n')
    after = compose(workbench, load_task(tmp_path), 'sapien', seed=1)
    assert before.objects == after.objects
    assert before.identity['task']['sha256'] != after.identity['task']['sha256']




def test_articulation_joint_state_and_geometry_survive_composition(workbench):
    task = load_task(ROOT / 'src/dexstation/contracts/examples/articulated_hinge.json')
    scene = compose(workbench, task, 'sapien', seed=1)
    obj = scene.objects[0]
    assert obj.type == 'articulation'
    assert obj.geometry == dict(urdf='workspace://assets/tasks/articulated_hinge/hinge.urdf',
                                fixed_root=True, collision_mesh='coacd')
    assert obj.extra['initial_joint_positions'] == {'hinge_joint': 0.25}
    task['objects'][0].pop('initial_joint_positions')
    zero = compose(workbench, task, 'sapien', seed=1)
    assert 'initial_joint_positions' not in zero.objects[0].extra




@pytest.mark.parametrize('recording', [None, {'format': 'task-owned-rec-v1'}])
def test_recording_declaration_loads_without_inventing_format(tmp_path, recording):
    task = deepcopy(EMPTY_TASK)
    if recording is not None:
        task['recording'] = recording
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(task))
    loaded = load_task(path)
    assert loaded.get('recording') == recording


@pytest.mark.parametrize('recording', [{}, {'format': ''}, {'format': 'v1', 'extra': True}])
def test_recording_declaration_rejects_invalid_profile(tmp_path, recording):
    task = deepcopy(EMPTY_TASK)
    task['recording'] = recording
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(task))
    with pytest.raises(ValueError):
        load_task(path)


def test_keep_out_box_schema_positive_and_negative(tmp_path):
    task = json.loads((ROOT/'src/dexstation/task_plugins/tabletop_cube_grasp/data/task.json').read_text())
    task['keep_out'] = [dict(name='zone', shape='box', frame='table_top',
        pose=[0, 0, .2, 1, 0, 0, 0], half_extents_m=[.1, .2, .3])]
    path = tmp_path/'task.json'
    path.write_text(json.dumps(task))
    assert load_task(path)['keep_out'] == task['keep_out']
    for key, value in [('shape', 'sphere'), ('half_extents_m', [-1, 1, 1]), ('pose', [0]*6)]:
        invalid = deepcopy(task); invalid['keep_out'][0][key] = value
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            load_task(path)


def test_task_digest_without_checkout_uses_packaged_ledger(tmp_path, monkeypatch):
    from dexstation.profiles import compose as profiles
    task = load_task('empty')
    expected = profile_digest(task, tmp_path)
    monkeypatch.setattr(profiles, 'ROOT', tmp_path / 'no-checkout')
    assert profile_digest(task, tmp_path) == expected


def test_compose_without_git_checkout_keeps_unknown_code_identity(tmp_path, monkeypatch):
    from dexstation.profiles import compose as profiles
    monkeypatch.setattr(profiles, 'ROOT', tmp_path)
    scene = profiles.compose(load_workbench(BENCH), load_task('empty'), 'sapien', seed=0)
    assert scene.identity['code'] == dict(git_commit=None, dirty=None)
