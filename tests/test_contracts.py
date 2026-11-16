"""Executable WP0 contracts; no hardware, physics, network or filesystem writer.

Prefer an already installed jsonschema. The fallback deliberately handles only
keywords used by these fixtures and fails closed on unknown assertion keywords.
It is not a general JSON Schema implementation or a production validator.
"""

from helpers import ContractValidationError, _json_equal, _minimal_validate, validate_instance, load_schema
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pytest

from dexstation.contracts.backend import CollectionBackend, FakeBackend
from dexstation.contracts.limits import FakeLimits, WorkspaceLimits
from dexstation.contracts.profiles import EMPTY_TASK
from dexstation.contracts.recording import EpisodeWriterProtocol, MemoryEpisodeWriter, WORKBENCH_ROW_KEYS, read_episode
from dexstation.contracts.takeover import ArmState, ArmTakeover, FakeTakeover

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / 'src/dexstation/contracts'
EXAMPLES = CONTRACTS / 'examples'
SCHEMAS = CONTRACTS / 'schemas'
DRAFT = 'https://json-schema.org/draft/2020-12/schema'











def load_example(name):
    return json.loads((EXAMPLES / f'{name}.json').read_text())




EXAMPLE_SCHEMAS = {
    'empty': 'task', 'xbox_gamepad_v3_3': 'task', 'tabletop_cube_grasp': 'task',
    'articulated_hinge': 'task',
    'ar5_08_g20_bench_bimanual': 'workbench', 'limits': 'limits', 'takeover': 'takeover',
    'calibration_teleop': 'calibration', 'calibration_real_robot': 'calibration',
    'episode_empty': 'episode', 'episode_gamepad': 'episode',
    **{name: 'http' for name in ('http_config', 'http_task_cells', 'http_task_web',
                                'http_session_start', 'http_task_action',
                                'http_task_action_recording', 'http_status', 'http_episodes')},
}


def test_every_example_is_covered():
    assert {p.stem for p in EXAMPLES.glob('*.json')} == EXAMPLE_SCHEMAS.keys()
    assert {p.name for p in SCHEMAS.glob('*.json')} == {
        f'{name}.schema.json' for name in {*EXAMPLE_SCHEMAS.values(), 'site'}}


@pytest.mark.parametrize('name,schema_name', EXAMPLE_SCHEMAS.items())
def test_example_passes_schema(name, schema_name):
    schema = load_schema(schema_name)
    assert schema['$schema'] == DRAFT
    validate_instance(load_example(name), schema)


# Both a missing field and a wrong type for every schema, including nested API fields.
BAD_CASES = [
    ('ar5_08_g20_bench_bimanual', ('manifest',), 'missing'),
    ('ar5_08_g20_bench_bimanual', ('robots', 'left_arm', 'joint_names'), 'type'),
    ('empty', ('id',), 'missing'), ('empty', ('objects',), 'type'),
    ('limits', ('angular',), 'missing'), ('limits', ('angular', 'elevation', 'max_up_deg'), 'type'),
    ('takeover', ('parts',), 'missing'), ('takeover', ('match_dwell_s',), 'type'),
    ('calibration_teleop', ('arm_base_links',), 'missing'),
    ('calibration_teleop', ('reference_base_orientation_wxyz', 'left'), 'type'),
    ('calibration_real_robot', ('hand_motor_reverse',), 'missing'),
    ('calibration_real_robot', ('hand_motor_reverse', 'right'), 'type'),
    ('episode_empty', ('layers',), 'missing'), ('episode_empty', ('rows',), 'type'),
    ('http_status', ('response', 'body', 'limits', 'report', 'angular_range'), 'missing'),
    ('http_status', ('response', 'status'), 'type'),
]


@pytest.mark.parametrize('name,keys,kind', BAD_CASES)
def test_bad_examples_are_rejected(name, keys, kind):
    data = load_example(name)
    target = data
    for key in keys[:-1]:
        target = target[key]
    if kind == 'missing':
        del target[keys[-1]]
    else:
        target[keys[-1]] = 'WRONG TYPE'
    with pytest.raises(ContractValidationError):
        validate_instance(data, load_schema(EXAMPLE_SCHEMAS[name]))


def test_fakes_satisfy_runtime_protocols():
    assert isinstance(FakeBackend(), CollectionBackend)
    assert isinstance(FakeLimits(), WorkspaceLimits)
    assert isinstance(FakeTakeover(), ArmTakeover)
    assert isinstance(MemoryEpisodeWriter(), EpisodeWriterProtocol)


def test_limits_takeover_fakes_and_report_schema():
    limits = FakeLimits()
    assert limits.with_task_keep_out([{'name': 'ignored'}]) is limits
    report = limits.check({'left': [999.] * 7}, {})
    assert report.ok and report.angular_range == {} and report.spatial == report.angular == []
    schema = load_schema('limits')['$defs']['report']
    validate_instance(asdict(report), schema)
    populated = asdict(report)
    populated['angular_range'] = {'left': [-70., 80.], 'right': [-80., 70.]}
    validate_instance(populated, schema)
    takeover = FakeTakeover()
    for side in ('left', 'right'):
        decision = takeover.update(1., side, [999.] * 7, [0.] * 7, False)
        assert decision.state is ArmState.FOLLOWING and decision.release
        assert decision.deviation_deg == [] and decision.reason is None
    takeover.restore(takeover.snapshot())
    assert takeover.snapshot() == {}


def test_fake_backend_is_memory_only_and_snapshots_both_layers():
    backend = FakeBackend()
    backend.apply({'left': [1.] * 7})
    backend.set_object_state('cube', [1, 2, 3, 1, 0, 0, 0], [0] * 6)
    backend.step(0.02)
    state = backend.snapshot_motion_state()
    assert state.keys() == {'workbench', 'task'}
    backend.reset()
    assert backend.observe().object_poses == {}
    backend.restore_motion_state(state)
    observation = backend.observe()
    assert observation.robot.sim_time_s == 0.02
    assert observation.object_poses['cube'] == [1, 2, 3, 1, 0, 0, 0]
    np.testing.assert_equal(observation.robot.arm_q, np.zeros((2, 7)))
    np.testing.assert_equal(backend.contact_force('left', 'tip', 'cube'), np.zeros(3))
    observation.object_poses['cube'][0] = -100
    assert backend.observe().object_poses['cube'][0] == 1
    backend.close()
    assert backend.closed


@pytest.mark.parametrize('task_id,task_format', [('empty', None), ('tabletop_cube_grasp', 'cube-example-rec-v1')])
def test_memory_episode_layered_roundtrip_and_join(task_id, task_format):
    backend = FakeBackend()
    writer = MemoryEpisodeWriter(task_id=task_id, task_format=task_format)
    assert isinstance(writer, EpisodeWriterProtocol)
    for index in range(3):
        backend.step(0.01)
        obs = backend.observe()
        if task_id != 'empty':
            obs.task = {'cube_pose': [index, 0, 0, 1, 0, 0, 0], 'cube_contact': [False, True]}
        commands = {'left': [index] * 7}
        writer.append(obs, commands, applied_monotonic_ns=index + 1,
                      input_sequences={'arm': index}, arm_clamps={'left': [False] * 7})
        commands['left'][0] = -100
    writer.append_inputs([{'sequence': 1}])
    writer.event('workbench', 'takeover', {'state': 'following'})
    if task_id != 'empty':
        writer.event('task', 'action', {'action': 'reset'})
    writer.note('Example note')
    writer.close()
    episode = read_episode(writer)
    assert episode.meta['status'] == 'complete' and episode.meta['rows'] == 3
    assert episode.inputs == [{'sequence': 1}] and episode.notes == ['Example note']
    assert len(episode.workbench.events) == 1
    assert tuple(episode.workbench.rows[0]) == WORKBENCH_ROW_KEYS
    assert [r['commands']['left'][0] for r in episode.workbench.rows] == [0, 1, 2]
    if task_id == 'empty':
        assert episode.task is None
    else:
        assert len(episode.task.rows) == 3
        assert all('wall_time_ns' not in row for row in episode.task.rows)
        # Deliberate permutation proves joining is by step, not zip/index.
        episode.task.rows.reverse()
    joined = episode.joined()
    assert [r['step'] for r in joined] == [0, 1, 2]
    for index, row in enumerate(joined):
        assert row['sim_time_s'] == row['workbench']['sim_time_s']
        if task_id != 'empty':
            assert row['task']['sim_time_s'] == row['sim_time_s']
            assert row['task']['cube_pose'][0] == index
        else:
            assert row['task'] is None
    episode.workbench.rows.clear()
    assert len(read_episode(writer).workbench.rows) == 3


def test_memory_join_rejects_different_times_or_missing_steps():
    writer = MemoryEpisodeWriter(task_id='cube', task_format='cube-example-rec-v1')
    backend = FakeBackend()
    writer.append(backend.observe(), {}, applied_monotonic_ns=None, input_sequences={}, arm_clamps={})
    episode = read_episode(writer)
    episode.task.rows[0]['sim_time_s'] += 1
    with pytest.raises(ValueError, match='times differ'):
        episode.joined()
    episode = read_episode(writer)
    episode.task.rows[0]['step'] = 1
    with pytest.raises(ValueError, match='different steps'):
        episode.joined()


def test_memory_writer_refuses_clock_overwrite_and_repeated_time():
    writer = MemoryEpisodeWriter(task_id='cube', task_format='cube-example-rec-v1')
    backend = FakeBackend()
    obs = backend.observe()
    obs.task = {'sim_time_s': 123}
    kwargs = dict(applied_monotonic_ns=None, input_sequences={}, arm_clamps={})
    with pytest.raises(ValueError, match='clock fields'):
        writer.append(obs, {}, **kwargs)
    obs.task = {}
    writer.append(obs, {}, **kwargs)
    with pytest.raises(ValueError, match='timestamps must increase'):
        writer.append(obs, {}, **kwargs)


def test_empty_constant_matches_example():
    assert EMPTY_TASK == load_example('empty')


def test_calibration_examples_preserve_current_physical_facts_and_loaders():
    from types import SimpleNamespace
    from dexstation.inputs.arm_mapping import load_teleop_calibration
    from dexstation.profiles.calibration import load_calibration

    teleop = load_example('calibration_teleop')
    real = load_example('calibration_real_robot')
    spec = SimpleNamespace(**{side + '_arm': SimpleNamespace(
        base_link=real['arm_base_links'][side][0],
        base_pose=(0, 0, 0, *real['measured_base_orientation_wxyz'][side])) for side in ('left', 'right')})
    offsets = load_teleop_calibration(EXAMPLES / 'calibration_teleop.json', spec)
    assert all(len(offsets[side]) == 7 for side in ('left', 'right'))
    result = load_calibration(EXAMPLES / 'calibration_real_robot.json', spec)
    np.testing.assert_allclose([result['arm'][side] for side in ('left', 'right')], np.zeros((2, 7)), atol=1e-7)
    assert result['hand'] == real['hand_motor_reverse']
    assert 'scene' not in teleop and 'scene' not in real


def test_http_coverage_and_reviewer_cell_state_ruling():
    examples = [load_example(name) for name in EXAMPLE_SCHEMAS if name.startswith('http_')]
    assert {(e['request']['method'], e['request']['path']) for e in examples} == {
        ('GET', '/api/v1/config'), ('GET', '/api/v1/tasks/tabletop_cube_grasp/cells'),
        ('GET', '/tasks/xbox_gamepad_v3_3/web/gamepad-overlay.mjs'),
        ('POST', '/api/v1/session/start'), ('POST', '/api/v1/task/action'),
        ('GET', '/api/v1/status'), ('GET', '/api/v1/episodes'),
    }
    status = load_example('http_status')
    task = status['response']['body']['task']
    assert isinstance(task['cell_state'], dict) and 'cells' not in task
    task['cells'] = task.pop('cell_state')
    with pytest.raises(ContractValidationError):
        validate_instance(status, load_schema('http'))





@pytest.mark.parametrize('name', ['empty', 'tabletop_cube_grasp'])
def test_task_cannot_define_cameras(name):
    data = load_example(name)
    data['cameras'] = []
    with pytest.raises(ContractValidationError):
        validate_instance(data, load_schema('task'))


@pytest.mark.parametrize('mutate', [
    lambda t: t['initial_robot_state']['hand_q'].update(right=[0] * 20),
    lambda t: t['initial_robot_state']['hand_q'].update(right=[0] * 22),
    lambda t: t['initial_robot_state']['hand_q'].update(center=[0] * 21),
    lambda t: t['objects'][0].update(collision='sdf'),
    lambda t: t['objects'][0]['collision'].update(robots=1),
    lambda t: t['objects'][0]['geometry'].pop('collision_mesh'),
    lambda t: t['objects'][0]['geometry'].update(collision_mesh='mesh'),
    lambda t: t['objects'][0].update(initial_joint_positions={'trigger': 0.2}),
    lambda t: t['objects'][0].update(release='legacy-root-field'),
])
def test_reviewer_task_contract_rejects_invalid_state_or_collision(mutate):
    data = load_example('xbox_gamepad_v3_3')
    mutate(data)
    with pytest.raises(ContractValidationError):
        validate_instance(data, load_schema('task'))


def test_articulation_geometry_and_named_initial_joints():
    data = load_example('empty')
    obj = dict(name='hinge', type='articulation',
               geometry=dict(urdf='workspace://assets/tasks/hinge/hinge.urdf', fixed_root=True),
               initial_pose=dict(frame='world', pose=[0, 0, 1, 1, 0, 0, 0]),
               collision=dict(robots=True, environment=True, objects=True))
    data['objects'] = [obj]
    validate_instance(data, load_schema('task'))
    obj['initial_joint_positions'] = {'hinge_joint': 0.25}
    obj['geometry']['collision_mesh'] = 'coacd'
    validate_instance(data, load_schema('task'))
    obj['initial_joint_positions']['hinge_joint'] = '0.25'
    with pytest.raises(ContractValidationError):
        validate_instance(data, load_schema('task'))
    obj.pop('initial_joint_positions')
    obj['geometry'].pop('urdf')
    with pytest.raises(ContractValidationError):
        validate_instance(data, load_schema('task'))


def test_create_backend_registry_dispatches_and_imports_sapien_lazily(monkeypatch):
    import sys
    from types import SimpleNamespace
    from dexstation.recording import session as backend
    calls = []
    marker = object()
    monkeypatch.setitem(sys.modules, 'dexstation.backends.sapien_physics',
        SimpleNamespace(CollectionSapienBackend=lambda scene, **kw: calls.append((scene, kw)) or marker))
    assert backend.create_registered_backend('sapien', 'scene', work_dir='scratch') is marker
    assert calls == [('scene', {'work_dir': 'scratch'})]
    from dexstation.backends import registry
    monkeypatch.setitem(registry._ENTRIES, 'fake', registry.BackendEntry('fake', lambda scene, **kw: FakeBackend()))
    assert isinstance(backend.create_registered_backend('fake', None), CollectionBackend)
    with pytest.raises(ValueError, match='unknown collection backend'):
        backend.create_registered_backend('unknown', None)





def test_limits_schema_accepts_configured_points_and_rejects_excess_elevation():
    data = load_example('limits')
    data['angular']['points'] = ['J3','J4','J5','J6','J7','flange']
    validate_instance(data, load_schema('limits'))
    data['angular']['elevation']['max_up_deg'] = 46
    with pytest.raises(ContractValidationError):
        validate_instance(data, load_schema('limits'))
