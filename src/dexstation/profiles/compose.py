"""Load workstation/task profiles and compose their backend-neutral scene.

Composition performs structural checks only. Backend self-checks, collision
queries for initial poses and randomization corners, home-robot overlaps and
keep-out checks belong to the backend/limits at session start (§5.1, ruling 4).
No hardware, physics backend or task plugin is started by this module.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from dexstation.backends.registry import backend_names, get_backend
from dexstation.runtime.plugins import task_plugin, activate_task
from dexstation.profiles.calibration import load_calibration, load_teleop_calibration
from dexstation.contracts.validation import _validate
from dexstation.contracts.versions import WORKBENCH_SCHEMA
from dexstation.backends.object_types import object_type, object_types
from dexstation.backends.channel_types import channel_type, channel_types
from dexstation.contracts.profiles import CameraPair, ChannelSpec, EMPTY_TASK, ObjectSpec, RuntimeScene
from dexstation.config.paths import resolve_workspace_path, package_path
from dexstation.model.mount_geometry import j1_offset_deg
from dexstation.config.site import repository_root
from dexstation.model.spec import compose as compose_pose, pose_to_matrix, spec_from_dict

ROOT = repository_root()
CONTRACTS = Path(__file__).parents[1] / 'contracts'


class _Profile(dict):
    """A JSON dict with source location kept outside the profile's schema."""

    def __init__(self, data, path, reference_paths=()):
        super().__init__(data)
        self.path = Path(path)
        self.reference_paths = tuple(reference_paths)


def _json_text(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(data):
    return hashlib.sha256(_json_text(data).encode('utf-8')).hexdigest()


def referenced_files(profile, base):
    """Byte checksums of profile references; only WP5 placeholders may be absent."""
    base = Path(base)
    paths = list(getattr(profile, 'reference_paths', ()))
    optional = set()
    if profile['schema'] == WORKBENCH_SCHEMA:
        paths.extend(profile['calibration'].values())
        paths.extend(c['name'] for c in profile['manifest']['components'].values()
                     if not c['name'].startswith('workspace://'))
        paths.extend(arm['base_orientation'] for key, arm in profile['robots'].items()
                     if key.endswith('_arm'))
        optional.update((profile['limits'], profile['takeover']))
        paths.extend(optional)
    elif isinstance(profile['fixtures'], str):
        paths.append(profile['fixtures'])
    if profile.get('appearance'):
        paths.append(profile['appearance'])
    if profile['schema'] == WORKBENCH_SCHEMA:
        paths.append(profile['robots']['motion_model'])
    result = {}
    for value in sorted(set(paths)):
        path = _reference(value, base)
        if not path.exists() and value in optional and not path.is_symlink():
            result[value] = None
        else:
            result[value] = hashlib.sha256(_regular(path).read_bytes()).hexdigest()
    return result


def profile_digest(profile, base):
    """Ruling 5 canonical digest, including referenced bytes and pinned assets."""
    content = deepcopy(dict(profile))
    if 'manifest' in content:
        content['manifest'].pop('sha256', None)
    content['referenced_files'] = referenced_files(profile, base)
    # The ledger pins external asset versions/SHAs; asset files are never hashed.
    ledger_path = Path(base) / 'sources/external_resources.json'
    if not ledger_path.is_file():
        ledger_path = ROOT / 'references/external_resources.json'
        if not ledger_path.is_file():
            ledger_path = Path(__file__).parents[1] / 'examples/ar5_g20/data/sources/external_resources.json'
    resources = _read(ledger_path)['resources']
    content['external_assets'] = {r['id']: {
        'revision': r.get('revision', r.get('version')),
        'files': {f['path']: f['sha256'] for f in r.get('files', []) if 'sha256' in f},
    } for r in resources}
    return _digest(content)


def _schema(data, name):
    schema = json.loads((CONTRACTS / f'schemas/{name}.schema.json').read_text())
    if name == 'task':
        entries = object_types()
        objects = schema['properties']['objects']['items']
        objects['properties']['type']['enum'] = list(entries)
        objects['oneOf'] = [dict(properties=dict(entry.schema, type={'const': entry.name}))
                            for entry in entries.values()]
        schema['properties']['observations']['items']['properties']['kind']['enum'] = list(channel_types())
    _validate(data, schema)
    if name == 'task':
        from dexstation.runtime.plugins import CORE_PANES
        cameras = data.get('auxiliary_cameras', [])
        ids = [c['id'] for c in cameras]
        slots = [c['atlas_slot'] for c in cameras]
        if len(set(ids)) != len(ids) or 'table' in ids or len(set(slots)) != len(slots):
            raise ValueError('Duplicate auxiliary camera id or atlas slot')
        objects = {o['name'] for o in data['objects']}
        for camera in cameras:
            if camera['object'] not in objects:
                raise ValueError(f"Unknown auxiliary camera object: {camera['object']}")
        panes = data.get('panes', [])
        pane_ids = [p['id'] for p in [*CORE_PANES, *panes]]
        if len(set(pane_ids)) != len(pane_ids):
            raise ValueError('Duplicate grid pane id')
        for pane in panes:
            if 'camera' in pane and pane['camera'] not in {*ids, 'table'}:
                raise ValueError(f"Unknown pane camera: {pane['camera']}")


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f'Profile resource must be a regular file: {path}')
    if not path.is_file():
        raise ValueError(f'Profile resource not found: {path}')
    return path


def _read(path):
    path = _regular(path)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path.name}: invalid JSON at line {exc.lineno}') from exc


def _base(profile, kind):
    default = package_path('configs/' + kind + '/' + (profile['manifest']['name'] if kind == 'workbenches' else profile['id']))
    return profile.path.parent if isinstance(profile, _Profile) else default


def _reference(value, base):
    if value.startswith('workspace://'):
        return Path(resolve_workspace_path(value))
    path = Path(value)
    if value.startswith('configs/'):
        return package_path(value)
    if value.startswith('motion_assets/'):
        return package_path(value)
    return path if path.is_absolute() else base / path


def load_workbench(path: str | Path) -> dict:
    """Load a workbench JSON (or directory), validating its schema/checksum."""
    path = Path(path)
    if path.is_dir():
        path = path / 'workbench.json'
    data = _read(path)
    _schema(data, 'workbench')
    if profile_digest(data, path.parent) != data['manifest']['sha256']:
        raise ValueError('Workbench manifest sha256 mismatch')
    for filename in data['calibration'].values():
        _schema(_read(_reference(filename, path.parent)), 'calibration')
    return _Profile(data, path.absolute())


def load_task(task_id_or_path: str | Path) -> dict:
    """Load a task; static fixtures become objects, in their original order."""
    value = str(task_id_or_path)
    if value == 'empty':
        return _Profile(deepcopy(EMPTY_TASK), Path(__file__).parents[1] / 'task_plugins/empty/data/task.json')
    path = Path(value)
    if re.fullmatch(r'[a-z][a-z0-9_]*', value):
        entry = task_plugin(value)
        path = Path(entry.data_directory) / 'task.json' if entry else package_path('configs/tasks/' + value + '/task.json')
    elif path.is_dir():
        path = path / 'task.json'
    data = _read(path)
    activate_task(data['id'])
    _schema(data, 'task')
    fixtures = data['fixtures']
    reference_paths = (fixtures,) if isinstance(fixtures, str) else ()
    if isinstance(fixtures, str):
        fixtures = _read(_reference(fixtures, path.parent))
        fixture_schema = json.loads((CONTRACTS / 'schemas/task.schema.json').read_text())
        _validate(fixtures, fixture_schema['properties']['fixtures']['anyOf'][0])
    # Static supports come first, preserving the old construction stages.
    data['objects'] = fixtures + data['objects']
    data['fixtures'] = []
    _schema(data, 'task')
    return _Profile(data, path.absolute(), reference_paths)


def _frames(workbench):
    frames = {workbench['world_frame']: (0., 0., 0., 1., 0., 0., 0.)}
    active = set()

    def resolve(name):
        if name in frames:
            return frames[name]
        if name in active:
            raise ValueError(f'Frame cycle: {name}')
        if name not in workbench['frames']:
            raise ValueError(f'Unknown workbench frame: {name}')
        active.add(name)
        frame = workbench['frames'][name]
        frames[name] = compose_pose(resolve(frame['parent']), frame['pose'])
        active.remove(name)
        return frames[name]

    for name in workbench['frames']:
        resolve(name)
    return frames


def _sample_pose(obj, frames, rng):
    initial = obj['initial_pose']
    if initial['frame'] not in frames:
        raise ValueError(f'Unknown workbench frame: {initial["frame"]}')
    local = list(initial['pose'])
    pose_to_matrix(local)  # Validate without changing the legacy numeric values.
    randomize = obj.get('randomize', {})
    if not randomize and initial['frame'] == next(iter(frames)):
        # Preserve the old scene's numbers; spec_from_dict normalizes once.
        return tuple(local)
    if set(randomize) - {'xy_m', 'yaw_deg'}:
        raise ValueError(f'Unknown randomization field for {obj["name"]}')
    for key, count in (('xy_m', 2), ('yaw_deg', 1)):
        if key not in randomize:
            continue
        intervals = np.asarray(randomize[key], dtype=float)
        shape = (2, 2) if count == 2 else (2,)
        if intervals.shape != shape or not np.isfinite(intervals).all() or np.any(intervals[..., 0] > intervals[..., 1]):
            raise ValueError(f'Invalid randomization interval: {obj["name"]}.{key}')
        sampled = rng.uniform(intervals[..., 0], intervals[..., 1])
        if key == 'xy_m':
            local[:2] = (np.asarray(local[:2]) + sampled).tolist()
        else:
            angle = math.radians(float(sampled)) / 2
            local = list(compose_pose(local, (0, 0, 0, math.cos(angle), 0, 0, math.sin(angle))))
    return compose_pose(frames[initial['frame']], local)


def _static_fixture(obj, pose):
    geometry = obj['geometry']
    kind = geometry.get('fixture_type', 'box' if 'box' in geometry else 'mesh')
    fixture = dict(name=obj['name'], type=kind, pose=list(pose),
                   collision=obj['collision']['robots'], status=obj.get('status', 'design'),
                   source=obj.get('source', f'task object {obj["name"]}'))
    if kind == 'box':
        fixture['half_extents'] = geometry['box']['half_extents']
    else:
        fixture['mesh'] = geometry['mesh']
        if 'convex_mesh' in geometry:
            fixture['convex_mesh'] = geometry['convex_mesh']
    if 'parent' in obj:
        fixture['parent'] = obj['parent']
    return fixture


def _project_none(obj, pose, data):
    pass


def _project_primitive(obj, pose, data):
    if object_type(obj['type']).static:
        data['fixtures'].append(_static_fixture(obj, pose))




def _channels(task):
    channels = []
    for channel in task['observations']:
        shape, dtype, unit = channel_type(channel['kind']).describe(channel)
        channels.append(ChannelSpec(channel['name'], channel['kind'], shape, dtype, unit, channel.get('object')))
    return tuple(channels)


def _code_identity():
    """Source checkouts have Git identity; installed wheels have no checkout."""
    if not (ROOT / '.git').exists():
        return dict(git_commit=None, dirty=None)
    commit = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True)
    if commit.returncode:
        return dict(git_commit=None, dirty=None)
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    return dict(git_commit=commit.stdout.strip(), dirty=bool(status))


def compose(workbench: dict, task: dict, backend: str, *, seed: int | None = None) -> RuntimeScene:
    """Compose profiles without starting plugins, physics or hardware.

    Backend self-checks and initial/random-corner collision, home and keep-out
    validation are performed at session start by the backend and limits, not
    here. Static objects are also projected into spec.fixtures for transition;
    consumers must avoid creating the same named static object twice.
    """
    _schema(workbench, 'workbench')
    _schema(task, 'task')
    if task['fixtures']:
        raise ValueError('Load task fixtures with load_task before composing')
    name = workbench['manifest']['name']
    if '*' not in task['workbenches'] and name not in task['workbenches']:
        raise ValueError(f'Task {task["id"]} does not support workbench {name}')
    if backend not in backend_names(online=True) or backend not in workbench['physics'] or backend not in task['backends']:
        raise ValueError(f'Unsupported online backend: {backend}')
    workbench_base, task_base = _base(workbench, 'workbenches'), _base(task, 'tasks')
    if profile_digest(workbench, workbench_base) != workbench['manifest']['sha256']:
        raise ValueError('Workbench manifest sha256 mismatch')
    teleop_path = _regular(_reference(workbench['calibration']['teleop'], workbench_base))
    real_path = _regular(_reference(workbench['calibration']['real_robot'], workbench_base))
    teleop_facts, real_facts = _read(teleop_path), _read(real_path)
    _schema(teleop_facts, 'calibration')
    _schema(real_facts, 'calibration')
    robots = workbench['robots']
    sides = robots['sides']
    if sides != ['left'] and sides != ['left', 'right']:
        raise ValueError('Workbench sides must be left or left/right')
    data = dict(name=f'{name}/{task["id"]}', world_frame=workbench['world_frame'],
                table=deepcopy(workbench['geometry']['table']), notes=[],
                fixtures=deepcopy(workbench['geometry']['fixtures']), cameras=[])
    for side in sides:
        arm, hand = robots[f'{side}_arm'], robots[f'{side}_hand']
        facts = _read(_reference(arm['base_orientation'], workbench_base))
        data[f'{side}_arm'] = {key: arm[key] for key in ('urdf', 'base_link', 'tip_link', 'joint_names')}
        data[f'{side}_arm'].update(base_pose=[*arm['base_position'], *facts['measured_base_orientation_wxyz'][side]],
                                   home_q=list(robots['home']['q_at_teleop_reference_mount'][side]))
        data[f'{side}_hand'] = {key: hand[key] for key in ('urdf', 'root_link', 'mount', 'mount_status')}
    reference_spec = spec_from_dict(data)
    teleop_offsets = load_teleop_calibration(teleop_path, reference_spec)
    real_offsets = load_calibration(real_path, reference_spec)
    initial_hands = task.get('initial_robot_state', {}).get('hand_q', {})
    if set(initial_hands) - set(sides):
        raise ValueError('Initial hand state references a missing robot side')
    for side in sides:
        phi = j1_offset_deg(data[f'{side}_arm']['base_pose'][3:],
                            teleop_facts['reference_base_orientation_wxyz'][side])
        data[f'{side}_arm']['home_q'][0] += math.radians(phi)
        if side in initial_hands:
            data[f'{side}_hand']['home_q'] = list(initial_hands[side])
    cameras = tuple(CameraPair(c['name'], deepcopy(c['real']), deepcopy(c['sim'])) for c in workbench['cameras'])
    for camera in cameras:
        if camera.sim is not None:
            data['cameras'].append(dict(camera.sim, name=camera.name,
                serial=camera.name, model=camera.real['model']))
    data['appearance'] = task.get('appearance', workbench['appearance'])
    frames = _frames(workbench)
    randomized = any(o.get('randomize') for o in task['objects'])
    actual_seed = int(np.random.SeedSequence().entropy) if seed is None and randomized else seed
    rng = np.random.default_rng(actual_seed)
    objects = []
    names = {f['name'] for f in data['fixtures']}
    if len(names) != len(data['fixtures']):
        raise ValueError('Duplicate workbench fixture name')
    for obj in task['objects']:
        if obj['name'] in names:
            raise ValueError(f'Duplicate scene object name: {obj["name"]}')
        names.add(obj['name'])
        pose = _sample_pose(obj, frames, rng)
        geometry = deepcopy(obj['geometry'])
        extra = {k: deepcopy(v) for k, v in obj.items() if k not in {
            'name', 'type', 'geometry', 'inertial', 'material', 'initial_pose', 'randomize', 'collision'}}
        objects.append(ObjectSpec(obj['name'], obj['type'], geometry, deepcopy(obj.get('inertial', {})),
            deepcopy(obj.get('material', {})), pose, deepcopy(obj.get('randomize')), deepcopy(obj['collision']), extra))
        object_type(obj['type']).invoke('compose', globals(), obj, pose, data)
    object_names = {o.name for o in objects}
    for channel in task['observations']:
        if channel.get('object') is not None and channel['object'] not in object_names:
            raise ValueError(f'Unknown observation object: {channel["object"]}')
    for action in task['actions'].values():
        if set(action.get('objects', [])) - object_names:
            raise ValueError('Task action references an unknown object')
    if task['evaluation'] is not None:
        for evaluation in task['evaluation'].values():
            if isinstance(evaluation, dict) and evaluation.get('object') not in object_names:
                if 'object' in evaluation:
                    raise ValueError('Task evaluation references an unknown object')
    channels = _channels(task)
    if len({c.name for c in channels}) != len(channels):
        raise ValueError('Duplicate observation channel name')
    # Keep portable, pre-normalization geometry for exact migration review.
    scene_text = _json_text(data)
    for key in ('left_arm', 'right_arm', 'left_hand', 'right_hand'):
        if key in data:
            data[key]['urdf'] = resolve_workspace_path(data[key]['urdf'])
    for fixture in data['fixtures']:
        for key in ('mesh', 'convex_mesh'):
            if key in fixture:
                fixture[key] = resolve_workspace_path(fixture[key])
    spec = replace(spec_from_dict(data), source_dir=ROOT)
    calibration = dict(teleop=teleop_offsets, real_robot=real_offsets,
        measured_rotation_deg_about_vendor=real_facts.get('measured_rotation_deg_about_vendor'),
        base_position_delta_m=None,
        base_position_delta_note='Calibration facts do not store reference base positions',
        sha256={key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in (
            ('teleop', teleop_path), ('real_robot', real_path))})
    identity = dict(workbench={k: workbench['manifest'][k] for k in ('name', 'revision', 'sha256')},
        task=dict(id=task['id'], version=task['version'], sha256=profile_digest(task, task_base)),
        backend=dict(name=backend, version=get_backend(backend).version),
        code=_code_identity(),
        scene_sha256=_digest(dict(scene=json.loads(scene_text), objects=[o.__dict__ for o in objects], seed=actual_seed)))
    return RuntimeScene(spec, deepcopy(dict(workbench)), deepcopy(dict(task)), tuple(objects), channels,
        cameras, tuple(deepcopy(task['cells'])), deepcopy(task['actions']),
        _reference(workbench['limits'], workbench_base), _reference(workbench['takeover'], workbench_base),
        calibration, identity, dict(workbench_text=_json_text(workbench), task_text=_json_text(task),
            workbench_referenced_files=referenced_files(workbench, workbench_base),
            task_referenced_files=referenced_files(task, task_base), seed=actual_seed,
            scene_text=scene_text))
