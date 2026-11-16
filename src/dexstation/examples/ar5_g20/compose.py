"""Compose a station scene from the vendor's standard workstation plus recorded deviations.

The vendor baseline (linker-sim-assets ``ar5_08_l25_bench_bimanual`` and its ``bench_table``)
is extracted once into ``references/vendor_bench_baseline.json``. A scene is then

    vendor baseline  +  deviations (configs/workcell/vendor_bench_overrides.json)  +  our layers

where "our layers" are the parts the vendor model does not have (gamepad, fixture, cameras,
home poses, floor, backdrop), taken unchanged from a base scene. Every value that differs
from the vendor must appear in the overrides file with its evidence.
"""
from dexstation.contracts.versions import WORKBENCH_SCHEMA, WORKBENCH_MANIFEST_VERSION
import hashlib
from copy import deepcopy
import math
import posixpath
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexstation.model.mount_geometry import j1_offset_deg

FRAME_PREFIX = 'station_table_frame_'
COLUMN_NAMES = {'bench_table_arm_column': 'station_column',
                'bench_table_mount_left': 'station_column_mount_left',
                'bench_table_mount_right': 'station_column_mount_right',
                'bench_table_mount_crossbar': 'station_column_mount_crossbar'}
TABLETOP = 'bench_table_tabletop'
SIDES = ('left', 'right')
ARM_TAGS = {'left': '08L', 'right': '08R'}
HAND_ROOTS = {'left': 'lh_hand_base_link', 'right': 'rh_hand_base_link'}


def _rpy_matrix(rpy):
    r, p, y = rpy
    cx, sx, cy, sy, cz, sz = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def _quat_wxyz(m):
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = np.copysign(x, m[2, 1] - m[1, 2]); y = np.copysign(y, m[0, 2] - m[2, 0]); z = np.copysign(z, m[1, 0] - m[0, 1])
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def _world_poses(urdf):
    parent = {}
    for joint in ET.parse(urdf).getroot().findall('joint'):
        origin = joint.find('origin'); m = np.eye(4)
        if origin is not None:
            m[:3, :3] = _rpy_matrix([float(v) for v in origin.get('rpy', '0 0 0').split()])
            m[:3, 3] = [float(v) for v in origin.get('xyz', '0 0 0').split()]
        parent[joint.find('child').get('link')] = (joint.find('parent').get('link'), m)
    poses = {}
    def pose(link):
        if link not in poses:
            poses[link] = np.eye(4) if link not in parent else pose(parent[link][0]) @ parent[link][1]
        return poses[link]
    for link in parent:
        pose(link)
    return poses


def _stl_bounds(path):
    data = Path(path).read_bytes(); count = struct.unpack('<I', data[80:84])[0]
    if 84 + 50 * count != len(data):
        raise ValueError(f'{path} is not a binary STL')
    v = np.frombuffer(data[84:], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', 9), ('a', '<u2')]))['v'].reshape(-1, 3)
    return v.min(0).astype(float), v.max(0).astype(float)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pose(xyz, quat):
    return [round(float(v), 9) for v in (*xyz, *quat)]


def extract_baseline(root, revision):
    """Read the downloaded vendor files under ``root`` (layout of the ``bench`` resource)."""
    root = Path(root)
    base_urdf = root / 'bench_table/base.urdf'
    bench_urdf = root / 'workstation/workstation.urdf'
    parts = []
    for c in ET.parse(base_urdf).getroot().find('link').findall('collision'):
        name = c.get('name'); origin = c.find('origin'); geometry = c.find('geometry')
        xyz = [float(v) for v in origin.get('xyz').split()] if origin is not None else [0.0, 0.0, 0.0]
        rpy = [float(v) for v in origin.get('rpy').split()] if origin is not None else [0.0, 0.0, 0.0]
        pose = _pose(xyz, _quat_wxyz(_rpy_matrix(rpy)))
        box, mesh = geometry.find('box'), geometry.find('mesh')
        if box is not None:
            parts.append(dict(name=name, type='box', pose=pose,
                              half_extents=[round(float(v) / 2, 9) for v in box.get('size').split()]))
        else:
            parts.append(dict(name=name, type='mesh', pose=pose, mesh=mesh.get('filename')))
    top = next(p for p in parts if p['name'] == TABLETOP)
    lo, hi = _stl_bounds(root / 'bench_table' / top['mesh'])
    world = _world_poses(bench_urdf)
    arms, hands = {}, {}
    for side in SIDES:
        prefix = f'arm_{side}_AR5_5_{ARM_TAGS[side]}_W4C4A6_'
        base = world[prefix + 'base']
        arms[side] = dict(base_link=f'AR5_5_{ARM_TAGS[side]}_W4C4A6_base', base_pose=_pose(base[:3, 3], _quat_wxyz(base[:3, :3])))
        mount = np.linalg.inv(world[prefix + 'tcp']) @ world[f'hand_{side}_{HAND_ROOTS[side]}']
        hands[side] = dict(mount=_pose(mount[:3, 3], _quat_wxyz(mount[:3, :3])))
    files = sorted([base_urdf, bench_urdf, root / 'bench_table/collision_manifest.json'] +
                   [root / 'bench_table' / p['mesh'] for p in parts if p['type'] == 'mesh'])
    return dict(
        source=dict(repository='https://github.com/linker-bot/linker-sim-assets', revision=revision,
                    workstation='src/linker_robot_assets/assets/workstations/ar5_08_l25_bench_bimanual',
                    table='src/linker_robot_assets/assets/components/bases/bench_table/variants/default',
                    sha256={str(f.relative_to(root)): _sha256(f) for f in files}),
        tabletop=dict(center=[round(float(v), 9) for v in (lo + hi) / 2],
                      half_extents=[round(float(v), 9) for v in (hi - lo) / 2]),
        parts=parts, arms=arms, hands=hands)


def _fixture_name(vendor_name):
    return COLUMN_NAMES.get(vendor_name, FRAME_PREFIX + vendor_name.removeprefix('bench_table_'))


def rotate_about_base_z(quaternion_wxyz, degrees):
    """Quaternion of a frame rotated by ``degrees`` about its own z axis."""
    w1, x1, y1, z1 = (float(v) for v in quaternion_wxyz)
    half = math.radians(degrees) / 2
    w2, z2 = math.cos(half), math.sin(half)
    return [w1*w2 - z1*z2, x1*w2 + y1*z2, y1*w2 - x1*z2, w1*z2 + z1*w2]


def compose_scene(base_scene, baseline, overrides, teleop_reference):
    """Return a new scene dict; ``base_scene`` supplies only the layers the vendor lacks."""
    scene = {k: v for k, v in base_scene.items()}
    scene['name'] = overrides['scene_name']
    top = baseline['tabletop']
    scene['table'] = dict(pose=[*top['center'], 1.0, 0.0, 0.0, 0.0], half_extents=list(top['half_extents']))
    keep = [f for f in base_scene['fixtures'] if f['name'] not in set(overrides['replaced_fixtures'])]
    unknown = set(overrides['replaced_fixtures']) - {f['name'] for f in base_scene['fixtures']}
    if unknown:
        raise ValueError(f'replaced_fixtures not in base scene: {sorted(unknown)}')
    vendor_fixtures = []
    for part in baseline['parts']:
        if part['name'] in overrides['vendor_parts_not_used']:
            continue
        fixture = dict(name=_fixture_name(part['name']), type=part['type'], pose=list(part['pose']), collision=True,
                       status='design', source=f"linker-sim-assets {baseline['source']['revision'][:8]} bench_table collision part {part['name']}")
        if part['type'] == 'box':
            fixture['half_extents'] = list(part['half_extents'])
        else:
            fixture['mesh'] = f"{overrides['bench_workspace']}/bench_table/{part['mesh']}"
        vendor_fixtures.append(fixture)
    scene['fixtures'] = vendor_fixtures + keep
    for side in SIDES:
        arm = dict(base_scene[f'{side}_arm']); tag = ARM_TAGS[side]
        names = overrides['arms'][side]
        arm['urdf'] = names['urdf']
        prefix = f'AR5_{side}_5_{tag}_W4C4A6_'
        arm['base_link'], arm['tip_link'] = prefix + 'base', prefix + 'tcp'
        arm['joint_names'] = [f'{prefix}joint_{i}' for i in range(1, 8)]
        # Rotate the vendor mount about the arm's own base z axis, which is the J1 axis.
        rotation = float(overrides['arm_base_rotation_deg'][side])
        vendor = baseline['arms'][side]['base_pose']
        arm['base_pose'] = [*vendor[:3], *(round(v, 12) for v in rotate_about_base_z(vendor[3:], rotation))]
        # Keep the home pose where it was in the world for the mount teleop was calibrated on:
        # a base rotation about J1 equals the same J1 offset, so J1 absorbs the difference.
        offset = j1_offset_deg(arm['base_pose'][3:], teleop_reference['reference_base_orientation_wxyz'][side])
        arm['home_q'] = [float(arm['home_q'][0]) + math.radians(offset), *map(float, arm['home_q'][1:])]
        scene[f'{side}_arm'] = arm
        hand = dict(base_scene[f'{side}_hand'])
        hand['mount'] = list(baseline['hands'][side]['mount'])
        hand['mount_status'] = 'unverified'
        scene[f'{side}_hand'] = hand
    scene['gamepad'] = dict(base_scene['gamepad'], release=overrides['gamepad_release'])
    scene['notes'] = list(overrides['notes'])
    return scene


def split_profiles(base_scene, baseline, overrides, teleop_reference, gamepad_task):
    """Generate workbench facts and static task fixtures from the pinned inputs.

    The 50 station fixtures precede the 69 v3.3 task pieces in the old scene.
    Meshes stay ordinary-file resources in our configured asset workspace;
    this transformation neither opens upstream files nor changes geometry.
    """
    scene = compose_scene(base_scene, baseline, overrides, teleop_reference)
    source = baseline['source']
    robots = dict(sides=list(SIDES), home=dict(
        intent='both arms horizontal, pointing world +x',
        q_at_teleop_reference_mount={side: list(base_scene[f'{side}_arm']['home_q']) for side in SIDES}),
        hand_mapping='station-g20-20261014', motion_model='motion_assets/station.json')
    for side in SIDES:
        arm, hand = scene[f'{side}_arm'], scene[f'{side}_hand']
        robots[f'{side}_arm'] = {k: deepcopy(arm[k]) for k in ('urdf', 'base_link', 'tip_link', 'joint_names')}
        robots[f'{side}_arm'].update(model=f'AR5-5_0.8{side[0].upper()}',
            base_position=list(arm['base_pose'][:3]), base_orientation='calibration/real_robot.json')
        robots[f'{side}_hand'] = {k: deepcopy(hand[k]) for k in ('urdf', 'root_link', 'mount', 'mount_status')}
        robots[f'{side}_hand'].update(model='G20', geometry='L25 (g20_pbr_description v0.1.1)')
    cameras = []
    for camera in scene['cameras']:
        cameras.append(dict(name=camera['name'], real=dict(
            port_key=f'camera.usb_ports.{camera["name"]}', model=camera['model'],
            start_by_default=camera['name'] == 'camera_3'),
            sim={k: deepcopy(v) for k, v in camera.items() if k not in ('name', 'serial', 'model')}))
    station, fixtures = [], []
    for fixture in scene['fixtures']:
        if not fixture['name'].startswith(('v3_proxy_', 'v3_visual_')):
            station.append(deepcopy(fixture))
            continue
        geometry = dict(fixture_type=fixture['type'])
        for key in ('mesh', 'convex_mesh'):
            if key in fixture:
                geometry[key] = fixture[key]
        if fixture['type'] == 'box':
            geometry['box'] = dict(half_extents=list(fixture['half_extents']))
        obj = dict(name=fixture['name'], type='static', geometry=geometry,
            initial_pose=dict(frame=scene['world_frame'], pose=list(fixture['pose'])),
            collision=dict(robots=fixture['collision'], environment=False, objects=False),
            status=fixture['status'], source=fixture['source'])
        if 'parent' in fixture:
            obj['parent'] = fixture['parent']
        fixtures.append(obj)
    top = scene['table']
    table_top = [*top['pose'][:3], *top['pose'][3:]]
    table_top[2] += top['half_extents'][2]
    workbench = dict(schema=WORKBENCH_SCHEMA, manifest=dict(
        schema_version=WORKBENCH_MANIFEST_VERSION, name='ar5_08_g20_bench_bimanual', revision='2026-11-12', composer_version='0.1.0',
        upstream=dict(repository=source['repository'], revision=source['revision'], workstation='ar5_08_l25_bench_bimanual'),
        components={}), world_frame=scene['world_frame'],
        geometry=dict(generated_by='scripts/vendor_bench_scene.py compose --workbench',
                      table=deepcopy(top), fixtures=station),
        frames=dict(table_top=dict(parent=scene['world_frame'], pose=table_top)),
        robots=robots, cameras=cameras,
        calibration=dict(teleop='calibration/teleop.json', real_robot='calibration/real_robot.json'),
        limits='limits.json', takeover='takeover.json',
        physics=dict(sapien=dict(timestep_s=0.001), mujoco=dict(timestep_s=0.001)),
        real_robot=dict(arms=dict(left_ip_key='real_robot.left_arm_ip', right_ip_key='real_robot.right_arm_ip',
            local_ip_key='real_robot.local_ip', reader_key='real_robot.arm_reader'),
            hands=dict(left=dict(can_key='station.left_can', can_id='0x28'),
                       right=dict(can_key='station.right_can', can_id='0x27')), impedance=None),
        estop_script_key='station.estop_script',
        appearance=posixpath.normpath(str(Path(overrides['output_scene']).parent / scene['appearance'])))
    task = deepcopy(gamepad_task)
    old = scene['gamepad']
    task['objects'] = [dict(name='gamepad', type='xbox_gamepad_physics', geometry=dict(
        release=old['release'], profile=old['profile'], collision_mesh=old['collision'], fixed_root=old['fixed_root']),
        initial_pose=dict(frame=scene['world_frame'], pose=list(old['root_pose'])),
        collision=dict(robots=True, environment=True, objects=False))]
    task['initial_robot_state'] = dict(hand_q={side: list(scene[f'{side}_hand']['home_q']) for side in SIDES
                                             if any(scene[f'{side}_hand'].get('home_q', []))})
    return workbench, task, fixtures
