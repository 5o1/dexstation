"""WP7 contracts, portable data and collision semantics, without hardware."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dexstation.contracts.profiles import EMPTY_TASK, ObjectSpec, RuntimeScene
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES, G20_MIMIC_JOINTS
from dexstation.model.spec import ArmSpec, HandSpec, TableSpec, WorkcellSpec, GamepadSpec, FixtureSpec
from dexstation.profiles import export

ROOT = Path(__file__).resolve().parents[1]


def scene_fixture(tmp_path, task='empty'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    arm = ET.Element('robot', name='arm')
    for i in range(8):
        link = ET.SubElement(arm, 'link', name=f'a{i}')
        _link_geometry(link)
        if i:
            joint = ET.SubElement(arm, 'joint', name=f'j{i}', type='revolute')
            ET.SubElement(joint, 'parent', link=f'a{i-1}')
            ET.SubElement(joint, 'child', link=f'a{i}')
            ET.SubElement(joint, 'origin', xyz='.06 0 0')
            ET.SubElement(joint, 'axis', xyz='0 0 1' if i % 2 else '0 1 0')
            ET.SubElement(joint, 'limit', lower='-2', upper='2', effort='500', velocity='3')
    ET.ElementTree(arm).write(tmp_path / 'arm.urdf')
    for side in ('left', 'right'):
        hand = ET.Element('robot', name='hand')
        _link_geometry(ET.SubElement(hand, 'link', name='palm'))
        parents = {f: 'palm' for f in ('thumb', 'index', 'middle', 'ring', 'pinky')}
        for i, name in enumerate(G20_JOINT_NAMES):
            finger = name.split('_')[0]
            terminal = name.endswith(('_ip', '_dip'))
            child = finger + '_distal' if terminal else 'h_' + name
            _link_geometry(ET.SubElement(hand, 'link', name=child))
            joint = ET.SubElement(hand, 'joint', name=name, type='revolute')
            ET.SubElement(joint, 'parent', link=parents[finger])
            ET.SubElement(joint, 'child', link=child)
            ET.SubElement(joint, 'origin', xyz=f'.03 {(i % 5) * .01 if parents[finger] == "palm" else 0} 0')
            ET.SubElement(joint, 'axis', xyz='0 1 0')
            ET.SubElement(joint, 'limit', lower='-2', upper='2', effort='60', velocity='3')
            if name in G20_MIMIC_JOINTS[side]:
                source, multiplier = G20_MIMIC_JOINTS[side][name]
                ET.SubElement(joint, 'mimic', joint=source, multiplier=str(multiplier))
            parents[finger] = child
        ET.ElementTree(hand).write(tmp_path / (side + '.urdf'))
    arm_spec = ArmSpec(str(tmp_path/'arm.urdf'), 'a0', 'a7', (0., -1., 2., 1., 0., 0., 0.), (0.,)*7, tuple(f'j{i}' for i in range(1, 8)))
    left = HandSpec(str(tmp_path/'left.urdf'), 'palm', (.05,0,0,1,0,0,0), 'unverified')
    right = replace(left, urdf=str(tmp_path/'right.urdf'))
    spec = WorkcellSpec('contract-bench', 'world', TableSpec((0.,0.,0.,1.,0.,0.,0.), (.5,.5,.01)),
        arm_spec, left, None, (), right_arm=replace(arm_spec, base_pose=(0.,1.,2.,1.,0.,0.,0.)), right_hand=right)
    limits = json.loads((ROOT/'src/dexstation/contracts/examples/limits.json').read_text())
    (tmp_path/'limits.json').write_text(json.dumps(limits))
    task_dict = dict(EMPTY_TASK)
    objects = ()
    if task == 'gamepad':
        gamepad = GamepadSpec('xbox_gamepad_physics==0.4.3', 'full_controls', 'sdf', (0.,0.,1.,1.,0.,0.,0.), True)
        spec = replace(spec, gamepad=gamepad)
        objects = (ObjectSpec('gamepad', 'xbox_gamepad_physics', {'collision_mesh': 'sdf'}, {}, {}, gamepad.root_pose, None,
            {'robots': True, 'environment': True, 'objects': True}, {'profile': 'full_controls', 'fixed_root': True}),)
        task_dict = dict(task_dict, id='xbox_gamepad_v3_3')
    return RuntimeScene(spec, {'physics': {'sapien': {'timestep_s': .001}, 'mujoco': {'timestep_s': .001}}}, task_dict,
        objects, (), (), (), {}, tmp_path/'limits.json', tmp_path/'takeover.json',
        {'measured_rotation_deg_about_vendor': {'left': -84.1, 'right': 85.85}, 'offset': [0., 0.]},
        {'workbench': 'contract-bench', 'task': task_dict['id']}, {'workbench_text': '{}', 'task_text': json.dumps(task_dict)})


def _link_geometry(link):
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'mass', value='.01')
    ET.SubElement(inertial, 'inertia', ixx='.00001', iyy='.00001', izz='.00001', ixy='0', ixz='0', iyz='0')
    geometry = ET.SubElement(ET.SubElement(link, 'collision'), 'geometry')
    ET.SubElement(geometry, 'box', size='.01 .01 .01')


def object_spec(name, flags, kind='rigid'):
    return ObjectSpec(name, kind, {'box': {'half_extents': [.01]*3}}, {'mass_kg': .1, 'inertia_diag': [.00001]*3},
        {}, (0.,0.,1.,1.,0.,0.,0.), None, dict(zip(('robots','environment','objects'), flags)), {})


def test_collision_boolean_truth_table_and_native_masks():
    fixture = FixtureSpec('support', 'box', (0,0,0,1,0,0,0), True, 'design', 'test', (.1,.1,.1))
    for flags_a, flags_b in itertools.product(itertools.product((False, True), repeat=3), repeat=2):
        a, b = object_spec('a', flags_a), object_spec('b', flags_b)
        objects = (a, b)
        pairs = export.allowed_collision_pairs(objects, (fixture,))
        pair = lambda x,y: tuple(sorted((x,y))) in pairs
        assert pair('object:a','object:b') == (flags_a[2] and flags_b[2])
        assert pair('object:a','robot:left') == flags_a[0]
        assert pair('object:a','environment:table') == flags_a[1]
        assert pair('object:a','environment:support') == flags_a[1]
        masks = export.collision_masks(objects, (fixture,))
        for x,y in itertools.combinations(masks, 2):
            ax, ay = masks[x], masks[y]
            assert bool((ax[0] & ay[1]) | (ay[0] & ax[1])) == pair(x,y)
        assert masks['robot:left'][0] & masks['robot:left'][1] == 0
        assert pair('robot:left', 'robot:right')


def test_legacy_gamepad_support_exclusion_and_geometry_transition():
    gamepad = replace(object_spec('gamepad', (True,True,True)), type='xbox_gamepad_physics', geometry={'collision_mesh':'coacd'})
    fixture = FixtureSpec('support', 'box', (0,0,0,1,0,0,0), True, 'design', 'test', (.1,.1,.1))
    pairs = export.allowed_collision_pairs((gamepad,), (fixture,))
    assert tuple(sorted(('object:gamepad','environment:support'))) not in pairs
    assert tuple(sorted(('object:gamepad','environment:table'))) in pairs
    assert export.gamepad_collision_mesh(gamepad) == 'coacd'
    assert export.gamepad_collision_mesh(replace(gamepad, geometry={}, collision='sdf')) == 'sdf'


@pytest.mark.parametrize('task', ['empty', 'gamepad'])
def test_sapien_bundle_and_independent_fk(tmp_path, task):
    scene = scene_fixture(tmp_path/'input', task)
    resources = tmp_path/'resources'
    resources.mkdir()
    (resources/'asset.txt').write_text('fake gamepad resource for contract export')
    profile = resources/'robotwin/full_controls'
    profile.mkdir(parents=True)
    (profile/'mobility.urdf').write_text('<robot name="fake"><link name="gamepad"/></robot>')
    output = tmp_path/'bundle'
    bundle = export.export_scene(scene, output, format='sapien', control_hz=60, resource_root=resources)
    assert set(('robots','objects','limits','calibration','calibration_files','control_hz','drives','fingertip_offsets','pose_convention','inventory','files','identity')) <= bundle.keys()
    assert bundle['limits'] == json.loads(scene.limits_path.read_text())
    assert bundle['limits']['spatial']['margin_m'] is None
    assert len(bundle['objects']) == (task == 'gamepad')
    assert bundle['calibration'] == scene.calibration
    assert bundle['control_hz'] == 60
    assert all(export._hash(output/name) == digest for name,digest in bundle['files'].items())
    assert not any(p.is_symlink() for p in output.rglob('*'))
    fk = json.loads((output/'fk_check.json').read_text())
    assert len(fk['samples']) == 3
    home = fk['samples'][0]
    # 7 arm offsets .06, mount .05, measured base x=0 => palm x=.47.
    np.testing.assert_allclose(home['palm_pose_world']['left'], [.47,-1,2,1,0,0,0], atol=1e-12)
    assert fk['samples'][1]['palm_pose_world']['left'] != home['palm_pose_world']['left']
    for sample in fk['samples']:
        for side,robot in bundle['robots'].items():
            poses = export.link_poses(output/robot['urdf'], sample['q'][side], robot['base_pose'])
            np.testing.assert_allclose(poses[robot['palm_link']], sample['palm_pose_world'][side], atol=1e-12)
            for finger,name in robot['tip_links'].items():
                expected = export.tip_pose(poses[name], bundle['fingertip_offsets'][side][finger])
                np.testing.assert_allclose(expected, sample['fingertip_pose_world'][side][finger], atol=1e-12)
    assert export._limit_check(output, bundle, home)['status'] == dict(spatial='disabled', angular='disabled', joint='disabled')


def test_fk_reference_detects_q_and_quaternion_sign(tmp_path):
    scene = scene_fixture(tmp_path/'input')
    urdf = tmp_path/'composed.urdf'
    export.compose_robot_urdf(scene.spec, urdf)
    a = export.link_poses(urdf, {}, scene.spec.left_arm.base_pose)['palm']
    b = export.link_poses(urdf, {'j1': .3}, scene.spec.left_arm.base_pose)['palm']
    assert np.linalg.norm(np.array(a[:3])-b[:3]) > .05
    opposite = [*a[:3], *(-np.array(a[3:]))]
    assert export._pose_error(opposite, a) == pytest.approx((0.,0.), abs=1e-12)


def test_inventory_compares_names_and_counts_across_engine_traversal_orders():
    expected = dict(joints=['a','b'], actuators=['drive_a'], objects={'gamepad':['root','stick','button']})
    actual = dict(joints=['b','a'], actuators=['drive_a'], objects={'gamepad':['root','button','stick']})
    assert export._inventory_matches(actual, expected)
    assert not export._inventory_matches(dict(actual, joints=['a','a','b']), expected)
    assert not export._inventory_matches(dict(actual, objects={'gamepad':['root','button','wrong']}), expected)


def test_assets_are_portable_regular_copies(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'part.obj').write_text('mtllib part.mtl\nv 0 0 0\n')
    (source/'part.mtl').write_text('map_Kd color.png\n')
    (source/'color.png').write_bytes(b'image')
    output = tmp_path/'bundle'
    output.mkdir()
    copied = export._Assets(output)
    obj = copied.copy(source/'part.obj', 'robots')
    assert (obj.parent/'part.mtl').is_file() and (obj.parent/'color.png').is_file()
    (source/'link.obj').symlink_to(source/'part.obj')
    with pytest.raises(ValueError, match='regular-file clone'):
        copied.copy(source/'link.obj', 'robots')


def test_vendor_convex_fixture_certificate_and_rejection(tmp_path):
    trimesh = pytest.importorskip('trimesh')
    pytest.importorskip('scipy')
    mesh = tmp_path/'part.stl'
    trimesh.creation.box(extents=[.1,.2,.3]).export(mesh)
    manifest = tmp_path/'collision_manifest.json'
    manifest.write_text(json.dumps(dict(method='semantic convex unions and conservative native primitives',
        parts=[dict(file=mesh.name, sha256=export._hash(mesh), max_added_distance_m=.002)])))
    fixture = FixtureSpec('vendor_part', 'mesh', (0,0,0,1,0,0,0), True, 'design', 'vendor', mesh=str(mesh))
    evidence = export.vendor_convex_fixture(fixture)
    assert evidence['check']['ok'] and evidence['check']['convex_certificate']
    assert evidence['check']['volume_ratio'] == pytest.approx(1.)
    assert evidence['check']['hull_to_original_max_distance_upper_m'] < 1e-10
    assert evidence['manifest_sha256'] == export._hash(manifest)
    trimesh.creation.annulus(r_min=.03, r_max=.1, height=.05, sections=8).export(mesh)
    manifest.write_text(json.dumps(dict(method='test only', parts=[dict(file=mesh.name,sha256=export._hash(mesh))])))
    with pytest.raises(ValueError, match='convexity acceptance gate failed'):
        export.vendor_convex_fixture(fixture)


def _l_fixture(tmp_path):
    trimesh = pytest.importorskip('trimesh')
    pytest.importorskip('manifold3d')
    a = trimesh.creation.box(extents=[.08,.04,.02])
    a.apply_translation([-.02,-.02,0])
    b = trimesh.creation.box(extents=[.04,.08,.02])
    original = trimesh.boolean.union([a,b], engine='manifold')
    path = tmp_path/'task_support.stl'
    original.export(path)
    fixture = FixtureSpec('task_support', 'mesh', (0,0,0,1,0,0,0), True, 'design', 'test', mesh=str(path))
    return fixture, original, [a,b]


def test_task_candidate_optimization_and_reuse(tmp_path, monkeypatch):
    coacd = pytest.importorskip('coacd')
    fixture, original, hulls = _l_fixture(tmp_path)
    calls = []
    def decompose(mesh, **parameters):
        calls.append(parameters)
        return [(h.vertices,h.faces) for h in hulls]
    monkeypatch.setattr(coacd, 'run_coacd', decompose)
    folder = tmp_path/'cache'/fixture.name
    record = export.task_fixture_collision(fixture, folder)
    assert [c['threshold'] for c in calls] == list(export.COACD_THRESHOLDS)
    assert all(dict(c, threshold=.00025) == export.COACD_PARAMETERS for c in calls)
    assert record['coacd_version'] == export.COACD_VERSION and record['seed'] == 0
    assert record['hull_count'] == 2 and record['selected'] == .00025
    assert len(record['candidates']) == 4 and sum(c['selected'] for c in record['candidates']) == 1
    assert record['check']['ok']
    assert record['check']['decomposition_volume_ratio'] == pytest.approx(1., abs=1e-6)
    assert record['check']['hull_sum_volume_ratio'] > 1.05
    assert record['check']['outward']['upper_m'] <= .002
    assert record['check']['coverage_gap']['upper_m'] <= .002
    cached = export.task_fixture_collision(fixture, tmp_path/'cached', cache_root=tmp_path/'cache')
    assert len(calls) == 4 and cached['hull_count'] == 2
    assert all(c['check']['ok'] for c in cached['candidates'])


def test_solid_metrics_reject_filled_L_corner(tmp_path):
    fixture, original, _ = _l_fixture(tmp_path)
    hull = original.convex_hull
    values,_ = export._MeshSolid(original).query(np.array([[-.04,.02,0]]))
    assert values[0] == pytest.approx(.02, abs=1e-8)
    check = export.solid_collision_check(original,[hull])
    assert not check['ok']
    assert check['outward']['lower_m'] > .002


def test_solid_metrics_reject_missing_L_arm(tmp_path):
    fixture, original, hulls = _l_fixture(tmp_path)
    check = export.solid_collision_check(original,[hulls[0]])
    assert not check['ok']
    assert check['coverage_gap']['lower_m'] > .002


def test_solid_metrics_accept_point_one_mm_internal_gap(tmp_path):
    trimesh = pytest.importorskip('trimesh')
    fixture,original,_ = _l_fixture(tmp_path)
    left = trimesh.creation.box(extents=[.03995,.04,.02])
    left.apply_translation([(-.06-.02005)/2,-.02,0])
    right = trimesh.creation.box(extents=[.03995,.08,.02])
    right.apply_translation([(-.01995+.02)/2,0,0])
    check = export.solid_collision_check(original,[left,right])
    assert check['ok'], check
    assert check['outward']['upper_m'] <= .002
    assert check['coverage_gap']['upper_m'] <= .002


def test_task_decomposition_rejects_all_infeasible_candidates(tmp_path, monkeypatch):
    coacd = pytest.importorskip('coacd')
    fixture, original, _ = _l_fixture(tmp_path)
    bad = original.convex_hull
    monkeypatch.setattr(coacd, 'run_coacd', lambda *args, **kw: [(bad.vertices,bad.faces)])
    with pytest.raises(ValueError, match='task decomposition acceptance gate failed'):
        export.task_fixture_collision(fixture, tmp_path/'collision')
    evidence = json.loads((tmp_path/'collision/collision_check.json').read_text())
    assert len(evidence['candidates']) == 4 and evidence['selected'] is None
    assert all(not c['check']['ok'] for c in evidence['candidates'])
    assert all(c['check']['outward']['lower_m'] > .002 for c in evidence['candidates'])


def test_real_coacd_on_known_nonconvex_solid(tmp_path):
    pytest.importorskip('coacd')
    fixture, original, _ = _l_fixture(tmp_path)
    record = export.task_fixture_collision(fixture, tmp_path/'collision')
    assert record['method'] == 'coacd' and record['check']['ok']
    assert record['check']['outward']['upper_m'] <= .002
    assert record['check']['coverage_gap']['upper_m'] <= .002
    assert record['check']['decomposition_volume_ratio'] <= 1.05
    feasible = [c['hull_count'] for c in record['candidates'] if c['check']['ok']]
    assert record['hull_count'] == min(feasible)


def test_mjcf_preserves_separate_task_collision_hulls(tmp_path, monkeypatch):
    coacd = pytest.importorskip('coacd')
    mj = pytest.importorskip('mujoco')
    fixture, original, hulls = _l_fixture(tmp_path)
    monkeypatch.setattr(coacd, 'run_coacd', lambda *args, **kw: [(h.vertices,h.faces) for h in hulls])
    scene = scene_fixture(tmp_path/'input')
    scene = replace(scene, spec=replace(scene.spec, fixtures=(fixture,)))
    bundle = export.export_scene(scene, tmp_path/'bundle', format='mjcf', control_hz=60)
    model = mj.MjModel.from_xml_path(str(tmp_path/'bundle/scene.mjcf.xml'))
    geoms = [model.geom(name) for name in ('task_support_convex_000','task_support_convex_001')]
    assert geoms[0].dataid[0] != geoms[1].dataid[0]
    assert all(g.contype[0] == 8 and g.conaffinity[0] == 6 for g in geoms)
    assert bundle['task_fixture_collision'][0]['hull_count'] == 2
    assert all(g.name in bundle['inventory']['geoms'] for g in geoms)
    result = export.validate_export(tmp_path/'bundle',position_atol_m=1e-8,
        rotation_atol_rad=1e-8,work_dir=tmp_path/'validate')
    assert result['ok'],result
    assert result['collision_export']['hulls']['task_support'] == dict(expected=2,loaded=2)
    assert result['collision_export']['minimum_count_ok']


def test_sapien_task_support_retains_original_triangles(tmp_path, monkeypatch):
    scene = scene_fixture(tmp_path/'input')
    source = tmp_path/'support.obj'
    source.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    fixture = FixtureSpec('task_support', 'mesh', (0,0,0,1,0,0,0), True, 'design', 'test', mesh=str(source))
    scene = replace(scene, spec=replace(scene.spec, fixtures=(fixture,)))
    def no_decomposition(*args, **kwargs):
        raise AssertionError('SAPIEN must not decompose task collision meshes')
    monkeypatch.setattr(export, 'task_fixture_collision', no_decomposition)
    bundle = export.export_scene(scene, tmp_path/'bundle', format='sapien', control_hz=60)
    desc = json.loads((tmp_path/'bundle/scene.json').read_text())
    assert export._hash(tmp_path/'bundle'/desc['fixtures'][0]['mesh']) == export._hash(source)
    assert bundle['task_fixture_collision'] == []


def test_empty_mjcf_native_construction_and_validation(tmp_path):
    pytest.importorskip('mujoco')
    scene = scene_fixture(tmp_path/'input')
    # A nontrivial measured mounting quaternion exposes six-digit MJCF writer
    # loss that identity mount fixtures cannot detect.
    quat = (.7061697403471531, -.7061697433171531, .03639087280100789, .03639087548000789)
    scene = replace(scene, spec=replace(scene.spec,
        left_arm=replace(scene.spec.left_arm, base_pose=(*scene.spec.left_arm.base_pose[:3], *quat))))
    output = tmp_path/'bundle'
    bundle = export.export_scene(scene, output, format='mjcf', control_hz=60)
    assert len(bundle['inventory']['joints']) == 56
    assert len(bundle['inventory']['actuators']) == 46
    result = export.validate_export(output, position_atol_m=1e-8, rotation_atol_rad=1e-8, work_dir=tmp_path/'validate')
    assert result['ok'], result
    assert result['timestep_s'] == .001


def test_task_objects_native_mjcf_collision_flags_and_pose(tmp_path):
    mj = pytest.importorskip('mujoco')
    scene = scene_fixture(tmp_path/'input')
    objects = (object_spec('a', (False,False,False)), replace(object_spec('b',(True,True,True)), initial_pose_world=(.2,0,1,1,0,0,0)))
    scene = replace(scene, objects=objects)
    output = tmp_path/'bundle'
    bundle = export.export_scene(scene, output, format='mjcf', control_hz=60)
    model = mj.MjModel.from_xml_path(str(output/'scene.mjcf.xml'))
    a,b = model.geom('a_collision').id, model.geom('b_collision').id
    assert not ((model.geom_contype[a] & model.geom_conaffinity[b]) | (model.geom_contype[b] & model.geom_conaffinity[a]))
    result = export.validate_export(output, position_atol_m=1e-8, rotation_atol_rad=1e-8, work_dir=tmp_path/'validate')
    assert result['ok'], result
