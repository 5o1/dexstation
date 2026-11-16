import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dexstation.control.workspace_limits as wl
from dexstation.contracts.limits import LimitReport, SpatialViolation
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES
from dexstation.model.spec import ArmSpec, HandSpec, TableSpec

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = json.loads((ROOT/'src/dexstation/examples/ar5_g20/data/limits.json').read_text())


@pytest.fixture
def toy(tmp_path, monkeypatch):
    # Seven real revolute joints and five real hand chains, no mesh dependency.
    arm = '<robot><link name="base"/>'
    for i in range(1, 8):
        xyz = '0 0 0' if i < 3 else '.1 0 0'
        axis = '0 0 1' if i == 1 else '0 1 0'
        arm += f'<link name="link{i}"/><joint name="j{i}" type="revolute"><parent link="{"base" if i==1 else "link"+str(i-1)}"/><child link="link{i}"/><origin xyz="{xyz}"/><axis xyz="{axis}"/><limit lower="-3" upper="3"/></joint>'
    arm += '</robot>'
    arm_path = tmp_path/'arm.urdf'; arm_path.write_text(arm)
    hand = '<robot><link name="hand_base_link"/>'
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        names = [n for n in G20_JOINT_NAMES if n.startswith(finger+'_')]
        parent = 'hand_base_link'
        for i, name in enumerate(names):
            child = finger+'_distal' if i==len(names)-1 else name+'_link'
            hand += f'<link name="{child}"/><joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/><origin xyz=".03 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>'
            parent = child
    hand += '</robot>'
    hand_path = tmp_path/'hand.urdf'; hand_path.write_text(hand)
    def points(path):
        if path == str(arm_path):
            return {'link'+str(i): np.array([[0., 0., 0.]]) for i in range(1, 8)}
        return {'hand_base_link': np.array([[0., 0., 0.]]), 'index_distal': np.array([[.04, 0., 0.]])}
    monkeypatch.setattr(wl, '_link_points', points)
    monkeypatch.setattr(wl, '_fixtures', lambda *args, **kw: [])
    arm_spec = ArmSpec(str(arm_path), 'base', 'link7', (0, 0, 1, 1, 0, 0, 0), (0.,)*7, tuple('j'+str(i) for i in range(1, 8)))
    hand_spec = HandSpec(str(hand_path), 'hand_base_link', (0, 0, 0, 1, 0, 0, 0), 'design')
    spec = SimpleNamespace(left_arm=arm_spec, right_arm=None, left_hand=hand_spec,
        table=TableSpec((.5, 0, .5, 1, 0, 0, 0), (.4, .6, .02)), fixtures=())
    scene = SimpleNamespace(spec=spec, workbench={'world_frame':'world', 'frames':{}}, task={'keep_out':[]}, objects=())
    return wl.WorkspaceLimits(PARAMETERS, scene, cache_dir=tmp_path/'field-cache')


def test_whole_table_azimuth_allows_both_sides_of_centre(toy):
    lo, hi = toy.angular_range['left']
    assert lo < -75 and hi > 75
    parts, frames = toy.robots['left'].world(np.radians([40, 0, 0, 0, 0, 0, 0]), np.zeros(21))
    assert not toy._angular('left', frames)
    _, frames = toy.robots['left'].world(np.radians([-40, 0, 0, 0, 0, 0, 0]), np.zeros(21))
    assert not toy._angular('left', frames)


def test_actual_finger_state_changes_collision_and_command_is_held(toy):
    arms = {'left': [0.]*7}
    old = {'left': [0.]*21}
    new = copy.deepcopy(old); new['left'][G20_JOINT_NAMES.index('index_mcp_roll')] = np.pi/2
    parts, _ = toy.robots['left'].world(arms['left'], new['left'])
    point = parts['index_distal'][1][0]
    toy.obstacles = [wl.Obstacle.box('finger_block', [*point, 1, 0, 0, 0], [.003]*3)]
    assert toy.check(arms, old).ok
    assert not toy.check(arms, new).ok
    result = wl.decide_release(toy, arms, old, arms, new)
    assert result.arm_release['left'] and not result.hand_release['left']
    assert result.hand_q == old and result.reasons['left'] == ['hand_limit']


def test_arm_limit_holds_both_and_does_not_mutate_inputs(toy):
    arms = {'left': [0.]*7}; hands = {'left': [0.]*21}
    candidate = {'left': np.radians([120, 0, 0, 0, 0, 0, 0]).tolist()}
    saved = copy.deepcopy(candidate)
    result = wl.decide_release(toy, arms, hands, candidate, hands)
    assert not result.arm_release['left'] and not result.hand_release['left']
    assert result.arm_q == arms and candidate == saved


def test_elevation_margin_and_duplicate_wrist_points(toy):
    robot = toy.robots['left']
    _, frames = robot.world([0.]*7, [0.]*21)
    for frame in frames.values():
        frame[:3, 3] = robot.shoulder + [.2, 0, .2]
    violations = toy._angular('left', frames)
    assert {v.joint for v in violations if v.reason == 'elevation_up'} == set(PARAMETERS['angular']['points'])
    toy.parameters['angular']['elevation']['margin_deg'] = 0
    assert not toy._angular('left', frames)


def test_keep_out_is_additive_and_uses_existing_frame_resolution(toy):
    keep = dict(name='zone', shape='box', frame='world', pose=[.5, 0, 1, 1, 0, 0, 0], half_extents_m=[.01]*3)
    updated = toy.with_task_keep_out([keep])
    assert not toy.keep_out and len(updated.keep_out) == 1
    assert len(updated.with_task_keep_out([dict(keep, name='second')]).keep_out) == 2
    assert not updated.check({'left':[0.]*7}, {'left':[0.]*21}).ok


def test_null_enabled_layer_is_rejected(toy):
    parameters = copy.deepcopy(PARAMETERS); parameters['angular']['exempt']['r_min_m'] = None
    with pytest.raises(ValueError, match='null'):
        wl.WorkspaceLimits(parameters, toy.scene)


@pytest.mark.parametrize('left,right,expected', [(1,0,('left',)), (0,1,('right',)), (1,1,('left','right'))])
def test_dual_attribution_single_side_and_combination(left, right, expected):
    # Deliberately chosen spatial oracle isolates attribution from geometry.
    class Limits:
        robots = {s:SimpleNamespace(arm_points={'arm'},hand_points={'finger'}) for s in ('left','right')}
        def check(self, arms, hands):
            l, r = arms['left'][0], arms['right'][0]
            bad = (l > 0 if left and not right else r > 0 if right and not left else l > 0 and r > 0)
            return LimitReport(not bad, [SpatialViolation('left','arm','right:arm',0)] if bad else [], [], {}, {})
    previous = {s:[0.]*7 for s in ('left','right')}; hands={s:[0.]*21 for s in previous}
    candidate={s:[.1]+[0.]*6 for s in previous}
    result=wl.decide_release(Limits(),previous,hands,candidate,hands)
    assert tuple(s for s in previous if not result.arm_release[s]) == expected


def test_initial_empty_and_geometry_failure_are_explicit(toy, monkeypatch):
    assert wl.validate_initial_state(toy.scene) == {'checked':True,'ok':True,'violations':[]}
    toy.scene.objects=(object(),)
    toy.scene.limits_path=Path('/nonexistent/wp5-limits')
    result=wl.validate_initial_state(toy.scene)
    assert result['checked'] and not result['ok'] and result['violations'][0]['reason']=='geometry_validation_failed'


def test_box_true_euclidean_faces_edges_corners_and_interior():
    box = wl.Obstacle.box('box', [0,0,0,1,0,0,0], [.1]*3)
    points = np.array([[.119,.119,0], [.11,.12,.13], [.12,0,0],
                       [0,0,0], [.09,.08,.07], [-.12,-.13,.04], [.1,.1,.1]])
    expected = [np.hypot(.019,.019), np.linalg.norm([.01,.02,.03]), .02,
                -.1, -.01, np.hypot(.02,.03), 0.]
    np.testing.assert_allclose(box.signed_distances(points), expected, atol=1e-14, rtol=0)


def test_rotated_box_true_distance_is_rigid_invariant():
    from dexstation.model.spec import pose_to_matrix
    pose = [.2,-.3,.7, np.cos(.3),0,0,np.sin(.3)]
    points = np.array([[.119,.119,0], [.12,0,0], [0,0,0]])
    box = wl.Obstacle.box('rotated', pose, [.1]*3)
    world = wl._transform(points, pose_to_matrix(pose))
    np.testing.assert_allclose(box.signed_distances(world), [np.sqrt(2)*.019,.02,-.1], atol=1e-14)


def test_closed_mesh_matches_box_euclidean_distance():
    trimesh = pytest.importorskip('trimesh')
    model = trimesh.creation.box(extents=[.2]*3)
    solid = wl.Obstacle.convex('mesh', model.triangles)
    points = np.array([[.119,.119,0], [.11,.12,.13], [.12,0,0], [0,0,0], [.09,.08,.07]])
    box = wl.Obstacle.box('box', [0,0,0,1,0,0,0], [.1]*3)
    np.testing.assert_allclose(solid.signed_distances(points), box.signed_distances(points), atol=1e-12, rtol=0)


def reference_check(self, arm_q, hand_q):
    spatial, angular, joint, bodies = [], [], {}, {}
    for side, robot in self.robots.items():
        q = wl._vector(arm_q[side], 7)
        parts, frames = robot.world(q, hand_q[side])
        bodies[side] = parts
        bounds = self.parameters['joint']['controller_soft_limits'][side]
        joint[side] = ([True]*7 if self._has_null(bounds) else
                       ((q >= np.radians(bounds['lower_deg'])) & (q <= np.radians(bounds['upper_deg']))).tolist())
        angular.extend(self._angular(side, frames))
        if not self.spatial_enabled:
            continue
        for link, (_, points) in parts.items():
            low, high = points.min(0), points.max(0)
            for obstacle in self.obstacles+self.keep_out:
                if link == robot.chain.joints[0]['child'] and obstacle.name.split('#')[0] == 'station_column_mount_'+side:
                    continue
                if np.any(low > obstacle.hi+self.margin) or np.any(high < obstacle.lo-self.margin):
                    continue
                distance = float(obstacle.distance(points))
                if distance < self.margin:
                    spatial.append(SpatialViolation(side, link, obstacle.name, distance))
    if self.spatial_enabled and len(bodies) == 2:
        # Still transform/check every point. Cache only repeated AABB reductions
        # in this oracle's pair loop; no new hierarchy or distance-field reuse.
        bounds = {side:{link:(p.min(0),p.max(0)) for link,(_,p) in parts.items()}
                  for side,parts in bodies.items()}
        for left, (_, a) in bodies['left'].items():
            for right, (_, b) in bodies['right'].items():
                if np.any(bounds['left'][left][0] > bounds['right'][right][1]+self.margin) or np.any(bounds['right'][right][0] > bounds['left'][left][1]+self.margin):
                    continue
                distance = wl._pair_distance(a, b, self.margin)
                if distance < self.margin:
                    spatial.append(SpatialViolation('left', left, 'right:'+right, distance))
    return LimitReport(not spatial and not angular and all(all(v) for v in joint.values()), spatial, angular, joint, dict(self.angular_range))


def test_trilinear_error_bound_prevents_a_false_negative(tmp_path):
    obstacle = wl.Obstacle.box('corner', [-.007]*3+[1,0,0,0], [.001]*3)
    field = wl.DistanceField([obstacle], [0,0,0], [.01]*3, tmp_path, spacing=.01)
    point = np.array([[.005]*3])
    exact = obstacle.signed_distances(point)[0]
    interpolated = field.query(point)[0]
    assert exact < .02 < interpolated  # e=0 would incorrectly discard this point.
    assert interpolated-field.error <= .02
    rng = np.random.default_rng(20261115)
    points = rng.uniform(0, .01, (2000,3))
    error = np.abs(field.query(points)-obstacle.signed_distances(points))
    assert error.max() <= field.error


def test_cluster_radius_is_required_for_conservative_screening(tmp_path):
    obstacle = wl.Obstacle.box('face', [0,0,0,1,0,0,0], [.083,1,1])
    clusters = wl.PointClusters.build(np.array([[.102,0,0],[.124,0,0]]))
    assert len(clusters.centres) == 1
    field = wl.DistanceField([obstacle], [.09,-.01,-.01], [.14,.01,.01], tmp_path)
    d = field.query(clusters.centres)[0]
    assert obstacle.distance(clusters.points) < .02
    assert d-field.error > .02  # r=0 misses a real violation.
    assert d-field.error-clusters.radii[0] <= .02


def _assert_reports_equal(a, b):
    assert a.ok == b.ok and a.joint == b.joint and a.angular == b.angular
    assert a.angular_range == b.angular_range
    left = {(v.side,v.link,v.obstacle):v.distance_m for v in a.spatial}
    right = {(v.side,v.link,v.obstacle):v.distance_m for v in b.spatial}
    assert left.keys() == right.keys()
    np.testing.assert_allclose(list(left.values()),list(right.values()),atol=1e-9,rtol=0)




def test_fast_fk_matches_repository_serial_chain(toy):
    robot=toy.robots['left']; rng=np.random.default_rng(20261115)
    for _ in range(30):
        arm,hand=rng.uniform(-1,1,7),rng.uniform(-1,1,21)
        frames,hand_frames=robot.frames(arm,hand)
        old=robot.chain.link_poses(arm)
        for name,pose in old.items():
            np.testing.assert_allclose(frames[name],robot.base@pose[0],atol=1e-12,rtol=0)
        positions=dict(zip(G20_JOINT_NAMES,hand))
        palm=frames[robot.arm.tip_link]@robot.mount
        for chain in robot.hand_chains:
            for name,pose in chain.link_poses([positions[n] for n in chain.active]).items():
                np.testing.assert_allclose(hand_frames[name],palm@pose[0],atol=1e-12,rtol=0)



def test_tiny_triangle_distance_does_not_snap_interior_projection_to_edge():
    triangle = np.array([[[0.,0.,0.],[.01,0.,0.],[.01,.00001,0.]]])
    point = np.array([[.005,.000002,.000001]])
    nearest = wl._triangle_closest_exact(triangle, point)
    np.testing.assert_allclose(nearest, [[.005,.000002,0.]], atol=1e-15, rtol=0)
    assert np.linalg.norm(nearest-point) == pytest.approx(.000001, abs=1e-15)


def test_temporal_displacement_is_required_to_prevent_a_miss(tmp_path):
    obstacle=wl.Obstacle.box('box',[0,0,0,1,0,0,0],[.1,1,1])
    field=wl.DistanceField([obstacle],[.1,-.01,-.01],[.17,.01,.01],tmp_path)
    cache=wl.TemporalBounds(anchor_every=30,jump_m=.05)
    radii=np.zeros(1)
    previous=np.array([[.15,0,0]])
    current=np.array([[.119,0,0]])
    possible,queries=cache.screen(previous,radii,field,.02)
    assert not possible[0] and queries==1
    assert obstacle.distance(current)<.02
    possible,queries=cache.screen(current,radii,field,.02)
    assert possible[0] and queries==1  # Delta=0 would reuse an unsafe old bound.


def test_temporal_anchor_jump_and_restore_clear_history(toy):
    arms={'left':[0.]*7};hands={'left':[0.]*21}
    _,first=toy.check_profiled(arms,hands)
    assert first['temporal_anchor']==1
    _,second=toy.check_profiled(arms,hands)
    assert second['temporal_anchor']==0 and second['field_queried_centres']==0
    for _ in range(28):toy.check(arms,hands)
    _,anchor=toy.check_profiled(arms,hands)
    assert anchor['temporal_anchor']==1
    jump={'left':np.radians([100,0,0,0,0,0,0]).tolist()}
    _,stats=toy.check_profiled(jump,hands)
    assert stats['temporal_anchor']==1
    state=toy.snapshot();toy.restore(state)
    assert toy._temporal.centres is None
    _,restored=toy.check_profiled(arms,hands)
    assert restored['temporal_anchor']==1


def test_cached_uncached_and_reference_agree_over_jumps_and_boundaries(toy):
    toy.obstacles = [wl.Obstacle.box('near', [.4,.1,1,1,0,0,0], [.03]*3)]
    rng = np.random.default_rng(20261115)
    for _ in range(60):
        arm={'left':rng.uniform(-.5,.5,7)}
        hand={'left':rng.uniform(-.5,.5,21)}
        _assert_reports_equal(toy.check(arm,hand),reference_check(toy,arm,hand))
    for clearance in (.019,.02,.021):
        toy.obstacles=[wl.Obstacle.box('margin', [.5+clearance+.01,0,1,1,0,0,0],[.01]*3)]
        arm={'left':[0.]*7}; hand={'left':[0.]*21}
        _assert_reports_equal(toy.check(arm,hand),reference_check(toy,arm,hand))
    toy.obstacles=[wl.Obstacle.box('block',[.5,.05,1,1,0,0,0],[.01]*3)]
    rng=np.random.default_rng(20261115)
    for i in range(100):
        arm={'left':rng.uniform(-1,1,7) if i%11==0 else np.zeros(7)}
        hand={'left':rng.uniform(-.5,.5,21)}
        actual=toy.check(arm,hand)
        _assert_reports_equal(actual,toy.check_uncached(arm,hand))
        _assert_reports_equal(actual,reference_check(toy,arm,hand))
    toy.restore({})
    for x in (.019,.02,.021):
        toy.obstacles=[wl.Obstacle.box('margin',[.51+x,0,1,1,0,0,0],[.01]*3)]
        _assert_reports_equal(toy.check({'left':[0.]*7},{'left':[0.]*21}),toy.check_uncached({'left':[0.]*7},{'left':[0.]*21}))


@pytest.fixture
def initial_scene(tmp_path, monkeypatch):
    pytest.importorskip('trimesh')
    pytest.importorskip('manifold3d')
    from dexstation.contracts.profiles import ObjectSpec
    def make(position=(0,0,.075), half=(.025,.025,.025), randomize=None, collision=None):
        collision=collision or dict(robots=True,environment=True,objects=True)
        source=dict(name='cube',type='rigid',geometry={'box':{'half_extents':list(half)}},
            initial_pose={'frame':'world','pose':[*position,1,0,0,0]},collision=collision)
        if randomize:source['randomize']=randomize
        obj=ObjectSpec('cube','rigid',source['geometry'],{}, {},tuple(source['initial_pose']['pose']),randomize,collision,{})
        return SimpleNamespace(spec=SimpleNamespace(table=TableSpec((0,0,0,1,0,0,0),(1,1,.05))),
            workbench={'world_frame':'world','frames':{'table_top':{'parent':'world','pose':[0,0,.05,1,0,0,0]}}},
            task={'objects':[source],'keep_out':[]},objects=(obj,),
            limits_path=ROOT/'src/dexstation/examples/ar5_g20/data/limits.json')
    fixed=[('table',wl._InitialShape(wl._box_triangles([0,0,0,1,0,0,0],[1,1,.05])))]
    robot=[('left:arm',wl._InitialShape(wl._box_triangles([.5,0,.25,1,0,0,0],[.1]*3)))]
    monkeypatch.setattr(wl,'_initial_environment',lambda scene:fixed)
    monkeypatch.setattr(wl,'_initial_robots',lambda scene:robot)
    return make,fixed,robot


def test_initial_whole_object_contact_and_below_table(initial_scene):
    make,_,_=initial_scene
    assert wl.validate_initial_state(make())['ok']
    invalid=wl.validate_initial_state(make(position=(0,0,.06)))
    assert not invalid['ok']
    assert any(v['reason']=='below_table_top' for v in invalid['violations'])


def test_initial_containment_and_edge_crossing_are_detected(initial_scene):
    make,fixed,_=initial_scene
    fixed.append(('contained_post',wl._InitialShape(wl._box_triangles([0,0,.2,1,0,0,0],[.01]*3))))
    result=wl.validate_initial_state(make(position=(0,0,.2),half=(.05,.05,.05)))
    assert any(v.get('obstacle')=='contained_post' for v in result['violations'])
    fixed[-1]=('crossbar',wl._InitialShape(wl._box_triangles([0,0,.2,1,0,0,0],[.01,.2,.01])))
    result=wl.validate_initial_state(make(position=(0,0,.2),half=(.2,.01,.01)))
    assert any(v.get('obstacle')=='crossbar' for v in result['violations'])


def test_initial_randomization_corners_use_full_geometry(initial_scene):
    make,fixed,_=initial_scene
    fixed.append(('corner_post',wl._InitialShape(wl._box_triangles([.6,0,.075,1,0,0,0],[.03]*3))))
    scene=make(randomize={'xy_m':[[0,.6],[0,0]],'yaw_deg':[0,45]})
    result=wl.validate_initial_state(scene)
    assert any(v.get('obstacle')=='corner_post' and abs(v['pose'][0]-.6)<1e-12 for v in result['violations'])


def test_initial_robot_overlap_and_existing_collision_switches(initial_scene):
    make,_,_=initial_scene
    invalid=wl.validate_initial_state(make(position=(.5,0,.25)))
    assert any(v['reason']=='initial_robot_overlap' for v in invalid['violations'])
    scene=make(position=(.5,0,.25),collision=dict(robots=False,environment=True,objects=True))
    assert wl.validate_initial_state(scene)['ok']


def test_initial_keepout_margin_frame_and_contained_zone(initial_scene):
    make,_,_=initial_scene
    scene=make()
    zone=dict(name='zone',shape='box',frame='table_top',pose=[.05,0,.025,1,0,0,0],half_extents_m=[.01]*3)
    scene.task['keep_out']=[zone]
    assert any(v['reason']=='keep_out' for v in wl.validate_initial_state(scene)['violations'])
    scene.task['keep_out']=[dict(zone,pose=[.055,0,.025,1,0,0,0])]
    assert wl.validate_initial_state(scene)['ok']  # exact 20 mm gap
    scene.task['keep_out']=[dict(zone,pose=[.055-5e-9,0,.025,1,0,0,0])]
    assert not wl.validate_initial_state(scene)['ok']  # a 5 nm deficit is not a new clearance allowance
    scene=make(half=(.05,.05,.025))
    scene.task['keep_out']=[dict(zone,pose=[0,0,.025,1,0,0,0],half_extents_m=[.001]*3)]
    assert any(v['reason']=='keep_out' for v in wl.validate_initial_state(scene)['violations'])


def _replay_module():
    import runpy
    return runpy.run_path(str(ROOT/'scripts/limits_replay.py'))


def test_replay_merges_first_hand_hold_in_final_decision(toy):
    from dexstation.control.takeover import ArmTakeover
    m=_replay_module();arms={'left':[0.]*7};hands={'left':[0.]*21}
    requested={'left':[0.]*21};requested['left'][5]=np.pi/2
    parts,_=toy.robots['left'].world(arms['left'],requested['left'])
    point=parts['index_distal'][1][0]
    toy.obstacles=[wl.Obstacle.box('finger_block',[*point,1,0,0,0],[.003]*3)]
    takeover=ArmTakeover(dict(match_tolerance_deg=2.,match_dwell_s=.2,deviation_budget_deg=10.))
    replay=m['ReplayController'](toy,takeover,arms,hands)
    result=replay.step(0.,arms,requested)
    assert not result['hand_release']['left'] and result['hand_q']==hands
    assert result['output_report']['ok']
    assert any(e['type']=='limit_hold' for e in result['events'])


def test_replay_logs_second_stage_following_to_hold():
    from dexstation.control.takeover import ArmTakeover
    m=_replay_module()
    class Limits:
        robots={s:SimpleNamespace(arm_points={'arm'},hand_points={'hand'}) for s in ('left','right')}
        def check(self,arms,hands):
            bad=abs(arms['left'][0])<.1 and hands['right'][0]>.05
            return LimitReport(not bad,[SpatialViolation('left','arm','right:hand',0.)] if bad else [],[],{}, {})
    arms={s:[0.]*7 for s in ('left','right')};hands={s:[0.]*21 for s in arms}
    takeover=ArmTakeover(dict(match_tolerance_deg=2.,match_dwell_s=0.,deviation_budget_deg=10.))
    replay=m['ReplayController'](Limits(),takeover,arms,hands)
    target=copy.deepcopy(arms);target['left'][0]=1.
    hand_target=copy.deepcopy(hands);hand_target['right'][0]=.1
    result=replay.step(0.,target,hand_target)
    transitions=[(e['before'],e['after']) for e in result['events'] if e['type']=='takeover' and e['side']=='right']
    assert transitions==[('waiting','following'),('following','hold')]
    assert result['takeover']['right']['state'].value=='hold'
    assert not result['hand_release']['right'] and result['output_report']['ok']


def test_recording_decoder_preserves_actual_hands_and_explicit_mount_conversion():
    m=_replay_module()
    class Mapper:
        def command(self,*args):raise AssertionError('mapped targets must not be remapped')
    row=dict(sim_time_s=1.,arm_q=[[0.]*7,[0.]*7],hand_q=[[.1]*21,[.2]*21],commands={
        'left_arm':dict(positions=[.3]+[0.]*6,raw_master_deg=None),
        'right_arm':dict(positions=[.4]+[0.]*6,raw_master_deg=None)})
    frame=m['decode_record'](row,Mapper(),[-5.90,19.15])
    assert frame['hand_targets']=={'left':[.1]*21,'right':[.2]*21}
    assert frame['arm_targets']['left'][0]==pytest.approx(.3+np.radians(-5.90))
    assert frame['observed_arm_q']['right'][0]==pytest.approx(np.radians(19.15))
    assert frame['mapping_source']['left']=='recorded_target_rad'


def test_replay_reference_roundtrip_accepts_json_array_ranges(toy):
    from dataclasses import asdict
    m=_replay_module();report=toy.check({'left':[0.]*7},{'left':[0.]*21})
    reference=json.loads(json.dumps(asdict(report)))
    assert m['compare_reports'](report,reference)['equivalent']
    reference['ok']=not reference['ok']
    assert not m['compare_reports'](report,reference)['equivalent']


def test_initial_articulation_fixed_only_chain_keeps_link_transform(tmp_path):
    pytest.importorskip('trimesh')
    path=tmp_path/'fixed.urdf'
    path.write_text('<robot name="fixed"><link name="root"><collision><geometry><box size=".04 .04 .04"/></geometry></collision></link>'
        '<link name="child"><collision><geometry><box size=".02 .02 .02"/></geometry></collision></link>'
        '<joint name="mount" type="fixed"><parent link="root"/><child link="child"/><origin xyz=".3 0 0"/></joint></robot>')
    obj=SimpleNamespace(type='articulation',geometry={'urdf':str(path),'fixed_root':True},extra={})
    pieces=wl._object_meshes(obj)
    assert len(pieces)==2
    assert pieces[1][...,0].mean()==pytest.approx(.3)


def test_initial_gamepad_reads_selected_sdf_not_urdf_coacd(tmp_path,monkeypatch):
    trimesh=pytest.importorskip('trimesh')
    resources=pytest.importorskip('xbox_gamepad_physics.resources')
    coarse=trimesh.creation.box(extents=[.02]*3);coarse.export(tmp_path/'coacd.obj')
    sdf=trimesh.creation.box(extents=[.02]*3);sdf.apply_translation([.2,0,0]);sdf.export(tmp_path/'sdf.obj')
    (tmp_path/'mobility.urdf').write_text('<robot name="pad"><link name="body"><collision><geometry><mesh filename="coacd.obj"/></geometry></collision></link></robot>')
    (tmp_path/'manifest.json').write_text(json.dumps({'entrypoint':'mobility.urdf','collision_variants':{'sdf':{'links':{'body':['sdf.obj']}}}}))
    monkeypatch.setattr(resources,'query_resource_path',lambda backend,profile,*parts:tmp_path/(parts[0] if parts else 'manifest.json'))
    obj=SimpleNamespace(type='xbox_gamepad_physics',geometry={'profile':'full_controls','collision_mesh':'sdf'},extra={})
    assert wl._object_meshes(obj)[0][...,0].mean()==pytest.approx(.2)
    obj.geometry['collision_mesh']='coacd'
    assert wl._object_meshes(obj)[0][...,0].mean()==pytest.approx(0.)


def test_recording_raw_master_uses_mapper_without_double_mount_shift():
    from dexstation.inputs.arm_mapping import ArmMapper
    m=_replay_module();home={s:[0.]*7 for s in ('left','right')}
    offsets={'left':[-5.90]+[0.]*6,'right':[19.15]+[0.]*6}
    mapper=ArmMapper(home,startup='none',joint_offset_deg=offsets)
    row=dict(sim_time_s=1.,arm_q=[[0.]*7,[0.]*7],hand_q=[[0.]*21,[0.]*21],commands={
        s+'_arm':dict(raw_master_deg=[0.]*7,positions=[999.]*7) for s in ('left','right')})
    frame=m['decode_record'](row,mapper,[-5.90,19.15])
    assert frame['arm_targets']['left'][0]==pytest.approx(np.radians(84.10))
    assert frame['arm_targets']['right'][0]==pytest.approx(np.radians(-70.85))
    assert frame['mapping_source']['left']=='raw_master_deg_via_mapper'


def test_recording_without_arm_command_passes_none_to_takeover():
    m=_replay_module()
    row=dict(sim_time_s=1.,arm_q=[[0.]*7,[0.]*7],hand_q=[[0.]*21,[0.]*21],commands={})
    frame=m['decode_record'](row,None,[0,0])
    assert frame['arm_targets']=={'left':None,'right':None}


def test_observed_mimic_child_is_not_replaced_by_command_coupling(toy,tmp_path):
    import xml.etree.ElementTree as ET
    from dexstation.model.kinematics import SerialChain
    robot=toy.robots['left'];path=Path(robot.hand.urdf)
    tree=ET.parse(path);root=tree.getroot()
    child=root.find("joint[@name='index_dip']")
    ET.SubElement(child,'mimic',joint='index_pip',multiplier='.89',offset='0')
    tree.write(path)
    updated=wl.RobotGeometry(toy.scene.spec,'left')
    hand=np.zeros(21);hand[G20_JOINT_NAMES.index('index_dip')]=.4
    _,frames=updated.frames(np.zeros(7),hand)
    # Independent measured-state FK: the same URDF with only the command-level
    # mimic declaration removed, so all observed joint angles are explicit.
    child.remove(child.find('mimic'));explicit=tmp_path/'observed.urdf';tree.write(explicit)
    chain=SerialChain(explicit,'index_distal')
    values=dict(zip(G20_JOINT_NAMES,hand));local=chain.forward([values[n] for n in chain.active])[0]
    arm_frames,_=updated.frames(np.zeros(7),np.zeros(21));palm=arm_frames[updated.arm.tip_link]@updated.mount
    np.testing.assert_allclose(frames['index_distal'],palm@local,atol=1e-12,rtol=0)
    assert not np.allclose(frames['index_distal'],updated.frames(np.zeros(7),np.zeros(21))[1]['index_distal'])


def test_initial_articulation_uses_explicit_child_positions(tmp_path):
    pytest.importorskip('trimesh')
    path=tmp_path/'mimic.urdf'
    path.write_text('<robot name="explicit"><link name="root"/><link name="parent"/>'
        '<link name="child"><collision><geometry><box size=".02 .02 .02"/></geometry></collision></link>'
        '<joint name="source" type="prismatic"><parent link="root"/><child link="parent"/><axis xyz="1 0 0"/><limit lower="0" upper="1"/></joint>'
        '<joint name="coupled" type="prismatic"><parent link="parent"/><child link="child"/><axis xyz="1 0 0"/><limit lower="0" upper="1"/><mimic joint="source" multiplier=".5"/></joint></robot>')
    obj=SimpleNamespace(type='articulation',geometry={'urdf':str(path),'fixed_root':True},extra={'initial_joint_positions':{'source':.1,'coupled':.3}})
    assert wl._object_meshes(obj)[0][...,0].mean()==pytest.approx(.4)
