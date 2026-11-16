from helpers import bench_data, bench_spec
"""Task body loading and state roundtrips without a renderer or hardware."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from dexstation.contracts.profiles import RuntimeScene, ObjectSpec, CameraPair, EMPTY_TASK
from dexstation.profiles.task_objects import TaskObject, load_task_object, render_payload, render_runtime, render_spec, collision_mesh
from dexstation.model.spec import load_spec

ROOT = Path(__file__).parents[1]
POSE = (0., 0., 1., 1., 0., 0., 0.)


def runtime(task='empty'):
    spec = bench_spec()
    obj = ObjectSpec('cube','rigid', {'box':{'half_extents':[.025]*3}}, {'mass_kg':.1},
        {'static_friction':.8,'dynamic_friction':.6,'restitution':0.}, POSE, None,
        dict(robots=True,environment=True,objects=True), {})
    if task == 'gamepad':
        pad = spec.gamepad
        obj = replace(obj, name='gamepad', type='xbox_gamepad_physics', geometry=dict(
            release=pad.release, profile=pad.profile, collision_mesh=pad.collision, fixed_root=pad.fixed_root),
            initial_pose_world=pad.root_pose)
    cameras = tuple(CameraPair(c.name, dict(model=c.model), dict(parent=c.parent,
        pose_optical=c.pose_optical, intrinsics={k:getattr(c.intrinsics,k) for k in c.intrinsics.__dataclass_fields__},
        source=c.source,status=c.status)) for c in spec.cameras)
    return RuntimeScene(spec, {}, dict(EMPTY_TASK, id=task), () if task == 'empty' else (obj,),
        (), cameras, (), {}, Path('.'), Path('.'), {}, {}, {})


@pytest.mark.parametrize('task',['empty','cube','gamepad'])
def test_render_payload_roundtrip_keeps_task_objects_and_camera_pairs(task):
    source = runtime(task)
    result = render_runtime(None, json.loads(json.dumps(render_payload(source))))
    definition = render_spec(result)
    assert (definition.gamepad is None) == (task != 'gamepad')
    assert [c.name for c in definition.cameras] == [c.name for c in source.cameras if c.sim]
    assert [o.name for o in result.objects] == [o.name for o in source.objects]
    assert definition.left_arm == source.spec.left_arm or np.allclose(definition.left_arm.home_q, source.spec.left_arm.home_q)


def test_uncalibrated_pair_does_not_create_a_sim_camera():
    scene = runtime()
    scene = replace(scene, cameras=(*scene.cameras, CameraPair('uncalibrated',{},None)))
    assert 'uncalibrated' not in [c.name for c in render_spec(scene).cameras]


def test_geometry_collision_mesh_wins_over_legacy_string():
    obj = runtime('gamepad').objects[0]
    assert collision_mesh(replace(obj,geometry={'collision_mesh':'coacd'},collision='sdf'))=='coacd'
    assert collision_mesh(replace(obj,geometry={},collision='sdf'))=='sdf'


class Pose:
    def __init__(self, p=(0,0,0), q=(1,0,0,0)):
        self.p,self.q=np.asarray(p,float),np.asarray(q,float)


class Body:
    def __init__(self, articulation=False):
        self.pose=Pose();self.v=np.zeros(3);self.w=np.zeros(3)
        self.q=np.zeros(1);self.dq=np.zeros(1)
    def get_pose(self):return self.pose
    get_root_pose=get_pose
    def set_pose(self, pose):self.pose=pose
    set_root_pose=set_pose
    def find_component_by_type(self, cls):return self
    def get_linear_velocity(self):return self.v
    def get_angular_velocity(self):return self.w
    def set_linear_velocity(self, v):self.v=v
    def set_angular_velocity(self, v):self.w=v
    set_root_linear_velocity=set_linear_velocity
    set_root_angular_velocity=set_angular_velocity


class Articulation(Body):
    def get_active_joints(self):return [NS(name='hinge')]
    def get_links(self):return [self]
    def get_qpos(self):return self.q
    def get_qvel(self):return self.dq
    def set_qpos(self,q):self.q=q
    def set_qvel(self,v):self.dq=v


@pytest.mark.parametrize('kind',['rigid','static','articulation','xbox_gamepad_physics'])
def test_load_and_restore_task_objects(monkeypatch,tmp_path,kind):
    body = Articulation() if kind in ('articulation','xbox_gamepad_physics') else Body()
    calls=[]
    monkeypatch.setitem(sys.modules,'sapien', NS(Pose=Pose,physx=NS(
        PhysxMaterial=lambda *a:a, PhysxRigidDynamicComponent=Body,PhysxRigidStaticComponent=Body)))
    builder=NS(add_box_collision=lambda **kw:calls.append(('collision',kw)),
        add_box_visual=lambda **kw:calls.append(('visual',kw)),
        set_mass_and_inertia=lambda *a:calls.append(('mass',a)),
        build=lambda **kw:body,build_static=lambda **kw:body)
    loader=NS(load=lambda path:body)
    scene=NS(create_actor_builder=lambda:builder,create_urdf_loader=lambda:loader)
    obj=replace(runtime('cube').objects[0],type=kind)
    if kind=='articulation':
        asset=tmp_path/'hinge.urdf';asset.write_text('<robot/>')
        obj=replace(obj,geometry=dict(urdf=str(asset),fixed_root=True),extra={'initial_joint_positions':{'hinge':.2}})
    if kind=='xbox_gamepad_physics':
        obj=runtime('gamepad').objects[0]
        def load(scene_arg,pose,**kw):
            calls.append(('asset',kw));body.set_pose(pose)
            return NS(articulation=body)
        monkeypatch.setitem(sys.modules,'xbox_gamepad_physics.robotwin',NS(load_into_scene=load))
    item=load_task_object(scene,obj,render=True,work_dir=tmp_path)
    np.testing.assert_allclose(item.pose(),obj.initial_pose_world)
    initial=item.snapshot()
    item.set_state([3,2,1,1,0,0,0], [1,2,3,4,5,6])
    assert item.pose()[0]==3
    item.restore(initial)
    np.testing.assert_allclose(item.pose(),obj.initial_pose_world)
    if kind=='rigid':
        args=next(value for name,value in calls if name=='mass')
        assert args[0]==.1
        np.testing.assert_allclose(args[2],np.full(3,.1/3*(2*.025**2)))
    if kind=='articulation':assert item.body.get_qpos()==pytest.approx([.2])
    if kind=='xbox_gamepad_physics':assert calls[-1][1]['attach_visual'] is True


@pytest.mark.parametrize('task',['empty','cube','gamepad'])
def test_real_sapien_backend_constructs_steps_and_restores(tmp_path,task):
    pytest.importorskip('sapien')
    pytest.importorskip('xbox_gamepad_physics')
    scene=runtime(task)
    if not Path(scene.spec.left_arm.urdf).is_file():
        pytest.skip('station assets required for real SAPIEN acceptance')
    from dexstation.contracts.backend import CollectionBackend, StepObservation
    from dexstation.recording.session import create_registered_backend as create_backend
    backend=create_backend('sapien',scene,work_dir=tmp_path/task)
    try:
        assert isinstance(backend,CollectionBackend)
        backend.step(.01)
        obs=backend.observe()
        assert isinstance(obs,StepObservation)
        assert obs.robot.sim_time_s==pytest.approx(.01)
        assert set(obs.object_poses)==set(o.name for o in scene.objects)
        saved=backend.snapshot_motion_state()
        backend.step(.01)
        backend.restore_motion_state(saved)
        assert backend.snapshot_motion_state()==saved
    finally:
        backend.close()


@pytest.mark.parametrize('policy',['sdf','coacd'])
def test_articulation_mesh_policy_keeps_cooking_in_work_dir(monkeypatch,tmp_path,policy):
    mesh=tmp_path/'source.obj';mesh.write_text('mesh test')
    asset=tmp_path/'source.urdf';asset.write_text('<robot/>')
    record=NS(type='convex_mesh',filename=str(mesh),decomposition='none')
    body=Articulation()
    builder=NS(link_builders=[NS(name='base',collision_records=[record])],build=lambda:body)
    loader=NS(parse=lambda path:([builder],[],[]))
    monkeypatch.setitem(sys.modules,'sapien',NS(Pose=Pose))
    obj=replace(runtime('cube').objects[0],type='articulation',
        geometry=dict(urdf=str(asset),fixed_root=True,collision_mesh=policy))
    item=load_task_object(NS(create_urdf_loader=lambda:loader),obj,render=True,work_dir=tmp_path/'work')
    assert record.type==('nonconvex_mesh' if policy=='sdf' else 'multiple_convex_meshes')
    assert record.decomposition==('none' if policy=='sdf' else 'coacd')
    assert Path(record.filename).is_relative_to(tmp_path/'work')
    assert Path(record.filename).read_bytes()==mesh.read_bytes()
    assert loader.fix_root_link is True


def test_rigid_mesh_mass_uses_native_volume_inertia(monkeypatch,tmp_path):
    mesh=tmp_path/'solid.obj';mesh.write_text('fake native mass properties')
    body=Body();body.mass=2.;body.inertia=np.array([.2,.4,.6])
    builder=NS(add_multiple_convex_collisions_from_file=lambda *a,**kw:None,
        build=lambda **kw:body)
    monkeypatch.setitem(sys.modules,'sapien',NS(Pose=Pose,physx=NS(
        PhysxMaterial=lambda *args:args,PhysxRigidDynamicComponent=Body,PhysxRigidStaticComponent=Body)))
    obj=replace(runtime('cube').objects[0],geometry=dict(mesh=str(mesh),collision_mesh='coacd'))
    item=load_task_object(NS(create_actor_builder=lambda:builder),obj,work_dir=tmp_path)
    assert body.mass==pytest.approx(.1)
    np.testing.assert_allclose(body.inertia,[.01,.02,.03])


@pytest.mark.parametrize('fixed_root',[False,True])
def test_real_urdf_articulation_root_and_joint_state(tmp_path,fixed_root):
    sapien=pytest.importorskip('sapien')
    source=tmp_path/'hinge.urdf'
    source.write_text('''<robot name="test_hinge">
      <link name="base"><inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial>
        <collision><geometry><box size=".05 .05 .05"/></geometry></collision></link>
      <link name="tip"><inertial><mass value=".1"/><inertia ixx=".001" iyy=".001" izz=".001" ixy="0" ixz="0" iyz="0"/></inertial>
        <collision><geometry><box size=".02 .02 .1"/></geometry></collision></link>
      <joint name="hinge" type="revolute"><parent link="base"/><child link="tip"/><origin xyz="0 0 .1"/>
        <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>''')
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);scene.set_timestep(.001)
    definition=replace(runtime('cube').objects[0],type='articulation',geometry=dict(
        urdf=str(source),fixed_root=fixed_root),extra={'initial_joint_positions':{'hinge':.2}})
    obj=load_task_object(scene,definition,work_dir=tmp_path)
    np.testing.assert_allclose(obj.body.get_qpos(),[.2],atol=1e-7)
    for _ in range(10):scene.step()
    assert np.isfinite(obj.body.get_qpos()).all()
    assert (obj.pose()[2] < POSE[2]) == (not fixed_root)
    frozen=obj.snapshot()
    obj.set_state([3,2,1,1,0,0,0],[1,2,3,4,5,6])
    obj.restore(frozen)
    np.testing.assert_allclose(obj.pose(),frozen['pose'],atol=1e-7)
    np.testing.assert_allclose(obj.velocity(),frozen['twist'],atol=1e-7)


def test_real_sapien_composed_gamepad_builds_each_static_object_once(tmp_path):
    """compose() also projects static objects into spec.fixtures; the backend must not build them twice."""
    pytest.importorskip('sapien')
    pytest.importorskip('xbox_gamepad_physics')
    from dexstation.profiles.compose import compose, load_task, load_workbench
    from dexstation.recording.session import create_registered_backend as create_backend
    scene=compose(load_workbench(ROOT/'src/dexstation/examples/ar5_g20/data'),
                  load_task('xbox_gamepad_v3_3'),'sapien')
    if not Path(scene.spec.left_arm.urdf).is_file():
        pytest.skip('station assets required for real SAPIEN acceptance')
    statics=[o.name for o in scene.objects if o.type=='static']
    assert statics
    backend=create_backend('sapien',scene,work_dir=tmp_path)
    try:
        names=[e.name for e in backend.scene.entities]
        assert {n:names.count(n) for n in statics}=={n:1 for n in statics}
    finally:
        backend.close()
