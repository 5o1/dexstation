"""Type extensions and non-gamepad scenes require no gamepad asset package."""
import builtins
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from helpers import bench_directory
from dexstation.backends import sapien_physics as native
from dexstation.backends.channel_types import ChannelType, CHANNEL_TYPES, register_channel_type
from dexstation.backends.object_types import ObjectType, OBJECT_TYPES, register_object_type
from dexstation.profiles.compose import compose, load_task, load_workbench


class Pose:
    def __init__(self, p=(0., 0., 0.), q=(1., 0., 0., 0.)):
        self.p, self.q = np.asarray(p), np.asarray(q)


class Actor:
    def __init__(self):
        self.pose = Pose()
    def set_pose(self, pose): self.pose = pose
    def get_pose(self): return self.pose
    def find_component_by_type(self, cls): return self
    def get_linear_velocity(self): return np.zeros(3)
    def get_angular_velocity(self): return np.zeros(3)
    def set_linear_velocity(self, value): pass
    def set_angular_velocity(self, value): pass
    def get_collision_shapes(self): return []


class Builder:
    def add_box_collision(self, **kwargs): pass
    def set_mass_and_inertia(self, *args): pass
    def build_static(self, **kwargs): return Actor()
    build = build_static


def block_gamepad_import(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] == 'xbox_gamepad_physics':
            raise AssertionError(f'Unexpected gamepad import: {name}')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


@pytest.mark.parametrize('task_id', ['empty', 'tabletop_cube_grasp'])
def test_empty_and_cube_compose_construct_and_step_without_gamepad(monkeypatch, tmp_path, task_id):
    """Run the real adapter constructor/step against CPU native API doubles.

    This specifically exercises the former unconditional import in __init__;
    it does not claim native dynamics validation without SAPIEN installed.
    """
    block_gamepad_import(monkeypatch)
    runtime = compose(load_workbench(bench_directory()), load_task(task_id), 'sapien', seed=7)
    robots = []
    for side in ('left', 'right'):
        names = list(getattr(runtime.spec, side + '_arm').joint_names)
        names += [('right_' if side == 'right' else '') + name for name in native.G20_JOINT_NAMES]
        joints = [NS(name=name, set_drive_property=lambda **kw: None,
            set_armature=lambda values: None, get_limits=lambda: [[-1., 1.]],
            set_drive_target=lambda value: None) for name in names]
        robots.append(NS(get_active_joints=lambda joints=joints: joints, get_links=lambda: [],
            get_qpos=lambda names=names: np.zeros(len(names)),
            set_qpos=lambda values: None, set_qvel=lambda values: None,
            set_qf=lambda value: None, compute_passive_force=lambda **kw: []))
    loaded = iter(robots)
    ticks = []
    scene = NS(set_timestep=lambda value: None, create_actor_builder=Builder,
        create_urdf_loader=lambda: NS(load=lambda path: next(loaded)),
        get_contacts=lambda: [], step=lambda: ticks.append(1))
    for robot in robots:
        robot.set_root_pose = lambda pose: None
    sapien = NS(Pose=Pose, Scene=lambda systems: scene, physx=NS(
        PhysxCpuSystem=lambda: None, PhysxMaterial=lambda *args: None,
        PhysxRigidDynamicComponent=Actor, PhysxRigidStaticComponent=Actor))
    monkeypatch.setitem(sys.modules, 'sapien', sapien)
    monkeypatch.setattr(native, 'fingertip_offsets', lambda spec: {})
    monkeypatch.setattr(native, 'compose_robot_urdf', lambda *args: 'unused.urdf')
    monkeypatch.setattr(native, '_physics_urdf', lambda source, target: target)
    monkeypatch.setattr(native, 'add_sapien_fixture_collision', lambda *args: None)
    monkeypatch.setattr(native, 'configure_sapien_shapes', lambda *args, **kwargs: None)
    monkeypatch.setattr(native, 'configure_category_shapes', lambda *args, **kwargs: None)
    backend = native.CollectionSapienBackend(runtime, work_dir=tmp_path)
    try:
        backend.step(.002)
        assert backend._ticks == 2 and len(ticks) == 2
        assert set(backend.objects) == {obj.name for obj in runtime.objects}
        assert backend.gamepad is None
    finally:
        backend.close()


def test_registered_object_and_channel_drive_composition_and_observation(monkeypatch):
    calls = []
    monkeypatch.setitem(OBJECT_TYPES, 'extension_body', ObjectType('extension_body', operations={
        'compose': lambda obj, pose, data: calls.append(obj['name']),
        'sapien_load': lambda scene, obj, **kwargs: ('external', obj.name),
        'mujoco_build': lambda parent, mj, obj, **kwargs: [obj.name + '_body'],
        'inventory': lambda obj, description, root: (['hinge'], ['extension_link'], [])}))
    monkeypatch.setitem(CHANNEL_TYPES, 'extension_value', ChannelType('extension_value',
        ((1,), 'float64', 'm'), lambda backend, obj, channel, raw: [42.]))
    task = load_task('tabletop_cube_grasp')
    task['objects'][0]['type'] = 'extension_body'
    task['observations'] = [dict(name='value', kind='extension_value', object=task['objects'][0]['name'])]
    runtime = compose(load_workbench(bench_directory()), task, 'sapien', seed=7)
    assert calls == [runtime.objects[0].name]
    assert runtime.channels[0].shape == (1,)
    from dexstation.profiles.task_objects import load_task_object
    assert load_task_object(None, runtime.objects[0]) == ('external', runtime.objects[0].name)
    adapter = native.CollectionSapienBackend.__new__(native.CollectionSapienBackend)
    adapter.runtime_scene = runtime
    adapter.objects = {runtime.objects[0].name: object()}
    assert adapter._task_observations() == {'value': [42.]}
    from dexstation.profiles.export import add_mujoco_objects, sapien_inventory
    obj = runtime.objects[0]
    assert add_mujoco_objects(None, None, [obj]) == {obj.name: [obj.name + '_body']}
    assert sapien_inventory({'robots': {}, 'objects': [obj.__dict__]}, None) == dict(
        joints=[obj.name + ':hinge'], actuators=[], objects={obj.name: ['extension_link']})


def test_registry_rejects_duplicate_entries():
    with pytest.raises(ValueError, match='already registered'):
        register_object_type(ObjectType('rigid'))
    with pytest.raises(ValueError, match='already registered'):
        register_channel_type(ChannelType('object_pose'))


@pytest.mark.parametrize('entry,register,message', [
    (ObjectType(''), register_object_type, 'nonempty'),
    (ObjectType('broken', operations={'compose': object()}), register_object_type, 'callable'),
    (ChannelType(''), register_channel_type, 'nonempty'),
    (ChannelType('broken', metadata=()), register_channel_type, 'metadata'),
    (ChannelType('broken', sapien_observer=object()), register_channel_type, 'callable'),
])
def test_invalid_entries_are_rejected_before_registration(entry, register, message):
    with pytest.raises((ValueError, TypeError), match=message):
        register(entry)


@pytest.mark.parametrize('render', [False, True])
def test_renderer_scene_creation_preserves_defaults_without_gamepad(monkeypatch, render):
    from dexstation.backends.sapien_render import _create_scene
    block_gamepad_import(monkeypatch)
    calls = []
    scene = NS(set_timestep=lambda value: calls.append(('timestep', value)),
        set_ambient_light=lambda value: calls.append(('ambient', value)),
        add_directional_light=lambda *args, **kw: calls.append(('directional', args, kw)),
        add_point_light=lambda *args, **kw: calls.append(('point', args, kw)))
    engine = NS(set_renderer=lambda renderer: None, create_scene=lambda cfg: scene)
    monkeypatch.setitem(sys.modules, 'sapien', NS(Engine=lambda: engine,
        SapienRenderer=lambda: object(), SceneConfig=lambda: object(), Scene=lambda: scene))
    assert _create_scene(timestep=1 / 240, render=render) is scene
    assert calls[0] == ('timestep', 1 / 240)
    assert calls[1:] == ([('ambient', [.5, .5, .5]),
        ('directional', ([0, .5, -1], [.5, .5, .5]), {'shadow': True}),
        ('point', ([1, 0, 1.8], [1, 1, 1]), {'shadow': True}),
        ('point', ([-1, 0, 1.8], [1, 1, 1]), {'shadow': True})] if render else [])
