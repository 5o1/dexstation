"""Task-owned SAPIEN bodies and render-state transport (design §§4.3, 5.2).

Composition has already resolved initial poses and sampled randomization. This
module never samples again and never chooses a new control target.
"""
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import numpy as np

from dexstation.contracts.profiles import ObjectSpec, CameraPair, RuntimeScene
from dexstation.config.paths import resolve_workspace_path, asset_path
from dexstation.config.site import repository_root
from dexstation.backends.object_types import object_type


def collision_mesh(obj):
    """Read the adjudicated geometry key; accept the pre-WP2 asset spelling."""
    if 'collision_mesh' in obj.geometry:
        return obj.geometry['collision_mesh']
    if isinstance(obj.collision, str):
        return obj.collision
    return obj.extra.get('collision_mesh', 'sdf')


def collision_options(obj):
    if isinstance(obj.collision, dict):
        return obj.collision
    # Only the old built-in asset used a string instead of category switches.
    return dict(robots=True, environment=True, objects=True)


def camera_definitions(scene):
    """Only calibrated members of real/sim pairs become render cameras."""
    from dexstation.model.spec import CameraSpec, IntrinsicsSpec
    definitions = []
    for pair in scene.cameras:
        if pair.sim is None:
            continue
        sim = pair.sim
        intr = dict(sim['intrinsics'])
        intr['distortion'] = tuple(intr.get('distortion', ()))
        definitions.append(CameraSpec(pair.name, '', pair.real.get('model', ''),
            sim['parent'], tuple(sim['pose_optical']), IntrinsicsSpec(**intr),
            sim['status'], sim['source']))
    return tuple(definitions)


def render_spec(scene):
    """Adapt WP0 to the established renderer without changing robot geometry."""
    from dexstation.model.spec import GamepadSpec
    pad = next((obj for obj in scene.objects if object_type(obj.type).legacy_gamepad), None)
    gamepad = None
    if pad is not None:
        cfg = dict(pad.extra, **pad.geometry)
        gamepad = GamepadSpec(cfg['release'], cfg['profile'], collision_mesh(pad),
                             tuple(pad.initial_pose_world), cfg['fixed_root'])
    return replace(scene.spec, gamepad=gamepad, cameras=camera_definitions(scene))


def render_payload(scene):
    """JSON value embedded in the existing operator configuration/mmap flow."""
    spec = asdict(scene.spec)
    spec['source_dir'] = str(scene.spec.source_dir) if scene.spec.source_dir else None
    return dict(spec=spec, objects=[asdict(o) for o in scene.objects],
                cameras=[asdict(c) for c in scene.cameras], task=scene.task)


def render_runtime(spec, payload):
    """Reconstruct only the rendering subset; no profile loader or compositor."""
    from dexstation.model.spec import WorkcellSpec, TableSpec, ArmSpec, HandSpec, FixtureSpec, GamepadSpec
    data = dict(payload['spec'])
    data['table'] = TableSpec(**data['table'])
    for side in ('left', 'right'):
        for kind, cls in (('arm', ArmSpec), ('hand', HandSpec)):
            key = side+'_'+kind
            if data[key] is not None:
                data[key] = cls(**data[key])
    data['gamepad'] = GamepadSpec(**data['gamepad']) if data['gamepad'] else None
    data['fixtures'] = tuple(FixtureSpec(**f) for f in data['fixtures'])
    data['cameras'] = ()  # Rebuilt exclusively from CameraPair below.
    data['source_dir'] = Path(data['source_dir']) if data['source_dir'] else None
    spec = WorkcellSpec(**data)
    return RuntimeScene(spec, {}, payload['task'],
        tuple(ObjectSpec(**o) for o in payload['objects']), (),
        tuple(CameraPair(**c) for c in payload['cameras']), (), {},
        Path('.'), Path('.'), {}, {}, {})


class TaskObject:
    def __init__(self, definition, body, *, gamepad=None):
        self.definition, self.body, self.gamepad = definition, body, gamepad
        self.articulation = body if hasattr(body, 'get_active_joints') else None
        self.initial = self.snapshot()

    def pose(self):
        pose = self.body.get_root_pose() if self.articulation is not None else self.body.get_pose()
        return [*map(float, pose.p), *map(float, pose.q)]

    def links(self):
        return self.body.get_links() if self.articulation is not None else [self.body]

    def rigid_body(self):
        if self.articulation is not None:
            return self.body.get_links()[0]
        import sapien
        for cls in (sapien.physx.PhysxRigidDynamicComponent, sapien.physx.PhysxRigidStaticComponent):
            body = self.body.find_component_by_type(cls)
            if body is not None:
                return body
        raise ValueError('task object has no rigid component')

    def velocity(self):
        body = self.rigid_body()
        if object_type(self.definition.type).static:
            return [0.] * 6
        return [*map(float, body.get_linear_velocity()), *map(float, body.get_angular_velocity())]

    def set_state(self, pose, twist=None):
        import sapien
        pose = sapien.Pose(list(pose[:3]), list(pose[3:]))
        if self.articulation is not None:
            self.body.set_root_pose(pose)
        else:
            self.body.set_pose(pose)
        if twist is not None and not object_type(self.definition.type).static:
            if self.articulation is not None:
                self.body.set_root_linear_velocity(np.asarray(twist[:3], float))
                self.body.set_root_angular_velocity(np.asarray(twist[3:], float))
            else:
                body = self.rigid_body()
                body.set_linear_velocity(np.asarray(twist[:3], float))
                body.set_angular_velocity(np.asarray(twist[3:], float))

    def snapshot(self):
        state = dict(pose=self.pose(), twist=self.velocity())
        if self.articulation is not None:
            state.update(q=self.body.get_qpos().tolist(), v=self.body.get_qvel().tolist())
        return state

    def restore(self, state):
        self.set_state(state['pose'], state['twist'])
        if self.articulation is not None:
            self.body.set_qpos(np.asarray(state['q'], float))
            self.body.set_qvel(np.asarray(state['v'], float))


def load_task_object(scene, obj, *, render=False, physics_scene=None, work_dir=None, collisions=True):
    return object_type(obj.type).invoke('sapien_load', globals(), scene, obj,
        render=render, physics_scene=physics_scene, work_dir=work_dir, collisions=collisions)




def _load_articulation(scene, obj, *, render=False, physics_scene=None, work_dir=None, collisions=True):
    import sapien
    pose = sapien.Pose(list(obj.initial_pose_world[:3]), list(obj.initial_pose_world[3:]))
    loader = scene.create_urdf_loader()
    loader.fix_root_link = obj.geometry['fixed_root']
    source = asset_path(obj.geometry['urdf'])
    if not render:
        from dexstation.backends.sapien_physics import _physics_urdf
        source = str(_physics_urdf(source, Path(work_dir) / (obj.name + '_physics.urdf')))
    mesh_policy = obj.geometry.get('collision_mesh')
    if mesh_policy is None:
        body = loader.load(source)
    else:
        builders, actors, cameras = loader.parse(source)
        if len(builders) != 1 or actors or cameras:
            raise ValueError('task URDF must describe one articulation')
        builder = builders[0]
        for link in builder.link_builders:
            for record in link.collision_records:
                if record.type in ('convex_mesh', 'multiple_convex_meshes', 'nonconvex_mesh'):
                    record.type = 'nonconvex_mesh' if mesh_policy == 'sdf' else 'multiple_convex_meshes'
                    record.decomposition = 'none' if mesh_policy == 'sdf' else 'coacd'
                    # Native cooking/decomposition may cache beside a mesh.
                    # Keep those products in this backend's own work dir.
                    directory = Path(work_dir) / obj.name / link.name
                    directory.mkdir(parents=True, exist_ok=True)
                    original = Path(record.filename)
                    target = directory / original.name
                    if original.resolve() != target.resolve():
                        shutil.copyfile(original, target)
                    record.filename = str(target)
        body = builder.build()
    body.set_root_pose(pose)
    initial = obj.extra.get('initial_joint_positions', {})
    joints = body.get_active_joints()
    body.set_qpos(np.asarray([initial.get(j.name, 0.) for j in joints]))
    body.set_qvel(np.zeros(len(joints)))
    return TaskObject(obj, body)


def _load_primitive(scene, obj, *, render=False, physics_scene=None, work_dir=None, collisions=True):
    import sapien
    pose = sapien.Pose(list(obj.initial_pose_world[:3]), list(obj.initial_pose_world[3:]))
    builder = scene.create_actor_builder()
    material = obj.material
    physical = sapien.physx.PhysxMaterial(material['static_friction'], material['dynamic_friction'],
                                         material['restitution']) if material else None
    geometry = obj.geometry
    if 'box' in geometry:
        size = list(geometry['box']['half_extents'])
        if collisions:
            builder.add_box_collision(half_size=size, material=physical)
        if render:
            builder.add_box_visual(half_size=size)
    elif 'mesh' in geometry:
        mesh = asset_path(geometry['mesh'])
        if collisions:
            if object_type(obj.type).static:
                builder.add_nonconvex_collision_from_file(mesh, material=physical)
            else:
                builder.add_multiple_convex_collisions_from_file(mesh, material=physical,
                    decomposition=collision_mesh(obj))
        if render:
            builder.add_visual_from_file(mesh)
    else:
        raise ValueError(f'unsupported task geometry: {obj.name}')
    if object_type(obj.type).mass:
        mass = obj.inertial['mass_kg']
        if 'box' in geometry:
            half = np.asarray(geometry['box']['half_extents'], float)
            inertia = mass / 3 * np.array([half[1]**2 + half[2]**2,
                half[0]**2 + half[2]**2, half[0]**2 + half[1]**2])
            builder.set_mass_and_inertia(mass, sapien.Pose(), inertia)
        body = builder.build(name=obj.name)
        if 'box' not in geometry:
            component = body.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            # SAPIEN derives the centre/principal inertia from collision volume.
            # Rescale its uniform-density mass properties to the supplied mass.
            component.inertia = np.asarray(component.inertia) * mass / component.mass
            component.mass = mass
    elif object_type(obj.type).static:
        body = builder.build_static(name=obj.name)
    else:
        raise ValueError(f'unsupported task object type: {obj.type}')
    body.set_pose(pose)
    return TaskObject(obj, body)
