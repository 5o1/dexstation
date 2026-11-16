"""Dual-arm collection adapter, with an explicitly renderer-free CPU path.

Only drive targets are changed during replay. Initial qpos assignment is limited
to reset. This separate adapter leaves the existing direction-pad task intact.
"""
from __future__ import annotations

from dexstation.contracts.versions import SAPIEN_SNAPSHOT_VERSION
import math
import copy
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

from dexstation.control.commands import ArmCommand, HandCommand
from dexstation.backends.object_types import object_type
from dexstation.backends.channel_types import channel_type
from dexstation.model.contact import has_robot_contact
from dexstation.recording.episode import FINGERS, SIDES, Observation, fingertip_offsets, tip_pose
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES, apply_mimic_joint_positions
from dexstation.model.robot_description import compose_robot_urdf
from dexstation.model.fixture_collision import add_sapien_fixture_collision, configure_sapien_shapes, configure_category_shapes
from dexstation.contracts.profiles import RuntimeScene
from dexstation.contracts.backend import RobotObservation, StepObservation
from dexstation.profiles.task_objects import load_task_object, collision_options, render_spec

PHYSICS_DT = 0.001
HAND_GAINS = dict(stiffness=240., damping=8., force_limit=60., armature=1e-4)
ARM_GAINS = dict(stiffness=2e4, damping=2e3, force_limit=500., armature=1e-2)


def _physics_urdf(source, destination):
    """Strip rendering input before SAPIEN parses it; preserve all physics."""
    source, destination = Path(source), Path(destination)
    root = ET.parse(source).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"visual", "material"}:
                parent.remove(child)
    for mesh in root.iter("mesh"):
        name = mesh.get("filename", "")
        if name and not name.startswith(("/", "package://")):
            mesh.set("filename", str((source.parent / name).resolve()))
    ET.ElementTree(root).write(destination, encoding="unicode")
    return destination


class _PhysicsLoader:
    """Public-loader adapter for asset URDFs that still contain visual tags."""
    def __init__(self, loader, directory):
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_directory", directory)

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __setattr__(self, name, value):
        setattr(self._loader, name, value)

    def _path(self, path):
        return str(_physics_urdf(path, self._directory / "gamepad_physics.urdf"))

    def parse(self, path, *args, **kwargs):
        return self._loader.parse(self._path(path), *args, **kwargs)

    def load(self, path, *args, **kwargs):
        return self._loader.load(self._path(path), *args, **kwargs)

    def load_multiple(self, path, *args, **kwargs):
        return self._loader.load_multiple(self._path(path), *args, **kwargs)


class _PhysicsScene:
    def __init__(self, scene, directory):
        self._scene, self._directory = scene, directory

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def create_urdf_loader(self):
        return _PhysicsLoader(self._scene.create_urdf_loader(), self._directory)


def _solver_record(gamepad, robots):
    """xbox-gamepad-physics >= 0.4.3 selects the gamepad's solver iterations from
    the scene timestep at load time. PhysX solves an island (and a merged batch)
    with the maximum iterations of its members, so the robots' counts matter too."""
    policy = getattr(gamepad, "solver_policy", None)
    record = dict(policy=None if policy is None else dict(
        timestep_s=policy.timestep_s, validated_timestep_s=policy.validated_timestep_s,
        iterations=None if policy.iterations is None else list(policy.iterations), source=policy.source))
    def readback(articulation):
        try:
            return [articulation.get_solver_position_iterations(), articulation.get_solver_velocity_iterations()]
        except AttributeError:
            return None
    record["gamepad_readback"] = readback(getattr(gamepad, "articulation", None))
    record["robot_readback"] = {side: readback(robot) for side, robot in robots.items()}
    return record


def _pose(sapien, values):
    return sapien.Pose(list(values[:3]), list(values[3:]))


def _world_pose(link):
    pose = link.get_entity_pose()
    values = np.asarray([*pose.p, *pose.q], dtype=float)
    values[3:] /= np.linalg.norm(values[3:])
    if values[3] < 0:
        values[3:] *= -1
    return values


def _observe_object_pose(backend, obj, channel, raw):
    return obj.pose()


def _observe_object_velocity(backend, obj, channel, raw):
    return obj.velocity()


def _observe_articulation_q(backend, obj, channel, raw):
    return obj.body.get_qpos().tolist()


def _observe_contact(backend, obj, channel, raw):
    targets = {getattr(link, 'entity', link) for link in obj.links()}
    value = []
    for group in raw[channel.name]['with']:
        side, kind = group.split('_', 1)
        entities = {link.entity for name, link in backend.links[side].items()
            if name.startswith('AR5_') == (kind == 'arm')}
        value.append(any(any(b.entity in entities for b in contact.bodies)
            and any(b.entity in targets for b in contact.bodies)
            and any(p.separation <= 0 for p in contact.points)
            for contact in backend.scene.get_contacts()))
    return value


class CollectionSapienBackend:
    """CPU PhysX collection backend; rendering is an explicit separate option."""
    name = "sapien"
    timestep_s = PHYSICS_DT
    timestep_source = "workbench 1 kHz"

    def __init__(self, spec, work_dir=None, render=False, resource_root=None,
                 environment_collisions=True, gamepad_collisions=None, motion_profile=None):
        if render:
            raise ValueError("collection physics is renderer-free; use OperatorView")
        import sapien

        self.runtime_scene = spec if isinstance(spec, RuntimeScene) else None
        self.object_definitions = self.runtime_scene.objects if self.runtime_scene else ()
        if self.runtime_scene is not None:
            spec = render_spec(self.runtime_scene)
        self.spec = spec
        self.objects = {}
        if spec.right_arm is None or spec.right_hand is None:
            raise ValueError("collection requires both arms and both hands")
        self.motion = None
        if motion_profile is not None and motion_profile != 'legacy':
            from dexstation.control.joint_motion import JointMotion, limits_from_spec
            self.motion = JointMotion(spec, motion_profile, limits=limits_from_spec(spec))
        self.environment_collisions = bool(environment_collisions)
        # gamepad-contact: robots touch only the gamepad, so a hanging master arm
        # passes through the table instead of jamming against it.
        self.gamepad_collisions = self.environment_collisions if gamepad_collisions is None else bool(gamepad_collisions)
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="collection_sapien_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.render_enabled = bool(render)
        systems = [sapien.physx.PhysxCpuSystem()]
        self.scene = sapien.Scene(systems)
        self.scene.set_timestep(PHYSICS_DT)
        self.tip_offsets = fingertip_offsets(spec)
        self._fixed_actors = []
        table = self.scene.create_actor_builder()
        if self.environment_collisions:
            table.add_box_collision(half_size=list(spec.table.half_extents))
        actor = table.build_static(name="table")
        actor.set_pose(_pose(sapien, spec.table.pose))
        configure_sapien_shapes(actor)
        self._fixed_actors.append(actor)
        # compose() also projects static task objects into spec.fixtures for the
        # legacy WorkcellSpec view; those are built once below by load_task_object.
        static_objects = {obj.name for obj in self.object_definitions if object_type(obj.type).static}
        for fixture in spec.fixtures:
            if not fixture.collision or fixture.name in static_objects:
                continue
            if fixture.parent not in (None, spec.world_frame):
                raise ValueError("collection collision fixture must be world-fixed")
            builder = self.scene.create_actor_builder()
            if self.environment_collisions:
                add_sapien_fixture_collision(builder, fixture)
            actor = builder.build_static(name=fixture.name)
            actor.set_pose(_pose(sapien, fixture.pose))
            configure_sapien_shapes(actor, exclude_fixture_gamepad=True)
            self._fixed_actors.append(actor)
        if self.runtime_scene is not None:
            for actor in self._fixed_actors:
                configure_category_shapes(actor, 'environment')
            for obj in self.object_definitions:
                if object_type(obj.type).static:
                    self.objects[obj.name] = load_task_object(self.scene, obj, work_dir=self.work_dir, collisions=self.environment_collisions)
        self.robots, self.joints, self.links, self.joint_order = {}, {}, {}, {}
        self._robot_entities = set()
        self.motion_drives = {}
        for index, side in enumerate(SIDES):
            arm, hand = getattr(spec, side + "_arm"), getattr(spec, side + "_hand")
            urdf = compose_robot_urdf(spec, self.work_dir / (side + "_robot.urdf"), side)
            urdf = _physics_urdf(urdf, self.work_dir / (side + "_physics.urdf"))
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            robot = loader.load(str(urdf))
            robot.set_root_pose(_pose(sapien, arm.base_pose))
            self.robots[side] = robot
            self.joints[side] = {j.name: j for j in robot.get_active_joints()}
            self.joint_order[side] = list(self.joints[side])
            self.links[side] = {link.name: link for link in robot.get_links()}
            prefix = "right_" if side == "right" else ""
            for name in arm.joint_names:
                self._configure(self.joints[side][name], ARM_GAINS)
            for name in G20_JOINT_NAMES:
                self._configure(self.joints[side][prefix + name], HAND_GAINS)
            if self.motion is not None:
                values = self.motion.drive_parameters(side,backend='sapien')
                for kind, names in (('arm', arm.joint_names), ('hand', [prefix+n for n in G20_JOINT_NAMES])):
                    for i, name in enumerate(names):
                        joint=self.joints[side][name]
                        joint.set_drive_property(stiffness=float(values[kind]['stiffness'][i]), damping=float(values[kind]['damping'][i]),
                            force_limit=float(np.finfo(np.float32).max) if values[kind]['force_limit'] is None else float(values[kind]['force_limit'][i]), mode='force')
                        joint.set_armature([float(values[kind]['armature'][i])]);joint.friction=0.
                self.motion_drives[side]={kind:{k:None if v is None else v.tolist() for k,v in settings.items()} for kind,settings in values.items()}
            for link in robot.get_links():
                if self.motion is not None:
                    # Vendor baseline specifies joint PD, not extra world-space body drag.
                    link.linear_damping = 0.
                    link.angular_damping = 0.
                configure_sapien_shapes(link)
                if self.runtime_scene is not None:
                    configure_category_shapes(link, 'robot')
                self._robot_entities.add(link.entity)
                for shape in link.get_collision_shapes():
                    groups = list(shape.get_collision_groups())
                    groups[2] |= 0x00010000 << index
                    shape.set_collision_groups(groups)
        self.teleop_hand_bounds = {s: np.array([self.joints[s][('right_' if s=='right' else '')+n].get_limits()[0]
                                  for n in G20_JOINT_NAMES]).T.tolist() for s in SIDES}
        if self.motion is not None:self._reset_robot_pose()
        previous_resources = os.environ.get("XBOX_GAMEPAD_PHYSICS_RESOURCES")
        if resource_root is not None:
            os.environ["XBOX_GAMEPAD_PHYSICS_RESOURCES"] = str(resource_root)
        self.gamepad = None
        try:
            if self.runtime_scene is None:
                from xbox_gamepad_physics.robotwin import load_into_scene
                self.gamepad = load_into_scene(
                    _PhysicsScene(self.scene, self.work_dir),
                    _pose(sapien, spec.gamepad.root_pose), fixed_root=spec.gamepad.fixed_root,
                    profile=spec.gamepad.profile, attach_visual=False,
                    collision=spec.gamepad.collision)
            else:
                for obj in self.object_definitions:
                    if not object_type(obj.type).static:
                        self.objects[obj.name] = load_task_object(self.scene, obj,
                            physics_scene=_PhysicsScene(self.scene, self.work_dir), work_dir=self.work_dir)
                self.gamepad = next((o.gamepad for o in self.objects.values() if o.gamepad is not None), None)
                for obj in self.objects.values():
                    for link in obj.links():
                        configure_sapien_shapes(link)
                        configure_category_shapes(link, 'object', collision_options(obj.definition))
        finally:
            if resource_root is not None:
                if previous_resources is None:
                    os.environ.pop("XBOX_GAMEPAD_PHYSICS_RESOURCES", None)
                else:
                    os.environ["XBOX_GAMEPAD_PHYSICS_RESOURCES"] = previous_resources
        self.gamepad_solver = _solver_record(self.gamepad, self.robots)
        self.solver_record = dict(robots=self.gamepad_solver['robot_readback'], objects={})
        for name, obj in self.objects.items():
            self.solver_record['objects'][name] = _solver_record(obj.gamepad,
                {'body': obj.articulation or obj.rigid_body()})
        if self.gamepad is not None:
            if self.runtime_scene is None:
                for link in self.gamepad.articulation.get_links():
                    configure_sapien_shapes(link, exclude_fixture_gamepad=True)
            self._initial_gamepad_q = dict(self.gamepad.joint_positions())
            if not self.gamepad_collisions:
                for link in self.gamepad.articulation.get_links():
                    for shape in link.get_collision_shapes():
                        groups = list(shape.get_collision_groups())
                        groups[0] = groups[1] = 0
                        shape.set_collision_groups(groups)
        self.reset()

    @staticmethod
    def _configure(joint, gains):
        joint.set_drive_property(stiffness=gains["stiffness"], damping=gains["damping"],
                                 force_limit=gains["force_limit"], mode="force")
        joint.set_armature([gains["armature"]])

    def _reset_robot_pose(self):
        for side in SIDES:
            arm, hand = getattr(self.spec, side + "_arm"), getattr(self.spec, side + "_hand")
            prefix = "right_" if side == "right" else ""
            targets = dict(zip(arm.joint_names, arm.home_q))
            targets.update(zip((prefix + n for n in G20_JOINT_NAMES), hand.home_q))
            robot = self.robots[side]
            robot.set_qpos(np.asarray([targets[n] for n in self.joint_order[side]]))
            robot.set_qvel(np.zeros(len(self.joint_order[side])))
            for name, value in targets.items():
                self.joints[side][name].set_drive_target(float(value))

    def reset(self):
        self._ticks = 0
        self._requested_time = 0.
        self._robot_contact = False
        self._reset_robot_pose()
        # Compatible with the installed asset release that predates ResetRequest.
        # This is initialization/reset, never a substitute for contact dynamics.
        if getattr(self, 'runtime_scene', None) is not None:
            for obj in self.objects.values():
                obj.restore(obj.initial)
        elif self.gamepad is not None:
            self.gamepad.set_joint_positions(self._initial_gamepad_q)
            self.gamepad.articulation.set_qvel(np.zeros(len(self._initial_gamepad_q)))
        self.applied_reference = {}
        if self.motion is not None:self.motion.reset()

    def apply(self, commands):
        if self.motion is not None:
            self.motion.accept(commands)
            return
        self._apply_targets(commands)

    def _apply_targets(self, commands):
        for command in commands:
            if command.side not in SIDES:
                raise ValueError("command side must be left or right")
            side = command.side
            if isinstance(command, HandCommand):
                values = apply_mimic_joint_positions(side, [command.joint_positions[n] for n in G20_JOINT_NAMES])
                prefix = "right_" if side == "right" else ""
                targets = zip((prefix + n for n in G20_JOINT_NAMES), values)
            elif isinstance(command, ArmCommand):
                values = np.asarray(command.positions, float)
                if values.shape != (7,):
                    raise ValueError("arm command requires seven radians")
                targets = zip(getattr(self.spec, side + "_arm").joint_names, values)
            else:
                raise TypeError("unsupported collection command")
            targets = list(targets)
            if not all(math.isfinite(float(value)) for _, value in targets):
                raise ValueError("drive targets must be finite")
            for name, value in targets:
                self.joints[side][name].set_drive_target(float(value))
            self.applied_reference[side+('_arm' if isinstance(command,ArmCommand) else '_hand')] = [float(v) for _,v in targets]

    def step(self, dt):
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("step duration must be finite and positive")
        self._requested_time += dt
        target_tick = math.floor(self._requested_time / PHYSICS_DT + 1e-9)
        self._robot_contact = False
        while self._ticks < target_tick:
            before_step = getattr(self, 'before_step', None)
            if before_step is not None:before_step(self._ticks*PHYSICS_DT)
            if self.motion is not None:
                self._apply_targets(self.motion.advance_to(self._ticks*PHYSICS_DT))
            after_input = getattr(self, 'after_input', None)
            if after_input is not None:after_input()
            for robot in self.robots.values():
                robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self.scene.step()
            self._ticks += 1
            self._robot_contact = self._robot_contact or self._has_robot_contact()
        if any(not np.isfinite(robot.get_qpos()).all() for robot in self.robots.values()):
            raise FloatingPointError("nonfinite SAPIEN articulation state")

    def contact_force(self, side, robot_link, obj, obj_link=None):
        """World force (N) that one robot link exerts on one gamepad link during the last physics step.

        Read-only sensing for input sources (the IK autopilot's stick force loop); nothing is recorded.
        """
        prefix = 'right_' if side == 'right' else ''
        robot = self.links[side][prefix + robot_link].entity
        if self.runtime_scene is None:
            targets = {next(link.entity for link in self.gamepad.articulation.get_links() if link.name == obj)}
        else:
            targets = {getattr(link, 'entity', link) for link in self.objects[obj].links()
                       if obj_link is None or link.name == obj_link}
        total = np.zeros(3)
        for contact in self.scene.get_contacts():
            entities = [body.entity for body in contact.bodies]
            if robot in entities and any(target in entities for target in targets):
                # New contract observes force ON the robot. Legacy autopilot asks
                # for force exerted BY it; preserve that independent old entry.
                sign = 1. if entities[0] is robot else -1.
                if self.runtime_scene is None:
                    sign = -sign
                for point in contact.points:
                    total += sign * np.asarray(point.impulse, float) / PHYSICS_DT
        return total

    def _has_robot_contact(self):
        # PhysX reports contact-offset candidates before surfaces meet. Match
        # MuJoCo's geometric dist <= 0 convention, not merely point presence.
        return has_robot_contact(
            (float(point.separation), any(body.entity in self._robot_entities for body in contact.bodies))
            for contact in self.scene.get_contacts() for point in contact.points)

    def observe(self):
        hands, arms, palms, tips = [], [], [], []
        for side in SIDES:
            q = dict(zip(self.joint_order[side], self.robots[side].get_qpos()))
            prefix = "right_" if side == "right" else ""
            hands.append([q[prefix + n] for n in G20_JOINT_NAMES])
            arms.append([q[n] for n in getattr(self.spec, side + "_arm").joint_names])
            palms.append(_world_pose(self.links[side][prefix + getattr(self.spec, side + "_hand").root_link]))
            tips.append([tip_pose(_world_pose(self.links[side][prefix + finger + "_distal"]),
                                  self.tip_offsets[side][finger]) for finger in FINGERS])
        if self.runtime_scene is not None:
            robot = RobotObservation(self._ticks * PHYSICS_DT, time.time_ns(),
                np.asarray(hands), np.asarray(arms), np.asarray(palms), np.asarray(tips),
                bool(self._robot_contact or self._has_robot_contact()))
            return StepObservation(robot, self._task_observations(),
                                   {name: obj.pose() for name, obj in self.objects.items()})
        state = self.gamepad.gamepad_state()
        observation = Observation(
            sim_time_s=self._ticks * PHYSICS_DT, wall_time_ns=time.time_ns(),
            hand_q=np.asarray(hands), arm_q=np.asarray(arms),
            palm_pose_world=np.asarray(palms), fingertip_pose_world=np.asarray(tips),
            gamepad_signal=np.asarray(state.values, float), gamepad_mask=np.asarray(state.available_mask, bool),
            control_joint_positions=self.gamepad.joint_positions(),
            robot_contact=bool(self._robot_contact or self._has_robot_contact()), backend=self.name)
        observation.validate()
        return observation

    def set_object_state(self, name, pose, twist=None):
        self.objects[name].set_state(pose, twist)

    def _task_observations(self):
        result = {}
        raw = {c['name']: c for c in self.runtime_scene.task.get('observations', [])}
        for channel in self.runtime_scene.channels:
            obj = self.objects[channel.object]
            value = channel_type(channel.kind).observe_sapien(self, obj, channel, raw, globals())
            result[channel.name] = value
        return result

    def snapshot_motion_state(self):
        """Observable physical and controller state; PhysX solver warm-start is not portable."""
        if self.runtime_scene is not None:
            return dict(workbench=dict(backend=self.name, ticks=self._ticks,
                requested_time=self._requested_time, robot_contact=self._robot_contact,
                robots={s: dict(q=r.get_qpos().tolist(), v=r.get_qvel().tolist()) for s, r in self.robots.items()},
                motion=None if self.motion is None else self.motion.snapshot(),
                applied_reference=copy.deepcopy(self.applied_reference)),
                task={name: obj.snapshot() for name, obj in self.objects.items()})
        return dict(version=SAPIEN_SNAPSHOT_VERSION,backend=self.name,ticks=self._ticks,requested_time=self._requested_time,
            robots={s:dict(q=self.robots[s].get_qpos().tolist(),v=self.robots[s].get_qvel().tolist()) for s in SIDES},
            gamepad=dict(q=self.gamepad.articulation.get_qpos().tolist(),v=self.gamepad.articulation.get_qvel().tolist()),
            motion=self.motion.snapshot(),applied_reference=copy.deepcopy(self.applied_reference))

    def restore_motion_state(self,state):
        if self.runtime_scene is not None:
            workbench = state['workbench']
            self._reset_robot_pose()
            for side, robot in self.robots.items():
                robot.set_qpos(np.asarray(workbench['robots'][side]['q']))
                robot.set_qvel(np.asarray(workbench['robots'][side]['v']))
            for name, obj in self.objects.items():
                obj.restore(state['task'][name])
            self._ticks = workbench['ticks']
            self._requested_time = workbench['requested_time']
            self._robot_contact = workbench['robot_contact']
            if self.motion is not None:
                self.motion.restore(workbench['motion'])
                self._apply_targets(self.motion.commands())
            self.applied_reference = copy.deepcopy(workbench['applied_reference'])
            for key, values in self.applied_reference.items():
                side, kind = key.split('_', 1)
                names = getattr(self.spec, side+'_arm').joint_names if kind == 'arm' else [
                    ('right_' if side == 'right' else '')+n for n in G20_JOINT_NAMES]
                for name, value in zip(names, values):
                    self.joints[side][name].set_drive_target(float(value))
            return
        if state['version']!=SAPIEN_SNAPSHOT_VERSION or state['backend']!=self.name:raise ValueError('incompatible physical snapshot')
        for side in SIDES:
            self.robots[side].set_qpos(np.array(state['robots'][side]['q']))
            self.robots[side].set_qvel(np.array(state['robots'][side]['v']))
        self.gamepad.articulation.set_qpos(np.array(state['gamepad']['q']))
        self.gamepad.articulation.set_qvel(np.array(state['gamepad']['v']))
        self._ticks=state['ticks'];self._requested_time=state['requested_time']
        self.motion.restore(state['motion'])
        self._apply_targets(self.motion.commands())
        self.applied_reference=copy.deepcopy(state['applied_reference'])

    def close(self):
        self.gamepad = None
        self.objects.clear()
        self.links.clear()
        self.joints.clear()
        self.robots.clear()
        self._robot_entities.clear()
        self._fixed_actors.clear()
        self.scene = None
