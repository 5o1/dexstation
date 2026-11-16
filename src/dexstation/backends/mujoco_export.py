"""Dual-arm collection dynamics, separate from the frozen direction-key task.

Only robot position actuators receive commands. Native gamepad displacement,
contact, and signals are observed after MuJoCo integration, without projection.
"""
from __future__ import annotations

from dexstation.contracts.versions import MUJOCO_SNAPSHOT_VERSION
import json
import copy
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from dexstation.backends.object_types import object_type

from dexstation.model.physics_utils import project_inertia_tensor, resolve_asset_files
from dexstation.control.commands import ArmCommand, HandCommand
from dexstation.model.contact import has_robot_contact
from dexstation.recording.episode import FINGERS, SIDES, Observation, fingertip_offsets, tip_pose
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES, G20_MIMIC_JOINTS, apply_mimic_joint_positions
from dexstation.config.paths import resolve_workspace_path
from dexstation.model.robot_description import compose_robot_urdf
from dexstation.model.fixture_collision import add_mujoco_fixture, CONTACT_OFFSET

# Same unmeasured drive placeholders as the existing SAPIEN workcell. These
# settings are local to collection and never alter the direction-key task.
HAND_GAINS = dict(kp=240., kv=8., force_limit=60., armature=1e-4)
ARM_GAINS = dict(kp=2e4, kv=2e3, force_limit=500., armature=1e-2)


def _mujoco_visuals(root):
    """MuJoCo cannot read glTF. g20_pbr_description ships an OBJ beside every GLB
    visual and a materials.json palette; use those, colouring by the visual's material."""
    for visual in root.iter('visual'):
        mesh = visual.find('geometry/mesh')
        if mesh is None or not mesh.get('filename', '').endswith('.glb'):
            continue
        glb = Path(mesh.get('filename'))
        if not glb.with_suffix('.obj').is_file():
            raise ValueError(f'MuJoCo needs an OBJ beside {glb}')
        mesh.set('filename', str(glb.with_suffix('.obj')))
        palette = glb.parents[3] / 'materials.json'
        if palette.is_file() and visual.find('material') is None:
            entries = json.loads(palette.read_text())
            key = next((k for k in entries if (visual.get('name') or '').endswith('_' + k)), None)
            if key:
                h = entries[key]['hex']
                material = ET.SubElement(visual, 'material', name='g20_' + key)
                ET.SubElement(material, 'color', rgba=' '.join(f'{int(h[i:i + 2], 16) / 255:.4f}' for i in (1, 3, 5)) + ' 1')


def _robot_urdf(spec, path, side):
    """Compose live arm/hand joints and repair only invalid input inertias."""
    compose_robot_urdf(spec, path, side)
    tree = ET.parse(path)
    corrections = []
    for link in tree.getroot().findall("link"):
        element = link.find("inertial/inertia")
        if element is None:
            continue
        a = {k: float(v) for k, v in element.attrib.items()}
        tensor = np.array([[a['ixx'], a['ixy'], a['ixz']],
                           [a['ixy'], a['iyy'], a['iyz']],
                           [a['ixz'], a['iyz'], a['izz']]])
        projected = project_inertia_tensor(tensor)
        if not np.array_equal(tensor, projected):
            corrections.append(dict(link=link.get('name'), before=tensor.tolist(),
                                    after=projected.tolist()))
            for key, i, j in [('ixx',0,0), ('iyy',1,1), ('izz',2,2),
                              ('ixy',0,1), ('ixz',0,2), ('iyz',1,2)]:
                element.set(key, str(projected[i, j]))
    _mujoco_visuals(tree.getroot())
    ext = ET.SubElement(tree.getroot(), 'mujoco')
    ET.SubElement(ext, 'compiler', discardvisual='false', fusestatic='false', strippath='false')
    tree.write(path)
    return corrections


def _scene_xml(parent, spec, work_dir):
    """Preserve URDF/mount frame precision across MjSpec's six-digit writer.

    These are the original geometric transforms, not fitted FK corrections.
    Restoring only base quaternions did not resolve the measured error; the
    URDF child-frame rotations must survive both XML round trips as well.
    """
    from dexstation.model.kinematics import _origin
    from dexstation.model.spec import matrix_to_pose
    root = ET.fromstring(parent.to_xml())
    for frame in root.find('worldbody').findall('frame'):
        child = frame.find('body')
        for side in SIDES:
            if child is not None and child.get('name', '').startswith(side + '_'):
                pose = getattr(spec, side + '_arm').base_pose
                frame.set('pos', ' '.join(repr(float(v)) for v in pose[:3]))
                frame.set('quat', ' '.join(repr(float(v)) for v in pose[3:]))
    for side in SIDES:
        source = Path(work_dir) / (side + '_arm_hand.urdf')
        for joint in ET.parse(source).getroot().findall('joint'):
            body = root.find(f".//body[@name='{side}_{joint.find('child').get('link')}']")
            if body is None:
                raise ValueError('compiled robot lost a URDF frame')
            pose = matrix_to_pose(_origin(joint))
            body.set('pos', ' '.join(repr(float(v)) for v in pose[:3]))
            body.set('quat', ' '.join(repr(float(v)) for v in pose[3:]))
    return ET.tostring(root, encoding='unicode')


class CollectionMujocoBackend:
    name = "mujoco"

    def __init__(self, spec, *, work_dir=None, render=False, resource_root=None,
                 environment_collisions=True, gamepad_collisions=None, motion_profile=None,
                 export_only=False, runtime_scene=None, fixture_meshes=None):
        if render:
            raise ValueError("collection physics is renderer-free; use OperatorView")
        import mujoco as mj

        if spec.right_arm is None or spec.right_hand is None:
            raise ValueError('collection requires both arms and both hands')
        if spec.gamepad is not None and not spec.gamepad.fixed_root:
            raise ValueError('collection v1 retains the fixed gamepad mount')
        self.mj, self.spec = mj, spec
        self.motion = None
        if motion_profile is not None and motion_profile != 'legacy':
            from dexstation.control.joint_motion import JointMotion, limits_from_spec
            self.motion = JointMotion(spec, motion_profile, limits=limits_from_spec(spec))
        self.environment_collisions = bool(environment_collisions)
        if gamepad_collisions is not None and bool(gamepad_collisions) != self.environment_collisions:
            raise NotImplementedError("the gamepad-contact variant is implemented for SAPIEN only")
        self.gamepad_collisions = self.environment_collisions
        self._temporary = tempfile.TemporaryDirectory(prefix='collection_mujoco_') if work_dir is None else None
        self.work_dir = Path(self._temporary.name if self._temporary else work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._renderer = None
        self._render_enabled = render
        self.tip_offsets = None if export_only else fingertip_offsets(spec)
        resource_root = resolve_workspace_path(str(resource_root or
            'workspace://assets/xbox_gamepad_physics_mujoco_84f43e6/resources'))
        parent = mj.MjSpec()
        corrections = {}
        for side, own_bit, other_bit in [('left', 2, 4), ('right', 4, 2)]:
            arm = getattr(spec, side + '_arm')
            source = self.work_dir / (side + '_arm_hand.urdf')
            corrections[side] = _robot_urdf(spec, source, side)
            robot = mj.MjSpec.from_file(str(source))
            for body in robot.bodies:
                body.gravcomp = 1.
            for geom in robot.geoms:
                if geom.contype or geom.conaffinity:
                    # Disable same-side self-collision, keep opposite robots,
                    # the gamepad, and the unchanged fixed support in contact.
                    geom.contype, geom.conaffinity = own_bit, 1 | other_bit | 8
                    geom.margin = CONTACT_OFFSET
            for joint in robot.joints:
                joint.armature = (ARM_GAINS if joint.name in arm.joint_names else HAND_GAINS)['armature']
            frame = parent.worldbody.add_frame(pos=arm.base_pose[:3], quat=arm.base_pose[3:])
            parent.attach(robot, prefix=side + '_', frame=frame)
        xml = ET.fromstring(_scene_xml(parent, spec, self.work_dir))
        equality = ET.SubElement(xml, 'equality')
        actuator = ET.SubElement(xml, 'actuator')
        self._hand_names, self._arm_names = {}, {}
        for side in SIDES:
            local = 'right_' if side == 'right' else ''
            hand_names = [side + '_' + local + n for n in G20_JOINT_NAMES]
            arm_names = [side + '_' + n for n in getattr(spec, side + '_arm').joint_names]
            self._hand_names[side], self._arm_names[side] = hand_names, arm_names
            for child, (source, multiplier) in G20_MIMIC_JOINTS[side].items():
                ET.SubElement(equality, 'joint', name=side + '_mimic_' + child,
                    joint1=side + '_' + local + child, joint2=side + '_' + local + source,
                    polycoef=f'0 {multiplier} 0 0 0')
            independent = [n for n in G20_JOINT_NAMES if n not in G20_MIMIC_JOINTS[side]]
            for name, gain in ([(side + '_' + local + n, HAND_GAINS) for n in independent] +
                               [(n, ARM_GAINS) for n in arm_names]):
                ET.SubElement(actuator, 'position', name='drive_' + name, joint=name,
                    kp=str(gain['kp']), kv=str(gain['kv']), forcelimited='true',
                    forcerange=f"{-gain['force_limit']} {gain['force_limit']}")
        parent = mj.MjSpec.from_string(ET.tostring(xml, encoding='unicode'))
        table = spec.table
        parent.worldbody.add_geom(name='table', type=mj.mjtGeom.mjGEOM_BOX,
            size=table.half_extents, pos=table.pose[:3], quat=table.pose[3:],
            contype=16, conaffinity=7, margin=CONTACT_OFFSET, rgba=[.12,.4,.2,1])
        for fixture in spec.fixtures:
            if fixture.parent and fixture.parent != spec.world_frame:
                if fixture.collision:
                    raise ValueError('attached collision fixtures are unsupported')
                continue  # Camera shells are optional visual geometry.
            if fixture_meshes and fixture.name in fixture_meshes:
                for index,mesh in enumerate(fixture_meshes[fixture.name]):
                    name = f'{fixture.name}_convex_{index:03d}'
                    parent.add_mesh(name=name, file=str(Path(mesh).resolve()))
                    parent.worldbody.add_geom(name=name, type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=name, pos=fixture.pose[:3], quat=fixture.pose[3:],
                        contype=8, conaffinity=6, margin=CONTACT_OFFSET)
                continue
            if fixture.type == 'mesh' and fixture.collision and fixture.convex_mesh is None:
                from dexstation.profiles.export import vendor_convex_fixture
                evidence = vendor_convex_fixture(fixture)
                if evidence is not None:
                    meshname = fixture.name + '_vendor_convex'
                    parent.add_mesh(name=meshname, file=fixture.mesh)
                    parent.worldbody.add_geom(name=fixture.name, type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=meshname, pos=fixture.pose[:3], quat=fixture.pose[3:],
                        contype=8, conaffinity=6, margin=CONTACT_OFFSET)
                    continue
            add_mujoco_fixture(parent, mj, fixture)
        attachment = None
        if spec.gamepad is not None:
            from xbox_gamepad_physics.mujoco import attach_to_spec
            from xbox_gamepad_physics.mujoco.numerics import apply_standalone_policy
            from dexstation.profiles.export import gamepad_collision_mesh
            gamepad_object = next((o for o in runtime_scene.objects
                if object_type(o.type).legacy_gamepad), None) if runtime_scene else None
            collision = gamepad_collision_mesh(gamepad_object) if gamepad_object else spec.gamepad.collision
            attachment = attach_to_spec(parent, resource_root=resource_root,
                fixed_root=True, pose=spec.gamepad.root_pose, profile=spec.gamepad.profile,
                attach_visual=render, collision=collision)
            for geom in parent.geoms:
                if geom.name.startswith(attachment.prefix) and (geom.contype or geom.conaffinity):
                    # Internal gamepad excludes remain owned by the asset.
                    geom.contype, geom.conaffinity = 1, 2 | 4 | 16
                    geom.margin = CONTACT_OFFSET
            apply_standalone_policy(parent.option, attachment.asset.numerical_policy)
            parent.option.sdf_iterations = attachment.asset.sdf_policy['sdf_iterations']
            parent.option.sdf_initpoints = attachment.asset.sdf_policy['sdf_initpoints']
        else:
            if runtime_scene is None:
                raise ValueError('empty MJCF construction requires RuntimeScene.workbench physics')
            parent.option.timestep = runtime_scene.workbench['physics']['mujoco']['timestep_s']
        self.object_bodies = {}
        if runtime_scene is not None:
            from dexstation.profiles.export import add_mujoco_objects, configure_mujoco_collision
            self.object_bodies = add_mujoco_objects(parent, mj, runtime_scene.objects,
                                                   attachment=attachment)
            configure_mujoco_collision(parent, runtime_scene, self.object_bodies)
        if not self.environment_collisions:
            for geom in parent.geoms:
                # Robot groups and same/opposite-side policy are unchanged.
                # Disable both masks for every environment/gamepad geometry.
                if geom.contype not in (2, 4):
                    geom.contype, geom.conaffinity = 0, 0
        parent.worldbody.add_light(pos=[.5,.6,2.5], dir=[0,0,-1], diffuse=[.8,.8,.8])
        self.model = parent.compile()
        self.data = mj.MjData(self.model)
        self.gamepad = attachment.bind(self.model, self.data) if attachment is not None else None
        if export_only:
            # Only construction is generalized here; the legacy online adapter
            # below retains its gamepad observation/reset contract.
            (self.work_dir / 'inertia_corrections.json').write_text(json.dumps(corrections, indent=2) + '\n')
            (self.work_dir / 'scene.xml').write_text(resolve_asset_files(_scene_xml(parent, spec, self.work_dir), resource_root))
            return
        self._hand_indices = {s: np.array([self.model.joint(n).qposadr[0]
                                          for n in self._hand_names[s]]) for s in SIDES}
        self._arm_indices = {s: np.array([self.model.joint(n).qposadr[0]
                                         for n in self._arm_names[s]]) for s in SIDES}
        self._robot_bodies = {b for b in range(self.model.nbody)
            if self.model.body(b).name.startswith(('left_', 'right_'))}
        self._palms, self._tips = {}, {}
        for side in SIDES:
            prefix = side + '_' + ('right_' if side == 'right' else '')
            self._palms[side] = self.model.body(prefix + getattr(spec, side + '_hand').root_link).id
            self._tips[side] = [self.model.body(prefix + f + '_distal').id for f in FINGERS]
        (self.work_dir / 'inertia_corrections.json').write_text(json.dumps(corrections, indent=2) + '\n')
        (self.work_dir / 'scene.xml').write_text(resolve_asset_files(_scene_xml(parent, spec, self.work_dir), resource_root))
        self.teleop_hand_bounds = {s: np.array([self.model.joint(n).range for n in self._hand_names[s]]).T.tolist()
                                  for s in SIDES}
        self.reset()

    def reset(self):
        from xbox_gamepad_physics.backends.common.handle import ResetRequest
        self.mj.mj_resetData(self.model, self.data)
        self.gamepad.reset(ResetRequest())
        for side in SIDES:
            for indices, names, values in [
                (self._arm_indices[side], self._arm_names[side], getattr(self.spec, side + '_arm').home_q),
                (self._hand_indices[side], self._hand_names[side], getattr(self.spec, side + '_hand').home_q),
            ]:
                self.data.qpos[indices] = values  # Initial condition only.
                for name, value in zip(names, values):
                    try:
                        self.data.ctrl[self.model.actuator('drive_' + name).id] = value
                    except KeyError:
                        pass  # Mimic joints have native equality, no actuator.
        self._robot_contact = False
        self._requested_time = 0.
        self._physics_ticks = 0
        self.mj.mj_forward(self.model, self.data)
        self.applied_reference = {}
        if self.motion is not None:
            self.motion.reset()
            self.motion_drives = {}
            for side in SIDES:
                values = self.motion.drive_parameters(side)
                for kind, names in (('arm', self._arm_names[side]), ('hand', self._hand_names[side])):
                    for i, name in enumerate(names):
                        joint = self.model.joint(name)
                        self.model.dof_armature[joint.dofadr[0]] = values[kind]['armature'][i]
                        self.model.dof_frictionloss[joint.dofadr[0]] = 0.
                        if kind == 'hand' and G20_JOINT_NAMES[i] in G20_MIMIC_JOINTS[side]:
                            continue
                        actuator = self.model.actuator('drive_'+name)
                        kp, kv = [float(values[kind][key][i]) for key in ('stiffness','damping')]
                        actuator.gainprm[0] = kp
                        actuator.biasprm[1:3] = [-kp, -kv]
                        cap = values[kind]['force_limit']
                        self.model.actuator_forcelimited[actuator.id] = cap is not None
                        if cap is not None:actuator.forcerange[:] = [-float(cap[i]),float(cap[i])]
                self.motion_drives[side] = {kind:{k:None if v is None else v.tolist() for k,v in settings.items()} for kind,settings in values.items()}
            self.mj.mj_forward(self.model, self.data)

    def apply(self, commands):
        if self.motion is not None:
            self.motion.accept(commands)
            return
        self._apply_targets(commands)

    def _apply_targets(self, commands):
        for command in commands:
            if command.side not in SIDES:
                raise ValueError('unknown robot side')
            if isinstance(command, HandCommand):
                values = np.array([command.joint_positions[n] for n in G20_JOINT_NAMES], float)
                names = self._hand_names[command.side]
                mimic = G20_MIMIC_JOINTS[command.side]
                selected = [(name, value) for n, name, value in zip(G20_JOINT_NAMES, names, values) if n not in mimic]
            elif isinstance(command, ArmCommand):
                values = np.asarray(command.positions, float)
                if values.shape != (7,):
                    raise ValueError('arm command requires seven radians')
                selected = list(zip(self._arm_names[command.side], values))
            else:
                raise TypeError('expected normalized HandCommand or ArmCommand')
            if not np.isfinite(values).all():
                raise ValueError('robot targets must be finite')
            for name, value in selected:
                joint = self.model.joint(name)
                if joint.limited[0]:
                    value = np.clip(value, *joint.range)
                self.data.ctrl[self.model.actuator('drive_' + name).id] = value
            if isinstance(command,ArmCommand):
                applied=[float(self.data.ctrl[self.model.actuator('drive_'+n).id]) for n in self._arm_names[command.side]]
            else:
                applied=list(values)
                for i,n in enumerate(self._hand_names[command.side]):
                    if G20_JOINT_NAMES[i] not in G20_MIMIC_JOINTS[command.side]:
                        applied[i]=float(self.data.ctrl[self.model.actuator('drive_'+n).id])
                applied=apply_mimic_joint_positions(command.side,applied)
            self.applied_reference[command.side+('_arm' if isinstance(command,ArmCommand) else '_hand')]=applied

    def step(self, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('control interval must be finite and positive')
        # Accumulate the requested clock, rather than rounding each 30 Hz frame
        # independently. Actual state remains on the native 0.5 ms grid.
        self._requested_time += dt
        target_ticks = round(self._requested_time / self.model.opt.timestep)
        count = target_ticks - self._physics_ticks
        self._robot_contact = False
        for tick in range(self._physics_ticks, target_ticks):
            before_step = getattr(self, 'before_step', None)
            if before_step is not None:before_step(tick*self.model.opt.timestep)
            if self.motion is not None:
                self._apply_targets(self.motion.advance_to(tick*self.model.opt.timestep))
            after_input = getattr(self, 'after_input', None)
            if after_input is not None:after_input()
            self.mj.mj_step(self.model, self.data)
            if not self._robot_contact:
                self._robot_contact = has_robot_contact(
                    (float(contact.dist), any(int(self.model.geom_bodyid[g]) in self._robot_bodies
                                              for g in contact.geom))
                    for contact in self.data.contact)
        self._physics_ticks = target_ticks
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('MuJoCo robot dynamics became non-finite')
        for warning in (self.mj.mjtWarning.mjWARN_BADQPOS, self.mj.mjtWarning.mjWARN_BADQVEL,
                        self.mj.mjtWarning.mjWARN_BADQACC):
            if self.data.warning[int(warning)].number:
                raise FloatingPointError('MuJoCo reported an invalid dynamics warning')
        self.mj.mj_forward(self.model, self.data)

    def _pose(self, body):
        pose = np.r_[self.data.xpos[body], self.data.xquat[body]].copy()
        if pose[3] < 0:
            pose[3:] *= -1
        return pose

    def observe(self):
        state = self.gamepad.gamepad_state()
        observation = Observation(float(self.data.time), time.time_ns(),
            np.array([self.data.qpos[self._hand_indices[s]] for s in SIDES]),
            np.array([self.data.qpos[self._arm_indices[s]] for s in SIDES]),
            np.array([self._pose(self._palms[s]) for s in SIDES]),
            np.array([[tip_pose(self._pose(b), self.tip_offsets[s][f])
                for f, b in zip(FINGERS, self._tips[s])] for s in SIDES]),
            np.asarray(state.values, float), np.asarray(state.available_mask, bool),
            self.gamepad.joint_positions(), self._robot_contact, self.name)
        observation.validate()
        return observation

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._temporary is not None:
            self._temporary.cleanup()

    def snapshot_motion_state(self):
        kind=self.mj.mjtState.mjSTATE_INTEGRATION
        data=np.empty(self.mj.mj_stateSize(self.model,kind))
        self.mj.mj_getState(self.model,self.data,data,kind)
        return dict(version=MUJOCO_SNAPSHOT_VERSION,backend=self.name,state_type=int(kind),integration=data.tolist(),
                    ticks=self._physics_ticks,requested_time=self._requested_time,
                    motion=self.motion.snapshot(),applied_reference=copy.deepcopy(self.applied_reference))

    def restore_motion_state(self,state):
        if state['version']!=MUJOCO_SNAPSHOT_VERSION or state['backend']!=self.name:raise ValueError('incompatible physical snapshot')
        self.mj.mj_setState(self.model,self.data,np.array(state['integration']),state['state_type'])
        self.mj.mj_forward(self.model,self.data)
        # mj_forward can update warm-start state. Restore the captured integration state.
        self.mj.mj_setState(self.model,self.data,np.array(state['integration']),state['state_type'])
        self._physics_ticks=state['ticks'];self._requested_time=state['requested_time']
        self.motion.restore(state['motion']);self.applied_reference=copy.deepcopy(state['applied_reference'])
