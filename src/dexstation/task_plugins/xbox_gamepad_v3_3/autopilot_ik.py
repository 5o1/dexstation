"""Kinematics and damped-least-squares IK for the IK autopilot input source.

numpy only. Arms: the scene's arm URDF chain plus the hand mount. Hands: one chain per
finger from ``hand_base_link`` to ``<finger>_distal``; G20 mimic joints (thumb_ip and
the four DIP joints) are folded into their source joints, exactly as the station's
coupled hand moves. Control geometry is in the gamepad body frame
(``assets/xbox_one_s_controls.json``) and moves with the scene's gamepad pose.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from dexstation.model.kinematics import Chain, _rotation
from dexstation.model.spec import pose_to_matrix

ASSETS = Path(__file__).with_name('assets')
FINGER_JOINTS = {
    'thumb': ('thumb_cmc_roll', 'thumb_cmc_yaw', 'thumb_cmc_pitch', 'thumb_mcp'),
    'index': ('index_mcp_roll', 'index_mcp_pitch', 'index_pip'),
    'middle': ('middle_mcp_roll', 'middle_mcp_pitch', 'middle_pip'),
    'ring': ('ring_mcp_roll', 'ring_mcp_pitch', 'ring_pip'),
    'pinky': ('pinky_mcp_roll', 'pinky_mcp_pitch', 'pinky_pip'),
}


def load_asset(name):
    return json.loads((ASSETS / name).read_text())


@dataclass
class IkResult:
    q: np.ndarray
    error_m: float
    angle_deg: float
    iterations: int


def solve_point(chain, q0, base, local, normal_local, target, push, *, lower=None, upper=None,
                iterations=30, tolerance_m=1e-4, normal_weight=.01, normal_cone_deg=35., damping=.004,
                rest=None, rest_weight=0., surface=None):
    """Move a tip-fixed point onto ``target`` with its outward normal near ``push``.

    The normal term is an inequality: it acts only outside ``normal_cone_deg``.
    ``rest`` adds a weak posture bias in the null space of the point task.
    With ``surface`` (tip-frame samples) the point is, at every iteration, the sample that leads along
    ``push``: the part of the fingertip that meets a surface facing it first, whatever the finger's tilt.
    """
    lower = chain.lower if lower is None else lower
    upper = chain.upper if upper is None else upper
    q = np.clip(np.array(q0, float), lower, upper)
    cosine = math.cos(math.radians(normal_cone_deg))
    n = len(q)
    for iteration in range(iterations + 1):
        if surface is not None:
            tip, _ = chain.frames(q, base)
            local = surface[int(np.argmax(surface @ (tip[:3, :3].T @ push)))]
        p, tip, motion = chain.point(q, base, local)
        normal = tip[:3, :3] @ normal_local if normal_local is not None else None
        error = p - target
        if np.linalg.norm(error) < tolerance_m and (normal is None or normal @ push >= cosine):
            break
        if iteration == iterations:
            break
        jp, jn = Chain.jacobian(motion, p, normal, n)
        rows, residual = [jp], [error]
        if normal is not None and normal @ push < cosine:
            rows.append(normal_weight * jn)
            residual.append(normal_weight * (normal - push))
        J, r = np.vstack(rows), np.concatenate(residual)
        dq = -J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(len(r)), r)
        if rest is not None and rest_weight:
            projector = np.eye(n) - np.linalg.pinv(jp) @ jp
            dq += rest_weight * projector @ (rest - q)
        step = np.abs(dq).max()
        if step > .2:
            dq *= .2 / step
        q = np.clip(q + dq, lower, upper)
    if surface is not None:
        tip, _ = chain.frames(q, base)
        local = surface[int(np.argmax(surface @ (tip[:3, :3].T @ push)))]
    p, tip, _ = chain.point(q, base, local)
    angle = 0. if normal_local is None else math.degrees(math.acos(np.clip((tip[:3, :3] @ normal_local) @ push, -1, 1)))
    return IkResult(q, float(np.linalg.norm(p - target)), angle, iteration)


def pose_error(current, target):
    """6-vector (position m, rotation rad) from current to target."""
    rotation = target[:3, :3] @ current[:3, :3].T
    angle = math.acos(max(-1., min(1., (np.trace(rotation) - 1) / 2)))
    if angle < 1e-9:
        omega = np.zeros(3)
    else:
        omega = angle / (2 * math.sin(angle)) * np.array([rotation[2, 1] - rotation[1, 2],
                                                           rotation[0, 2] - rotation[2, 0],
                                                           rotation[1, 0] - rotation[0, 1]])
    return np.concatenate([target[:3, 3] - current[:3, 3], omega])


class ArmModel:
    """World pose of ``hand_base_link`` for seven arm joints; 6-D pose IK."""

    def __init__(self, spec, side, margin_deg=3.):
        arm, hand = getattr(spec, side + '_arm'), getattr(spec, side + '_hand')
        self.side = side
        self.chain = Chain(arm.urdf, arm.tip_link)
        self.base = pose_to_matrix(arm.base_pose)
        self.mount = pose_to_matrix(hand.mount)
        margin = math.radians(margin_deg)
        self.lower, self.upper = self.chain.lower + margin, self.chain.upper - margin
        self.home = np.asarray(arm.home_q, float)

    def hand_pose(self, q):
        tip, _ = self.chain.frames(q, self.base)
        return tip @ self.mount

    def solve(self, target, q0, *, iterations=200, tolerance=(2e-4, 2e-3), damping=.01):
        q = np.clip(np.array(q0, float), self.lower, self.upper)
        for iteration in range(iterations + 1):
            tip, motion = self.chain.frames(q, self.base)
            pose = tip @ self.mount
            error = pose_error(pose, target)
            if np.linalg.norm(error[:3]) < tolerance[0] and np.linalg.norm(error[3:]) < tolerance[1]:
                break
            if iteration == iterations:
                break
            J = np.zeros((6, 7))
            for axis, origin, column, gain in motion:
                J[:3, column] += gain * np.cross(axis, pose[:3, 3] - origin)
                J[3:, column] += gain * axis
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), error)
            step = np.abs(dq).max()
            if step > .1:
                dq *= .1 / step
            q = np.clip(q + dq, self.lower, self.upper)
        error = pose_error(self.hand_pose(q), target)
        return IkResult(q, float(np.linalg.norm(error[:3])), math.degrees(np.linalg.norm(error[3:])), iteration)


class HandModel:
    """Finger chains in the hand_base_link frame, with each finger's contact pad."""

    def __init__(self, spec, side, pads=None):
        urdf = getattr(spec, side + '_hand').urdf
        self.side = side
        self.fingers = {f: Chain(urdf, f + '_distal') for f in FINGER_JOINTS}
        for finger, chain in self.fingers.items():
            if tuple(chain.active) != FINGER_JOINTS[finger]:
                raise ValueError(f'unexpected {finger} joints {chain.active}')
        pads = pads or load_asset('g20_finger_pads.json')['pads'][side]
        self.pads = {f: (np.asarray(pads[f]['point'], float), np.asarray(pads[f]['normal'], float)) for f in FINGER_JOINTS}
        self.surface = {f: np.asarray(pads[f]['surface'], float) for f in FINGER_JOINTS if 'surface' in pads[f]}

    def pad(self, finger, q, palm):
        local, normal = self.pads[finger]
        point, tip, _ = self.fingers[finger].point(q, palm, local)
        return point, tip[:3, :3] @ normal

    def solve(self, finger, q0, palm, target, push, *, leading=False, **kwargs):
        """Finger IK for the pad point, or with ``leading`` for the fingertip sample that leads along ``push``."""
        local, normal = self.pads[finger]
        surface = self.surface.get(finger) if leading else None
        return solve_point(self.fingers[finger], q0, palm, local, normal, target, push, surface=surface, **kwargs)


class GamepadGeometry:
    """Contact targets for control states, in world coordinates."""

    def __init__(self, spec, controls=None):
        self.data = controls or load_asset('xbox_one_s_controls.json')
        self.controls = self.data['controls']
        self.pose = pose_to_matrix(spec.gamepad.root_pose)
        self.rotation = self.pose[:3, :3]

    def world(self, point):
        return self.rotation @ np.asarray(point, float) + self.pose[:3, 3]

    def direction(self, vector):
        vector = self.rotation @ np.asarray(vector, float)
        return vector / np.linalg.norm(vector)

    def rim_target(self, control, direction, amount=0., depth=0.):
        """Stick tilt by pushing the cap's vertical side wall opposite ``direction``, horizontally in the gamepad
        frame. The contact (mid-height of the side wall) turns with the cap as it tilts by ``amount`` of its
        limit; the push stays horizontal, so it adds nothing along the click axis (gamepad -z).
        """
        c = self.controls[control]
        d = np.array([direction[0], direction[1], 0.])
        d /= np.linalg.norm(d)
        pivot = np.asarray(c['pivot'], float)
        side = np.array([pivot[0], pivot[1], sum(c['cap_side_z']) / 2]) - c['cap_radius_m'] * d
        rotation = _rotation(np.cross([0., 0., 1.], d), amount * c['tilt_limit_rad'])
        point = pivot + rotation @ (side - pivot)
        return self.world(point + depth * d), self.direction(d)

    def target(self, control, *, depth=0., direction=None, amount=0.):
        """Pad-frame contact target and push direction, mapped to world.

        ``depth`` is along the push direction (negative hovers above the surface).
        Sticks: ``direction`` is a unit (x, y) in the gamepad frame (+y toward the bumpers)
        and ``amount`` the tilt as a fraction of the mechanical limit. Triggers: ``amount``
        is the fraction of hinge travel. D-pad and buttons ignore ``amount``.
        """
        c = self.controls[control]
        point, push = np.asarray(c['point'], float), np.asarray(c['push'], float)
        if c['kind'] == 'stick' and direction is not None and amount:
            d = np.array([direction[0], direction[1], 0.])
            d /= np.linalg.norm(d)
            rotation = _rotation(np.cross([0., 0., 1.], d), amount * c['tilt_limit_rad'])
            pivot = np.asarray(c['pivot'], float)
            point = pivot + rotation @ (point - pivot)
            push = rotation @ push
        elif c['kind'] == 'trigger' and amount:
            hinge, axis = np.asarray(c['hinge'], float), np.asarray(c['axis'], float)
            rotation = _rotation(axis, amount * c['travel_rad'])
            point = hinge + rotation @ (point - hinge)
            push = rotation @ push
        return self.world(point + depth * push), self.direction(push)
