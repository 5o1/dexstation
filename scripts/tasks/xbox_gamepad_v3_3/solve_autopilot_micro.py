#!/usr/bin/env python3
"""Per-direction wrist adjustments for the stick play poses (numpy only).

A rim push (autopilot_ik.GamepadGeometry.rim_target) toward some directions is out of the thumb's reach from
the stick pose itself. For each of the eight directions this solves the smallest arm change from the play pose
for which the thumb's leading fingertip sample reaches the rim push at rest and at STICK_COVERAGE tilt: first
with a strong pull back to the play pose (no change when the thumb alone suffices), then a weak one. The result
is stored per stick mode as hand poses in the gamepad frame, so runtime arm IK places the wrist in the scene.

    GAMEPAD_VLA_WORKSPACE=... python scripts/tasks/xbox_gamepad_v3_3/solve_autopilot_micro.py SCENE_JSON PLAY_POSES_JSON
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_ik import ArmModel, Chain, GamepadGeometry, HandModel  # noqa: E402
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_plan import ASSIGNMENT, DIRECTIONS, RIM_PUSH_M, RIM_REACH_MARGIN_M, STICK_COVERAGE  # noqa: E402
from dexstation.model.spec import load_spec  # noqa: E402


def adjust(arm, hand, finger, q_play, finger_q, goals, regularization, iterations=300):
    """Arm q (and per-goal finger q) placing the leading fingertip sample on every goal; error (m)."""
    chain, surface = hand.fingers[finger], hand.surface[finger]
    size = len(chain.active)
    x = np.concatenate([q_play] + [finger_q] * len(goals))
    lower = np.concatenate([arm.lower] + [chain.lower] * len(goals))
    upper = np.concatenate([arm.upper] + [chain.upper] * len(goals))
    for _ in range(iterations + 1):
        tip, arm_motion = arm.chain.frames(x[:7], arm.base)
        palm = tip @ arm.mount
        rows, residual = [], []
        for i, (target, push) in enumerate(goals):
            q = x[7 + size * i:7 + size * (i + 1)]
            ftip, _ = chain.frames(q, palm)
            local = surface[int(np.argmax(surface @ (ftip[:3, :3].T @ push)))]
            point, _, motion = chain.point(q, palm, local)
            J = np.zeros((3, len(x)))
            J[:, :7] = Chain.jacobian(arm_motion, point, None, 7)[0]
            J[:, 7 + size * i:7 + size * (i + 1)] = Chain.jacobian(motion, point, None, size)[0]
            rows.append(J)
            residual.append(point - target)
        error = max(np.linalg.norm(r) for r in residual)
        rows.append(np.hstack([regularization * np.eye(7), np.zeros((7, len(x) - 7))]))
        residual.append(regularization * (x[:7] - q_play))
        J, r = np.vstack(rows), np.concatenate(residual)
        dx = -J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(len(r)), r)
        step = np.abs(dx).max()
        if step > .05:
            dx *= .05 / step
        x = np.clip(x + dx, lower, upper)
    return x[:7], error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene')
    parser.add_argument('poses')
    args = parser.parse_args()
    spec = load_spec(args.scene)
    pad = GamepadGeometry(spec)
    doc = json.loads(Path(args.poses).read_text())
    for side, value in doc['sides'].items():
        arm, hand = ArmModel(spec, side), HandModel(spec, side)
        for mode in value['modes']:
            sticks = [c for c in mode['controls'] if pad.controls[c]['kind'] == 'stick']
            if not sticks:
                continue
            stick = sticks[0]
            finger = ASSIGNMENT[side][stick]
            q_play = arm.solve(pad.pose @ np.asarray(mode['hand_in_gamepad']), mode['arm_q']).q
            finger_q = np.asarray(mode['approach']['away_fingers'][finger], float)
            micro = {}
            for name, direction in DIRECTIONS.items():
                goals = [pad.rim_target(stick, direction, amount, RIM_PUSH_M + RIM_REACH_MARGIN_M) for amount in (0., STICK_COVERAGE)]
                q, error = adjust(arm, hand, finger, q_play, finger_q, goals, .02)
                if error > 5e-4:
                    q, error = adjust(arm, hand, finger, q_play, finger_q, goals, .002)
                if error > 5e-4:
                    q, error = adjust(arm, hand, finger, q_play, finger_q, goals, .0005, iterations=600)
                change = float(np.degrees(np.abs(q - q_play).max()))
                print(side, mode['name'], name, 'reach error %.2f mm' % (error * 1e3), 'arm change %.2f deg' % change, flush=True)
                if change >= .5:    # smaller changes fall inside the station's 0.5 deg command deadband
                    micro[name] = dict(hand_in_gamepad=(np.linalg.inv(pad.pose) @ arm.hand_pose(q)).tolist(),
                                       arm_q=q.tolist(), reach_error_mm=round(error * 1e3, 3), arm_change_deg=round(change, 2))
            mode['micro'] = micro
    Path(args.poses).write_text(json.dumps(doc, indent=1) + '\n')


if __name__ == '__main__':
    main()
