#!/usr/bin/env python3
"""Offline search for the IK autopilot's per-hand play pose (numpy only).

For each hand, find one arm configuration from which the assigned fingers reach every
assigned control state by finger motion alone (feasibility study 2026-11-11, option a'':
thumb on the face cluster, index on the bumper, middle on the trigger). The arm is solved
for three simultaneous hover targets from random seeds; each candidate is then scored by
finger-only coverage of all control states, joint margins and palm clearance above the
gamepad. The result stores the hand pose in the gamepad frame (so runtime arm IK places the
wrist in any scene that keeps the gamepad pose), the arm seed, and per-state finger seeds.

    GAMEPAD_VLA_WORKSPACE=... python scripts/tasks/xbox_gamepad_v3_3/solve_autopilot_poses.py SCENE_JSON OUTPUT_JSON
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_ik import ArmModel, Chain, GamepadGeometry, HandModel  # noqa: E402
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_plan import ASSIGNMENT, control_states, hover_depth  # noqa: E402
from dexstation.model.spec import load_spec  # noqa: E402
from autopilot_clearance import ExactPad, HandPoints, PadBody  # noqa: E402

THUMB_HOVER_M = .012


def cluster_target(pad, side):
    """Midway between the D-pad/face-button group and the stick, above both."""
    thumb = [c for c, f in ASSIGNMENT[side].items() if f == 'thumb']
    group = np.array([pad.controls[c]['point'] for c in thumb if pad.controls[c]['kind'] in ('dpad', 'button')
                      and c not in ('l3', 'r3')])
    stick = [pad.controls[c]['point'] for c in thumb if pad.controls[c]['kind'] == 'stick']
    points = [p for p in ([group.mean(0)] if len(group) else []) + stick]
    centre = np.mean(points, axis=0)
    centre[2] = max(p[2] for p in (list(group) + stick))
    return pad.world(centre + THUMB_HOVER_M * np.array([0, 0, 1.])), pad.direction([0, 0, -1.])


def finger_targets(pad, side):
    targets = {'thumb': cluster_target(pad, side)} if 'thumb' in ASSIGNMENT[side].values() else {}
    for control, finger in ASSIGNMENT[side].items():
        if finger != 'thumb':
            targets[finger] = pad.target(control, depth=hover_depth(finger))
    return targets


def state_goal(pad, state):
    if state.get('rim'):
        return pad.rim_target(state['control'], state['direction'], state['amount'], state['depth'])
    return pad.target(state['control'], depth=state['depth'], direction=state.get('direction'), amount=state.get('amount', 0.))


def finger_collision(hand, finger, q, palm, pad, points, body, target_link, tolerance, exact=None):
    """(link, gamepad link, clearance mm) of the worst finger-link penetration, or None if clear."""
    inverse = np.linalg.inv(pad.pose)
    worst = None
    for link, pose in hand.fingers[finger].link_frames(q, palm).items():
        local_points = points.links.get(link)
        if local_points is None:
            continue
        local = (local_points @ pose[:3, :3].T + pose[:3, 3]) @ inverse[:3, :3].T + inverse[:3, 3]
        clearance, owner = body.clearance(local)
        if exact is not None and (clearance < tolerance).any():
            suspect = clearance < tolerance
            clearance, owner = clearance.copy(), owner.copy()
            clearance[suspect], owner[suspect] = exact.clearance(local[suspect])
        sign = -1. if hand.side == 'left' else 1.
        side_distance = (sign * local[:, 0] - MIDLINE_M).min()
        if side_distance < 0 and (worst is None or side_distance < worst[2] / 1e3):
            worst = (link, 'midline', round(float(side_distance) * 1e3, 2))
        for value, other in zip(clearance, owner):
            # The pressing fingertip may meet its own control and rest on the static shell around it (the
            # shell only limits depth); touching any other control or link would actuate or wedge it.
            if link.endswith('distal') and other == target_link:
                continue
            allowed = (SHELL_CONTACT_M if link.endswith('distal') else max(SHELL_REST_M, -tolerance)) \
                if other == 'controller_body' else -tolerance
            if value < -allowed and (worst is None or value < worst[2] / 1e3):
                worst = (link, other, round(float(value) * 1e3, 2))
    return worst


def solve_multi(arm, hand, x0, targets, iterations=400, normal_weight=.02, cone_deg=30.):
    fingers = list(targets)
    sizes = [len(hand.fingers[f].active) for f in fingers]
    lower = np.concatenate([arm.lower] + [hand.fingers[f].lower for f in fingers])
    upper = np.concatenate([arm.upper] + [hand.fingers[f].upper for f in fingers])
    x = np.clip(np.array(x0, float), lower, upper)
    cosine = math.cos(math.radians(cone_deg))
    for iteration in range(iterations):
        tip, arm_motion = arm.chain.frames(x[:7], arm.base)
        palm = tip @ arm.mount
        rows, residual, offset = [], [], 7
        for finger, size in zip(fingers, sizes):
            local, normal_local = hand.pads[finger]
            point, ftip, motion = hand.fingers[finger].point(x[offset:offset + size], palm, local)
            normal = ftip[:3, :3] @ normal_local
            target, push = targets[finger]
            J = np.zeros((6, len(x)))
            jp, jn = Chain.jacobian(arm_motion, point, normal, 7)
            J[:3, :7], J[3:, :7] = jp, jn
            jp, jn = Chain.jacobian(motion, point, normal, size)
            J[:3, offset:offset + size], J[3:, offset:offset + size] = jp, jn
            active = normal @ push < cosine
            rows.append(J if active else J[:3])
            residual.append(np.concatenate([point - target, normal_weight * (normal - push)]) if active else point - target)
            if active:
                rows[-1] = J.copy()
                rows[-1][3:] *= normal_weight
            offset += size
        J, r = np.vstack(rows), np.concatenate(residual)
        dx = -J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(r)), r)
        step = np.abs(dx).max()
        if step > .15:
            dx *= .15 / step
        x = np.clip(x + dx, lower, upper)
    errors, offset = {}, 7
    palm = arm.hand_pose(x[:7])
    for finger, size in zip(fingers, sizes):
        point, normal = hand.pad(finger, x[offset:offset + size], palm)
        errors[finger] = (float(np.linalg.norm(point - targets[finger][0])),
                          math.degrees(math.acos(np.clip(normal @ targets[finger][1], -1, 1))))
        offset += size
    return x, errors, fingers, sizes


def coverage(hand, pad, side, palm, seeds, points=None, body=None, tolerance=-.0005, exact=None):
    """Finger-only reach of every control state from one palm pose.

    With ``points``/``body`` a state also needs the operating finger clear of the gamepad (convex pieces,
    ``tolerance``), except its distal link against the control it operates.
    """
    result, worst = {}, 0.
    for name, state in control_states(side).items():
        finger = state['finger']
        target, push = state_goal(pad, state)
        lead = finger == 'thumb'          # as at runtime (autopilot_source.LEADING_CONTACT)
        ik = hand.solve(finger, seeds[finger], palm, target, push, iterations=80, tolerance_m=5e-5, normal_cone_deg=40.)
        if lead:
            ik = hand.solve(finger, ik.q, palm, target, push, iterations=80, tolerance_m=5e-5, normal_cone_deg=90.,
                            leading=True)
        lower, upper = hand.fingers[finger].lower, hand.fingers[finger].upper
        margin = float(np.degrees(np.minimum(ik.q - lower, upper - ik.q).min()))
        ok = ik.error_m < 5e-4 and (lead or ik.angle_deg < 50.)
        hit = None
        if points is not None and ok:
            hit = finger_collision(hand, finger, ik.q, palm, pad, points, body, pad.controls[state['control']]['link'],
                                   tolerance, exact)
            ok = hit is None
        result[name] = dict(finger=finger, q=ik.q.tolist(), error_mm=ik.error_m * 1e3, angle_deg=ik.angle_deg,
                            margin_deg=margin, ok=ok, collision=hit)
        worst = max(worst, ik.error_m)
    return result, worst


IDLE = {'left': {'thumb': 'ls', 'index': 'lt', 'middle': 'lb'}, 'right': {'thumb': 'rs', 'index': 'rt', 'middle': 'rb'}}
GRIP = {'ring': np.zeros(3), 'pinky': np.zeros(3)}


def collision_rows(arm_motion, palm, hand, finger_q, columns, size, points, body, pad, margin, weight=3.):
    """One row per hand link whose worst point is closer than ``margin`` to the gamepad (convex pieces)."""
    inverse = np.linalg.inv(pad.pose)
    rows, residual, worst = [], [], {}
    frames = {'hand_base_link': (palm, None, 0)}
    for finger, chain in hand.fingers.items():
        q = finger_q[finger]
        _, motion = chain.frames(q, palm)
        for link, pose in chain.link_frames(q, palm).items():
            frames[link] = (pose, (finger, motion[:chain.moving_through[link]]), len(q))
    for link, (pose, finger_motion, count) in frames.items():
        local_points = points.links.get(link)
        if local_points is None:
            continue
        world = local_points @ pose[:3, :3].T + pose[:3, 3]
        local = world @ inverse[:3, :3].T + inverse[:3, 3]
        near = np.all((local > body.lo - margin - .01) & (local < body.hi + margin + .01), axis=1)
        if not near.any():
            continue
        distance, normals, owners = body.nearest_plane(local[near])
        # Resting on the fixed shell is allowed up to SHELL_REST_M; controls keep the full margin.
        allowance = np.array([SHELL_REST_M if o == 'controller_body' else 0. for o in owners])
        k = int((distance + allowance).argmin())
        worst[link] = (float(distance[k] + allowance[k]), owners[k])
        if distance[k] + allowance[k] >= margin:
            continue
        p = world[near][k]
        normal = pad.rotation @ normals[k]
        row = np.zeros(size)
        row[:7] = normal @ Chain.jacobian(arm_motion, p, None, 7)[0]
        if finger_motion is not None:
            finger, motion = finger_motion
            row[columns[finger]:columns[finger] + count] = normal @ Chain.jacobian(motion, p, None, count)[0]
        rows.append(-weight * row)
        residual.append(weight * (margin - distance[k] - allowance[k]))
    # Each hand stays on its own side of the gamepad's midline (x = 0), so both can play at once.
    sign = -1. if hand.side == 'left' else 1.
    for link, (pose, finger_motion, count) in frames.items():
        local_points = points.links.get(link)
        if local_points is None:
            continue
        world = local_points @ pose[:3, :3].T + pose[:3, 3]
        side_distance = sign * (world @ inverse[:3, :3].T + inverse[:3, 3])[:, 0] - MIDLINE_M
        k = int(side_distance.argmin())
        worst[link] = min(worst.get(link, (1., None)), (float(side_distance[k]), 'midline'))
        if side_distance[k] >= 0:
            continue
        normal = pad.rotation @ np.array([sign, 0., 0.])
        row = np.zeros(size)
        row[:7] = normal @ Chain.jacobian(arm_motion, world[k], None, 7)[0]
        if finger_motion is not None:
            finger, motion = finger_motion
            row[columns[finger]:columns[finger] + count] = normal @ Chain.jacobian(motion, world[k], None, count)[0]
        rows.append(-weight * row)
        residual.append(weight * -side_distance[k])
    return rows, residual, worst


def solve_shared(arm, hand, pad, side, q_arm, seeds, iterations=150, normal_weight=.02, cone_deg=35.,
                 points=None, body=None, margin=.003, grip=None, max_step=.05):
    """One arm configuration for every control state; each state keeps its own finger joints.

    With ``points``/``body`` the hand at rest (idle hovers plus ring/pinky) is also pushed clear of the
    gamepad by ``margin``; ring and pinky joints are then variables too. Returns (arm q, ring/pinky q, worst
    clearance per link).
    """
    states = control_states(side)
    names = list(states)
    sizes = [len(hand.fingers[states[n]['finger']].active) for n in names]
    # Fingers that operate nothing in this group are free variables, constrained only by clearance.
    free = [f for f in hand.fingers if f not in IDLE[side]]
    grip = {f: np.zeros(len(hand.fingers[f].active)) for f in free} | {f: np.array(v, float) for f, v in (grip or GRIP).items()
                                                                        if f in free}
    x = np.concatenate([q_arm] + [np.asarray(seeds[states[n]['finger']], float) for n in names] + [grip[f] for f in free])
    lower = np.concatenate([arm.lower] + [hand.fingers[states[n]['finger']].lower for n in names] +
                           [hand.fingers[f].lower for f in free])
    upper = np.concatenate([arm.upper] + [hand.fingers[states[n]['finger']].upper for n in names] +
                           [hand.fingers[f].upper for f in free])
    offsets = dict(zip(names, np.cumsum([7] + sizes[:-1])))
    columns = {f: offsets[f'{IDLE[side][f]}:hover'] for f in IDLE[side]}
    start = len(x) - sum(len(grip[f]) for f in free)
    for f in free:
        columns[f] = start
        start += len(grip[f])
    goals = {n: state_goal(pad, s) for n, s in states.items()}
    cosine = math.cos(math.radians(cone_deg))
    worst = {}
    for iteration in range(iterations + 1):
        tip, arm_motion = arm.chain.frames(x[:7], arm.base)
        palm = tip @ arm.mount
        rows, residual = [], []
        if points is not None:
            finger_q = {f: x[c:c + len(hand.fingers[f].active)] for f, c in columns.items()}
            crow, cres, worst = collision_rows(arm_motion, palm, hand, finger_q, columns, len(x), points, body, pad, margin)
            rows += [r[None] for r in crow]
            residual += [np.array([v]) for v in cres]
        if iteration == iterations:
            break
        for name, size in zip(names, sizes):
            offset = offsets[name]
            finger = states[name]['finger']
            local, normal_local = hand.pads[finger]
            point, ftip, motion = hand.fingers[finger].point(x[offset:offset + size], palm, local)
            normal = ftip[:3, :3] @ normal_local
            target, push = goals[name]
            J = np.zeros((6, len(x)))
            J[:3, :7], J[3:, :7] = Chain.jacobian(arm_motion, point, normal, 7)
            J[:3, offset:offset + size], J[3:, offset:offset + size] = Chain.jacobian(motion, point, normal, size)
            rows.append(J[:3])
            residual.append(point - target)
            if normal @ push < cosine:
                rows.append(normal_weight * J[3:])
                residual.append(normal_weight * (normal - push))
        J, r = np.vstack(rows), np.concatenate(residual)
        dx = -J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(r)), r)
        step = np.abs(dx).max()
        if step > max_step:
            dx *= max_step / step
        x = np.clip(x + dx, lower, upper)
    return x[:7], {f: x[columns[f]:columns[f] + len(grip[f])] for f in free}, worst


def clearance(pad, palm):
    """Palm origin height above the gamepad face plane, along the gamepad +z."""
    local = pad.rotation.T @ (palm[:3, 3] - pad.pose[:3, 3])
    return float(local[2])


AWAY_M = {'thumb': .025, 'index': .015, 'middle': .015}
SHELL_CONTACT_M = .0025
SHELL_REST_M = 0.                      # allowed penetration (convex pieces) of resting hand links into the fixed shell; --contact-mm
MIDLINE_M = .003                       # each hand keeps this far from the gamepad's x = 0 plane
RETREATS = ((0, 0, 1), (0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1), (0, 1, 0), (0, 2, 1), (0, 1, 2))


def away_fingers(hand, pad, side, palm, seeds, grip=None):
    """Finger joints that hold each operating finger off its idle control while the arm moves."""
    result = {f: np.asarray((grip or {}).get(f, np.zeros(len(hand.fingers[f].active))), float)
              for f in hand.fingers if f not in IDLE[side]}
    for finger, control in IDLE[side].items():
        point, push = pad.target(control, depth=-AWAY_M.get(finger, .015))
        result[finger] = hand.solve(finger, np.asarray(seeds[finger], float), palm, point, push, iterations=200).q
    return result


def hand_clear(points, body, hand, finger_q, palm, pad):
    """Smallest clearance to the gamepad, or to the midline when the hand is near the gamepad."""
    inverse = np.linalg.inv(pad.pose)
    sign = -1. if hand.side == 'left' else 1.
    worst = 1.
    for link, world in points.world(hand, finger_q, palm).items():
        local = world @ inverse[:3, :3].T + inverse[:3, 3]
        value, owner = body.clearance(local)
        worst = min(worst, float((value + SHELL_REST_M * (owner == 'controller_body')).min()))
        near = np.linalg.norm(local, axis=1) < .25
        if near.any():
            worst = min(worst, float((sign * local[near, 0] - MIDLINE_M).min()))
    return worst


def approach_path(arm, hand, pad, side, play_q, seeds, points, body, lift=.12, steps=12, home_steps=30, grip=None):
    """Retreat direction (gamepad frame) whose straight path from the play pose, and the joint path from home
    to its far end, keep the hand (fingers held away) clear of the gamepad. Returns (direction, report) or None."""
    palm = arm.hand_pose(play_q)
    fingers = away_fingers(hand, pad, side, palm, seeds, grip)
    results = []
    for direction in RETREATS:
        d = np.asarray(direction, float)
        d /= np.linalg.norm(d)
        q, worst, ok = np.asarray(play_q, float), hand_clear(points, body, hand, fingers, palm, pad), True
        for k in range(1, steps + 1):
            pose = palm.copy()
            pose[:3, 3] += pad.rotation @ d * lift * k / steps
            ik = arm.solve(pose, q, iterations=200)
            if ik.error_m > 1e-3 or ik.angle_deg > .5:
                ok = False
                break
            q = ik.q
            worst = min(worst, hand_clear(points, body, hand, fingers, pose, pad))
        if not ok:
            continue
        home_worst = min(hand_clear(points, body, hand, fingers, arm.hand_pose(arm.home + (q - arm.home) * s), pad)
                         for s in np.linspace(0, 1, home_steps))
        results.append((min(worst, home_worst), direction, worst, home_worst, q))
    if not results:
        return None
    best = max(results, key=lambda r: r[0])
    return dict(direction=list(best[1]), lift_m=lift, path_clearance_mm=round(best[2] * 1e3, 2),
                home_clearance_mm=round(best[3] * 1e3, 2), staging_q=best[4].tolist(),
                away_fingers={f: v.tolist() for f, v in fingers.items()})


_WORKER = {}
PRESETS = {'thumb': ((0., 0., 0., 0.), (1.2, 1.4, .6, .9), (.6, .2, .2, .2), (1.4, 1.57, .8, 1.2)),
           'other': ((0., 0., 0.), (0., .6, .8), (0., 1.2, 1.6), (0., 1.22, 1.74))}


def free_finger_presets(hand, pad, side, palm, points, body):
    """Per free finger, the preset posture with the most clearance to the gamepad and the midline."""
    inverse = np.linalg.inv(pad.pose)
    sign = -1. if side == 'left' else 1.
    chosen = {}
    for finger in hand.fingers:
        if finger in IDLE[side]:
            continue
        best = None
        for preset in PRESETS['thumb' if finger == 'thumb' else 'other']:
            q = np.asarray(preset, float)
            worst = 1.
            for link, pose in hand.fingers[finger].link_frames(q, palm).items():
                if link not in points.links:
                    continue
                local = (points.links[link] @ pose[:3, :3].T + pose[:3, 3]) @ inverse[:3, :3].T + inverse[:3, 3]
                worst = min(worst, float(body.clearance(local)[0].min()), float((sign * local[:, 0] - MIDLINE_M).min()))
            if best is None or worst > best[0]:
                best = (worst, q)
        chosen[finger] = best[1]
    return chosen


def restrict(side, controls):
    """Keep only ``controls`` (one play pose per control group); idle fingers stay on their controls."""
    kept = {c: ASSIGNMENT[side][c] for c in controls}
    ASSIGNMENT[side].clear()
    ASSIGNMENT[side].update(kept)
    for finger in list(IDLE[side]):
        mine = [c for c, f in kept.items() if f == finger]
        if mine:
            IDLE[side][finger] = IDLE[side][finger] if IDLE[side][finger] in mine else mine[0]
        else:
            del IDLE[side][finger]


def _init_worker(scene, resources, side, assignment=None, controls=None, contact=0.):
    global SHELL_REST_M
    SHELL_REST_M = contact
    if assignment:
        ASSIGNMENT[side].update(assignment)
        IDLE[side].update({finger: control for control, finger in assignment.items()})
    if controls:
        restrict(side, controls)
    spec = load_spec(scene)
    _WORKER.update(spec=spec, pad=GamepadGeometry(spec), arm=ArmModel(spec, side), hand=HandModel(spec, side), side=side,
                   points=HandPoints(getattr(spec, side + '_hand').urdf, size=.007), body=PadBody(resources),
                   exact=ExactPad(resources))


def palm_above_seed(arm, pad, side, rng, family='above'):
    """Arm configuration for a random palm pose clear of the gamepad: above the face with the fingers
    toward it ('above'), or beyond the top edge with the fingers reaching back over it ('edge')."""
    sign = -1 if side == 'left' else 1
    if family in ('side', 'below', 'around'):
        # Palm beside the gamepad, underneath it, or anywhere on a shell around it, in any orientation.
        if family == 'side':
            position = np.array([sign * rng.uniform(95, 190), rng.uniform(-60, 80), rng.uniform(-60, 70)]) / 1e3
        elif family == 'below':
            position = np.array([sign * rng.uniform(0, 110), rng.uniform(-90, 90), rng.uniform(-170, -60)]) / 1e3
        else:
            u = rng.normal(size=3)
            position = u / np.linalg.norm(u) * rng.uniform(.11, .22)
            position[0] = sign * abs(position[0])
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, position
        world = pad.pose @ pose
        best = None
        for start in (arm.home, rng.uniform(arm.lower, arm.upper)):
            ik = arm.solve(world, start, iterations=150)
            if best is None or ik.error_m < best.error_m:
                best = ik
        return best.q
    if family == 'shoulder':
        # Palm beyond the top edge, fingers pointing back at the bumper/trigger faces, spread across the
        # gamepad's thickness (index low on the trigger, middle high on the bumper).
        position = np.array([sign * rng.uniform(15, 80), rng.uniform(170, 260), rng.uniform(-35, 25)]) / 1e3
        fingers = np.array([rng.normal(0, .15), -1., rng.normal(0, .25)])
        fingers /= np.linalg.norm(fingers)
        spread = np.array([rng.normal(0, .2), 0., 1.]) * rng.choice([-1, 1])
        spread -= fingers * (spread @ fingers)
        spread /= np.linalg.norm(spread)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([np.cross(spread, fingers), spread, fingers], axis=1)
        pose[:3, 3] = position
        world = pad.pose @ pose
        best = None
        for start in (arm.home, rng.uniform(arm.lower, arm.upper)):
            ik = arm.solve(world, start, iterations=150)
            if best is None or ik.error_m < best.error_m:
                best = ik
        return best.q
    if family == 'edge':
        position = np.array([sign * rng.uniform(15, 90), rng.uniform(70, 160), rng.uniform(-50, 70)]) / 1e3
        finger_dir = np.array([-sign * rng.uniform(0, .5), -rng.uniform(.3, 1.), rng.uniform(-.6, .6)])
    else:
        position = np.array([sign * rng.uniform(20, 120), rng.uniform(-40, 140), rng.uniform(110, 220)]) / 1e3
        finger_dir = np.array([-sign * rng.uniform(0, .6), rng.uniform(-.8, .3), -rng.uniform(.4, 1.)])
    finger_dir /= np.linalg.norm(finger_dir)
    palmar = np.cross(finger_dir, rng.normal(size=3))
    palmar /= np.linalg.norm(palmar)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([palmar, np.cross(finger_dir, palmar), finger_dir], axis=1)
    pose[:3, 3] = position
    world = pad.pose @ pose
    best = None
    for start in (arm.home, rng.uniform(arm.lower, arm.upper)):
        ik = arm.solve(world, start, iterations=150)
        if best is None or ik.error_m < best.error_m:
            best = ik
    return best.q


def candidate(job):
    """Search from one random arm seed; returns a scored candidate or None."""
    k, seed_arm = job
    w = _WORKER
    arm, hand, pad, side = w['arm'], w['hand'], w['pad'], w['side']
    if isinstance(seed_arm, str):
        seed_arm = palm_above_seed(arm, pad, side, np.random.default_rng(10_000 + k), seed_arm)
    targets = finger_targets(pad, side)
    seed = np.concatenate([seed_arm] + [(hand.fingers[f].lower + hand.fingers[f].upper) / 2 for f in targets])
    x, errors, fingers, sizes = solve_multi(arm, hand, seed, targets)
    if max(e for e, _ in errors.values()) > 1e-3:
        return None
    seeds, offset = {}, 7
    for finger, size in zip(fingers, sizes):
        seeds[finger] = x[offset:offset + size]
        offset += size
    q, _, _ = solve_shared(arm, hand, pad, side, x[:7], seeds, iterations=120)
    grip = free_finger_presets(hand, pad, side, arm.hand_pose(q), w['points'], w['body'])
    q, grip, rest = solve_shared(arm, hand, pad, side, q, seeds, iterations=150, points=w['points'], body=w['body'], grip=grip)
    palm = arm.hand_pose(q)
    collision = min(v for v, _ in rest.values()) if rest else 1.
    states, _ = coverage(hand, pad, side, palm, seeds, w['points'], w['body'], exact=w['exact'])
    covered = sum(s['ok'] for s in states.values())
    arm_margin = float(np.degrees(np.minimum(q - arm.lower, arm.upper - q).min()))
    finger_margin = min(s['margin_deg'] for s in states.values())
    travel = float(np.degrees(np.abs(q - arm.home).max()))
    return dict(k=k, score=(covered, collision >= 0., arm_margin >= 10., -travel, collision, arm_margin),
                arm_q=q.tolist(), finger_seed={f: v.tolist() for f, v in seeds.items()},
                hand_in_gamepad=(np.linalg.inv(pad.pose) @ palm).tolist(), states=states,
                arm_margin_deg=arm_margin, finger_margin_deg=finger_margin, max_travel_from_home_deg=travel,
                grip={f: v.tolist() for f, v in grip.items()},
                rest_clearance_mm={link: round(v * 1e3, 2) for link, (v, _) in rest.items()})


def main():
    from multiprocessing import get_context
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene')
    parser.add_argument('output')
    parser.add_argument('--seeds', type=int, default=200)
    parser.add_argument('--sides', default='left,right')
    parser.add_argument('--rng', type=int, default=7)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--families', default='above,edge,home,any', help='seed families: above, edge, home, any')
    parser.add_argument('--keep', type=int, default=4, help='candidates kept for physical verification')
    parser.add_argument('--contact-mm', type=float, default=0., help='allowed resting penetration into the fixed shell')
    parser.add_argument('--controls', help='comma-separated control group for one play pose (default: all of the side)')
    parser.add_argument('--assignment', help='JSON per side, e.g. {"left": {"lb": "index", "lt": "middle"}}')
    parser.add_argument('--resources', required=True, help='xbox_gamepad_physics robotwin/full_controls resource directory')
    args = parser.parse_args()
    spec = load_spec(args.scene)
    output = dict(schema='dexhand-autopilot-play-poses-v1', scene=spec.name, gamepad_root_pose=list(spec.gamepad.root_pose),
                  assignment=ASSIGNMENT, rule='scripts/tasks/xbox_gamepad_v3_3/solve_autopilot_poses.py --seeds %d --rng %d' % (args.seeds, args.rng),
                  sides={})
    if Path(args.output).exists():
        previous = json.loads(Path(args.output).read_text())
        if previous.get('gamepad_root_pose') == output['gamepad_root_pose']:
            output['sides'] = previous['sides']
    for side in args.sides.split(','):
        arm = ArmModel(spec, side)
        rng = np.random.default_rng(args.rng)
        # Seeds: palm above the face (collision-free family), near home, anywhere.
        families = args.families.split(',')
        jobs = [(k, families[k % len(families)]) for k in range(args.seeds)]
        for k, (index, family) in enumerate(jobs):
            if family == 'home':
                jobs[k] = (index, np.clip(arm.home + rng.uniform(-1.75, 1.75, 7), arm.lower, arm.upper))
            elif family == 'any':
                jobs[k] = (index, rng.uniform(arm.lower, arm.upper))
        started, found = time.time(), []
        assignment = json.loads(args.assignment).get(side) if args.assignment else None
        if assignment:
            ASSIGNMENT[side].update(assignment)
            output['assignment'] = ASSIGNMENT
        controls = args.controls.split(',') if args.controls else None
        if controls:
            restrict(side, controls)
            output['assignment'] = ASSIGNMENT
        global SHELL_REST_M
        SHELL_REST_M = args.contact_mm / 1e3
        with get_context('spawn').Pool(args.workers, _init_worker,
                                       (args.scene, args.resources, side, assignment, controls, SHELL_REST_M)) as pool:
            for result in pool.imap_unordered(candidate, jobs):
                if result is None:
                    continue
                found.append(result)
                print(side, result['k'], 'covered', result['score'][0], 'arm margin %.1f' % result['arm_margin_deg'],
                      'travel %.0f deg' % result['max_travel_from_home_deg'], 'clearance %.1f mm' % (result['score'][4] * 1e3),
                      'q', np.degrees(result['arm_q']).round(0).tolist(), flush=True)
        found.sort(key=lambda r: r['score'], reverse=True)
        _init_worker(args.scene, args.resources, side, assignment, controls, SHELL_REST_M)
        w = _WORKER
        fine = HandPoints(getattr(w['spec'], side + '_hand').urdf, size=.003)
        candidates = []
        for result in found[:args.keep * 2]:
            # Re-check the pose at rest with dense hand samples (the search uses 7 mm samples).
            palm = w['arm'].hand_pose(np.array(result['arm_q']))
            away = away_fingers(w['hand'], w['pad'], side, palm, result['finger_seed'], result['grip'])
            dense = hand_clear(fine, w['body'], w['hand'], away, palm, w['pad'])
            approach = approach_path(w['arm'], w['hand'], w['pad'], side, np.array(result['arm_q']), result['finger_seed'],
                                     w['points'], w['body'], grip=result['grip'])
            print(side, result['k'], 'dense rest clearance %.1f mm' % (dense * 1e3), 'approach',
                  approach and {k: v for k, v in approach.items() if 'q' not in k and 'fingers' not in k}, flush=True)
            if approach is not None and approach['path_clearance_mm'] > 0 and approach['home_clearance_mm'] > 5 and dense > -.0005:
                candidates.append(dict(result, approach=approach, dense_rest_clearance_mm=round(dense * 1e3, 2)))
            if len(candidates) == args.keep:
                break
        if not found:
            print(side, 'no seed converged on the hover targets', flush=True)
            output['sides'][side] = None
            Path(args.output).write_text(json.dumps(output, indent=1) + '\n')
            continue
        best = candidates[0] if candidates else dict(found[0], approach=None)
        best = dict(best, candidates=candidates[1:], searched=dict(seeds=args.seeds, converged=len(found), families=args.families,
                                                                     contact_mm=args.contact_mm))
        print(side, 'best', best['score'], 'from', len(found), 'in %.0f s' % (time.time() - started), flush=True)
        output['sides'][side] = best
        Path(args.output).write_text(json.dumps(output, indent=1) + '\n')


if __name__ == '__main__':
    main()
