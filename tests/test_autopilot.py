import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dexstation.task_plugins.xbox_gamepad_v3_3 import autopilot_eval as ev
from dexstation.task_plugins.xbox_gamepad_v3_3 import autopilot_plan as plan
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_ik import Chain, GamepadGeometry, load_asset, pose_error, solve_point
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_tasks import Step, load_tasks, task_steps, validate
from dexstation.task_plugins.xbox_gamepad_v3_3.signals import SDL3_NAMES

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'src/dexstation/task_plugins/xbox_gamepad_v3_3/data/autopilot/game_combos_v1.json'
CONTROLS = load_asset('xbox_one_s_controls.json')['controls']
AXES = load_asset('xbox_one_s_controls.json')['signal_axes']

ARM = """<robot name="arm">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="tip"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-3" upper="3"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.2" rpy="0.3 0 0"/><axis xyz="0 1 0"/><limit lower="-3" upper="3"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="tip"/>
    <origin xyz="0 0 0.1"/><axis xyz="1 0 0"/><limit lower="-3" upper="3"/>
    <mimic joint="j3" multiplier="0.5" offset="0.1"/></joint>
</robot>"""


@pytest.fixture
def chain(tmp_path):
    path = tmp_path / 'arm.urdf'
    path.write_text(ARM)
    return Chain(path, 'tip')


def test_point_jacobian_matches_finite_differences(chain):
    base = np.eye(4)
    base[:3, 3] = [.1, -.2, .3]
    q, local, normal = np.array([.3, -.4, .7]), np.array([.01, .02, .03]), np.array([0., 0., 1.])
    point, tip, motion = chain.point(q, base, local)
    jp, jn = Chain.jacobian(motion, point, tip[:3, :3] @ normal, 3)
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = 1e-6
        p2, t2, _ = chain.point(q + dq, base, local)
        np.testing.assert_allclose((p2 - point) / 1e-6, jp[:, i], atol=1e-5)
        np.testing.assert_allclose((t2[:3, :3] @ normal - tip[:3, :3] @ normal) / 1e-6, jn[:, i], atol=1e-5)


def test_point_ik_reaches_a_reachable_target(chain):
    goal_q = np.array([.5, .3, -.6])
    target, tip, _ = chain.point(goal_q, np.eye(4), np.zeros(3))
    result = solve_point(chain, np.zeros(3), np.eye(4), np.zeros(3), None, target, None, iterations=100)
    assert result.error_m < 1e-4


def test_pose_error_is_zero_for_identical_poses_and_small_rotation():
    pose = np.eye(4)
    np.testing.assert_allclose(pose_error(pose, pose), 0)
    turned = np.eye(4)
    turned[:2, :2] = [[math.cos(.1), -math.sin(.1)], [math.sin(.1), math.cos(.1)]]
    np.testing.assert_allclose(pose_error(pose, turned)[3:], [0, 0, .1], atol=1e-9)


class Pad:
    gamepad = type('G', (), dict(root_pose=(0., 0., 0., 1., 0., 0., 0.)))


def test_control_targets_follow_press_tilt_and_pull():
    pad = GamepadGeometry(Pad)
    a, push = pad.target('a', depth=.001)
    np.testing.assert_allclose(a, np.add(CONTROLS['a']['point'], [0, 0, -.001]), atol=1e-7)
    np.testing.assert_allclose(push, [0, 0, -1], atol=1e-4)
    tilted, _ = pad.target('ls', direction=(1., 0.), amount=1.)
    assert tilted[0] > CONTROLS['ls']['point'][0] + .002
    pulled, push = pad.target('rt', amount=1.)
    rest, _ = pad.target('rt')
    assert np.linalg.norm(pulled - rest) > .005 and abs(np.linalg.norm(push) - 1) < 1e-9


def test_game_task_set_is_valid_and_covers_the_requested_operations():
    doc = load_tasks(TASKS)
    steps = [s for i in range(len(doc['tasks'])) for s in task_steps(doc, i, 0.)]
    actions = {(s.action, CONTROLS[s.control]['kind']) for s in steps}
    assert {('press', 'button'), ('hold', 'button'), ('tilt', 'stick'), ('hold', 'trigger'), ('press', 'dpad')} <= actions
    controls = {s.control for s in steps}
    assert {'lb', 'rb', 'l3', 'r3', 'dpad_up', 'dpad_down', 'dpad_left', 'dpad_right', 'lt', 'rt', 'ls', 'rs'} <= controls
    by_task = {}
    for i, task in enumerate(doc['tasks']):
        by_task[task['id']] = task_steps(doc, i, 0.)
    categories = {ev.category(s, by_task[t], CONTROLS) for t, ss in by_task.items() for s in ss}
    assert {'tap', 'hold', 'mash', 'stick_tilt', 'trigger', 'bumper', 'dpad', 'stick_click'} <= categories
    # Both hands at once, and a left-stick tilt while the right thumb presses a face button.
    assert any(a.hand != b.hand and a.start < b.end and b.start < a.end for ss in by_task.values() for a in ss for b in ss)
    assert any(a.control == 'ls' and b.control in 'abxy' and a.start < b.start < a.end
               for ss in by_task.values() for a in ss for b in ss)


@pytest.mark.parametrize('change, message', [
    (lambda s: s.update(finger='index'), 'operated by'),
    (lambda s: s.update(action='tilt'), 'sticks take tilt'),
    (lambda s: s.update(control='warp'), 'unknown control'),
    (lambda s: s.update(t=100.), 'ends after'),
])
def test_invalid_steps_are_rejected(change, message):
    doc = json.loads(TASKS.read_text())
    change(doc['tasks'][0]['steps'][0])
    with pytest.raises(ValueError, match=message):
        validate(doc)


def test_overlapping_steps_of_one_finger_are_rejected_except_a_click_inside_a_tilt():
    doc = json.loads(TASKS.read_text())
    task = doc['tasks'][0]
    base = dict(hand='right', finger='thumb', duration=.3)
    task['steps'] = [dict(base, t=.5, control='a', action='press'), dict(base, t=.6, control='b', action='press')]
    with pytest.raises(ValueError, match='overlapping'):
        validate(doc)
    left = dict(hand='left', finger='thumb')
    task['steps'] = [dict(left, t=.5, control='ls', action='tilt', duration=2., direction='up', amount=.9),
                     dict(left, t=1., control='l3', action='press', duration=.2)]
    validate(doc)


def step(control, action, start, end, **kw):
    side = plan.CONTROL_SIDE[control]
    return Step(0, kw.pop('index', 0), control, side, plan.ASSIGNMENT[side][control], action, start, end, **kw)


def test_press_profile_hovers_presses_and_releases():
    s = step('a', 'press', 1., 1.12)
    hover = plan.finger_target(s.finger, [s], .5, 'a', CONTROLS)
    assert hover['phase'] == 'hover' and hover['depth'] == plan.HOVER_M
    pressed = plan.finger_target(s.finger, [s], 1.08, 'a', CONTROLS)
    assert pressed['phase'] == 'press' and pressed['depth'] == pytest.approx(CONTROLS['a']['travel_m'] + plan.PRESS_EXTRA_M)
    released = plan.finger_target(s.finger, [s], 1.2, 'a', CONTROLS)
    assert released['depth'] == plan.HOVER_M


def test_tilt_lands_first_and_a_nested_click_goes_deeper():
    tilt = step('ls', 'tilt', 1., 3., direction=(0., 1.), amount=.9)
    click = step('l3', 'press', 2., 2.2, index=1)
    landing = plan.finger_target('thumb', [tilt, click], 1. - plan.LAND_S / 2, 'ls', CONTROLS)
    assert landing['phase'] == 'land' and plan.HOVER_M < landing['depth'] < plan.DISH_TILT_M and not landing['rim']
    tilted = plan.finger_target('thumb', [tilt, click], 1.5, 'ls', CONTROLS)
    assert tilted['amount'] == pytest.approx(.9) and tilted['depth'] == plan.DISH_TILT_M and not tilted['rim']
    clicking = plan.finger_target('thumb', [tilt, click], 2.1, 'ls', CONTROLS)
    assert clicking['depth'] == pytest.approx(plan.CLICK_PRESS_M) and clicking['amount'] == pytest.approx(.9)
    assert clicking['step'] is click and not clicking['rim']


def test_side_push_is_horizontal_and_its_contact_turns_with_the_cap():
    pad = GamepadGeometry(Pad)
    c = CONTROLS['ls']
    point, push = pad.rim_target('ls', (0., 1.), 0.)
    assert point[1] == pytest.approx(c['pivot'][1] - c['cap_radius_m']) and c['cap_side_z'][0] < point[2] < c['cap_side_z'][1]
    assert push[1] == pytest.approx(1.) and abs(push[2]) < 1e-9
    tilted, _ = pad.rim_target('ls', (0., 1.), 1.)
    assert tilted[1] > point[1] + .004


def test_trigger_pull_follows_amount():
    s = step('rt', 'hold', 1., 2., amount=.6)
    assert plan.finger_target(s.finger, [s], 1.5, 'rt', CONTROLS)['amount'] == pytest.approx(.6)
    assert plan.finger_target(s.finger, [s], 2.5, 'rt', CONTROLS)['amount'] == 0


def signals_for(times, **channels):
    values = np.zeros((len(times), len(SDL3_NAMES)))
    for name, fn in channels.items():
        values[:, SDL3_NAMES.index(name)] = [fn(t) for t in times]
    return values


def test_scoring_reports_success_latency_and_false_triggers():
    times = np.arange(0, 3, 1 / 60)
    tap = step('a', 'press', 1., 1.12)
    signals = signals_for(times, south=lambda t: 1. if 1.05 <= t < 1.15 else 0.,
                          west=lambda t: 1. if 2. <= t < 2.1 else 0.)
    result = ev.score_step(tap, [tap], times, signals, CONTROLS, AXES)
    assert result['success'] and result['latency_s'] == pytest.approx(times[times >= 1.05][0] - 1.)
    false = ev.false_triggers([tap], times, signals, CONTROLS, AXES, (0., 3.))
    assert [e['control'] for e in false] == ['x']


def test_stick_scoring_uses_the_measured_direction():
    times = np.arange(0, 3, 1 / 60)
    tilt = step('ls', 'tilt', 1., 2., direction=(0., 1.), amount=.8)
    # Pushing up on the gamepad face reads as negative left_y (SDL: +y is down).
    up = signals_for(times, left_y=lambda t: -.8 if 1.1 <= t < 2. else 0.)
    assert ev.score_step(tilt, [tilt], times, up, CONTROLS, AXES)['success']
    right = signals_for(times, left_x=lambda t: .8 if 1.1 <= t < 2. else 0.)
    assert not ev.score_step(tilt, [tilt], times, right, CONTROLS, AXES)['success']


def test_unreleased_hold_and_missed_press_fail():
    times = np.arange(0, 3, 1 / 60)
    hold = step('x', 'hold', 1., 1.5)
    stuck = signals_for(times, west=lambda t: 1. if t >= 1.05 else 0.)
    assert ev.score_step(hold, [hold], times, stuck, CONTROLS, AXES)['reason'] == 'not released in time'
    nothing = signals_for(times)
    assert not ev.score_step(hold, [hold], times, nothing, CONTROLS, AXES)['success']
    summary = ev.summarize([ev.score_step(hold, [hold], times, nothing, CONTROLS, AXES)], [])
    assert summary['categories']['hold']['success_rate'] == 0


def test_leading_contact_puts_the_first_touching_sample_on_target(chain):
    surface = np.array([[0., 0., 0.], [.01, 0., .005], [-.01, .002, .004], [0., .01, .006]])
    push = np.array([0., 0., -1.])
    goal, _ = chain.frames(np.array([.4, .5, -.3]), np.eye(4))
    reached = surface @ goal[:3, :3].T + goal[:3, 3]
    target = reached[np.argmax(reached @ push)]
    result = solve_point(chain, np.array([.3, .4, -.2]), np.eye(4), surface[0], None, target, push,
                         iterations=200, surface=surface)
    tip, _ = chain.frames(result.q, np.eye(4))
    world = surface @ tip[:3, :3].T + tip[:3, 3]
    lowest = world[np.argmax(world @ push)]
    assert result.error_m < 1e-4 and np.linalg.norm(lowest - target) < 1e-4


def test_side_wall_tilt_lands_behind_the_cap(monkeypatch):
    monkeypatch.setattr(plan, 'TILT_SIDE', True)
    tilt = step('ls', 'tilt', 1., 3., direction=(0., 1.), amount=.9)
    landing = plan.finger_target('thumb', [tilt], 1. - plan.LAND_S / 2, 'ls', CONTROLS)
    assert landing['rim'] and plan.RIM_HOVER_M < landing['depth'] < plan.RIM_PUSH_M
    assert plan.finger_target('thumb', [tilt], 1.5, 'ls', CONTROLS)['depth'] == plan.RIM_PUSH_M
