"""Finger assignment and per-control contact parameters for the IK autopilot."""
import math

# Thumbs work the face (D-pad, A/B/X/Y, sticks and clicks); each index finger works its bumper and trigger.
# One play pose per control group and hand (scripts/solve_autopilot_poses.py --controls): in this scene a
# single grip covering a stick and a trigger together embeds the palm 7-9 mm in the grip (2026-11-12).
ASSIGNMENT = {
    'left': {'dpad_up': 'thumb', 'dpad_down': 'thumb', 'dpad_left': 'thumb', 'dpad_right': 'thumb',
             'ls': 'thumb', 'l3': 'thumb', 'lb': 'index', 'lt': 'index'},
    'right': {'a': 'thumb', 'b': 'thumb', 'x': 'thumb', 'y': 'thumb',
              'rs': 'thumb', 'r3': 'thumb', 'rb': 'index', 'rt': 'index'},
}
CONTROL_SIDE = {control: side for side, controls in ASSIGNMENT.items() for control in controls}

# Depths along the press direction, metres (negative = above the surface).
HOVER_M = -.006
REST_M = -.003               # index/middle rest just off the bumper and trigger, as players do
PRESS_EXTRA_M = .0012        # beyond full travel, so the arm's compliance cannot stop short
DPAD_PRESS_M = .0025
CLICK_PRESS_M = .0030        # stick click needs its 7.4 N preload overcome
STICK_DISH_M = .0003         # thumb in the cap dish for a click inside a tilt
RIM_PUSH_M = .0005           # into the cap's side wall while tilting, horizontally (see autopilot_ik.rim_target)
RIM_HOVER_M = -.008          # back off the rim along the push before landing and after lifting
RIM_REACH_MARGIN_M = .002    # stick poses must reach rim pushes this much deeper (the real palm sits a few mm off)
STICK_COVERAGE = .85         # tilt fraction every direction must reach (signal about 0.85)
TRIGGER_EXTRA_M = .0010

DIRECTIONS = {
    'up': (0., 1.), 'down': (0., -1.), 'left': (-1., 0.), 'right': (1., 0.),
    'up_left': (-math.sqrt(.5), math.sqrt(.5)), 'up_right': (math.sqrt(.5), math.sqrt(.5)),
    'down_left': (-math.sqrt(.5), -math.sqrt(.5)), 'down_right': (math.sqrt(.5), -math.sqrt(.5)),
}


def hover_depth(finger):
    return HOVER_M if finger == 'thumb' else REST_M


def press_depth(control, kind, travel_m=0.):
    if control in ('l3', 'r3'):
        return CLICK_PRESS_M
    if kind == 'dpad':
        return DPAD_PRESS_M
    if kind == 'trigger':
        return TRIGGER_EXTRA_M
    return travel_m + PRESS_EXTRA_M


def control_states(side, controls=None):
    """Named coverage states: hover, full press, and for sticks every direction at ``STICK_COVERAGE`` tilt (dish
    drag), or with ``TILT_SIDE`` the side-wall push of every direction at rest and at full coverage tilt."""
    from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_ik import load_asset
    controls = controls or load_asset('xbox_one_s_controls.json')['controls']
    states = {}
    for control, finger in ASSIGNMENT[side].items():
        c = controls[control]
        states[control + ':hover'] = dict(control=control, finger=finger, depth=hover_depth(finger))
        if c['kind'] == 'stick' and not TILT_SIDE:
            for name, direction in DIRECTIONS.items():
                states[f'{control}:{name}'] = dict(control=control, finger=finger, depth=DISH_TILT_M,
                                                   direction=direction, amount=STICK_COVERAGE)
        elif c['kind'] == 'stick':
            for name, direction in DIRECTIONS.items():
                for suffix, amount in (('', STICK_COVERAGE), ('@0', 0.)):
                    states[f'{control}:{name}{suffix}'] = dict(control=control, finger=finger, depth=RIM_PUSH_M + RIM_REACH_MARGIN_M,
                                                               direction=direction, amount=amount, rim=True)
        elif c['kind'] == 'trigger':
            states[control + ':full'] = dict(control=control, finger=finger, depth=TRIGGER_EXTRA_M, amount=1.)
        else:
            states[control + ':press'] = dict(control=control, finger=finger,
                                              depth=press_depth(control, c['kind'], c.get('travel_m', 0.)))
    return states


# Timeline shaping (seconds).
RAMP_S = .03             # hover <-> press depth
TRIGGER_RAMP_S = .06     # trigger pull and release
TILT_RAMP_S = .06        # stick centre <-> requested tilt
LAND_S = .08             # seat the thumb on the cap before a tilt starts
LINGER_S = 0.            # the thumb always lifts and lands again between tilts
TILT_SIDE = False        # tilt by the cap's side wall (True) or by dragging the dish (False; see _tilt_dish)
TILT_FORCE = True        # dish drag with the thumb's push along the cap axis regulated to STICK_FORCE_N (needs
                         # contact sensing from the backend; the depth then only seeds the loop)
STICK_FORCE_N = 1.5      # dish drag normal force: friction (mu 0.5) carries ~0.7 N; click needs 7.4 N
CLICK_FORCE_N = 9.5      # intentional click
DISH_TILT_M = .0003      # dish drag: leading thumb sample this far into the dish bottom along the cap axis
                         # (station sweep 2026-11-12: -0.2 mm loses grip, 0..0.2 mm no better than 0.3 mm)


def smooth(x):
    x = min(1., max(0., x))
    return x * x * (3 - 2 * x)


def _profile(step, t, ramp):
    """0 -> 1 over ``ramp`` from the step start, 1 until its end, back to 0 over ``ramp``."""
    if t < step.start:
        return 0.
    if t < step.end:
        return smooth((t - step.start) / ramp)
    return 1 - smooth((t - step.end) / ramp)


def finger_target(finger, steps, t, idle_control, controls):
    """Contact target of one finger at time ``t`` from its steps (sorted by start).

    Returns ``dict(control, depth, direction, amount, step, phase)``; ``step`` is the step
    being executed (for closed-loop corrections and labels) or None.
    """
    kinds = {name: controls[name]['kind'] for name in controls}

    def tail(step):
        return step.end + (TILT_RAMP_S + RAMP_S if step.action == 'tilt' else
                           TRIGGER_RAMP_S if kinds[step.control] == 'trigger' else RAMP_S)

    clicks = [s for s in steps if s.control in ('l3', 'r3') and s.start <= t < tail(s)]
    if clicks:
        return _press(clicks[0], steps, t, kinds, controls)
    previous = upcoming = None
    for step in steps:
        if step.control in ('l3', 'r3'):
            if step.start > t and upcoming is None:
                upcoming = step
            elif step.start <= t:
                previous = step
            continue
        lead = LAND_S if step.action == 'tilt' else 0.
        if step.start - lead <= t < tail(step):
            if step.action == 'tilt':
                return _tilt(step, steps, t, controls)
            return _press(step, steps, t, kinds, controls)
        if step.start > t:
            upcoming = step if upcoming is None or step.start < upcoming.start else upcoming
            break
        previous = step
    control = (upcoming or previous).control if (upcoming or previous) else idle_control
    if upcoming is not None and upcoming.action == 'tilt' and previous is not None and previous.control == upcoming.control \
            and upcoming.start - LAND_S - t < LINGER_S and previous.action == 'tilt':
        return dict(control=control, depth=STICK_DISH_M, direction=None, amount=0., step=None, phase='seated')
    return dict(control=control, depth=hover_depth(finger), direction=None, amount=0.,
                step=None, phase='hover' if upcoming or previous else 'idle')


def _press(step, steps, t, kinds, controls):
    kind = kinds[step.control]
    if kind == 'trigger':
        p = _profile(step, t, TRIGGER_RAMP_S)
        depth = REST_M + (TRIGGER_EXTRA_M - REST_M) * smooth(3 * p)
        return dict(control=step.control, depth=depth, direction=None, amount=step.amount * p, step=step,
                    phase='press' if t < step.end else 'release')
    p = _profile(step, t, RAMP_S)
    if step.control in ('l3', 'r3'):
        stick = 'ls' if step.control == 'l3' else 'rs'
        tilt = next((s for s in steps if s.control == stick and s.start <= step.start and step.end <= s.end), None)
        base = _tilt(tilt, steps, t, controls) if tilt else dict(control=stick, depth=HOVER_M, direction=None, amount=0.)
        start_depth = STICK_DISH_M if tilt else HOVER_M
        # A click presses down the shaft from the dish of the (tilted) cap, not from its rim.
        return dict(base, depth=start_depth + (CLICK_PRESS_M - start_depth) * p, step=step, rim=False,
                    phase='press' if t < step.end else 'release',
                    force=CLICK_FORCE_N if TILT_FORCE and t < step.end and p > .5 else None)
    hover = hover_depth(step.finger)
    depth = press_depth(step.control, kind, controls[step.control].get('travel_m', 0.))
    return dict(control=step.control, depth=hover + (depth - hover) * p, direction=None, amount=0., step=step,
                phase='press' if t < step.end else 'release')


def _tilt(step, steps, t, controls):
    if not TILT_SIDE:
        return _tilt_dish(step, t)
    if t < step.start:
        return dict(control=step.control, depth=RIM_HOVER_M + (RIM_PUSH_M - RIM_HOVER_M) * smooth((t - step.start + LAND_S) / LAND_S),
                    direction=step.direction, amount=0., step=step, phase='land', rim=True)
    amount = step.amount * _profile(step, t, TILT_RAMP_S)
    if t < step.end + TILT_RAMP_S:
        return dict(control=step.control, depth=RIM_PUSH_M, direction=step.direction, amount=amount, step=step,
                    phase='tilt' if t < step.end else 'return', rim=True)
    following = next((s for s in steps if s.start > step.start and s.control == step.control and s.action == 'tilt'), None)
    if following is not None and following.start - LAND_S - step.end < LINGER_S:
        return dict(control=step.control, depth=STICK_DISH_M, direction=None, amount=0., step=step, phase='seated')
    lift = smooth((t - step.end - TILT_RAMP_S) / RAMP_S)
    return dict(control=step.control, depth=RIM_PUSH_M + (RIM_HOVER_M - RIM_PUSH_M) * lift, direction=step.direction,
                amount=0., step=step, phase='lift', rim=True)


def _tilt_dish(step, t):
    """Tilt by dragging the cap's dish with the thumb seated just above its bottom (no press along the shaft).
    The amount is bounded at runtime by the measured tilt plus a lead (autopilot_source), so the thumb never
    runs ahead of the cap into its rim or the cone."""
    if t < step.start:
        return dict(control=step.control, depth=HOVER_M + (DISH_TILT_M - HOVER_M) * smooth((t - step.start + LAND_S) / LAND_S),
                    direction=step.direction, amount=0., step=step, phase='land', rim=False,
                    force=STICK_FORCE_N if TILT_FORCE and t >= step.start - LAND_S / 3 else None)
    amount = step.amount * _profile(step, t, TILT_RAMP_S)
    if t < step.end + TILT_RAMP_S:
        return dict(control=step.control, depth=DISH_TILT_M, direction=step.direction, amount=amount, step=step,
                    phase='tilt' if t < step.end else 'return', rim=False, force=STICK_FORCE_N if TILT_FORCE else None)
    lift = smooth((t - step.end - TILT_RAMP_S) / RAMP_S)
    return dict(control=step.control, depth=DISH_TILT_M + (HOVER_M - DISH_TILT_M) * lift, direction=None,
                amount=0., step=step, phase='lift', rim=False)
