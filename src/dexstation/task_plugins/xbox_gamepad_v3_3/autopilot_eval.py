"""Score IK autopilot steps against the gamepad's own signals (sdl3-gamepad-core-v1).

A step succeeds only if its control's signal does what the step asked; any other control
rising inside a task window is a false trigger. Latency is from the scheduled step start to
the first sample that crosses the threshold, on the simulation clock.
"""
import math

import numpy as np

from dexstation.task_plugins.xbox_gamepad_v3_3.signals import SDL3_NAMES

BUTTON_ON = .5
BUTTON_OFF = .3
TRIGGER_ON = .5          # fraction of the requested pull
TRIGGER_REACH = .85      # fraction of the requested pull that must be reached
TRIGGER_FALSE = .3
STICK_ON = .5            # fraction of the requested tilt, in the requested direction
STICK_REACH = .75
STICK_ANGLE_DEG = 25.
STICK_FALSE = .3
PRESS_WINDOW_S = .25     # a press must register this soon after the step ends at the latest
RELEASE_WINDOW_S = .35
HOLD_FRACTION = .9
FALSE_GUARD_S = .05
MASH_PERIOD_S = .3


def channel(name):
    return SDL3_NAMES.index(name)


def stick_vector(signals, control, axes):
    """Measured stick deflection in the gamepad face frame (+x right, +y toward the bumpers)."""
    spec = axes[control]
    x = signals[:, channel(spec['x'][0])] * spec['x'][1]
    y = signals[:, channel(spec['y'][0])] * spec['y'][1]
    return np.stack([x, y], axis=1)


def category(step, steps, controls):
    kind = controls[step.control]['kind']
    if step.control in ('l3', 'r3'):
        return 'stick_click'
    if kind == 'stick':
        return 'stick_tilt'
    if kind == 'trigger':
        return 'trigger'
    if step.control in ('lb', 'rb'):
        return 'bumper'
    if kind == 'dpad':
        return 'dpad'
    if step.action == 'hold':
        return 'hold'
    same = [s for s in steps if s.control == step.control and s is not step]
    if any(abs(s.start - step.start) < MASH_PERIOD_S for s in same):
        return 'mash'
    return 'tap'


def _first(mask, times, after):
    index = np.flatnonzero(mask & (times >= after))
    return None if not len(index) else int(index[0])


def score_step(step, steps, times, signals, controls, axes):
    """Return a result dict for one step (times are absolute, as in ``step``)."""
    kind = controls[step.control]['kind']
    result = dict(step=step.key, control=step.control, hand=step.hand, finger=step.finger, action=step.action,
                  start=step.start, end=step.end, category=category(step, steps, controls),
                  concurrent=any(s.hand != step.hand and s.start < step.end and step.start < s.end for s in steps),
                  same_hand_stick_trigger=kind in ('stick', 'trigger') and any(
                      s.hand == step.hand and controls[s.control]['kind'] in ('stick', 'trigger') and
                      controls[s.control]['kind'] != kind and s.start < step.end and step.start < s.end for s in steps),
                  success=False, latency_s=None, reason=None)
    nxt = min((s.start for s in steps if s.control == step.control and s.start > step.start), default=math.inf)
    window = (times >= step.start - FALSE_GUARD_S) & (times <= min(step.end + PRESS_WINDOW_S, nxt))
    if kind == 'stick':
        vec = stick_vector(signals, step.control, axes)
        want = np.asarray(step.direction) * step.amount
        along = vec @ np.asarray(step.direction)
        angle = np.degrees(np.arctan2(np.abs(vec[:, 0] * step.direction[1] - vec[:, 1] * step.direction[0]), along))
        on = (along >= STICK_ON * step.amount) & (angle <= STICK_ANGLE_DEG)
        first = _first(on & window, times, step.start - FALSE_GUARD_S)
        if first is None:
            result['reason'] = 'stick never reached half of the requested tilt in direction'
            result['peak'] = float(np.max(along[window], initial=0.))
            return result
        result['latency_s'] = float(times[first] - step.start)
        hold = (times >= times[first]) & (times <= step.end)
        good = (along >= STICK_REACH * step.amount) & (angle <= STICK_ANGLE_DEG)
        result['held_fraction'] = float(good[hold].mean()) if hold.any() else 0.
        result['peak'] = float(along[window].max())
        result['mean_error'] = float(np.linalg.norm(vec[hold] - want, axis=1).mean()) if hold.any() else None
        result['success'] = result['held_fraction'] >= .5
        if not result['success']:
            result['reason'] = 'stick tilt not held'
        return result
    value = signals[:, channel(controls[step.control]['channel'])]
    if kind == 'trigger':
        on, reach = value >= TRIGGER_ON * step.amount, TRIGGER_REACH * step.amount
        off = value < TRIGGER_FALSE
    else:
        on, reach, off = value >= BUTTON_ON, BUTTON_ON, value < BUTTON_OFF
    first = _first(on & window, times, step.start - FALSE_GUARD_S)
    result['peak'] = float(value[window].max()) if window.any() else 0.
    if first is None:
        result['reason'] = 'signal never crossed the threshold'
        return result
    result['latency_s'] = float(times[first] - step.start)
    if result['peak'] < reach:
        result['reason'] = 'requested pull not reached'
        return result
    if step.action == 'hold':
        hold = (times >= times[first]) & (times <= step.end)
        result['held_fraction'] = float(on[hold].mean()) if hold.any() else 0.
        if result['held_fraction'] < HOLD_FRACTION:
            result['reason'] = 'released before the end of the hold'
            return result
    released = _first(off, times, step.end)
    if released is None or times[released] > min(step.end + RELEASE_WINDOW_S, nxt + FALSE_GUARD_S):
        result['reason'] = 'not released in time'
        return result
    result['release_s'] = float(times[released] - step.end)
    result['success'] = True
    return result


def false_triggers(steps, times, signals, controls, axes, window):
    """Rising edges of controls that no step asked for at that time, inside ``window`` (start, end)."""
    inside = (times >= window[0]) & (times <= window[1])
    events = []
    for name, c in controls.items():
        if name in ('l3', 'r3') or c['kind'] == 'button' or c['kind'] == 'dpad':
            level, on, off = signals[:, channel(c['channel'])], BUTTON_ON, BUTTON_OFF
        elif c['kind'] == 'trigger':
            level, on, off = signals[:, channel(c['channel'])], TRIGGER_FALSE, TRIGGER_FALSE * .5
        else:
            level, on, off = np.linalg.norm(stick_vector(signals, name, axes), axis=1), STICK_FALSE, STICK_FALSE * .5
        allowed = np.zeros(len(times), bool)
        for s in steps:
            if s.control == name:
                allowed |= (times >= s.start - FALSE_GUARD_S) & (times <= s.end + RELEASE_WINDOW_S)
        high = False
        for i in np.flatnonzero(inside):
            if not high and level[i] >= on:
                high = True
                if not allowed[i]:
                    events.append(dict(control=name, time=float(times[i]), value=float(level[i])))
            elif high and level[i] < off:
                high = False
    return events


def summarize(results, false_events, *, task_count=None):
    by_category = {}
    for r in results:
        for key in (r['category'], 'all') + (('concurrent',) if r['concurrent'] else ()) + \
                (('same_hand_stick_trigger',) if r.get('same_hand_stick_trigger') else ()):
            c = by_category.setdefault(key, dict(steps=0, success=0, latencies=[]))
            c['steps'] += 1
            c['success'] += int(r['success'])
            if r['latency_s'] is not None:
                c['latencies'].append(r['latency_s'])
    for c in by_category.values():
        lat = np.asarray(c.pop('latencies'))
        c['success_rate'] = c['success'] / c['steps'] if c['steps'] else None
        c['latency_s'] = None if not len(lat) else dict(median=float(np.median(lat)), p90=float(np.percentile(lat, 90)),
                                                         max=float(lat.max()))
    false_by_control = {}
    for e in false_events:
        false_by_control[e['control']] = false_by_control.get(e['control'], 0) + 1
    return dict(tasks=task_count, categories=by_category, false_triggers=len(false_events),
                false_by_control=false_by_control)


def task_instances(events):
    """(cycle, task id, start) of every task the autopilot announced in its event labels."""
    seen = {}
    for event in events:
        if event.get('source') != 'ik_autopilot':
            continue
        fields = dict(part.split('=', 1) for part in event['header']['frame_id'].split(';')[1:] if '=' in part)
        if fields.get('task', 'none') != 'none':
            seen.setdefault((int(fields['cycle']), fields['task'], float(fields['task_start'])), None)
    return sorted(seen, key=lambda k: k[2])


def evaluate_episode(path, tasks_path=None):
    """Score a saved episode: every task wholly inside the recording, from its rows' signals."""
    import re
    from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_ik import load_asset
    from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_tasks import load_tasks, task_steps
    from dexstation.recording.episode import read_episode, read_input_events
    from dexstation.contracts.versions import GAMEPAD_RECORDING_FORMAT
    episode = read_episode(path)
    meta, _ = episode
    if episode.task is None or episode.task.format != GAMEPAD_RECORDING_FORMAT:
        raise ValueError('gamepad evaluation requires an xbox-gamepad-rec-v1 task layer')
    rows = episode.task.rows
    events = read_input_events(path, meta=meta)
    note = next((n['text'] for n in meta.get('operator_notes', []) if n['text'].startswith('IK autopilot task set')), '')
    match = re.match(r'IK autopilot task set (\S+) \((.+), sha256 ([0-9a-f]+)\)', note)
    doc = load_tasks(tasks_path or (match and match.group(2)))
    if match and not doc['sha256'].startswith(match.group(3)):
        raise ValueError('task set file differs from the one recorded in the episode note')
    asset = load_asset('xbox_one_s_controls.json')
    controls, axes = asset['controls'], asset['signal_axes']
    times = np.array([r['sim_time_s'] for r in rows])
    signals = np.array([r['gamepad_signal'] for r in rows])
    index = {t['id']: i for i, t in enumerate(doc['tasks'])}
    results, false_events, scored = [], [], []
    for cycle, task_id, start in task_instances(events):
        task = doc['tasks'][index[task_id]]
        end = start + task['duration']
        if start < times[0] or end + .5 > times[-1]:
            continue
        steps = task_steps(doc, index[task_id], start)
        window = (times >= start - .2) & (times <= end + .5)
        for s in steps:
            r = score_step(s, steps, times[window], signals[window], controls, axes)
            results.append(dict(r, task=task_id, cycle=cycle))
        for e in false_triggers(steps, times[window], signals[window], controls, axes, (start, end + .5)):
            false_events.append(dict(e, task=task_id, cycle=cycle))
        scored.append(dict(task=task_id, cycle=cycle, start=start, end=end))
    summary = summarize(results, false_events, task_count=len(scored))
    summary['tasks_detail'] = [dict(t, steps=sum(1 for r in results if r['task'] == t['task'] and r['cycle'] == t['cycle']),
                                    success=sum(1 for r in results if r['task'] == t['task'] and r['cycle'] == t['cycle'] and r['success']),
                                    false_triggers=sum(1 for e in false_events if e['task'] == t['task'] and e['cycle'] == t['cycle']))
                               for t in scored]
    summary['realtime'] = meta.get('realtime_metrics')
    summary['notes'] = [n['text'] for n in meta.get('operator_notes', [])]
    return summary, results, false_events
