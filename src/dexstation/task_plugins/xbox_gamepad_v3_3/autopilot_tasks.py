"""Game-style gamepad task sets for the IK autopilot: loading, validation and scheduling."""
from dexstation.contracts.versions import AUTOPILOT_TASKS_SCHEMA
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_plan import ASSIGNMENT, CONTROL_SIDE, DIRECTIONS

SCHEMA = AUTOPILOT_TASKS_SCHEMA
ACTIONS = ('press', 'hold', 'tilt')
STICKS = {'ls': 'l3', 'rs': 'r3'}
TRIGGERS = ('lt', 'rt')
MIN_SAME_FINGER_GAP_S = .08


@dataclass(frozen=True)
class Step:
    task: int
    index: int
    control: str
    hand: str
    finger: str
    action: str
    start: float
    end: float
    direction: tuple | None = None
    amount: float = 1.

    @property
    def key(self):
        return f'{self.task}.{self.index}'


def _number(value, label, low=0., high=math.inf, inclusive_low=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if value < low or (value == low and not inclusive_low) or value > high:
        raise ValueError(f'{label} out of range')
    return float(value)


def validate(doc):
    """Return the task set unchanged if every step is executable; raise ValueError otherwise."""
    if doc.get('schema') != SCHEMA:
        raise ValueError(f'task set schema must be {SCHEMA}')
    if not isinstance(doc.get('name'), str) or not doc['name']:
        raise ValueError('task set needs a name')
    _number(doc.get('gap_s', 1.), 'gap_s')
    tasks = doc.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('task set needs at least one task')
    ids = set()
    for ti, task in enumerate(tasks):
        label = f"task {task.get('id', ti)!r}"
        if not isinstance(task.get('id'), str) or task['id'] in ids:
            raise ValueError(f'{label}: id must be a unique string')
        ids.add(task['id'])
        duration = _number(task.get('duration'), label + ' duration', 0., inclusive_low=False)
        steps = task.get('steps')
        if not isinstance(steps, list) or not steps:
            raise ValueError(f'{label}: needs steps')
        for si, step in enumerate(steps):
            where = f'{label} step {si}'
            control = step.get('control')
            if control not in CONTROL_SIDE:
                raise ValueError(f'{where}: unknown control {control!r}')
            hand, finger = CONTROL_SIDE[control], ASSIGNMENT[CONTROL_SIDE[control]][control]
            if step.get('hand') != hand or step.get('finger') != finger:
                raise ValueError(f'{where}: {control} is operated by the {hand} {finger}')
            action = step.get('action')
            if action not in ACTIONS:
                raise ValueError(f'{where}: action must be one of {ACTIONS}')
            if (action == 'tilt') != (control in STICKS):
                raise ValueError(f'{where}: sticks take tilt; other controls take press or hold')
            start = _number(step.get('t'), where + ' t')
            length = _number(step.get('duration'), where + ' duration', 0., inclusive_low=False)
            if start + length > duration + 1e-9:
                raise ValueError(f'{where}: ends after the task')
            if action == 'tilt':
                direction = step.get('direction')
                if isinstance(direction, str):
                    if direction not in DIRECTIONS:
                        raise ValueError(f'{where}: direction must be one of {sorted(DIRECTIONS)} or [x, y]')
                elif not (isinstance(direction, list) and len(direction) == 2 and
                          math.hypot(*(_number(v, where + ' direction', -1., 1.) for v in direction)) > .5):
                    raise ValueError(f'{where}: direction must be a name or a unit [x, y]')
                _number(step.get('amount'), where + ' amount', 0., 1., inclusive_low=False)
            elif 'amount' in step:
                if control not in TRIGGERS:
                    raise ValueError(f'{where}: amount applies to triggers and stick tilts')
                _number(step['amount'], where + ' amount', 0., 1., inclusive_low=False)
        _check_fingers(task, label)
    return doc


def _check_fingers(task, label):
    by_finger = {}
    for step in task['steps']:
        by_finger.setdefault((step['hand'], step['finger']), []).append(step)
    for (hand, finger), steps in by_finger.items():
        steps = sorted(steps, key=lambda s: s['t'])
        for a_i, a in enumerate(steps):
            for b in steps[a_i + 1:]:
                if b['t'] >= a['t'] + a['duration'] + MIN_SAME_FINGER_GAP_S:
                    break
                pair = {a['control'], b['control']}
                stick = next((s for s in STICKS if s in pair), None)
                nested = stick and STICKS[stick] in pair and _inside(*sorted((a, b), key=lambda s: s['control'] != stick))
                if not nested:
                    raise ValueError(f"{label}: the {hand} {finger} has overlapping steps on {sorted(pair)}; "
                                     'only a stick click inside a tilt of the same stick may overlap')


def _inside(tilt, click):
    return tilt['t'] <= click['t'] and click['t'] + click['duration'] <= tilt['t'] + tilt['duration']


def load_tasks(path):
    path = Path(path)
    text = path.read_text()
    doc = validate(json.loads(text))
    return dict(doc, path=str(path.resolve()), sha256=hashlib.sha256(text.encode()).hexdigest())


def direction_vector(value):
    x, y = DIRECTIONS[value] if isinstance(value, str) else value
    norm = math.hypot(x, y)
    return (x / norm, y / norm)


def task_steps(doc, task_index, origin):
    """Steps of one task with absolute times."""
    steps = []
    for index, s in enumerate(doc['tasks'][task_index]['steps']):
        steps.append(Step(task_index, index, s['control'], s['hand'], s['finger'], s['action'],
                          origin + s['t'], origin + s['t'] + s['duration'],
                          direction_vector(s['direction']) if s['action'] == 'tilt' else None,
                          float(s.get('amount', 1.))))
    return steps
