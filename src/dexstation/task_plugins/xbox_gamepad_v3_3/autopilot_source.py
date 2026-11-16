"""IK autopilot: a collection input source that plays game-style gamepad tasks.

Same event interface as the ROS/UDP/replay sources. Each control tick it emits one arm and
one hand command per side, in the station's own command formats (seven slave radians; twenty
G20 SDK motor values), scheduled on the simulation clock like a stateful replay. Each hand has a few
play poses (one per control group); arm IK places the wrist at the pose the next step needs, along an
offline-checked approach line, while fingers follow hover -> contact -> press -> hold -> release
targets solved by finger IK against the measured palm pose every tick. The gamepad's own signals
close the loop on analog amounts and confirm every step (``autopilot_eval``).
"""
from collections import deque
from dataclasses import asdict
import math
import threading
import time

import numpy as np

from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_eval import BUTTON_ON, channel, false_triggers, score_step, stick_vector, summarize
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_ik import ArmModel, GamepadGeometry, HandModel, load_asset
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_plan import ASSIGNMENT, finger_target
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_tasks import load_tasks, task_steps
from dexstation.runtime.health import InputHealth
from dexstation.control.commands import ArmCommand, HandCommand, hand_command_from_packet
from dexstation.recording.episode import SIDES
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES, apply_mimic_joint_positions, joint_positions_to_motor_positions
from dexstation.control.protocol import HandCommandPacket
from dexstation.model.spec import pose_to_matrix

SOURCE = 'ik_autopilot'
APPROACH_STEPS = 12                    # waypoints on the straight retreat line (offline-checked direction and length)
APPROACH_SPEED_DEG_S = 30.             # streamed arm target speed, below the station's 40 deg/s cap
APPROACH_STEP_DEG = 1.                 # spacing of streamed arm targets (station deadband is 0.5 deg)
READY_TOLERANCE_DEG = 1.5
SETTLE_TOLERANCE_DEG = 10.            # at rest within this of the target also counts (contact with the grip)
SETTLE_MOTION_DEG = .1
SETTLE_S = .4
START_DELAY_S = .5
TRANSIT_S = .12
TRANSIT_BOW_M = .008                   # mid-transit offset along the controls' hover directions
IDLE_PREFERENCE = ('ls', 'rs', 'dpad_up', 'a', 'lt', 'rt', 'lb', 'rb')   # where an idle operating finger hovers
# Closed loop on the gamepad's signals (per step; reset when a step starts).
BUTTON_DEEPEN_M_S, BUTTON_DEEPEN_MAX_M, BUTTON_CHECK_S = .02, .0015, .05
ANALOG_GAIN, ANALOG_MAX = 4., .35
MAX_TILT = .95                         # never drive a stick into its mechanical stop (23 deg cone, rigid)
RIM_LEAD = .15                         # side-wall target leads the measured tilt by at most this fraction of the
                                       # limit (about 1 mm at the contact height): a bounded overlap, a bounded force
WRIST_LEAD_S = .45                     # start a stick-direction wrist adjustment this long before the tilt
FORCE_GAIN_M_PER_NS = .002             # stick force loop: depth change per newton of error per second
FORCE_ADJUST_MAX_M = .003
LEAD_FROM_M = -1.                      # leading-contact IK from this depth (now always: switching models mid-air
                                       # cost tilt success and added clicks in the station ablation)
MAX_RETRIES_PER_TARGET = 3             # full IK restarts while a target stays out of reach
IK_HOLD_M = 1.                         # hold the finger when the best solution misses by more than this (off)
FINGER_MAX_STEP_RAD = 9.               # per-tick finger joint change limit (off; both lowered stick success)
LEADING_CONTACT = ('thumb',)           # thumbs press with the fingertip sample that leads along the push; index
                                       # fingers on bumpers/triggers keep the pad point the play poses were solved for
CLICK_GUARD_M_S, CLICK_GUARD_MAX_M = .004, .0006  # lift a tilting thumb at once when its stick reports a click,
                                                  # settle back slowly (station sweep: R3 false clicks 22 -> 9)


def _direction_name(direction):
    from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_plan import DIRECTIONS
    return max(DIRECTIONS, key=lambda n: np.dot(DIRECTIONS[n], direction))


def _arc_length(path):
    """Cumulative max-norm joint distance (deg) along a joint path."""
    return np.concatenate([[0.], np.cumsum(np.degrees(np.abs(np.diff(path, axis=0)).max(1)))])


class Timeline:
    """Sequential tasks with a gap; optionally looping. Times are simulation seconds."""

    def __init__(self, doc, loop):
        self.doc, self.loop = doc, loop
        self.gap = float(doc.get('gap_s', 1.))
        self.origin = None
        self.index = 0
        self.cycle = 0
        self.done = False

    def start(self, t):
        self.origin, self.index, self.cycle, self.done = t, 0, 0, False

    def advance(self, t):
        """Move to the next task once the current one (plus gap) is over; return finished task windows."""
        finished = []
        while self.origin is not None and not self.done:
            task = self.doc['tasks'][self.index]
            end = self.origin + task['duration']
            if t < end + self.gap:
                break
            finished.append((self.cycle, self.index, self.origin, end))
            self.origin = end + self.gap
            self.index += 1
            if self.index == len(self.doc['tasks']):
                if self.loop:
                    self.index, self.cycle = 0, self.cycle + 1
                else:
                    self.done = True
        return finished

    def steps(self):
        """Steps of the current and the next task (the next one gives early hover targets)."""
        if self.origin is None or self.done:
            return []
        steps = task_steps(self.doc, self.index, self.origin)
        nxt = self.index + 1
        if nxt < len(self.doc['tasks']) or self.loop:
            nxt %= len(self.doc['tasks'])
            steps += task_steps(self.doc, nxt, self.origin + self.doc['tasks'][self.index]['duration'] + self.gap)
        return steps

    def current(self):
        if self.origin is None or self.done:
            return None
        task = self.doc['tasks'][self.index]
        return dict(index=self.index, id=task['id'], cycle=self.cycle, start=self.origin, end=self.origin + task['duration'])


class IkAutopilotSource:
    def status_fields(self):
        return dict(autopilot=self.autopilot_status())

    @staticmethod
    def contact_reader(backend):
        return lambda side, link, pad_link: -backend.contact_force(side, link, 'gamepad', pad_link)

    def __init__(self, spec, tasks_path, *, loop=True, poses=None, controls=None, clock=time.monotonic):
        self.tasks = load_tasks(tasks_path)
        self.poses = poses or load_asset('play_poses.json')
        self.geometry = GamepadGeometry(spec, controls)
        self.controls = self.geometry.controls
        self.axes = self.geometry.data['signal_axes']
        if not np.allclose(pose_to_matrix(self.poses['gamepad_root_pose']), self.geometry.pose, atol=1e-6):
            raise ValueError('IK autopilot play poses were solved for a different gamepad pose; '
                             'rerun scripts/tasks/xbox_gamepad_v3_3/solve_autopilot_poses.py for this scene')
        self.arms = {s: ArmModel(spec, s) for s in SIDES}
        self.hands = {s: HandModel(spec, s) for s in SIDES}
        self.modes = {s: [self._prepare_mode(s, i, m) for i, m in enumerate(self.poses['sides'][s]['modes'])] for s in SIDES}
        missing = [(s.hand, s.control) for i in range(len(self.tasks['tasks'])) for s in task_steps(self.tasks, i, 0.)
                   if not any(s.control in m['controls'] for m in self.modes[s.hand])]
        if missing:
            raise ValueError(f'no play pose covers {sorted(set(missing))}; extend the play poses or the task set')
        self.mode = {s: self._first_mode(s) for s in SIDES}
        self.finger_q = {s: {f: q.copy() for f, q in self.modes[s][self.mode[s]]['away'].items()} for s in SIDES}
        self.path = {s: np.array([self.arms[s].home] + self.modes[s][self.mode[s]]['descent']) for s in SIDES}
        self.path_s = {s: 0. for s in SIDES}
        self.path_length = {s: _arc_length(self.path[s]) for s in SIDES}
        self.clock = clock
        self.health = InputHealth(clock)
        self.decode_errors = self.received_packets = 0
        self.latest_events, self.polled_sequences = {}, {}
        self._events, self._lock = [], threading.Lock()
        self.timeline = Timeline(self.tasks, loop)
        self.arm_state = {s: 'moving' for s in SIDES}
        self.transitions = 0
        self.switch_at, self.finger_mode = {}, dict(self.mode)
        self.arm_target = {s: self.path[s][0].copy() for s in SIDES}
        self.arm_reached = {s: None for s in SIDES}
        self.settle_q = {}
        self.palm = {s: self.arms[s].hand_pose(self.arms[s].home) for s in SIDES}
        self.arm_q = {s: self.arms[s].home.copy() for s in SIDES}
        self.signals = None
        self.corrections = {}
        self.stick_force = {}
        self.force_adjust = {}
        self.force_log = deque(maxlen=20000)
        self.retries = {}
        self.lift = {}
        self.commanded = {}
        self.transit = {}
        self.history = deque()
        self.results, self.false_events, self.task_runs = [], [], 0
        self.ik_times = deque(maxlen=4000)
        self.ik_failures = 0
        self.ik_failure_kinds = {}
        self.last_t = 0.
        self.restart_requested = False

    # ---- input-source interface -------------------------------------------------
    def drain_events(self):
        with self._lock:
            events, self._events = self._events, []
        return events

    def boundary(self):
        return self.drain_events(), self.received_packets

    def close(self):
        pass

    def sense_contacts(self, contact_force):
        """Thumb-on-stick contact force per side (gamepad frame, N on the stick), from the backend."""
        for side, stick in (('left', 'left_stick'), ('right', 'right_stick')):
            self.stick_force[side] = self.geometry.rotation.T @ contact_force(side, 'thumb_distal', stick)

    def observe(self, observation):
        """Measured state after each control tick (palm poses, arm joints, gamepad signals)."""
        t = float(observation.sim_time_s)
        for i, side in enumerate(SIDES):
            self.palm[side] = pose_to_matrix(np.asarray(observation.palm_pose_world[i]).tolist())
            self.arm_q[side] = np.asarray(observation.arm_q[i], float)
        self.signals = np.asarray(observation.gamepad_signal, float)
        self._update_arms(t)
        self._update_corrections(t)
        if self.timeline.origin is not None:
            self.history.append((t, self.signals.copy()))
            for cycle, index, start, end in self.timeline.advance(t):
                self._score_task(index, start, end)
                for key in [k for k in self.corrections if k.startswith(f'{index}.')]:
                    del self.corrections[key]

    def poll(self, now):
        t = float(now)
        self.last_t = t
        if self.timeline.origin is None:
            # (Re)start at the first task once each hand is ready in the play pose that task begins with.
            for side in SIDES:
                wanted = self._first_mode(side)
                if wanted != self.mode[side] and self.arm_state[side] != 'moving':
                    self._transition(side, wanted)
            if all(self.arm_state[s] == 'ready' and self.mode[s] == self._first_mode(s) for s in SIDES):
                self.restart_requested = False
                self._start_timeline(t)
        started = time.perf_counter()
        commands, labels = [], {}
        steps = self.timeline.steps() if self.timeline.origin is not None and t >= self.timeline.origin - 1. else []
        for side in SIDES:
            self._plan_mode(side, t, [s for s in steps if s.hand == side])
            self._wrist(side, t, [s for s in steps if s.hand == side])
        for side in SIDES:
            commands.append(ArmCommand(side, tuple(float(v) for v in self.arm_target[side]), t))
            hand, label = self._hand(side, t, [s for s in steps if s.hand == side])
            commands.append(hand)
            labels[side] = label
        self.ik_times.append(time.perf_counter() - started)
        task = self.timeline.current()
        frame = f'{SOURCE};set={self.tasks["name"]}' + (
            f";task={task['id']};cycle={task['cycle']};task_start={task['start']:.6f}" if task else ';task=none')
        with self._lock:
            for command in commands:
                kind = 'hand' if isinstance(command, HandCommand) else 'arm'
                extra = f';steps={labels[command.side]}' if kind == 'hand' and labels[command.side] else ''
                self._emit(command, t, frame + extra)
        return commands

    def recording_started(self):
        """Each recording begins with the first task, once both arms are in place."""
        self.timeline.origin = None
        self.restart_requested = True
        self.history.clear()
        self.results, self.false_events, self.task_runs = [], [], 0
        self.ik_times.clear()
        self.ik_failures = 0
        self.ik_failure_kinds = {}

    def episode_note(self):
        return (f"IK autopilot task set {self.tasks['name']} ({self.tasks['path']}, sha256 {self.tasks['sha256'][:12]}); "
                f"finger assignment {ASSIGNMENT}")

    def summary_note(self):
        status = self.autopilot_status()
        return 'IK autopilot summary: ' + str(dict(status, task=None))

    # ---- arms and play poses ---------------------------------------------------------
    def _prepare_mode(self, side, index, mode):
        """Runtime arm IK for one play pose and its approach line; idle hover controls; away finger joints."""
        arm = self.arms[side]
        target = self.geometry.pose @ np.asarray(mode['hand_in_gamepad'])
        play = arm.solve(target, mode['arm_q'])
        if play.error_m > 1e-3 or play.angle_deg > .5:
            raise ValueError(f"{side} arm cannot reach play pose {mode['name']} in this scene "
                             f'({play.error_m * 1e3:.1f} mm, {play.angle_deg:.1f} deg)')
        approach = mode['approach']
        direction = self.geometry.rotation @ np.asarray(approach['direction'], float)
        direction /= np.linalg.norm(direction)
        q, chain = play.q, []
        for k in range(1, APPROACH_STEPS + 1):
            pose = target.copy()
            pose[:3, 3] += direction * approach['lift_m'] * k / APPROACH_STEPS
            ik = arm.solve(pose, q, iterations=300)
            if ik.error_m > 1e-3 or ik.angle_deg > .5:
                raise ValueError(f"{side} approach line of {mode['name']} is not reachable ({ik.error_m * 1e3:.1f} mm)")
            q = ik.q
            chain.append(q.copy())
        controls = list(mode['controls'])
        fingers = {ASSIGNMENT[side][c] for c in controls}
        idle = {f: next(c for c in IDLE_PREFERENCE + tuple(controls) if c in controls and ASSIGNMENT[side][c] == f)
                for f in fingers}
        away = {f: np.asarray(v, float) for f, v in approach['away_fingers'].items()}
        # Small wrist moves that bring a rim push within the thumb's reach (scripts/solve_autopilot_micro.py).
        micro = {}
        for name, value in mode.get('micro', {}).items():
            ik = arm.solve(self.geometry.pose @ np.asarray(value['hand_in_gamepad']), play.q)
            if ik.error_m < 1e-3 and ik.angle_deg < .5:
                micro[name] = ik.q
        return dict(name=mode['name'], index=index, controls=controls, fingers=fingers, idle=idle, play_q=play.q,
                    retreat=chain, descent=chain[::-1] + [play.q], away=away, micro=micro)

    def _wrist(self, side, t, steps):
        """In a stick pose, move the wrist to the adjustment of the tilt in progress or about to land."""
        if self.arm_state[side] != 'ready':
            return
        from dexstation.task_plugins.xbox_gamepad_v3_3 import autopilot_plan
        mode = self.modes[side][self.mode[side]]
        if not mode['micro'] or not autopilot_plan.TILT_SIDE:     # adjustments were solved for side-wall pushes
            return
        tilts = sorted((s for s in steps if s.action == 'tilt' and s.control in mode['controls']
                        and s.start - WRIST_LEAD_S <= t <= s.end + .2), key=lambda s: s.start)
        name = _direction_name(tilts[0].direction) if tilts else None
        self.arm_target[side] = mode['micro'].get(name, mode['play_q'])

    def _mode_for(self, side, control, current):
        if control is None:
            return current or 0
        if current is not None and control in self.modes[side][current]['controls']:
            return current
        return next(i for i, m in enumerate(self.modes[side]) if control in m['controls'])

    def _plan_mode(self, side, t, steps):
        """Move the arm to the play pose the side's next step needs, once its fingers are free."""
        if self.arm_state[side] == 'moving' or self.timeline.origin is None:
            return
        mode = self.modes[side][self.mode[side]]
        busy = any(s.start - .2 <= t <= s.end + .25 for s in steps if s.control in mode['controls'])
        upcoming = sorted((s for s in steps if s.end + .25 >= t), key=lambda s: s.start)
        if busy or not upcoming:
            return
        wanted = self._mode_for(side, upcoming[0].control, self.mode[side])
        if wanted != self.mode[side]:
            self._transition(side, wanted)

    def _first_mode(self, side):
        """Play pose of the side's first step in the task set."""
        for i in range(len(self.tasks['tasks'])):
            steps = sorted((s for s in task_steps(self.tasks, i, 0.) if s.hand == side), key=lambda s: s.start)
            if steps:
                return self._mode_for(side, steps[0].control, None)
        return 0

    def _transition(self, side, wanted):
        """Retreat along the current approach line, cross to the new staging pose, descend to the new play pose."""
        old, new = self.modes[side][self.mode[side]], self.modes[side][wanted]
        self.path[side] = np.array([self.arm_target[side]] + old['retreat'] + new['descent'])
        self.path_s[side], self.path_length[side] = 0., _arc_length(self.path[side])
        self.switch_at[side] = self.path_length[side][len(old['retreat']) + 1]
        self.mode[side], self.arm_state[side], self.arm_reached[side] = wanted, 'moving', None
        self.finger_q[side] = {f: q.copy() for f, q in old['away'].items()}
        self.transitions += 1

    def _path_point(self, side, length):
        path, cumulative = self.path[side], self.path_length[side]
        i = min(int(np.searchsorted(cumulative, length, side='right')) - 1, len(path) - 2)
        span = cumulative[i + 1] - cumulative[i]
        return path[i] + (0. if span <= 0 else (length - cumulative[i]) / span) * (path[i + 1] - path[i])

    def _update_arms(self, t):
        dt = max(0., t - getattr(self, '_last_arm_t', t))
        self._last_arm_t = t
        for side in SIDES:
            path = self.path[side]
            if self.arm_state[side] == 'moving':
                # Stream targets along the joint path at a bounded speed. Targets sit at whole multiples of
                # APPROACH_STEP_DEG (max-norm) counted back from the end, so every update clears the station's
                # 0.5 deg deadband and the last one lands exactly on the play pose.
                progress = self.path_s[side] + APPROACH_SPEED_DEG_S * dt
                total = self.path_length[side][-1]
                self.path_s[side] = min(progress, total)
                snapped = total - math.ceil((total - self.path_s[side]) / APPROACH_STEP_DEG - 1e-9) * APPROACH_STEP_DEG
                self.arm_target[side] = self._path_point(side, max(0., snapped))
                if self.path_s[side] >= self.switch_at.get(side, 0.) and self.finger_mode.get(side) != self.mode[side]:
                    # Past the old staging pose: hold the fingers away from the new play pose's controls.
                    self.finger_q[side] = {f: q.copy() for f, q in self.modes[side][self.mode[side]]['away'].items()}
                    self.finger_mode[side] = self.mode[side]
                if self.path_s[side] >= total:
                    self.arm_state[side] = 'settle'
                continue
            if self.arm_state[side] == 'settle':
                # Ready once on target, or once at rest near it (the palm may rest against the grip).
                error = np.degrees(np.abs(self.arm_q[side] - self.arm_target[side]).max())
                moved = np.degrees(np.abs(self.arm_q[side] - self.settle_q.get(side, self.arm_q[side])).max())
                if error > SETTLE_TOLERANCE_DEG or (error > READY_TOLERANCE_DEG and moved > SETTLE_MOTION_DEG):
                    self.arm_reached[side] = None
                    self.settle_q[side] = self.arm_q[side].copy()
                    continue
                if self.arm_reached[side] is None:
                    self.arm_reached[side] = t
                    self.settle_q[side] = self.arm_q[side].copy()
                if t - self.arm_reached[side] >= SETTLE_S:
                    self.arm_state[side] = 'ready'

    # ---- timeline and fingers ------------------------------------------------------
    def _start_timeline(self, t):
        self.timeline.start(t + START_DELAY_S)
        self.history.clear()
        self.corrections.clear()

    def _hand(self, side, t, steps):
        hand, mode = self.hands[side], self.modes[side][self.mode[side]]
        joints = dict.fromkeys(G20_JOINT_NAMES, 0.)
        labels = []
        ready = self.arm_state[side] == 'ready'
        for finger in hand.fingers:
            q = self.finger_q[side].get(finger, np.zeros(len(hand.fingers[finger].active)))
            if ready and finger in mode['fingers']:
                own = sorted((s for s in steps if s.finger == finger and s.control in mode['controls']), key=lambda s: s.start)
                target = finger_target(finger, own, t, mode['idle'][finger], self.controls)
                step = target['step']
                if step is not None and target['phase'] not in ('hover', 'idle', 'seated'):
                    labels.append(f'{step.key}:{step.control}:{target["phase"]}')
                point, push = self._target_point(side, finger, target, t)
                # Leading contact only near the surface; in the air the pad point is cheaper and always reachable.
                lead = finger in LEADING_CONTACT and target['depth'] >= LEAD_FROM_M
                ik = hand.solve(finger, q, self.palm[side], point, push, iterations=25, tolerance_m=5e-5, leading=lead)
                if lead and ik.error_m > 1e-3 and self.retries.get((side, finger), 0) < MAX_RETRIES_PER_TARGET:
                    # The leading sample jumps with orientation; restart from the pad-point solution.
                    seed = hand.solve(finger, q, self.palm[side], point, push, iterations=100, tolerance_m=5e-5).q
                    retry = hand.solve(finger, seed, self.palm[side], point, push, iterations=100, tolerance_m=5e-5, leading=True)
                    ik = retry if retry.error_m < ik.error_m else ik
                    self.retries[(side, finger)] = self.retries.get((side, finger), 0) + 1
                elif not lead or ik.error_m <= 1e-3:
                    self.retries[(side, finger)] = 0
                if ik.error_m > 1e-3:
                    self.ik_failures += 1
                    if ik.error_m > IK_HOLD_M:
                        # Far out of reach: a best-effort solution can be anywhere; hold the finger instead.
                        ik = type(ik)(q, ik.error_m, ik.angle_deg, ik.iterations)
                    key = f'{side}_{finger}:{target["control"]}:{target["phase"]}'
                    self.ik_failure_kinds[key] = self.ik_failure_kinds.get(key, 0) + 1
                step_q = np.clip(ik.q - q, -FINGER_MAX_STEP_RAD, FINGER_MAX_STEP_RAD)
                q = self.finger_q[side][finger] = q + step_q
            joints.update(zip(hand.fingers[finger].active, q))
        values = apply_mimic_joint_positions(side, [joints[n] for n in G20_JOINT_NAMES])
        # Whole SDK motor units, converted back exactly as a received station hand packet.
        motors = [float(round(m)) for m in joint_positions_to_motor_positions(side, values)]
        packet = HandCommandPacket(side, 'motor', [], motors, [], [], t)
        return hand_command_from_packet(packet), ','.join(labels)

    def _target_point(self, side, finger, target, t):
        """World contact target; closed-loop corrections and lifted transits between controls."""
        step = target['step']
        depth, amount, direction = target['depth'], target['amount'], target['direction']
        if step is not None:
            fix = self.corrections.get(step.key)
            kind = self.controls[step.control]['kind']
            if kind == 'stick' and target['step'] is step and not any(
                    c.control in ('l3', 'r3') and c.start <= t <= c.end + .1 for c in self.timeline.steps()):
                depth -= self.lift.get(step.key, 0.)
            if fix is not None and kind == 'stick':
                if direction is not None and amount > 0:
                    # Scale the correction with the ramp so it vanishes as the stick returns to centre.
                    want = np.asarray(direction) * amount + fix * (amount / max(step.amount, 1e-9))
                    amount = min(MAX_TILT, float(np.linalg.norm(want)))
                    direction = tuple(want / amount) if amount > 1e-6 else direction
            if kind == 'stick' and step.action == 'tilt' and target.get('rim') and direction is not None and self.signals is not None:
                # Lead the cap's measured tilt by a bounded amount: the thumb stays on the rim and pushes it,
                # instead of jumping to where the tilted rim would be (and leaving the cap behind).
                unit = np.asarray(direction) / np.linalg.norm(direction)
                measured = float(stick_vector(self.signals[None], step.control, self.axes)[0] @ unit)
                amount = min(amount, MAX_TILT, max(0., measured) + RIM_LEAD)
            elif fix is not None and kind == 'trigger':
                amount = min(1., amount + float(fix) * (amount / max(step.amount, 1e-9)))
            elif fix is not None and target['phase'] == 'press':
                depth += float(fix)
        if target.get('force') is not None and side in self.stick_force:
            # Force loop through the joint targets: deepen or back off along the push until the thumb's push
            # down the stick (gamepad -z) matches the target; the depth from the plan only seeds it.
            down = -float(self.stick_force[side][2])
            dt = max(0., t - self.force_adjust.get(side, (t, 0.))[0])
            adjust = self.force_adjust.get(side, (t, 0.))[1] + FORCE_GAIN_M_PER_NS * (target['force'] - down) * dt
            adjust = float(np.clip(adjust, -FORCE_ADJUST_MAX_M, FORCE_ADJUST_MAX_M))
            self.force_adjust[side] = (t, adjust)
            depth += adjust
            self.force_log.append((side, target['phase'], target['force'], down, adjust))
        elif finger == 'thumb':
            self.force_adjust.pop(side, None)
        rim = target.get('rim') and direction is not None
        local = self._local_target(target['control'], depth, direction, amount, rim)
        key = (side, finger)
        previous = self.commanded.get(key)
        identity = (target['control'], tuple(direction) if rim else None)
        if previous is not None and previous[0] != identity and target['phase'] in ('hover', 'idle', 'land'):
            # Also between the back of a stick cap and its centre: go over the cap, not through it.
            self.transit[key] = (t, previous[1], identity, previous[0][0])
        move = self.transit.get(key)
        if move is not None:
            begin, origin, aim, source = move
            control = aim[0]
            s = (t - begin) / TRANSIT_S
            if aim != identity or s >= 1:
                self.transit.pop(key)
            else:
                # Straight line in the gamepad frame, bowed out along both controls' hover directions so the
                # finger clears what lies between them (stick caps on the face; the bumper between the trigger
                # and the bumper hover).
                path = origin + (local - origin) * s
                away = -np.asarray(self.controls[source]['push']) - np.asarray(self.controls[control]['push'])
                away /= np.linalg.norm(away)
                path += 4 * s * (1 - s) * TRANSIT_BOW_M * away
                local = path
        self.commanded[key] = (identity, local)
        push = (self.geometry.rim_target(target['control'], direction, amount)[1] if rim else
                self.geometry.target(target['control'], depth=0., direction=direction, amount=amount)[1])
        return self.geometry.world(local), push

    def _local_target(self, control, depth, direction, amount, rim=False):
        world = (self.geometry.rim_target(control, direction, amount, depth)[0] if rim else
                 self.geometry.target(control, depth=depth, direction=direction, amount=amount)[0])
        return self.geometry.rotation.T @ (world - self.geometry.pose[:3, 3])

    def _update_corrections(self, t):
        if self.timeline.origin is None or self.signals is None:
            return
        dt = max(0., t - getattr(self, '_last_correction_t', t))
        self._last_correction_t = t
        for step in self.timeline.steps():
            if not step.start <= t <= step.end:
                continue
            kind = self.controls[step.control]['kind']
            if kind == 'stick' and step.action == 'tilt':
                click = self.signals[channel(self.controls['l3' if step.control == 'ls' else 'r3']['channel'])]
                lift = self.lift.get(step.key, 0.)
                self.lift[step.key] = min(CLICK_GUARD_MAX_M, lift + CLICK_GUARD_MAX_M) if click > .3 else max(0., lift - CLICK_GUARD_M_S * dt)
                if t - step.start < .1:
                    continue
                measured = stick_vector(self.signals[None], step.control, self.axes)[0]
                if np.linalg.norm(measured) >= MAX_TILT:
                    continue                  # anti-windup: at the cone, more command only adds force
                error = np.asarray(step.direction) * min(step.amount, MAX_TILT) - measured
                fix = self.corrections.get(step.key, np.zeros(2)) + ANALOG_GAIN * error * dt
                norm = np.linalg.norm(fix)
                self.corrections[step.key] = fix * min(1., ANALOG_MAX / norm) if norm > 0 else fix
            elif kind == 'trigger':
                if t - step.start < .1:
                    continue
                measured = self.signals[channel(self.controls[step.control]['channel'])]
                fix = self.corrections.get(step.key, 0.) + ANALOG_GAIN * (step.amount - measured) * dt
                self.corrections[step.key] = float(np.clip(fix, -ANALOG_MAX, ANALOG_MAX))
            elif t - step.start >= BUTTON_CHECK_S:
                measured = self.signals[channel(self.controls[step.control]['channel'])]
                if measured < BUTTON_ON + .2:
                    fix = self.corrections.get(step.key, 0.) + BUTTON_DEEPEN_M_S * dt
                    self.corrections[step.key] = min(BUTTON_DEEPEN_MAX_M, fix)

    # ---- evaluation and status -------------------------------------------------------
    def _score_task(self, index, start, end):
        window_end = end + .5
        rows = [(t, s) for t, s in self.history if start - .2 <= t <= window_end]
        while self.history and self.history[0][0] < end - 1.:
            self.history.popleft()
        if not rows:
            return
        times = np.array([t for t, _ in rows])
        signals = np.array([s for _, s in rows])
        steps = task_steps(self.tasks, index, start)
        results = [score_step(s, steps, times, signals, self.controls, self.axes) for s in steps]
        for r in results:
            r['task'] = self.tasks['tasks'][index]['id']
        events = false_triggers(steps, times, signals, self.controls, self.axes, (start, window_end))
        for e in events:
            e['task'] = self.tasks['tasks'][index]['id']
        self.results += results
        self.false_events += events
        self.task_runs += 1

    def autopilot_status(self):
        summary = summarize(self.results, self.false_events, task_count=self.task_runs)
        times = np.asarray(self.ik_times) * 1e3
        all_ = summary['categories'].get('all', {})
        return dict(set=self.tasks['name'], arms=dict(self.arm_state), task=self.timeline.current(),
                    play_pose={s: self.modes[s][self.mode[s]]['name'] for s in SIDES}, transitions=self.transitions,
                    tasks_scored=self.task_runs, steps=all_.get('steps', 0), success=all_.get('success', 0),
                    false_triggers=summary['false_triggers'], categories=summary['categories'],
                    ik_ms=None if not len(times) else dict(median=float(np.median(times)), p95=float(np.percentile(times, 95)),
                                                           max=float(times.max())),
                    ik_failures=self.ik_failures)

    def _emit(self, command, t, frame_id):
        received_mono, received_wall = time.monotonic_ns(), time.time_ns()
        kind = 'hand' if isinstance(command, HandCommand) else 'arm'
        event = dict(sequence=self.received_packets, channel=f'{command.side}_{kind}',
                     received_monotonic_ns=received_mono, received_wall_time_ns=received_wall,
                     header=dict(stamp=dict(sec=received_wall // 10**9, nanosec=received_wall % 10**9), frame_id=frame_id),
                     raw=asdict(command), converted_command=asdict(command), error=None,
                     scheduled_sim_time_s=t, source=SOURCE)
        self.received_packets += 1
        self.health.receive(event['channel'])
        self._events.append(event)
        self.latest_events[event['channel']] = event
        self.polled_sequences[event['channel']] = event['sequence']
