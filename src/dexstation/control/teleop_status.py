"""Display-only telemetry, sampled with observations; never written to episodes."""
import math

from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES

# One source for both the backend timer and the browser's status decisions.
TELEOP_THRESHOLDS = dict(arm_error_deg=10., hand_error_deg=15., error_duration_s=.5,
                         input_stale_s=.1, limit_epsilon_rad=1e-6)
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def arm_input_status(health):
    """Display the source's measured availability without inferring from angles."""
    return {side: dict(available=row['input_available'], reason=row.get('input_reason'),
                       linkerta=row.get('linkerta'))
            for side in ('left', 'right')
            if (row := health.get(side+'_arm', {})).get('input_available') is not None}


class TeleopMonitor:
    def __init__(self):
        self.since = {}
        self.latest = None

    def _duration(self, key, exceeds, now):
        if not exceeds:
            self.since.pop(key, None)
            return 0.
        return max(0., now-self.since.setdefault(key, now))

    def update(self, observation, commands, applied_reference, hand_bounds, now):
        """Measure actual minus effective target, on a monotonic wall clock.

        Timers reset at every sampled recovery. They do not advance on repeated
        SSE reads of the same observation. Missing channels remain unavailable.
        """
        result = dict(thresholds=dict(TELEOP_THRESHOLDS), arms={}, hands={})
        active_keys = set()
        for si, side in enumerate(('left', 'right')):
            for kind in ('arm', 'hand'):
                channel = side+'_'+kind
                command, target = commands.get(channel), applied_reference.get(channel)
                if command is None or target is None:
                    continue
                actual = getattr(observation, kind+'_q')[si]
                errors = [math.degrees(float(q)-float(t)) for q, t in zip(actual, target)]
                if kind == 'arm':
                    durations = []
                    for i, error in enumerate(errors):
                        key = (channel, i);active_keys.add(key)
                        durations.append(self._duration(key, abs(error)>TELEOP_THRESHOLDS['arm_error_deg'], now))
                    result['arms'][side] = dict(effective_target_deg=[math.degrees(float(t)) for t in target],
                                               error_deg=errors, error_duration_s=durations)
                else:
                    bounds = hand_bounds.get(side)
                    desired = command['joint_positions']
                    fingers = {}
                    for finger in FINGERS:
                        indices = [i for i, n in enumerate(G20_JOINT_NAMES) if n.startswith(finger+'_')]
                        error = max(abs(errors[i]) for i in indices)
                        key = (channel, finger);active_keys.add(key)
                        duration = self._duration(key, error>TELEOP_THRESHOLDS['hand_error_deg'], now)
                        eps = TELEOP_THRESHOLDS['limit_epsilon_rad']
                        # A normalized target on/outside a model endpoint is a
                        # limit indication, not a claim about contact forces.
                        limited = None if bounds is None else any(desired[G20_JOINT_NAMES[i]] <= bounds[0][i]+eps or
                                      desired[G20_JOINT_NAMES[i]] >= bounds[1][i]-eps for i in indices)
                        fingers[finger] = dict(max_error_deg=error, error_duration_s=duration, limited=limited)
                    result['hands'][side] = dict(fingers=fingers)
        self.since = {k: v for k, v in self.since.items() if k in active_keys}
        self.latest = result
        return result
