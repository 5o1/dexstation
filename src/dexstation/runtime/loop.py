"""Deterministic fixed-rate workstation collection loop."""

from collections import deque


LIVE_INPUT_MAX_DELAY_S = .100


class LatestInput:
    """Live targets are sample-and-hold values, sampled before each native step.

    Receipt stamps remain untouched in the journal. The optional scheduling and
    discard fields let replay reproduce the decisions on the simulation clock.
    Silence holds the last target; it does not manufacture a fresh input.
    """
    def __init__(self, max_delay_s=LIVE_INPUT_MAX_DELAY_S):
        self.max_delay_s = max_delay_s
        self.dropped = 0
        self.applied_received_ns = {}
        self.selected = {}
        self.applied_sequences = {}
        self.max_applied_delay_s = 0.

    def select(self, events, now_ns, sim_time_s):
        latest = {}
        for event in events:
            if event.get('converted_command') is not None and not event.get('error'):
                latest[event['channel']] = event
        selected = []
        journal = []
        for original in events:
            event = dict(original, scheduled_sim_time_s=sim_time_s)
            if original.get('converted_command') is not None and not original.get('error'):
                age = max(0., (now_ns-original['received_monotonic_ns'])/1e9)
                discard = latest[original['channel']] is not original or age > self.max_delay_s
                event['realtime_discarded'] = discard
                if discard:
                    self.dropped += 1
                else:
                    selected.append(event)
                    self.selected[event['channel']] = (event['sequence'],original['received_monotonic_ns'])
            journal.append(event)
        return selected, journal

    def mark_applied(self, sequences, now_ns):
        for channel,(sequence,received_ns) in self.selected.items():
            if sequences.get(channel)==sequence and self.applied_sequences.get(channel)!=sequence:
                self.applied_sequences[channel]=sequence
                self.applied_received_ns[channel]=received_ns
                self.max_applied_delay_s=max(self.max_applied_delay_s,max(0.,(now_ns-received_ns)/1e9))

    def status(self, now_ns):
        ages = {k: max(0., (now_ns-v)/1e9) for k,v in self.applied_received_ns.items()}
        fresh = [age for age in ages.values() if age <= self.max_delay_s]
        return dict(input_delay_s=max(fresh) if fresh else None, input_age_s=ages,
                    stale_channels=[k for k,v in ages.items() if v > self.max_delay_s],
                    dropped_inputs=self.dropped, input_delay_limit_s=self.max_delay_s,
                    max_applied_input_delay_s=self.max_applied_delay_s)


class RealtimeFactor:
    """Trailing one-second wall-clock window, including pacing and Python work."""
    def __init__(self, wall_s, sim_s):
        self.samples = deque([(wall_s, sim_s)])
        self.value = None

    def update(self, wall_s, sim_s):
        self.samples.append((wall_s, sim_s))
        while len(self.samples) > 2 and self.samples[1][0] <= wall_s-1.:
            self.samples.popleft()
        wall0, sim0 = self.samples[0]
        self.value = (sim_s-sim0)/(wall_s-wall0) if wall_s > wall0 else None
        return self.value
