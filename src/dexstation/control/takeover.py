"""Arm alignment protocol shared by simulation and future hardware callers.

Reviewer ruling 1: Stop is a session action. The controller stops calling update
and discards this object; a new session constructs both arms in WAITING. There
are no hardware-only states. Caller-owned freshness is passed as target=None.
Hands do not align; workspace_limits owns their collision/hold decisions.
"""
from dexstation.contracts.versions import TAKEOVER_SNAPSHOT_VERSION
import copy
import json
from pathlib import Path

import numpy as np

from dexstation.contracts.takeover import ArmDecision, ArmState


class ArmTakeover:
    def __init__(self, parameters):
        self.parameters = copy.deepcopy(parameters)
        for key in ('match_tolerance_deg', 'match_dwell_s', 'deviation_budget_deg'):
            if parameters.get(key) is None:
                raise ValueError(f'takeover not configured: {key} is null')
        self.tolerance = self._threshold(parameters['match_tolerance_deg'])
        self.budget = self._threshold(parameters['deviation_budget_deg'])
        self.dwell = float(parameters['match_dwell_s'])
        if not np.isfinite(self.dwell) or self.dwell < 0:
            raise ValueError('match_dwell_s must be finite and nonnegative')
        self._arms = {side: dict(state='waiting', aligned_since=None, time=None)
                      for side in ('left', 'right')}

    @staticmethod
    def _threshold(value):
        result = np.broadcast_to(np.asarray(value, float), (7,)).copy()
        if not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError('takeover thresholds must be finite and nonnegative')
        return result

    def update(self, t, side, master_target_rad, current_reference_rad, limit_ok):
        arm = self._arms[side]
        t = float(t)
        if not np.isfinite(t) or (arm['time'] is not None and t < arm['time']):
            raise ValueError('takeover timestamps must be finite and monotonic per side')
        arm['time'] = t
        if master_target_rad is None:
            arm.update(state='waiting', aligned_since=None)
            return ArmDecision(ArmState.WAITING, False, [], 'input_not_fresh')
        target, reference = (np.asarray(q, float) for q in (master_target_rad, current_reference_rad))
        if target.shape != (7,) or reference.shape != (7,) or not np.isfinite([target, reference]).all():
            raise ValueError('takeover requires two finite 7-vectors')
        # Factory-limited joints are not continuous; do not wrap differences.
        deviation = np.degrees(target - reference)
        state = ArmState(arm['state'])
        reason = None
        if not limit_ok:
            state, reason = ArmState.HOLD, 'limit'
            arm['aligned_since'] = None
        elif state is ArmState.FOLLOWING and np.any(np.abs(deviation) > self.budget):
            state, reason = ArmState.HOLD, 'deviation_budget'
            arm['aligned_since'] = None
        elif state is not ArmState.FOLLOWING:
            # Hold remains observable for the triggering cycle; recovery always
            # starts a new matching dwell, including a configured zero dwell.
            was_hold = state is ArmState.HOLD
            state = ArmState.MATCHING
            if np.all(np.abs(deviation) <= self.tolerance):
                if arm['aligned_since'] is None:
                    arm['aligned_since'] = t
                if not was_hold and t - arm['aligned_since'] >= self.dwell:
                    state = ArmState.FOLLOWING
            else:
                arm['aligned_since'] = None
        arm['state'] = state.value
        return ArmDecision(state, state is ArmState.FOLLOWING, deviation.tolist(), reason)

    def snapshot(self):
        return dict(version=TAKEOVER_SNAPSHOT_VERSION, parameters=copy.deepcopy(self.parameters), arms=copy.deepcopy(self._arms))

    def restore(self, state):
        if state['version'] != TAKEOVER_SNAPSHOT_VERSION or state['parameters'] != self.parameters:
            raise ValueError('takeover snapshot parameters/version mismatch')
        for side in ('left', 'right'):
            ArmState(state['arms'][side]['state'])
        self._arms = copy.deepcopy(state['arms'])


def load_takeover(path):
    return ArmTakeover(json.loads(Path(path).read_text()))
