"""Project motion equations in Python/numpy; no compiler or device dependencies.

Units follow LinkerTA TeleArmNode.cpp:164-189 at 109766554f56728da458aac6d74780d30378d1b1
(linker-bot/linkerbot-teleop-arm) and the radian command interface in
RokaeRobot/xCoreSDK-CPP, 061c2fb11f360f1585aae2776a4bb713b9e7c49c,
example/rt/joint_impedance_control.cpp:60-96. The scalar RC equation is
independently implemented from the first-order low-pass equation documented by
frankarobotics/libfranka, e88d08f64cbf842b8f53c0b551963684064038d6,
src/lowpass_filter.cpp:11-29. No SDK implementation or validation code is copied.
Project offsets/limits are used for degree targets; ramp uses a 1 ms cycle,
a 5% minimum ratio and nanosecond-truncated elapsed time for replay continuity.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from dexstation.model.constants import LOW_DEG, HIGH_DEG

ASSETS = Path(__file__).parents[1] / 'examples/ar5_g20/motion_assets'


class StationMath:
    def __init__(self, cache=None):
        # Keep the cache argument for existing callers; Python needs no cache files.
        self.digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    @staticmethod
    def vector(value):
        result = np.ascontiguousarray(value, dtype=np.float64)
        if result.shape != (7,) or not np.isfinite(result).all():
            raise ValueError('expected seven finite joint values')
        return result

    def map(self, side, raw, margin=3.):
        if side not in ('left', 'right'):
            raise ValueError('unknown arm side')
        values = self.vector(raw).copy()
        values[0] += 90. if side == 'left' else -90.
        values[1] += 90.
        values[2] += -90. if side == 'left' else 90.
        # Ordered bounds match scalar min(high, value), then max(low, ...).
        return np.maximum(LOW_DEG + margin, np.minimum(HIGH_DEG - margin, values))

    def step(self, current, target, max_step):
        start, goal = self.vector(current), self.vector(target)
        delta = np.maximum(-max_step, np.minimum(max_step, goal - start))
        return start + delta

    def difference(self, a, b):
        return float(np.max(np.abs(self.vector(a) - self.vector(b))))

    def ramp_step_deg(self, speed, ramp, elapsed):
        if ramp <= 0. or elapsed >= ramp:
            return float(speed * .001)
        seconds = int(elapsed * 1_000_000_000) / 1_000_000_000
        ratio = max(.05, min(1., seconds / ramp))
        return float(speed * ratio * .001)

    def lowpass(self, dt, value, previous, cutoff):
        # numpy division retains the old scalar ABI's IEEE results at zero denominators.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            period = np.float64(dt)
            coefficient = period / (period + 1. / (2. * math.pi * np.float64(cutoff)))
            output = coefficient * value + (1. - coefficient) * previous
        return float(output)

    def provenance(self):
        return {'code_sha256': self.digest, 'clone': json.loads((ASSETS / 'provenance.json').read_text())}
