"""Project degree-to-radian arm mapping using public SDK unit contracts.

LinkerTA legacy positions are degrees (linker-bot/linkerbot-teleop-arm,
109766554f56728da458aac6d74780d30378d1b1, TeleArmNode.cpp:164-189).
xCoreSDK accepts radians (RokaeRobot/xCoreSDK-CPP,
061c2fb11f360f1585aae2776a4bb713b9e7c49c,
example/rt/joint_impedance_control.cpp:60-96).
Offsets, margins, startup neighborhoods and calibration are project configuration.
"""
import numpy as np

from dexstation.control.commands import ArmCommand
from dexstation.profiles.calibration import load_teleop_calibration
from dexstation.model.constants import LOW_DEG, HIGH_DEG
from dexstation.contracts.versions import ARM_MAPPING_VERSION

VERSION = ARM_MAPPING_VERSION
OFFSETS = {
    'left': np.asarray([90., 90., -90., 0., 0., 0., 0.]),
    'right': np.asarray([-90., 90., 90., 0., 0., 0., 0.]),
}


class ArmMapper:
    def __init__(self, startup_rad, *, margin_deg=3., max_delta_deg=60., limit_around_initial=True,
                 startup="spec", station_math=None, joint_offset_deg=None):
        if startup not in ('spec', 'first_command', 'none'):
            raise ValueError("arm startup must be 'spec', 'first_command' or 'none'")
        self.startup = startup
        self.station_math = station_math
        self.initial = {}
        self.joint_offset = {}
        for side in OFFSETS:
            self.initial[side] = None if startup == 'first_command' else np.rad2deg(self._vector(startup_rad[side]))
            self.joint_offset[side] = np.zeros(7) if joint_offset_deg is None else self._vector(joint_offset_deg[side])
        self.margin = float(margin_deg)
        self.max_delta = float(max_delta_deg)
        self.limit_around_initial = bool(limit_around_initial) if startup != 'none' else False
        if not (0 <= self.margin < 60) or not np.isfinite(self.max_delta) or self.max_delta < 0:
            raise ValueError('invalid arm mapping limits')

    @staticmethod
    def _vector(value):
        result = np.asarray(value, dtype=float)
        if result.shape != (7,) or not np.isfinite(result).all():
            raise ValueError('arm input must contain 7 finite values')
        return result

    def command(self, side, raw_deg, stamp):
        raw = self._vector(raw_deg)
        desired = raw + OFFSETS[side]
        lower, upper = LOW_DEG + self.margin, HIGH_DEG - self.margin
        target = np.clip(desired, lower, upper)
        if self.station_math is not None:
            target = self.station_math.map(side, raw, self.margin)
        correction = self.joint_offset[side]
        if correction.any():
            desired = desired + correction
            target = np.clip(target + correction, lower, upper)
        absolute_flags = (target != desired).tolist()
        if self.initial[side] is None:
            self.initial[side] = target.copy()
        absolute_target = target.copy()
        if self.limit_around_initial:
            center = self.initial[side]
            target = np.clip(target, center - self.max_delta, center + self.max_delta)
        flags = {'absolute': absolute_flags, 'startup': (target != absolute_target).tolist()}
        return ArmCommand(side, tuple(np.deg2rad(target)), float(stamp), tuple(raw), flags)

    def metadata(self):
        homes = {side: None if degrees is None else np.deg2rad(degrees).tolist()
                 for side, degrees in self.initial.items()}
        return {'version': VERSION, 'startup': self.startup, 'startup_rad': homes,
                'margin_deg': self.margin, 'max_delta_deg': self.max_delta,
                'limit_around_initial': self.limit_around_initial}
