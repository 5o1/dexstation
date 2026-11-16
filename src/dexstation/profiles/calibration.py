"""Pure measured calibration transforms for the bench robot contract."""
import json
from pathlib import Path

from dexstation.model.constants import SIDES
from dexstation.model.mount_geometry import j1_offset_deg


def load_teleop_calibration(path, spec) -> dict[str, list[float]]:
    """Derive teleop -> sim J1 offsets from the current and reference mounts."""
    data = json.loads(Path(path).read_text())
    offsets = {}
    for side in SIDES:
        arm = getattr(spec, side + '_arm', None)
        if arm is None:
            offsets[side] = [0.] * 7
            continue
        if arm.base_link not in data['arm_base_links'][side]:
            raise ValueError(f'calibration does not cover arm model {arm.base_link}; recalibrate')
        offsets[side] = [j1_offset_deg(arm.base_pose[3:], data['reference_base_orientation_wxyz'][side]),
                         *([0.] * 6)]
    return offsets


def load_calibration(path, spec):
    """Derive real robot -> sim J1 offsets; retain the measured hand rules."""
    data = json.loads(Path(path).read_text())
    arm = {}
    for side in SIDES:
        scene_arm = getattr(spec, side + '_arm', None)
        if scene_arm is None:
            arm[side] = [0.] * 7
            continue
        if scene_arm.base_link not in data['arm_base_links'][side]:
            raise ValueError(f'calibration does not cover arm model {scene_arm.base_link}; recalibrate')
        arm[side] = [j1_offset_deg(scene_arm.base_pose[3:], data['measured_base_orientation_wxyz'][side]),
                     *([0.] * 6)]
    hand = {side: sorted(int(i) for i in data['hand_motor_reverse'][side]) for side in SIDES}
    if any(len(v) != 7 for v in arm.values()) or any(not 0 <= i < 20 for v in hand.values() for i in v):
        raise ValueError('calibration needs 7 arm offsets and hand slots 0-19 per side')
    return dict(arm=arm, hand=hand)
