"""G20 byte commands and URDF radians, implemented from public HandCore contracts.

Public data: linker-bot/linkerhand-telop-ros2
12300b1e45b263a253cc88fdb7635f95587d0741,
src/linkerhand_retarget/linkerhand_retarget/config/hand_config.yml and
assets/robots/hands/linker_hand/g20_{left,right}/linkerhand_g20_{left,right}.urdf.
Conversion basis: linkerhand/handcore.py:236-265 and utils.py:147-152.
MIT, Copyright (c) 2024 Linker Robotics; full notice in THIRD_PARTY_NOTICES.md.

Project configuration: endpoint directions and left-thumb teleop reversal define
motor decoding; the default motor pose supplies reserved slots during encoding.
Equal input endpoints select the output lower endpoint. Nonfinite inputs retain
the historical scalar clamp behavior; this API adds no finite-value validation.
"""
from collections.abc import Mapping, Sequence

from dexstation.examples.ar5_g20.model import G20_JOINT_NAMES as _CANONICAL_JOINT_NAMES

G20_MOTOR_COUNT = 20

G20_JOINT_NAMES = ['thumb_cmc_roll',
 'thumb_cmc_yaw',
 'thumb_cmc_pitch',
 'thumb_mcp',
 'thumb_ip',
 'index_mcp_roll',
 'index_mcp_pitch',
 'index_pip',
 'index_dip',
 'middle_mcp_roll',
 'middle_mcp_pitch',
 'middle_pip',
 'middle_dip',
 'ring_mcp_roll',
 'ring_mcp_pitch',
 'ring_pip',
 'ring_dip',
 'pinky_mcp_roll',
 'pinky_mcp_pitch',
 'pinky_pip',
 'pinky_dip']

G20_DEFAULT_MOTOR_POSITION = [255.0,
 255.0,
 255.0,
 255.0,
 255.0,
 255.0,
 193.0,
 148.0,
 105.0,
 42.0,
 245.0,
 255.0,
 255.0,
 255.0,
 255.0,
 255.0,
 255.0,
 255.0,
 255.0,
 255.0]

LOGICAL_SLOT_FROM_JOINT = {'thumb_cmc_pitch': 18,
 'index_mcp_pitch': 1,
 'middle_mcp_pitch': 9,
 'ring_mcp_pitch': 13,
 'pinky_mcp_pitch': 5,
 'thumb_cmc_roll': 16,
 'index_mcp_roll': 0,
 'middle_mcp_roll': 8,
 'ring_mcp_roll': 12,
 'pinky_mcp_roll': 4,
 'thumb_cmc_yaw': 17,
 'thumb_mcp': 19,
 'index_pip': 2,
 'middle_pip': 10,
 'ring_pip': 14,
 'pinky_pip': 6,
 'thumb_ip': 20,
 'index_dip': 3,
 'middle_dip': 11,
 'ring_dip': 15,
 'pinky_dip': 7}

G20_COMMAND_LOWER = {'right': [255,
           255,
           255,
           255,
           255,
           255,
           255,
           255,
           255,
           255,
           255,
           None,
           None,
           None,
           None,
           255,
           255,
           255,
           255,
           255],
 'left': [255,
          255,
          255,
          255,
          255,
          255,
          0,
          0,
          0,
          0,
          255,
          None,
          None,
          None,
          None,
          255,
          255,
          255,
          255,
          255]}

G20_COMMAND_UPPER = {'right': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, 0, 0, 0, 0, 0],
 'left': [0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, None, None, None, None, 0, 0, 0, 0, 0]}

G20_COMMAND_SOURCE_INDEX = {'right': [18, 1, 9, 13, 5, 16, 0, 8, 12, 4, 17, None, None, None, None, 19, 2, 10, 14, 6],
 'left': [18, 1, 9, 13, 5, 16, 0, 8, 12, 4, 17, None, None, None, None, 19, 2, 10, 14, 6]}

G20_URDF_INDEX = {'right': [2, 6, 10, 14, 18, 0, 5, 9, 13, 17, 1, None, None, None, None, 3, 7, 11, 15, 19],
 'left': [2, 6, 10, 14, 18, 0, 5, 9, 13, 17, 1, None, None, None, None, 3, 7, 11, 15, 19]}

G20_JOINT_LIMITS = {'right': {'thumb_cmc_roll': (0.0, 1.39),
           'thumb_cmc_yaw': (0.0, 1.57),
           'thumb_cmc_pitch': (0.0, 0.83),
           'thumb_mcp': (0.0, 1.25),
           'thumb_ip': (0.0, 1.29),
           'index_mcp_roll': (-0.23, 0.23),
           'index_mcp_pitch': (0.0, 1.22),
           'index_pip': (0.0, 1.75),
           'index_dip': (0.0, 1.55),
           'middle_mcp_roll': (-0.23, 0.23),
           'middle_mcp_pitch': (0.0, 1.22),
           'middle_pip': (0.0, 1.75),
           'middle_dip': (0.0, 1.55),
           'ring_mcp_roll': (-0.23, 0.23),
           'ring_mcp_pitch': (0.0, 1.22),
           'ring_pip': (0.0, 1.75),
           'ring_dip': (0.0, 1.55),
           'pinky_mcp_roll': (-0.23, 0.23),
           'pinky_mcp_pitch': (0.0, 1.22),
           'pinky_pip': (0.0, 1.75),
           'pinky_dip': (0.0, 1.55)},
 'left': {'thumb_cmc_roll': (0.0, 1.4),
          'thumb_cmc_yaw': (0.0, 1.57),
          'thumb_cmc_pitch': (0.0, 0.84),
          'thumb_mcp': (0.0, 1.26),
          'thumb_ip': (0.0, 1.29),
          'index_mcp_roll': (-0.23, 0.23),
          'index_mcp_pitch': (0.0, 1.22),
          'index_pip': (0.0, 1.74),
          'index_dip': (0.0, 1.55),
          'middle_mcp_roll': (-0.23, 0.23),
          'middle_mcp_pitch': (0.0, 1.22),
          'middle_pip': (0.0, 1.74),
          'middle_dip': (0.0, 1.55),
          'ring_mcp_roll': (-0.23, 0.23),
          'ring_mcp_pitch': (0.0, 1.22),
          'ring_pip': (0.0, 1.74),
          'ring_dip': (0.0, 1.55),
          'pinky_mcp_roll': (-0.23, 0.23),
          'pinky_mcp_pitch': (0.0, 1.22),
          'pinky_pip': (0.0, 1.74),
          'pinky_dip': (0.0, 1.55)}}

G20_MIMIC_JOINTS = {'right': {'thumb_ip': ('thumb_mcp', 1.03),
           'index_dip': ('index_pip', 0.89),
           'middle_dip': ('middle_pip', 0.89),
           'ring_dip': ('ring_pip', 0.89),
           'pinky_dip': ('pinky_pip', 0.89)},
 'left': {'thumb_ip': ('thumb_mcp', 1.02),
          'index_dip': ('index_pip', 0.89),
          'middle_dip': ('middle_pip', 0.89),
          'ring_dip': ('ring_pip', 0.89),
          'pinky_dip': ('pinky_pip', 0.89)}}

JOINT_FROM_LOGICAL_SLOT = dict(zip(LOGICAL_SLOT_FROM_JOINT.values(), LOGICAL_SLOT_FROM_JOINT))
TELEOP_MOTOR_REVERSE = {"left": (5,), "right": ()}
if tuple(G20_JOINT_NAMES) != _CANONICAL_JOINT_NAMES:
    raise RuntimeError("station G20 joint order differs from dexstation.examples.ar5_g20.model")


def _validate_hand(hand):
    if hand not in ("left", "right"):
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")


def _clamp(value, lower, upper):
    # Ordered comparisons also preserve the established NaN and inverted-bound behavior.
    capped = min(upper, value)
    return max(lower, capped)


def _scale(value, in_lower, in_upper, out_lower, out_upper):
    span = in_upper - in_lower
    if in_upper == in_lower:
        return out_lower
    numerator = (value - in_lower) * (out_upper - out_lower)
    return numerator / span + out_lower


def teleop_motor_positions(hand: str, motor_positions: Sequence[float]) -> list[float]:
    """Apply the project's left-thumb convention to a teleop motor packet."""
    _validate_hand(hand)
    result = list(map(float, motor_positions))
    for slot in TELEOP_MOTOR_REVERSE[hand]:
        result[slot] = 255.0 - result[slot]
    return result


def _apply_mimic(hand, logical_qpos):
    for name, (driver, ratio) in G20_MIMIC_JOINTS[hand].items():
        logical_qpos[LOGICAL_SLOT_FROM_JOINT[name]] = _clamp(
            logical_qpos[LOGICAL_SLOT_FROM_JOINT[driver]] * ratio,
            *G20_JOINT_LIMITS[hand][name])


def motor_positions_to_joint_positions(hand: str, motor_positions: Sequence[float],
                                       joint_names: Sequence[str] = G20_JOINT_NAMES) -> list[float]:
    """Decode the 16 active motor slots, then fill the five coupled URDF joints."""
    _validate_hand(hand)
    count = len(motor_positions)
    if count < G20_MOTOR_COUNT:
        raise ValueError(f"G20 motor command requires {G20_MOTOR_COUNT} values, got {count}")
    logical = [0.0 for _ in LOGICAL_SLOT_FROM_JOINT]
    channels = zip(G20_COMMAND_SOURCE_INDEX[hand], G20_URDF_INDEX[hand],
                   G20_COMMAND_LOWER[hand], G20_COMMAND_UPPER[hand])
    for motor, (slot, joint, start, end) in enumerate(channels):
        if None in (slot, joint, start, end):
            continue
        bounded = _clamp(float(motor_positions[motor]), min(start, end), max(start, end))
        name = G20_JOINT_NAMES[joint]
        logical[slot] = _scale(bounded, float(start), float(end), *G20_JOINT_LIMITS[hand][name])
    _apply_mimic(hand, logical)
    return [_clamp(logical[LOGICAL_SLOT_FROM_JOINT[name]], *G20_JOINT_LIMITS[hand][name])
            for name in joint_names]


def apply_mimic_joint_positions(hand: str, joint_positions: Sequence[float],
                                joint_names: Sequence[str] = G20_JOINT_NAMES) -> list[float]:
    """Return the requested named order with coupled joints recomputed."""
    _validate_hand(hand)
    positions = dict(zip(joint_names, map(float, joint_positions)))
    for name, (driver, ratio) in G20_MIMIC_JOINTS[hand].items():
        positions[name] = _clamp(positions.get(driver, 0.0) * ratio, *G20_JOINT_LIMITS[hand][name])
    return [_clamp(positions.get(name, 0.0), *G20_JOINT_LIMITS[hand][name]) for name in joint_names]


def joint_positions_to_motor_positions(hand: str, joint_positions: Sequence[float] | Mapping[str, float],
                                       joint_names: Sequence[str] = G20_JOINT_NAMES) -> list[float]:
    """Encode float motor values; integer truncation belongs to HandCore's API."""
    _validate_hand(hand)
    entries = joint_positions.items() if isinstance(joint_positions, Mapping) else zip(joint_names, joint_positions)
    positions = {name: float(value) for name, value in entries}
    result = G20_DEFAULT_MOTOR_POSITION.copy()
    for motor, (slot, start, end) in enumerate(zip(G20_COMMAND_SOURCE_INDEX[hand],
                                                G20_COMMAND_LOWER[hand], G20_COMMAND_UPPER[hand])):
        if None in (slot, start, end):
            continue
        name = JOINT_FROM_LOGICAL_SLOT[slot]
        low, high = G20_JOINT_LIMITS[hand][name]
        radians = _clamp(positions.get(name, 0.0), low, high)
        result[motor] = _clamp(_scale(radians, low, high, float(start), float(end)),
                               min(start, end), max(start, end))
    return result
