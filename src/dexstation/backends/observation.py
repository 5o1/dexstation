"""Observation adapter used by the maintained SAPIEN renderer."""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Observation:
    stamp: float
    hand_joint_positions: dict[str, float]
    arm_joint_positions: tuple[float, ...] | None
    hand_root_pose_world: tuple[float, ...] | None
    gamepad_signal: np.ndarray | None
    gamepad_mask: np.ndarray | None
    backend: str
    extra: dict[str, "Observation"] = field(default_factory=dict)
