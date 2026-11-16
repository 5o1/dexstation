"""Backend protocol and a deterministic, non-physical in-memory fake (§9)."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np



@dataclass
class RobotObservation:
    sim_time_s: float
    wall_time_ns: int
    hand_q: np.ndarray
    arm_q: np.ndarray
    palm_pose_world: np.ndarray
    fingertip_pose_world: np.ndarray
    robot_contact: bool


@dataclass
class StepObservation:
    robot: RobotObservation
    task: dict[str, object]
    object_poses: dict[str, list[float]]


@runtime_checkable
class CollectionBackend(Protocol):
    name: str
    timestep_s: float
    timestep_source: str
    solver_record: dict
    motion: object | None

    def reset(self) -> None: ...
    def apply(self, commands: object) -> None: ...
    def step(self, dt: float) -> None: ...
    def observe(self) -> StepObservation: ...
    def contact_force(self, side: str, robot_link: str, obj: str, obj_link: str | None = None) -> np.ndarray: ...
    def set_object_state(self, name: str, pose, twist=None) -> None: ...
    def snapshot_motion_state(self) -> dict: ...
    def restore_motion_state(self, state: dict) -> None: ...
    def close(self) -> None: ...


class FakeBackend:
    """Stores supplied observations/commands; step only advances the fake clock.

    Default robot arrays retain the existing bimanual SI/world xyz-wxyz shapes.
    No joint integration, mapping, contact, rendering, or hardware is performed.
    """

    name = "fake"
    timestep_s = 0.001
    timestep_source = "fake"

    def __init__(self, observation: StepObservation | None = None):
        if observation is None:
            pose = np.array([0., 0., 0., 1., 0., 0., 0.])
            observation = StepObservation(RobotObservation(
                0., 0, np.zeros((2, 21)), np.zeros((2, 7)),
                np.tile(pose, (2, 1)), np.tile(pose, (2, 5, 1)), False,
            ), {}, {})
        self._initial = deepcopy(observation)
        self.solver_record: dict = {}
        self.motion = None
        self.closed = False
        self.reset()

    def reset(self) -> None:
        self._observation = deepcopy(self._initial)
        self.commands = None
        self.object_twists: dict = {}

    def apply(self, commands: object) -> None:
        self.commands = deepcopy(commands)

    def step(self, dt: float) -> None:
        self._observation.robot.sim_time_s += dt

    def observe(self) -> StepObservation:
        return deepcopy(self._observation)

    def contact_force(self, side: str, robot_link: str, obj: str, obj_link: str | None = None) -> np.ndarray:
        return np.zeros(3)

    def set_object_state(self, name: str, pose, twist=None) -> None:
        self._observation.object_poses[name] = list(deepcopy(pose))
        self.object_twists[name] = deepcopy(twist)

    def snapshot_motion_state(self) -> dict:
        return deepcopy({
            "workbench": {"robot": self._observation.robot, "commands": self.commands},
            "task": {"channels": self._observation.task, "object_poses": self._observation.object_poses,
                     "object_twists": self.object_twists},
        })

    def restore_motion_state(self, state: dict) -> None:
        state = deepcopy(state)
        self._observation = StepObservation(state["workbench"]["robot"], state["task"]["channels"],
                                            state["task"]["object_poses"])
        self.commands = state["workbench"]["commands"]
        self.object_twists = state["task"]["object_twists"]

    def close(self) -> None:
        self.closed = True
