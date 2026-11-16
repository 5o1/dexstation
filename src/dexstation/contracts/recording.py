"""Layered recording contracts and memory-only read/write fakes (§7).

read_episode accepts an in-memory Episode or MemoryEpisodeWriter here. Filesystem
reading, integrity recovery and real writer implementation belong to WP4.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from dexstation.contracts.backend import StepObservation

from dexstation.contracts.versions import EPISODE_FORMAT, WORKBENCH_FORMAT
FILES = {"episode": "episode.json", "meta": "meta.json", "steps": "steps.jsonl",
         "commands": "commands.jsonl", "events": "events.jsonl", "integrity": "integrity.json"}
WORKBENCH_ROW_KEYS = ("step", "sim_time_s", "wall_time_ns", "hand_q", "arm_q", "palm_pose_world",
                      "fingertip_pose_world", "robot_contact", "commands", "applied_monotonic_ns",
                      "input_sequences", "arm_clamps", "motion_reference")


@runtime_checkable
class EpisodeWriterProtocol(Protocol):
    def append(self, obs: StepObservation, command_state: dict, *, applied_monotonic_ns, input_sequences,
               arm_clamps, motion_reference=None) -> None: ...
    def append_inputs(self, events: list[dict]) -> None: ...
    def event(self, layer: str, kind: str, payload: dict) -> None: ...
    def note(self, text: str) -> None: ...
    def close(self, status: str = "complete") -> None: ...


@dataclass
class EpisodeLayer:
    format: str
    rows: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Episode:
    workbench: EpisodeLayer
    task: EpisodeLayer | None
    meta: dict
    inputs: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def joined(self) -> list[dict]:
        """Fake reader: rows contain step/time plus nested workbench and task.

        Lookup is by step, not list position; task=None for empty. Nested layers
        avoid inventing a cross-layer channel name collision policy.
        """
        workbench = {row["step"]: row for row in self.workbench.rows}
        task = {row["step"]: row for row in self.task.rows} if self.task else None
        if len(workbench) != len(self.workbench.rows) or (self.task and len(task) != len(self.task.rows)):
            raise ValueError("Duplicate step in episode layer")
        if task is not None and task.keys() != workbench.keys():
            raise ValueError("Episode layers have different steps")
        result = []
        for step, row in sorted(workbench.items()):
            other = task[step] if task is not None else None
            if other is not None and other["sim_time_s"] != row["sim_time_s"]:
                raise ValueError("Episode layer simulation times differ")
            result.append(deepcopy({"step": step, "sim_time_s": row["sim_time_s"],
                                    "workbench": row, "task": other}))
        return result


class MemoryEpisodeWriter:
    """In-memory test double; metadata is minimal, not a production manifest.

    Non-empty tasks require their own explicit format. Writes copy inputs and
    observations, without creating files or performing production recovery.
    """

    def __init__(self, *, task_id: str = "empty", task_format: str | None = None):
        if task_id != "empty" and task_format is None:
            raise ValueError("Non-empty task needs a recording format")
        self.episode = Episode(EpisodeLayer(WORKBENCH_FORMAT),
                               EpisodeLayer(task_format) if task_id != "empty" else None,
                               {"format": EPISODE_FORMAT, "task_id": task_id, "status": "recording", "rows": 0})
        self.closed = False

    def append(self, obs: StepObservation, command_state: dict, *, applied_monotonic_ns, input_sequences,
               arm_clamps, motion_reference=None) -> None:
        if self.closed:
            raise ValueError("Episode writer is closed")
        rows = self.episode.workbench.rows
        if rows and obs.robot.sim_time_s <= rows[-1]["sim_time_s"]:
            raise ValueError("Simulation timestamps must increase")
        if self.episode.task is None and obs.task:
            raise ValueError("Empty task cannot record task channels")
        if {"step", "sim_time_s", "wall_time_ns"}.intersection(obs.task):
            raise ValueError("Task channels cannot replace recording clock fields")
        step = len(rows)
        row = {"step": step, **asdict(obs.robot), "commands": command_state,
               "applied_monotonic_ns": applied_monotonic_ns, "input_sequences": input_sequences,
               "arm_clamps": arm_clamps, "motion_reference": motion_reference}
        rows.append(deepcopy(row))
        if self.episode.task is not None:
            self.episode.task.rows.append(deepcopy({"step": step, "sim_time_s": obs.robot.sim_time_s, **obs.task}))
        self.episode.meta["rows"] = len(rows)

    def append_inputs(self, events: list[dict]) -> None:
        self.episode.inputs.extend(deepcopy(events))

    def event(self, layer: str, kind: str, payload: dict) -> None:
        if layer not in ("workbench", "task"):
            raise ValueError("Unknown recording layer")
        target = getattr(self.episode, layer)
        if target is None:
            raise ValueError("Empty task has no task layer")
        target.events.append(deepcopy({"kind": kind, "payload": payload}))

    def note(self, text: str) -> None:
        self.episode.notes.append(text)

    def close(self, status: str = "complete") -> None:
        self.episode.meta["status"] = status
        self.closed = True


def read_episode(path: str | Path | Episode | MemoryEpisodeWriter) -> Episode:
    """Memory reader only; WP4 provides read_episode(path) for real episodes."""
    if isinstance(path, MemoryEpisodeWriter):
        return deepcopy(path.episode)
    if isinstance(path, Episode):
        return deepcopy(path)
    raise NotImplementedError("Filesystem episode reading belongs to WP4")
