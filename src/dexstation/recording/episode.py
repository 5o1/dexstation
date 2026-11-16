"""Layered collection: SI units, world poses in xyz/wxyz, left then right.

Unavailable original commands use null, never reconstructed measurements.
Fingertips use distal-frame orientation and the collision mesh's +Z extremum
at local x=y=0; the resulting offsets are recorded in episode metadata.
"""
from dataclasses import dataclass, asdict, is_dataclass
import math
import re
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from dexstation.contracts.backend import RobotObservation, StepObservation
from dexstation.contracts.recording import EPISODE_FORMAT, WORKBENCH_FORMAT, FILES, WORKBENCH_ROW_KEYS, Episode, EpisodeLayer

from dexstation.model.constants import SIDES
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class Observation:
    sim_time_s: float
    wall_time_ns: int
    hand_q: np.ndarray
    arm_q: np.ndarray
    palm_pose_world: np.ndarray
    fingertip_pose_world: np.ndarray
    gamepad_signal: np.ndarray
    gamepad_mask: np.ndarray
    control_joint_positions: dict
    robot_contact: bool
    backend: str

    def validate(self):
        for name, shape in (("hand_q", (2, 21)), ("arm_q", (2, 7)),
                            ("palm_pose_world", (2, 7)), ("fingertip_pose_world", (2, 5, 7)),
                            ("gamepad_signal", (32,)), ("gamepad_mask", (32,))):
            a = np.asarray(getattr(self, name))
            if a.shape != shape or (name != "gamepad_mask" and not np.isfinite(a).all()):
                raise ValueError(f"invalid {name}: expected finite {shape}")
        for a in (self.palm_pose_world, self.fingertip_pose_world):
            if not np.allclose(np.linalg.norm(np.asarray(a)[..., 3:], axis=-1), 1, atol=1e-5):
                raise ValueError("world pose quaternion must have unit norm")
        if not np.isfinite(self.sim_time_s) or self.sim_time_s < 0 or self.wall_time_ns <= 0:
            raise ValueError("invalid observation clock")
        if not all(np.isfinite(v) for v in self.control_joint_positions.values()):
            raise ValueError("nonfinite control displacement")

    def to_dict(self):
        self.validate()
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(self).items()}


def fingertip_offsets(spec):
    import trimesh
    from dexstation.model.kinematics import rpy_to_matrix
    result = {}
    for side in SIDES:
        urdf = Path(getattr(spec, side + "_hand").urdf)
        root = ET.parse(urdf).getroot()
        result[side] = {}
        for finger in FINGERS:
            col = root.find(f"link[@name='{finger}_distal']/collision")
            mesh = col.find("geometry/mesh")
            vertices = np.asarray(trimesh.load(urdf.parent / mesh.get("filename"), force="mesh").vertices).copy()
            vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            origin = col.find("origin")
            if origin is not None:
                vertices = vertices @ rpy_to_matrix(np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).T
                vertices += np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            result[side][finger] = [0., 0., float(vertices[:, 2].max())]
    return result


def tip_pose(link_pose, offset):
    from dexstation.model.spec import pose_to_matrix
    pose = np.asarray(link_pose, float).copy()
    pose[:3] += pose_to_matrix(pose.tolist())[:3, :3] @ offset
    if pose[3] < 0:
        pose[3:] *= -1
    return pose


def _json(value):
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"not JSON serializable: {type(value).__name__}")
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      default=encode,
                      separators=(",", ":"))


def _store_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def as_step_observation(observation, *, include_task=True):
    """Bridge the current controller's observation API, never an old disk format."""
    if isinstance(observation, StepObservation):
        return observation
    row = observation.to_dict()
    robot = RobotObservation(**{key: row[key] for key in RobotObservation.__dataclass_fields__})
    task = {key: row[key] for key in ("gamepad_signal", "gamepad_mask", "control_joint_positions")} if include_task else {}
    return StepObservation(robot, task, {})


def _validate_robot(row):
    for name, shape in (("hand_q", (2, 21)), ("arm_q", (2, 7)),
                        ("palm_pose_world", (2, 7)), ("fingertip_pose_world", (2, 5, 7))):
        value = np.asarray(row[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {name}: expected finite {shape}")
    for name in ("palm_pose_world", "fingertip_pose_world"):
        if not np.allclose(np.linalg.norm(np.asarray(row[name])[..., 3:], axis=-1), 1, atol=1e-5):
            raise ValueError("world pose quaternion must have unit norm")
    if not math.isfinite(row["sim_time_s"]) or row["sim_time_s"] < 0 or row["wall_time_ns"] < 0:
        raise ValueError("invalid observation clock")
    flags = row["arm_clamps"]
    if set(flags) != set(SIDES):
        raise ValueError("invalid arm clamp sides")
    for side in SIDES:
        if flags[side] is not None:
            for kind in ("absolute", "startup"):
                mask = flags[side][kind]
                if len(mask) != 7 or any(type(v) is not bool for v in mask):
                    raise ValueError("invalid arm clamp flags")


class DiskEpisode(Episode):
    """WP0 Episode plus an in-memory adapter for unmigrated Python callers.

    Tuple unpacking/indexing exposes metadata and workbench rows for existing
    robot replay callers. Task data stays in .task or WP0's nested .joined();
    there is no cross-layer channel overwrite rule or old on-disk format.
    """
    def _compat(self):
        meta = dict(self.workbench.meta, **self.meta)
        meta["input_messages"] = self.workbench.meta["input_messages"]
        if "initial_state" in self.workbench.meta:
            meta["motion_initial_state"] = (dict(workbench=self.workbench.meta["initial_state"],
                task=self.task.meta.get("initial_state", {})) if self.task else self.workbench.meta["initial_state"])
        return meta, self.workbench.rows

    def __iter__(self):
        return iter(self._compat())

    def __getitem__(self, index):
        return self._compat()[index]


class EpisodeWriter:
    """Synchronous write of one snapshot to each layer (WP0 protocol).

    meta holds session/driver metadata for the current controller. New callers
    supply RuntimeScene via scene, or explicit identity, workbench_meta and
    task_meta. Non-empty tasks must declare their recording format, as in WP0.
    """
    def __init__(self, path, meta, *, scene=None, task_id=None, task_format=None,
                 workbench_meta=None, task_meta=None):
        self.path = Path(path)
        self.meta = copy.deepcopy(meta)
        self.workbench_meta = copy.deepcopy(workbench_meta or {})
        self.task_meta = copy.deepcopy(task_meta or {})
        if scene is not None:
            self.meta["identity"] = copy.deepcopy(scene.identity)
            task_id = scene.task["id"]
            self.workbench_meta.update(workbench_profile_text=scene.frozen["workbench_text"],
                geometry=asdict(scene.spec), calibration=scene.calibration,
                cameras=[asdict(pair) for pair in scene.cameras])
            self.task_meta.update(task_profile_text=scene.frozen["task_text"],
                task_profile=scene.task, channels=[asdict(channel) for channel in scene.channels])
        self.task_id = task_id or self.meta.get("identity", {}).get("task", {}).get("id", "empty")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.task_id):
            raise ValueError("invalid task id")
        self.task_format = task_format or self.task_meta.get("format")
        if self.task_id != "empty" and not self.task_format:
            raise ValueError("Non-empty task needs a recording format")
        variant = self.meta.get("session", {}).get("scene_variant", self.meta.get("scene_variant", "full"))
        if isinstance(variant, dict):
            variant = variant["name"]
        if variant != "full":
            raise ValueError("only the full scene variant may be recorded")
        self.meta.update(status="recording", rows=0, input_messages=0)
        self.meta.setdefault("fps", 30)
        if not math.isfinite(self.meta["fps"]) or self.meta["fps"] <= 0:
            raise ValueError("fps must be positive")
        self.meta.setdefault("operator_notes", [])
        self.meta.setdefault("created_wall_time_ns", time.time_ns())
        self.meta.update(side_order=list(SIDES), finger_order=list(FINGERS),
                         pose_convention="world_from_frame:[x,y,z,qw,qx,qy,qz];m;rad")
        if "scene_spec_text" in self.meta:
            self.meta["scene_spec_sha256"] = _hash(self.meta["scene_spec_text"])
        for layer_meta, key in ((self.workbench_meta, "workbench_profile_text"),
                                (self.task_meta, "task_profile_text")):
            if key in layer_meta:
                layer_meta[key.replace("_text", "_sha256")] = _hash(layer_meta[key])
        self.path.mkdir(parents=True, exist_ok=False)
        self.layers = [{"name": "workbench", "format": WORKBENCH_FORMAT, "path": "workbench/"}]
        if self.task_id != "empty":
            self.layers.append({"name": "task", "task": self.task_id,
                                "format": self.task_format, "path": f"task/{self.task_id}/"})
        self.streams = {}
        self.event_streams = {}
        for layer in self.layers:
            directory = self.path / layer["path"]
            directory.mkdir(parents=True)
            self.streams[layer["name"]] = (directory / FILES["steps"]).open("x", encoding="utf-8", buffering=1)
            self.event_streams[layer["name"]] = (directory / FILES["events"]).open("x", encoding="utf-8", buffering=1)
        self.stream = self.streams["workbench"]
        self.input_stream = (self.path / "workbench" / FILES["commands"]).open("x", encoding="utf-8", buffering=1)
        self._last_input_sequence = self._last_receive_ns = -1
        self.closed = False
        self.last_sim_time = -1.
        self._write_meta()

    def _write_meta(self):
        session = copy.deepcopy(self.meta.get("session", {}))
        for key in ("input_source", "motion_profile", "retarget", "started_by", "stopped_by", "forced"):
            if key in self.meta:
                session[key] = self.meta[key]
        session["scene_variant"] = "full"
        manifest = dict(format=EPISODE_FORMAT, episode_id=self.path.parent.name + "~" + self.path.name,
            status=self.meta["status"], rows=self.meta["rows"], fps=self.meta["fps"], layers=self.layers,
            clock={"step": "record index from 0, identical in every layer",
                   "sim_time_s": "physics time of the completed step, identical in every layer",
                   "wall_time_ns": "workbench layer only"}, identity=self.meta.get("identity", {}),
            session=session, operator_notes=self.meta["operator_notes"],
            realtime_metrics=self.meta.get("realtime_metrics", {}))
        # Keep legacy controller metadata in the workbench journal, not in the
        # index. WP8a will supply full RuntimeScene provenance at construction.
        excluded = set(manifest) | {"session", "motion_initial_state", "gamepad_solver"}
        workbench = dict(self.workbench_meta, **{k: v for k, v in self.meta.items() if k not in excluded})
        assets = copy.deepcopy(workbench.get("asset_versions", {}))
        gamepad_version = assets.pop("gamepad", None)
        if "asset_versions" in workbench:
            workbench["asset_versions"] = assets
        workbench["format"] = WORKBENCH_FORMAT
        initial = self.meta.get("motion_initial_state")
        if initial is not None:
            workbench["initial_state"] = initial.get("workbench", initial)
        _store_json(self.path / "workbench" / FILES["meta"], workbench)
        if self.task_id != "empty":
            task = dict(self.task_meta, format=self.task_format)
            if gamepad_version is not None:
                task["asset_versions"] = dict(task.get("asset_versions", {}), gamepad=gamepad_version)
            if "gamepad_solver" in self.meta:
                task["gamepad_solver"] = self.meta["gamepad_solver"]
            if initial is not None and "task" in initial:
                task["initial_state"] = initial["task"]
            _store_json(self.path / "task" / self.task_id / FILES["meta"], task)
        _store_json(self.path / FILES["episode"], manifest)

    def append(self, obs, command_state, overlay=None, *, applied_monotonic_ns=None,
               input_sequences=None, arm_clamps=None, motion_reference=None):
        if self.closed:
            raise ValueError("Episode writer is closed")
        obs = as_step_observation(obs, include_task=self.task_id != "empty")
        if self.task_id == "empty" and obs.task:
            raise ValueError("Empty task cannot record task channels")
        if {"step", "sim_time_s", "wall_time_ns"}.intersection(obs.task):
            raise ValueError("Task channels cannot replace recording clock fields")
        row = dict(step=self.meta["rows"], **asdict(obs.robot), commands=command_state,
            applied_monotonic_ns=applied_monotonic_ns, input_sequences=dict(input_sequences or {}),
            arm_clamps=arm_clamps if arm_clamps is not None else dict.fromkeys(SIDES),
            motion_reference=motion_reference)
        _validate_robot(row)
        if obs.robot.sim_time_s <= self.last_sim_time:
            raise ValueError("simulation timestamps must increase")
        # Encode all layers before writing any bytes: invalid task values must
        # never create a half-written snapshot.
        encoded = {"workbench": _json(row)}
        if self.task_id != "empty":
            encoded["task"] = _json(dict(step=row["step"], sim_time_s=row["sim_time_s"], **obs.task))
        for layer, text in encoded.items():
            self.streams[layer].write(text + "\n")
        self.last_sim_time = obs.robot.sim_time_s
        self.meta["rows"] += 1

    def append_inputs(self, events):
        if self.closed:
            raise ValueError("Episode writer is closed")
        for event in events:
            if event["sequence"] <= self._last_input_sequence or event["received_monotonic_ns"] < self._last_receive_ns:
                raise ValueError("input events must preserve receive order")
            self.input_stream.write(_json(event) + "\n")
            self._last_input_sequence, self._last_receive_ns = event["sequence"], event["received_monotonic_ns"]
            self.meta["input_messages"] += 1

    def event(self, layer, kind, payload):
        if self.closed:
            raise ValueError("Episode writer is closed")
        if layer not in self.event_streams:
            raise ValueError("Unknown or absent recording layer")
        self.event_streams[layer].write(_json(dict(kind=kind, payload=payload)) + "\n")

    def note(self, text):
        if self.closed:
            raise ValueError("Episode writer is closed")
        note = {"wall_time_ns": time.time_ns(), "text": str(text)}
        self.meta["operator_notes"].append(note)
        self.event("workbench", "note", note)
        self._write_meta()

    def close(self, status="complete"):
        if not self.closed:
            for stream in [*self.streams.values(), *self.event_streams.values(), self.input_stream]:
                stream.close()
            self.meta["status"] = status
            self._write_meta()
            self.closed = True


def _read_jsonl(path, *, prefix=False):
    rows = []
    with path.open("rb") as stream:
        for line in stream:
            try:
                if not line.endswith(b"\n"):
                    raise ValueError(f"truncated journal: {path}")
                row = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                if not isinstance(row, dict):
                    raise ValueError(f"expected object in {path}")
            except (ValueError, UnicodeError):
                if prefix:
                    break
                raise ValueError(f"corrupt or truncated journal: {path}") from None
            rows.append(row)
    return rows


def _read_layers(path, *, prefix=False):
    if not (path / FILES["episode"]).is_file():
        raise ValueError("expected layered episode.json; old formats are archived")
    manifest = json.loads((path / FILES["episode"]).read_text(encoding="utf-8"))
    if manifest.get("format") != EPISODE_FORMAT:
        raise ValueError(f"expected {EPISODE_FORMAT}; old formats are archived")
    layers = manifest["layers"]
    if not 1 <= len(layers) <= 2 or layers[0] != {"name": "workbench", "format": WORKBENCH_FORMAT, "path": "workbench/"}:
        raise ValueError("invalid workbench layer")
    if len(layers) == 2:
        layer = layers[1]
        task_id = layer["task"]
        if (layer["name"] != "task" or task_id == "empty" or
                not re.fullmatch(r"[a-z][a-z0-9_]*", task_id) or
                layer["path"] != f"task/{task_id}/" or not layer["format"]):
            raise ValueError("invalid task layer")
    declared_task = manifest.get("identity", {}).get("task", {}).get("id")
    if declared_task is not None and declared_task != (layers[1]["task"] if len(layers) == 2 else "empty"):
        raise ValueError("task identity and layers differ")
    result = []
    for layer in layers:
        directory = path / layer["path"]
        meta = json.loads((directory / FILES["meta"]).read_text(encoding="utf-8"))
        if meta["format"] != layer["format"]:
            raise ValueError("layer format differs from index")
        for key in ("scene_spec", "workbench_profile", "task_profile"):
            if key + "_text" in meta and _hash(meta[key + "_text"]) != meta[key + "_sha256"]:
                raise ValueError(f"{key} hash mismatch")
        result.append(EpisodeLayer(layer["format"], _read_jsonl(directory / FILES["steps"], prefix=prefix),
                                   _read_jsonl(directory / FILES["events"], prefix=prefix), meta))
    return manifest, result


def _check_rows(layers, *, start_step=0, last=-1.):
    if len({len(layer.rows) for layer in layers}) != 1:
        raise ValueError("episode layers have different row counts")
    for step, rows in enumerate(zip(*(layer.rows for layer in layers)), start=start_step):
        time_s = rows[0]["sim_time_s"]
        if any(type(row["step"]) is not int or row["step"] != step or row["sim_time_s"] != time_s for row in rows):
            raise ValueError("episode layers have different clocks or non-continuous steps")
        if not math.isfinite(time_s) or time_s <= last:
            raise ValueError("simulation timestamps must increase")
        if set(rows[0]) != set(WORKBENCH_ROW_KEYS):
            raise ValueError("invalid workbench row keys")
        _validate_robot(rows[0])
        if len(rows) > 1 and "wall_time_ns" in rows[1]:
            raise ValueError("wall_time_ns belongs only to workbench")
        last = time_s


def read_episode(path):
    """Strict, read-only reader. Interrupted journals require explicit recovery."""
    path = Path(path)
    manifest, layers = _read_layers(path)
    _check_rows(layers)
    if manifest["status"] == "recording" or manifest["rows"] != len(layers[0].rows):
        raise ValueError("episode length/status indicates an interrupted recording")
    inputs = read_input_events(path, meta=layers[0].meta)
    episode = DiskEpisode(layers[0], layers[1] if len(layers) == 2 else None, manifest, inputs,
                          [note["text"] for note in manifest["operator_notes"]])
    return episode


def read_input_events(path, *, meta=None):
    path = Path(path)
    # The metadata argument remains accepted for existing Python callers. The
    # authoritative command count belongs to the workbench layer.
    meta = json.loads((path / "workbench" / FILES["meta"]).read_text(encoding="utf-8"))
    events = _read_jsonl(path / "workbench" / FILES["commands"])
    if len(events) != meta["input_messages"]:
        raise ValueError("input journal length differs from metadata")
    previous_sequence = previous_receive = -1
    for event in events:
        if event["sequence"] <= previous_sequence or event["received_monotonic_ns"] < previous_receive:
            raise ValueError("input journal receive order is invalid")
        if event["received_wall_time_ns"] <= 0 or not isinstance(event["header"]["stamp"]["nanosec"], int):
            raise ValueError("invalid input clocks")
        previous_sequence, previous_receive = event["sequence"], event["received_monotonic_ns"]
    return events
