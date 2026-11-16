"""Profile value types and the empty-task contract."""

from dexstation.contracts.versions import TASK_SCHEMA
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCENE_VARIANT = "full"
DEBUG_SCENE_VARIANTS = ("free-space",)

# Same JSON value as examples/empty.json (§4.5.1). No mutable default arguments.
EMPTY_TASK = {
    "schema": TASK_SCHEMA, "id": "empty", "version": "1.0",
    "title": "Empty table", "workbenches": ["*"],
    "backends": {"sapien": {}, "mjcf_export": {}},
    "objects": [], "fixtures": [], "observations": [], "sources": [],
    "cells": [], "actions": {}, "evaluation": None, "keep_out": [],
    "real_robot": {"twin_objects": "none"},
}


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    unit: str
    object: str | None


@dataclass(frozen=True)
class ObjectSpec:
    """Task body; collision contains only robots/environment/objects booleans.

    Geometry owns collision_mesh (sdf/coacd). For articulations, extra may
    contain initial_joint_positions in radians; omission means URDF zero.
    Static task bodies are also projected into WorkcellSpec.fixtures during
    the transition. Backends must create each named body only once.
    """
    name: str
    type: str
    geometry: dict
    inertial: dict
    material: dict
    initial_pose_world: tuple[float, ...]
    randomize: dict | None
    collision: dict
    extra: dict


@dataclass(frozen=True)
class CameraPair:
    name: str
    real: dict
    sim: dict | None


@dataclass(frozen=True)
class RuntimeScene:
    # The composed scene carries an implementation-owned spec by reference.
    spec: "WorkcellSpec"
    workbench: dict
    task: dict
    objects: tuple[ObjectSpec, ...]
    channels: tuple[ChannelSpec, ...]
    cameras: tuple[CameraPair, ...]
    cells: tuple[dict, ...]
    actions: dict
    limits_path: Path
    takeover_path: Path
    calibration: dict
    identity: dict
    frozen: dict
