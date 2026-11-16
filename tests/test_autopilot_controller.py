from helpers import bench_spec
"""IK autopilot through the real controller and JointMotion with a kinematic stand-in for physics.

Needs the scene's arm and hand URDFs from the configured workspace; skipped where they are absent.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexstation.runtime.controller import CollectionController, CollectionOptions
from dexstation.backends.sapien_physics import CollectionSapienBackend
from dexstation.recording.episode import SIDES, Observation
from dexstation.control.joint_motion import JointMotion, load_profile, limits_from_spec
from dexstation.model.spec import load_spec, matrix_to_pose

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'src/dexstation/task_plugins/xbox_gamepad_v3_3/data/autopilot/game_combos_v1.json'






def test_input_source_meta_names_the_replay_file_only_for_replay():
    from dexstation.runtime.controller import input_source_meta
    assert input_source_meta('ik_autopilot', '/data/august/states.npz', True) == dict(kind='ik_autopilot', replay=None, loop=False)
    assert input_source_meta('ros2', '/data/august/states.npz', True) == dict(kind='ros2', replay=None, loop=False)
    assert input_source_meta('replay', '/data/a.npz', True) == dict(kind='replay', replay='/data/a.npz', loop=True)
