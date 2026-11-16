"""Field-by-field RuntimeScene guards, without loading physics or robot assets."""
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

from helpers import ROOT, GOLDEN_ROOT, assert_golden, bench_directory, golden_scene, updating_golden


TASKS = sorted((ROOT / 'src/dexstation/task_plugins').glob('*/data/task.json'))
BENCH = 'src/dexstation/examples/ar5_g20/data'


def scene_json(value, tmp_path, trail=()):
    """Walk every declared field, retaining numeric precision and dictionary keys."""
    if trail == ('identity', 'code', 'git_commit'):
        return '<git-commit>'
    if trail == ('identity', 'code', 'dirty'):
        return '<git-dirty>'
    if is_dataclass(value):
        return {field.name: scene_json(getattr(value, field.name), tmp_path, trail + (field.name,))
                for field in fields(value)}
    if isinstance(value, dict):
        return {key: scene_json(child, tmp_path, trail + (key,)) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [scene_json(child, tmp_path, trail + (index,)) for index, child in enumerate(value)]
    if isinstance(value, np.ndarray):
        return scene_json(value.tolist(), tmp_path, trail)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        # Replace only path prefixes on complete path components, never digests.
        for prefix, replacement in ((str(bench_directory()), BENCH),
                                    (str(ROOT), '.'), (str(tmp_path), '<tmp>')):
            if value == prefix:
                return replacement
            if value.startswith(prefix + '/'):
                suffix = value[len(prefix) + 1:]
                return suffix if replacement == '.' else replacement + '/' + suffix
    return value


def test_scene_profile_inventory():
    assert TASKS, 'No task profiles discovered under src/dexstation/task_plugins'
    expected = {path.parent.parent.name + '.json' for path in TASKS}
    directory = GOLDEN_ROOT / 'scenes'
    if updating_golden():
        for path in directory.glob('*.json'):
            if path.name not in expected:
                path.unlink()
        return
    assert {path.name for path in directory.glob('*.json')} == expected, 'Task profile/golden inventory changed'


@pytest.mark.parametrize('task_path', TASKS, ids=lambda path: path.parent.parent.name)
def test_composed_scene(task_path, tmp_path, monkeypatch):
    scene = golden_scene(task_path, tmp_path, monkeypatch)
    assert_golden(f'scenes/{task_path.parent.parent.name}.json', scene_json(scene, tmp_path), numeric=True)
