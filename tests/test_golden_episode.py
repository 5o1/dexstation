"""Production Session/recorder layout guards driven by deterministic FakeBackend."""
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest

from dexstation.recording import session as collection
from dexstation.control.commands import ArmCommand, HandCommand
from dexstation.inputs.replay import command_event
from dexstation.contracts.backend import FakeBackend
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES
from helpers import ROOT, assert_golden, canonical_golden, golden_scene


def recursive_keys(value):
    """Scalar leaves are null; arrays retain only distinct recursive key shapes."""
    if isinstance(value, dict):
        return {key: recursive_keys(child) for key, child in value.items()}
    if isinstance(value, list):
        shapes = {canonical_golden(recursive_keys(child)) for child in value}
        return [json.loads(shape) for shape in sorted(shapes)]
    return None


def episode_layout(path):
    files = sorted(file.relative_to(path).as_posix() for file in path.rglob('*') if file.is_file())
    keys = {}
    for relative in files:
        file = path / relative
        if file.name in ('episode.json', 'meta.json'):
            keys[relative] = recursive_keys(json.loads(file.read_text(encoding='utf-8')))
        elif file.name in ('steps.jsonl', 'events.jsonl', 'commands.jsonl'):
            with file.open(encoding='utf-8') as stream:
                first = next(stream, None)
            keys[relative] = None if first is None else recursive_keys(json.loads(first))
    return dict(files=files, keys=keys)


@pytest.mark.parametrize('task', ['empty', 'xbox_gamepad_v3_3'])
def test_episode_layout(task, tmp_path, monkeypatch):
    scene = golden_scene(ROOT / f'src/dexstation/task_plugins/{task}/data/task.json', tmp_path, monkeypatch)
    # Fingertip mesh extraction is the only asset-dependent Session operation.
    monkeypatch.setattr(collection, 'fingertip_offsets', lambda spec: {})
    monkeypatch.setattr(collection, 'ProcessPoolExecutor',
                        lambda **kwargs: ThreadPoolExecutor(max_workers=1))
    backend = FakeBackend()
    if task != 'empty':
        backend._initial.task = dict(gamepad_signal=np.arange(32, dtype=float) / 31,
                                     gamepad_mask=np.arange(32) % 2 == 0,
                                     control_joint_positions={'button_a': .001})
        backend.reset()
    commands = [command for side in ('left', 'right') for command in (
        ArmCommand(side, (0.,) * 7, 0., (0.,) * 7,
                   dict(absolute=[False] * 7, startup=[False] * 7)),
        HandCommand(side, dict.fromkeys(G20_JOINT_NAMES, 0.), 0., (0.,) * 20))]
    session = collection.Session(tmp_path / 'session', scene, backend, {'kind': 'replay'}, fps=50,
        task_format=scene.task.get('recording', {}).get('format'),
        workbench_meta={'scene_check': dict(checked=False, reason='initial-state geometry check pending WP5')},
        task_meta={'seed': 17, 'initial_poses': {obj.name: obj.initial_pose_world for obj in scene.objects}})
    try:
        session.control(dict(action='start', client='cli'))
        session.control(dict(action='note', text='golden episode'))
        if task != 'empty':
            session.writer.event('task', 'reset', {'seed': 17})
        for step in range(2):
            backend.step(.02)
            session.record_inputs([command_event(commands[0], step)])
            session.record(backend.observe(), commands, applied_monotonic_ns=step + 1,
                           input_sequences={'left_arm': step})
        session.writer.meta.update(received_input_messages=2, receive_start_sequence=0, receive_end_sequence=2)
        session.control(dict(action='stop', client='cli'))
        session.wait_saved()
        assert session.current_integrity['passed'], session.current_integrity
    finally:
        session.close()
        backend.close()
    assert_golden(f'episode/{task}.json', episode_layout(session.output / 'episode-000000'))
