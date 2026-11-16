"""Fresh-process public acceptance with private imports blocked before startup."""
import importlib.abc
import json
import os
from pathlib import Path
import sys


attempts = []


class NoStation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('dexstation_'):
            attempts.append(fullname)
            raise AssertionError('Private implementation requested: ' + fullname)


sys.meta_path.insert(0, NoStation())

import asyncio
from dataclasses import replace
from aiohttp.test_utils import TestClient, TestServer
from helpers import ROOT, bench_directory
from test_session_compose import SceneBackend, SavePool, replay_npz
from dexstation.profiles import compose as profiles
from dexstation.recording import session as collection
from dexstation.backends import registry as backend_registry
from dexstation.backends.registry import backend_names
from dexstation.runtime.controller import CollectionController, CollectionOptions
from dexstation.server.app import SessionManager, make_app, task_catalog
from dexstation.inputs.registry import source_names
from dexstation.recording.episode import read_episode


def prepare_bench():
    original = profiles.load_workbench
    committed = ROOT / 'src/dexstation/examples/ar5_g20/data'
    def load(path):
        candidate = Path(path).absolute()
        if candidate in (committed, committed / 'workbench.json'):
            path = bench_directory()
        return original(path)
    profiles.load_workbench = load


async def main(root):
    prepare_bench()
    collection.fingertip_offsets = lambda spec: {}
    collection.ProcessPoolExecutor = SavePool
    entry = backend_registry.get_backend('sapien')
    backend_registry._ENTRIES['sapien'] = replace(entry, factory=lambda scene, **kw: SceneBackend(scene),
        imports=(), versions={}, check=None)
    assert {'sapien', 'mujoco'} <= set(backend_names())
    assert source_names(api=True) == ('ros2', 'replay', 'udp', 'ik_autopilot')
    assert {'empty', 'tabletop_cube_grasp', 'xbox_gamepad_v3_3'} <= {t['id'] for t in task_catalog()}
    client = TestClient(TestServer(make_app(SessionManager(root/'api'), enable_remote_control=True)))
    async with client:
        page = await (await client.get('/')).text()
        assert 'id="canDialog"' not in page
        assert 'service-presentation' in page
        camera_module = await (await client.get('/camera-panel.mjs')).text()
        assert 'USB port' not in camera_module
        config = await (await client.get('/api/v1/config')).json()
        assert config['defaults']['source'] == 'udp'
        assert config['defaults']['retarget'] == 'manus_anatomical'
        assert config['retarget_methods'] == ['manus_anatomical']
        assert config['signal_sources'] == [{'value': 'teleop', 'label': 'Teleop'}]
        camera = await (await client.get('/api/v1/camera/status')).json()
        assert camera['available'] is False and camera['mode'] == 'unavailable'
        response = await client.post('/api/v1/can/bring-up', json={'role': 'linkerta'},
                                     headers={'X-Gamepad-VLA-Client': 'cli'})
        assert response.status == 409
        assert (await response.json())['error_code'] == 'bring_up_unavailable'
    results = {}
    for task in ('empty', 'tabletop_cube_grasp'):
        options = CollectionOptions('sapien', 'udp', root/task, .04, task=task,
            rate_hz=50, udp_port=0, record=True, motion_profile='legacy')
        controller = CollectionController(options)
        result = controller.run()
        episode = read_episode(root/task/'episode-000000')
        assert episode.meta['identity']['task']['id'] == task
        assert len(episode.workbench.rows) == 2
        robot = episode.workbench.rows[0]
        assert len(robot['arm_q']) == 2 and len(robot['hand_q']) == 2
        results[task] = result
    replay = replay_npz(root/'replay.npz')
    result = CollectionController(CollectionOptions('sapien', 'replay', root/'replayed', .04,
        replay=replay, rate_hz=50, motion_profile='legacy')).run()
    assert result['ticks'] == 2
    assert not attempts, attempts
    assert not any(name.startswith('dexstation_')
                   for name in sys.modules)
    print(json.dumps(dict(private_import_attempts=attempts, tasks=list(results), replay_steps=result['ticks'],
                          sources=config['sources'], backends=[b['name'] for b in config['backends']])))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    site = root/'site.json'
    site.write_text('{}')
    os.environ.update(DEXSTATION_SITE_CONFIG=str(site), DEXSTATION_DATA=str(root/'data'),
                      DEXSTATION_WORKSPACE=str(root/'workspace'))
    asyncio.run(main(root))
