"""Task discovery, controller integration and isolation without native physics."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dexstation.runtime import plugins
from dexstation.runtime.plugins import TaskPlugin
from test_registries import entry_points, endpoint
from test_session_compose import simulated, SceneBackend, ComposedDesk


@pytest.fixture
def task_entries(entry_points, monkeypatch):
    plugins.task_plugins()
    monkeypatch.setattr(plugins, '_TASKS', dict(plugins._TASKS))
    monkeypatch.setattr(plugins, '_active_tasks', set(plugins._active_tasks))
    monkeypatch.setattr(plugins, '_discovered', plugins._discovered - {'dexstation.tasks'})
    return entry_points


def test_gamepad_registry_source_drives_controller_on_simulation_clock(simulated, monkeypatch, tmp_path):
    """Keep the actual source constructor, poll, event stream and observation loop.

    Only URDF numerical models/precomputed mode preparation use small stand-ins.
    This checks integration and scheduling, not IK accuracy or native dynamics.
    """
    from dexstation.runtime import controller
    from dexstation.task_plugins.xbox_gamepad_v3_3 import autopilot_source as source
    from dexstation.inputs.registry import get_source
    polls, batches = [], []
    monkeypatch.setattr(source, 'ArmModel', lambda spec, side: NS(
        home=np.array(getattr(spec, side+'_arm').home_q), hand_pose=lambda q: np.eye(4)))
    monkeypatch.setattr(source, 'HandModel', lambda spec, side: NS(fingers={}))
    def prepare(self, side, index, mode):
        target = self.arms[side].home + .2
        return dict(name=mode['name'], controls=set(self.controls), away={},
            descent=[target], retreat=[target], fingers=(), micro={}, play_q=target)
    monkeypatch.setattr(source.IkAutopilotSource, '_prepare_mode', prepare)
    original = source.IkAutopilotSource.poll
    def poll(self, now):
        polls.append(now)
        commands = original(self, now)
        batches.append(commands)
        return commands
    monkeypatch.setattr(source.IkAutopilotSource, 'poll', poll)
    tasks = plugins.task_plugin('xbox_gamepad_v3_3').task_sets('ik_autopilot')[0]
    options = controller.CollectionOptions('sapien', 'ik_autopilot', tmp_path/'run', .2,
        task='xbox_gamepad_v3_3', autopilot_tasks=Path(tasks), rate_hz=30, motion_profile='legacy', record=True)
    c = controller.CollectionController(options)
    assert c.run()['ticks'] == 6
    assert type(c.source) is source.IkAutopilotSource
    assert polls == pytest.approx([i/30 for i in range(6)])
    assert all(len(commands) == 4 for commands in batches)
    assert all(command.stamp == pytest.approx(t) for commands, t in zip(batches, polls) for command in commands)
    assert batches[-1][0].positions != batches[0][0].positions
    assert c.source.received_packets == 24
    assert c.backend.commands == batches[-1]
    assert c.status()['autopilot'] == c.source.autopilot_status()
    episode = tmp_path/'run/episode-000000'
    assert (episode/'task/xbox_gamepad_v3_3/meta.json').is_file()
    summary, results, false_events = plugins.task_plugin('xbox_gamepad_v3_3').evaluate(episode, tasks_path=tasks)
    assert results == false_events == []  # Short approach has no completed task to score.
    assert summary['tasks'] == 0


def external_profile(tmp_path, task_id):
    from dexstation.contracts.profiles import EMPTY_TASK
    data = json.loads(json.dumps(EMPTY_TASK))
    data.update(id=task_id, title='Lab task', cells=[dict(id='lab',title='Lab',module='web/lab.mjs')])
    root = tmp_path/task_id;root.mkdir()
    (root/'task.json').write_text(json.dumps(data))
    web = root/'web';web.mkdir()
    (web/'lab.mjs').write_text('export function mount(root, api) { root.textContent = "Lab cell"; }')
    return root, web


def test_external_task_entry_point_lists_profile_and_serves_cell(task_entries, tmp_path, caplog):
    from aiohttp.test_utils import TestClient, TestServer
    from dexstation.server.app import make_app, SessionManager
    root, web = external_profile(tmp_path, 'lab_task')
    loads = []
    entry = TaskPlugin('lab_task', NS(API_VERSION=1, ACTIONS={}, cell_state=lambda task:{}),root,web)
    task_entries['dexstation.tasks'] = [endpoint('lab_task',entry,loads=loads)]
    caplog.set_level('INFO', logger=plugins.__name__)
    async def scenario():
        async with TestClient(TestServer(make_app(SessionManager(tmp_path/'sessions'),frame_provider=lambda:{}))) as client:
            result = await (await client.get('/api/v1/config')).json()
            assert dict(id='lab_task',title='Lab task',version='1.0') in result['tasks']
            cells = await (await client.get('/api/v1/tasks/lab_task/cells')).json()
            url = cells['cells'][0]['module']
            assert url == '/tasks/lab_task/web/lab.mjs'
            response = await client.get(url)
            assert response.status == 200 and 'Lab cell' in await response.text()
            assert (await client.get('/tasks/lab_task/web/%2e%2e%2ftask.json')).status == 404
    asyncio.run(scenario())
    assert loads == ['lab_task']
    assert 'Loaded plugin dexstation.tasks:lab_task' in caplog.text


def test_broken_task_is_reported_and_only_its_session_fails(task_entries, simulated, tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from dexstation.server.app import make_app
    task_entries['dexstation.tasks'] = [endpoint('broken_task',error=ImportError('missing lab dependency'))]
    async def scenario():
        manager = ComposedDesk(tmp_path/'sessions')
        async with TestClient(TestServer(make_app(manager,enable_remote_control=True,frame_provider=lambda:{}))) as client:
            result = await client.get('/api/v1/config')
            assert result.status == 200
            body = await result.json()
            assert body['plugin_status']['errors'] == [dict(group='dexstation.tasks',name='broken_task',reason='missing lab dependency')]
            headers = {'X-Gamepad-VLA-Client':'test'}
            failed = await client.post('/api/v1/session/start',json={'task':'broken_task','source':'udp'},headers=headers)
            assert failed.status >= 400
            assert 'missing lab dependency' in await failed.text()
            assert not manager.worker.is_alive()
            good = await client.post('/api/v1/session/start',json={'task':'empty','source':'udp'},headers=headers)
            assert good.status == 200
            assert (await client.get('/api/v1/config')).status == 200
            await manager.stop()
    asyncio.run(scenario())


@pytest.mark.parametrize('task_id', ['empty','tabletop_cube_grasp'])
def test_empty_and_cube_run_without_importing_gamepad_plugin(tmp_path, task_id):
    """Fresh processes prevent earlier tests from concealing eager imports."""
    code = r'''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(('dexstation.task_plugins.xbox_gamepad_v3_3', 'xbox_gamepad_physics')):
            raise AssertionError('unexpected task import: ' + fullname)
sys.meta_path.insert(0, Block())
from pathlib import Path
from unittest.mock import patch
from dexstation.runtime.controller import CollectionOptions, CollectionController
from dexstation.contracts.backend import FakeBackend
from dexstation.inputs.replay import CollectionReplay
import numpy as np
root=Path(sys.argv[1]); replay=root/'input.npz'
np.savez(replay,action=np.zeros((2,54)),timestamp=[0.,.02],names=[],units=[],episode_index=[0,0])
# A deterministic source and in-memory backend; no ROS, native renderer or assets.
from dexstation.inputs.registry import SourceEntry, register_source
from dexstation.runtime.health import InputHealth
class Source:
    health=InputHealth(); decode_errors=received_packets=0
    def poll(self,t): return []
    def drain_events(self): return []
    def boundary(self): return [],0
    def close(self): pass
register_source(SourceEntry('inert',lambda context:Source()))
with patch('dexstation.runtime.controller.configure_runtime'), patch('dexstation.runtime.controller.create_backend',lambda *a,**k:FakeBackend()), patch('dexstation.recording.session.fingertip_offsets',lambda spec:{}):
    c=CollectionController(CollectionOptions('sapien','inert',root/'run',.1,rate_hz=30,task=sys.argv[2],motion_profile='legacy'))
    assert c.run()['ticks']==3
assert not any(n.startswith(('dexstation.task_plugins.xbox_gamepad_v3_3','xbox_gamepad_physics')) for n in sys.modules)
'''
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path),task_id],env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+os.environ.get('PYTHONPATH','')),text=True,capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr


def test_declared_modules_and_offline_evaluation_resolve_through_entry(monkeypatch):
    entry = plugins.task_plugin('xbox_gamepad_v3_3')
    provider = entry.resolve('dexstation.task_plugins.xbox_gamepad_v3_3.autopilot')
    assert callable(provider.make_source)
    module = entry.resolve('dexstation.task_plugins.xbox_gamepad_v3_3.evaluate')
    calls = []
    monkeypatch.setattr(module,'evaluate_episode',lambda episode,**kwargs:calls.append((episode,kwargs)) or ('summary',[],[]))
    assert entry.evaluate(Path('/fixture/episode'), tasks_path='/fixture/tasks.json') == ('summary',[],[])
    assert calls == [(Path('/fixture/episode'),{'tasks_path':'/fixture/tasks.json'})]
    with pytest.raises(ValueError,match='does not provide module'):
        entry.resolve('unrelated_lab.module')


@pytest.mark.parametrize('fault', ['version','hook','source'])
def test_invalid_task_hooks_are_reported_without_mutating_registries(task_entries, tmp_path, fault):
    from dexstation.server.app import SessionManager
    from dexstation.inputs.registry import SourceEntry, source_names
    from test_registries import config
    root, web = external_profile(tmp_path,'invalid_task')
    provider = NS(API_VERSION=2 if fault=='version' else 1,ACTIONS={})
    if fault=='hook':provider.cell_state=42
    sources = (SourceEntry('invalid_input',None),) if fault=='source' else ()
    task_entries['dexstation.tasks'] = [endpoint('invalid_task',TaskPlugin('invalid_task',provider,root,web,sources=sources))]
    result=asyncio.run(config(SessionManager(tmp_path/'sessions')))
    assert result['plugin_status']['errors'][0]['name']=='invalid_task'
    assert 'invalid_task' not in plugins.task_plugins()
    assert 'invalid_input' not in source_names()


def test_cold_process_resolves_task_types_for_render_and_replay():
    code = '''
import sys
from dexstation.backends.object_types import object_type
from dexstation.backends.channel_types import channel_type
assert object_type('xbox_gamepad_physics').legacy_gamepad
assert channel_type('gamepad_sdl3_core_v1').describe({}) == ((32,), 'float32', 'normalized')
assert 'xbox_gamepad_physics' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+os.environ.get('PYTHONPATH','')),text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_task_activation_failure_is_reported_after_start(task_entries, simulated, tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from dexstation.server.app import make_app
    root, web = external_profile(tmp_path,'activation_failure')
    def activate():raise ImportError('missing task type dependency')
    task_entries['dexstation.tasks']=[endpoint('activation_failure',TaskPlugin(
        'activation_failure',NS(API_VERSION=1,ACTIONS={}),root,web,activate=activate))]
    async def scenario():
        manager=ComposedDesk(tmp_path/'sessions')
        async with TestClient(TestServer(make_app(manager,enable_remote_control=True,frame_provider=lambda:{}))) as client:
            assert (await client.get('/api/v1/config')).status==200
            response=await client.post('/api/v1/session/start',json={'task':'activation_failure'},headers={'X-Gamepad-VLA-Client':'test'})
            assert response.status>=400 and 'missing task type dependency' in await response.text()
            body=await (await client.get('/api/v1/config')).json()
            assert body['plugin_status']['errors']==[dict(group='dexstation.tasks',name='activation_failure',reason='missing task type dependency')]
            assert (await client.post('/api/v1/session/start',json={'task':'empty'},headers={'X-Gamepad-VLA-Client':'test'})).status==200
            await manager.stop()
    asyncio.run(scenario())
