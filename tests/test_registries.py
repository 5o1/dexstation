"""Extension discovery and capabilities, using inert sources and backend doubles."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexstation.backends import registry as backend_registry
from dexstation.runtime import plugins
from dexstation.inputs import registry as source_registry
from dexstation.inputs.registry import SourceEntry, SourceCapabilities
from test_session_compose import simulated


@pytest.fixture
def entry_points(monkeypatch):
    # A new discovery lifetime per test; restore process-wide state afterward.
    monkeypatch.setattr(plugins, '_discovered', set())
    monkeypatch.setattr(plugins, '_errors', [])
    monkeypatch.setattr(source_registry, '_ENTRIES', dict(source_registry._ENTRIES))
    monkeypatch.setattr(backend_registry, '_ENTRIES', dict(backend_registry._ENTRIES))
    entries = {}
    monkeypatch.setattr(plugins.metadata, 'entry_points', lambda *, group: entries.get(group, ()))
    return entries


def endpoint(name, entry=None, error=None, loads=None):
    def load():
        if loads is not None:
            loads.append(name)
        if error:
            raise error
        return entry
    return SimpleNamespace(name=name, load=load)


def cli_module(filename):
    path = Path(__file__).parents[1] / 'scripts' / filename
    spec = importlib.util.spec_from_file_location('registry_cli_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def config(manager):
    from aiohttp.test_utils import TestClient, TestServer
    from dexstation.server.app import make_app
    async with TestClient(TestServer(make_app(manager, frame_provider=lambda: {}))) as client:
        result = await client.get('/api/v1/config')
        assert result.status == 200
        return await result.json()


@pytest.mark.parametrize('clock,lease', [('simulation', True), ('wall', False)])
def test_fake_entry_point_reaches_cli_api_and_controller(entry_points, simulated, monkeypatch, tmp_path, caplog, clock, lease):
    from dexstation.runtime import controller
    from dexstation.server import app as server
    from dexstation.runtime.health import InputHealth
    calls, polls, leases, loads = [], [], [], []

    class Source:
        health = InputHealth()
        decode_errors = received_packets = 0
        def drain_events(self): return []
        def boundary(self): return [], 0
        def poll(self, now):
            polls.append(now)
            return []
        def close(self): calls.append('closed')

    class Lease:
        def acquire(self, **kwargs): leases.append(kwargs)
        def status(self): return dict(held=True, owner='test')
        def close(self): leases.append('closed')

    def factory(context):
        calls.append(context)
        return Source()

    entry_points['dexstation.sources'] = [endpoint('lab_input', SourceEntry('lab_input', factory,
        SourceCapabilities(clock=clock, hardware_lease=lease)), loads=loads)]
    caplog.set_level('INFO', logger=plugins.__name__)
    names = source_registry.source_names()
    assert names == ('ros2', 'udp', 'replay', 'ik_autopilot', 'lab_input')
    cli = cli_module('collection_ctl.py')
    assert cli.parser().parse_args(['session-start', '--backend', 'sapien', '--source', 'lab_input']).source == 'lab_input'
    from dexstation.config import defaults
    assert defaults.COLLECTION_SOURCES == names
    result = asyncio.run(config(server.SessionManager(tmp_path / 'api')))
    assert result['sources'] == ['ros2', 'replay', 'udp', 'ik_autopilot', 'lab_input']
    assert 'plugin_status' not in result
    monkeypatch.setattr(controller, 'HardwareLease', Lease)
    c = controller.CollectionController(controller.CollectionOptions('sapien', 'lab_input', tmp_path / 'run', .1, rate_hz=30))
    assert c.run()['ticks'] == 3
    assert len(polls) == 3
    if clock == 'simulation':
        assert polls == pytest.approx([0., 1/30, 2/30])
    else:
        assert polls[0] > 1 and polls == sorted(polls)
    assert bool(leases) == lease
    if lease:
        assert leases == [dict(port=8780, task='empty'), 'closed']
    assert calls[-1] == 'closed'
    assert loads == ['lab_input']
    assert 'Loaded plugin dexstation.sources:lab_input' in caplog.text


def test_broken_plugins_are_status_not_api_failures(entry_points, tmp_path):
    from dexstation.server.app import SessionManager
    groups = ('dexstation.sources', 'dexstation.backends', 'dexstation.object_types', 'dexstation.channel_types')
    for group in groups:
        entry_points[group] = [endpoint('broken', error=RuntimeError('bad extension'))]
    entry_points['dexstation.sources'].append(endpoint('invalid', object()))
    result = asyncio.run(config(SessionManager(tmp_path)))
    errors = result['plugin_status']['errors']
    assert {e['group'] for e in errors} == set(groups)
    assert len(errors) == 5
    assert result['sources'] == ['ros2', 'replay', 'udp', 'ik_autopilot']
    assert [e['name'] for e in result['backends']] == ['sapien']
    assert len(plugins.plugin_status()['plugin_status']['errors']) == 5


def test_backend_entry_probes_versions_and_online_policy(entry_points, monkeypatch):
    loads = []
    entry_points['dexstation.backends'] = [endpoint('lab_engine', backend_registry.BackendEntry(
        'lab_engine', lambda scene, **kwargs: (scene, kwargs), versions={'lab-engine': '1.2'},
        imports=('lab_engine',), version='1.2'))]
    monkeypatch.setattr(backend_registry.importlib, 'import_module', lambda module: loads.append(module))
    monkeypatch.setattr(backend_registry.metadata, 'version', lambda name: '1.2')
    assert backend_registry.backend_names(online=True) == ('sapien', 'lab_engine')
    entry = backend_registry.get_backend('lab_engine')
    assert entry.self_check() == dict(name='lab_engine', available=True, reason=None, check='import+version')
    assert loads == ['lab_engine']
    assert backend_registry.create_registered_backend('lab_engine', 'scene', render=False) == ('scene', {'render': False})
    monkeypatch.setattr(backend_registry.metadata, 'version', lambda name: 'wrong')
    assert 'required 1.2' in entry.self_check()['reason']
    with pytest.raises(ValueError, match='unknown collection backend'):
        backend_registry.create_registered_backend('mujoco', None)


def test_builtin_sources_keep_capabilities_and_order(entry_points):
    from dexstation.config import defaults
    assert defaults.COLLECTION_SOURCES == ('ros2', 'udp', 'replay', 'ik_autopilot')
    assert defaults.API_COLLECTION_SOURCES == ('ros2', 'replay', 'udp', 'ik_autopilot')
    caps = {e.name: e.capabilities for e in source_registry.source_entries()}
    assert caps['ros2'].hardware_lease and caps['ros2'].real_robot_mirror
    assert caps['ros2'].teleop_calibration and caps['udp'].teleop_calibration
    assert caps['replay'].clock == caps['ik_autopilot'].clock == 'simulation'
    assert caps['ik_autopilot'].task_declared
    assert not caps['udp'].hardware_lease




@pytest.mark.parametrize('bad_entry', [
    SourceEntry('bad_order', lambda context: None, api_order='bad'),
    SourceEntry('bad_validator', lambda context: None, validate=object()),
])
def test_invalid_source_hooks_are_reported_before_use(entry_points, tmp_path, bad_entry):
    from dexstation.server.app import SessionManager
    entry_points['dexstation.sources'] = [endpoint(bad_entry.name, bad_entry)]
    result = asyncio.run(config(SessionManager(tmp_path)))
    assert bad_entry.name not in result['sources']
    assert result['plugin_status']['errors'][0]['name'] == bad_entry.name
