"""Kernel flock contention, lifecycle and HTTP 409; no input/hardware nodes."""
import asyncio
import multiprocessing as mp
from pathlib import Path
import queue
import threading
from types import SimpleNamespace as N

import pytest

from dexstation.runtime.health import HardwareLease, HardwareConflict
from dexstation.server.app import SessionManager, make_app


def competing_process(path, result):
    lease=HardwareLease(path)
    try:lease.acquire(port=9002,task='empty')
    except HardwareConflict as exc:result.put(exc.owner)
    else:result.put('unexpected acquisition');lease.close()


def test_flock_cross_process_owner_and_release(tmp_path):
    path=tmp_path/'locks/hardware.lock'
    a=HardwareLease(path)
    a.acquire(port=9001,task='xbox_gamepad_v3_3')
    context=mp.get_context('spawn');result=context.Queue()
    child=context.Process(target=competing_process,args=(path,result));child.start();child.join(10)
    assert child.exitcode==0
    owner=result.get(timeout=2)
    assert owner['port']==9001 and owner['task']=='xbox_gamepad_v3_3'
    a.close()
    b=HardwareLease(path);b.acquire(port=9002,task='empty');b.close()
    assert path.is_file()


class Worker:
    exitcode=0
    def __init__(self,*args):self.alive=False
    def start(self):self.alive=True
    def is_alive(self):return self.alive
    def join(self,timeout):self.alive=False


def manager_at(path):
    manager=SessionManager(path)
    manager.input_chain_factory=lambda:None
    manager.ctx=N(Queue=lambda **kwargs:queue.Queue(),Event=threading.Event,
        Process=lambda **kwargs:Worker())
    return manager


def test_manager_hardware_conflict_is_409_and_replay_has_no_lock(monkeypatch,tmp_path):
    import dexstation.config.site as site
    monkeypatch.setattr(site,'hardware_lock_path',lambda:tmp_path/'station-hardware.lock')
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer
    async def scenario():
        a,b=manager_at(tmp_path/'a'),manager_at(tmp_path/'b')
        a.service_port=9001
        await a.start(dict(backend='sapien',source='ros2',task='xbox_gamepad_v3_3'))
        service=TestServer(make_app(b,enable_remote_control=True,frame_provider=lambda:{}))
        await service.start_server()
        try:
            async with ClientSession() as client:
                async with client.post(service.make_url('/api/v1/session/start'),json={'backend':'sapien','source':'ros2','task':'empty'},headers={'X-Gamepad-VLA-Client':'test'}) as response:
                    value=await response.json()
                    assert response.status==409 and value['error_code']=='hardware_busy'
                    assert value['owner']['port']==9001 and value['owner']['task']=='xbox_gamepad_v3_3'
            replay=tmp_path/'states.npz';replay.touch()
            c=manager_at(tmp_path/'c')
            await c.start(dict(backend='sapien',source='replay',replay=str(replay)))
            assert c.hardware is None and not c.refresh()['hardware_lease']['held']
            assert c.output.parent.name=='empty'
            await c.stop()
        finally:
            await service.close();await a.stop()
        assert not a.hardware.status()['held']
        lease=HardwareLease();lease.acquire(port=9003,task='empty');lease.close()
    asyncio.run(scenario())


def test_preflight_failure_releases_hardware_lock(monkeypatch,tmp_path):
    import dexstation.config.site as site
    from dexstation.inputs.supervisor import InputConflict
    monkeypatch.setattr(site,'hardware_lock_path',lambda:tmp_path/'station-hardware.lock')
    def preflight():raise InputConflict('test conflict')
    manager=manager_at(tmp_path/'sessions')
    manager.input_chain_factory=lambda:N(preflight=preflight)
    with pytest.raises(InputConflict):
        asyncio.run(manager.start(dict(backend='sapien',source='ros2')))
    lease=HardwareLease();lease.acquire(port=9003,task='empty');lease.close()


@pytest.mark.parametrize('failure', ['mkdir', 'open', 'flock'])
def test_lock_unavailable_returns_503_json_without_starting_inputs(monkeypatch,tmp_path,failure):
    import errno
    import fcntl
    import dexstation.config.site as site
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer
    monkeypatch.setenv('DEXSTATION_HOME',str(tmp_path/'dexhand'))
    monkeypatch.setenv('DEXSTATION_WORKSPACE',str(tmp_path/'workspace'))
    path=site.hardware_lock_path()
    if failure=='mkdir':
        (tmp_path/'workspace').write_text('Not a directory')
    elif failure=='open':
        path.mkdir(parents=True)
    else:
        def unavailable(*args):raise OSError(errno.EROFS,'read-only\nfilesystem')
        monkeypatch.setattr(fcntl,'flock',unavailable)
    starts=[]
    class Chain:
        def preflight(self):starts.append('preflight')
        def launch(self):starts.append('launch')
    async def scenario():
        manager=manager_at(tmp_path/'sessions')
        manager.input_chain_factory=Chain
        manager.ctx.Process=lambda **kwargs:starts.append('worker')
        service=TestServer(make_app(manager,enable_remote_control=True,frame_provider=lambda:{}))
        await service.start_server()
        try:
            async with ClientSession() as client:
                async with client.post(service.make_url('/api/v1/session/start'),json={'backend':'sapien','source':'ros2','task':'empty'},headers={'X-Gamepad-VLA-Client':'test'}) as response:
                    body=await response.json()
                    assert response.status==503
                    assert set(body)=={'error'}
                    assert body['error'].startswith('hardware lock unavailable: ')
                    assert '\n' not in body['error']
            assert starts==[]
            assert manager.worker is None and manager.input_task is None
            assert manager.output is None and manager.hardware.stream is None
        finally:await service.close()
    asyncio.run(scenario())
