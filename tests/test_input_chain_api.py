"""Real HTTP/SSE lifecycle with inert input/worker processes and temporary files."""
import asyncio
from pathlib import Path

import pytest
pytest.importorskip('aiohttp')
from aiohttp import ClientSession
from aiohttp.test_utils import TestServer

from dexstation.server.app import make_app
from dexstation.inputs.supervisor import registered_rows, processes, receiver_holders
from helpers import fake_chain, proc_entry, session_manager, SurrogateChain, until


def test_conflict_http_409_preserves_original_pid_and_message(tmp_path,isolated_station_paths):
    async def scenario():
        chain,device=fake_chain(tmp_path/'fixture')
        proc=proc_entry(chain.proc_root,4242,['python3','/opt/workstation/original_manus.py'])
        (proc/'fd/5').symlink_to(device)
        manager=session_manager(tmp_path,chain)
        server=TestServer(make_app(manager,enable_remote_control=True,frame_provider=lambda:{}))
        await server.start_server()
        try:
            async with ClientSession() as client:
                async with client.post(server.make_url('/api/v1/session/start'),json={},headers={'X-Gamepad-VLA-Client':'test'}) as response:
                    value=await response.json()
                    assert response.status==409 and value['error_code']=='input_busy'
                    assert 'PID 4242' in value['message'] and 'original_manus.py' in value['message']
                    assert value['state']['input_chain']['error']==value['message']
                    assert manager.worker is None
                async with client.get(server.make_url('/api/v1/events')) as response:
                    assert 'PID 4242' in (await response.content.readline()).decode()
            assert proc.exists()
        finally:await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('ending',['idle','service_exit'])
def test_http_lifecycle_releases_all_owned_groups_and_receiver(tmp_path,ending,isolated_station_paths):
    async def scenario():
        chain=SurrogateChain(tmp_path/'fixture',delay=.05)
        manager=session_manager(tmp_path,chain,prepare=.05)
        now=[0.]
        app=make_app(manager,enable_remote_control=True,frame_provider=lambda:{},clock=lambda:now[0],idle_timeout_s=1)
        server=TestServer(app);await server.start_server()
        try:
            async with ClientSession() as client:
                async with client.post(server.make_url('/api/v1/session/start'),json={},headers={'X-Gamepad-VLA-Client':'test'}) as response:
                    assert response.status==200
                await until(lambda:chain.launched.is_set() and chain.status()['state']=='ready')
                manager.latest.update(session_state='running')
                if ending=='idle':
                    now[0]=2
                    await app['stop_idle']()
                else:
                    await server.close()
            assert not registered_rows(chain.registry(),processes())
            assert not receiver_holders(processes(),[chain.device])
            assert chain.status()['receiver_released']
        finally:await server.close()
    asyncio.run(scenario())


def test_service_cleanup_error_still_releases_inputs(tmp_path,isolated_station_paths):
    async def scenario():
        from types import SimpleNamespace
        chain=SurrogateChain(tmp_path/'fixture',delay=.05)
        manager=session_manager(tmp_path,chain,prepare=.05)
        def camera_failure():raise RuntimeError('camera shutdown test failure')
        server=TestServer(make_app(manager,enable_remote_control=True,frame_provider=lambda:{},camera=SimpleNamespace(stop=camera_failure)))
        await server.start_server()
        await manager.start({})
        await until(lambda:chain.launched.is_set())
        with pytest.raises(RuntimeError,match='camera shutdown test failure'):
            await server.close()
        assert not registered_rows(chain.registry(),processes())
        assert not receiver_holders(processes(),[chain.device])
    asyncio.run(scenario())




def test_foreign_ros_graph_blocks_session_with_structured_reason(tmp_path,isolated_station_paths):
    from dexstation.inputs.supervisor import ForeignPublisherError
    from dexstation.inputs.linkerta_reader import publisher_evidence
    from types import SimpleNamespace as N
    graph=N(get_publishers_info_by_topic=lambda _: [N(node_name='linkerta_node',endpoint_gid=b'foreign')])
    chain=SurrogateChain(tmp_path/'chain',delay=.01)
    def inspect():
        rows=publisher_evidence(graph,{s:f'/{s}_arm_joint_control' for s in ('left','right')})
        raise ForeignPublisherError(sorted({n for r in rows.values() for n in r['foreign']}))
    chain.check_publishers=inspect
    manager=session_manager(tmp_path,chain)
    async def scenario():
        server=TestServer(make_app(manager,enable_remote_control=True,frame_provider=lambda:{}));await server.start_server()
        try:
            async with ClientSession() as client:
                async with client.post(server.make_url('/api/v1/session/start'),json={},headers={'X-Gamepad-VLA-Client':'test'}) as response:
                    body=await response.json()
                    assert response.status==409 and body['input']['foreign_publishers']==['linkerta_node']
                    assert body['input']['fix']=='stop_foreign'
                    assert manager.worker is None and not chain.launched.is_set()
        finally:await server.close()
    asyncio.run(scenario())


def test_linkerta_reason_is_forwarded_by_status_and_sse(tmp_path):
    manager=session_manager(tmp_path)
    manager.latest.update(source='ros2',input_health=dict(left_arm=dict(input_available=False,input_reason='can9 down',linkerta={'interface':{'name':'can9'}})))
    async def scenario():
        server=TestServer(make_app(manager,frame_provider=lambda:{}));await server.start_server()
        try:
            async with ClientSession() as client:
                async with client.get(server.make_url('/api/v1/status')) as response:
                    body=await response.json()
                    assert body['arm_input']['left']['reason']=='can9 down'
                    assert body['input_health']['left_arm']['linkerta']['interface']['name']=='can9'
                async with client.get(server.make_url('/api/v1/events')) as response:
                    assert b'can9 down' in await response.content.readline()
        finally:await server.close()
    asyncio.run(scenario())
