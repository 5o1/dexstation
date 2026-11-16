"""Shared view publishes measured state without touching collection physics."""
import json
from types import SimpleNamespace
import pytest
from dexstation.render.operator_view import OperatorView


def test_collection_view_publishes_completed_bilateral_state_and_view_control(tmp_path):
    published=[]
    view=OperatorView.__new__(OperatorView)
    view.children=[];view.step_id=0;view.snapshot=SimpleNamespace(write=published.append)
    view.backend=None
    view.run=tmp_path;view._event_offset=0;view.collection=False
    row=dict(hand_q=[[1]*21,[2]*21],arm_q=[[3]*7,[4]*7],gamepad_signal=[.9]+[0]*31,
             control_joint_positions={'button_a_joint':.003},applied_monotonic_ns=123,
             overlay={'values':[.9]+[0]*31})
    view.publish(row)
    result=json.loads(published[0]);assert result['hand_q']==row['hand_q'] and result['arm_q']==row['arm_q']
    assert result['signals']==row['gamepad_signal'] and result['applied_ns']==123
    empty=dict(hand_q=[[0]*21]*2,arm_q=[[0]*7]*2,applied_monotonic_ns=124,
               task={},object_poses={'cube':[1,2,3,1,0,0,0]})
    view.publish(empty)
    state=json.loads(published[-1])
    assert state['object_poses']==empty['object_poses'] and state['signals']==[]
    view.set_view('operator');assert json.loads((tmp_path/'view.json').read_text())=={'view':'operator'}
    with pytest.raises(ValueError):view.set_view('other')
    view.collection=True
    with pytest.raises(ValueError,match='fixed'):view.set_view('operator')
    view.set_view('camera_3')
    (tmp_path/'controls.jsonl').write_text('{"action":"stop"}\n')
    assert view.events()==[{'action':'stop'}] and view.events()==[]




def test_render_state_transports_free_bodies_and_articulations_without_gamepad():
    from dexstation.backends.sapien_physics import CollectionSapienBackend
    from dexstation.backends.mujoco_export import CollectionMujocoBackend
    from dexstation.render.operator_view import apply_render_state
    for backend_type in (CollectionSapienBackend,CollectionMujocoBackend):
        with pytest.raises(ValueError,match='OperatorView'):
            backend_type(None,render=True)
    from dexstation.contracts.backend import FakeBackend
    calls = []
    backend = SimpleNamespace(gamepad=None, runtime_scene=None,
        scene=SimpleNamespace(update_render=lambda:calls.append('render')),
        set_review_state=lambda s:calls.append(s), set_object_state=lambda *a:calls.append(a),
        objects={'hinge': SimpleNamespace(body=SimpleNamespace(set_qpos=lambda q:calls.append(q.tolist())))})
    state = dict(arm_q=[[0]*7]*2, hand_q=[[0]*21]*2,
        object_poses={'cube':[1,2,3,1,0,0,0]}, object_q={'hinge':[.4]})
    apply_render_state(backend, state)
    assert ('cube', state['object_poses']['cube']) in calls
    assert [.4] in calls and calls[-1] == 'render'




def test_fake_backend_observation_publishes_numpy_robot_arrays():
    from dexstation.contracts.backend import FakeBackend
    frames = []
    backend = FakeBackend()
    backend.set_object_state('cube', [1,2,3,1,0,0,0])
    view = OperatorView.__new__(OperatorView)
    view.backend=backend;view.children=[];view.step_id=0;view.applied_ns=42
    view.snapshot=SimpleNamespace(write=frames.append)
    observation=view.observe()
    state=json.loads(frames[-1])
    assert state['arm_q']==observation.robot.arm_q.tolist()
    assert state['object_poses']==observation.object_poses


@pytest.mark.parametrize('cfg, expected', [
    ('missing', ['table']),
    ({'auxiliary_views': ['face', 'back', 'left', 'right',  'table']},
     ['face', 'back', 'left', 'right',  'table']),
    ({'auxiliary_views': ['table']}, ['table']),
], ids=['missing-cfg', 'gamepad', 'empty'])
def test_auxiliary_websocket_filters_views_with_optional_config(cfg, expected):
    import asyncio
    pytest.importorskip('aiohttp')
    from aiohttp import web, WSMsgType
    from aiohttp.test_utils import TestClient, TestServer
    from dexstation.render.operator_stream import OperatorStream

    async def scenario():
        stream = OperatorStream.__new__(OperatorStream)
        stream.closing = False
        stream.auxiliary = True
        stream.peers = {}
        if cfg != 'missing':
            stream.cfg = cfg
        demands = asyncio.Queue()
        stream.demand = SimpleNamespace(write=lambda data: demands.put_nowait(json.loads(data)))
        app = web.Application()
        app.router.add_get('/aux-video', stream.video)
        async with TestClient(TestServer(app)) as client:
            async with client.ws_connect('/aux-video') as ws:
                await ws.send_json({'views': ['table', 'face', 'front',
                                             'back', 'left', 'right', 'face']})
                assert await asyncio.wait_for(demands.get(), 2) == expected
                assert not ws.closed
                await ws.send_json({'views': []})
                assert await asyncio.wait_for(demands.get(), 2) == []
                await ws.send_json({'views': 'face'})
                message=await ws.receive(timeout=2)
                assert message.type in (WSMsgType.CLOSE,WSMsgType.CLOSED)

    asyncio.run(scenario())
