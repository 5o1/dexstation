"""WP8a composed CLI/controller/API acceptance; no physics or hardware devices."""
import asyncio
from concurrent.futures import Future
from dataclasses import asdict, replace
import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
from types import SimpleNamespace as N, ModuleType

import numpy as np
import pytest

from dexstation.runtime import controller
from dexstation.server import app as server
from dexstation.recording.session import Session
from dexstation.runtime.health import InputHealth
from dexstation.recording.integrity import check_integrity
from dexstation.contracts.backend import FakeBackend
from dexstation.contracts.profiles import RuntimeScene
from dexstation.contracts.limits import FakeLimits, LimitReport, AngularViolation
from dexstation.contracts.takeover import FakeTakeover
from dexstation.recording.episode import read_episode
from dexstation.control.commands import ArmCommand, HandCommand
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES
from dexstation.profiles.compose import load_task
from dexstation.runtime import plugins


def register_test_plugin(monkeypatch, module):
    monkeypatch.setitem(plugins._TASKS, "tabletop_cube_grasp", plugins.TaskPlugin(
        "tabletop_cube_grasp", module, Path(__file__).resolve().parents[1]/"src/dexstation/task_plugins/tabletop_cube_grasp/data", Path("/unused")))


class SavePool:
    def __init__(self, **kwargs):pass
    def submit(self, fn, *args):
        future=Future()
        try:future.set_result(fn(*args))
        except Exception as exc:future.set_exception(exc)
        return future
    def shutdown(self, **kwargs):pass


class SceneBackend(FakeBackend):
    name='sapien'
    def __init__(self, scene):
        super().__init__()
        if scene.task['id']=='xbox_gamepad_v3_3':
            self._initial.task=dict(gamepad_signal=np.arange(32,dtype=float)/31,
                gamepad_mask=np.arange(32)%2==0, control_joint_positions={'button_a':.001})
        elif scene.task['id']=='tabletop_cube_grasp':
            self._initial.task=dict(cube_pose=list(scene.objects[0].initial_pose_world),cube_contact=[False,False])
        self.reset()
    def restore_motion_state(self,state):
        from dexstation.contracts.backend import RobotObservation
        state=dict(state,workbench=dict(state['workbench']))
        if isinstance(state['workbench'].get('robot'),dict):
            state['workbench']['robot']=RobotObservation(**state['workbench']['robot'])
        super().restore_motion_state(state)
    def step(self, dt):
        super().step(dt)
        self._observation.robot.wall_time_ns+=round(dt*1e9)


@pytest.fixture
def simulated(monkeypatch):
    import dexstation.recording.session as collection
    monkeypatch.setattr(collection,'fingertip_offsets',lambda spec:{})
    monkeypatch.setattr(collection,'ProcessPoolExecutor',SavePool)
    monkeypatch.setattr(controller,'configure_runtime',lambda:None)
    monkeypatch.setattr(controller,'create_backend',lambda name,scene,**kw:SceneBackend(scene))


def scene_for(task='empty', seed=17):
    return controller.compose_options(controller.CollectionOptions('sapien','udp',Path('/tmp/unused'),1.,task=task,seed=seed))


def replay_npz(path):
    names=[];units=[]
    for side in ('left','right'):
        names.extend(f'{side}_arm_joint_{j}' for j in range(1,8));units.extend(['rad']*7)
        names.extend(f'{side}_hand_motor_{j}' for j in range(20));units.extend(['motor_0_255']*20)
    np.savez(path,action=np.zeros((2,54)),timestamp=[0.,.02],names=names,units=units,episode_index=[0,0])
    return path


@pytest.mark.parametrize('task', ['empty','xbox_gamepad_v3_3'])
def test_cli_records_composed_task_layers(simulated, monkeypatch, tmp_path, task):
    path=Path(__file__).parents[1]/'scripts/run_collection.py'
    spec=importlib.util.spec_from_file_location('wp8a_cli',path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    replay=replay_npz(tmp_path/'states.npz');output=tmp_path/'session'
    monkeypatch.setattr(sys,'argv',[str(path),'--backend','sapien','--task',task,'--seed','17',
        '--source','replay','--replay',str(replay),'--output',str(output),'--record','--duration','.1','--rate-hz','50','--arm-startup','none'])
    monkeypatch.setattr(sys,'stdin',io.StringIO())
    cli.main()
    episode=read_episode(output/'episode-000000')
    assert check_integrity(output/'episode-000000')['passed']
    assert episode.meta['identity']['task']['id']==task
    assert episode.workbench.meta['scene_check']==dict(checked=False,reason='initial-state geometry check pending WP5')
    if task=='empty':
        assert episode.task is None and len(episode.meta['layers'])==1
    else:
        assert len(episode.meta['layers'])==2
        assert episode.task.format=='xbox-gamepad-rec-v1'
        for wb,row in zip(episode.workbench.rows,episode.task.rows):
            assert len(row['gamepad_signal'])==len(row['gamepad_mask'])==32
            assert (wb['step'],wb['sim_time_s'])==(row['step'],row['sim_time_s'])
    assert not (output/'episode-000000/meta.json').exists()


def test_frozen_scene_reconstruction_ignores_current_profiles(simulated,monkeypatch,tmp_path):
    replay=replay_npz(tmp_path/'states.npz')
    output=tmp_path/'session'
    c=controller.CollectionController(controller.CollectionOptions('sapien','replay',output,.06,task='xbox_gamepad_v3_3',record=True,replay=replay,rate_hz=50))
    c.run()
    episode=read_episode(output/'episode-000000')
    import dexstation.profiles.compose as profiles
    monkeypatch.setattr(profiles,'load_workbench',lambda *args:pytest.fail('read current workbench'))
    monkeypatch.setattr(profiles,'load_task',lambda *args:pytest.fail('read current task'))
    frozen=controller.scene_from_episode(episode)
    assert frozen.task==c.scene.task
    assert frozen.identity==c.scene.identity
    assert [asdict(o) for o in frozen.objects]==[asdict(o) for o in c.scene.objects]


def test_missing_task_format_and_invalid_initial_state_fail_before_backend(simulated,monkeypatch,tmp_path):
    import dexstation.profiles.compose as profiles
    original=profiles.load_task
    def no_format(task):
        value=original(task);value.pop('recording',None);return value
    monkeypatch.setattr(profiles,'load_task',no_format)
    with pytest.raises(ValueError,match='xbox_gamepad_v3_3.*recording format'):
        scene_for('xbox_gamepad_v3_3')
    monkeypatch.setattr(profiles,'load_task',original)
    monkeypatch.setattr(controller,'validate_initial_state',lambda scene:dict(checked=True,ok=False,reason='test collision'))
    monkeypatch.setattr(controller,'create_backend',lambda *a,**kw:pytest.fail('backend started'))
    options=controller.CollectionOptions('sapien','replay',tmp_path/'out',.04,replay=replay_npz(tmp_path/'states.npz'))
    with pytest.raises(ValueError,match='test collision'):
        controller.CollectionController(options).run()


def test_limits_hold_only_violating_arm_and_record_workbench_event(simulated,tmp_path):
    class Limits(FakeLimits):
        def check(self, arm_q, hand_q):
            bad=arm_q['left'][0]>.2
            return LimitReport(not bad,[],[AngularViolation('left','J1',120.,0.,'azimuth')] if bad else [],{}, {})
    class Takeover(FakeTakeover):
        def __init__(self):self.calls=[]
        def update(self,*args):self.calls.append(args);return super().update(*args)
    scene=scene_for();backend=SceneBackend(scene)
    session=Session(tmp_path/'session',scene,backend,{'kind':'fake'})
    session.control({'action':'start'})
    takeover=Takeover();gate=controller.ControlGate(scene,backend,session,Limits(),takeover)
    hand=HandCommand('left',dict.fromkeys(G20_JOINT_NAMES,.3),0.)
    first=gate.filter([ArmCommand('left',(.1,0,0,0,0,0,0),0.),hand])
    second=gate.filter([ArmCommand('left',(.5,0,0,0,0,0,0),0.),ArmCommand('right',(.4,0,0,0,0,0,0),0.),hand])
    assert first[0].positions[0]==second[0].positions[0]==.1
    assert second[1].positions[0]==.4 and second[2] is hand
    assert any(args[1]=='left' and args[-1] is False for args in takeover.calls)
    session.close()
    events=[json.loads(line) for line in (session.output/'episode-000000/workbench/events.jsonl').read_text().splitlines()]
    assert any(e['kind']=='limit' and e['payload']['side']=='left' for e in events)


def test_plugin_actions_and_builtin_rng_state(simulated,monkeypatch):
    scene=scene_for('tabletop_cube_grasp')
    task=dict(scene.task,actions=dict(scene.actions,nudge={'plugin':'nudge','while_recording':'allowed'}))
    scene=replace(scene,task=task,actions=task['actions'])
    module=ModuleType('dexstation.task_plugins.tabletop_cube_grasp')
    module.nudge=lambda ctx,params: (ctx.set_object_state('cube',params['pose']), {'value':params['value']})[1]
    module.cell_state=lambda task:{'extra':{'channel_count':len(task['channels'])}}
    register_test_plugin(monkeypatch, module)
    a,b=controller.TaskRuntime(scene),controller.TaskRuntime(scene)
    a.backend=SceneBackend(scene);b.backend=SceneBackend(scene)
    a.latest={'cube_pose':[1]*7,'cube_contact':[False,True]}
    state=a.status()
    assert state['channels']['cube_pose']==[1]*7
    assert state['cell_state']['cube']['cube_pose']==[1]*7
    assert state['cell_state']['result']=={'value':None,'error':'online evaluation not available'}
    assert state['cell_state']['extra']=={'channel_count':2}
    assert a.action('reset',{})==b.action('reset',{})
    assert a.action('nudge',{'pose':[0,0,1,1,0,0,0],'value':7})=={'value':7}
    assert a.backend.observe().object_poses['cube']==[0,0,1,1,0,0,0]
    a.cell_callback=lambda task:{'cube':{'plugin':True}}
    assert a.status()['cell_state']['cube']=={'plugin':True}
    module.nudge=1
    with pytest.raises(ValueError,match='not callable'):
        controller.TaskRuntime(scene)


def test_module_cells_have_no_default_payload_and_empty_has_no_plugin_import(monkeypatch):
    task=controller.TaskRuntime(scene_for('xbox_gamepad_v3_3'))
    task.latest={'gamepad_signal':[0]*32}
    assert task.status()['cell_state']=={'signals':{'gamepad_signal':[0]*32}}
    import importlib
    monkeypatch.setattr(importlib,'import_module',lambda *a:pytest.fail('imported optional plugin'))
    value=controller.TaskRuntime(scene_for()).status()
    assert value['channels']==value['actions']==value['cell_state']=={}


def test_null_limits_are_disabled_without_altering_targets(tmp_path):
    scene=scene_for()
    path=tmp_path/'limits.json';path.write_text(json.dumps({'spatial':{'enabled':True,'margin_m':None}}))
    limits,status=controller.load_control_limits(replace(scene,limits_path=path))
    assert limits is None and status['enabled'] is False and 'null' in status['reason']


def test_debug_variants_and_seed_validation(tmp_path):
    for variant in ('gamepad-contact','other'):
        with pytest.raises(ValueError,match='scene variant'):
            controller.CollectionOptions('sapien','udp',tmp_path,1.,scene_variant=variant).validate()
    with pytest.raises(ValueError,match='cannot be recorded'):
        controller.CollectionOptions('sapien','udp',tmp_path,1.,scene_variant='free-space',record=True).validate()
    with pytest.raises(ValueError,match='seed'):
        controller.CollectionOptions('sapien','udp',tmp_path,1.,seed=True).validate()


class ComposedDesk(server.SessionManager):
    """Real controller/Session/task runtime, fake backend and input at API boundary."""
    def __init__(self, path):
        super().__init__(path)
        self.alive=False
        self.worker=N(is_alive=lambda:self.alive)
        self.controller=None
    def refresh(self):
        return self.controller.status() if self.controller else self.latest
    async def start(self, values):
        task=values.get('task','empty')
        self.output=self.root/task/'session-api'
        options=controller.CollectionOptions('sapien','udp',self.output,None,task=task,seed=values.get('seed'),scene_variant=values.get('scene_variant','full'))
        c=self.controller=controller.CollectionController(options)
        c.scene=controller.compose_options(options)
        c.task_runtime=controller.TaskRuntime(c.scene)
        c.backend=SceneBackend(c.scene);c.task_runtime.backend=c.backend
        c.task_runtime.on_plugin_error=c.record_plugin_error
        c.session=Session(self.output,c.scene,c.backend,{'kind':'fake'},fps=50,
            task_format=c.scene.task.get('recording',{}).get('format'))
        c.source=N(health=InputHealth(),boundary=lambda:([],0))
        c.state='running';self.alive=True
        self.scene=c.scene
        return {'started':True}
    async def stop(self, client='service'):
        if self.alive:
            self.controller.session.close();self.controller.state='stopped';self.alive=False
        return {'stopped':True}
    def control(self, event):
        self.controller.control(event)
        if event['action']=='stop':self.controller.session.wait_saved()
    async def command(self, event):
        c=self.controller;c.control(event)
        if event['action']=='start' and c.session.writer:
            for _ in range(2):
                c.backend.step(.02)
                obs=c.backend.observe();c.task_runtime.latest=obs.task
                c.last_row=c.session.record(obs,[])
        elif event['action']=='stop':
            c.session.wait_saved()
        return c.control_result


@pytest.mark.parametrize('task', ['empty','xbox_gamepad_v3_3'])
def test_composed_api_record_stop_list_detail_void_export(simulated,monkeypatch,tmp_path,task):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer
    original=server.site_value
    monkeypatch.setattr(server,'site_value',lambda key,default=None:sys.executable if key=='collection.export_python' else original(key,default))
    exports=[]
    async def export(*args,**kw):
        exports.append(args)
        async def communicate():return b'fake export completed',None
        return N(returncode=0,communicate=communicate)
    monkeypatch.setattr(server.asyncio,'create_subprocess_exec',export)
    async def scenario():
        m=ComposedDesk(tmp_path/'sessions')
        service=TestServer(server.make_app(m,enable_remote_control=True,frame_provider=lambda:{}))
        await service.start_server()
        headers={'X-Gamepad-VLA-Client':'test'}
        async with ClientSession() as client:
            async def call(path,body=None):
                async with client.request('GET' if body is None else 'POST',service.make_url('/api/v1/'+path),json=body,headers=headers) as r:
                    result=await r.json();assert r.status==200,result;return result
            config=await call('config')
            assert config['defaults']['task']=='empty'
            assert config['workbench']['name']=='ar5_08_g20_bench_bimanual'
            assert config['workbench']['cameras'][0]['name']=='camera_3'
            assert config['backends'][0]['check']=='import+version'
            assert {item['id'] for item in config['tasks']}=={'empty','xbox_gamepad_v3_3','tabletop_cube_grasp'}
            cells=await call('tasks/'+task+'/cells')
            for cell in cells['cells']:
                if 'module' in cell:assert cell['module'].startswith('/tasks/'+task+'/web/')
            await call('session/start',{'task':task,'seed':23})
            assert m.controller.scene.task['id']==task and m.controller.options.seed==23
            state=await call('status')
            assert state['task']['id']==task
            assert all(k in state for k in ('takeover','limits','hardware_lease','scene_check','workbench'))
            started=await call('recording/start',{})
            identifier=started['episode_id']
            assert identifier.startswith(task+'~session-')
            await call('recording/stop',{})
            rows=await call('episodes?task='+task)
            assert len(rows)==1 and rows[0]['task']==task
            assert await call('episodes?task=unused')==[]
            detail=await call('episodes/'+identifier)
            assert detail['meta']['status']=='complete' and detail['integrity']['passed']
            assert (await call('episodes/'+identifier+'/export',{}))['exported']
            assert (await call('episodes/'+identifier+'/void',{'reason':'synthetic'}))['discarded']
            await call('session/stop',{})
        await service.close()
        assert len(exports)==1 and exports[0][2]==str(m.output/'episode-000000')
        assert read_episode(m.output/'episode-000000').task is None if task=='empty' else True
    asyncio.run(scenario())


def test_task_web_traversal_permissions_and_action_events(simulated,monkeypatch,tmp_path):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer
    web=tmp_path/'web';web.mkdir();(web/'test.mjs').write_text('export const value=1;')
    (tmp_path/'secret.txt').write_text('private')
    monkeypatch.setattr(server,'task_web_directory',lambda task:web)
    async def scenario():
        m=ComposedDesk(tmp_path/'sessions')
        service=TestServer(server.make_app(m,frame_provider=lambda:{}))
        await service.start_server()
        async with ClientSession() as remote:
            async with remote.post(service.make_url('/api/v1/task/action'),json={'action':'reset'},headers={'X-Gamepad-VLA-Client':'test'}) as r:assert r.status==403
            for path,code in [('/task-cells.mjs',200),('/tasks/tabletop_cube_grasp/web/test.mjs',200),
                ('/tasks/tabletop_cube_grasp/web/missing.mjs',404),('/tasks/tabletop_cube_grasp/web/%2e%2e%2fsecret.txt',404)]:
                async with remote.get(service.make_url(path)) as r:assert r.status==code
        # Preserve the established cookie/Origin/header authorization protocol.
        from aiohttp import CookieJar
        async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as browser:
            await browser.get(service.make_url('/'))
            headers={'X-Gamepad-VLA-Client':'web','Origin':str(service.make_url('')).rstrip('/')}
            async def call(path,body):
                async with browser.post(service.make_url('/api/v1/'+path),json=body,headers=headers) as r:return r.status,await r.json()
            assert (await call('session/start',{'task':'tabletop_cube_grasp','seed':7}))[0]==200
            assert (await call('recording/start',{}))[0]==200
            code,reply=await call('task/action',{'action':'reset','params':{}})
            assert code==409 and not reply['accepted']
            assert (await call('recording/stop',{}))[0]==200
            code,reply=await call('task/action',{'action':'reset','params':{}})
            assert code==200 and reply['accepted'] and 'cube' in reply['result']['object_poses']
            state=m.refresh()['task']
            assert state['cell_state']['result']['value'] is None
            events=[json.loads(line) for line in (m.output/'episode-000000/task/tabletop_cube_grasp/events.jsonl').read_text().splitlines()]
            assert any(e['kind']=='action' and not e['payload']['reply']['accepted'] for e in events)
            assert len([json.loads(line) for line in (m.output/'events.jsonl').read_text().splitlines() if json.loads(line)['action']=='task/action'])==2
        await service.close()
    asyncio.run(scenario())


def test_backend_probe_reports_import_version_scope(monkeypatch):
    import importlib
    import importlib.metadata
    monkeypatch.setattr(importlib,'import_module',lambda name:N())
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'3.0.3+concon.physx5.6.1.a3' if name=='sapien' else '0.4.3')
    assert controller.backend_availability()==dict(name='sapien',available=True,reason=None,check='import+version')
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'incorrect')
    value=controller.backend_availability()
    assert not value['available'] and 'required 3.0.3+' in value['reason']


def test_gate_checks_each_native_control_period_and_uses_physics_time(simulated,tmp_path):
    scene=scene_for();backend=SceneBackend(scene);backend._ticks=0
    calls=[]
    class Limits(FakeLimits):
        def check(self, arm_q, hand_q):calls.append((dict(arm_q),dict(hand_q)));return super().check(arm_q,hand_q)
    times=[]
    class Takeover(FakeTakeover):
        def update(self,t,*args):times.append(t);return super().update(t,*args)
    session=Session(tmp_path/'session',scene,backend,{'kind':'fake'})
    gate=controller.ControlGate(scene,backend,session,Limits(),Takeover())
    for tick in range(3):
        backend._ticks=tick
        gate.filter([ArmCommand('left',(.1,0,0,0,0,0,0),0.)])
    assert len(calls)==3 and times==[0.,0.,.001,.001,.002,.002]
    session.close()


def test_camera_name_routes_forward_selection_without_changing_started_set(monkeypatch,tmp_path):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer
    calls=[]
    class Camera:
        def status(self,camera_name=None):calls.append(('status',camera_name));return {'selected_camera':camera_name}
        def preview_jpeg(self,camera_name=None):calls.append(('preview',camera_name));return b'jpeg'
        def photos(self,camera_name=None):calls.append(('photos',camera_name));return []
        def photo(self,name,camera_name=None):calls.append(('photo',camera_name));return image
        def snapshot(self,label='',camera_name=None):calls.append(('snapshot',camera_name));return {'label':label}
        def start(self):calls.append(('start',None));return {}
        def stop(self):return {}
        def record_start(self):return {}
        def record_stop(self):return {}
        def record_state(self):return {}
    image=tmp_path/'photo.jpg';image.write_bytes(b'jpeg')
    async def scenario():
        m=server.SessionManager(tmp_path/'sessions')
        service=TestServer(server.make_app(m,camera=Camera(),enable_remote_control=True,frame_provider=lambda:{}))
        await service.start_server()
        async with ClientSession() as client:
            for route in ('status','preview.jpg','photos','photos/photo.jpg'):
                async with client.get(service.make_url('/api/v1/camera/'+route+'?camera_name=camera_7')) as r:assert r.status==200
            async with client.post(service.make_url('/api/v1/camera/snapshot?camera_name=camera_7'),json={'label':'test'},headers={'X-Gamepad-VLA-Client':'test'}) as r:assert r.status==200
            async with client.post(service.make_url('/api/v1/camera/start?camera_name=camera_7'),json={},headers={'X-Gamepad-VLA-Client':'test'}) as r:assert r.status==200
        await service.close()
        assert all((name,'camera_7') in calls for name in ('status','preview','photos','photo','snapshot'))
        assert ('start',None) in calls
    asyncio.run(scenario())


def test_composed_view_receives_robot_and_object_state(simulated,monkeypatch,tmp_path):
    import dexstation.render.operator_view as operator
    frames=[]
    class View:
        def __init__(self,backend,*args,**kw):assert isinstance(backend,SceneBackend)
        def wait_ready(self,*args):pass
        def events(self):return []
        def publish(self,row):frames.append(row)
        def close(self,**kwargs):pass
    monkeypatch.setattr(operator,'OperatorView',View)
    options=controller.CollectionOptions('sapien','replay',tmp_path/'out',.04,
        task='xbox_gamepad_v3_3',replay=replay_npz(tmp_path/'states.npz'),view=True,rate_hz=50)
    controller.CollectionController(options).run()
    assert frames and all('hand_q' in row and 'arm_q' in row and 'object_poses' in row for row in frames)
    assert frames[-1]['task']['id']=='xbox_gamepad_v3_3'
    assert len(frames[-1]['task']['channels']['gamepad_signal'])==32


def test_task_action_replay_runs_at_recorded_boundary(simulated,tmp_path):
    scene=scene_for('tabletop_cube_grasp')
    options=controller.CollectionOptions('sapien','udp',tmp_path/'session',1.)
    c=controller.CollectionController(options)
    c.scene=scene;c.task_runtime=controller.TaskRuntime(scene);c.backend=SceneBackend(scene)
    c.task_runtime.backend=c.backend
    c.session=Session(options.output,scene,c.backend,{'kind':'fake'},task_format=scene.task['recording']['format'])
    c.replay_task_origin=1.
    c.replay_task_events=[dict(kind='action',payload=dict(name='reset',params={},reply={'accepted':True},sim_time_s=1.02))]
    c.replay_actions(.01)
    assert c.replay_task_index==0 and c.backend.observe().object_poses=={}
    c.replay_actions(.02)
    assert c.replay_task_index==1 and 'cube' in c.backend.observe().object_poses
    poses=c.backend.observe().object_poses
    c.replay_actions(.03)
    assert c.backend.observe().object_poses==poses
    c.session.close()


def test_layered_episode_replay_records_same_task_without_current_profiles(simulated,monkeypatch,tmp_path):
    replay=replay_npz(tmp_path/'states.npz')
    first=tmp_path/'first'
    controller.CollectionController(controller.CollectionOptions('sapien','replay',first,.06,
        task='xbox_gamepad_v3_3',record=True,replay=replay,rate_hz=50)).run()
    import dexstation.profiles.compose as profiles
    monkeypatch.setattr(profiles,'load_workbench',lambda *a:pytest.fail('read current workbench'))
    monkeypatch.setattr(profiles,'load_task',lambda *a:pytest.fail('read current task'))
    second=tmp_path/'second'
    controller.CollectionController(controller.CollectionOptions('sapien','replay',second,.06,
        record=True,replay=first/'episode-000000',rate_hz=50)).run()
    episode=read_episode(second/'episode-000000')
    assert episode.task is not None and episode.meta['identity']['task']['id']=='xbox_gamepad_v3_3'
    assert check_integrity(second/'episode-000000')['passed']


def test_empty_rejects_task_action_while_recording_without_absent_layer_error(simulated,tmp_path):
    async def scenario():
        m=ComposedDesk(tmp_path)
        await m.start({'task':'empty'})
        await m.command({'action':'start'})
        reply=await m.command({'action':'task-action','name':'reset','params':{}})
        assert not reply['accepted'] and reply['error']=='Unknown task action'
        await m.command({'action':'stop'});await m.stop()
        assert read_episode(m.output/'episode-000000').task is None
    asyncio.run(scenario())


def test_debug_free_space_api_cannot_start_recording(simulated,tmp_path):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer
    async def scenario():
        manager=ComposedDesk(tmp_path)
        service=TestServer(server.make_app(manager,enable_remote_control=True,frame_provider=lambda:{}))
        await service.start_server()
        try:
            async with ClientSession() as client:
                headers={'X-Gamepad-VLA-Client':'test'}
                async with client.post(service.make_url('/api/v1/session/start'),json={'task':'empty','scene_variant':'free-space'},headers=headers) as r:assert r.status==200
                async with client.post(service.make_url('/api/v1/recording/start'),json={},headers=headers) as r:
                    value=await r.json()
                    assert r.status==409 and value['error_code']=='debug_variant' and 'free-space' in value['message']
                assert manager.controller.session.writer is None
        finally:await service.close()
    asyncio.run(scenario())


@pytest.fixture
def plugin_desk(simulated, monkeypatch, tmp_path):
    module=ModuleType('dexstation.task_plugins.tabletop_cube_grasp')
    module.cell_state=lambda task:{}
    register_test_plugin(monkeypatch, module)
    c=controller.CollectionController(controller.CollectionOptions('sapien','udp',tmp_path/'session',None,task='tabletop_cube_grasp'))
    scene=scene_for('tabletop_cube_grasp')
    cells=(*scene.cells,{'id':'module','title':'Module','module':'web/fake.mjs'})
    c.scene=replace(scene,cells=cells,task=dict(scene.task,cells=list(cells)))
    c.backend=SceneBackend(c.scene)
    c.session=Session(c.options.output,c.scene,c.backend,{'kind':'fake'},task_format=c.scene.task['recording']['format'])
    c.task_runtime=controller.TaskRuntime(c.scene)
    c.task_runtime.backend=c.backend
    c.task_runtime.on_plugin_error=c.record_plugin_error
    c.task_runtime.latest={'cube_pose':[1]*7,'cube_contact':[False,True]}
    yield c
    c.session.close()


def builtin_plugin_payloads():
    return {'cube':{'cube_pose':[1]*7,'cube_contact':[False,True]},
        'result':{'value':None,'error':'online evaluation not available'}}


def plugin_events(c):
    return [row for row in map(json.loads,(c.session.output/'events.jsonl').read_text().splitlines())
            if row['action']=='task/plugin-error']


def fail_plugin(task):
    raise ValueError('plugin\nfailed')


def test_plugin_first_failure_uses_only_builtin_payloads(plugin_desk):
    c=plugin_desk
    c.task_runtime.cell_callback=fail_plugin
    for _ in range(3):
        c.backend.step(.001)
        state=c.status()['task']
        assert state['cell_state']==builtin_plugin_payloads()
        assert state['plugin_error']=='ValueError: plugin failed'
        assert 'module' not in state['cell_state']
    assert len(plugin_events(c))==1
    c.control({'action':'note','text':'controller still responds'})
    assert c.control_result['accepted']
    assert len(plugin_events(c))==1


def test_plugin_success_then_failure_discards_every_plugin_result(plugin_desk):
    c=plugin_desk
    c.task_runtime.cell_callback=lambda task:{'cube':{'override':True},'module':{'value':7}}
    state=c.status()['task']
    assert state['plugin_error'] is None
    assert state['cell_state']['cube']=={'override':True}
    assert state['cell_state']['module']=={'value':7}
    c.task_runtime.cell_callback=fail_plugin
    state=c.status()['task']
    assert state['cell_state']==builtin_plugin_payloads()
    assert state['plugin_error']=='ValueError: plugin failed'
    assert len(plugin_events(c))==1


@pytest.mark.parametrize('bad', [None,[],17,'bad',{1:{}},{'cube':{'override':True},'module':object()}, {'cube':float('nan')}])
def test_plugin_invalid_result_is_rejected_as_a_whole(plugin_desk,bad):
    c=plugin_desk
    c.task_runtime.cell_callback=lambda task:bad
    state=c.status()['task']
    assert state['cell_state']==builtin_plugin_payloads()
    assert state['plugin_error'].startswith(('TypeError:','ValueError:'))
    assert '\n' not in state['plugin_error'] and len(plugin_events(c))==1


def test_plugin_recovers_and_logs_only_changed_errors(plugin_desk):
    c=plugin_desk
    c.task_runtime.cell_callback=fail_plugin
    for _ in range(5):c.status()
    def other_error(task):raise RuntimeError('second error')
    c.task_runtime.cell_callback=other_error
    for _ in range(5):c.status()
    c.task_runtime.cell_callback=lambda task:{'module':{'value':'recovered'}}
    for _ in range(5):
        task=c.status()['task']
        assert task['plugin_error'] is None and task['cell_state']['module']=={'value':'recovered'}
    assert [event['plugin_error'] for event in plugin_events(c)]==[
        'ValueError: plugin failed','RuntimeError: second error',None]
    c.session.close()
    c.task_runtime.cell_callback=fail_plugin
    assert c.status()['task']['plugin_error']=='ValueError: plugin failed'


def test_plugin_error_does_not_break_http_status_or_note(simulated,monkeypatch,tmp_path):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer
    module=ModuleType('dexstation.task_plugins.tabletop_cube_grasp')
    mode=[True]
    def callback(task):
        if mode[0]:raise ValueError('plugin\nfailed')
        return {}
    module.cell_state=callback
    register_test_plugin(monkeypatch, module)
    async def scenario():
        manager=ComposedDesk(tmp_path/'sessions')
        service=TestServer(server.make_app(manager,enable_remote_control=True,frame_provider=lambda:{}))
        await service.start_server()
        try:
            async with ClientSession() as client:
                headers={'X-Gamepad-VLA-Client':'test'}
                async with client.post(service.make_url('/api/v1/session/start'),json={'task':'tabletop_cube_grasp'},headers=headers) as response:
                    assert response.status==200
                    assert (await response.json())['state']['task']['plugin_error']=='ValueError: plugin failed'
                for _ in range(3):
                    async with client.get(service.make_url('/api/v1/status')) as response:
                        assert response.status==200
                        assert (await response.json())['task']['plugin_error']=='ValueError: plugin failed'
                async with client.post(service.make_url('/api/v1/notes'),json={'text':'still running'},headers=headers) as response:
                    assert response.status==200
                mode[0]=False
                async with client.get(service.make_url('/api/v1/status')) as response:
                    assert response.status==200 and (await response.json())['task']['plugin_error'] is None
                assert [event['plugin_error'] for event in plugin_events(manager.controller)]==['ValueError: plugin failed',None]
        finally:await service.close()
    asyncio.run(scenario())
