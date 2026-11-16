"""API v1 contract tests with the real recorder and a CPU observation source."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip('aiohttp')
from aiohttp import ClientSession, CookieJar
from aiohttp.test_utils import TestServer

from dexstation.server.app import SessionManager, make_app
from dexstation.runtime.controller import CollectionController, CollectionOptions
from dexstation.runtime.health import InputHealth
from session_helpers import session_for
from helpers import observation


class Desk(SessionManager):
    def __init__(self,path):
        super().__init__(path)
        self.alive=False
        self.worker=SimpleNamespace(is_alive=lambda:self.alive)
        self.controller=None
    def refresh(self):
        return self.controller.status() if self.controller else self.latest
    async def start(self,v):
        self.output=self.root/'session-test';self.output.mkdir(exist_ok=True)
        options=CollectionOptions(v.get('backend','sapien'),v.get('source','replay'),self.output,None)
        self.controller=CollectionController(options)
        s=session_for(self.output)
        self.controller.session=s;self.controller.source=SimpleNamespace(boundary=lambda:([],7),health=InputHealth())
        self.controller.state='running';self.alive=True
    async def stop(self,client='service'):
        if self.alive:
            if self.controller.session.writer:await self.command(dict(action='stop',client=client))
            self.controller.session.close();self.controller.state='stopped';self.alive=False
        return {}
    async def command(self,event):
        self.controller.control(event)
        if event['action']=='start' and self.controller.session.writer:
            s=self.controller.session
            for _ in range(2):s.record(observation(s.last_sim_time_s+.02),[])
        return self.controller.control_result
    def control(self,event):
        self.controller.control(event)
    async def export(self,identifier):
        self.episode(identifier)
        return dict(exported=True,output='export-fixture')


def test_all_api_contracts_and_audit_force(tmp_path):
    async def scenario():
        m=Desk(tmp_path);server=TestServer(make_app(m));await server.start_server()
        url=str(server.make_url('')).rstrip('/')
        headers={'X-Gamepad-VLA-Client':'cli','Content-Type':'application/json'}
        async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as browser,ClientSession() as remote:
            async def call(path,data=None,*,session=remote,h=headers):
                async with session.request('GET' if data is None else 'POST',url+'/api/v1/'+path,json=data,headers=h) as r:
                    return r.status,await r.json()
            assert (await call('status'))[1]['remote_control_enabled'] is False
            for path,data in [('session/start',{}),('session/stop',{}),('recording/start',{}),('recording/stop',{}),('notes',{'text':'n'}),('episodes/missing/void',{}),('episodes/missing/export',{})]:
                code,r=await call(path,data);assert code==403 and r['error_code']=='remote_control_disabled'
                assert set(r)=={'error_code','message','state','safety'}
            assert (await call('remote-control',{'enabled':True}))[0]==403
            page=await browser.get(url+'/')
            cookie=page.cookies['collection_web_session']
            assert cookie['httponly'] and cookie['samesite']=='Strict'
            web_headers=dict(headers,**{'X-Gamepad-VLA-Client':'web','Origin':url})
            assert (await call('remote-control',{'enabled':True},session=browser,h=web_headers))[0]==200
            assert (await call('session/stop',{},h={'Content-Type':'application/json'}))[0]==403
            assert (await call('session/stop',{},h=dict(headers,Origin='https://invalid.example')))[0]==403
            assert (await call('session/stop',{},h={'X-Gamepad-VLA-Client':'cli'}))[0]==200  # json= supplies required Content-Type
            async with remote.post(url+'/api/v1/session/stop',data='{}',headers={'X-Gamepad-VLA-Client':'cli','Content-Type':'text/plain'}) as invalid:
                assert invalid.status==403
            assert (await call('recording/start',{}))[0]==409
            assert (await call('session/start',dict(backend='sapien',source='ros2')))[0]==200
            assert (await call('session/start',{}))[0]==409
            code,r=await call('recording/start',{});assert code==409 and r['error_code']=='input_unhealthy'
            assert all(k in r['message'] for k in ('left_hand','left_arm','right_hand','right_arm'))
            code,r=await call('recording/start',dict(force=True,note='forced test'));assert code==200
            identifier=r['episode_id'];assert identifier
            assert (await call('episodes/'+identifier+'/void',{}))[0]==409
            assert (await call('recording/start',{}))[0]==409
            assert (await call('notes',dict(text='note from CLI')))[0]==200
            assert (await call('recording/stop',{}))[0]==200
            m.controller.session.wait_saved()
            code,detail=await call('episodes/'+identifier)
            assert code==200 and detail['integrity']['passed']
            assert detail['meta']['forced'] and len(detail['meta']['recording_start_health'])==4
            assert detail['meta']['started_by']==detail['meta']['stopped_by']=='cli'
            assert (await call('recording/stop',{}))[0]==200
            assert (await call('episodes/missing'))[0]==400
            assert (await call('episodes/missing/void',{}))[0]==400
            assert (await call('episodes/missing/export',{}))[0]==400
            assert (await call('notes',{}))[0]==400
            assert (await call('remote-control',dict(enabled='yes'),session=browser,h=web_headers))[0]==400
            async with remote.get(url+'/api/v1/events') as r:
                assert r.headers['Content-Type'].startswith('text/event-stream')
                assert (await r.content.readline()).startswith(b'data: ')
            assert (await call('session/stop',{}))[0]==200
            assert (await call('session/stop',{}))[0]==200
            assert (await call('notes',dict(text='n')))[0]==409
            assert (await call('remote-control',dict(enabled=False),session=browser,h=web_headers))[0]==200
            assert (await call('session/stop',{}))[0]==403
        await server.close()
        logs=[json.loads(line) for line in (tmp_path/'api_audit.jsonl').read_text().splitlines()]
        assert not any(e['method']=='GET' and e['result']['http_status']<400 for e in logs)
        assert any(e['method']=='GET' and e['result']['http_status']>=400 for e in logs)
        assert any(e['client']=='cli' and e['result']['http_status']==409 for e in logs)
        assert any(e['endpoint']=='/api/v1/remote-control' and e['client']=='web' for e in logs)
        assert all(k in logs[0] for k in ('wall_time_ns','endpoint','parameters','result','client'))
        server=TestServer(make_app(Desk(tmp_path)));await server.start_server()
        async with ClientSession() as c:
            async with c.get(server.make_url('/api/v1/status')) as r:assert not (await r.json())['remote_control_enabled']
        await server.close()
    asyncio.run(scenario())


def test_idle_stop_and_read_polling_with_fake_clock(tmp_path):
    async def scenario():
        now=[0.]
        m=Desk(tmp_path)
        app=make_app(m,enable_remote_control=True,clock=lambda:now[0])
        server=TestServer(app);await server.start_server()
        headers={'X-Gamepad-VLA-Client':'cli','Content-Type':'application/json'}
        async with ClientSession() as cli:
            async with cli.post(server.make_url('/api/v1/session/start'),json={'source':'ros2'},headers=headers) as r:assert r.status==200
            # A live SSE connection keeps the simulation alive even without writes.
            async with cli.get(server.make_url('/api/v1/events')) as sse:
                assert (await sse.content.readline()).startswith(b'data: ')
                now[0]=10000
                await app['stop_idle']()
                assert m.alive
            # Wait for the server to observe the disconnect.
            for _ in range(100):
                if app['idle_session'].connections==0:break
                await asyncio.sleep(.01)
            assert app['idle_session'].connections==0
            # Browser polling also counts as presence; CLI read polling does not.
            async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as browser:
                async with browser.get(server.make_url('/')) as r:assert r.status==200
                now[0]+=7000
                async with browser.get(server.make_url('/api/v1/status')) as r:assert r.status==200
                now[0]+=7000
                await app['stop_idle']();assert m.alive
            now[0]+=200
            for path in ('status','episodes','config'):
                async with cli.get(server.make_url('/api/v1/'+path)) as r:assert r.status==200
            await app['stop_idle']()
            assert not m.alive
            async with cli.get(server.make_url('/api/v1/status')) as r:
                status=await r.json()
                assert status['recent_control_events'][-1]['operation']=='session/idle-stop'
        await server.close()
        control=[json.loads(line) for line in (m.output/'events.jsonl').read_text().splitlines()]
        assert sum(e.get('action')=='session/idle-stop' for e in control)==1
        audit=[json.loads(line) for line in (tmp_path/'api_audit.jsonl').read_text().splitlines()]
        assert all(e['method']=='POST' for e in audit)
    asyncio.run(scenario())


def test_service_commit_is_captured_at_startup(tmp_path,monkeypatch):
    from dexstation.runtime import provenance
    current={'git_commit':'a'*40,'dirty':False,'source_sha256':'test'}
    monkeypatch.setattr(provenance,'code_provenance',lambda root:dict(current))
    async def scenario():
        manager=SessionManager(tmp_path/'version')
        app=make_app(manager,frame_provider=lambda:{})
        server=TestServer(app);await server.start_server()
        try:
            current['git_commit']='b'*40
            async with ClientSession() as client:
                async with client.get(server.make_url('/api/v1/status')) as response:
                    state=await response.json()
                    assert state['service_code']['git_commit']=='a'*40
        finally:await server.close()
    asyncio.run(scenario())


def test_web_default_motion_reaches_worker_and_explicit_legacy_is_preserved(tmp_path):
    import queue
    import threading
    from dexstation.control.joint_motion import load_profile, resolve_profile

    async def scenario():
        captured=[]
        def no_hardware_chain():
            raise AssertionError('UDP test must not construct an input chain')
        manager=SessionManager(tmp_path,default_values={'source':'udp','replay':None},input_chain_factory=no_hardware_chain)
        class Worker:
            exitcode=0
            def __init__(self,options,statuses):self.options=options;self.statuses=statuses;self.alive=False
            def start(self):captured.append(self.options);self.alive=True
            def is_alive(self):return self.alive
            def join(self,timeout):self.alive=False;self.statuses.put({'session_state':'stopped','recording_status':{'state':'idle'}})
        manager.ctx=SimpleNamespace(Queue=lambda **kw:queue.Queue(),Event=threading.Event,
            Process=lambda *,target,args:Worker(args[0],args[3]))
        server=TestServer(make_app(manager));await server.start_server()
        try:
            async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as browser:
                async with browser.get(server.make_url('/')) as response:assert response.status==200
                async with browser.get(server.make_url('/api/v1/config')) as response:
                    config=await response.json()
                    assert config['defaults']['motion_profile']=='auto'
                    assert config['motion_profiles']==['auto','station','legacy']
                headers={'X-Gamepad-VLA-Client':'web'}
                for selected in (None,'legacy','station'):
                    body={} if selected is None else {'motion_profile':selected}
                    async with browser.post(server.make_url('/api/v1/session/start'),json=body,headers=headers) as response:assert response.status==200
                    options=captured[-1]
                    assert options.motion_profile==('auto' if selected is None else selected)
                    assert resolve_profile(options.motion_profile,options.source)==(None if selected=='legacy' else load_profile('station'))
                    async with browser.post(server.make_url('/api/v1/session/stop'),json={},headers=headers) as response:assert response.status==200
                assert not (await (await browser.get(server.make_url('/api/v1/status'))).json())['remote_control_enabled']
        finally:await server.close()
    asyncio.run(scenario())


def test_page_and_modules_revalidate_after_deploy(tmp_path):
    async def scenario():
        server=TestServer(make_app(Desk(tmp_path)));await server.start_server()
        try:
            async with ClientSession() as client:
                for path in ('/','/auxiliary-panel.mjs'):
                    async with client.get(server.make_url(path)) as response:
                        assert response.status==200
                        assert response.headers.get('Cache-Control')=='no-cache'
        finally:await server.close()
    asyncio.run(scenario())


def test_autopilot_source_is_offered_and_only_listed_task_sets_start(tmp_path):
    import queue
    import threading
    from dexstation.server.app import task_set_files
    from dexstation.control.joint_motion import load_profile, resolve_profile
    async def scenario():
        captured=[]
        manager=SessionManager(tmp_path,default_values={'source':'udp','replay':None},
                               input_chain_factory=lambda:(_ for _ in ()).throw(AssertionError('no input chain')))
        class Worker:
            exitcode=0
            def __init__(self,options,statuses):self.options=options;self.statuses=statuses;self.alive=False
            def start(self):captured.append(self.options);self.alive=True
            def is_alive(self):return self.alive
            def join(self,timeout):self.alive=False;self.statuses.put({'session_state':'stopped','recording_status':{'state':'idle'}})
        manager.ctx=SimpleNamespace(Queue=lambda **kw:queue.Queue(),Event=threading.Event,
            Process=lambda *,target,args:Worker(args[0],args[3]))
        server=TestServer(make_app(manager));await server.start_server()
        try:
            async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as browser:
                async with browser.get(server.make_url('/')) as response:assert response.status==200
                config=await (await browser.get(server.make_url('/api/v1/config'))).json()
                assert 'ik_autopilot' in config['sources'] and config['autopilot_tasks']==task_set_files()
                assert any(path.endswith('game_combos_v1.json') for path in config['autopilot_tasks'])
                headers={'X-Gamepad-VLA-Client':'web'}
                body={'source':'ik_autopilot','autopilot_tasks':config['autopilot_tasks'][0]}
                async with browser.post(server.make_url('/api/v1/session/start'),json=body,headers=headers) as response:assert response.status==200
                options=captured[-1]
                assert options.source=='ik_autopilot' and str(options.autopilot_tasks)==config['autopilot_tasks'][0]
                assert resolve_profile(options.motion_profile,options.source)==load_profile('station')
                async with browser.post(server.make_url('/api/v1/session/stop'),json={},headers=headers) as response:assert response.status==200
                async with browser.post(server.make_url('/api/v1/session/start'),json=dict(body,autopilot_tasks='/etc/passwd'),headers=headers) as response:
                    assert response.status>=400
                assert len(captured)==1
        finally:await server.close()
    asyncio.run(scenario())
