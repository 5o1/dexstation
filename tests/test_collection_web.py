"""Collection web data-quality, lifecycle and read-only safety contracts."""
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import queue
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dexstation.runtime.controller import CollectionController, CollectionOptions
from dexstation.runtime.health import InputHealth, detect_safety, require_loopback
from dexstation.recording.integrity import check_integrity, discard_episode
from dexstation.recording.episode import EpisodeWriter, read_episode
from dexstation.inputs.arm_mapping import ArmMapper
from helpers import observation
from session_helpers import session_for


def episode(path, *, events=()):
    writer = EpisodeWriter(path, dict(scene_spec_text='{}', fps=50, recording_start_sim_time_s=0,
        recording_end_sim_time_s=.04, received_input_messages=len(events),
        receive_start_sequence=events[0]['sequence'] if events else 0,
        receive_end_sequence=events[-1]['sequence']+1 if events else 0))
    writer.append_inputs(events)
    for t in (.02,.04):
        writer.append(observation(t), {}, {})
    writer.close()
    return path


def test_discard_does_not_change_raw_data(tmp_path):
    path=episode(tmp_path/'ep')
    before={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}
    discard_episode(path,'bad trial')
    assert json.loads((path/'discarded.json').read_text())['discarded']
    assert before=={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file() and p!=path/'discarded.json'}
    assert read_episode(path)[0]['status']=='complete'


def test_health_frequency_errors_and_stale():
    clock=[0.];health=InputHealth(lambda:clock[0])
    assert all(c['warning'] for c in health.snapshot().values())
    for i in range(200):
        clock[0]=i/100;health.receive('left_arm',error=i==5)
    c=health.snapshot()['left_arm']
    assert c['hz']==pytest.approx(100) and c['count']==200 and c['decode_errors']==1
    clock[0]+=.251
    assert health.snapshot()['left_arm']['warning']




@pytest.mark.parametrize('host',['0.0.0.0','192.0.2.1','::'])
def test_nonloopback_refused(host):
    with pytest.raises(ValueError):require_loopback(host)


def test_controller_receive_boundaries_and_stop(tmp_path):
    # Use the real lifecycle/recorder with the simulator replaced by observations.
    session=session_for(tmp_path)
    source=SimpleNamespace(boundary=lambda:([],7),health=InputHealth())
    controller=CollectionController(CollectionOptions('sapien','udp',tmp_path,1.))
    controller.session=session;controller.source=source
    controller.control(dict(action='start'))
    for t in (.02,.04):session.record(observation(t),[])
    controller.control(dict(action='stop'))
    assert session.recording_state=='saving'
    session.wait_saved()
    assert session.recording_state=='saved'
    meta,rows=read_episode(tmp_path/'episode-000000')
    assert meta['received_input_messages']==0 and meta['receive_start_sequence']==7
    assert check_integrity(tmp_path/'episode-000000')['passed']
    # Optional timing statistics must survive EpisodeWriter.close and stay local
    # to each recording, including the first frame's drops.
    session.record_runtime_metrics(dict(rtf=1.,input_delay_s=.01,dropped_inputs=7))
    session.control(dict(action='start'))
    for t,rtf,delay,drops in ((.06,.8,.02,9),(.08,1.,.01,10)):
        session.record_runtime_metrics(dict(rtf=rtf,input_delay_s=delay,dropped_inputs=drops,input_delay_limit_s=.1))
        session.record(observation(t),[])
    # Finish directly so the injected counters aren't replaced by this test's
    # controller (which has no actual live source).
    session.finish();session.wait_saved()
    stats=read_episode(tmp_path/'episode-000001')[0]['realtime_metrics']
    assert stats['rtf']['mean']==pytest.approx(.9)
    assert stats['input_delay_s']['max']==.02 and stats['dropped_inputs']==3
    session.close()


def test_manager_single_session_and_path_boundary(tmp_path, monkeypatch):
    from dexstation.server.app import SessionManager
    m=SessionManager(tmp_path)
    m.worker=SimpleNamespace(is_alive=lambda:True)
    with pytest.raises(ValueError,match='only one'):
        asyncio.run(m.start({}))
    with pytest.raises(ValueError,match='within'):
        m.episode('../outside')
def test_manager_export_environment_for_layered_episode(tmp_path, monkeypatch):
    from dexstation.server.app import SessionManager
    m=SessionManager(tmp_path)
    import dexstation.server.app as server
    monkeypatch.setattr(server, 'site_value', lambda *args: str(tmp_path/'missing-python'))
    path=episode(tmp_path/'session-1/episode-000000')
    with pytest.raises(ValueError,match='environment unavailable'):
        asyncio.run(m.export(str(path.relative_to(tmp_path))))


def test_controller_final_status_does_not_publish_to_closed_view(tmp_path,monkeypatch):
    import dexstation.runtime.controller as module
    closed=[False]
    class Source:
        decode_errors=0;health=InputHealth()
        def boundary(self):return [],0
        def drain_events(self):return []
        def poll(self,now):return []
        def close(self):pass
    class Backend:
        def reset(self):pass
        def apply(self,commands):pass
        def step(self,dt):pass
        def observe(self):return observation()
        def close(self):pass
    class Recorder:
        writer=None
        meta={}
        def __init__(self,*args,**kwargs):pass
        def record_inputs(self,events):pass
        def record_runtime_metrics(self,metrics):pass
        def record(self,obs,commands,**kwargs):return obs.to_dict()
        def status(self):return dict(state='idle')
        def close(self,status):pass
    class View:
        def __init__(self,*args,**kwargs):pass
        def wait_ready(self,stop):pass
        def publish(self,row):assert not closed[0]
        def events(self):return []
        def close(self,**kwargs):closed[0]=True
    import dexstation.render.operator_view as stream
    monkeypatch.setattr(stream,'OperatorView',View)
    monkeypatch.setattr(module,'configure_runtime',lambda:None)
    monkeypatch.setattr(__import__('dexstation.inputs.factories', fromlist=['_']), 'CollectionReplay',lambda *a,**kw:Source())
    monkeypatch.setattr(module,'create_backend',lambda *a,**kw:Backend())
    monkeypatch.setattr(module,'Session',Recorder)
    controller=CollectionController(CollectionOptions('sapien','replay',tmp_path,.02,replay=tmp_path/'fake',view=True,rate_hz=50))
    assert controller.run()['ticks']==1
    assert controller.state=='stopped' and closed[0]


def test_profile_test_double_without_arm_geometry_keeps_zero_offsets(tmp_path, monkeypatch):
    from dataclasses import replace
    from helpers import bench_scene, spec
    import dexstation.runtime.controller as module
    import dexstation.control.joint_motion as motion

    scene = replace(bench_scene('empty'), spec=spec())
    monkeypatch.setattr(module, 'configure_runtime', lambda: None)
    monkeypatch.setattr(motion, 'resolve_profile', lambda *args: None)
    monkeypatch.setattr(module, 'load_control_limits', lambda scene: (None, {}))

    class SourceReached(RuntimeError):
        pass

    def source(*args, arm_mapper, **kwargs):
        assert all(offset.tolist() == [0.] * 7 for offset in arm_mapper.joint_offset.values())
        raise SourceReached('zero-offset mapper reached the inert source boundary')

    monkeypatch.setattr(__import__('dexstation.inputs.factories', fromlist=['_']), 'CollectionUdpSource', source)
    controller = CollectionController(CollectionOptions('sapien', 'udp', tmp_path, .01,
                                                        runtime_scene=scene))
    with pytest.raises(SourceReached):
        controller.run()


def test_manager_stopped_means_child_has_exited(tmp_path):
    from dexstation.server.app import SessionManager
    m=SessionManager(tmp_path)
    alive=[True]
    m.worker=SimpleNamespace(is_alive=lambda:alive[0],exitcode=0)
    m.latest=dict(session_state='stopped')
    assert m.refresh()['session_state']=='stopping'
    alive[0]=False
    assert m.refresh()['session_state']=='stopped'


def test_stop_closes_only_owned_operator_children(tmp_path):
    from dexstation.render.operator_view import OperatorView
    import io
    calls=[]
    view=OperatorView.__new__(OperatorView)
    view.run=tmp_path;view.collection=True;view.backend=None;view.roles=('render',)
    view.children=[SimpleNamespace(wait=lambda **kw:calls.append('wait'),returncode=0)]
    view.logs=[];view.timing=io.StringIO();view.snapshot=SimpleNamespace(close=lambda:calls.append('close'))
    view.close(cancelled=True)
    assert (tmp_path/'stop').exists() and calls==['wait','close']


def test_episode_list_uses_actual_active_measurements(tmp_path):
    from dexstation.server.app import SessionManager
    m=SessionManager(tmp_path)
    m.output=tmp_path/'session-live'
    path=m.output/'episode-000000'
    writer=EpisodeWriter(path,dict(scene_spec_text='{}',recording_start_sim_time_s=10.,fps=50))
    m.latest=dict(recording_status=dict(episode='episode-000000',rows=100,duration_s=2.,notes=[]))
    row=m.episodes()[0]
    assert row['rows']==100 and row['duration_s']==2.
    m.latest={}
    assert m.episodes()[0]['duration_s'] is None
    writer.close()




def test_color_conversion_preserves_clipping_and_rounding():
    pytest.importorskip('cv2')
    from dexstation.cameras.image import color_to_rgb8
    picture=np.random.default_rng(3).uniform(-.1,1.1,(48,64,4)).astype(np.float32)
    picture.reshape(-1)[:8]=[-1,0,1,2,.5,.1,127.5/255,128.5/255]
    expected=np.rint(np.clip(picture[...,:3],0.,1.)*255).astype(np.uint8)
    assert np.array_equal(color_to_rgb8(picture),expected)
    rgba8=np.random.default_rng(4).integers(0,256,(48,64,4),dtype=np.uint8)
    assert np.array_equal(color_to_rgb8(rgba8),rgba8[...,:3])


def test_two_recordings_never_reuse_previous_integrity(tmp_path):
    from concurrent.futures import Future
    from dexstation.recording.integrity import check_and_store, store_integrity
    s=session_for(tmp_path/'session')
    class Pool:
        future=None;path=None
        def submit(self,function,path):self.future=Future();self.path=path;return self.future
        def shutdown(self,**kwargs):pass
    pool=Pool();s.save_pool=pool
    for index in (0,1):
        s.control(dict(action='start'))
        current=s.status()['integrity'];assert current['state']=='pending' and current['episode_id'].endswith(f'{index:06d}')
        for _ in range(2):s.record(observation(s.last_sim_time_s+.02),[])
        s.writer.meta.update(received_input_messages=0,receive_start_sequence=0,receive_end_sequence=0)
        s.control(dict(action='stop'))
        store_integrity(pool.path,dict(current,state='running'));s.last_integrity_poll=0
        status=s.status()
        assert status['state']=='checking' and status['integrity']['state']=='running'
        assert status['integrity']['episode_id']==current['episode_id'] and status['integrity']['passed'] is None
        if index==1:
            assert status['last_saved']['episode_id']==current['episode_id']
        pool.future.set_result(check_and_store(pool.path));s.update_save()
        assert s.status()['integrity']['state']=='passed'
    s.close()



def test_safety_policy_is_supplied_by_the_caller(tmp_path):
    from dexstation.runtime.health import hardware_process_identity
    argv = ['python', '-m', 'example_driver.node']
    assert hardware_process_identity(argv) is None
    assert hardware_process_identity(argv, prefixes=('example_driver',)) == 'example_driver.node'
    process = tmp_path / '123'
    process.mkdir()
    (process / 'cmdline').write_bytes(b'python\0-m\0example_driver.node\0')
    assert detect_safety(tmp_path)['safe']
    state = detect_safety(tmp_path, identity=lambda args: hardware_process_identity(args, prefixes=('example_driver',)))
    assert not state['safe'] and state['matches'][0]['identity'] == 'example_driver.node'
