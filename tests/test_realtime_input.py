"""Controlled slow native physics, real controller and JointMotion; no hardware.

The native-step path uses a deterministic virtual clock, including a recording stall.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexstation.runtime.controller import CollectionController, CollectionOptions
from dexstation.runtime.health import InputHealth
from dexstation.backends.sapien_physics import CollectionSapienBackend
from dexstation.control.commands import ArmCommand
from dexstation.control.joint_motion import JointMotion
from helpers import observation
from helpers import spec


@pytest.mark.parametrize('source_kind', ['ros2', 'udp'])
def test_slow_physics_tracks_latest_input_through_recording_stall(monkeypatch, tmp_path, source_kind):
    import dexstation.runtime.controller as module
    import dexstation.config.site as site
    import io
    monkeypatch.setattr(site,'hardware_lock_path',lambda:tmp_path/'station-hardware.lock')
    clock=[10.]
    now = lambda:clock[0]
    ns = lambda:round(clock[0]*1e9)
    def sleep(dt):clock[0]+=dt
    monkeypatch.setattr(module.time,'monotonic',now)
    monkeypatch.setattr(module.time,'monotonic_ns',ns)
    monkeypatch.setattr(module.time,'sleep',sleep)
    origin=now();origin_ns=ns()
    m=JointMotion(spec(),cache=tmp_path/'native')
    journal=[];measurements=[];published=[]
    def target(index):return .3 if index%2 else .7
    class Source:
        health=InputHealth(now);decode_errors=0
        received_packets=0
        def drain_events(self):
            events=[]
            due=math.floor((now()-origin)*100+1e-6)
            while self.received_packets <= due:
                i=self.received_packets
                command=ArmCommand('left',(target(i),0,0,0,0,0,0),i/100)
                events.append(dict(sequence=i,channel='left_arm',converted_command=asdict(command),
                    received_monotonic_ns=origin_ns+i*10_000_000,received_wall_time_ns=i*10_000_000,
                    raw={},header={},error=None))
                self.received_packets+=1
            return events
        def boundary(self):return self.drain_events(),self.received_packets
        def poll(self,now):return []
        def close(self):pass
    source=Source()
    class SlowBackend:
        motion=m;motion_drives={};robots={};applied_reference={}
        _has_robot_contact=lambda self:False
        def __init__(self):
            self.scene=SimpleNamespace(step=self.native_step)
            self._ticks=0;self._requested_time=0.
            self.deadline=now()
        def reset(self):pass
        def native_step(self):
            sequence=m.input_sequences.get('left_arm')
            if sequence is not None:
                age=(ns()-(origin_ns+sequence*10_000_000))/1e9
                newest=source.received_packets-1  # latest available at the step's receive boundary
                measurements.append((age,sequence,newest,float(m.requested['left_arm'][0])))
            # 1 ms physics costs 1.25 ms wall time: RTF = 0.8.
            self.deadline+=.00125
            sleep(.00125)
        def _apply_targets(self,commands):pass
        def step(self,dt):CollectionSapienBackend.step(self,dt)
        def observe(self):return observation(self._ticks*.001)
        def close(self):pass
    class Recorder:
        writer=None
        blocked=False
        def __init__(self,*a,**kw):
            self.meta={};self.commands={};self.control_log=io.StringIO();Path(a[0]).mkdir()
        def record_inputs(self,events):
            journal.extend(events)
            if source_kind=='udp' and len(journal)>2000 and not self.blocked:
                self.blocked=True;sleep(.25)  # recording I/O stalls, receiver keeps running
        def record_runtime_metrics(self,*a):pass
        def record(self,obs,commands,**kw):return obs.to_dict()
        def status(self):return dict(state='idle')
        def close(self,*a):pass
    backend=SlowBackend()
    monkeypatch.setattr(module,'configure_runtime',lambda:None)
    monkeypatch.setattr(module,'create_backend',lambda *a,**kw:backend)
    monkeypatch.setattr(module,'Session',Recorder)
    monkeypatch.setattr(__import__('dexstation.inputs.factories', fromlist=['_']), 'Ros2Source',lambda *a,**kw:source)
    monkeypatch.setattr(__import__('dexstation.inputs.factories', fromlist=['_']), 'CollectionUdpSource',lambda *a,**kw:source)
    controller=CollectionController(CollectionOptions('sapien',source_kind,tmp_path/'out',48.,rate_hz=30),publish=published.append)
    result=controller.run()
    assert now()-origin >= 60.-1e-6
    assert measurements and max(x[0] for x in measurements) <= .100
    assert all(seq == newest and requested == pytest.approx(target(seq)) for _,seq,newest,requested in measurements)
    assert len(m.pending)<=1  # final close boundary may snapshot one unstepped target
    metrics=published[-1]['realtime_metrics']
    assert metrics['rtf']==pytest.approx(.8,abs=.03)
    assert metrics['max_applied_input_delay_s'] <= .100
    if source_kind == 'udp':assert controller.session.blocked


def test_stall_coalesces_channels_and_drops_stale_targets_but_keeps_raw_journal(tmp_path):
    from dexstation.runtime.loop import LatestInput
    from dexstation.inputs.replay import command_event
    policy=LatestInput()
    events=[]
    for i in range(40):
        for side in ('left','right'):
            e=command_event(ArmCommand(side,(i/100,0,0,0,0,0,0),0),len(events))
            e['received_monotonic_ns']=i*10_000_000
            events.append(e)
    selected,journal=policy.select(events,400_000_000,.32)
    assert [e['sequence'] for e in selected]==[78,79]
    assert policy.dropped==78
    assert len(journal)==len(events) and 'realtime_discarded' not in events[0]
    assert all(e['scheduled_sim_time_s']==.32 for e in journal)
    policy.mark_applied({'left_arm':78,'right_arm':79},400_000_000)
    assert policy.select([dict(events[-1],sequence=80)],600_000_000,.4)[0]==[]
    assert policy.dropped==79 and set(policy.status(600_000_000)['stale_channels'])=={'left_arm','right_arm'}
    assert policy.status(600_000_000)['input_delay_s'] is None


def test_replay_keeps_every_event_and_reproduces_live_discard_decisions(tmp_path):
    from dexstation.inputs.replay import CollectionReplay, command_event
    from dexstation.recording.episode import EpisodeWriter
    events=[command_event(ArmCommand('left',(v,0,0,0,0,0,0),0),i)
        for i,v in enumerate((.2,.8,.4))]
    for i,e in enumerate(events):
        e['received_monotonic_ns']=i*10_000_000
        e['scheduled_sim_time_s']=i*.01
    for discarded in (False,True):
        copied=[dict(e) for e in events]
        if discarded:copied[1]['realtime_discarded']=True
        path=tmp_path/str(discarded)
        w=EpisodeWriter(path,dict(scene_spec_text='{}',fps=50,
            motion_initial_state={'input_clock_origin_ns':0},recording_start_sim_time_s=0.))
        w.append_inputs(copied)
        for t in (.02,.04):w.append(observation(t),dict.fromkeys(('left_arm','right_arm','left_hand','right_hand')),{})
        w.close()
        replay=CollectionReplay(path,stateful=True)
        replay.poll(.03);drained=replay.drain_events()
        m=JointMotion(spec(),cache=tmp_path/'native')
        c=CollectionController(CollectionOptions('sapien','replay',tmp_path/'out',.03,replay=path))
        c.backend=SimpleNamespace(motion=m);c.session=SimpleNamespace(record_inputs=lambda e:None);c.input_clock_origin_ns=0
        c.consume_events(drained)
        trace=[]
        for tick in range(30):
            m.advance_to(tick*.001);trace.append(m.requested['left_arm'][0])
        assert trace[:10]==pytest.approx([.2]*10)
        assert trace[10:20]==pytest.approx([.2 if discarded else .8]*10)
        assert trace[20:]==pytest.approx([.4]*10)
        assert len(drained)==3  # even discarded inputs remain auditable


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_ros_arm_nonfinite_is_decode_error_and_never_a_command(bad):
    from dexstation.inputs.replay import Ros2Source
    callbacks={}
    class Node:
        def create_subscription(self,typ,topic,callback,qos):callbacks[topic]=callback
        def destroy_node(self):pass
    mapper=SimpleNamespace(command=lambda *a:pytest.fail('nonfinite reached mapper'))
    source=Ros2Source(mapper,node=Node(),joint_state_type=object,status_type=object)
    for side in ('left','right'):
        for joint in range(7):
            values=[0.]*7;values[joint]=bad
            callbacks[f'/{side}_arm_joint_control'](SimpleNamespace(name=[],position=values))
    assert source.poll(source.clock())==[] and source.decode_errors==14
    assert all(event['converted_command'] is None and event['error'] for event in source.drain_events())
    assert all(not value['input_available'] for key,value in source.health.snapshot().items() if key.endswith('_arm'))
    source.close()


def test_ros_sides_stale_independently_and_linkerta_reason_reaches_health():
    from dexstation.inputs.replay import Ros2Source
    from dexstation.inputs.arm_mapping import ArmMapper
    callbacks={};clock=[10.]
    class Node:
        def create_subscription(self,typ,topic,callback,qos):callbacks[topic]=callback
        def destroy_node(self):pass
    source=Ros2Source(ArmMapper({s:[0.]*7 for s in ('left','right')}),node=Node(),
        joint_state_type=object,status_type=object,clock=lambda:clock[0],topic_prefix='/test')
    callbacks['/test/left_arm_joint_control'](SimpleNamespace(name=[],position=[0.]*7))
    status=dict(interface=dict(name='can9',up=False),sides={s:dict(publishing=False,reason='can9 down') for s in ('left','right')})
    callbacks['/test/linkerta/status'](SimpleNamespace(data=json.dumps(status)))
    health=source.health.snapshot()
    assert health['left_arm']['input_available'] and not health['right_arm']['input_available']
    assert health['right_arm']['input_reason']=='can9 down'
    assert len(source.poll(clock[0]))==1  # measured zero is valid
    clock[0]+=.26
    assert source.poll(clock[0])==[]
    assert source.health.snapshot()['left_arm']['input_reason']=='can9 down'
    clock[0]+=2.
    assert source.health.snapshot()['left_arm']['input_reason']=='LinkerTA status stale'
    source.close()


def test_arm_endpoint_gid_is_required_and_foreign_publisher_immediately_removes_input():
    from dexstation.inputs.replay import Ros2Source, snapshot_input_events
    from dexstation.inputs.arm_mapping import ArmMapper
    from dexstation.inputs.linkerta_reader import READER_NODE
    N = SimpleNamespace
    callbacks={};clock=[10.];own=N(node_name=READER_NODE,endpoint_gid=b'own')
    foreign=N(node_name='linkerta_node',endpoint_gid=b'vendor')
    graph={s:[own] for s in ('left','right')}
    class Node:
        def create_subscription(self,typ,topic,callback,qos):callbacks[topic]=callback
        def get_publishers_info_by_topic(self,topic):return graph['left' if 'left' in topic else 'right']
        def destroy_node(self):pass
    source=Ros2Source(ArmMapper({s:[0.]*7 for s in ('left','right')}),node=Node(),joint_state_type=object,clock=lambda:clock[0])
    msg=SimpleNamespace(name=[],position=[0.]*7)
    cb=callbacks['/left_arm_joint_control']
    cb(msg,N(publisher_gid=b'own'))
    assert len(source.poll(clock[0]))==1
    # The new sender is blocked even before the 1 Hz graph refresh is due.
    graph['left']=[own,foreign]
    cb(msg,N(publisher_gid=b'vendor'))
    assert source.poll(clock[0])==[]
    assert 'left_arm' not in snapshot_input_events(source)
    assert source.health.snapshot()['left_arm']['input_reason']=='foreign publisher on /left_arm_joint_control: linkerta_node'
    cb(msg,N(publisher_gid=b'own'))
    assert source.poll(clock[0])==[]  # entire side is blocked while both exist
    graph['left']=[own];clock[0]+=1.01
    cb(msg,N(publisher_gid=b'own'))
    assert len(source.poll(clock[0]))==1
    # A GID missing from graph evidence never becomes a command.
    cb(msg,N(publisher_gid=b'not-yet-discovered'))
    assert source.poll(clock[0])==[]
    source.close()
