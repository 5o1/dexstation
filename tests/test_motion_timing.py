"""Actual receiver/backend methods under controlled clocks and fake native surfaces."""
from types import SimpleNamespace
import json,math
import numpy as np
import pytest
from dexstation.control.joint_motion import JointMotion, load_profile
from dexstation.control.commands import ArmCommand
from dexstation.inputs.arm_mapping import ArmMapper
from dexstation.inputs.replay import Ros2Source
from dexstation.runtime.controller import CollectionController, CollectionOptions
from dexstation.backends.sapien_physics import CollectionSapienBackend
from dexstation.backends.mujoco_export import CollectionMujocoBackend
from helpers import spec

@pytest.fixture
def cache(tmp_path):return tmp_path/'cache'

def motion(cache,**arm):
 p=load_profile('station');p['arm'].update(arm);return JointMotion(spec(),p,cache=cache)

def test_100hz_pulse_reaches_control_on_30_and_60hz_display(monkeypatch,tmp_path,cache):
 results=[]
 for fps in (30,60):
  clock=[0.];callbacks={}
  class Node:
   def create_subscription(self,typ,topic,cb,qos):callbacks[topic]=cb
   def destroy_node(self):pass
  source=Ros2Source(ArmMapper({s:[0.]*7 for s in ('left','right')},startup='none'),node=Node(),joint_state_type=object,clock=lambda:clock[0])
  monkeypatch.setattr('dexstation.inputs.replay.time.monotonic_ns',lambda:round(clock[0]*1e9))
  m=motion(cache,filter_hz=None,deadband_deg=0.,max_speed_deg_s=1e6,speed_ramp_s=0)
  controller=CollectionController(CollectionOptions('sapien','ros2',tmp_path/f'out{fps}',.1))
  journal=[];controller.backend=SimpleNamespace(motion=m);controller.session=SimpleNamespace(record_inputs=lambda e:journal.extend(e));controller.input_clock_origin_ns=0
  events=[(0.,-90.),(.010,-80.),(.020,-90.)];next_event=0;trace=[];tick=0
  for frame in range(fps//10):
   end=(frame+1)/fps
   while next_event<len(events) and events[next_event][0]<=end:
    t,q=events[next_event];clock[0]=t
    callbacks['/left_arm_joint_control'](SimpleNamespace(name=[],position=[q,-90.,90.,0,0,0,0]));next_event+=1
   controller.consume_events(source.drain_events())
   while (tick+1)*.001<=end+1e-10:
    m.advance_to(tick*.001);trace.append(m.filtered['left_arm'][0]);tick+=1
  source.close();assert len(journal)==3
  assert sum(abs(x-math.radians(10))<1e-10 for x in trace)==10
  results.append(trace)
 np.testing.assert_allclose(results[0],results[1],atol=1e-12)

def test_actual_backend_step_methods_share_update_phase(cache):
 outputs={}
 for name,dt in [('sapien',.001),('mujoco',.0005)]:
  m=motion(cache,filter_hz=None,speed_ramp_s=0)
  m.accept([ArmCommand('left',(1.,0,0,0,0,0,0),0)])
  ref=[0.];records=[]
  def apply(commands):
   if commands:ref[0]=commands[0].positions[0]
  if name=='sapien':
   robot=SimpleNamespace(compute_passive_force=lambda **kw:[0.],set_qf=lambda _:None,get_qpos=lambda:np.zeros(1))
   f=SimpleNamespace(motion=m,_ticks=0,_requested_time=0.,robots={'left':robot},_has_robot_contact=lambda:False,_apply_targets=apply,
    scene=SimpleNamespace(step=lambda:records.append(ref[0])))
   CollectionSapienBackend.step(f,.002)
  else:
   enum=SimpleNamespace(mjWARN_BADQPOS=0,mjWARN_BADQVEL=1,mjWARN_BADQACC=2)
   mj=SimpleNamespace(mj_step=lambda *a:records.append(ref[0]),mj_forward=lambda *a:None,mjtWarning=enum)
   f=SimpleNamespace(motion=m,_physics_ticks=0,_requested_time=0.,_apply_targets=apply,mj=mj,model=SimpleNamespace(opt=SimpleNamespace(timestep=dt)),
    data=SimpleNamespace(qpos=np.zeros(1),qvel=np.zeros(1),contact=[],warning=[SimpleNamespace(number=0) for _ in range(3)]))
   CollectionMujocoBackend.step(f,.002)
  outputs[name]=records
 assert outputs['mujoco'][::2]==outputs['sapien']
 assert outputs['mujoco'][1::2]==outputs['sapien']

def test_filter_frequency_gain_and_phase_match_sdk_oracle(cache):
 from pathlib import Path
 oracle=json.loads((Path(__file__).parent/'fixtures/motion/sdk_frequency_reference.json').read_text())
 t=np.arange(1000)*.001
 # The cutoff-frequency case pins both gain and phase; scalar tests span cutoffs.
 for item in oracle['samples'][2:3]:
  freq=item['frequency_hz'];m=motion(cache,deadband_deg=0.,max_speed_deg_s=1e6,speed_ramp_s=0)
  values=[]
  for i,now in enumerate(t):
   m.accept([ArmCommand('left',(.5+.1*math.sin(2*math.pi*freq*now),0,0,0,0,0,0),now)]);m.advance_to(float(now));values.append(m.filtered['left_arm'][0])
  ratio=(2*np.mean(np.asarray(values)[500:]*np.exp(-2j*np.pi*freq*t[500:])))/(-.1j)
  assert abs(ratio)==pytest.approx(item['gain'],abs=1e-12)
  assert np.angle(ratio)==pytest.approx(item['phase_rad'],abs=1e-12)

def test_post_backend_reference_is_the_actual_clamped_actuator_target():
 from dexstation.model.constants import LOW_DEG, HIGH_DEG
 names=[f'joint_{i}' for i in range(7)]
 bounds=dict(zip(names,np.deg2rad([LOW_DEG,HIGH_DEG]).T))
 fake=SimpleNamespace(_arm_names={'left':names},data=SimpleNamespace(ctrl=np.zeros(7)),applied_reference={},model=SimpleNamespace(
  joint=lambda n:SimpleNamespace(limited=[True],range=bounds[n]),actuator=lambda n:SimpleNamespace(id=names.index(n.removeprefix('drive_')))))
 CollectionMujocoBackend._apply_targets(fake,[ArmCommand('left',(0,0,0,0,0,math.radians(67),math.radians(67)),0)])
 assert fake.applied_reference['left_arm'][5]==pytest.approx(bounds[names[5]][1])
 assert fake.applied_reference['left_arm']==fake.data.ctrl.tolist()

def test_udp_reception_is_independent_of_parent_poll(monkeypatch):
 import queue
 import time
 from dexstation.inputs.replay import CollectionUdpSource
 from dexstation.control.protocol import ArmCommandPacket, encode
 from dexstation.inputs import udp as sources
 datagrams = queue.Queue()
 completed = queue.Queue()
 class Socket:
  def bind(self, address):pass
  def setblocking(self, value):assert value is False
  def getsockname(self):return ('127.0.0.1', 12345)
  def recvfrom(self, size):
   try:return datagrams.get_nowait(), ('127.0.0.1', 54321)
   except queue.Empty:raise BlockingIOError from None
  def close(self):pass
 monkeypatch.setattr(sources.socket, 'socket', lambda *args: Socket())
 original = sources.UdpSource.poll
 def receive_step(receiver, now):
  before = receiver.received_packets
  try:
   result = original(receiver, now)
  except Exception as error:
   completed.put(error)
   raise
  if receiver.received_packets > before:
   completed.put(None)
  return result
 monkeypatch.setattr(sources.UdpSource, 'poll', receive_step)
 source = CollectionUdpSource('127.0.0.1', 0, stale_after_s=1., continuous=True)
 try:
  for i in range(2):
   datagrams.put(encode(ArmCommandPacket('left', [i*.1]*7, 0, 'slave-target-rad')))
   try:
    error = completed.get(timeout=2.)
   except queue.Empty:
    source.close()
    source.poll(time.monotonic())  # Surface the receiver's exception before a timeout failure.
    pytest.fail('receiver did not complete the injected datagram')
   if error is not None:
    source.close()
    source.poll(time.monotonic())
    pytest.fail(f'receiver failed without propagating its error: {error}')
   # Reception must finish before the parent polls; no scheduling latency assertion.
   assert source.received_packets == i + 1
   source.poll(time.monotonic())
  events = source.drain_events()
  assert len(events) == 2 and events[0]['received_monotonic_ns'] < events[1]['received_monotonic_ns']
 finally:
  source.close()
 assert not source._thread.is_alive()


def test_udp_receiver_failure_reaches_parent_poll(monkeypatch):
 import threading
 from dexstation.inputs.replay import CollectionUdpSource
 from dexstation.inputs.udp import UdpSource
 entered = threading.Event()
 failure = OSError('injected receiver failure')
 def fail(receiver, now):
  entered.set()
  raise failure
 monkeypatch.setattr(UdpSource, 'poll', fail)
 source = CollectionUdpSource('127.0.0.1', 0, stale_after_s=1., continuous=True)
 try:
  assert entered.wait(timeout=2.)
  source.close()  # Join synchronizes publication of the thread error.
  with pytest.raises(RuntimeError, match='UDP receiver failed') as caught:
   source.poll(0.)
  assert caught.value.__cause__ is failure
 finally:
  source.close()

def test_replay_physics_and_view_use_the_recorded_scene(monkeypatch,tmp_path):
 import dexstation.runtime.controller as module
 import dexstation.render.operator_view as view_module
 from dexstation.model.spec import load_spec, spec_to_dict
 from dexstation.runtime.health import InputHealth
 from helpers import observation
 from pathlib import Path
 from dataclasses import replace
 from dexstation.recording.episode import EpisodeWriter
 from dexstation.contracts.profiles import RuntimeScene
 scene=module.compose_options(module.CollectionOptions('sapien','udp',tmp_path/'unused',1.))
 scene=replace(scene,spec=replace(scene.spec,name='frozen-recorded-scene'))
 episode=tmp_path/'episode'
 profile=load_profile('station')
 writer=EpisodeWriter(episode,dict(fps=50,recording_start_sim_time_s=0.,motion_model={'profile':profile},
     motion_initial_state=dict(workbench=dict(physical={},commands={},input_clock_origin_ns=0),task={})),scene=scene)
 writer.append(observation(.02),{});writer.append(observation(.04),{});writer.close()
 seen={};clock=[2.]
 class Source:
  health=InputHealth();decode_errors=0
  def poll(self,now):return []
  def drain_events(self):return []
  def boundary(self):return [],0
  def close(self):pass
 class Backend:
  def __init__(self,spec,**kw):
   seen['physics']=(spec.spec if isinstance(spec,RuntimeScene) else spec).name;self.work_dir=Path(kw['work_dir']);self.work_dir.mkdir()
   seen['collisions']=(kw['environment_collisions'],kw['gamepad_collisions'])
   self.motion_drives={};self.applied_reference={}
   self.motion=SimpleNamespace(metadata=lambda:{'profile':profile},schedule_events=lambda *a:None,input_commands={},input_sequences={},status=lambda:{'control_time_s':2.,'reference':{}})
  def reset(self):pass
  def apply(self,commands):pass
  def restore_motion_state(self,state):seen['restored']=True
  def snapshot_motion_state(self):return {}
  def observe(self):return observation(clock[0])
  def step(self,dt):clock[0]+=dt
  def close(self):pass
 class Recorder:
  writer=None
  def __init__(self,*a,**kw):self.meta={};self.commands={};Path(a[0]).mkdir()
  def record_inputs(self,*a):pass
  def record_runtime_metrics(self,*a):pass
  def record(self,obs,commands,**kw):return obs.to_dict()
  def close(self,*a):pass
  def status(self):return {'state':'idle'}
 class View:
  def __init__(self,*args,**kw):seen['view']=args[3].spec.name;seen['scene']=args[3]
  def wait_ready(self,*a):pass
  def events(self):return []
  def publish(self,*a):pass
  def close(self,**kw):pass
 monkeypatch.setattr(module,'configure_runtime',lambda:None)
 monkeypatch.setattr(__import__('dexstation.inputs.factories', fromlist=['_']), 'CollectionReplay',lambda *a,**kw:Source())
 monkeypatch.setattr(module,'create_backend',lambda name,spec,**kw:Backend(spec,**kw))
 monkeypatch.setattr(module,'Session',Recorder);monkeypatch.setattr(view_module,'OperatorView',View)
 options=module.CollectionOptions('sapien','replay',tmp_path/'out',.02,replay=episode,rate_hz=50,view=True)
 controller=module.CollectionController(options);controller.run()
 assert seen['physics']==seen['view']=='frozen-recorded-scene'
 assert seen['restored'] and isinstance(seen['scene'],RuntimeScene)
 assert seen['collisions']==(True,True) and options.scene_variant=='full'
 meta=json.loads((episode/'episode.json').read_text());meta['identity']['backend']['name']='mujoco'
 (episode/'episode.json').write_text(json.dumps(meta))
 with pytest.raises(ValueError,match='recorded physics backend'):
  module.CollectionController(options).run()

def test_sapien_reset_restores_state_without_rewriting_articulation_properties():
 calls=[]
 m=SimpleNamespace(reset=lambda:calls.append('motion'),drive_parameters=lambda *a:(_ for _ in ()).throw(AssertionError('immutable properties must be configured before activation')))
 b=SimpleNamespace(motion=m,_reset_robot_pose=lambda:calls.append('pose'),_initial_gamepad_q={},
  gamepad=SimpleNamespace(set_joint_positions=lambda q:None,articulation=SimpleNamespace(set_qvel=lambda q:None)))
 CollectionSapienBackend.reset(b)
 assert calls==['pose','motion'] and b._ticks==0 and b._requested_time==0 and b.applied_reference=={}
