"""Source-oracle, timing, limits and stateful replay tests; no hardware nodes."""
from helpers import spec
import copy
import json
import math
from types import SimpleNamespace
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pytest

from dexstation.inputs.arm_mapping import ArmMapper, HIGH_DEG, LOW_DEG
from dexstation.control.commands import ArmCommand, hand_command_from_packet
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES
from dexstation.control.joint_motion import JointMotion, load_profile, resolve_profile
from dexstation.control.protocol import HandCommandPacket
from dexstation.model.station_math import ASSETS, StationMath

@pytest.fixture(scope='module')
def native(tmp_path_factory):return StationMath(tmp_path_factory.mktemp('motion-native'))


def model(native,**arm):
    p=load_profile('station');p['arm'].update(arm)
    return JointMotion(spec(),p,cache=None)

def test_native_mapping_matches_live_python_mapper(native):
    mapper=ArmMapper({s:[0]*7 for s in ('left','right')},startup='none')
    for side in ('left','right'):
        for raw in [[0]*7,[-1000]*7,[1000]*7,*np.random.default_rng(447).uniform(-300,300,(20,7))]:
            np.testing.assert_allclose(native.map(side,raw),np.rad2deg(mapper.command(side,raw,0).positions),atol=1e-12)

def test_filter_matches_frozen_actual_sdk_outputs(native):
    ref=json.loads((Path(__file__).parent/'fixtures/motion/sdk_filter_reference.json').read_text())
    # One signed sample per cutoff pins the SDK coefficient across its range.
    for x in ref['samples'][::8]:
        assert native.lowpass(x['dt'],x['input'],x['previous'],x['cutoff'])==pytest.approx(x['output'],abs=1e-14)
    assert native.lowpass(.001,1,0,100)==pytest.approx(.3858695450950375)

def test_configured_gains_reach_drives_and_effort_caps_remain_unmodeled(native):
    m=model(native);values=m.drive_parameters('left');profile=load_profile('station')
    for kind,count in (('arm',7),('hand',21)):
        for field,key in (('stiffness','stiffness_nm_rad'),('damping','damping_nm_s_rad')):
            np.testing.assert_array_equal(values[kind][field],[profile[kind][key]]*count)
    assert values['arm']['force_limit'] is values['hand']['force_limit'] is None

def test_per_axis_speed_ramp_and_deadband(native):
    m=model(native,filter_hz=None)
    def send(angle):m.accept([ArmCommand('left',(math.radians(angle),0,0,0,0,0,0),0)])
    send(.2);assert not m.active
    send(.6);send(.9);assert math.degrees(m.requested['left_arm'][0])==pytest.approx(.6)
    send(120)
    q=[m.advance_to(i*.001)[0].positions[0] for i in range(1000)]
    velocity=np.diff([0,*q])/.001
    assert velocity.max()<=math.radians(40)+1e-10
    assert math.degrees(velocity[0])==pytest.approx(2.)
    assert 19<math.degrees(q[-1])<21
    assert math.degrees(m.advance_to(1.999)[0].positions[0]-q[-1])==pytest.approx(40.)

def test_numerical_adapter_is_explicit_engine_specific_and_frozen(native):
    m=model(native)
    raw=m.drive_parameters('left');mj=m.drive_parameters('left',backend='mujoco');sp=m.drive_parameters('left',backend='sapien')
    np.testing.assert_array_equal(raw['hand']['armature'],[3.4707e-8]*21)
    np.testing.assert_array_equal(mj['hand']['armature'],raw['hand']['armature'])
    np.testing.assert_array_equal(sp['hand']['armature'],[1e-4]*21)
    for kind in ('arm','hand'):
        for key in ('stiffness','damping'):
            np.testing.assert_array_equal(sp[kind][key],raw[kind][key])
    frozen=m.metadata()['profile'];frozen['numerical_adapters']['sapien']['hand_armature_floor_kg_m2']=3e-5
    restored=JointMotion(spec(),frozen,cache=None)
    np.testing.assert_array_equal(restored.drive_parameters('left',backend='sapien')['hand']['armature'],[3e-5]*21)
    assert m.profile['numerical_adapters']['sapien']['hand_armature_floor_kg_m2']==1e-4
    frozen['numerical_adapters']['sapien']['hand_armature_floor_kg_m2']=float('nan')
    with pytest.raises(ValueError):load_profile(frozen)

def test_model_limits_intersect_mapping_config_and_margin(native):
    m=model(native,filter_hz=None,speed_ramp_s=0)
    # A narrower loaded model is a controlled input to the runtime's limit intersection.
    ranges=np.deg2rad([LOW_DEG,HIGH_DEG])*.9
    m.limits={s+'_arm':ranges.tolist() for s in ('left','right')}
    upper=ranges[1,5];margin=math.radians(m.profile['arm']['joint_limit_margin_deg'])
    m.accept([ArmCommand('left',(100.,)*7,0)])
    assert m.requested['left_arm'][5]==pytest.approx(upper-margin)
    assert m.requested['left_arm'][5]<upper
    m.advance_to(10.)
    assert m.filtered['left_arm'][5]<=upper-margin+1e-12

def test_hand_quantization_and_coupled_speed_bound(native):
    m=model(native);packet=HandCommandPacket('right','motor',[],[100.9]*20,[],[],0)
    m.accept([hand_command_from_packet(packet)])
    expected=hand_command_from_packet(HandCommandPacket('right','motor',[],[100.]*20,[],[],0))
    np.testing.assert_allclose(m.requested['right_hand'],list(expected.joint_positions.values()))
    p=copy.deepcopy(m.profile);p['hand']['max_speed_rad_s']=.5
    m=JointMotion(spec(),p,cache=None);m.accept([hand_command_from_packet(packet)])
    old=np.zeros(21)
    for i in range(100):
        q=np.array([m.advance_to(i*.001)[0].joint_positions[n] for n in G20_JOINT_NAMES])
        assert np.max(np.abs(q-old))<=.0005+1e-12;old=q

def test_auto_uses_station_for_live_and_preserves_episode_compatibility(tmp_path):
    assert resolve_profile('auto','ros2')==load_profile('station')
    assert resolve_profile('auto','udp')==load_profile('station')
    assert resolve_profile('legacy','ros2') is None
    (tmp_path/'meta.json').write_text(json.dumps({'motion_model':{'profile':load_profile('station')}}))
    assert resolve_profile('auto','replay',tmp_path) is None
    meta={'motion_initial_state':{'version':2},'motion_model':{'profile':load_profile('station')}}
    (tmp_path/'meta.json').write_text(json.dumps(meta))
    assert resolve_profile('auto','replay',tmp_path)==meta['motion_model']['profile']

@pytest.mark.parametrize('change',[dict(filter_hz=1001),dict(max_speed_deg_s=0),dict(deadband_deg=-1),dict(stiffness_nm_rad=-1),dict(damping_nm_s_rad=None)])
def test_invalid_profile_rejected(change):
    p=load_profile('station');p['arm'].update(change)
    with pytest.raises(ValueError):load_profile(p)

def test_bypass_and_backwards_clock(native):
    m=model(native,filter_hz=1000,speed_ramp_s=0)
    m.accept([ArmCommand('left',(1.,0,0,0,0,0,0),0)]);m.advance_to(0)
    np.testing.assert_array_equal(m.reference['left_arm'],m.filtered['left_arm'])
    with pytest.raises(ValueError):m.advance_to(-1)

def test_snapshot_restores_filter_ramp_targets_and_pending_events(native):
    live=model(native);live.accept([ArmCommand('left',(2.,0,0,0,0,0,0),0)])
    live.schedule(ArmCommand('left',(.2,0,0,0,0,0,0),0),2.015)
    live.advance_to(2.)
    replay=model(native);replay.restore(json.loads(json.dumps(live.snapshot())))
    for i in range(1,81):
        a=live.advance_to(2.+i*.0005);b=replay.advance_to(2.+i*.0005)
        assert a==b
        assert live.snapshot()==replay.snapshot()

def test_stateful_episode_replay_uses_input_journal_and_boundary_state(native,tmp_path):
    from dexstation.recording.episode import EpisodeWriter, read_episode
    from dexstation.inputs.replay import CollectionReplay, command_event
    from helpers import observation
    live=model(native);initial_command=ArmCommand('left',(2.,0,0,0,0,0,0),0)
    live.accept([initial_command]);live.schedule(ArmCommand('left',(.7,0,0,0,0,0,0),0),2.015);live.advance_to(2.)
    initial=live.snapshot();commands={s+'_'+k:None for s in ('left','right') for k in ('arm','hand')};commands['left_arm']=asdict(initial_command)
    meta={'scene_spec_text':'{}','fps':50,'motion_model':live.metadata(),'recording_start_sim_time_s':2.,
          'motion_initial_state':{'version':2,'physical':{'motion':initial},'commands':commands,'input_clock_origin_ns':0}}
    writer=EpisodeWriter(tmp_path/'episode',meta)
    events=[]
    for i,(when,q) in enumerate([(2.021,.3),(2.031,1.2)]):
        e=command_event(ArmCommand('left',(q,0,0,0,0,0,0),0),i);e['scheduled_sim_time_s']=when;events.append(e)
    writer.append_inputs(events);live.schedule_events(events,0);expected=[]
    for frame in range(2):
        for tick in range(20):live.advance_to(2.+(frame*20+tick+1)*.001)
        state=live.status();expected.append(state)
        obs=observation(2.+(frame+1)*.02)
        writer.append(obs,commands,{},motion_reference={k:state[k] for k in ('control_time_s','reference')})
    writer.close();source=CollectionReplay(tmp_path/'episode');replay=model(native);replay.restore(initial)
    for frame in range(2):
        source.poll((frame+1)*.02);replay.schedule_events(source.drain_events(),0)
        for tick in range(20):replay.advance_to(2.+(frame*20+tick+1)*.001)
        assert replay.status()==expected[frame]
    _,rows=read_episode(tmp_path/'episode');assert rows[0]['motion_reference']['reference']==expected[0]['reference']
    with pytest.raises(ValueError,match='cannot loop'):CollectionReplay(tmp_path/'episode',loop=True)


def test_status_exposes_actual_motion_mode_instead_of_requested_auto(tmp_path):
    from dexstation.runtime.controller import CollectionController, CollectionOptions
    controller=CollectionController(CollectionOptions('sapien','udp',tmp_path/'output',1.))
    assert controller.status()['motion_profile_resolved'] is None
    controller.backend=SimpleNamespace(motion=SimpleNamespace(profile=load_profile('station')))
    assert controller.status()['motion_profile_resolved']=='station'
    assert controller.status()['motion_profile_name']=='station-factory-baseline-v2'
    controller.backend=SimpleNamespace(motion=None)
    assert controller.status()['motion_profile_resolved']=='legacy'


def test_factory_joint_limits_have_one_source():
    import dexstation.inputs.arm_mapping as mapping
    import dexstation.control.joint_motion as motion
    from dexstation.control.workspace_limits import LOW_DEG, HIGH_DEG
    for module in (mapping, motion):
        np.testing.assert_array_equal(module.LOW_DEG, LOW_DEG)
        np.testing.assert_array_equal(module.HIGH_DEG, HIGH_DEG)
    np.testing.assert_array_equal(LOW_DEG, [-178,-120,-178,-60,-178,-60,-60])
    np.testing.assert_array_equal(HIGH_DEG, [178,120,178,145,178,60,60])
