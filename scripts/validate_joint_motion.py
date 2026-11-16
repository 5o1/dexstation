#!/usr/bin/env python3
"""Bounded CPU validation of the experimental joint model. No ROS or hardware nodes.

The one-joint force-law fixture is a measurement rig, not another collection or
rendering pipeline. Full-body cases reuse the production collection backend.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import importlib.metadata
import numpy as np

from dexstation.config.defaults import DEFAULT_TASK, DEFAULT_WORKBENCH
from dexstation.recording.session import create_backend, Session
from dexstation.inputs.replay import CollectionReplay, command_event
from dexstation.control.commands import ArmCommand, HandCommand
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES, G20_MIMIC_JOINTS
from dexstation.control.joint_motion import load_profile
from dexstation.config.site import configure_runtime
from dexstation.profiles.compose import compose, load_task, load_workbench
from dexstation.recording.episode import read_episode, read_input_events


def rig(backend,kp,kd,armature):
    """Known inertia/zero-gravity hinge with force-mode PD and no contacts."""
    if backend=='sapien':
        import sapien
        scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);dt=.001;scene.set_timestep(dt)
        b=scene.create_articulation_builder();root=b.create_link_builder();root.set_name('base')
        link=b.create_link_builder(root);link.set_name('test_link');link.set_joint_name('joint')
        link.set_mass_and_inertia(1.,sapien.Pose(),[1.,1.,1.])
        link.set_joint_properties('revolute',limits=[[-2.,2.]],pose_in_parent=sapien.Pose(),pose_in_child=sapien.Pose())
        robot=b.build(fix_root_link=True);joint=robot.get_active_joints()[0]
        def gains(p,d):joint.set_drive_property(stiffness=p,damping=d,force_limit=float(np.finfo(np.float32).max),mode='force')
        gains(kp,kd);joint.set_armature([armature]);joint.set_drive_target(0.)
        defaults=[dict(linear=float(x.linear_damping),angular=float(x.angular_damping)) for x in robot.get_links()]
        default_joint_friction=float(joint.friction)
        def no_drag():
            for x in robot.get_links():x.linear_damping=0.;x.angular_damping=0.
            joint.friction=0.
        def reset(q,v):robot.set_qpos(np.array([q]));robot.set_qvel(np.array([v]))
        def step(torque):robot.set_qf(np.array([torque]));scene.step()
        def state():return float(robot.get_qpos()[0]),float(robot.get_qvel()[0])
    # Constant-speed torque balance isolates D without assuming a solver's P integration.
    gains(0.,kd);reset(0.,1.);step(kd);_,with_defaults=state()
    no_drag();reset(0.,1.);step(kd);_,balanced=state()
    inferred=(kd-(1+armature)*(balanced-1.)/dt)/((1.+balanced)/2.)
    # Keep the original 1% coefficient budget, expressed in Nm*s/rad rather
    # than an arbitrary bit-level velocity tolerance on an iterative solver.
    assert abs(inferred-kd)<=max(.02,.01*kd),(backend,'damping coefficient',inferred,kd)
    gains(kp,kd)
    loads=[]
    for torque in (1.,5.):
        reset(0.,0.)
        tail=[]
        for tick in range(round(6/dt)):
            step(torque)
            if tick>=round(5.75/dt):tail.append(state()[0])
        q,v=state();expected=torque/kp;mean=float(np.mean(tail));span=float(np.ptp(tail))
        budget=abs(expected)*.01+1e-7
        assert abs(mean-expected)<=budget,(backend,'compliance',torque,mean,expected)
        assert span<=budget,(backend,'unsettled load response',torque,span,budget)
        loads.append(dict(torque_nm=torque,expected_offset_rad=expected,measured_offset_rad=mean,settled_window_s=.25,settled_position_span_rad=span,final_velocity_rad_s=v))
    return dict(kp=kp,kd=kd,armature=armature,default_body_drag=defaults,default_joint_friction=default_joint_friction,velocity_with_default_drag=with_defaults,velocity_after_drag_removed=balanced,damping_inferred=inferred,damping_relative_budget=.01,damping_balance_torque_nm=kd,load_cases=loads,passed=True)


def body_checks(backend,scene,profile,root,resources,full):
    spec=scene.spec
    def make(where):return create_backend(backend,scene,work_dir=where,render=False,environment_collisions=False,gamepad_collisions=False,motion_profile=profile,resource_root=resources)
    b=make(root/'body');cases=[]
    focus={'read_velocity':None,'key':None,'index':0,'peak':0.,'peak_reference':0.,'last_time':-.001,'last_reference':0.}
    def sample_native_step():
        if focus['read_velocity'] is None:return
        velocity=float(focus['read_velocity']());assert math.isfinite(velocity)
        focus['peak']=max(focus['peak'],abs(velocity))
        now=b.motion.control_time;value=float(b.motion.filtered[focus['key']][focus['index']])
        if now>focus['last_time']:
            focus['peak_reference']=max(focus['peak_reference'],abs(value-focus['last_reference'])/(now-focus['last_time']))
            focus['last_time']=now;focus['last_reference']=value
    class StepProbe:
        def __init__(self,wrapped,method):self.wrapped=wrapped;self.method=method
        def __getattr__(self,name):
            original=getattr(self.wrapped,name)
            if name!=self.method:return original
            def measured(*args,**kwargs):
                result=original(*args,**kwargs);sample_native_step();return result
            return measured
    b.scene=StepProbe(b.scene,'step')
    try:
        channels=[]
        for side in ('left','right'):
            channels += [(side,'arm',i) for i in range(7)]
            channels += [(side,'hand',i) for i,n in enumerate(G20_JOINT_NAMES) if n not in G20_MIMIC_JOINTS[side]]
        if not full:channels=[('left','arm',0),('right','arm',6),('left','hand',0),('right','hand',6)]
        for side,kind,index in channels:
            print('body',side,kind,index,flush=True)
            b.reset();obs=b.observe().robot;i=('left','right').index(side)
            initial=np.array(obs.arm_q[i] if kind=='arm' else obs.hand_q[i]);target=initial.copy()
            low,high=b.motion.bounds(side+'_'+kind);target[index]=np.clip(target[index]+.03,low[index],high[index])
            command=ArmCommand(side,tuple(target),0) if kind=='arm' else HandCommand(side,dict(zip(G20_JOINT_NAMES,target)),0)
            b.apply([command]);last=None;errors=[]
            key=side+'_'+kind
            name=getattr(spec,side+'_arm').joint_names[index] if kind=='arm' else ('right_' if side=='right' else '')+G20_JOINT_NAMES[index]
            col=b.joint_order[side].index(name)
            read_velocity=lambda s=side,c=col:b.robots[s].get_qvel()[c]
            focus.update(read_velocity=read_velocity,key=key,index=index,peak=0.,peak_reference=0.,last_time=-.001,last_reference=float(b.motion.filtered[key][index]))
            for frame in range(100):
                b.step(.02);obs=b.observe().robot;q=obs.arm_q[i] if kind=='arm' else obs.hand_q[i]
                ref=np.array(b.motion.status()['reference'][side+'_'+kind]);accepted=np.array(b.applied_reference[side+'_'+kind])
                assert np.isfinite(q).all()
                np.testing.assert_allclose(ref,accepted,atol=1e-10)
                assert np.all(ref>=low-1e-10) and np.all(ref<=high+1e-10)
                if kind=='arm':
                    if last is not None:assert np.max(np.abs(ref-last))<=math.radians(40)*.02+1e-9
                    last=ref.copy()
                errors.append(float(np.max(np.abs(q-ref))))
            if kind=='arm':assert focus['peak_reference']<=math.radians(40)+1e-8
            cases.append(dict(native_peak_velocity_rad_s=focus['peak'],reference_peak_velocity_rad_s=focus['peak_reference'],side=side,kind=kind,joint=index,target=target.tolist(),final_actual=q.tolist(),final_reference=ref.tolist(),final_error_rad=errors[-1],max_error_rad=max(errors),reference_contract_passed=True))
        # A mid-session recording, with future inputs in its snapshot and journal.
        focus['read_velocity']=None
        from concurrent.futures import ThreadPoolExecutor
        b.reset();b.apply([ArmCommand('left',tuple(np.array(spec.left_arm.home_q)+np.array([.05,0,0,0,0,0,0])),0)])
        session=Session(root/'session',scene,b,{'kind':'synthetic'},fps=50)
        session.meta.update(motion_model=dict(b.motion.metadata(),drives=b.motion_drives),input_clock_origin_ns=0,controller_placeholders={})
        session.save_pool=ThreadPoolExecutor(max_workers=1)
        for _ in range(25):b.step(.02);session.record(b.observe(),list(b.motion.input_commands.values()))
        start=b.observe().robot.sim_time_s
        target=tuple(np.array(spec.left_arm.home_q)+np.array([.10,0,0,0,0,0,0]))
        b.motion.schedule(ArmCommand('left',target,0),start+.015)
        def initial_state():
            physical=b.snapshot_motion_state()
            return dict(workbench=dict(version=2,physical=physical['workbench'],
                commands=copy.deepcopy(session.commands),input_clock_origin_ns=0),
                task=dict(physical=physical['task']))
        session.initial_state_provider=initial_state
        session.control({'action':'start'})
        expected=[]
        for frame in range(5):
            if frame==1:
                event=command_event(ArmCommand('left',tuple(spec.left_arm.home_q),0),0);event['scheduled_sim_time_s']=start+.031
                b.motion.schedule_events([event],0);session.record_inputs([event])
            b.step(.02);obs=b.observe().robot;reference=dict(control_time_s=b.motion.control_time,reference=copy.deepcopy(b.applied_reference))
            session.record(b.observe(),list(b.motion.input_commands.values()),motion_reference=reference)
            expected.append(dict(q=obs.arm_q.copy(),hand=obs.hand_q.copy(),reference=reference))
        session.control({'action':'stop'});session.wait_saved();session.close()
        episode=root/'session/episode-000000';recording=read_episode(episode)
        source=CollectionReplay(episode,stateful=False)
        initial=recording.workbench.meta['initial_state']
        source.motion_journal=read_input_events(episode,meta=recording[0])
        source._record_origin_ns=initial['input_clock_origin_ns']
        source._record_start_s=recording.workbench.meta['recording_start_sim_time_s']
        source.start=0.;source.emit_all=True
        clone=make(root/'restore')
        try:
            clone.restore_motion_state(dict(workbench=initial['physical'],
                task=recording.task.meta['initial_state']['physical'] if recording.task else {}));max_q_error=0.;max_reference_error=0.
            for frame in range(5):
                source.poll((frame+1)*.02);clone.motion.schedule_events(source.drain_events(),0);clone.step(.02);obs=clone.observe().robot
                err=max(float(np.max(np.abs(obs.arm_q-expected[frame]['q']))),float(np.max(np.abs(obs.hand_q-expected[frame]['hand']))));max_q_error=max(max_q_error,err)
                for key,ref in expected[frame]['reference']['reference'].items():max_reference_error=max(max_reference_error,float(np.max(np.abs(np.asarray(clone.applied_reference[key])-ref))))
            assert max_reference_error<1e-12,max_reference_error
            tolerance=1e-3
            assert max_q_error<tolerance,(backend,'restored dynamics',max_q_error)
            resume=dict(passed=True,reference_max_error_rad=max_reference_error,physical_max_error_rad=max_q_error,physical_tolerance_rad=tolerance,note='PhysX solver warm-start not serialized')
        finally:clone.close();source.close()
        return dict(cases=cases,resume=resume,scope='reference limits/velocity/actuator acceptance and snapshot restoration; tracking errors reported, not compared to real hardware')
    finally:b.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('sapien',),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',default=DEFAULT_TASK,help='Task profile; empty isolates the robot joint model');p.add_argument('--resource-root',type=Path);p.add_argument('--full',action='store_true');p.add_argument('--rig-only',action='store_true');a=p.parse_args()
    configure_runtime();a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();profile=load_profile('station')
    result=dict(backend=a.backend,version=importlib.metadata.version(a.backend),asset_version=importlib.metadata.version('xbox-gamepad-physics'),hardware_executed=False,profile=profile)
    try:
        result['force_law']={}
        for kind,v in [('arm',profile['arm']),('hand',profile['hand'])]:
            print('force-law',kind,flush=True)
            armature=v['armature_kg_m2']
            if a.backend=='sapien' and kind=='hand':armature=max(armature,profile.get('numerical_adapters',{}).get('sapien',{}).get('hand_armature_floor_kg_m2',0.))
            result['force_law'][kind]=rig(a.backend,v['stiffness_nm_rad'],v['damping_nm_s_rad'],armature)
        if not a.rig_only:result['body']=body_checks(a.backend,compose(load_workbench(Path(__file__).resolve().parents[1]/'src/dexstation/examples/ar5_g20' / DEFAULT_WORKBENCH),load_task(a.task),'sapien',seed=0),profile,a.output,a.resource_root,a.full)
        result['passed']=True
    except Exception as exc:
        result['passed']=False;result['error']=str(exc);raise
    finally:
        result['wall_s']=time.monotonic()-start;(a.output/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,backend=a.backend,cases=len(result.get('body',{}).get('cases',[])),wall_s=result['wall_s'])))

if __name__=='__main__':main()
