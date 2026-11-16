"""Timestamped joint motion. Station host rules and vendor simulation gains are separate."""
from dexstation.contracts.versions import MOTION_PROFILE_VERSION, MOTION_SNAPSHOT_VERSION
import copy
import heapq
import json
import math
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from dexstation.control.commands import ArmCommand, HandCommand, command_from_event
from dexstation.model.constants import LOW_DEG, HIGH_DEG
from dexstation.inputs.registry import source_names, get_source
from dexstation.backends.registry import backend_names, get_backend
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES, G20_MIMIC_JOINTS, apply_mimic_joint_positions, motor_positions_to_joint_positions
from dexstation.model.station_math import ASSETS, StationMath


def vector(value, count, name):
    result = np.broadcast_to(np.asarray(value, float), (count,)).copy()
    if not np.isfinite(result).all():
        raise ValueError(name+' must be finite')
    return result


def load_profile(value):
    if value is None or value == 'legacy':
        return None
    profile = copy.deepcopy(value) if isinstance(value, dict) else json.loads(
        (ASSETS/'station.json' if value == 'station' else Path(value).expanduser()).read_text())
    if profile['control_dt_s'] != .001:
        raise ValueError('station host math uses a 1 ms control period')
    if profile.get('version') != MOTION_PROFILE_VERSION:
        raise ValueError('Profile v1 is unverified; use legacy target replay or the corrected station profile')
    arm, hand = profile['arm'], profile['hand']
    for key in ('max_speed_deg_s', 'ramp_update_s'):
        if not math.isfinite(arm[key]) or arm[key] <= 0:
            raise ValueError(key+' must be positive')
    for key in ('speed_ramp_s', 'deadband_deg', 'joint_limit_margin_deg'):
        if not math.isfinite(arm[key]) or arm[key] < 0:
            raise ValueError(key+' must be nonnegative')
    if arm['joint_limit_margin_deg'] >= 60:
        raise ValueError('invalid joint limit margin')
    for section, count in ((arm,7),(hand,21)):
        for key in ('stiffness_nm_rad','damping_nm_s_rad','armature_kg_m2'):
            if section.get(key) is None or np.any(vector(section[key],count,key)<0):
                raise ValueError(key+' requires explicit nonnegative values with a recorded source')
        cap=section.get('effort_limit_nm')
        if cap is not None and np.any(vector(cap,count,'effort')<=0):
            raise ValueError('effort limit must be positive, or null for an unmodeled cap')
        cutoff=section.get('filter_hz')
        if cutoff is not None and (not math.isfinite(cutoff) or not 0<cutoff<=1000):
            raise ValueError('SDK client cutoff must be positive and at most 1000; null/1000 bypasses')
        if section.get('filter_model')!='rokae-sdk-rc':
            raise ValueError('expected the verified SDK RC filter')
    speed=hand.get('max_speed_rad_s')
    if speed is not None and np.any(vector(speed,21,'hand speed')<=0):
        raise ValueError('hand speed must be positive or null when not identified')
    floor=profile.get('numerical_adapters',{}).get('sapien',{}).get('hand_armature_floor_kg_m2',0.)
    if not math.isfinite(floor) or floor<0:
        raise ValueError('SAPIEN hand armature floor must be finite and nonnegative')
    return profile


def resolve_profile(value, source, replay=None):
    if value == 'auto':
        value=get_source(source).capabilities.motion_profile if source in source_names() else 'legacy'
        if source in source_names() and get_source(source).capabilities.replay and replay is not None and Path(replay).is_dir():
            meta=json.loads((Path(replay)/'meta.json').read_text())
            if meta.get('motion_initial_state'):
                value=meta['motion_model']['profile']
    return load_profile(value)


def limits_from_spec(spec):
    result={}
    for side in ('left','right'):
        for kind,names in (('arm',getattr(spec,side+'_arm').joint_names),('hand',G20_JOINT_NAMES)):
            root=ET.parse(getattr(spec,side+'_'+kind).urdf).getroot()
            joints={j.get('name'):j for j in root.findall('joint')}
            result[side+'_'+kind]=[[float(joints[n].find('limit').get(k)) for n in names] for k in ('lower','upper')]
    return result


class JointMotion:
    """Control updates occur at t=0,1ms,... before the physical step starting at t."""
    def __init__(self,spec,profile='station',*,cache=None,limits=None):
        self.profile=load_profile(profile)
        self.math=StationMath(cache)
        self.spec=spec
        self.limits=copy.deepcopy(limits) if limits is not None else None
        self.reset()

    def reset(self):
        self.ticks=0  # index of NEXT control update
        self.ramp_start=None
        self.active=set()
        self.requested,self.reference,self.filtered,self.velocity={},{},{},{}
        self.pending=[];self._serial=0;self._last_advance=-1.
        self.input_commands={};self.input_sequences={}
        for side in ('left','right'):
            for kind,count in (('arm',7),('hand',21)):
                key=side+'_'+kind;home=vector(getattr(self.spec,key).home_q,count,key)
                self.requested[key]=home.copy();self.reference[key]=home.copy();self.filtered[key]=home.copy();self.velocity[key]=np.zeros(count)

    def bounds(self,key):
        if key.endswith('_arm'):
            margin=math.radians(self.profile['arm']['joint_limit_margin_deg'])
            low,high=np.deg2rad(LOW_DEG),np.deg2rad(HIGH_DEG)
            if self.limits is not None:
                low=np.maximum(low,self.limits[key][0]);high=np.minimum(high,self.limits[key][1])
            return low+margin,high-margin
        if self.limits is not None:return np.asarray(self.limits[key][0]),np.asarray(self.limits[key][1])
        from dexstation.examples.ar5_g20.mapping import G20_JOINT_LIMITS
        side=key.split('_')[0]
        return tuple(np.array([G20_JOINT_LIMITS[side][n][i] for n in G20_JOINT_NAMES]) for i in (0,1))

    def schedule(self,command,time_s,sequence=None):
        if not math.isfinite(time_s):raise ValueError('input time must be finite')
        heapq.heappush(self.pending,(float(time_s),self._serial,command,sequence));self._serial+=1

    def schedule_events(self,events,origin_ns):
        for event in events:
            if event.get('realtime_discarded'):continue
            command=command_from_event(event)
            if command is None:continue
            when=event.get('scheduled_sim_time_s')
            if when is None:when=(event['received_monotonic_ns']-origin_ns)/1e9
            self.schedule(command,max(0.,when),event.get('sequence'))

    def accept(self,commands):
        now=self.ticks*.001
        for command in commands:
            if command.side not in ('left','right'):raise ValueError('unknown side')
            if isinstance(command,ArmCommand):
                key=command.side+'_arm';target=np.clip(vector(command.positions,7,key),*self.bounds(key))
                self.input_commands[key]=command
                if self.math.difference(np.rad2deg(target),np.rad2deg(self.requested[key]))<self.profile['arm']['deadband_deg']:continue
                if self.ramp_start is None:self.ramp_start=now
            elif isinstance(command,HandCommand):
                key=command.side+'_hand';target=vector([command.joint_positions[n] for n in G20_JOINT_NAMES],21,key)
                self.input_commands[key]=command
                if self.profile['hand']['motor_quantization'] and command.motor_positions is not None:
                    motors=vector(command.motor_positions,20,'hand motors')
                    if np.any((motors<0)|(motors>255)):raise ValueError('hand motors outside 0..255')
                    target=np.asarray(motor_positions_to_joint_positions(command.side,np.trunc(motors)))
                target=np.clip(apply_mimic_joint_positions(command.side,target),*self.bounds(key))
            else:raise TypeError('expected arm or hand command')
            self.requested[key]=target;self.active.add(key)

    def advance_to(self,time_s):
        if not math.isfinite(time_s) or time_s<self._last_advance-1e-10:raise ValueError('invalid or backwards motion clock')
        self._last_advance=float(time_s)
        while self.ticks*.001<=time_s+1e-10:
            now=self.ticks*.001
            while self.pending and self.pending[0][0]<=now+1e-10:
                _,_,command,sequence=heapq.heappop(self.pending)
                self.accept([command])
                if sequence is not None:self.input_sequences[command.side+('_arm' if isinstance(command,ArmCommand) else '_hand')]=sequence
            arm,hand=self.profile['arm'],self.profile['hand']
            for key in sorted(self.active):
                target,old=self.requested[key],self.reference[key]
                if key.endswith('_arm'):
                    elapsed=max(0.,now-self.ramp_start)
                    elapsed=math.floor(elapsed/arm['ramp_update_s']+1e-8)*arm['ramp_update_s']
                    step=math.radians(self.math.ramp_step_deg(arm['max_speed_deg_s'],arm['speed_ramp_s'],elapsed))
                    value=self.math.step(old,target,step)
                else:
                    speed=hand.get('max_speed_rad_s')
                    if speed is None:value=target.copy()
                    else:
                        speed=vector(speed,21,'hand speed')
                        for child,(parent,ratio) in G20_MIMIC_JOINTS[key.split('_')[0]].items():
                            i,j=G20_JOINT_NAMES.index(parent),G20_JOINT_NAMES.index(child);speed[i]=min(speed[i],speed[j]/abs(ratio))
                        value=old+np.clip(target-old,-speed*.001,speed*.001)
                    value=np.asarray(apply_mimic_joint_positions(key.split('_')[0],value))
                value=np.clip(value,*self.bounds(key));self.velocity[key]=(value-old)/.001;self.reference[key]=value
                cutoff=(arm if key.endswith('_arm') else hand).get('filter_hz')
                if cutoff is None or cutoff>=1000:self.filtered[key]=value.copy()
                else:self.filtered[key]=np.array([self.math.lowpass(.001,x,y,cutoff) for x,y in zip(value,self.filtered[key])])
                self.filtered[key]=np.clip(self.filtered[key],*self.bounds(key))
            self.ticks+=1
        return self.commands()

    def commands(self):
        result=[]
        for key in sorted(self.active):
            side,kind=key.split('_');value=self.filtered[key]
            result.append(ArmCommand(side,tuple(value),self.control_time) if kind=='arm' else HandCommand(side,dict(zip(G20_JOINT_NAMES,value)),self.control_time))
        return result

    @property
    def control_time(self):return max(0,self.ticks-1)*.001

    def drive_parameters(self,side,*,backend=None):
        result={}
        for kind,count in (('arm',7),('hand',21)):
            p=self.profile[kind];cap=p.get('effort_limit_nm')
            result[kind]=dict(stiffness=vector(p['stiffness_nm_rad'],count,'stiffness'),damping=vector(p['damping_nm_s_rad'],count,'damping'),force_limit=None if cap is None else vector(cap,count,'effort'),armature=vector(p['armature_kg_m2'],count,'armature'))
        adapter=get_backend(backend).motion_adapter if backend in backend_names() else None
        if adapter:
            floor=self.profile.get('numerical_adapters',{}).get(adapter,{}).get('hand_armature_floor_kg_m2',0.)
            result['hand']['armature']=np.maximum(result['hand']['armature'],floor)
        return result

    def snapshot(self):
        return dict(version=MOTION_SNAPSHOT_VERSION,ticks=self.ticks,ramp_start=self.ramp_start,active=sorted(self.active),last_advance=self._last_advance,serial=self._serial,
                    **{name:{k:v.tolist() for k,v in getattr(self,name).items()} for name in ('requested','reference','filtered','velocity')},
                    input_sequences=dict(self.input_sequences),
                    input_commands={k:asdict(c) for k,c in self.input_commands.items()},
                    pending=[dict(time=t,serial=s,sequence=seq,channel=c.side+('_arm' if isinstance(c,ArmCommand) else '_hand'),converted_command=asdict(c)) for t,s,c,seq in sorted(self.pending)])

    def restore(self,state):
        if state['version']!=MOTION_SNAPSHOT_VERSION:raise ValueError('unsupported motion snapshot')
        self.ticks=state['ticks'];self.ramp_start=state['ramp_start'];self.active=set(state['active']);self._last_advance=state['last_advance'];self._serial=state['serial']
        for name in ('requested','reference','filtered','velocity'):setattr(self,name,{k:np.array(v,float) for k,v in state[name].items()})
        self.input_sequences=dict(state['input_sequences'])
        self.input_commands={k:command_from_event(dict(channel=k,converted_command=c)) for k,c in state['input_commands'].items()}
        self.pending=[(p['time'],p['serial'],command_from_event(p),p['sequence']) for p in state['pending']];heapq.heapify(self.pending)

    def metadata(self):
        return dict(version=MOTION_PROFILE_VERSION,profile=copy.deepcopy(self.profile),provenance=self.math.provenance(),limits=self.limits,calibration='Vendor simulation baseline; not verified hardware equivalence')

    def status(self):
        return dict(name=self.profile['name'],control_time_s=self.control_time,requested={k:self.requested[k].tolist() for k in sorted(self.active)},reference={k:self.filtered[k].tolist() for k in sorted(self.active)})
