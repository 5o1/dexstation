#!/usr/bin/env python3
"""Offline mapping -> limits -> takeover replay; never starts a backend.

Arrays may be raw master degrees or already mapped current-geometry radians.
Recordings use actual hand_q. Raw master commands go through ArmMapper; when
raw_master_deg is absent, recorded target radians are preserved and their J1
mount conversion must be explicitly supplied. Never invert a clipped mapping.

Takeover thresholds have no replay defaults: --takeover must name an explicitly
configured file. The workbench's shipped null thresholds still refuse enablement.
Legacy prototype arrays are diagnostics with different margins/hand geometry;
they are not an actual-finger equivalence reference.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from dexstation.config.defaults import DEFAULT_TASK
from dexstation.inputs.arm_mapping import ArmMapper
from dexstation.contracts.takeover import ArmState
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES
from dexstation.profiles.compose import compose, load_task, load_workbench
from dexstation.control.takeover import load_takeover
from dexstation.control.workspace_limits import decide_release, load_limits

SIDES = ('left', 'right')


def compare_reports(report, reference):
    """Compare reports from the same geometry/actual hands/Euclidean definition."""
    current = report if isinstance(report,dict) else asdict(report)
    left = {(v['side'],v['link'],v['obstacle']):v['distance_m'] for v in current['spatial']}
    right = {(v['side'],v['link'],v['obstacle']):v['distance_m'] for v in reference['spatial']}
    common = left.keys() & right.keys()
    error = max((abs(left[k]-right[k]) for k in common), default=0.)
    same_angular = current['angular'] == reference['angular']
    same_range = current['angular_range'].keys()==reference['angular_range'].keys() and all(np.array_equal(current['angular_range'][s],reference['angular_range'][s]) for s in current['angular_range'])
    return dict(ok_equal=current['ok']==reference['ok'], angular_equal=same_angular,
        spatial_keys_equal=left.keys()==right.keys(), max_distance_error_m=error,
        equivalent=(current['ok']==reference['ok'] and same_angular and left.keys()==right.keys()
                    and current['joint']==reference['joint'] and same_range
                    and error<=1e-9))


class ReplayController:
    """Kinematic acceptance/reference replay, without ramps or physics steps."""
    def __init__(self, limits, takeover, arm_q, hand_q):
        self.limits, self.takeover = limits, takeover
        self.arm_q = {s:np.asarray(q,float).tolist() for s,q in arm_q.items()}
        self.hand_q = {s:np.asarray(q,float).tolist() for s,q in hand_q.items()}
        if not limits.check(self.arm_q,self.hand_q).ok:
            raise ValueError('replay requires a legal initial profile state')

    def step(self, t, arm_targets, hand_targets):
        sides=tuple(self.arm_q)
        proposed={s:self.arm_q[s] if arm_targets[s] is None else arm_targets[s] for s in sides}
        first=decide_release(self.limits,self.arm_q,self.hand_q,proposed,hand_targets)
        events=[];take={}
        def update(side,limit_ok):
            before=self.takeover.snapshot()['arms'][side]['state']
            decision=self.takeover.update(float(t),side,arm_targets[side],self.arm_q[side],limit_ok)
            if decision.state.value!=before:
                events.append(dict(type='takeover',t=float(t),side=side,before=before,
                    after=decision.state.value,reason=decision.reason,
                    deviation_deg=decision.deviation_deg))
            return decision
        for side in sides:take[side]=update(side,first.arm_release[side])
        gated={s:first.arm_q[s] if take[s].release else self.arm_q[s] for s in sides}
        final=decide_release(self.limits,self.arm_q,self.hand_q,gated,first.hand_q)
        for side in sides:
            if not final.arm_release[side]:take[side]=update(side,False)
        reasons={s:list(dict.fromkeys(first.reasons[s]+final.reasons[s])) for s in sides}
        for side in sides:
            if reasons[side]:
                events.append(dict(type='limit_hold',t=float(t),side=side,reasons=reasons[side],
                    arm_held=not (first.arm_release[side] and final.arm_release[side]),
                    hand_held=not (first.hand_release[side] and final.hand_release[side])))
        output_report=self.limits.check(final.arm_q,final.hand_q)
        if not output_report.ok:
            raise RuntimeError('release decision produced an illegal output: '+json.dumps(asdict(output_report)))
        self.arm_q,self.hand_q=final.arm_q,final.hand_q
        return dict(arm_q=self.arm_q,hand_q=self.hand_q,
            arm_release={s:take[s].release and first.arm_release[s] and final.arm_release[s] for s in sides},
            hand_release={s:first.hand_release[s] and final.hand_release[s] for s in sides},
            reasons=reasons,report=asdict(first.report),final_gate_report=asdict(final.report),
            output_report=asdict(output_report),takeover={s:asdict(d) for s,d in take.items()},
            events=events,takeover_snapshot=self.takeover.snapshot())


def decode_record(row, mapper, j1_shift_deg):
    """Preserve measured hands and distinguish raw commands from mapped targets."""
    t=float(row['sim_time_s']);commands=row.get('commands',{})
    observed=np.asarray(row['arm_q'],float).copy()
    hand=np.asarray(row['hand_q'],float)
    if observed.shape!=(2,7) or hand.shape!=(2,21) or not np.isfinite([*observed.ravel(),*hand.ravel(),t]).all():
        raise ValueError('recording requires finite (2,7) arm_q, (2,21) hand_q and time')
    targets={};sources={}
    for i,side in enumerate(SIDES):
        observed[i,0]+=np.radians(j1_shift_deg[i])
        command=commands.get(side+'_arm')
        if command is not None and command.get('raw_master_deg') is not None:
            targets[side]=list(mapper.command(side,command['raw_master_deg'],t).positions)
            sources[side]='raw_master_deg_via_mapper'
        elif command is None:
            targets[side]=None
            sources[side]='no_recorded_arm_command'
        else:
            q=np.asarray(command['positions'],float).copy()
            if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('invalid recorded arm target')
            q[0]+=np.radians(j1_shift_deg[i]);targets[side]=q.tolist()
            sources[side]='recorded_target_rad'
    return dict(t=t,arm_targets=targets,hand_targets={s:hand[i].tolist() for i,s in enumerate(SIDES)},
                observed_arm_q={s:observed[i].tolist() for i,s in enumerate(SIDES)},mapping_source=sources)


class LegacyPrototype:
    def __init__(self, root, tag):
        self.data={}
        for side in SIDES:
            for name in ('request','ok','hold','projection'):
                self.data[side+'_'+name]=np.load(Path(root)/f'replay_{tag}_{side}_{name}.npy',mmap_mode='r',allow_pickle=False)

    def frame(self, raw_index, observed):
        if raw_index%3:return None
        i=raw_index//3
        if any(i>=len(v) for v in self.data.values()):return None
        result=dict(legacy_index=i,spatial_margin_m=.04,angle_margin_deg=2.,hand_geometry='envelope',
                    actual_hand_spatial_equivalence_available=False,sides={})
        for side in SIDES:
            request=np.asarray(self.data[side+'_request'][i])
            result['sides'][side]=dict(request_rad=request.tolist(),ok=bool(self.data[side+'_ok'][i]),
                hold_rad=self.data[side+'_hold'][i].tolist(),projection_rad=self.data[side+'_projection'][i].tolist(),
                observed_request_error_rad=float(np.abs(np.asarray(observed[side])-request).max()))
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbench',required=True)
    parser.add_argument('--task',default=DEFAULT_TASK)
    parser.add_argument('--recording',type=Path,help='directory containing steps.jsonl, or the JSONL file')
    parser.add_argument('--input-kind',choices=('master_deg','target_rad'))
    for side in SIDES:
        parser.add_argument('--'+side+'-request',type=Path)
        parser.add_argument('--'+side+'-hand',type=Path)
    parser.add_argument('--j1-shift-deg',type=float,nargs=2,
                        help='explicit old->current mount conversion; required for recordings')
    parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--max-frames',type=int,help='bounded smoke replay; omission reads the whole source')
    time_group=parser.add_mutually_exclusive_group()
    time_group.add_argument('--timestamps',type=Path);time_group.add_argument('--dt',type=float)
    parser.add_argument('--takeover',required=True,type=Path)
    parser.add_argument('--initial-state',type=Path,help='explicit recorded arm/hand/takeover snapshot; omission keeps the profile initial state')
    parser.add_argument('--reference-reports',type=Path,help='optional actual-hand Euclidean candidate reports, one per selected frame')
    parser.add_argument('--prototype-root',type=Path);parser.add_argument('--prototype-tag',choices=('rec1','rec2'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--summary',type=Path)
    args=parser.parse_args(argv)
    if args.stride<1 or (args.max_frames is not None and args.max_frames<1):parser.error('stride/max-frames must be positive')
    if args.recording and args.j1_shift_deg is None:parser.error('recordings require explicit --j1-shift-deg')
    if args.prototype_root and (not args.recording or not args.prototype_tag):parser.error('prototype comparison requires recording and prototype tag')
    scene=compose(load_workbench(args.workbench),load_task(args.task),'sapien',seed=0)
    limits=load_limits(scene.limits_path,scene).with_task_keep_out(scene.task['keep_out'])
    takeover=load_takeover(args.takeover)
    arms={s:list(getattr(scene.spec,s+'_arm').home_q) for s in SIDES}
    hands={s:list(getattr(scene.spec,s+'_hand').home_q) for s in SIDES}
    if args.initial_state:
        initial=json.loads(args.initial_state.read_text())
        arms,hands=initial['arm_q'],initial['hand_q']
        takeover.restore(initial['takeover'])
        limits.restore(initial.get('limits',{}))
    replay=ReplayController(limits,takeover,arms,hands)
    mapper=ArmMapper(arms,startup='none',joint_offset_deg=scene.calibration['teleop'])
    shift=args.j1_shift_deg or [0.,0.]
    def frames():
        if args.recording:
            path=args.recording/'steps.jsonl' if args.recording.is_dir() else args.recording
            with path.open() as f:
                for index,line in enumerate(f):
                    if index%args.stride:continue
                    yield index,decode_record(json.loads(line),mapper,shift)
        else:
            if not args.input_kind or any(getattr(args,s+'_'+k) is None for s in SIDES for k in ('request','hand')):
                raise ValueError('arrays require input-kind, both request arrays and both actual hand arrays')
            requests={s:np.load(getattr(args,s+'_request'),allow_pickle=False) for s in SIDES}
            hand={s:np.load(getattr(args,s+'_hand'),allow_pickle=False) for s in SIDES};count=len(requests['left'])
            if any(q.shape!=(count,7) for q in requests.values()) or any(q.shape!=(count,21) for q in hand.values()):
                raise ValueError('expected equal-length (N,7) requests and (N,21) hands')
            if args.timestamps:timestamps=np.load(args.timestamps,allow_pickle=False)
            elif args.dt is not None and np.isfinite(args.dt) and args.dt>0:timestamps=np.arange(count)*args.dt
            else:raise ValueError('arrays require timestamps or a positive finite dt')
            if timestamps.shape!=(count,) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps)<0):
                raise ValueError('timestamps must be finite and monotonic')
            for i in range(0,count,args.stride):
                target={}
                for j,side in enumerate(SIDES):
                    q=requests[side][i]
                    if not np.isfinite(q).all():target[side]=None
                    elif args.input_kind=='master_deg':target[side]=list(mapper.command(side,q,timestamps[i]).positions)
                    else:
                        q=q.copy();q[0]+=np.radians(shift[j]);target[side]=q.tolist()
                yield i,dict(t=float(timestamps[i]),arm_targets=target,hand_targets={s:hand[s][i].tolist() for s in SIDES},
                             observed_arm_q=None,mapping_source={s:args.input_kind for s in SIDES})
    legacy=LegacyPrototype(args.prototype_root,args.prototype_tag) if args.prototype_root else None
    reference=args.reference_reports.open() if args.reference_reports else None
    summary=dict(frames=0,events=0,limit_hold_events=0,following_frames={s:0 for s in SIDES},
                 held_hand_frames={s:0 for s in SIDES},reference_mismatches=0,illegal_outputs=0,
                 source='recording_actual_hand_q' if args.recording else 'array_actual_hand_q',j1_shift_deg=shift,
                 takeover_parameters=takeover.parameters,prototype_actual_hand_equivalence=False,
                 initial_state=dict(arm_q=replay.arm_q,hand_q=replay.hand_q,takeover=takeover.snapshot(),limits=limits.snapshot()))
    last_time=None
    try:
        with args.output.open('x') as output:
            for index,frame in frames():
                t=frame['t']
                if last_time is not None and t<last_time:raise ValueError('recording timestamps must be monotonic')
                last_time=t
                result=replay.step(t,frame['arm_targets'],frame['hand_targets'])
                payload=dict(type='frame',frame=summary['frames'],source_row_index=index,t=t,
                    mapping_source=frame['mapping_source'],observed_arm_q=frame['observed_arm_q'],actual_hand_q=frame['hand_targets'],**result)
                if frame['observed_arm_q'] is not None:
                    payload['observed_report']=asdict(limits.check_uncached(frame['observed_arm_q'],frame['hand_targets']))
                if reference is not None:
                    line=reference.readline()
                    if not line:raise ValueError('reference has fewer selected frames')
                    payload['comparison']=compare_reports(result['report'],json.loads(line))
                    summary['reference_mismatches']+=int(not payload['comparison']['equivalent'])
                if legacy:payload['legacy_prototype']=legacy.frame(index,frame['observed_arm_q'])
                output.write(json.dumps(payload,allow_nan=False)+'\n')
                summary['frames']+=1;summary['events']+=len(result['events'])
                summary['limit_hold_events']+=sum(e['type']=='limit_hold' for e in result['events'])
                for side in SIDES:
                    summary['following_frames'][side]+=int(result['takeover'][side]['state']==ArmState.FOLLOWING)
                    summary['held_hand_frames'][side]+=int(not result['hand_release'][side])
                if args.max_frames is not None and summary['frames']>=args.max_frames:break
        if reference is not None and args.max_frames is None and reference.readline():raise ValueError('reference has extra frames')
    finally:
        if reference is not None:reference.close()
    if args.summary:
        with args.summary.open('x') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':main()
