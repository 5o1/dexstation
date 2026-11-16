"""Collection session lifecycle shared by CPU simulators and operator controls."""
from dexstation.contracts.versions import SDL3_LAYOUT_ID
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from dexstation.recording.integrity import check_and_store, pending_integrity, read_integrity, store_integrity, episode_id
from dexstation.control.commands import ArmCommand, HandCommand
from dexstation.recording.episode import EpisodeWriter, SIDES, fingertip_offsets, _json
from dexstation.contracts.backend import StepObservation
from dexstation.contracts.profiles import RuntimeScene
from dexstation.inputs.arm_mapping import VERSION as ARM_MAPPING_VERSION
from dexstation.model.spec import spec_to_dict


# Forwarding names preserve callers while the registry owns all backend entries.
from dexstation.backends.registry import create_registered_backend, get_backend


def create_backend(name, spec, **kwargs):
    return get_backend(name).factory(spec, **kwargs)


def overlay_state(obs):
    if isinstance(obs, StepObservation):
        if 'gamepad_signal' not in obs.task:
            return None
        signal, mask, sim_time = obs.task['gamepad_signal'], obs.task['gamepad_mask'], obs.robot.sim_time_s
    else:
        signal, mask, sim_time = obs.gamepad_signal, obs.gamepad_mask, obs.sim_time_s
    return dict(layout_id=SDL3_LAYOUT_ID, device_id=1,
                values=np.asarray(signal).tolist(),
                available_mask=np.asarray(mask, bool).tolist(),
                task_mask=[False] * 32, source="post_physics_contact", sim_time_s=sim_time)


class Session:
    def __init__(self, output, spec, backend, source, *, mapper=None, fps=30, task_format=None,
                 workbench_meta=None, task_meta=None):
        self.runtime_scene = spec if isinstance(spec, RuntimeScene) else None
        self.task_format = task_format
        self.workbench_meta, self.task_meta = workbench_meta, task_meta
        if self.runtime_scene is not None:
            spec = self.runtime_scene.spec
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        scene = spec_to_dict(spec)
        if scene.get('appearance') and not Path(scene['appearance']).is_absolute() and spec.source_dir:
            scene['appearance']=str((spec.source_dir/scene['appearance']).resolve())
        environment_collisions = getattr(backend, "environment_collisions", True)
        gamepad_collisions = getattr(backend, "gamepad_collisions", environment_collisions)
        name = ("full" if environment_collisions else
                "gamepad-contact" if gamepad_collisions else "free-space")
        scene["collection_variant"] = dict(name=name,
            environment_collisions=environment_collisions, gamepad_collisions=gamepad_collisions,
            robot_self_collision="backend_existing",
            definition="free-space: table, gamepad, fixtures have no collision; gamepad-contact: only the "
                       "gamepad collides with robots; model/mount unchanged; drive profile recorded separately")
        self.meta = dict(scene_spec_text=json.dumps(scene, sort_keys=True),
                         scene_variant=scene["collection_variant"],
                         backend_name=backend.name, backend_version=(self.runtime_scene.identity['backend']['version']
                            if self.runtime_scene else importlib.metadata.version(backend.name)),
                         asset_versions=({"robot": "v4.4-AR5-0.8R-L25-right"} if self.runtime_scene else
                            {"gamepad": importlib.metadata.version("xbox-gamepad-physics"), "robot": "v4.4-AR5-0.8R-L25-right"}),
                         mapping_versions={"arm": ARM_MAPPING_VERSION, "hand": "station-g20-20261014"},
                         arm_mapping=None if mapper is None else mapper.metadata(),
                         input_source=source, fps=fps, operator_notes=[],
                         right_hand_mount_status="assumed", fingertip_offsets_m=fingertip_offsets(spec),
                         fingertip_definition="distal orientation; local [0,0,max_z(collision_mesh)]",
                         units={"hand_q": "rad", "arm_q": "rad", "hand_motor": "motor_0_255",
                                "raw_master": "deg", "slave_target": "rad", "pose_position": "m",
                                "control_joints": "hinge:rad;slide:m", "wall_time_ns": "Unix epoch ns"})
        module = sys.modules[type(backend).__module__]
        self.meta["controller_placeholders"] = {kind: dict(getattr(module, kind, {})) for kind in ("HAND_GAINS", "ARM_GAINS")}
        self.meta["physics_timestep_s"] = (float(backend.timestep_s) if self.runtime_scene else
                                         get_backend(backend.name).timestep(backend))
        if self.runtime_scene:
            self.meta.update(identity=self.runtime_scene.identity, solver_record=backend.solver_record,
                             timestep_source=backend.timestep_source)
        if getattr(backend, "gamepad_solver", None) is not None:
            self.meta["gamepad_solver"] = backend.gamepad_solver
        self.input_sequences = {}
        self.commands = {f"{side}_{kind}": None for side in SIDES for kind in ("hand", "arm")}
        self.writer = None
        self.last_sim_time_s = 0.
        self.episode_index = 0
        self.pending_notes = []
        self.recording_state = "idle"
        self.last_saved = None
        self.on_status = lambda: None
        self.save_pool = self.save_future = None
        # Session alone creates the journal. Both it and the manager append:
        # a startup timing may be written before our first control event.
        descriptor=os.open(self.output/'events.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_APPEND,0o666)
        self.control_log = os.fdopen(descriptor,"w",buffering=1)
        self.initial_state_provider = backend.snapshot_motion_state if self.runtime_scene else None
        self.runtime_metrics = None
        self.latest_runtime_metrics = {}
        self.runtime_dropped_start = 0

    def control(self, event):
        action = event["action"]
        self.control_log.write(json.dumps(dict(event, wall_time_ns=time.time_ns()), ensure_ascii=False) + "\n")
        if action == "start" and self.writer is None and self.recording_state not in {"saving","checking"}:
            initial = self.initial_state_provider() if getattr(self,'initial_state_provider',None) else None
            self.writer = EpisodeWriter(self.output / f"episode-{self.episode_index:06d}",
                                        dict(self.meta, recording_start_sim_time_s=self.last_sim_time_s,
                                             motion_initial_state=initial,
                                             started_by=event.get('client','cli'), stopped_by=None,
                                             forced=bool(event.get('force',False)),
                                             recording_start_health=event.get('health_snapshot',{})),
                                        scene=getattr(self, 'runtime_scene', None),
                                        task_format=getattr(self, 'task_format', None),
                                        workbench_meta=getattr(self, 'workbench_meta', None),
                                        task_meta=getattr(self, 'task_meta', None))
            self.episode_index += 1
            self.runtime_metrics = None
            self.runtime_dropped_start = getattr(self,'latest_runtime_metrics',{}).get('dropped_inputs',0)
            self.recording_state = "recording"
            self.current_integrity = pending_integrity(self.writer.path,"recording")
            for note in self.pending_notes:
                self.writer.note(note)
            self.pending_notes.clear()
        elif action == "stop" and self.writer is not None:
            self.writer.meta["stopped_by"] = event.get("client", "cli")
            self.finish()
        elif action == "note":
            if self.writer:
                self.writer.note(event.get("text", ""))
            else:
                self.pending_notes.append(event.get("text", ""))
        elif action not in {"start", "stop", "note"}:
            raise ValueError("unknown recording event")

    def record_inputs(self, events):
        if self.writer:
            self.writer.append_inputs(events)

    def record_runtime_metrics(self, metrics):
        """Optional episode metadata; original observation/action schema is unchanged."""
        self.latest_runtime_metrics = metrics
        if not self.writer:return
        if self.runtime_metrics is None:
            self.runtime_metrics = dict(samples=0, rtf={}, input_delay_s={},
                dropped_inputs=0, dropped_start=self.runtime_dropped_start)
        stats=self.runtime_metrics
        stats['samples']+=1
        for name in ('rtf','input_delay_s'):
            value=metrics.get(name)
            if value is None:continue
            s=stats[name]
            s.update(count=s.get('count',0)+1, sum=s.get('sum',0.)+value,
                     min=min(s.get('min',value),value), max=max(s.get('max',value),value))
            s['mean']=s['sum']/s['count']
        stats['dropped_inputs']=metrics.get('dropped_inputs',0)-stats['dropped_start']
        stats['input_delay_limit_s']=metrics.get('input_delay_limit_s')
        self.writer.meta['realtime_metrics']=dict(stats, sampling='post-frame and recording boundaries; RTF trailing 1 s wall window')

    def record(self, observation, commands, *, applied_monotonic_ns=None, input_sequences=None,
               motion_reference=None):
        self.update_save()
        self.last_sim_time_s = observation.robot.sim_time_s if isinstance(observation, StepObservation) else observation.sim_time_s
        for c in commands:
            kind = "hand" if isinstance(c, HandCommand) else "arm"
            self.commands[c.side + "_" + kind] = asdict(c)
        self.input_sequences.update(input_sequences or {})
        overlay = overlay_state(observation)
        if self.writer:
            self.writer.append(observation, self.commands, applied_monotonic_ns=applied_monotonic_ns,
                               input_sequences=self.input_sequences, arm_clamps=self.arm_clamps(),
                               motion_reference=motion_reference)
        values = json.loads(_json(asdict(observation))) if isinstance(observation, StepObservation) else observation.to_dict()
        if isinstance(observation, StepObservation):
            values = dict(values, sim_time_s=observation.robot.sim_time_s)
        return dict(values, commands=dict(self.commands), overlay=overlay,
                    recording=self.writer is not None, arm_clamps=self.arm_clamps(), recording_status=self.status(), applied_monotonic_ns=applied_monotonic_ns,
                    input_sequences=dict(self.input_sequences))

    def arm_clamps(self):
        return {side: None if self.commands[side+'_arm'] is None else self.commands[side+'_arm'].get('clamp_flags')
                for side in SIDES}

    def status(self):
        self.update_save()
        writer = self.writer
        return dict(state=self.recording_state, integrity=getattr(self,"current_integrity",None),
                    episode=writer.path.name if writer else (self.last_saved or {}).get('episode'),
                    rows=writer.meta['rows'] if writer else (self.last_saved or {}).get('rows', 0),
                    notes=writer.meta.get('operator_notes',[]) if writer else (self.last_saved or {}).get('notes',[]),
                    duration_s=max(0., self.last_sim_time_s-writer.meta['recording_start_sim_time_s']) if writer else
                               (self.last_saved or {}).get('duration_s', 0.), last_saved=self.last_saved)

    def finish(self, status='complete'):
        writer = self.writer
        self.recording_state = 'saving'
        self.current_integrity = pending_integrity(writer.path,'saving')
        self.on_status()
        if writer.meta.get('stopped_by') is None:
            writer.meta['stopped_by'] = 'session_end'
        writer.meta['recording_end_sim_time_s'] = self.last_sim_time_s
        writer.close(status)
        self.last_saved = dict(episode=writer.path.name, episode_id=episode_id(writer.path), rows=writer.meta['rows'], notes=writer.meta.get('operator_notes',[]),
                               duration_s=self.last_sim_time_s-writer.meta['recording_start_sim_time_s'], integrity=self.current_integrity)
        self.integrity_path=writer.path
        store_integrity(writer.path,self.current_integrity)
        self.writer = None
        if getattr(self,'save_pool',None) is None:
            self.save_pool = ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'))
        self.save_future = self.save_pool.submit(check_and_store,writer.path)
        self.on_status()

    def update_save(self):
        future=getattr(self,'save_future',None)
        if future is None:return
        if future.done():
            try:result=future.result()
            except Exception as exc:result=dict(self.current_integrity,state='failed',passed=False,errors=[str(exc)])
            if result.get('episode_id') != self.current_integrity['episode_id']:
                result=dict(self.current_integrity,state='failed',passed=False,errors=['integrity episode mismatch'])
            self.save_future=None
            self.current_integrity=result
            self.last_saved=dict(self.last_saved,integrity=result)
            self.recording_state='saved' if result['state']=='passed' else 'failed'
        elif time.monotonic()-getattr(self,'last_integrity_poll',0)>=.05:
            self.last_integrity_poll=time.monotonic()
            self.current_integrity=read_integrity(self.integrity_path)
            self.last_saved=dict(self.last_saved,integrity=self.current_integrity)
            if self.current_integrity['state']=='running':self.recording_state='checking'

    def wait_saved(self):
        future=getattr(self,'save_future',None)
        if future is not None:
            try:future.result()
            except Exception:pass
            self.update_save()
        if getattr(self,'save_pool',None) is not None:
            self.save_pool.shutdown(wait=True)
            self.save_pool=None
        self.on_status()

    def close(self, status="complete"):
        if self.writer:
            self.finish(status)
        self.wait_saved()
        self.control_log.close()
