"""Single simulation/recording path used by CLI and local web sessions."""
from dexstation.contracts.versions import CONTROLLER_SNAPSHOT_VERSION
from dataclasses import dataclass, asdict
import json
import copy
from pathlib import Path
import queue
import time

from dexstation.config.site import configure_runtime, repository_root, site_value
from dexstation.inputs.arm_mapping import ArmMapper
from dexstation.recording.session import Session, create_backend as _legacy_create_backend
from dexstation.contracts.profiles import RuntimeScene
from dexstation.contracts.backend import StepObservation
from dexstation.contracts.takeover import FakeTakeover
from dexstation.runtime.health import HardwareLease
from dexstation.config.service_defaults import configured_services
from dexstation.runtime.health import require_loopback
from dexstation.runtime.loop import LatestInput, RealtimeFactor
from dexstation.control.teleop_status import TeleopMonitor
from dexstation.config.defaults import DEFAULT_WORKBENCH, DEFAULT_TASK
from dexstation.backends.registry import backend_availability, backend_names, get_backend
from dexstation.inputs.registry import get_source, source_capabilities, source_names, validate_source, SourceContext


def preflight_linkerta():
    """Forward the existing entry point for external callers."""
    from dexstation.inputs.linkerta_reader import preflight_linkerta as preflight
    return preflight()


def input_source_meta(source, replay, loop):
    """Episode input_source. The site default replay file is not this episode's input unless the source is replay."""
    if not source_capabilities(source).replay:
        return dict(kind=source, replay=None, loop=False)
    return dict(kind=source, replay=None if replay is None else str(replay), loop=bool(loop))


def create_backend(name, scene, **kwargs):
    if isinstance(scene, RuntimeScene):
        from dexstation.recording.session import create_registered_backend as registered_backend
        check = backend_availability(name)
        if not check['available']:
            raise ValueError(f"Backend import+version check failed: {check['reason']}")
        return registered_backend(name, scene, **kwargs)
    return _legacy_create_backend(name, scene, **kwargs)


def validate_initial_state(scene):
    """WP5 replaces this with its shared collision/keep-out geometry check."""
    return dict(checked=False, reason='initial-state geometry check pending WP5')


def compose_options(options):
    from dexstation.profiles.compose import load_workbench, load_task, compose
    path = Path(options.workbench)
    if not path.exists() and len(path.parts) == 1:
        path = Path(__file__).parents[1]/'examples/ar5_g20/data' if str(path) == DEFAULT_WORKBENCH else repository_root()/'configs/workbenches'/path
    scene = compose(load_workbench(path), load_task(options.task), options.backend, seed=options.seed)
    if scene.task['id'] != 'empty' and not scene.task.get('recording', {}).get('format'):
        raise ValueError(f"Task {scene.task['id']}: Non-empty task needs a recording format")
    return scene


def load_control_limits(scene):
    """Never enable incomplete configuration or silently use a fake as safety."""
    path = scene.limits_path
    if not path.is_file():
        return None, dict(enabled=False, reason='limits configuration missing; pending WP5')
    config = json.loads(path.read_text())
    def incomplete(value):
        if isinstance(value, dict):
            return any(incomplete(v) for v in value.values())
        if isinstance(value, list):
            return any(incomplete(v) for v in value)
        return value is None
    if incomplete(config) or not any(config.get(k, {}).get('enabled', False) for k in ('spatial', 'angular')):
        return None, dict(enabled=False, reason='limits parameters are null or disabled; pending WP5')
    from dexstation.control.workspace_limits import load_limits
    return load_limits(path, scene.spec).with_task_keep_out(scene.task['keep_out']), dict(enabled=True, reason=None)


def _json_value(value):
    from dexstation.recording.episode import _json
    return json.loads(_json(value))


def scene_from_episode(episode):
    """Rebuild from frozen layer metadata, without loading current profiles."""
    from dataclasses import replace
    from dexstation.contracts.profiles import ChannelSpec
    from dexstation.profiles.task_objects import render_runtime
    wb = episode.workbench.meta
    task_meta = episode.task.meta if episode.task else {}
    task = task_meta.get('task_profile')
    if task is None:
        from dexstation.contracts.profiles import EMPTY_TASK
        task = copy.deepcopy(EMPTY_TASK)
    partial = render_runtime(None, dict(spec=wb['geometry'], task=task,
        objects=task_meta.get('runtime_objects', []), cameras=wb['cameras']))
    workbench = json.loads(wb['workbench_profile_text'])
    return replace(partial, workbench=workbench, channels=tuple(ChannelSpec(**c) for c in task_meta.get('channels', [])),
        objects=tuple(replace(obj,initial_pose_world=tuple(obj.initial_pose_world)) for obj in partial.objects),
        cells=tuple(task['cells']), actions=task['actions'], calibration=wb['calibration'],
        identity=episode.meta['identity'], frozen=dict(workbench_text=wb['workbench_profile_text'],
            task_text=task_meta.get('task_profile_text', json.dumps(task)), seed=task_meta.get('seed')))


class TaskRuntime:
    """Task callbacks run exclusively in the physics controller process."""
    def __init__(self, scene):
        from dexstation.runtime.plugins import task_plugin
        import numpy as np
        self.scene = scene
        self.backend = None
        self.generator = np.random.default_rng(scene.frozen.get('seed'))
        self.initial_rng_state = copy.deepcopy(self.generator.bit_generator.state)
        self.actions = {}
        self.module = None
        self.plugin = task_plugin(scene.task['id'])
        plugin_actions = {name: action['plugin'] for name, action in scene.actions.items() if 'plugin' in action}
        try:
            self.module = self.plugin.resolve() if self.plugin else None
            if plugin_actions and self.module is None:
                raise ValueError('no registered task plugin')
            for name, symbol in plugin_actions.items():
                callback = getattr(self.module, 'ACTIONS', {}).get(symbol, getattr(self.module, symbol, None))
                if not callable(callback):
                    raise TypeError(f'{symbol} is not callable')
                self.actions[name] = callback
        except Exception as exc:
            raise ValueError(f"Task {scene.task['id']} plugin load failed: {' '.join(str(exc).split())}") from exc
        self.cell_callback = getattr(self.module, 'cell_state', None)
        self.on_plugin_error = None
        self.last_plugin_error = None
        self.latest = {}

    def set_object_state(self, name, pose, twist=None):
        if name not in {obj.name for obj in self.scene.objects}:
            raise ValueError(f'Unknown task object: {name}')
        self.backend.set_object_state(name, pose, twist)

    def state(self):
        return self.backend.observe()

    def rng(self):
        return self.generator

    def status(self):
        task = self.scene.task
        if task['id'] == 'empty':
            return dict(id='empty', version=task['version'], title=task['title'], channels={}, actions={}, cell_state={}, plugin_error=None)
        value = dict(id=task['id'], version=task['version'], title=task['title'],
                     channels=_json_value(self.latest),
                     actions={name: dict(while_recording=action.get('while_recording')) for name, action in self.scene.actions.items()})
        cells = {}
        for cell in self.scene.cells:
            if 'builtin' not in cell:
                continue
            source = cell.get('source', {})
            if isinstance(source, dict) and 'channels' in source:
                cells[cell['id']] = {k: value['channels'].get(k) for k in source['channels']}
            elif isinstance(source, dict) and source.get('evaluation') == 'online':
                cells[cell['id']] = dict(value=None, error='online evaluation not available')
        plugin_error = None
        if callable(self.cell_callback):
            try:
                plugin_cells = self.cell_callback(copy.deepcopy(dict(value, plugin_error=None)))
                if not isinstance(plugin_cells, dict) or any(not isinstance(key, str) for key in plugin_cells):
                    raise TypeError('cell_state must return an object with string cell ids')
                # Validate the entire result before merging any plugin payload.
                plugin_cells = _json_value(plugin_cells)
            except Exception as exc:
                plugin_error = f"{type(exc).__name__}: {' '.join(str(exc).split())}"
            else:
                cells.update(plugin_cells)
        if plugin_error != self.last_plugin_error:
            self.last_plugin_error = plugin_error
            if self.on_plugin_error is not None:
                self.on_plugin_error(plugin_error)
        return dict(value, cell_state=cells, plugin_error=plugin_error)

    def action(self, name, params):
        action = self.scene.actions[name]
        if 'plugin' in action:
            return self.actions[name](self, params)
        from dexstation.profiles.compose import _sample_pose, _frames
        objects = {obj['name']: obj for obj in self.scene.task['objects']}
        initial = {obj.name: obj.initial_pose_world for obj in self.scene.objects}
        poses = {}
        for obj in action.get('objects', []):
            if action['builtin'] == 'resample_objects':
                poses[obj] = _sample_pose(objects[obj], _frames(self.scene.workbench), self.generator)
            elif action['builtin'] == 'restore_initial':
                poses[obj] = initial[obj]
            else:
                raise ValueError(f"Unknown builtin action: {action['builtin']}")
        for obj, pose in poses.items():
            self.set_object_state(obj, pose, [0.] * 6)
        return dict(object_poses=poses)


class ControlGate:
    """Arm-only protocol hold after candidate generation; hands always pass."""
    def __init__(self, scene, backend, session, limits, takeover):
        from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES
        self.scene, self.backend, self.session = scene, backend, session
        self.limits, self.takeover = limits, takeover
        self.last = {s: list(getattr(backend,'applied_reference',{}).get(s+'_arm',getattr(scene.spec,s+'_arm').home_q)) for s in ('left','right')}
        self.hands = {s: list(getattr(scene.spec, s+'_hand').home_q) for s in ('left', 'right')}
        self.joint_names = G20_JOINT_NAMES
        self.decisions = {}
        self.report = None
        self.previous_events = {}

    def filter(self, commands):
        from dexstation.control.commands import ArmCommand, HandCommand
        from dataclasses import asdict, replace
        commands = list(commands)
        arm = copy.deepcopy(self.last)
        for c in commands:
            if isinstance(c, ArmCommand):
                arm[c.side] = list(c.positions)
            elif isinstance(c, HandCommand):
                self.hands[c.side] = [c.joint_positions[n] for n in self.joint_names]
        report = self.limits.check(arm, self.hands) if self.limits is not None else None
        self.report = None if report is None else asdict(report)
        result = {}
        for side in ('left', 'right'):
            violated = report is not None and not report.ok and (
                any(v.side == side for v in [*report.spatial, *report.angular]) or
                any(report.joint.get(side, [])) or not (report.spatial or report.angular or report.joint))
            current = getattr(self.backend, 'applied_reference', {}).get(side+'_arm', self.last[side])
            time_s = self.backend._ticks*self.backend.timestep_s if hasattr(self.backend,'_ticks') and hasattr(self.backend,'timestep_s') else getattr(self.session,'last_sim_time_s',0.)
            decision = self.takeover.update(time_s, side, arm[side], current, not violated)
            released = decision.release and not violated
            if released:
                self.last[side] = arm[side]
            self.decisions[side] = dict(asdict(decision), state=decision.state.value, release=released)
            if violated:
                self.decisions[side].update(state='hold', reason=decision.reason or 'Workspace limit violation')
            signature = (self.decisions[side]['state'], released, self.decisions[side]['reason'])
            if commands and signature != self.previous_events.get(side):
                payload = dict(side=side, decision=self.decisions[side], report=None if report is None else asdict(report))
                self.session.control_log.write(json.dumps(dict(action='control/decision', **_json_value(payload)))+'\n')
                if self.session.writer:
                    self.session.writer.event('workbench', 'limit' if violated else 'takeover', _json_value(payload))
                self.previous_events[side] = signature
            result[side] = tuple(self.last[side])
        return [replace(c, positions=result[c.side]) if isinstance(c, ArmCommand) else c for c in commands]


@dataclass
class CollectionOptions:
    backend: str
    source: str
    output: Path
    duration: float | None
    workbench: str = DEFAULT_WORKBENCH
    task: str = DEFAULT_TASK
    seed: int | None = None
    hardware_lease: dict | None = None
    runtime_scene: RuntimeScene | None = None
    scene_variant: str = 'full'
    mujoco_workers: int = 1
    replay: Path | None = None
    ros_topic_prefix: str = ''
    loop_replay: bool = False
    resource_root: Path | None = None
    rate_hz: int = 30
    realtime: bool = False
    record: bool = False
    note: str = ''
    view: bool = False
    diagnostics: bool = False
    host: str = '127.0.0.1'
    port: int = 8780
    udp_host: str = '127.0.0.1'
    udp_port: int = 50200
    arm_startup: str = 'spec'
    motion_profile: str | dict = 'auto'
    retarget: str | None = None
    autopilot_tasks: Path | None = None

    def validate(self, services=None):
        services = services or configured_services()
        if self.retarget is None:
            self.retarget = services.defaults.get('retarget', 'manus_anatomical')
        if self.retarget not in services.retarget_methods:
            raise ValueError('invalid retarget method')
        if self.backend not in backend_names() or self.source not in source_names():
            raise ValueError('invalid simulation backend or source')
        if self.scene_variant not in ('full', 'free-space'):
            raise ValueError('invalid scene variant')
        if (self.duration is not None and (self.duration <= 0 or not __import__('math').isfinite(self.duration))) or self.rate_hz <= 0:
            raise ValueError('duration and rate must be positive (null duration means unlimited)')
        if self.mujoco_workers < 1 or (not get_backend(self.backend).supports_workers and self.mujoco_workers != 1):
            raise ValueError('--mujoco-workers must be positive and applies only to MuJoCo')
        validate_source(self)
        if self.record and self.scene_variant == 'free-space':
            raise ValueError('free-space cannot be recorded')
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError('seed must be a nonnegative integer')
        if type(self.diagnostics) is not bool:
            raise ValueError('diagnostics must be a boolean')
        require_loopback(self.host)


class CollectionController:
    def __init__(self, options, *, controls=None, stop=None, publish=None, limits=None, takeover=None, services=None):
        self.services = services or configured_services()
        options.validate(self.services)
        self.options = options
        self.controls = controls if controls is not None else queue.Queue()
        self.stop = stop
        self.publish = publish or (lambda state: None)
        self.session = self.source = self.backend = self.view = None
        self.last_row = {}
        self.state = 'starting'
        self.error = None
        self.receiver_start = 0
        self.last_publish = 0.
        self.control_result = None
        self.input_clock_origin_ns = None
        self.live_input = None
        self.rtf = None
        self.live_sequences = {}
        self.live_commands = {}
        self.pending_input_journal = []
        self.teleop_monitor = TeleopMonitor()
        self.scene = self.task_runtime = self.gate = None
        self.limits = limits
        self.takeover = takeover or FakeTakeover()
        self.limits_status = dict(enabled=False, reason='limits not loaded')
        self.scene_check = dict(checked=False, reason='initial-state geometry check pending WP5')
        self.hardware = None
        self.replay_task_events = []
        self.replay_task_index = 0
        self.replay_task_origin = 0.

    def record_plugin_error(self, error):
        journal = self.session.control_log
        if not journal.closed:
            journal.write(json.dumps(dict(action='task/plugin-error',
                task=self.scene.task['id'], plugin_error=error, wall_time_ns=time.time_ns()), ensure_ascii=False)+'\n')

    def observation_time(self):
        observation = self.backend.observe()
        return observation.robot.sim_time_s if isinstance(observation, StepObservation) else observation.sim_time_s

    def realtime_metrics(self):
        return dict(rtf=None if self.rtf is None else self.rtf.value,
            **(self.live_input.status(time.monotonic_ns()) if self.live_input else {}))

    def status(self, *, include_inputs=False):
        motion=getattr(self.backend,'motion',None)
        result = dict(self.last_row, session_state=self.state, error=self.error,
                    backend=self.options.backend, scene_variant=self.options.scene_variant, source=self.options.source, retarget=self.options.retarget,
                    motion_profile_resolved=None if self.backend is None else ('station' if motion is not None else 'legacy'),
                    motion_profile_name=getattr(motion,'profile',{}).get('name'),
                    control_result=self.control_result,
                    recording_status=self.session.status() if self.session else dict(state='idle'),
                    input_health=self.source.health.snapshot() if self.source else {},
                    glove_health=self.source.glove_status() if hasattr(self.source, 'glove_status') else {},
                    signal_source=getattr(self.source, 'mode', 'teleop'),
                    real_robot=self.source.real_status() if hasattr(self.source, 'real_status') else {},
                    autopilot=None,
                    output=str(self.options.output))
        if hasattr(self.source, 'status_fields'):
            result.update(self.source.status_fields())
        result['realtime_metrics'] = self.realtime_metrics()
        result.update(task=self.task_runtime.status() if self.task_runtime else dict(id=self.options.task, channels={}, actions={}, cell_state={}, plugin_error=None),
            workbench=None if self.scene is None else self.scene.identity['workbench'],
            takeover={} if self.gate is None else self.gate.decisions, limits=dict(self.limits_status,report=None if self.gate is None else self.gate.report), scene_check=self.scene_check,
            hardware_lease=self.hardware.status() if self.hardware else self.options.hardware_lease or dict(held=False, owner=None))
        if self.teleop_monitor.latest is not None:
            result['teleop'] = self.teleop_monitor.latest
        if include_inputs:
            from dexstation.inputs.replay import snapshot_input_events, ros_command_topics
            result['input_topics'] = ros_command_topics(self.options.ros_topic_prefix) if source_capabilities(self.options.source).ros_topics else {}
            result['glove_overlay'] = getattr(self.source, 'glove_samples', {})
            result['glove_overlay_error'] = getattr(self.source, 'glove_error', None)
            result['glove_layers'] = getattr(self, 'glove_layers', None)
            result['snapshot_inputs'] = snapshot_input_events(self.source) if self.source else {}
            result['mapping_versions'] = self.session.meta.get('mapping_versions', {}) if self.session else {}
            result['arm_mapping'] = self.session.meta.get('arm_mapping') if self.session else None
        return result

    def emit(self, force=False):
        if force or time.monotonic()-self.last_publish >= .1:
            self.publish(self.status())
            if force and self.view and self.last_row:
                self.view.publish(self.status(include_inputs=True))
            self.last_publish = time.monotonic()

    def boundary(self):
        events, count = self.source.boundary()
        self.consume_events(events)
        self.flush_input_journal()
        return count

    def flush_input_journal(self):
        if self.pending_input_journal:
            self.session.record_inputs(self.pending_input_journal)
            self.pending_input_journal.clear()

    def consume_events(self,events,sim_time_s=None):
        motion=getattr(self.backend,'motion',None)
        if self.live_input is not None:
            if sim_time_s is None:
                sim_time_s=motion.ticks*.001 if motion is not None else self.observation_time()
            selected,journal=self.live_input.select(events,time.monotonic_ns(),sim_time_s)
            if motion is not None:
                motion.schedule_events(selected,self.input_clock_origin_ns)
            else:
                from dexstation.control.commands import command_from_event
                self.live_commands.update({e['channel']:command_from_event(e) for e in selected})
                if selected:self.backend.apply([command_from_event(e) for e in selected])
                self.live_sequences.update({e['channel']:e['sequence'] for e in selected})
            # Filesystem stalls must happen outside the native-step input/apply
            # boundary. The next step samples fresh targets after a slow flush.
            self.pending_input_journal.extend(journal)
            return
        if motion is not None and self.input_clock_origin_ns is not None:
            motion.schedule_events(events,self.input_clock_origin_ns)
        self.session.record_inputs(events)

    def replay_actions(self, sim_time_s):
        while self.replay_task_index < len(self.replay_task_events):
            payload=self.replay_task_events[self.replay_task_index]['payload']
            if payload['sim_time_s']-self.replay_task_origin > sim_time_s+1e-10:break
            self.replay_task_index+=1
            if payload['reply'].get('accepted'):
                self.control(dict(action='task-action',name=payload['name'],params=payload['params'],client='replay'))

    def before_live_step(self,sim_time_s):
        self.consume_events(self.source.drain_events(),sim_time_s)

    def after_live_input(self):
        motion=getattr(self.backend,'motion',None)
        sequences=motion.input_sequences if motion is not None else self.live_sequences
        self.live_input.mark_applied(sequences,time.monotonic_ns())

    def snapshot_recording_state(self):
        physical=self.backend.snapshot_motion_state()
        value=dict(version=CONTROLLER_SNAPSHOT_VERSION,physical=physical,commands=copy.deepcopy(self.session.commands),
                   input_clock_origin_ns=self.input_clock_origin_ns,takeover=self.takeover.snapshot(),
                   released_targets=None if self.gate is None else copy.deepcopy(self.gate.last))
        if self.scene is not None:
            value['physical']=physical['workbench']
            return dict(workbench=value, task=dict(physical=physical.get('task', {}),
                rng=copy.deepcopy(self.task_runtime.generator.bit_generator.state)))
        return value

    def control(self, event):
        action = event.get('action')
        request_id = event.get('request_id')
        self.control_result = dict(request_id=request_id, accepted=True, changed_wall_time_ns=time.time_ns())
        if action == 'task-action':
            name, params = event.get('name'), event.get('params', {})
            recording = self.session.writer is not None
            try:
                if self.task_runtime is None or name not in self.scene.actions:
                    raise ValueError('Unknown task action')
                if recording and self.scene.actions[name].get('while_recording') == 'forbidden':
                    self.control_result.update(accepted=False, error_code='recording_active', error='Task action forbidden while recording')
                else:
                    self.control_result['result'] = _json_value(self.task_runtime.action(name, params))
            except Exception as exc:
                self.control_result.update(accepted=False, error=' '.join(str(exc).split()))
            payload = _json_value(dict(name=name, params=params, reply=self.control_result,sim_time_s=self.observation_time()))
            self.session.control_log.write(json.dumps(dict(action='task/action', **payload))+'\n')
            if recording and self.scene is not None and self.scene.task['id']!='empty':
                self.session.writer.event('task', 'action', payload)
            self.emit(True)
            return
        if action == 'retarget':
            if self.session.writer or self.session.recording_state in ('recording','saving','checking'):
                self.control_result.update(accepted=False,error_code='recording_active')
                self.emit(True)
                return
            self.options.retarget = event['method']
            self.source.retarget = event['method']
            self.source.glove_geometry = None
            self.session.meta['retarget'] = event['method']
            self.session.control_log.write(json.dumps(dict(event,wall_time_ns=time.time_ns()))+'\n')
            self.session.control_log.flush()
            self.emit(True)
            return
        if action == 'signal-source':
            recording = self.session.writer or self.session.recording_state in ('recording','saving','checking')
            if recording or not hasattr(self.source, 'switch'):
                self.control_result.update(accepted=False,error_code='recording_active' if recording else 'source_fixed')
                self.emit(True)
                return
            try:
                self.source.switch(event['source'])
            except (ValueError, OSError) as exc:
                self.control_result.update(accepted=False,error_code='source_failed',error=str(exc))
                self.emit(True)
                return
            self.session.control_log.write(json.dumps(dict(event,wall_time_ns=time.time_ns()))+'\n')
            self.session.control_log.flush()
            self.emit(True)
            return
        if action == 'glove-overlay':
            self.glove_layers = event['layers']
            self.emit(True)
            return
        if action == 'view':
            if self.view is None:
                raise ValueError('this session has no operator view')
            self.view.set_view(event['view'])
            self.emit(True)
            return
        if action == 'start':
            if self.options.scene_variant == 'free-space':
                self.control_result.update(accepted=False,error_code='debug_variant',error='free-space cannot be recorded')
                self.emit(True)
                return
            health = self.source.health.snapshot()
            unhealthy = [k for k,c in health.items() if c['warning']]
            error = ('recording_active' if self.session.writer or self.session.recording_state in ('saving','checking') else
                     'real_robot_source' if getattr(self.source, 'mode', 'teleop') == 'real_robot' else
                     'input_unhealthy' if source_capabilities(self.options.source).health_gate and unhealthy and not event.get('force') else None)
            if error:
                self.control_result = dict(request_id=request_id, accepted=False, error_code=error, unhealthy=unhealthy)
                self.emit(True)
                return
            event = dict(event, health_snapshot=health)
        if action == 'start' and self.session.writer is None:
            self.receiver_start = self.boundary()
            self.session.record_runtime_metrics(self.realtime_metrics())
            self.session.control(event)
            self.session.writer.meta.update(receive_start_sequence=self.receiver_start,
                                             receive_end_sequence=self.receiver_start, received_input_messages=0)
            if hasattr(self.source, 'recording_started'):
                self.source.recording_started()
                self.session.control(dict(action='note', text=self.source.episode_note()))
        elif action == 'stop' and self.session.writer:
            if hasattr(self.source, 'summary_note'):
                self.session.control(dict(action='note', text=self.source.summary_note()))
            count = self.boundary()
            self.session.record_runtime_metrics(self.realtime_metrics())
            self.session.writer.meta.update(receive_end_sequence=count, received_input_messages=count-self.receiver_start)
            self.session.control(event)
        else:
            self.session.control(event)
        self.emit(True)

    def run(self):
        a = self.options
        entry = get_source(a.source)
        capabilities = entry.capabilities
        success = False
        ticks = 0
        begin = time.monotonic()
        try:
            configure_runtime()
            from dexstation.control.joint_motion import resolve_profile
            replay_episode = None
            replay_meta = {}
            if capabilities.replay and a.replay.is_dir():
                if (a.replay/'episode.json').is_file():
                    from dexstation.recording.episode import read_episode
                    replay_episode=read_episode(a.replay)
                    replay_meta=replay_episode[0]
            motion_option=a.motion_profile
            if replay_episode is not None and motion_option == 'auto':
                motion_option=(replay_meta.get('motion_model') or {}).get('profile') or 'legacy'
            profile=resolve_profile(motion_option,a.source,None if replay_episode is not None else a.replay)
            initial=(replay_episode.workbench.meta.get('initial_state') if replay_episode is not None else
                     replay_meta.get('motion_initial_state') if profile else None)
            environment_collisions=a.scene_variant=='full'
            gamepad_collisions=a.scene_variant!='free-space'
            if replay_episode is not None:
                if replay_episode.meta['identity']['backend']['name'] != a.backend:
                    raise ValueError('Stateful replay requires the recorded physics backend')
                self.scene=scene_from_episode(replay_episode)
                if replay_episode.task:
                    self.replay_task_events=[e for e in replay_episode.task.events if e['kind']=='action']
                    self.replay_task_origin=0. if initial else replay_meta.get('recording_start_sim_time_s',0.)
                spec=self.scene.spec
            else:
                self.scene=a.runtime_scene or compose_options(a)
            if self.scene:
                self.task_runtime=TaskRuntime(self.scene)
                if capabilities.task_declared and not any(item['id']==a.source for item in self.scene.task['sources']):
                    raise ValueError(f"Task {self.scene.task['id']} does not declare source {a.source}")
                if capabilities.task_declared:
                    declaration = next(item for item in self.scene.task['sources'] if item['id'] == a.source)
                    if self.task_runtime.plugin is None:
                        raise ValueError('task source needs a registered plugin')
                    self.task_runtime.plugin.resolve(declaration['module'])
                self.scene_check=validate_initial_state(self.scene)
                if self.scene_check.get('checked') and self.scene_check.get('ok') is False:
                    raise ValueError('Initial-state geometry check failed: '+self.scene_check.get('reason', 'invalid scene'))
                if self.limits is None:
                    self.limits,self.limits_status=load_control_limits(self.scene)
                else:
                    self.limits=self.limits.with_task_keep_out(self.scene.task['keep_out'])
                    self.limits_status=dict(enabled=True,reason=None)
                spec=self.scene.spec
            from dexstation.model.station_math import StationMath
            # Test doubles without arm geometry use ArmMapper's default zero offsets.
            no_arm_geometry = all(not hasattr(getattr(spec, s+'_arm', None), 'base_pose')
                                  for s in ('left', 'right'))
            mapper = ArmMapper({s: getattr(spec, s+'_arm').home_q for s in ('left', 'right')},
                startup=a.arm_startup, station_math=StationMath() if profile else None,
                joint_offset_deg=(None if not capabilities.teleop_calibration or no_arm_geometry else self.scene.calibration['teleop']),
                margin_deg=profile['arm']['joint_limit_margin_deg'] if profile else 3.)
            if capabilities.hardware_lease and not a.hardware_lease:
                self.hardware=HardwareLease()
                self.hardware.acquire(port=a.port,task=self.scene.task['id'] if self.scene else a.task)
            context = SourceContext(a, spec, mapper, self.scene, profile,
                replay_episode, replay_meta, initial, self.services.geometry_provider)
            self.source = entry.factory(context)
            if capabilities.real_robot_mirror:
                self.source = self.services.wrap_source(self.source, context)
            self.backend = create_backend(a.backend, self.scene or spec, work_dir=a.output.with_name(a.output.name+'-physics'),
                render=False, resource_root=a.resource_root, environment_collisions=environment_collisions,
                gamepad_collisions=gamepad_collisions, motion_profile=profile)
            view_spec=self.scene
            self.backend.reset()
            if initial:
                physical=initial['physical']
                if replay_episode is not None:
                    physical=dict(workbench=physical, task=replay_episode.task.meta.get('initial_state', {}).get('physical', {}) if replay_episode.task else {})
                self.backend.restore_motion_state(physical)
                self.takeover.restore(initial.get('takeover', {}))
            hand_bounds = getattr(self.backend, 'teleop_hand_bounds', {})
            if get_backend(a.backend).supports_workers and a.mujoco_workers > 1:
                self.backend.mj.mju_threadpool(self.backend.data, a.mujoco_workers)
            self.session = Session(a.output, self.scene or spec, self.backend,
                input_source_meta(a.source, a.replay, a.loop_replay),
                mapper=mapper, fps=a.rate_hz, **(dict(task_format=self.scene.task.get('recording', {}).get('format'),
                    workbench_meta=dict(scene_check=self.scene_check, limits=self.limits_status,started_cameras=[]),
                    task_meta=dict(solver_record={'objects':getattr(self.backend,'solver_record',{}).get('objects',{})},seed=self.scene.frozen.get('seed'), action_rng_initial_state=self.task_runtime.initial_rng_state, runtime_objects=_json_value([asdict(o) for o in self.scene.objects]))) if self.scene else {}))
            if self.scene:
                if replay_episode is not None and replay_episode.task and 'initial_state' in replay_episode.task.meta:
                    rng_state=replay_episode.task.meta['initial_state'].get('rng')
                    if rng_state:self.task_runtime.generator.bit_generator.state=rng_state
                self.session.meta['solver_record']={'robots':getattr(self.backend,'solver_record',{}).get('robots',{})}
                self.task_runtime.backend=self.backend
                self.task_runtime.on_plugin_error=self.record_plugin_error
                self.gate=ControlGate(self.scene,self.backend,self.session,self.limits,self.takeover)
                if initial and initial.get('released_targets'):self.gate.last=copy.deepcopy(initial['released_targets'])
                target_method='_apply_targets' if hasattr(self.backend,'_apply_targets') else 'apply'
                apply_targets=getattr(self.backend,target_method)
                setattr(self.backend,target_method,lambda commands: apply_targets(self.gate.filter(commands)))
            self.session.meta['native_cpu_workers'] = a.mujoco_workers
            motion = getattr(self.backend, 'motion', None)
            self.session.meta['motion_model'] = (dict(motion.metadata(), drives=self.backend.motion_drives)
                                                  if motion is not None else None)
            if motion is not None or self.scene is not None:
                self.session.initial_state_provider=self.snapshot_recording_state
                self.session.last_sim_time_s=self.observation_time()
                self.session.meta['controller_placeholders']={}
                self.session.meta['input_sequence_semantics']='last event delivered to the controller, not latest received'
                if initial:self.session.commands=copy.deepcopy(initial['commands'])
            self.session.meta['retarget'] = a.retarget
            self.session.on_status = lambda: self.emit(True)
            if a.note:
                self.control(dict(action='note', text=a.note))
            if a.view:
                from dexstation.render.operator_view import OperatorView
                self.view = OperatorView(self.backend, Path(__file__).parents[1]/'render/data/operator_view.yaml',
                    a.output/'view', view_spec, a.port, collection=True, host=a.host, resources=a.resource_root, diagnostics=a.diagnostics,
                    geometry_provider=self.services.geometry_provider, retarget=a.retarget)
                self.emit(True)
                self.view.wait_ready(self.stop)
            self.state = 'running'
            begin = time.monotonic()
            physics_start=self.observation_time()
            self.input_clock_origin_ns=time.monotonic_ns()-round(physics_start*1e9)
            self.session.meta['input_clock_origin_ns']=self.input_clock_origin_ns
            self.rtf=RealtimeFactor(begin,physics_start)
            if capabilities.latest_input:
                self.live_input=LatestInput()
                self.backend.before_step=self.before_live_step
                self.backend.after_input=self.after_live_input
                self.session.meta['realtime_input_policy']=dict(mode='latest_per_native_step',
                    max_delay_s=self.live_input.max_delay_s,
                    stale='discard new stale targets; hold previous target; report stale_channels')
            if self.replay_task_events:self.backend.before_step=self.replay_actions
            if a.record:self.control(dict(action='start'))
            n = None if a.duration is None else max(1, round(a.duration*a.rate_hz))
            while (n is None or ticks < n) and not (self.stop and self.stop.is_set()):
                if a.realtime:
                    delay = max(0., begin+(ticks+(1 if motion is not None else 0))/a.rate_hz-time.monotonic())
                    if self.stop:
                        if self.stop.wait(delay):
                            break
                    else:
                        time.sleep(delay)
                if motion is None and self.live_input is None:self.consume_events(self.source.drain_events())
                while True:
                    try:
                        self.control(self.controls.get_nowait())
                    except queue.Empty:
                        break
                if self.view:
                    for event in self.view.events():
                        self.control(event)
                now = {
                    'simulation': lambda: ((ticks+(1 if motion is not None else 0))/a.rate_hz
                        if capabilities.replay else self.last_row.get('sim_time_s', physics_start)),
                    'wall': time.monotonic,
                }[capabilities.clock]()
                commands = self.source.poll(now)
                if self.live_input is None:
                    if motion is not None:self.consume_events(self.source.drain_events())
                    else:self.backend.apply(commands)
                if self.replay_task_events:self.replay_actions(self.observation_time())
                applied_ns = time.monotonic_ns()
                self.backend.step(1/a.rate_hz)
                self.flush_input_journal()
                motion_state = None if motion is None else motion.status()
                if motion is not None:commands=list(motion.input_commands.values())
                elif self.live_input is not None:commands=list(self.live_commands.values())
                observation=self.backend.observe()
                if hasattr(self.source,'sense_contacts') and hasattr(self.backend,'contact_force'):
                    contact=self.backend.contact_force
                    if hasattr(self.source, 'contact_reader'):
                        contact=self.source.contact_reader(self.backend)
                    self.source.sense_contacts(contact)
                if hasattr(self.source,'observe'):
                    if isinstance(observation,StepObservation) and capabilities.flattened_observation:
                        from types import SimpleNamespace
                        self.source.observe(SimpleNamespace(**asdict(observation.robot),**observation.task))
                    else:self.source.observe(observation)
                if self.task_runtime and isinstance(observation,StepObservation):
                    self.task_runtime.latest=observation.task
                self.rtf.update(time.monotonic(),observation.robot.sim_time_s if isinstance(observation,StepObservation) else observation.sim_time_s)
                self.session.record_runtime_metrics(self.realtime_metrics())
                self.last_row = self.session.record(observation, commands,
                    applied_monotonic_ns=applied_ns, input_sequences=motion.input_sequences if motion is not None else
                        self.live_sequences if self.live_input is not None else getattr(self.source, 'polled_sequences', None),
                    motion_reference=None if motion_state is None else
                        dict(control_time_s=motion_state['control_time_s'],reference=copy.deepcopy(self.backend.applied_reference)))
                if isinstance(observation,StepObservation):
                    self.last_row.update(_json_value(asdict(observation.robot)))
                    self.last_row.update(_json_value(observation.task))
                if motion is not None:
                    self.last_row['motion_state'] = motion_state
                self.teleop_monitor.update(observation.robot if isinstance(observation,StepObservation) else observation, self.last_row.get('commands', {}),
                    getattr(self.backend, 'applied_reference', {}), hand_bounds, time.monotonic())
                if self.view:
                    self.view.publish(self.status(include_inputs=True))
                ticks += 1
                self.emit()
            success = True
        except KeyboardInterrupt:
            success = False  # CLI retains interrupted status on Ctrl+C
        except Exception as exc:
            self.error = str(exc)
            raise
        finally:
            try:
                if self.source:
                    self.source.close()
                if self.session:
                    count = self.boundary()
                    if self.session.writer:
                        self.session.record_runtime_metrics(self.realtime_metrics())
                        self.session.writer.meta.update(receive_end_sequence=count, received_input_messages=count-self.receiver_start)
                    self.session.close('complete' if success else 'interrupted')
            finally:
                try:
                    if self.view:
                        closing_view, self.view = self.view, None
                        closing_view.close(cancelled=bool(self.stop and self.stop.is_set()))
                finally:
                    try:
                        if self.backend:
                            self.backend.close()
                    finally:
                        if self.hardware:self.hardware.close()
                    self.state = 'stopped' if success or not self.error else 'failed'
                    self.emit(True)
        result = dict(ticks=ticks, sim_time_s=self.last_row.get('sim_time_s', 0),
                      wall_elapsed_s=time.monotonic()-begin, decode_errors=self.source.decode_errors,
                      source_cycles=getattr(self.source, 'cycles', 0))
        (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        return result


def run_managed(options, controls, stop, statuses, services=None):
    """Spawn target: all simulator and ROS ownership remains in this child."""
    def publish(state):
        # Health/status is latest-only; command journals never use this queue.
        try:
            statuses.put_nowait(state)
        except queue.Full:
            try:
                statuses.get_nowait()
                statuses.put_nowait(state)
            except (queue.Empty, queue.Full):
                pass
    try:
        controller = CollectionController(options, controls=controls, stop=stop, publish=publish, services=services)
        controller.run()
    except Exception as exc:
        publish(dict(session_state='failed', error=str(exc), recording_status=dict(state='failed')))
        raise
