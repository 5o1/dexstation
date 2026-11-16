"""Loopback collection desk; owns simulation and registered read-only inputs."""
import asyncio
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import secrets
from collections import deque
from dataclasses import asdict
import socket
import sys
import time
from uuid import uuid4

from dexstation.config.site import site_value, workspace_root, repository_root, data_root
from dexstation.runtime.controller import CollectionOptions, run_managed, compose_options, scene_from_episode, TaskRuntime, backend_availability
from dexstation.runtime.health import HardwareLease, HardwareConflict, HardwareLockUnavailable, detect_safety, require_loopback, DISK_MIN_BYTES, SAFETY_INTERVAL_S
from dexstation.recording.integrity import discard_episode, read_integrity
from dexstation.contracts.services import CameraConflict, InputConflict, ForeignPublisherError
from dexstation.config.service_defaults import configured_services
from dexstation.inputs.glove_recording import WebGloveRecording, GloveRecordingConflict, PROTOCOL
from dexstation.config.defaults import DEFAULT_WORKBENCH, DEFAULT_TASK
from dexstation.backends.registry import backend_entries
from dexstation.inputs.registry import source_names, source_capabilities
from dexstation.runtime.plugins import plugin_status, task_plugin, task_plugins
from dexstation.contracts.versions import API_PREFIX


def defaults():
    return dict(workbench=site_value('collection.workbench',DEFAULT_WORKBENCH),
                task=site_value('collection.task',DEFAULT_TASK), seed=None, backend=site_value('collection.backend'), source=site_value('collection.source'),
                scene_variant='full', duration=site_value('collection.duration'),
                rate_hz=site_value('collection.rate_hz'), replay=site_value('collection.replay'),
                loop_replay=site_value('collection.loop_replay'),
                motion_profile=site_value('collection.motion_profile','auto'), retarget='manus_anatomical',
                ros_topic_prefix=site_value('collection.ros_topic_prefix', ''),
                autopilot_tasks=task_set_files()[0] if task_set_files() else None)


def task_set_files(task_id=None, source=None):
    entries = task_plugins()
    selected = [entries[task_id]] if task_id in entries else entries.values()
    return sorted(path for entry in selected for path in entry.task_sets(source))


def episode_metadata(path):
    manifest=json.loads((path/'episode.json').read_text())
    workbench=json.loads((path/'workbench/meta.json').read_text())
    return dict(workbench, **manifest, **manifest.get('session', {}))


def task_web_directory(task_id):
    entry = task_plugin(task_id, required=True)
    return Path(entry.web_directory)


def task_catalog():
    from dexstation.contracts.profiles import EMPTY_TASK
    profiles = {p.parent.parent.name: p for p in (Path(__file__).parents[1]/'task_plugins').glob('*/data/task.json')}
    profiles.update({entry.id: Path(entry.data_directory)/'task.json' for entry in task_plugins().values()})
    tasks = {'empty': EMPTY_TASK}
    for task_id, path in profiles.items():
        try:
            tasks[task_id] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return [{k: tasks[key][k] for k in ('id','title','version')} for key in sorted(tasks)]


class IdleSession:
    """Only browser connections/activity and accepted writes keep a session alive."""
    def __init__(self, timeout_s=7200, clock=time.monotonic):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('idle timeout must be finite and positive')
        self.timeout_s, self.clock = timeout_s, clock
        self.connections = 0
        self.touch()

    def touch(self):
        self.last_activity = self.clock()

    def expired(self, state, pending=False):
        if self.connections or pending or state.get('session_state') not in ('starting','running') or state.get('recording_status', {}).get('state') in ('recording','saving','checking'):
            self.touch()
            return False
        return self.clock() - self.last_activity >= self.timeout_s


class SessionManager:
    def __init__(self, root=None, default_values=None, *, input_chain_factory=None, services=None):
        self.services = services or configured_services()
        self.root = Path(root or (self.services.data_root or data_root)()/'sessions').expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ctx = mp.get_context('spawn')
        self.worker = self.controls = self.stop_event = self.statuses = None
        self.latest = dict(session_state='idle', recording_status=dict(state='idle'))
        self.port = None
        self.output = None
        self.lock = asyncio.Lock()
        self.export_process = None
        self.pending_start = False
        self.active_options = {}
        self.default_values = dict(self.services.defaults, **(default_values or {}))
        self.input_chain_factory = input_chain_factory or self.services.input_chain_factory
        self.input_chain = None
        self.input_task = None
        self.input_end = None
        self.input_state = dict(state='stopped', components={})
        self.timings = {}
        self.timing_pending = {}
        self.started_monotonic_s = None
        self.hardware = None
        self.service_port = None
        self.scene = None

    def timing(self, name, monotonic_s):
        if name in self.timings or self.started_monotonic_s is None:
            return
        self.timings[name] = max(0., monotonic_s-self.started_monotonic_s)
        self.timing_pending[name] = monotonic_s
        self.flush_timings()

    def flush_timings(self):
        if not self.timing_pending or not self.output or not self.output.is_dir():
            return
        # Directory creation is earlier than journal readiness. Never create
        # this file here: Session owns its exclusive creation.
        try:
            descriptor=os.open(self.output/'events.jsonl',os.O_WRONLY|os.O_APPEND)
        except FileNotFoundError:
            return
        # At most four records per session; never a per-frame production log.
        with os.fdopen(descriptor,'w') as stream:
            for name, stamp in self.timing_pending.items():
                stream.write(json.dumps(dict(action='session/timing', name=name,
                    elapsed_s=self.timings[name], monotonic_s=stamp,
                    started_monotonic_s=self.started_monotonic_s))+'\n')
        self.timing_pending.clear()

    async def manage_inputs(self, chain):
        try:
            await asyncio.to_thread(chain.launch)
            while not self.input_end.is_set():
                self.input_state = await asyncio.to_thread(chain.status)
                for name, component in self.input_state.get('components', {}).items():
                    stamp = component.get('first_data_monotonic_s')
                    if stamp is not None:
                        self.timing('first_'+name+'_data_s', stamp)
                self.refresh()
                if self.input_state.get('state') == 'failed' or not self.worker.is_alive():
                    break
                await asyncio.sleep(.05)
        except Exception as exc:
            self.input_state = dict(self.input_state, state='failed', error=str(exc))
        finally:
            failure = self.input_state.get('error')
            try:
                await asyncio.to_thread(chain.stop)
                self.input_state = await asyncio.to_thread(chain.status)
                if failure:
                    self.input_state.update(state='failed', error=failure)
            except Exception as exc:
                self.input_state.update(state='failed', error=str(exc), receiver_released=False)

    async def finish_inputs(self):
        if self.input_task:
            self.input_end.set()
            await asyncio.shield(self.input_task)

    def refresh(self):
        if self.statuses:
            while True:
                try:
                    self.latest = self.statuses.get_nowait()
                except queue.Empty:
                    break
        if self.worker and not self.worker.is_alive() and self.latest.get('session_state') not in ('stopped','failed'):
            self.latest = dict(self.latest, session_state='failed', error=f'session worker exited: {self.worker.exitcode}')
        if self.latest.get('recording_status',{}).get('state') == 'recording':
            self.pending_start = False
        self.flush_timings()
        if self.output and 'first_frame_s' not in self.timings:
            marker = self.output/'view/first_frame'
            if marker.exists():
                # The existing renderer writes its monotonic timestamp here.
                try:
                    stamp = float(marker.read_text().strip())
                except (ValueError, FileNotFoundError):
                    pass  # writer may have created the marker before filling it
                else:
                    self.timing('first_frame_s', stamp)
        self.latest.update(input_chain=dict(self.input_state, timings=dict(self.timings),
            started_monotonic_s=self.started_monotonic_s))
        if self.worker and self.worker.is_alive() and self.latest.get('session_state') == 'stopped':
            return dict(self.latest, session_state='stopping')
        if self.hardware and self.worker and not self.worker.is_alive() and (not self.input_task or self.input_task.done()):
            self.hardware.close()
        self.latest['hardware_lease']=self.hardware.status() if self.hardware else dict(held=False,owner=None)
        return self.latest

    def episode(self, identifier):
        path = (self.root/identifier.replace('~','/')).resolve()
        if not path.is_relative_to(self.root) or not (path/'episode.json').is_file():
            raise ValueError('episode not found within collection root')
        return path

    def episodes(self, task=None):
        items = []
        paths = [*self.root.glob('session-*/episode-*/episode.json'),
                 *self.root.glob('*/session-*/episode-*/episode.json')]
        for path in sorted(paths, reverse=True):
            try:
                meta=episode_metadata(path.parent)
                task_id=meta.get('identity',{}).get('task',{}).get('id','empty')
                if task is not None and task_id != task:continue
                current=self.latest.get('recording_status',{})
                live=(meta.get('status')=='recording' and self.output==path.parent.parent and current.get('episode')==path.parent.name)
                duration=(meta['recording_end_sim_time_s']-meta['recording_start_sim_time_s']
                          if 'recording_end_sim_time_s' in meta else None)
                items.append(dict(id=str(path.parent.relative_to(self.root)).replace('/','~'), episode=path.parent.name,
                    task=task_id, created_wall_time_ns=meta.get('created_wall_time_ns'),
                    rows=current.get('rows') if live else meta.get('rows') if meta.get('status')!='recording' else None,
                    duration_s=current.get('duration_s') if live else duration,
                    notes=current.get('notes',[]) if live else meta.get('operator_notes',[]), status=meta.get('status'),
                    integrity=read_integrity(path.parent), discarded=path.with_name('discarded.json').exists()))
            except (OSError,ValueError) as exc:
                items.append(dict(id=str(path.parent.relative_to(self.root)).replace('/','~'),status='failed',error=str(exc)))
        return items

    async def start(self, values):
        clicked = time.monotonic()
        async with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError('only one simulation session is allowed; stop it first')
            v = dict(defaults(), **self.default_values)
            v.update(values)
            replay = v.get('replay')
            if v.get('episode'):
                path = self.episode(v['episode'])
                meta = episode_metadata(path)
                v.update(source='replay', replay=str(path), backend=meta['backend_name'],
                    scene_variant=meta['scene_variant']['name'] if isinstance(meta['scene_variant'],dict) else meta['scene_variant'],
                    task=meta.get('identity',{}).get('task',{}).get('id','empty'),rate_hz=int(meta['fps']),
                    duration=meta['rows']/float(meta['fps']), loop_replay=False)
                replay = v['replay']
            stamp = time.strftime('%Y%m%d-%H%M%S')
            output = self.root/str(v.get('task','empty'))/f'session-{stamp}-{uuid4().hex[:6]}'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0))
                port = sock.getsockname()[1]
            tasks = v.get(source_capabilities(v['source']).task_set_option) if source_capabilities(v['source']).task_set_option else None
            if tasks is not None and str(tasks) not in task_set_files(v['task'], v['source']):
                raise ValueError('unknown task set; choose one listed by the station configuration')
            options = CollectionOptions(backend=v['backend'], source=v['source'], output=output,
                duration=None if v['duration'] is None else float(v['duration']),
                rate_hz=int(v['rate_hz']), scene_variant=v['scene_variant'],
                replay=Path(replay).expanduser().resolve() if replay else None,
                ros_topic_prefix=v.get('ros_topic_prefix',''), loop_replay=bool(v.get('loop_replay')),
                workbench=v['workbench'], task=v['task'], seed=v.get('seed'),
                arm_startup=v.get('arm_startup') or site_value('collection.arm_startup','spec'),
                motion_profile=v.get('motion_profile',site_value('collection.motion_profile','auto')),
                retarget=v.get('retarget','manus_anatomical'),
                autopilot_tasks=Path(tasks) if tasks else None,
                resource_root=Path(site_value('collection.resource_root')) if site_value('collection.resource_root') else None,
                view=True, realtime=True, host='127.0.0.1', port=port,
                diagnostics=values.get('diagnostics', False),
                udp_host=site_value('udp.bind_host','127.0.0.1'), udp_port=site_value('udp.port',50200))
            options.validate(self.services)
            if options.replay and options.replay.is_dir() and (options.replay/'episode.json').is_file():
                from dexstation.recording.episode import read_episode
                self.scene=scene_from_episode(read_episode(options.replay))
            else:
                self.scene=compose_options(options)
            TaskRuntime(self.scene)  # Resolve plugin actions before any input or physics start.
            from dexstation.runtime.controller import validate_initial_state
            check=validate_initial_state(self.scene)
            if check.get('checked') and check.get('ok') is False:
                raise ValueError('Initial-state geometry check failed: '+check.get('reason','invalid scene'))
            options.runtime_scene=self.scene
            output=self.root/self.scene.task['id']/output.name
            options.output=output
            if options.replay and not options.replay.exists():
                raise ValueError('replay file does not exist; select a file in station configuration')
            await self.finish_inputs()
            chain = self.input_chain_factory() if source_capabilities(options.source).input_chain else None
            if source_capabilities(options.source).hardware_lease:
                self.hardware=HardwareLease()
                options.hardware_lease=self.hardware.acquire(port=self.service_port,task=options.task)
            if chain:
                chain.retarget = options.retarget
                # Only read-only conflict inspection precedes simulation spawn.
                # Environment preparation and input readiness run concurrently.
                try:
                    await asyncio.to_thread(chain.preflight)
                    if hasattr(chain, 'check_publishers'):
                        await asyncio.to_thread(chain.check_publishers)
                    if hasattr(chain, 'check_can'):
                        await asyncio.to_thread(chain.check_can)
                except InputConflict as exc:
                    self.input_state = dict(state='failed', components={}, error=str(exc))
                    if self.hardware:self.hardware.close()
                    raise
                except Exception:
                    if self.hardware:self.hardware.close()
                    raise
            self.controls = self.ctx.Queue(maxsize=32)
            self.statuses = self.ctx.Queue(maxsize=8)
            self.stop_event = self.ctx.Event()
            self.latest = dict(session_state='starting', recording_status=dict(state='idle'))
            self.port, self.output = port, output
            # Session() owns creation of output. A sibling journal would split
            # provenance, so timing records are deferred until it creates it.
            self.started_monotonic_s = clicked
            self.timings = {}
            self.timing_pending = {}
            self.input_chain = chain
            self.input_state = dict(state='starting' if chain else 'stopped', components={})
            self.active_options = dict(backend=options.backend, source=options.source, scene_variant=options.scene_variant,
                                       motion_profile=options.motion_profile, retarget=options.retarget)
            self.pending_start = False
            self.worker = self.ctx.Process(target=run_managed, args=(options,self.controls,self.stop_event,self.statuses,self.services))
            previous = {key:os.environ.get(key) for key in ('ROS_DOMAIN_ID','ROS_LOCALHOST_ONLY')}
            try:
                if chain:
                    os.environ.update(ROS_DOMAIN_ID='88',ROS_LOCALHOST_ONLY='1')
                self.worker.start()
            except BaseException:
                if self.hardware:self.hardware.close()
                raise
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key,None)
                    else:
                        os.environ[key] = value
            if chain:
                self.input_end = asyncio.Event()
                self.input_task = asyncio.create_task(self.manage_inputs(chain))
            return dict(started=True)

    async def stop(self, client="service"):
        try:
            return await self.stop_worker(client)
        finally:
            await self.finish_inputs()
            if self.hardware and (not self.worker or not self.worker.is_alive()):self.hardware.close()
            self.refresh()

    async def stop_worker(self, client):
        if getattr(self, "glove_recorder", None):
            await self.glove_recorder.cancel()
        if self.worker and self.worker.is_alive() and self.refresh().get("recording_status",{}).get("state")=="recording":
            await self.command(dict(action="stop",client=client))
            deadline=time.monotonic()+30
            while self.refresh().get("recording_status",{}).get("state") in ("recording","saving","checking"):
                if time.monotonic()>deadline:
                    raise RuntimeError("Saving timed out. Check the collection log.")
                await asyncio.sleep(.025)
        async with self.lock:
            if self.worker and self.worker.is_alive():
                self.stop_event.set()
                await asyncio.to_thread(self.worker.join, 30)
                if self.worker.is_alive():
                    # No pgrep/pkill or hardware stop. Keep ownership for retry/diagnosis.
                    raise RuntimeError('owned session did not stop in 30 s; inspect its log')
            self.refresh()
            return dict(stopped=True, state=self.latest)

    async def command(self, event):
        event = dict(event, request_id=uuid4().hex)
        self.control(event)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            result=self.refresh().get('control_result') or {}
            if result.get('request_id')==event['request_id']:
                return result
            if not self.worker.is_alive():
                raise RuntimeError('Session exited. Check the log before restarting.')
            await asyncio.sleep(.01)
        raise RuntimeError('Command acknowledgement timed out. Check status before retrying.')

    def control(self, event):
        if event.get('action') not in ('start','stop','note','view','glove-overlay','retarget','signal-source','task-action'):
            raise ValueError('expected start/stop/note/view/glove-overlay/retarget/signal-source')
        if not self.worker or not self.worker.is_alive():
            raise ValueError('start a simulation session first')
        self.controls.put_nowait(event)
        return dict(queued=True)

    async def export(self, identifier):
        path = self.episode(identifier)
        if json.loads((path/'episode.json').read_text())['status'] != 'complete':
            raise ValueError('cannot export an incomplete episode')
        executable = Path(site_value('collection.export_python') or workspace_root()/'envs/data-tools-py310/bin/python').expanduser()
        if not executable.is_file():
            raise ValueError('LeRobot export environment unavailable. Set collection.export_python to the existing data-tools Python; no packages were installed.')
        if self.export_process and self.export_process.returncode is None:
            raise ValueError('an export is already running')
        target = path.parent/(path.name+'-lerobot')
        self.export_process = await asyncio.create_subprocess_exec(str(executable),
            str(repository_root()/'scripts/export_lerobot.py'), str(path), '--output', str(target),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await self.export_process.communicate()
        if self.export_process.returncode:
            raise ValueError('LeRobot export failed (check data-tools environment): '+output.decode(errors='replace')[-2000:])
        return dict(exported=True, output=str(target.relative_to(self.root)))


class ApiError(Exception):
    def __init__(self, status, code, message):
        self.status,self.code,self.message=status,code,message


def make_app(manager=None, *, enable_remote_control=False, video_handler=None, auxiliary_video_handler=None, frame_provider=None, managed=True, camera=None, idle_timeout_s=None, clock=time.monotonic, snapshots=None, glove_recorder=None, can_system=None, services=None):
    from aiohttp import web, ClientSession, ClientTimeout, ClientError, WSMsgType
    from dexstation.runtime.provenance import code_provenance
    from dexstation.inputs.linkerta_reader import CanPreflightError
    from dexstation.control.teleop_status import arm_input_status
    services = services or (manager.services if manager is not None else configured_services())
    can_system = can_system or services.can_system
    startup_code = code_provenance(repository_root())
    if manager is not None and manager.services is not services:
        if manager.input_chain_factory == manager.services.input_chain_factory:
            manager.input_chain_factory = services.input_chain_factory
        manager.services = services
        manager.default_values = dict(services.defaults, **manager.default_values)
    manager=manager or SessionManager(services=services)
    root=Path(__file__).with_name('web')
    from dexstation.runtime.plugins import task_panes
    pane_weights=site_value('web.pane_weights', {})
    scene=compose_options(CollectionOptions('sapien','udp',manager.root,None,
        workbench=getattr(manager,'default_values',{}).get('workbench',defaults()['workbench']),
        task=getattr(manager,'default_values',{}).get('task',defaults()['task'])))
    auxiliary_root = services.auxiliary_root(manager.root) if services.auxiliary_root else manager.root.parent
    camera=camera or services.camera_factory((services.data_root or data_root)(), cameras=scene.cameras)
    from dexstation.render.snapshots import SceneSnapshots, SnapshotConflict
    snapshots = snapshots or SceneSnapshots(
        auxiliary_root/'snapshots',
        delay_s=site_value('snapshots.delay_s',30), max_bytes=site_value('snapshots.max_bytes',64*1024*1024),
        width=site_value('snapshots.width',1280), height=site_value('snapshots.height',720))
    glove_recorder = glove_recorder or services.glove_recorder_factory(
        auxiliary_root/"glove_raw")
    manager.glove_recorder = glove_recorder
    cookie=secrets.token_urlsafe(32)
    remote_enabled=bool(enable_remote_control)
    safety={}
    stream=None
    frame_row={}
    client=None
    recent=deque(maxlen=10)
    revision=0
    audit=(manager.root/'api_audit.jsonl').open('a',buffering=1)
    recording_lock=asyncio.Lock()
    idle=IdleSession(site_value('collection.idle_timeout_s',7200) if idle_timeout_s is None else idle_timeout_s, clock)

    def snapshot():
        current=dict(manager.refresh(), **manager.active_options)
        current.setdefault('task',dict(id=scene.task['id'],version=scene.task['version'],title=scene.task['title'],channels={},actions={},cell_state={},plugin_error=None))
        current.setdefault('workbench',scene.identity['workbench'])
        current.setdefault('takeover',{})
        current.setdefault('limits',dict(enabled=False,reason='limits not loaded'))
        current.setdefault('hardware_lease',dict(held=False,owner=None))
        current.setdefault('scene_check',dict(checked=False,reason='initial-state geometry check pending WP5'))
        if frame_provider:
            current.update(frame_provider())
        if frame_row:
            for key in ('overlay','gamepad_signal','gamepad_mask','sim_time_s','arm_clamps','view'):
                if key in frame_row:current[key]=frame_row[key]
        current['arm_input'] = arm_input_status(current.get('input_health', {}))
        free=shutil.disk_usage(manager.root).free
        status=current.get('recording_status',{})
        saved=status.get('last_saved') or {}
        current.update(service_code=dict(startup_code), server_wall_time_ns=time.time_ns(), safety=safety, stream=current.get('stream') if frame_provider else stream, remote_control_enabled=remote_enabled,
            disk=dict(free_bytes=free,warning=free<DISK_MIN_BYTES),
            integrity=status.get('integrity') or dict(episode_id=None,state='pending',phase='idle',passed=None), recent_control_events=list(recent), revision=revision,
            scene_snapshot=snapshots.status(), glove_recording=glove_recorder.status(),
            session_id=manager.output.name if manager.output else None,
            episode_id=(str(manager.output.relative_to(manager.root)).replace('/','~')+'~'+status['episode']) if manager.output and status.get('episode') else None)
        return current

    def event(endpoint, who, values):
        nonlocal revision
        revision+=1
        idle.touch()
        recent.append(dict(wall_time_ns=time.time_ns(),operation=endpoint,client=who,parameters=values))

    def response(value=None):
        state=snapshot()
        return web.json_response(dict(value or {},state=state,safety=state['safety']))

    @web.middleware
    async def revalidate_page_code(request, handler):
        # Pages and modules change on every deploy; revalidate them so a
        # browser never runs a script from the previous release.
        result = await handler(request)
        path = request.path
        if (path == '/' or path.endswith(('.html', '.mjs', '.js', '.css'))) and 'Cache-Control' not in result.headers:
            result.headers['Cache-Control'] = 'no-cache'
        return result

    @web.middleware
    async def presence(request, handler):
        browser = secrets.compare_digest(request.cookies.get('collection_web_session',''),cookie)
        connected = request.path in (API_PREFIX + '/events','/video','/aux-video') or (request.method in ('GET','HEAD') and (browser or request.path=='/'))
        if connected:
            idle.connections += 1
            idle.touch()
        try:
            return await handler(request)
        finally:
            if connected:
                idle.connections -= 1
                idle.touch()

    @web.middleware
    async def contract(request,handler):
        is_api=request.path.startswith(API_PREFIX + '/')
        values=dict(request.query)
        who=request.headers.get('X-Dexstation-Client',request.headers.get('X-Gamepad-VLA-Client','')).strip()
        result=None
        try:
            if request.method=='POST':
                raw=await request.read()
                try:values=json.loads(raw) if raw else {}
                except ValueError:values={'raw_body':raw.decode(errors='replace')}
            origin=request.headers.get('Origin')
            if origin and origin!=f'{request.scheme}://{request.host}':
                raise ApiError(403,'cross_origin','Request origin does not match this page. Call the API locally on the station.')
            if request.method=='POST':
                if request.content_type!='application/json' or not who:
                    raise ApiError(403,'client_header_required','Write requests require JSON and a client header. Use collection_ctl.py or the local web page.')
                try:values=await request.json()
                except (ValueError,TypeError):raise ApiError(400,'invalid_json','Invalid JSON. Correct the request and retry.')
                if not isinstance(values,dict):raise ApiError(400,'invalid_json','Request body must be a JSON object.')
                request['values']=values
                request['client']=who
                is_web=secrets.compare_digest(request.cookies.get('collection_web_session',''),cookie)
                request['is_web']=is_web
                if request.path==API_PREFIX + '/remote-control' and not is_web:
                    raise ApiError(403,'web_session_required','Only the local web page can toggle remote control. Ask the station operator.')
                if not is_web and not remote_enabled:
                    raise ApiError(403,'remote_control_disabled','Ask the station operator to enable remote control on the web page.')
            if request.method not in ('GET','HEAD','OPTIONS'):
                idle.touch()
            result=await handler(request)
        except CanPreflightError as exc:
            result=web.json_response(dict(error='linkerta_can_unavailable', can=exc.can),status=409)
        except ApiError as exc:
            result=web.json_response(dict(error_code=exc.code,message=exc.message,state=snapshot(),safety=safety),status=exc.status)
        except (ValueError,KeyError,TypeError,queue.Full) as exc:
            result=web.json_response(dict(error_code='invalid_request',message=f'Invalid parameters or data. Check and retry: {exc}',state=snapshot(),safety=safety),status=400)
        except GloveRecordingConflict as exc:
            result=web.json_response(dict(error_code='glove_recording_busy',message=str(exc),state=snapshot(),safety=safety),status=409)
        except SnapshotConflict as exc:
            result=web.json_response(dict(error_code='snapshot_busy',message=str(exc),state=snapshot(),safety=safety),status=409)
        except CameraConflict as exc:
            result=web.json_response(dict(error_code='camera_busy',message=str(exc),state=snapshot(),safety=safety),status=409)
        except HardwareLockUnavailable as exc:
            result=web.json_response(dict(error=str(exc)),status=503)
        except HardwareConflict as exc:
            result=web.json_response(dict(error_code='hardware_busy',message=str(exc),owner=exc.owner,state=snapshot(),safety=safety),status=409)
        except ForeignPublisherError as exc:
            result=web.json_response(dict(error='foreign_linkerta_publisher', input=exc.input),status=409)
        except InputConflict as exc:
            result=web.json_response(dict(error_code='input_busy',message=str(exc),state=snapshot(),safety=safety),status=409)
        except RuntimeError as exc:
            result=web.json_response(dict(error_code='operation_failed',message=f'Operation failed. Check the service log before retrying: {exc}',state=snapshot(),safety=safety),status=503)
        except web.HTTPException as exc:
            result=web.json_response(dict(error_code='not_found' if exc.status==404 else 'http_error',message='Endpoint or method not found. Check the API v1 documentation.',state=snapshot(),safety=safety),status=exc.status)
        if is_api and (request.method not in ('GET','HEAD','OPTIONS') or result.status>=400):
            body=json.loads(result.text) if isinstance(result,web.Response) and result.content_type=='application/json' else {}
            audit.write(json.dumps(dict(wall_time_ns=time.time_ns(),endpoint=request.path,method=request.method,
                parameters=({'role':'linkerta'} if request.path==API_PREFIX + '/can/bring-up' else values),client=who or 'unspecified',result=dict(http_status=result.status,
                    error_code=body.get('error_code') if isinstance(body,dict) else None,
                    session_state=manager.refresh().get('session_state'),episode_id=snapshot()['episode_id'])),ensure_ascii=False)+'\n')
        return result

    async def index(request):
        page = (root/'index.html').read_text()
        presentation = json.dumps(services.web_presentation).replace('<', '\\u003c')
        page = page.replace('<!-- service-presentation -->', '<script type="application/json" id="service-presentation">'+presentation+'</script>')
        for marker, fragment in services.web_fragments.items():
            if marker == 'controls':
                page = page.replace('/* service-controls */\nfunction offerCan(error){throw error;}', fragment)
            else:
                page = page.replace('<!-- service-'+marker+' -->', fragment)
        result=web.Response(text=page, content_type='text/html')
        result.set_cookie('collection_web_session',cookie,httponly=True,samesite='Strict')
        return result

    async def config(request):
        discovery = plugin_status()
        files=site_value('collection.replay_files',[])
        default=dict(defaults(),**getattr(manager,"default_values",{}))
        if default['replay'] and default['replay'] not in files:files=[default['replay'],*files]
        profiles=['auto','station','legacy']
        if default['motion_profile'] not in profiles:profiles.append(default['motion_profile'])
        return response(dict(managed=managed,defaults=default,replay_files=files,motion_profiles=profiles,
            retarget_methods=list(services.retarget_methods),
            signal_sources=list(services.signal_sources),
            workbench=dict(scene.identity['workbench'],cameras=[asdict(c) for c in scene.cameras]),
            panes=task_panes({},pane_weights),
            pane_source_capabilities={name:dict(teleop_calibration=source_capabilities(name).teleop_calibration) for name in source_names(api=True)},
            tasks=task_catalog(),backends=[backend_availability(e.name) for e in backend_entries(online=True)],scene_variants=['full','free-space'],sources=list(source_names(api=True)),
            autopilot_tasks=task_set_files(),
            glove_recording_defaults=dict(duration_s=sum(d for _, d in PROTOCOL),countdown_s=5),
            snapshot_defaults=dict(delay_s=snapshots.default_delay,width=snapshots.width,height=snapshots.height,max_bytes=snapshots.max_bytes),
            estop_hint=services.estop_hint, **discovery))

    async def task_cells(request):
        from dexstation.profiles.compose import load_task
        task=load_task(request.match_info['task_id'])
        cells=json.loads(json.dumps(task['cells']))
        panes=task_panes(task,pane_weights)
        for cell in [*cells,*panes]:
            for key in ('module','css'):
                if key in cell:
                    cell[key]='/tasks/'+task['id']+'/'+cell[key]
        return web.json_response(dict(cells=cells,panes=panes,actions=task['actions']))

    async def task_web(request):
        from dexstation.profiles.compose import load_task
        try:task=load_task(request.match_info['task_id'])
        except ValueError:raise web.HTTPNotFound()
        base=task_web_directory(task['id']).resolve()
        path=(base/request.match_info['file']).resolve()
        if not path.is_relative_to(base) or not path.is_file():raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def task_action(request):
        running()
        values=request['values']
        if not isinstance(values.get('action'),str) or not isinstance(values.get('params',{}),dict):
            raise ApiError(400,'invalid_request','Task action needs an action string and params object.')
        reply=await manager.command(dict(action='task-action',name=values['action'],params=values.get('params',{}),client=request['client']))
        event('task/action',request['client'],values)
        if reply.get('error_code')=='recording_active':
            return web.json_response(reply,status=409)
        return web.json_response(reply)

    async def status(request):
        return web.json_response(snapshot())

    async def start(request):
        if manager.worker and manager.worker.is_alive():
            raise ApiError(409,'session_active','Simulation session already active. Stop it first.')
        v=dict(request['values'])
        if 'duration_s' in v:v['duration']=v.pop('duration_s')
        manager.service_port=request.url.port
        await manager.start(v)
        event('session/start',request['client'],v)
        return response()

    async def can_bring_up(request):
        values = request['values']
        if values.get('role') != 'linkerta':
            raise ApiError(400, 'invalid_role', 'Only the LinkerTA CAN role is supported.')
        if not hasattr(can_system, 'bring_up') or not services.can_bring_up:
            raise ApiError(409, 'bring_up_unavailable', 'CAN bring-up not available')
        result = await asyncio.to_thread(can_system.bring_up, 'linkerta', values.get('password'))
        return web.json_response(result, status=200 if result['ok'] else 409)

    async def stop(request):
        async with recording_lock:
            await glove_recorder.cancel()
            await manager.stop(request['client'])
        event('session/stop',request['client'],{})
        return response()

    def running():
        s=manager.refresh()
        if not manager.worker or not manager.worker.is_alive() or s.get('session_state')!='running':
            raise ApiError(409,'session_not_running','Simulation session not ready. Start it and wait until ready.')
        return s

    async def recording_start(request):
        async with recording_lock:
            s=running()
            if manager.pending_start or s.get('recording_status',{}).get('state') in ('recording','saving','checking'):
                raise ApiError(409,'recording_active','Recording or saving already in progress. Check status first.')
            if manager.input_state.get('components',{}).get('retarget',{}).get('state') == 'starting':
                raise ApiError(409,'retarget_switching','Wait for retarget data before recording.')
            v=request['values']
            if type(v.get('force',False)) is not bool:
                raise ApiError(400,'invalid_request','force must be a boolean.')
            result=await manager.command(dict(action='start',client=request['client'],force=v.get('force',False)))
            if not result['accepted']:
                code=result['error_code']
                if code=='debug_variant':
                    raise ApiError(409,code,result['error'])
                if code=='real_robot_source':
                    raise ApiError(409,code,'Recording is off while the signal source is Real robot. Switch to Teleop first.')
                raise ApiError(409,code,'Unhealthy inputs: '+', '.join(result.get('unhealthy',[]))+'; restore inputs before retrying, or explicitly use force if authorized.')
            if v.get('note'):
                await manager.command(dict(action='note',client=request['client'],text=str(v['note'])))
            event('recording/start',request['client'],v)
            return response(dict(episode_id=snapshot()['episode_id']))

    async def retarget_switch(request):
        async with recording_lock:
            state=running()
            if manager.pending_start or state.get('recording_status',{}).get('state') in ('recording','saving','checking'):
                raise ApiError(409,'recording_active','Stop recording before switching retarget.')
            method=request['values'].get('method')
            if method not in services.retarget_methods:
                raise ApiError(400,'invalid_request','Unknown retarget method.')
            if not manager.input_chain or manager.input_chain.manual:
                raise ApiError(409,'input_busy','Switch requires the session-owned ROS input chain.')
            result=await manager.command(dict(action='retarget',method=method,client=request['client']))
            if not result['accepted']:
                raise ApiError(409,'recording_active','Stop recording before switching retarget.')
            await asyncio.to_thread(manager.input_chain.switch_retarget,method)
            manager.input_state=dict(manager.input_state,retarget=method,state='starting',
                components=dict(manager.input_state.get('components',{}),retarget=dict(state='starting')))
            manager.active_options['retarget']=method
            event('retarget/switch',request['client'],dict(method=method))
            return response()

    async def signal_source_switch(request):
        async with recording_lock:
            state=running()
            if manager.pending_start or state.get('recording_status',{}).get('state') in ('recording','saving','checking'):
                raise ApiError(409,'recording_active','Stop recording before switching the signal source.')
            source=request['values'].get('source')
            if source not in ('teleop','real_robot'):
                raise ApiError(400,'invalid_request','Unknown signal source.')
            if not source_capabilities(manager.active_options.get('source')).real_robot_mirror:
                raise ApiError(409,'source_fixed','Signal source can only be switched in a ROS2 session.')
            result=await manager.command(dict(action='signal-source',source=source,client=request['client']))
            if not result['accepted']:
                code=result.get('error_code','source_failed')
                messages=dict(recording_active='Stop recording before switching the signal source.',
                              source_fixed='Signal source can only be switched in a ROS2 session.')
                raise ApiError(409,code,messages.get(code,'Could not switch the signal source: '+str(result.get('error'))))
            manager.active_options['signal_source']=source
            event('signal-source/switch',request['client'],dict(source=source))
            return response()

    async def recording_stop(request):
        async with recording_lock:
            s=manager.refresh()
            episode_id=snapshot()['episode_id']
            if manager.worker and manager.worker.is_alive() and s.get('recording_status',{}).get('state')=='recording':
                # Returns immediately; saving/checking runs in the controller process.
                manager.control(dict(action='stop',client=request['client']))
            event('recording/stop',request['client'],{})
            return response(dict(episode_id=episode_id))

    async def notes(request):
        running()
        text=request['values'].get('text')
        if not isinstance(text,str):raise ApiError(400,'invalid_request','Note text must be a string.')
        await manager.command(dict(action='note',client=request['client'],text=text))
        event('notes',request['client'],dict(text=text))
        return response()

    async def episodes(request):
        return web.json_response(await asyncio.to_thread(manager.episodes, **({'task':request.query['task']} if 'task' in request.query else {})))

    async def episode(request):
        path=manager.episode(request.match_info['id'])
        meta=episode_metadata(path)
        check=path/'integrity.json'
        return response(dict(id=request.match_info['id'],meta=meta,
            integrity=read_integrity(path),voided=(path/'discarded.json').exists()))

    async def void(request):
        path=manager.episode(request.match_info['id'])
        if json.loads((path/'episode.json').read_text())['status']=='recording':
            raise ApiError(409,'episode_recording','Episode is still recording. Stop and save it before marking it as void.')
        value=discard_episode(path,request['values'].get('reason',''))
        event('episodes/void',request['client'],dict(id=request.match_info['id']))
        return response(value)

    async def export(request):
        identifier=request.match_info['id']
        event('episodes/export',request['client'],dict(id=identifier))
        return response(await manager.export(identifier))

    async def glove_recording_start(request):
        async with recording_lock:
            if glove_recorder.active:
                raise GloveRecordingConflict('A glove recording is already running')
            state = running()
            source = manager.active_options.get('source', state.get('source'))
            if not source_capabilities(source).glove_recording:
                raise ApiError(409, 'glove_recording_requires_ros2', 'Glove recording needs the ROS2 source')
            values = request['values']
            hand = values.get('hand')
            if hand not in ('left', 'right'):
                raise ValueError('hand must be left or right')
            if state.get('glove_health', {}).get(hand, {}).get('state') != 'ready':
                raise ApiError(409, 'glove_unhealthy', ('Left' if hand == 'left' else 'Right')+' glove has no data or stale data')
            if not manager.output:
                raise ApiError(409, 'session_not_running', 'Simulation session directory is not ready')
            def alive():
                return bool(manager.worker and manager.worker.is_alive() and
                            manager.refresh().get('session_state') == 'running')
            result = glove_recorder.start(hand, values.get('label', 'subject01'), manager.output, alive)
            event('glove-recording', request['client'], values)
            return response(dict(glove_recording=result))

    async def glove_recording_cancel(request):
        async with recording_lock:
            result = await glove_recorder.cancel()
            event('glove-recording/cancel', request['client'], {})
            return response(dict(glove_recording=result))

    async def scene_snapshot_start(request):
        running()
        run = getattr(manager,'view_run',None) or (manager.output/'view' if manager.output else None)
        if run is None or not (run/'snapshot.mmap').is_file():
            raise ApiError(409,'session_not_running','Wait until the simulation view is ready before taking a snapshot.')
        values=request['values']
        result=snapshots.start(run,delay_s=values.get('delay_s'),label=values.get('label',''))
        event('snapshot',request['client'],dict(delay_s=result['delay_s'],label=result['label']))
        return response(dict(snapshot=result))

    async def scene_snapshot_cancel(request):
        result=await snapshots.cancel()
        event('snapshot/cancel',request['client'],{})
        return response(dict(snapshot=result))

    async def scene_snapshots(request):
        return web.json_response(await asyncio.to_thread(snapshots.list))

    async def scene_snapshot_file(request):
        name=request.match_info['name']
        try:
            if name.endswith(('.png','.json')):
                return web.FileResponse(snapshots.file(name))
            meta=snapshots.file(name+'/meta.json')
            return web.json_response(dict(name=name,meta=json.loads(meta.read_text()),
                files=sorted(p.name for p in meta.parent.iterdir() if p.is_file())))
        except FileNotFoundError:
            raise web.HTTPNotFound()

    async def remote(request):
        nonlocal remote_enabled
        value=request['values'].get('enabled')
        if type(value) is not bool:raise ApiError(400,'invalid_request','enabled must be a boolean.')
        remote_enabled=value
        event('remote-control',request['client'],dict(enabled=value))
        return response()

    async def events(request):
        result=web.StreamResponse(headers={'Content-Type':'text/event-stream','Cache-Control':'no-cache','X-Accel-Buffering':'no'})
        await result.prepare(request)
        try:
            while True:
                data=json.dumps(snapshot(),ensure_ascii=False,separators=(',',':'))
                await asyncio.wait_for(result.write(('data: '+data+'\n\n').encode()),2)
                await asyncio.sleep(.05)
        except (ConnectionError,asyncio.TimeoutError,asyncio.CancelledError):pass
        return result

    async def video(request):
        if not manager.worker or not manager.worker.is_alive():raise web.HTTPServiceUnavailable(text='no active simulation')
        try:
            upstream=await client.ws_connect(f'http://127.0.0.1:{manager.port}{request.path}')
        except (ClientError,OSError,asyncio.TimeoutError):return web.Response(status=503,text='stream unavailable')
        result=web.WebSocketResponse()
        await result.prepare(request)
        async def forward():
            async for message in upstream:
                if message.type==WSMsgType.BINARY:await result.send_bytes(message.data)
        async def closed():
            async for message in result:
                if request.path=='/aux-video' and message.type==WSMsgType.TEXT:
                    await upstream.send_str(message.data)
        tasks=[asyncio.create_task(forward()),asyncio.create_task(closed())]
        try:await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            await upstream.close();await result.close()
        return result

    async def glove_overlay(request):
        running()
        from dexstation.inputs.glove_overlay import DEFAULT_LAYERS
        layers = request['values'].get('layers')
        if isinstance(layers, dict):
            # Pages loaded before layer B was removed still send it; it has no effect.
            layers = {k: v for k, v in layers.items() if k != 'B'}
        if not isinstance(layers, dict) or set(layers) != set(DEFAULT_LAYERS) or any(type(v) is not bool for v in layers.values()):
            raise ValueError('expected enabled/A/C booleans')
        await manager.command(dict(action='glove-overlay', layers=layers, client=request['client']))
        return response()

    async def view(request):
        running()
        value=request['values'].get('view')
        if value != 'camera_3':raise ApiError(400,'invalid_request','Collection main view is fixed to camera_3. Use the positioning panes.')
        await manager.command(dict(action='view',view=value,client=request['client']))
        event('view',request['client'],dict(view=value))
        return response()

    def camera_kwargs(request):
        return dict(camera_name=request.query['camera_name']) if 'camera_name' in request.query else {}

    async def camera_module(request):
        module = (root/'camera-panel.mjs').read_text().replace('/* camera-service-presentation */', services.camera_presentation)
        return web.Response(text=module, content_type='text/javascript')

    async def camera_status(request):
        return web.json_response(await asyncio.to_thread(camera.status, **camera_kwargs(request)), headers={'Cache-Control':'no-store'})

    async def camera_preview(request):
        pixels=await asyncio.to_thread(camera.preview_jpeg, **camera_kwargs(request))
        return web.Response(body=pixels,content_type='image/jpeg',headers={'Cache-Control':'no-store'})

    async def camera_photos(request):
        return web.json_response(await asyncio.to_thread(camera.photos, **camera_kwargs(request)))

    async def camera_photo(request):
        try:
            path=camera.photo(request.match_info['name'], **camera_kwargs(request))
        except FileNotFoundError:
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def camera_write(request):
        action=request.match_info['action']
        methods={'start':camera.start,'stop':camera.stop,'snapshot':camera.snapshot,
                 'record/start':camera.record_start,'record/stop':camera.record_stop}
        method=methods.get(action)
        if method is None:raise web.HTTPNotFound()
        args=(request['values'].get('label',''),) if action=='snapshot' else ()
        value=await asyncio.to_thread(method,*args,**(camera_kwargs(request) if action=='snapshot' else {}))
        event('camera/'+action,request['client'],request['values'])
        # Recording responses remain exactly as returned by the station tool.
        return web.json_response(value,headers={'Cache-Control':'no-store'})

    async def camera_record_state(request):
        return web.json_response(await asyncio.to_thread(camera.record_state),headers={'Cache-Control':'no-store'})

    async def stop_idle():
        # Serialize with recording/start so an accepted start cannot race the stop.
        async with recording_lock:
            if managed and manager.worker and manager.worker.is_alive() and idle.expired(manager.refresh(), manager.pending_start):
                await glove_recorder.cancel()
                await manager.stop('idle-timeout')
                values=dict(idle_timeout_s=idle.timeout_s)
                event('session/idle-stop','service',values)
                # A control event is production audit, not a per-frame diagnostic.
                if manager.output:
                    with (manager.output/'events.jsonl').open('a') as journal:
                        journal.write(json.dumps(dict(action='session/idle-stop',client='service',
                            wall_time_ns=time.time_ns(),**values))+'\n')

    async def inspect():
        nonlocal safety
        while True:
            safety=await asyncio.to_thread(services.safety_check or detect_safety)
            manager.refresh()
            await stop_idle()
            await asyncio.sleep(SAFETY_INTERVAL_S)

    async def measure_stream():
        nonlocal stream,frame_row
        while True:
            if manager.worker and manager.worker.is_alive() and manager.port:
                try:
                    async with client.get(f'http://127.0.0.1:{manager.port}/state',timeout=ClientTimeout(total=2)) as r:
                        row=await r.json()
                        stream=row.get('stream');frame_row=row
                except (ClientError,OSError,asyncio.TimeoutError):stream=None
            else:stream=None;frame_row={}
            await asyncio.sleep(.05)

    async def lifecycle(app):
        nonlocal client
        client=ClientSession()
        tasks=[asyncio.create_task(inspect())]
        if not frame_provider:tasks.append(asyncio.create_task(measure_stream()))
        yield
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        try:
            await glove_recorder.cancel()
            await snapshots.close()
            await asyncio.to_thread(camera.stop)
        finally:
            try:
                if managed:await manager.stop()
                if manager.export_process and manager.export_process.returncode is None:
                    await manager.export_process.wait()
            finally:
                await client.close();audit.close()

    app=web.Application(middlewares=[revalidate_page_code,presence,contract])
    app['glove_recorder']=glove_recorder
    app['scene_snapshots']=snapshots
    app['idle_session']=idle
    app['stop_idle']=stop_idle
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get('/',index)
    app.router.add_get('/video',video_handler or video)
    app.router.add_get('/aux-video',auxiliary_video_handler or video)
    app.router.add_get('/auxiliary-panel.mjs',lambda request:web.FileResponse(root/'auxiliary-panel.mjs'))
    app.router.add_get('/teleop-table.mjs',lambda request:web.FileResponse(root/'teleop-table.mjs'))
    app.router.add_post(API_PREFIX + '/can/bring-up',can_bring_up)
    app.router.add_get('/camera-panel.mjs',camera_module)
    app.router.add_get('/task-cells.mjs',lambda request:web.FileResponse(root/'task-cells.mjs'))
    app.router.add_get(API_PREFIX + '/tasks/{task_id}/cells',task_cells)
    app.router.add_get('/tasks/{task_id}/web/{file:.+}',task_web)
    app.router.add_post(API_PREFIX + '/task/action',task_action)
    app.router.add_post(API_PREFIX + '/glove-recording',glove_recording_start)
    app.router.add_post(API_PREFIX + '/glove-recording/cancel',glove_recording_cancel)
    app.router.add_post(API_PREFIX + '/snapshot',scene_snapshot_start)
    app.router.add_post(API_PREFIX + '/snapshot/cancel',scene_snapshot_cancel)
    app.router.add_get(API_PREFIX + '/snapshots',scene_snapshots)
    app.router.add_get(API_PREFIX + '/snapshots/{name:.+}',scene_snapshot_file)
    app.router.add_get(API_PREFIX + '/camera/status',camera_status)
    app.router.add_get(API_PREFIX + '/camera/preview.jpg',camera_preview)
    app.router.add_get(API_PREFIX + '/camera/photos',camera_photos)
    app.router.add_get(API_PREFIX + '/camera/photos/{name}',camera_photo)
    app.router.add_get(API_PREFIX + '/camera/record/state',camera_record_state)
    app.router.add_post(API_PREFIX + '/camera/{action:start|stop|snapshot|record/start|record/stop}',camera_write)
    app.router.add_static('/overlay',root/'overlay')
    app.router.add_static('/web',root.parents[1]/'render/web')
    for path,handler in [('config',config),('status',status),('events',events),('episodes',episodes),('episodes/{id}',episode)]:
        app.router.add_get(API_PREFIX + '/'+path,handler)
    for path,handler in [('retarget/switch',retarget_switch),('signal-source/switch',signal_source_switch),('session/start',start),('session/stop',stop),('recording/start',recording_start),('recording/stop',recording_stop),
                         ('notes',notes),('episodes/{id}/void',void),('episodes/{id}/export',export),('remote-control',remote),('view',view),('glove-overlay',glove_overlay)]:
        app.router.add_post(API_PREFIX + '/'+path,handler)
    return app


def main():
    import argparse
    from aiohttp import web
    from dexstation.config.site import configure_runtime
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default=site_value('operator.bind_host','127.0.0.1'))
    parser.add_argument('--port',type=int,default=site_value('operator.port',8780))
    parser.add_argument('--enable-remote-control',action='store_true')
    parser.add_argument('--idle-timeout-s',type=float,default=site_value('collection.idle_timeout_s',7200))
    parser.add_argument('--source',choices=source_names())
    parser.add_argument('--workbench',default=DEFAULT_WORKBENCH)
    parser.add_argument('--task',default=site_value('collection.task',DEFAULT_TASK))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--scene-variant',choices=('full','free-space'))
    parser.add_argument('--arm-startup',choices=('spec','first_command','none'))
    parser.add_argument('--motion-profile',help='auto, station, legacy, or a maintained profile JSON')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--ros-topic-prefix')
    args=parser.parse_args();require_loopback(args.host);configure_runtime()
    values={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k in ('source','workbench','task','seed','scene_variant','arm_startup','motion_profile','ros_topic_prefix') and v is not None}
    manager=SessionManager(args.output_root,default_values=values)
    manager.service_port=args.port
    web.run_app(make_app(manager,enable_remote_control=args.enable_remote_control,idle_timeout_s=args.idle_timeout_s),host=args.host,port=args.port,shutdown_timeout=2,access_log=None)
