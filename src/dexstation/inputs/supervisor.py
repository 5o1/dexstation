"""Owned, read-only MANUS/retarget/LinkerTA lifecycle (no hand SDK or Rokae).

The service and the manual shell wrappers share this implementation. Device
inspection reads sysfs and /proc symlinks; it never opens the USB receiver.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from uuid import uuid4

from dexstation.config.site import repository_root, site_value, workspace_root
from dexstation.runtime.logs import CappedProcessLog

MODULE = 'dexstation.inputs.supervisor'
COMPONENTS = ('manus', 'retarget', 'linkerta')
START_TIMEOUT_S = 60
STOP_TIMEOUTS_S = (10, 3, 2)
RETARGET_METHODS = ('manus_anatomical',)


from dexstation.contracts.services import InputConflict, ForeignPublisherError


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    temporary.replace(path)


def processes(proc_root=Path('/proc')):
    result = []
    for path in Path(proc_root).iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            argv = [s.decode(errors='replace') for s in (path/'cmdline').read_bytes().split(b'\0') if s]
            result.append(dict(pid=int(path.name), pgid=int(fields[2]), session=int(fields[3]),
                               start_ticks=int(fields[19]), argv=argv, command=shlex.join(argv)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def receiver_devices(sys_root=Path('/sys/bus/usb/devices'), dev_root=Path('/dev/bus/usb')):
    devices = []
    if not Path(sys_root).is_dir():
        return devices
    for path in Path(sys_root).iterdir():
        try:
            if (path/'idVendor').read_text().strip().lower() == '3325':
                devices.append(Path(dev_root)/f'{int((path/"busnum").read_text()):03d}'/
                               f'{int((path/"devnum").read_text()):03d}')
        except FileNotFoundError:
            continue
    return devices


def receiver_holders(rows, devices, proc_root=Path('/proc'), *, unreadable=None):
    targets = {str(Path(p).resolve()) for p in devices}
    holders = []
    if not targets:
        return holders
    for row in rows:
        try:
            for fd in (Path(proc_root)/str(row['pid'])/'fd').iterdir():
                try:
                    if str(fd.resolve(strict=True)) in targets:
                        holders.append(row)
                        break
                except FileNotFoundError:
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            if unreadable is not None and row['argv']:
                unreadable.append(dict(pid=row['pid'], command=row['command']))
    return holders


def input_identity(argv):
    if not argv:
        return None
    program = Path(argv[0]).name
    args = argv[1:]
    if re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', program):
        i = 0
        while i < len(args):
            if args[i] in ('-c', '-'):
                return None
            if args[i] == '-m':
                if i+1 < len(args) and args[i+1] == 'dexstation.inputs.retarget_node':
                    return 'manus_anatomical_retarget'
                return None
            if args[i] in ('-W','-X','--check-hash-based-pycs'):
                i += 2
                continue
            if not args[i].startswith('-'):
                program = Path(args[i]).name
                args = args[i+1:]
                break
            i += 1
    if program == 'ros2' and len(args) >= 3 and args[0] in ('run', 'launch'):
        return args[1] if args[1] in ('manus_ros2', 'linkerta') else None
    return program if program in ('manus_data_publisher', 'linkerta_node', 'linkerta_reader_node.py') else None


def registration_row(pid, rows=None):
    """The /proc row for a just-started child, or None if it already exited."""
    return next((r for r in (processes() if rows is None else rows) if r['pid'] == pid), None)


def registered_rows(registry, rows, *, legacy_filter=None):
    groups = registry.get('groups', {})
    result = []
    for name, record in groups.items():
        pgid = record['pgid']
        leader = next((r for r in rows if r['pid'] == pgid), None)
        if leader and record.get('start_ticks') is not None and leader['start_ticks'] != record['start_ticks']:
            continue
        members = [r for r in rows if r['pgid'] == pgid and r['session'] == pgid]
        if registry.get('legacy') and (legacy_filter is None or not legacy_filter(name, members)):
            continue
        result.extend(members)
    return result


def describe(rows):
    return '; '.join(f'PID {r["pid"]} {r["command"]}' for r in rows)


class InputChain:
    """Narrow blocking API; SessionManager runs inspection/launch/stop in threads."""
    retarget_methods = RETARGET_METHODS
    def check_can(self):
        if self.can_system is None:
            from dexstation.inputs.linkerta_reader import CanSystem
            self.can_system = CanSystem()
        return self.can_system.require('linkerta')

    def check_publishers(self):
        script = 'set -e; source "$1"; source "$2"; exec "$3" "$4" --inspect-publishers'
        env = dict(os.environ, ROS_DOMAIN_ID=str(site_value('input_example.ros_domain_id', 88)), ROS_LOCALHOST_ONLY='1',
                   PYTHONPATH=str(repository_root()/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
        result = subprocess.run(['bash','-c',script,'input-graph',str(self.ros_setup),
            str(self.ros_environment),str(self.python),
            str(repository_root()/'scripts/linkerta_reader_node.py')],env=env,
            capture_output=True,text=True,timeout=10,check=True)
        names = json.loads(result.stdout.splitlines()[-1])
        if names:
            raise ForeignPublisherError(names)

    def __init__(self, *, log_root=None, ros_environment=None, retarget_python=None, ros_setup=None,
                 proc_root=Path('/proc'), sys_root=Path('/sys/bus/usb/devices'), dev_root=Path('/dev/bus/usb'),
                 retarget='manus_anatomical', can_system=None):
        if retarget not in self.retarget_methods:
            raise ValueError('invalid retarget method')
        self.can_system = can_system
        self.retarget = retarget
        self.log_root = Path(log_root or site_value('input_example.log_root') or workspace_root()/'logs').expanduser().resolve()
        self.ros_environment = Path(ros_environment or site_value('input_example.ros_environment') or '/dev/null').expanduser().resolve()
        self.python = Path(retarget_python or site_value('input_example.python') or sys.executable).expanduser().resolve()
        self.ros_setup = Path(ros_setup or site_value('input_example.ros_setup') or '/opt/ros/humble/setup.bash').expanduser().resolve()
        self.proc_root, self.sys_root, self.dev_root = Path(proc_root), Path(sys_root), Path(dev_root)
        self.run = None
        self.manual = False
        self.manual_run = None
        self.child = None
        self.owner = 'service-'+uuid4().hex
        self.inspection_unreadable = []

    @contextmanager
    def locked(self):
        self.log_root.mkdir(parents=True, exist_ok=True)
        with (self.log_root/'input-chain.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def registry(self, run=None):
        run = run or self.run
        if run is None:
            return {}
        value = read_json(run/'registry.json')
        if value is not None:
            return value
        # Compatibility with the old m2 pids file: those groups were manual.
        try:
            groups = {name: dict(pgid=int(pid)) for name, pid in
                      (line.split() for line in (run/'pids').read_text().splitlines())}
        except FileNotFoundError:
            return {}
        return dict(owner='manual', groups=groups, legacy=True)

    def preflight(self):
        self.manual = False
        rows = processes(self.proc_root)
        legacy = [r for r in rows if input_identity(r['argv']) in ('linkerta_node','linkerta')]
        if legacy:
            raise ForeignPublisherError(['linkerta_node'], describe(legacy))
        pointer = self.log_root/'m2-current'
        candidate = Path(pointer.read_text().strip()) if pointer.exists() else None
        registry = self.registry(candidate)
        allowed = registered_rows(registry, rows) if registry.get('owner') in ('manual', self.owner) else []
        allowed_pids = {r['pid'] for r in allowed}
        foreign = {r['pid']: r for r in rows if input_identity(r['argv']) and r['pid'] not in allowed_pids}
        devices = receiver_devices(self.sys_root, self.dev_root)
        self.inspection_unreadable = []
        for row in receiver_holders(rows, devices, self.proc_root, unreadable=self.inspection_unreadable):
            if row['pid'] not in allowed_pids:
                foreign[row['pid']] = row
        if foreign:
            raise InputConflict('MANUS receiver or input node is in use; refusing to start: '+describe(foreign.values()))
        if allowed:
            method = read_json(candidate/'config.json', {}).get('retarget', 'manus_anatomical')
            if method != self.retarget:
                raise InputConflict(f'Registered input chain uses {method}, selected {self.retarget}; stop that chain before switching')
            live_groups = {r['pgid'] for r in allowed}
            if not all(registry.get('groups', {}).get(name, {}).get('pgid') in live_groups for name in COMPONENTS):
                raise InputConflict('Registered input chain is incomplete; stop it with m2/stop.sh and retry: '+describe(allowed))
            self.run, self.manual = candidate, registry['owner'] == 'manual'
        return self.manual

    def launch(self, *, manual=False):
        with self.locked():
            reused = self.preflight()
            if reused and manual:
                return
            self.manual_run = self.run if reused else None
            self.manual = manual or reused
            owner = 'manual' if manual else self.owner
            self.run = self.log_root/f'm2-{time.strftime("%Y%m%d-%H%M%S")}-{uuid4().hex[:8]}'
            self.run.mkdir()
            write_json(self.run/'config.json', dict(ros_environment=str(self.ros_environment), python=str(self.python),
                ros_setup=str(self.ros_setup), ros_domain_id=site_value('input_example.ros_domain_id', 88),
                owner=owner, log_root=str(self.log_root), observe_only=reused, retarget=self.retarget))
            write_json(self.run/'state.json', dict(state='starting', components={k:dict(state='starting') for k in COMPONENTS}))
            if not reused:
                (self.log_root/'m2-current').write_text(str(self.run)+'\n')
            env = dict(os.environ, PYTHONPATH=str(repository_root()/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
            self.child = subprocess.Popen([sys.executable, '-m', MODULE, 'supervise', '--run', str(self.run)],
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            # Register immediately, so stop during preparation cannot lose a child.
            row = registration_row(self.child.pid)
            groups = {'supervisor':dict(pgid=row['pid'], start_ticks=row['start_ticks'])} if row else {}
            write_json(self.run/'registry.json', dict(owner=owner, groups=groups))

    def status(self):
        value = read_json(self.run/'state.json', {}) if self.run else {}
        if self.run and self.child and self.child.poll() is not None and value.get('state') not in ('stopped','failed'):
            value = dict(value, state='failed', error=f'Input chain manager exited: {self.child.returncode}')
        request=read_json(self.run/'retarget-request.json',{}) if self.run else {}
        if request.get('id') and request['id'] != value.get('switch_id'):
            value=dict(value,retarget=request['method'],state='starting',components=dict(value.get('components',{}),retarget=dict(state='starting')))
        return dict(value or dict(state='stopped', components={k:dict(state='stopped') for k in COMPONENTS}),
                    retarget=value.get('retarget',self.retarget), ownership='manual' if self.manual else 'service', label='Started manually' if self.manual else 'Managed by session',
                    inspection_scope='Readable /proc only; unreadable fds (including protected same-user processes) cannot be inspected',
                    inspection_unreadable=self.inspection_unreadable,
                    log_dir=str(self.run) if self.run else None, manual_log_dir=str(self.manual_run) if self.manual_run else None)

    def registered_processes(self, registry, rows):
        return registered_rows(registry, rows)

    def stop_processes(self, run):
        return stop_registered(run, proc_root=self.proc_root)

    def stop(self, *, include_manual=False):
        if self.run is None or (self.registry().get('owner') == 'manual' and not include_manual):
            return
        self.stop_processes(self.run)
        if self.child:
            self.child.wait(timeout=1)
        self.inspection_unreadable = []
        holders = receiver_holders(processes(self.proc_root), receiver_devices(self.sys_root, self.dev_root), self.proc_root,
                                   unreadable=self.inspection_unreadable)
        value = self.status()
        # A reused manual chain is deliberately still holding its receiver.
        manual_pids = {r['pid'] for r in self.registered_processes(self.registry(self.manual_run), processes(self.proc_root))} if self.manual_run else set()
        unexpected = [r for r in holders if r['pid'] not in manual_pids]
        value.update(state='failed' if unexpected else 'stopped', receiver_released=not holders,
                     manual_still_running=bool(manual_pids))
        for component in value.get('components',{}).values():
            component['state'] = 'stopped'
        if unexpected:
            value['error'] = 'MANUS receiver not yet released: '+describe(holders)
        write_json(self.run/'state.json', value)
        if unexpected:
            raise RuntimeError(value['error'])


    def switch_retarget(self, method):
        if method not in self.retarget_methods:
            raise ValueError('invalid retarget method')
        if self.manual or self.run is None:
            raise InputConflict('A manual or stopped input chain cannot be switched from the web page')
        write_json(self.run/'retarget-request.json', dict(method=method, id=uuid4().hex))


def retarget_command(config):
    method = config.get('retarget', 'manus_anatomical')
    if method not in RETARGET_METHODS:
        raise ValueError('invalid retarget method')
    return [config['python'], '-m', 'dexstation.inputs.retarget_node', '--method', method,
            '--ros-args', '--disable-external-lib-logs']


def replace_retarget(run, config, children, logs, env, *, command_factory=None):
    """Supervisor-only replacement; stop only its registered retarget group."""
    child = children['retarget']
    record = read_json(run/'registry.json')['groups']['retarget']
    rows = registered_rows(dict(groups={'retarget':record}),processes())
    if rows:
        for sig, delay in zip((signal.SIGINT,signal.SIGTERM,signal.SIGKILL),STOP_TIMEOUTS_S):
            if not registered_rows(dict(groups={'retarget':record}),processes()):break
            os.killpg(record['pgid'],sig)
            try:child.wait(timeout=delay)
            except subprocess.TimeoutExpired:pass
    child.wait(timeout=1)
    observed=read_json(run/'observed.json',{});observed.pop('retarget',None)
    write_json(run/'observed.json',observed)
    write_json(run/'retarget-generation.json',dict(time=time.monotonic()))
    log=CappedProcessLog(run/'retarget.log');logs.append(log)
    new=subprocess.Popen((command_factory or retarget_command)(config),env=env,cwd=run,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children['retarget']=new
    row=next(r for r in processes() if r['pid']==new.pid)
    registry=read_json(run/'registry.json');registry['groups']['retarget']=dict(pgid=new.pid,start_ticks=row['start_ticks'])
    write_json(run/'registry.json',registry)


def stop_registered(run, *, proc_root=Path('/proc'), exclude=(), timeouts=STOP_TIMEOUTS_S, registered=None):
    """Escalate only registered sessions, including children of dead leaders."""
    registered = registered or registered_rows
    for sig, timeout in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), timeouts):
        registry = read_json(run/'registry.json')
        if registry is None:
            registry = InputChain(proc_root=proc_root).registry(run)
        rows = [r for r in registered(registry, processes(proc_root)) if r['pgid'] not in exclude]
        for pgid in {r['pgid'] for r in rows}:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            # Re-read: a supervisor may have been registering a child at stop.
            registry = read_json(run/'registry.json', registry)
            remaining = [r for r in registered(registry, processes(proc_root)) if r['pgid'] not in exclude]
            if not remaining:
                return
            time.sleep(.05)
    raise RuntimeError('Input process group did not exit: '+describe(remaining))


def prepare(config, run):
    """Use explicitly installed ROS packages, without reading an upstream checkout."""
    env = dict(os.environ)
    for key in ('AMENT_PREFIX_PATH','COLCON_PREFIX_PATH','CMAKE_PREFIX_PATH','PYTHONPATH'):
        env.pop(key, None)
    command = 'set -e; source "$1"; source "$2"; exec "$3" -c "import os,json;print(json.dumps(dict(os.environ)))"'
    result = subprocess.run(['bash','-c',command,'input-chain',config['ros_setup'],
        config['ros_environment'],sys.executable],env=env,capture_output=True,text=True,timeout=30,check=True)
    env = json.loads(result.stdout.splitlines()[-1])
    env.update(ROS_DOMAIN_ID=str(config.get('ros_domain_id', 88)), ROS_LOCALHOST_ONLY='1',
               ROS_LOG_DIR=str(run/'ros'), ROS_HOME=str(Path(config['log_root']).parent/'cache/ros'),
               PYTHONPATH=str(repository_root()/'src')+os.pathsep+env.get('PYTHONPATH',''),
               OVERRIDE_LAUNCH_PROCESS_OUTPUT='screen')
    for directory in (env['ROS_LOG_DIR'], env['ROS_HOME']):
        Path(directory).mkdir(parents=True, exist_ok=True)
    return env


def manus_diagnostics(run):
    path = run/'manus.log'
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-65536))
            lines = stream.read().decode(errors='replace').splitlines()
    except FileNotFoundError:
        return []
    pattern = re.compile(r'error|fail|could not|cannot|unable|busy|denied|permission|occupied|\u5360\u7528|\u6743\u9650', re.I)
    return [line for line in lines if pattern.search(line)][-8:]


def supervise(run, *, prepare_environment=None, command_factory=None,
              replace=None, stop_processes=None, module=MODULE, start_timeout_s=None,
              default_retarget='manus_anatomical'):
    """Detached supervisor keeps bounded output drains alive for manual launches."""
    prepare_environment = prepare_environment or prepare
    command_factory = command_factory or retarget_command
    replace = replace or replace_retarget
    stop_processes = stop_processes or stop_registered
    start_timeout_s = START_TIMEOUT_S if start_timeout_s is None else start_timeout_s
    config = read_json(run/'config.json')
    stopping = False
    def requested(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, requested)
    signal.signal(signal.SIGTERM, requested)
    children, logs = {}, []
    state = read_json(run/'state.json')
    deadline = time.monotonic()+start_timeout_s
    try:
        # The launcher writes the initial registration immediately after Popen.
        while not (run/'registry.json').exists() and not stopping:
            time.sleep(.01)
        if stopping:
            return
        env = prepare_environment(config, run)
        no_disk_log = ['--ros-args', '--disable-external-lib-logs']
        commands = dict(probe=[config['python'],'-m',module,'probe','--run',str(run)],
            manus=['ros2','run','manus_ros2','manus_data_publisher',*no_disk_log],
            retarget=command_factory(config),
            linkerta=[config['python'],str(repository_root()/'scripts/linkerta_reader_node.py'),
                      '--ros-args','--params-file',str(Path(__file__).with_name('linkerta_assets')/'linkerta.yaml')])
        for name, command in commands.items():
            if stopping:
                break
            if config.get('observe_only') and name != 'probe':
                continue
            log = CappedProcessLog(run/f'{name}.log')
            logs.append(log)
            child = subprocess.Popen(command, env=env, cwd=run, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children[name] = child
            # A child that exits at once has no group left to register; the
            # loop below reports it as failed without stopping the others.
            row = registration_row(child.pid)
            if row:
                registry = read_json(run/'registry.json')
                registry['groups'][name] = dict(pgid=row['pid'],start_ticks=row['start_ticks'])
                write_json(run/'registry.json', registry)
        (run/'pids').write_text(''.join(f'{k} {v.pid}\n' for k,v in children.items() if k in COMPONENTS))
        # Components are independent: missing glove data or one failed input
        # never stops the others, so LinkerTA keeps driving the arms.
        failed = {}
        switch_id = None
        fatal = re.compile(r'permission denied|access denied|device.*busy|resource busy|failed to open|cannot open|unable to open|LIBUSB_ERROR_(?:BUSY|ACCESS)', re.I)
        while not stopping:
            request = read_json(run/'retarget-request.json',{})
            if request.get('id') and request['id'] != switch_id:
                config['retarget'] = request['method']
                state['components']['retarget'] = dict(state='starting')
                state.update(retarget=config['retarget'],state='starting')
                write_json(run/'state.json',state)
                replace(run,config,children,logs,env)
                write_json(run/'config.json',config)
                switch_id=request['id'];state['switch_id']=switch_id;failed.pop('retarget',None)
                deadline=time.monotonic()+start_timeout_s
            critical = manus_diagnostics(run)
            if 'manus' not in failed and any(fatal.search(line) for line in critical):
                failed['manus'] = 'MANUS receiver cannot be opened: it may be held by another user\'s process, or device permissions are insufficient. '+'\n'.join(critical)
            for name, child in children.items():
                if child.poll() is None or name in failed:
                    continue
                if name not in COMPONENTS:
                    raise RuntimeError(f'{name} exited ({child.returncode}); see {run}/{name}.log')
                detail = '\n'.join(critical) if name == 'manus' else ''
                failed[name] = f'{name} exited ({child.returncode}); see {run}/{name}.log'+('\n'+detail if detail else '')
            if all(name in failed for name in COMPONENTS):
                raise RuntimeError('\n'.join(failed[name] for name in COMPONENTS))
            observed = read_json(run/'observed.json', {})
            overdue = time.monotonic() >= deadline
            waiting_text = dict(
                manus='No MANUS data: gloves are off, or the receiver is held by another user\'s process. '+'\n'.join(critical),
                retarget='No retarget output' + ('' if observed.get('manus') else ': waiting for MANUS data'),
                linkerta='No LinkerTA data')
            components = {}
            for name in COMPONENTS:
                if name in failed:
                    components[name] = dict(state='failed', first_data_monotonic_s=observed.get(name), error=failed[name])
                elif observed.get(name):
                    components[name] = dict(state='ready', first_data_monotonic_s=observed[name])
                elif overdue:
                    components[name] = dict(state='waiting', first_data_monotonic_s=None, error=waiting_text[name])
                else:
                    components[name] = dict(state='starting', first_data_monotonic_s=None)
            states = {c['state'] for c in components.values()}
            overall = 'ready' if states == {'ready'} else 'degraded' if states & {'failed', 'waiting'} else 'starting'
            current = dict(manus_log_lines=critical, state=overall, components=components, retarget=config.get('retarget',default_retarget), switch_id=switch_id)
            if current != state:
                state = current
                write_json(run/'state.json', state)
            # Launch and native ROS files are bounded as well as stdout pipes.
            for path in (run/'ros').rglob('*'):
                if path.is_file() and path.stat().st_size > 1024*1024:
                    with path.open('r+b') as stream:
                        stream.truncate(0)
            time.sleep(.1)
    except Exception as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        state.update(state='failed', error=str(detail).strip())
        for component in state.get('components',{}).values():
            if component.get('state') != 'ready':
                component.update(state='failed',error=state['error'])
    finally:
        try:
            stop_processes(run, exclude=(os.getpgrp(),))
        except Exception as exc:
            state.update(state='failed', error=str(exc))
        for child in children.values():
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                state.update(state='failed', error=f'Input child process did not exit: PID {child.pid}')
        for log in logs:
            log.close()
        if state.get('state') != 'failed':
            state['state'] = 'stopped'
        write_json(run/'state.json',state)


def probe(run):
    """Observe real messages, never infer raw MANUS readiness from retarget Hz."""
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs.msg import JointState
    rclpy.init()
    node = rclpy.create_node('gamepad_input_chain_health', enable_rosout=False, start_parameter_services=False)
    seen, subscriptions, topics = {}, [], set()
    def receive(kind, message):
        generation=read_json(run/'retarget-generation.json',{}).get('time',0) if kind=='retarget' else 0
        if kind not in seen or seen[kind] < generation:
            seen[kind] = time.monotonic()
            write_json(run/'observed.json', seen)
    def subscribe(topic, message_type, kind):
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT if kind == 'linkerta' else ReliabilityPolicy.RELIABLE)
        subscriptions.append(node.create_subscription(message_type, topic, lambda msg: receive(kind,msg), qos))
        topics.add(topic)
    for side in ('left','right'):
        subscribe(f'/cb_{side}_hand_control_cmd',JointState,'retarget')
        subscribe(f'/{side}_arm_joint_control',JointState,'linkerta')
    try:
        while rclpy.ok():
            for topic, types in node.get_topic_names_and_types():
                if re.fullmatch(r'/manus_glove_\d+',topic) and topic not in topics and types == ['manus_ros2_msgs/msg/ManusGlove']:
                    subscribe(topic,get_message(types[0]),'manus')
            rclpy.spin_once(node,timeout_sec=.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    start = actions.add_parser('start')
    for name in ('log-root','ros-environment','retarget-python','ros-setup'):
        start.add_argument('--'+name, type=Path)
    start.add_argument('--retarget', choices=RETARGET_METHODS, default='manus_anatomical')
    actions.add_parser('stop').add_argument('--log-root',type=Path)
    for action in ('supervise','probe'):
        actions.add_parser(action).add_argument('--run',type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'supervise':
        supervise(args.run)
        return
    if args.action == 'probe':
        probe(args.run)
        return
    chain = InputChain(**{key:getattr(args,key,None) for key in ('log_root','ros_environment','retarget_python','ros_setup')},
                       retarget=getattr(args,'retarget','manus_anatomical'))
    try:
        if args.action == 'start':
            chain.launch(manual=True)
            while chain.status()['state'] == 'starting':
                time.sleep(.1)
            if chain.status()['state'] == 'failed':
                raise RuntimeError(chain.status().get('error','Input chain is not ready'))
        else:
            pointer = chain.log_root/'m2-current'
            if pointer.exists():
                chain.run = Path(pointer.read_text().strip())
                chain.manual = chain.registry().get('owner') == 'manual'
                chain.stop(include_manual=True)
        print(json.dumps(chain.status(),ensure_ascii=False))
    except (RuntimeError, OSError, ValueError) as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
