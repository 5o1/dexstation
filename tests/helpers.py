"""Shared test builders and validators; no hardware or rendering runtime required."""
import asyncio
import atexit
from functools import cache
import runpy
import shutil
import tempfile
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from types import SimpleNamespace as NS

import numpy as np

from dexstation.inputs.supervisor import COMPONENTS, InputChain, processes, write_json
from dexstation.server.app import SessionManager
from dexstation.recording.episode import Observation, EpisodeWriter
from dexstation.contracts.backend import FakeBackend

ROOT = Path(os.environ.get('DEXSTATION_TEST_PUBLIC_ROOT', Path(__file__).resolve().parents[1]))
EXAMPLES = ROOT / 'src/dexstation/contracts/examples'
SCHEMAS = ROOT / 'src/dexstation/contracts/schemas'

if importlib.util.find_spec('jsonschema') is not None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError
else:
    Draft202012Validator = None


def copy_site_runtime(package):
    """Clone the site's minimal import/data dependencies into an isolated test checkout."""
    for name in ('config/site.py', 'config/defaults.py', 'config/environment.py', 'config/__init__.py', 'contracts/__init__.py',
                 'contracts/validation.py', 'contracts/schemas/site.schema.json'):
        target = Path(package) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'src/dexstation' / name, target)

def observation(t=.02):
    pose = np.array([0., 0., 1., 1., 0., 0., 0.])
    return Observation(t, 123456, np.zeros((2,21)), np.zeros((2,7)), np.tile(pose, (2,1)),
                       np.tile(pose, (2,5,1)), np.zeros(32), np.ones(32, bool), {"button": .001}, False, "test")


def spec():
    return SimpleNamespace(**{s+'_'+k:SimpleNamespace(home_q=[0.]*n) for s in ('left','right') for k,n in (('arm',7),('hand',21))})


def gamepad_camera_task():
    """Read presentation data without importing a task implementation."""
    return json.loads((ROOT / 'src/dexstation/task_plugins/xbox_gamepad_v3_3/data/task.json').read_text())


def camera_backend(backend):
    """Supply the generic runtime/object camera interface for numerical tests."""
    pose = getattr(getattr(getattr(backend, 'spec', None), 'gamepad', None),
                   'root_pose', (0, 0, 0, 1, 0, 0, 0))
    backend.runtime_scene = NS(task=gamepad_camera_task())
    backend.objects = {'gamepad': NS(definition=NS(initial_pose_world=pose))}
    return backend


def fake_mesh(backend, monkeypatch):
    """Supply SAPIEN's mesh/pose API without loading a rendering runtime."""
    monkeypatch.setitem(sys.modules, 'sapien.render', NS(RenderBodyComponent=object))
    pose = NS(to_transformation_matrix=lambda: np.eye(4))
    part = NS(vertices=np.array([[-.08,-.04,-.025],[.08,.04,.025],[-.08,.04,.025],[.08,-.04,-.025]]))
    shape = NS(parts=[part], scale=np.ones(3), local_pose=pose)
    entity = NS(pose=pose, find_component_by_type=lambda kind: NS(render_shapes=[shape]))
    articulation = getattr(getattr(backend, 'gamepad', None), 'articulation', NS())
    articulation.get_links = lambda: [NS(entity=entity)]
    backend.gamepad = NS(articulation=articulation)
    camera_backend(backend)
    backend.objects['gamepad'].links = articulation.get_links
    backend.table = NS(per_scene_id=9001)
    if not hasattr(backend,'fixtures'):backend.fixtures=[]
    return shape, entity


def usb_tree(root):
    usb = root/'sys/1-2'
    usb.mkdir(parents=True)
    for key, value in dict(idVendor='3325', busnum='1', devnum='2').items():
        (usb/key).write_text(value)
    device = root/'usb/001/002'
    device.parent.mkdir(parents=True)
    device.touch()
    return root/'sys', root/'usb', device


def proc_entry(root, pid, argv, *, pgid=None, session=None, start=100):
    path = root/str(pid)
    (path/'fd').mkdir(parents=True)
    values = ['S','1',str(pgid or pid),str(session or pgid or pid)]+['0']*15+[str(start)]
    (path/'stat').write_text(f'{pid} (test process) '+ ' '.join(values))
    (path/'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv)+b'\0')
    return path


def fake_chain(tmp_path):
    proc = tmp_path/'proc';proc.mkdir(parents=True)
    sys_root, dev_root, device = usb_tree(tmp_path)
    chain = InputChain(log_root=tmp_path/'logs',proc_root=proc,sys_root=sys_root,dev_root=dev_root)
    chain.check_can = lambda: dict(ready=True)  # fake system boundary
    chain.check_publishers = lambda: None
    return chain,device


class SurrogateChain(InputChain):
    """Use production registry/stop logic with inert receiver/ready substitutes."""
    def __init__(self, root, delay=5, never=False):
        sys_root,dev_root,self.device=usb_tree(root)
        super().__init__(log_root=root/'logs',sys_root=sys_root,dev_root=dev_root)
        self.delay,self.never=delay,never
        self.children=[]
        self.launched=threading.Event()
    def preflight(self):return False
    def check_can(self):return dict(ready=True)
    def check_publishers(self):pass
    def launch(self):
        self.run=self.log_root/'surrogate';self.run.mkdir(parents=True)
        registry=dict(owner=self.owner,groups={})
        for name in COMPONENTS:
            marker=self.run/(name+'.ready')
            script='''import pathlib,sys,time
f=open(sys.argv[1],'rb'); pathlib.Path(sys.argv[2]+'.opened').touch()
time.sleep(float(sys.argv[3]))
if sys.argv[4]=='0':
 p=pathlib.Path(sys.argv[2]+'.tmp'); p.write_text(str(time.monotonic())); p.replace(sys.argv[2])
time.sleep(300)
'''
            child=subprocess.Popen([sys.executable,'-c',script,str(self.device),str(marker),str(self.delay),str(int(self.never))],
                start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            self.children.append(child)
            row=next(r for r in processes() if r['pid']==child.pid)
            registry['groups'][name]=dict(pgid=child.pid,start_ticks=row['start_ticks'])
        write_json(self.run/'registry.json',registry)
        self.launched.set()
    def status(self):
        if self.run and (self.run/'state.json').exists():return super().status()
        components={name:dict(state='ready' if self.run and (self.run/(name+'.ready')).exists() else 'starting',
            first_data_monotonic_s=float((self.run/(name+'.ready')).read_text()) if self.run and (self.run/(name+'.ready')).exists() else None) for name in COMPONENTS}
        return dict(state='ready' if all(v['state']=='ready' for v in components.values()) else 'starting',components=components)
    def stop(self):
        try:super().stop()
        finally:
            for child in self.children:child.wait(timeout=3)


class FakeWorker:
    def __init__(self,options,stop_event,statuses,prepare=3):
        self.options,self.stop_event,self.statuses=options,stop_event,statuses
        self.process=None;self.prepare=prepare
    def start(self):
        script="""import pathlib,sys,time
time.sleep(float(sys.argv[2]))
output=pathlib.Path(sys.argv[1]);output.mkdir(parents=True)
(output/'events.jsonl').open('x').close()
view=output/'view';view.mkdir()
(view/'first_frame').write_text(str(time.monotonic()))
time.sleep(300)
"""
        self.process=subprocess.Popen([sys.executable,'-c',script,str(self.options.output),str(self.prepare)],
            start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    @property
    def exitcode(self):return self.process.poll() if self.process else None
    def is_alive(self):return self.process is not None and self.process.poll() is None
    def join(self,timeout):
        if self.stop_event.is_set() and self.is_alive():self.process.send_signal(signal.SIGINT)
        try:self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:pass
    def crash(self):self.process.kill();self.process.wait()


def session_manager(root,chain=None,prepare=3):
    manager=SessionManager(root/'sessions',default_values=dict(backend='sapien',source='ros2',scene_variant='full',duration=None,rate_hz=60))
    manager.input_chain_factory=lambda:chain
    manager.ctx=SimpleNamespace(Queue=lambda **kw:queue.Queue(),Event=threading.Event,
        Process=lambda *,target,args:FakeWorker(args[0],args[2],args[3],prepare))
    return manager


async def until(predicate, timeout=10):
    deadline=time.monotonic()+timeout
    while not predicate():
        assert time.monotonic()<deadline,'condition did not become ready'
        await asyncio.sleep(.02)


def metadata(task='empty'):
    index=json.loads((EXAMPLES/('episode_empty.json' if task=='empty' else 'episode_gamepad.json')).read_text())
    index.update(fps=50, scene_spec_text='{}', recording_start_sim_time_s=0., recording_end_sim_time_s=.04,
                 physics_timestep_s=.001, received_input_messages=0, receive_start_sequence=0, receive_end_sequence=0)
    return index


def writer_for(path, task='empty'):
    profile=json.loads((EXAMPLES/(task+'.json')).read_text())
    return EpisodeWriter(path, metadata(task), task_id=task,
        task_format=None if task=='empty' else 'xbox-gamepad-rec-v1',
        workbench_meta={'workbench_profile_text':'{}'},
        task_meta={'task_profile':profile,'task_profile_text':json.dumps(profile),
                   'asset_versions':{'gamepad':'0.4.3'},'seed':17,'initial_poses':{}})


def observations(task='empty'):
    backend=FakeBackend()
    for step in range(2):
        backend.step(.02)
        obs=backend.observe()
        obs.robot.wall_time_ns=123456+step
        if task!='empty':
            obs.task={'gamepad_signal':np.arange(32,dtype=float)/31,
                      'gamepad_mask':np.arange(32)%2==0,
                      'control_joint_positions':{'button':.001,'axis':-0.0}}
        yield obs


def append(writer, obs, *, clamps=None, commands=None):
    writer.append(obs, commands or {}, applied_monotonic_ns=99, input_sequences={},
                  arm_clamps=clamps or {'left':None,'right':None})




class ContractValidationError(ValueError):
    pass


def _json_equal(left, right):
    # JSON Schema does not equate true with 1 (Python normally does).
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[k], right[k]) for k in left)
    return left == right


def _minimal_validate(value, schema, *, root=None, path='$'):
    """Subset: local refs, types, objects, arrays, unions and scalar constraints."""
    if isinstance(schema, bool):
        if not schema:
            raise ContractValidationError(f'{path}: forbidden value')
        return
    root = schema if root is None else root
    supported = {'$schema', '$id', '$defs', '$ref', 'title', 'description',
                 'type', 'const', 'enum', 'required', 'properties', 'additionalProperties',
                 'items', 'prefixItems', 'minItems', 'maxItems', 'uniqueItems',
                 'minimum', 'maximum', 'exclusiveMinimum', 'minLength', 'pattern',
                 'anyOf', 'oneOf'}
    if unknown := schema.keys() - supported:
        raise RuntimeError(f'Unsupported schema keywords: {sorted(unknown)}')
    if '$ref' in schema:
        reference = schema['$ref']
        if not reference.startswith('#/'):
            raise RuntimeError('Minimal validator supports only local references')
        target = root
        for key in reference[2:].split('/'):
            target = target[key.replace('~1', '/').replace('~0', '~')]
        _minimal_validate(value, target, root=root, path=path)
    for keyword in ('anyOf', 'oneOf'):
        if keyword in schema:
            matches = 0
            for option in schema[keyword]:
                try:
                    _minimal_validate(value, option, root=root, path=path)
                except ContractValidationError:
                    continue
                matches += 1
            if (keyword == 'anyOf' and matches == 0) or (keyword == 'oneOf' and matches != 1):
                raise ContractValidationError(f'{path}: failed {keyword} ({matches} matches)')
    types = {
        'null': lambda v: v is None,
        'boolean': lambda v: isinstance(v, bool),
        'integer': lambda v: not isinstance(v, bool) and isinstance(v, (int, float)) and float(v).is_integer(),
        'number': lambda v: not isinstance(v, bool) and isinstance(v, (int, float)),
        'string': lambda v: isinstance(v, str),
        'array': lambda v: isinstance(v, list),
        'object': lambda v: isinstance(v, dict),
    }
    if 'type' in schema:
        expected = schema['type']
        expected = expected if isinstance(expected, list) else [expected]
        if not any(types[t](value) for t in expected):
            raise ContractValidationError(f'{path}: expected {expected}')
    if 'const' in schema and not _json_equal(value, schema['const']):
        raise ContractValidationError(f'{path}: invalid constant')
    if 'enum' in schema and not any(_json_equal(value, v) for v in schema['enum']):
        raise ContractValidationError(f'{path}: value outside enum')
    if isinstance(value, dict):
        for key in schema.get('required', []):
            if key not in value:
                raise ContractValidationError(f'{path}: missing {key}')
        properties = schema.get('properties', {})
        for key, item in value.items():
            subschema = properties.get(key, schema.get('additionalProperties', True))
            _minimal_validate(item, subschema, root=root, path=f'{path}.{key}')
    if isinstance(value, list):
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', float('inf')):
            raise ContractValidationError(f'{path}: invalid array length')
        if schema.get('uniqueItems') and any(_json_equal(value[i], value[j])
                                             for i in range(len(value)) for j in range(i)):
            raise ContractValidationError(f'{path}: duplicate items')
        prefix = schema.get('prefixItems', [])
        for index, item in enumerate(value):
            subschema = prefix[index] if index < len(prefix) else schema.get('items', True)
            _minimal_validate(item, subschema, root=root, path=f'{path}[{index}]')
    if isinstance(value, str):
        if len(value) < schema.get('minLength', 0):
            raise ContractValidationError(f'{path}: string too short')
        if 'pattern' in schema and not re.search(schema['pattern'], value):
            raise ContractValidationError(f'{path}: pattern mismatch')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get('minimum', -float('inf')) or value > schema.get('maximum', float('inf')):
            raise ContractValidationError(f'{path}: number outside bounds')
        if 'exclusiveMinimum' in schema and value <= schema['exclusiveMinimum']:
            raise ContractValidationError(f'{path}: number below exclusive minimum')


def validate_instance(value, schema):
    if Draft202012Validator is not None:
        Draft202012Validator.check_schema(schema)
        try:
            Draft202012Validator(schema).validate(value)
        except ValidationError as exc:
            raise ContractValidationError(str(exc)) from exc
    else:
        _minimal_validate(value, schema)


def load_schema(name):
    return json.loads((SCHEMAS / f'{name}.schema.json').read_text())



@cache
def bench_directory():
    """Use a valid regular-file bench copy without policing committed hashes."""
    temporary = tempfile.TemporaryDirectory(prefix='dexhand-c2-bench-')
    atexit.register(temporary.cleanup)
    directory = Path(temporary.name) / 'bench'
    shutil.copytree(ROOT / 'src/dexstation/examples/ar5_g20/data', directory)
    rehash = runpy.run_path(str(ROOT / 'scripts/rehash_workbench.py'))['rehash']
    rehash(directory)
    return directory


def bench_scene(task='xbox_gamepad_v3_3'):
    """Compose maintained profiles for tests that need the actual bench geometry."""
    from dexstation.profiles.compose import compose, load_task, load_workbench
    return compose(load_workbench(bench_directory()),
                   load_task(task), 'sapien', seed=0)


def bench_spec():
    return bench_scene().spec


def bench_data():
    return json.loads(bench_scene().frozen['scene_text'])


GOLDEN_ROOT = ROOT / 'tests/fixtures/golden'


def updating_golden():
    import os
    return os.environ.get('DEXSTATION_UPDATE_GOLDEN') == '1'


def canonical_golden(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'


def assert_frozen_numeric(actual, expected, path='value'):
    """Compare frozen floats within both absolute and relative 1e-12 bounds.

    Container/scalar types, array dtypes/shapes, keys, and nonfloats stay exact.
    Both bounds must hold, so a large magnitude never widens the absolute
    allowance past 1e-12. No fixture or runtime value is rounded.
    """
    import math
    assert type(actual) is type(expected), f'{path}: type changed'
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, f'{path}: dtype changed'
        assert actual.shape == expected.shape, f'{path}: shape changed'
        assert_frozen_numeric(actual.tolist(), expected.tolist(), path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f'{path}: keys changed'
        for key in expected:
            assert_frozen_numeric(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), f'{path}: length changed'
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_frozen_numeric(a, b, f'{path}[{index}]')
    elif isinstance(expected, float):
        if actual == expected:
            return
        assert math.isfinite(actual) and math.isfinite(expected), f'{path}: nonfinite float changed'
        difference = abs(actual - expected)
        tolerance = min(1e-12, 1e-12 * max(abs(actual), abs(expected)))
        assert difference <= tolerance, f'{path}: float exceeds absolute/relative 1e-12 bounds: {actual!r} != {expected!r}'
    else:
        assert actual == expected, f'{path}: exact value changed'


def assert_golden(relative, value, *, numeric=False):
    """Compare canonical JSON with a unified diff; updates are explicitly opt-in."""
    import difflib
    path = GOLDEN_ROOT / relative
    actual = canonical_golden(value)
    if updating_golden():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding='utf-8')
        return
    assert path.is_file(), f'Missing golden: {path}. Run DEXSTATION_UPDATE_GOLDEN=1 after review.'
    expected = path.read_text(encoding='utf-8')
    diff = ''.join(difflib.unified_diff(expected.splitlines(True), actual.splitlines(True),
                                        fromfile=str(path), tofile='actual'))
    if numeric:
        try:
            assert_frozen_numeric(json.loads(actual), json.loads(expected))
        except AssertionError as error:
            raise AssertionError(f'{error}\n{diff}') from error
        return
    assert expected == actual, f'Golden changed; review before using DEXSTATION_UPDATE_GOLDEN=1.\n{diff}'


def golden_scene(task_path, tmp_path, monkeypatch):
    """Compose real profiles with isolated path settings and the C2 bench copy."""
    from dexstation.profiles.compose import compose, load_task, load_workbench
    site = tmp_path / 'site.json'
    site.write_text('{}', encoding='utf-8')
    monkeypatch.setenv('DEXSTATION_SITE_CONFIG', str(site))
    monkeypatch.setenv('DEXSTATION_WORKSPACE', str(tmp_path / 'workspace'))
    monkeypatch.setenv('DEXSTATION_DATA', str(tmp_path / 'data'))
    return compose(load_workbench(bench_directory()), load_task(task_path), 'sapien', seed=17)
