"""Entry-point discovery and task entries, with isolated extension failures."""
from dataclasses import dataclass, field
import importlib
from importlib import metadata
import logging
from threading import RLock
from pathlib import Path

_LOG = logging.getLogger(__name__)
_GROUP_MODULES = {
    'dexstation.sources': 'inputs.registry',
    'dexstation.backends': 'backends.registry',
    'dexstation.object_types': 'backends.object_types',
    'dexstation.channel_types': 'backends.channel_types',
    'dexstation.tasks': 'runtime.plugins',
}
_register = {}
_discovered = set()
_errors = []
_lock = RLock()


def register_group(group, register):
    """Connect a registry's single-entry registration function to discovery."""
    _register[group] = register


def discover_plugins(group=None):
    """Load each group's entry points once; isolate and report broken entries."""
    with _lock:
        for name in ((group,) if group else _GROUP_MODULES):
            if name in _discovered:
                continue
            if name not in _register:
                importlib.import_module('dexstation.' + _GROUP_MODULES[name])
            _discovered.add(name)  # Also prevents recursive discovery while loading.
            if name == 'dexstation.sources':
                registry = importlib.import_module('dexstation.inputs.registry')
                registry.register_task_sources()
            try:
                entries = metadata.entry_points(group=name)
            except Exception as exc:
                _failure(name, '<discovery>', exc)
                continue
            for entry in entries:
                try:
                    _register[name](entry.load())
                except Exception as exc:
                    _failure(name, entry.name, exc)
                else:
                    _LOG.info('Loaded plugin %s:%s', name, entry.name)


def _failure(group, name, exc):
    error = dict(group=group, name=name, reason=' '.join(str(exc).split()))
    _errors.append(error)
    _LOG.warning('Could not load plugin %s:%s: %s', group, name, error['reason'])


def plugin_status():
    """Omit status for the unchanged built-in API; include actionable failures."""
    discover_plugins()
    return {'plugin_status': {'errors': list(_errors)}} if _errors else {}


def lazy_factory(module, name):
    """Resolve optional implementations only when their factory is called."""
    def factory(*args, **kwargs):
        return getattr(importlib.import_module(module), name)(*args, **kwargs)
    return factory


@dataclass(frozen=True)
class TaskPlugin:
    """A task's profile/resources and optional Python hooks.

    Providers may be module names or objects. Full module names declared in a
    profile must be inside this entry's package; IDs never determine imports.
    """
    id: str
    package: str | object
    data_directory: Path
    web_directory: Path
    sources: tuple = ()
    activate: object = None
    api_version: int = 1
    modules: dict = field(default_factory=dict)

    def resolve(self, module=None):
        provider = self.package if module is None else self.modules.get(module, module)
        if isinstance(provider, str):
            if module is not None and module not in self.modules:
                if not isinstance(self.package, str) or not provider.startswith(self.package + '.'):
                    raise ValueError(f'Task {self.id} does not provide module {module}')
            provider = importlib.import_module(provider)
        if getattr(provider, 'API_VERSION', self.api_version) != 1:
            raise ValueError(f'Task {self.id} needs task API version 1')
        if module is None:
            actions = getattr(provider, 'ACTIONS', {})
            if not isinstance(actions, dict) or any(not callable(hook) for hook in actions.values()):
                raise ValueError('task ACTIONS must map symbols to callables')
            for name in ('cell_state', 'task_sets'):
                if hasattr(provider, name) and not callable(getattr(provider, name)):
                    raise ValueError(f'task {name} must be callable')
        return provider

    def task_sets(self, source=None):
        hook = getattr(self.resolve(), 'task_sets', None)
        return list(hook(self.data_directory, source)) if hook else []

    def evaluate(self, episode, **kwargs):
        import json
        profile = json.loads((Path(self.data_directory) / 'task.json').read_text())
        module = self.resolve(profile['evaluation']['offline'])
        return module.evaluate(episode, **kwargs)


_TASKS = {}
_task_builtins = False
_active_tasks = set()


def register_task(entry):
    import json
    from dexstation.inputs.registry import validate_source_entry, _ENTRIES
    if not isinstance(entry, TaskPlugin) or not isinstance(entry.id, str) or not entry.id:
        raise ValueError('task entry needs a TaskPlugin with a nonempty id')
    if entry.api_version != 1:
        raise ValueError('task entry needs task API version 1')
    if entry.activate is not None and not callable(entry.activate):
        raise ValueError('task activation hook must be callable')
    if entry.package is None or entry.package == '':
        raise ValueError('task entry needs a package or hook object')
    if entry.id in _TASKS:
        raise ValueError(f'task already registered: {entry.id}')
    profile = json.loads((Path(entry.data_directory) / 'task.json').read_text())
    if profile.get('id') != entry.id or not {'title', 'version'} <= profile.keys():
        raise ValueError('task entry needs a matching profile with title and version')
    names = set(_ENTRIES) | {source.name for task in _TASKS.values() for source in task.sources}
    for source in entry.sources:
        validate_source_entry(source)
        if source.name in names:
            raise ValueError(f'source already registered: {source.name}')
        names.add(source.name)
    _TASKS[entry.id] = entry


def _register_task_extension(entry):
    if not isinstance(entry, TaskPlugin):
        raise ValueError('task entry must be a TaskPlugin')
    entry.resolve()
    register_task(entry)


register_group('dexstation.tasks', _register_task_extension)


def task_plugins():
    global _task_builtins
    if not _task_builtins:
        _task_builtins = True
        # Only metadata loads here, never a task implementation or asset package.
        manifest = importlib.import_module('dexstation.task_plugins')
        for entry in manifest.builtin_tasks():
            register_task(entry)
    discover_plugins('dexstation.tasks')
    return dict(_TASKS)


def task_plugin(task_id, *, required=False):
    entry = task_plugins().get(task_id)
    errors = [e for e in _errors if e['group'] == 'dexstation.tasks' and e['name'] == task_id]
    if errors:
        raise ValueError(f"Task {task_id} plugin load failed: {errors[-1]['reason']}")
    if required and entry is None:
        raise ValueError(f'Task {task_id} has no registered plugin')
    return entry


def activate_task(task_id):
    entry = task_plugin(task_id)
    if entry and entry.activate and task_id not in _active_tasks:
        try:
            entry.activate()
        except Exception as exc:
            _failure('dexstation.tasks', task_id, exc)
            raise ValueError(f'Task {task_id} activation failed: {exc}') from exc
        _active_tasks.add(task_id)
    return entry


def task_sources():
    return tuple(source for entry in task_plugins().values() for source in entry.sources)


def activate_task_type(name, category):
    """Resolve a declared task type in a fresh renderer/export/replay process."""
    import json
    key = 'type' if category == 'objects' else 'kind'
    for entry in task_plugins().values():
        if entry.activate is None:
            continue
        profile = json.loads((Path(entry.data_directory) / 'task.json').read_text())
        if any(item[key] == name for item in profile[category]):
            activate_task(entry.id)
            return


# Core pane defaults live here; task-owned panes live exclusively in task.json.
CORE_PANES = (
    dict(id='table', title='Table top', weight=60, camera='table'),
    dict(id='arm_takeover', title='Arm takeover', weight=90,
         visible_when={'source_capabilities.teleop_calibration': True}),
    dict(id='status', title='Status', weight=100),
)


def ordered_panes(panes, overrides=None):
    """Hugo order: weighted first, ascending weight, then ID for every tie."""
    result = [dict(p, **({'weight': overrides[p['id']]} if p['id'] in (overrides or {}) else {}))
              for p in panes]
    return sorted(result, key=lambda p: ('weight' not in p, p.get('weight', 0), p['id']))


def task_panes(task, overrides=None):
    return ordered_panes([*CORE_PANES, *task.get('panes', [])], overrides)


def known_pane_ids():
    """Read task manifests without activating their Python or asset packages."""
    import json
    ids = {p['id'] for p in CORE_PANES}
    for entry in task_plugins().values():
        task = json.loads((Path(entry.data_directory) / 'task.json').read_text())
        ids.update(p['id'] for p in task.get('panes', []))
    # Declarative tasks need no Python entry.
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'task_plugins').glob('*/data/task.json'):
        ids.update(p['id'] for p in json.loads(path.read_text()).get('panes', []))
    return ids
