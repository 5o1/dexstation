"""Backend entries own construction, availability and dependency constraints."""
from dataclasses import dataclass, field
from collections.abc import Mapping
import importlib
from importlib import metadata
from typing import Callable

from dexstation.runtime.plugins import discover_plugins, lazy_factory, register_group


@dataclass(frozen=True)
class BackendEntry:
    name: str
    factory: Callable
    versions: Mapping[str, str] = field(default_factory=dict)
    imports: tuple[str, ...] = ()
    online: bool = True
    version: str = ''
    check: Callable | None = None
    supports_workers: bool = False
    timestep: Callable = lambda backend: float(backend.scene.get_timestep())
    motion_adapter: str | None = None

    def self_check(self):
        result = dict(name=self.name, available=False, reason=None, check='import+version')
        try:
            if self.check is not None:
                self.check()
            for module in self.imports:
                importlib.import_module(module)
            for package, required in self.versions.items():
                actual = metadata.version(package)
                if actual != required:
                    raise ValueError(f'{package} version {actual}; required {required}')
            result['available'] = True
        except Exception as exc:
            result['reason'] = ' '.join(str(exc).split())
        return result


SAPIEN_VERSION = '3.0.3+concon.physx5.6.1.a3'
_ENTRIES = {}


def register_backend(entry):
    if not isinstance(entry, BackendEntry) or not isinstance(entry.name, str) or not entry.name.strip() or not callable(entry.factory):
        raise ValueError('backend entry needs a name and callable factory')
    if not isinstance(entry.versions, Mapping):
        raise ValueError('backend versions must map distribution names to versions')
    if entry.check is not None and not callable(entry.check):
        raise ValueError('backend check must be callable')
    if not callable(entry.timestep):
        raise ValueError('backend timestep reader must be callable')
    if entry.name in _ENTRIES:
        raise ValueError(f'backend already registered: {entry.name}')
    _ENTRIES[entry.name] = entry


def backend_entries(*, online=False):
    discover_plugins('dexstation.backends')
    return tuple(e for e in _ENTRIES.values() if not online or e.online)


def backend_names(*, online=False):
    return tuple(e.name for e in backend_entries(online=online))


def get_backend(name, *, online=False):
    discover_plugins('dexstation.backends')
    entry = _ENTRIES.get(name)
    if entry is None or (online and not entry.online):
        raise ValueError(f'unknown collection backend: {name}')
    return entry


def backend_availability(name='sapien'):
    try:
        return get_backend(name, online=True).self_check()
    except ValueError as exc:
        return dict(name=name, available=False, reason=str(exc), check='import+version')


def create_registered_backend(name, scene, **kwargs):
    return get_backend(name, online=True).factory(scene, **kwargs)


register_backend(BackendEntry('sapien', lazy_factory('dexstation.backends.sapien_physics', 'CollectionSapienBackend'),
    versions={'sapien': SAPIEN_VERSION, 'xbox-gamepad-physics': '0.4.3'},
    version=SAPIEN_VERSION, imports=('sapien',), motion_adapter='sapien'))
register_backend(BackendEntry('mujoco', lazy_factory('dexstation.backends.mujoco_export', 'CollectionMujocoBackend'),
    online=False, supports_workers=True, timestep=lambda backend: float(backend.model.opt.timestep)))
register_group('dexstation.backends', register_backend)
