"""Input source factories and the capabilities consumed by orchestration."""
from dataclasses import dataclass
from typing import Callable

from dexstation.runtime.plugins import discover_plugins, lazy_factory, register_group, task_sources


@dataclass(frozen=True)
class SourceCapabilities:
    clock: str = 'wall'
    hardware_lease: bool = False
    teleop_calibration: bool = False
    motion_profile: str = 'legacy'
    real_robot_mirror: bool = False
    task_declared: bool = False
    latest_input: bool = False
    replay: bool = False
    ros_topics: bool = False
    input_chain: bool = False
    health_gate: bool = False
    glove_recording: bool = False
    flattened_observation: bool = False
    task_set_option: str | None = None


@dataclass(frozen=True)
class SourceEntry:
    name: str
    factory: Callable
    capabilities: SourceCapabilities = SourceCapabilities()
    validate: Callable | None = None
    api_order: int = 100


@dataclass
class SourceContext:
    options: object
    spec: object
    mapper: object
    scene: object
    profile: object
    replay_episode: object = None
    replay_meta: object = None
    initial: object = None
    geometry_provider: object = None


_ENTRIES = {}


def validate_source_entry(entry):
    if not isinstance(entry, SourceEntry) or not isinstance(entry.name, str) or not entry.name.strip() or not callable(entry.factory):
        raise ValueError('source entry needs a name and callable factory')
    if not isinstance(entry.capabilities, SourceCapabilities) or entry.capabilities.clock not in ('wall', 'simulation'):
        raise ValueError('source capabilities need a wall or simulation clock')
    if entry.validate is not None and not callable(entry.validate):
        raise ValueError('source validator must be callable')
    if type(entry.api_order) is not int:
        raise ValueError('source API order must be an integer')


def register_source(entry):
    validate_source_entry(entry)
    if entry.name in _ENTRIES:
        raise ValueError(f'source already registered: {entry.name}')
    _ENTRIES[entry.name] = entry


def register_task_sources():
    for entry in task_sources():
        if entry.name not in _ENTRIES:
            register_source(entry)


def source_entries(*, api=False):
    register_task_sources()
    discover_plugins('dexstation.sources')
    entries = tuple(_ENTRIES.values())
    return tuple(sorted(entries, key=lambda e: e.api_order)) if api else entries


def source_names(*, api=False):
    return tuple(e.name for e in source_entries(api=api))


def get_source(name):
    try:
        return next(entry for entry in source_entries() if entry.name == name)
    except StopIteration:
        raise ValueError(f'unknown collection source: {name}') from None


def source_capabilities(name):
    return get_source(name).capabilities


def validate_source(options):
    entry = get_source(options.source)
    if entry.validate:
        entry.validate(options)


def _require_replay(options):
    if options.replay is None:
        raise ValueError('--replay is required')


register_source(SourceEntry('ros2', lazy_factory('dexstation.inputs.factories', 'ros2'),
    SourceCapabilities(hardware_lease=True, teleop_calibration=True, motion_profile='station',
        real_robot_mirror=True, latest_input=True, ros_topics=True, input_chain=True, health_gate=True, glove_recording=True), api_order=0))
register_source(SourceEntry('udp', lazy_factory('dexstation.inputs.factories', 'udp'),
    SourceCapabilities(teleop_calibration=True, motion_profile='station', latest_input=True), api_order=2))
register_source(SourceEntry('replay', lazy_factory('dexstation.inputs.factories', 'replay'),
    SourceCapabilities(clock='simulation', replay=True), validate=_require_replay, api_order=1))
register_group('dexstation.sources', register_source)
