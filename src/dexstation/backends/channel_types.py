"""Observation channel metadata and engine reader registry."""
from dataclasses import dataclass
from typing import Callable

from dexstation.runtime.plugins import register_group, discover_plugins, activate_task_type


@dataclass(frozen=True)
class ChannelType:
    name: str
    metadata: tuple | Callable | None = None
    sapien_observer: Callable | str | None = None

    def describe(self, channel):
        defaults = self.metadata(channel) if callable(self.metadata) else self.metadata
        if defaults is None:
            if not {'shape', 'dtype', 'unit'} <= channel.keys():
                raise ValueError(f'Channel {channel["name"]} requires shape, dtype and unit')
            defaults = tuple(channel['shape']), channel['dtype'], channel['unit']
        shape, dtype, unit = defaults
        return tuple(channel.get('shape', shape)), channel.get('dtype', dtype), channel.get('unit', unit)

    def observe_sapien(self, backend, obj, channel, raw, namespace):
        handler = self.sapien_observer
        if handler is None:
            raise ValueError(f'unsupported task channel: {self.name}')
        return (namespace[handler] if isinstance(handler, str) else handler)(backend, obj, channel, raw)


CHANNEL_TYPES = {}


def register_channel_type(entry):
    if not isinstance(entry, ChannelType):
        raise TypeError('channel type entry must be a ChannelType')
    if not isinstance(entry.name, str) or not entry.name.strip():
        raise ValueError('channel type name must be a nonempty string')
    if entry.metadata is not None and not callable(entry.metadata):
        if not isinstance(entry.metadata, tuple) or len(entry.metadata) != 3:
            raise TypeError('channel metadata must be a (shape, dtype, unit) tuple or callable')
    handler = entry.sapien_observer
    if handler is not None and not callable(handler) and not (isinstance(handler, str) and handler.strip()):
        raise TypeError('channel observer must be callable or a local handler name')
    if entry.name in CHANNEL_TYPES:
        raise ValueError(f'Channel type already registered: {entry.name}')
    CHANNEL_TYPES[entry.name] = entry


def channel_types():
    discover_plugins('dexstation.channel_types')
    return dict(CHANNEL_TYPES)


def channel_type(name):
    if name not in channel_types():
        activate_task_type(name, 'observations')
    try:
        return channel_types()[name]
    except KeyError:
        raise ValueError(f'unsupported task channel: {name}') from None


def _contact_metadata(channel, force=False):
    if 'with' not in channel:
        raise ValueError(f'Channel {channel["name"]} requires with groups')
    size = len(channel['with'])
    return ((size, 3), 'float64', 'N') if force else ((size,), 'bool', '')


register_group('dexstation.channel_types', register_channel_type)
register_channel_type(ChannelType('object_pose', ((7,), 'float64', 'm,wxyz'), '_observe_object_pose'))
register_channel_type(ChannelType('object_velocity', ((6,), 'float64', 'm/s,rad/s'), '_observe_object_velocity'))
register_channel_type(ChannelType('articulation_q', sapien_observer='_observe_articulation_q'))
register_channel_type(ChannelType('contact', _contact_metadata, '_observe_contact'))
register_channel_type(ChannelType('contact_force', lambda c: _contact_metadata(c, force=True)))
register_channel_type(ChannelType('plugin'))
