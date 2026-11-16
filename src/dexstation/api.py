"""Supported Python entry types and helpers for extension authors."""
from dexstation.backends.channel_types import ChannelType, register_channel_type
from dexstation.backends.object_types import ObjectType, register_object_type
from dexstation.backends.registry import BackendEntry, register_backend
from dexstation.contracts.services import Services
from dexstation.inputs.registry import SourceCapabilities, SourceContext, SourceEntry, register_source
from dexstation.runtime.health import InputHealth
from dexstation.runtime.plugins import TaskPlugin, register_task, task_plugin, task_plugins

__all__ = [
    'BackendEntry', 'ChannelType', 'InputHealth', 'ObjectType', 'Services',
    'SourceCapabilities', 'SourceContext', 'SourceEntry', 'TaskPlugin',
    'register_backend', 'register_channel_type', 'register_object_type',
    'register_source', 'register_task', 'task_plugin', 'task_plugins',
]
