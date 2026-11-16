"""Object type contracts and operations shared by composition and engines.

Handlers may be callables supplied by an extension or local operation names
implemented by the consuming engine. Importing entries never loads an asset.
"""
from dataclasses import dataclass, field
from typing import Callable

from dexstation.runtime.plugins import register_group, discover_plugins, activate_task_type


@dataclass(frozen=True)
class ObjectType:
    name: str
    schema: dict = field(default_factory=dict)
    static: bool = False
    mass: bool = False
    legacy_gamepad: bool = False
    environment_table_only: bool = False
    operations: dict[str, Callable | str] = field(default_factory=dict)

    def invoke(self, operation, namespace, *args, **kwargs):
        handler = self.operations.get(operation)
        if handler is None:
            raise ValueError(f'Object type {self.name} does not support {operation}')
        return (namespace[handler] if isinstance(handler, str) else handler)(*args, **kwargs)


OBJECT_TYPES = {}


def register_object_type(entry):
    if not isinstance(entry, ObjectType):
        raise TypeError('object type entry must be an ObjectType')
    if not isinstance(entry.name, str) or not entry.name.strip():
        raise ValueError('object type name must be a nonempty string')
    if not isinstance(entry.schema, dict) or not isinstance(entry.operations, dict):
        raise TypeError('object type schema and operations must be dictionaries')
    for operation, handler in entry.operations.items():
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError('object operation name must be a nonempty string')
        if not callable(handler) and not (isinstance(handler, str) and handler.strip()):
            raise TypeError(f'object operation {operation} must be callable or a local handler name')
    if entry.name in OBJECT_TYPES:
        raise ValueError(f'Object type already registered: {entry.name}')
    OBJECT_TYPES[entry.name] = entry


def object_types():
    discover_plugins('dexstation.object_types')
    return dict(OBJECT_TYPES)


def object_type(name):
    if name not in object_types():
        activate_task_type(name, 'objects')
    try:
        return object_types()[name]
    except KeyError:
        raise ValueError(f'unsupported task object type: {name}') from None


register_group('dexstation.object_types', register_object_type)

_ARTICULATION_SCHEMA = {'type': {'const': 'articulation'},
 'geometry': {'type': 'object',
              'properties': {'urdf': {'type': 'string', 'minLength': 1},
                             'fixed_root': {'type': 'boolean'},
                             'collision_mesh': {'enum': ['sdf', 'coacd']}},
              'required': ['urdf', 'fixed_root'],
              'additionalProperties': False}}
_PRIMITIVE_SCHEMA = {'type': {'enum': ['rigid', 'static']}, 'initial_joint_positions': False}

for _name in ('rigid', 'static'):
    register_object_type(ObjectType(_name, schema=dict(_PRIMITIVE_SCHEMA,
        type={'const': _name}), static=_name == 'static', mass=_name == 'rigid', operations={
        'compose': '_project_primitive', 'sapien_load': '_load_primitive',
        'mujoco_build': '_add_mujoco_primitive', 'sapien_export_load': '_load_export_primitive',
        'inventory': '_inventory_primitive'}))
register_object_type(ObjectType('articulation', schema=_ARTICULATION_SCHEMA, operations={
    'compose': '_project_none', 'sapien_load': '_load_articulation',
    'mujoco_build': '_add_mujoco_articulation', 'sapien_export_load': '_load_export_articulation',
    'inventory': '_inventory_articulation'}))

# Built-ins use the same entry contract as external packages.
