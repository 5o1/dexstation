"""Gamepad-owned registry entries; the physics asset loads only on use."""
from dexstation.contracts.versions import GAMEPAD_CHANNEL_KIND


def register_object_types():
    from dexstation.backends.object_types import ObjectType, register_object_type
    from dexstation.runtime.plugins import lazy_factory
    operation = lambda name: lazy_factory(__package__ + ".operations", name)
    schema = {'type': {'const': 'xbox_gamepad_physics'},
     'initial_joint_positions': False,
     'geometry': {'type': 'object',
                  'properties': {'release': {'type': 'string', 'minLength': 1},
                                 'profile': {'type': 'string', 'minLength': 1},
                                 'collision_mesh': {'enum': ['sdf', 'coacd']},
                                 'fixed_root': {'type': 'boolean'}},
                  'required': ['release', 'profile', 'collision_mesh', 'fixed_root'],
                  'additionalProperties': False}}
    register_object_type(ObjectType('xbox_gamepad_physics', schema=schema,
        legacy_gamepad=True, environment_table_only=True, operations={
        'compose': operation('_project_gamepad'), 'sapien_load': operation('_load_gamepad'),
        'mujoco_build': operation('_add_mujoco_gamepad'), 'sapien_export_load': operation('_load_export_gamepad'),
        'inventory': operation('_inventory_gamepad'), 'export_resources': operation('_export_gamepad_resources')}))


def register_channel_types():
    from dexstation.backends.channel_types import ChannelType, register_channel_type
    register_channel_type(ChannelType(GAMEPAD_CHANNEL_KIND,
        ((32,), 'float32', 'normalized'), _observe_gamepad))


def _observe_gamepad(backend, obj, channel, raw):
    import numpy as np
    state = obj.gamepad.gamepad_state()
    return dict(values=np.asarray(state.values).tolist(),
        available_mask=np.asarray(state.available_mask, bool).tolist(),
        control_joint_positions=obj.gamepad.joint_positions())


def register_types():
    register_object_types()
    register_channel_types()
