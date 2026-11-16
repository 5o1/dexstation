"""Built-in task manifests; implementations stay unloaded until selected."""
from pathlib import Path


def builtin_tasks():
    from dexstation.runtime.plugins import TaskPlugin, lazy_factory
    from dexstation.inputs.registry import SourceCapabilities, SourceEntry
    package = __name__ + '.xbox_gamepad_v3_3'
    return (TaskPlugin('xbox_gamepad_v3_3', package,
        Path(__file__).parent / 'xbox_gamepad_v3_3/data',
        Path(__file__).parent / 'xbox_gamepad_v3_3/web',
        sources=(SourceEntry('ik_autopilot', lazy_factory(package + '.autopilot', 'make_source'),
            SourceCapabilities(clock='simulation', motion_profile='station', task_declared=True,
                flattened_observation=True, task_set_option='autopilot_tasks'),
            validate=lazy_factory(package + '.autopilot', 'validate_options'), api_order=3),),
        activate=lazy_factory(package + '.gamepad_types', 'register_types')),)
