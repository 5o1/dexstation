"""Registered IK source adapter declared by task.json."""
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_source import IkAutopilotSource

API_VERSION = 1


def validate_options(options):
    if options.autopilot_tasks is None:
        raise ValueError('the IK autopilot needs a task set')


def make_source(context):
    return IkAutopilotSource(context.spec, context.options.autopilot_tasks)
