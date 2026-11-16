"""Offline evaluation hook declared by task.json."""
from dexstation.task_plugins.xbox_gamepad_v3_3.autopilot_eval import evaluate_episode

API_VERSION = 1


def evaluate(episode, *, tasks_path=None):
    return evaluate_episode(episode, tasks_path=tasks_path)
