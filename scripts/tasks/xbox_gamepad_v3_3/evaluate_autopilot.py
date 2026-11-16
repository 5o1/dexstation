#!/usr/bin/env python3
"""Score an IK autopilot episode from its recorded gamepad signals.

    python scripts/tasks/xbox_gamepad_v3_3/evaluate_autopilot.py EPISODE_DIR [--tasks TASKS_JSON] [--steps]
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from dexstation.runtime.plugins import task_plugin  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--steps', action='store_true', help='also print every step result and false trigger')
    args = parser.parse_args()
    summary, results, false_events = task_plugin('xbox_gamepad_v3_3').evaluate(args.episode, tasks_path=args.tasks)
    output = dict(summary=summary)
    if args.steps:
        output.update(steps=results, false_triggers=false_events)
    print(json.dumps(output, indent=1))


if __name__ == '__main__':
    main()
