#!/usr/bin/env python3
"""Collect per-group play-pose search results into the runtime asset (task_plugins/xbox_gamepad_v3_3/assets/play_poses.json).

    python scripts/tasks/xbox_gamepad_v3_3/assemble_autopilot_poses.py OUTPUT NAME=SIDE:RESULT_JSON [NAME=SIDE:RESULT_JSON ...]

Each RESULT_JSON comes from ``solve_autopilot_poses.py --controls ...`` (one control group, one side).
Only the fields the runtime uses are kept, plus the offline checks that justify them.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output')
    parser.add_argument('modes', nargs='+', help='NAME=SIDE:RESULT_JSON[#CANDIDATE] (candidate 0 = the search best)')
    args = parser.parse_args()
    out = None
    for item in args.modes:
        name, rest = item.split('=', 1)
        side, path = rest.split(':', 1)
        path, _, pick = path.partition('#')
        result = json.loads(Path(path).read_text())
        best = result['sides'][side]
        if pick and int(pick):
            best = best['candidates'][int(pick) - 1]
        if out is None:
            out = dict(schema='dexhand-autopilot-play-poses-v2', scene=result['scene'],
                       gamepad_root_pose=result['gamepad_root_pose'], assignment={}, sides={'left': {'modes': []},
                                                                                       'right': {'modes': []}})
        if result['gamepad_root_pose'] != out['gamepad_root_pose']:
            raise ValueError(f'{path} was solved for a different gamepad pose')
        if not best.get('approach'):
            raise ValueError(f'{path} has no checked approach line')
        controls = list(result['assignment'][side])
        out['assignment'].setdefault(side, {}).update(result['assignment'][side])
        states = best['states']
        out['sides'][side]['modes'].append(dict(
            name=name, controls=controls, arm_q=best['arm_q'], hand_in_gamepad=best['hand_in_gamepad'],
            approach={k: best['approach'][k] for k in ('direction', 'lift_m', 'away_fingers', 'path_clearance_mm',
                                                       'home_clearance_mm')},
            checks=dict(states_covered=sum(s['ok'] for s in states.values()), states=len(states),
                        failed=sorted(k for k, s in states.items() if not s['ok']),
                        arm_margin_deg=round(best['arm_margin_deg'], 2),
                        rest_clearance_mm=min(best['rest_clearance_mm'].values()),
                        dense_rest_clearance_mm=best.get('dense_rest_clearance_mm'),
                        source=f'{Path(path).name}#{pick or 0} (scripts/tasks/xbox_gamepad_v3_3/solve_autopilot_poses.py --controls {",".join(controls)})')))
    Path(args.output).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
