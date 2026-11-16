#!/usr/bin/env python3
"""Export a WP2-composed workbench and task without starting an online session."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


from dexstation.config.defaults import DEFAULT_TASK


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbench', required=True)
    parser.add_argument('--task', default=DEFAULT_TASK)
    parser.add_argument('--format', choices=('sapien', 'mjcf'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-hz', type=float, required=True,
                        help='collection control rate, distinct from the physics timestep')
    parser.add_argument('--resource-root', type=Path)
    parser.add_argument('--task-collision-cache', type=Path,
                        help='reuse measured CoACD parts with identical source SHA256 and fixed parameters')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    from dexstation.profiles.compose import compose, load_task, load_workbench
    from dexstation.profiles.export import export_scene
    scene = compose(load_workbench(args.workbench), load_task(args.task), 'sapien', seed=args.seed)
    export_scene(scene, args.output, format=args.format, control_hz=args.control_hz,
                 resource_root=args.resource_root, task_collision_cache=args.task_collision_cache)
    print(args.output / 'bundle.json')


if __name__ == '__main__':
    main()
