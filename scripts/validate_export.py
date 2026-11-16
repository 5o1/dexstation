#!/usr/bin/env python3
"""CPU-only scene inventory, independent FK, home contact and limit checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--position-atol-m', type=float, required=True)
    parser.add_argument('--rotation-atol-rad', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from dexstation.profiles.export import validate_export
    result = validate_export(args.bundle, position_atol_m=args.position_atol_m,
        rotation_atol_rad=args.rotation_atol_rad, work_dir=args.output.parent / (args.output.stem + '_work'))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
