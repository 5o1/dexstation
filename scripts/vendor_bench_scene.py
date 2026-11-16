#!/usr/bin/env python3
"""Refresh the vendor baseline snapshot; maintained profiles are never regenerated."""
import argparse
import json
from pathlib import Path

from dexstation.config.defaults import DEFAULT_WORKBENCH
from dexstation.examples.ar5_g20.compose import extract_baseline

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'src/dexstation/examples/ar5_g20' / DEFAULT_WORKBENCH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    extract = commands.add_parser('extract')
    extract.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    sources = BENCH / 'sources'
    manifest = json.loads((sources / 'external_resources.json').read_text())
    bench = next(resource for resource in manifest['resources'] if resource['id'] == 'bench')
    baseline = extract_baseline(args.workspace / bench['destination'], bench['revision'])
    target = sources / 'baseline.json'
    target.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + '\n')
    print(target)


if __name__ == '__main__':
    main()
