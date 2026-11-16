#!/usr/bin/env python3
"""Fetch the pinned linker-sim AR5_08 left/right arms and derive the scene arm URDFs.

The upstream left and right variants share joints, limits and masses; they differ in
mirrored meshes and centres of mass. Each derived URDF keeps the upstream geometry,
kinematics and inertia and only renames links and joints with the scene side
(AR5_5_08L_... -> AR5_left_5_08L_...) and points meshes at the downloaded copy.

  python scripts/derive_ar5_08_arms.py --workspace WORKSPACE [--fetch]
"""
import argparse
import hashlib
import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ID = 'ar5_08'
SIDES = {'left': 'left', 'right': 'right'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource(manifest):
    return next(r for r in json.loads(Path(manifest).read_text())['resources'] if r['id'] == RESOURCE_ID)


def fetch(entry, destination):
    for item in entry['files']:
        target = destination / item['path']
        if target.exists() and sha256(target) == item['sha256']:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(item['url'], timeout=120) as response:
            data = response.read()
        target.write_bytes(data)


def verify(entry, destination):
    bad = [item['path'] for item in entry['files']
           if not (destination / item['path']).exists() or sha256(destination / item['path']) != item['sha256']]
    if bad:
        raise SystemExit(f'missing or mismatched files under {destination}: {bad}')


def derive(source_urdf, side, mesh_prefix):
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    rename = lambda name: name.replace('AR5_', f'AR5_{side}_', 1)
    for element in root.iter():
        if element.tag in ('link', 'joint') and element.get('name', '').startswith('AR5_'):
            element.set('name', rename(element.get('name')))
        if element.tag in ('parent', 'child') and element.get('link', '').startswith('AR5_'):
            element.set('link', rename(element.get('link')))
        if element.tag == 'mesh':
            element.set('filename', f"{mesh_prefix}/{element.get('filename')}")
    return tree


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, required=True, help='directory that workspace:// resolves to')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'references/external_resources.json')
    parser.add_argument('--fetch', action='store_true', help='download missing or mismatched files first')
    args = parser.parse_args()
    entry = resource(args.manifest)
    source = args.workspace / entry['destination']
    if args.fetch:
        fetch(entry, source)
    verify(entry, source)
    output = args.workspace / entry['derived']
    output.mkdir(parents=True, exist_ok=True)
    record = dict(resource=RESOURCE_ID, repository=entry['repository'], revision=entry['revision'],
                  rule='links and joints AR5_* renamed AR5_<side>_*; geometry, kinematics, limits and inertia unchanged',
                  sides={})
    for side, variant in SIDES.items():
        source_urdf = source / variant / 'arm.urdf'
        mesh_prefix = Path(os.path.relpath(source / variant, output)).as_posix()
        target = output / f'{side}_arm.urdf'
        derive(source_urdf, side, mesh_prefix).write(target, encoding='utf-8', xml_declaration=True)
        record['sides'][side] = dict(variant=variant, source=f'{variant}/arm.urdf', source_sha256=sha256(source_urdf),
                                     urdf=target.name, sha256=sha256(target))
    (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
