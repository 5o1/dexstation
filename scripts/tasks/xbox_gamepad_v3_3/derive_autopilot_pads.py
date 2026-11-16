#!/usr/bin/env python3
"""Derive each G20 finger's contact pad (point and outward normal, distal-link frame).

The pad is the collision-mesh surface patch that faces ``cos(a) * palmar + sin(a) * distal``
(palmar = +x, the direction a fingertip moves under positive flexion about the distal joint's
+y axis; distal = +z, along the finger). The point is the extreme vertex along that direction
and the normal is the area-weighted face normal within ``--patch-mm`` of it.

    python scripts/tasks/xbox_gamepad_v3_3/derive_autopilot_pads.py HAND_URDF_DIR OUTPUT_JSON
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def load_stl(path):
    data = Path(path).read_bytes()
    count = struct.unpack('<I', data[80:84])[0]
    if 84 + 50 * count == len(data):
        record = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
        return np.frombuffer(data[84:84 + 50 * count], dtype=record)['v'].astype(float)
    values = [list(map(float, line.split()[1:4])) for line in data.decode().splitlines()
              if line.strip().startswith('vertex')]
    return np.asarray(values).reshape(-1, 3, 3)


def pad(triangles, angle_deg, patch):
    a = math.radians(angle_deg)
    direction = np.array([math.cos(a), 0., math.sin(a)])
    vertices = triangles.reshape(-1, 3)
    point = vertices[np.argmax(vertices @ direction)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(normals, axis=1)
    keep = area > 1e-14
    centres, normals, area = triangles[keep].mean(1), normals[keep] / area[keep, None], area[keep]
    flip = np.einsum('ij,ij->i', normals, centres - vertices.mean(0)) < 0
    normals[flip] *= -1
    near = np.linalg.norm(centres - point, axis=1) < patch
    normal = (normals[near] * area[near, None]).sum(0)
    return point, normal / np.linalg.norm(normal)


def surface(triangles, spacing=.0025):
    """Vertices on the palmar/tip side of the distal mesh, voxel-thinned: the candidates for the first contact."""
    vertices = triangles.reshape(-1, 3)
    vertices = vertices[vertices @ np.array([1., 0., 1.]) / math.sqrt(2) > (vertices @ np.array([1., 0., 1.]) / math.sqrt(2)).max() - .010]
    key = np.floor(vertices / spacing).astype(np.int64)
    _, index = np.unique(key, axis=0, return_index=True)
    return vertices[np.sort(index)]


def derive(urdf_dir, angle_deg=45., patch_mm=3.):
    urdf_dir = Path(urdf_dir)
    result, sources = {}, {}
    for side in ('left', 'right'):
        urdf = urdf_dir / f'g20_{side}.urdf'
        sources[urdf.name] = hashlib.sha256(urdf.read_bytes()).hexdigest()
        root = ET.parse(urdf).getroot()
        result[side] = {}
        for finger in FINGERS:
            collision = root.find(f"link[@name='{finger}_distal']/collision")
            origin = collision.find('origin')
            if origin is not None and any(float(v) for v in origin.get('xyz', '0 0 0').split() + origin.get('rpy', '0 0 0').split()):
                raise ValueError('distal collision origins must be identity')
            mesh = (urdf.parent / collision.find('geometry/mesh').get('filename')).resolve()
            sources[f'{side}/{mesh.name}'] = hashlib.sha256(mesh.read_bytes()).hexdigest()
            triangles = load_stl(mesh)
            point, normal = pad(triangles, angle_deg, patch_mm / 1000)
            result[side][finger] = dict(point=point.tolist(), normal=normal.tolist(),
                                        surface=np.round(surface(triangles), 5).tolist())
    return dict(schema='dexhand-autopilot-finger-pads-v1', frame='<finger>_distal link', units='m',
                surface='collision-mesh vertices within 10 mm of the fingertip along palmar+distal, 2.5 mm voxels',
                rule=dict(direction_angle_deg_from_palmar_toward_distal=angle_deg, patch_mm=patch_mm),
                source=dict(package='g20_pbr_description v0.1.1 collision meshes', sha256=sources), pads=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('urdf_dir')
    parser.add_argument('output')
    parser.add_argument('--angle-deg', type=float, default=45.)
    parser.add_argument('--patch-mm', type=float, default=3.)
    args = parser.parse_args()
    result = derive(args.urdf_dir, args.angle_deg, args.patch_mm)
    head = json.dumps({k: v for k, v in result.items() if k != 'pads'}, indent=1)[:-2]
    Path(args.output).write_text(head + ',\n "pads": ' + json.dumps(result['pads'], separators=(',', ':')) + '\n}\n')


if __name__ == '__main__':
    main()
