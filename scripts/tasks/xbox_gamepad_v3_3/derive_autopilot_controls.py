#!/usr/bin/env python3
"""Derive the IK autopilot's gamepad control geometry from xbox_gamepad_physics resources.

Reads the RoboTwin ``full_controls`` resource (``mobility.urdf`` joint origins/axes/limits
and the exact SDF collision meshes the collection scene loads) and writes contact points,
press directions and travel per control, in the gamepad body frame. Geometry rules follow
the 2026-11-11 intent-IK feasibility study (``scripts/geom.py``: top surface along the
press axis; D-pad pressed at 65 % of its arm; stick cap surface on the axis; trigger face
whose normal opposes the hinge velocity).

    python scripts/tasks/xbox_gamepad_v3_3/derive_autopilot_controls.py RESOURCES_DIR OUTPUT_JSON
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

SDF_FILES = {'a': ['a_0'], 'b': ['b_0'], 'x': ['x_0'], 'y': ['y_0'], 'dpad': ['dpad_0'],
             'left_shoulder': ['left_shoulder_0'], 'right_shoulder': ['right_shoulder_0'],
             'left_stick': ['left_stick_0', 'left_stick_1'], 'right_stick': ['right_stick_0', 'right_stick_1'],
             'left_trigger': ['left_trigger_0'], 'right_trigger': ['right_trigger_0'],
             'controller_body': ['shell_0', 'shell_front_plate_0']}


def load_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith('v '):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith('f '):
            faces.append([int(t.split('/')[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(vertices)[np.asarray(faces)]


def joints(urdf):
    result = {}
    for joint in ET.parse(urdf).getroot().findall('joint'):
        origin, axis, limit = joint.find('origin'), joint.find('axis'), joint.find('limit')
        result[joint.get('name')] = dict(
            parent=joint.find('parent').get('link'), child=joint.find('child').get('link'),
            xyz=np.array([float(v) for v in origin.get('xyz').split()]),
            axis=np.array([float(v) for v in axis.get('xyz').split()]),
            limit=(float(limit.get('lower')), float(limit.get('upper'))))
    return result


def link_offsets(js):
    offsets, changed = {'controller_body': np.zeros(3)}, True
    while changed:
        changed = False
        for j in js.values():
            if j['parent'] in offsets and j['child'] not in offsets:
                offsets[j['child']] = offsets[j['parent']] + j['xyz']
                changed = True
    return offsets


def derive(resources):
    root = Path(resources)
    urdf = root / 'mobility.urdf'
    js = joints(urdf)
    offset = link_offsets(js)
    files = {}

    def tri(link):
        parts = []
        for name in SDF_FILES[link]:
            path = root / f'collision/sdf/{name}.obj'
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            parts.append(load_obj(path))
        return np.concatenate(parts) + offset[link]

    def top(link, outward, band=4e-4):
        v = tri(link).reshape(-1, 3)
        h = v @ outward
        c = v[h > h.max() - band].mean(0)
        return c + (h.max() - c @ outward) * outward

    controls = {}
    for name, link, joint, channel in (('a', 'a', 'a_press', 'south'), ('b', 'b', 'b_press', 'east'),
                                       ('x', 'x', 'x_press', 'west'), ('y', 'y', 'y_press', 'north'),
                                       ('lb', 'left_shoulder', 'left_shoulder_press', 'left_shoulder'),
                                       ('rb', 'right_shoulder', 'right_shoulder_press', 'right_shoulder')):
        axis = js[joint]['axis'] / np.linalg.norm(js[joint]['axis'])
        controls[name] = dict(kind='button', link=link, joint=joint, channel=channel,
                              point=top(link, -axis, 8e-4 if 'shoulder' in link else 4e-4).tolist(),
                              push=axis.tolist(), travel_m=js[joint]['limit'][1])
    v = tri('dpad').reshape(-1, 3)
    pivot = offset['dpad']
    for name, direction in (('dpad_up', (0, 1, 0)), ('dpad_down', (0, -1, 0)),
                            ('dpad_left', (-1, 0, 0)), ('dpad_right', (1, 0, 0))):
        d = np.array(direction, float)
        r = (v - pivot) @ d
        r0 = .65 * r.max()
        band = v[np.abs(r - r0) < .0015]
        band = band[np.abs((band - pivot) @ np.cross([0, 0, 1.], d)) < .002]
        point = pivot + r0 * d
        point[2] = band[:, 2].max()
        controls[name] = dict(kind='dpad', link='dpad', joint='dpad_vertical' if d[1] else 'dpad_horizontal',
                              channel=name, point=point.tolist(), push=[0., 0., -1.], pivot=pivot.tolist(),
                              tilt_limit_rad=js['dpad_vertical']['limit'][1])
    for prefix in ('left', 'right'):
        link = prefix + '_stick'
        pivot = offset[link]
        vv = tri(link).reshape(-1, 3)
        radial = np.linalg.norm((vv - pivot)[:, :2], axis=1)
        near = vv[radial < .0025]
        cap = np.array([pivot[0], pivot[1], near[:, 2].max()])
        rim = vv[vv[:, 2] > cap[2] - .004]
        # The cap's outer side is a vertical cylinder; tilting pushes belong there (horizontal push, no click force).
        outer = np.linalg.norm((rim - pivot)[:, :2], axis=1)
        side = rim[outer > outer.max() - 1e-4]
        controls[prefix[0] + 's'] = dict(
            kind='stick', link=link, joints=[prefix + '_stick_x', prefix + '_stick_y'],
            channels=[prefix + '_x', prefix + '_y'], pivot=pivot.tolist(), point=cap.tolist(), push=[0., 0., -1.],
            cap_rim_top_z=float(vv[:, 2].max()), cap_radius_m=float(outer.max()),
            cap_side_z=[float(side[:, 2].min()), float(side[:, 2].max())],
            tilt_limit_rad=js[prefix + '_stick_x']['limit'][1],
            x_axis=js[prefix + '_stick_x']['axis'].tolist(), y_axis=js[prefix + '_stick_y']['axis'].tolist())
        controls[prefix[0] + '3'] = dict(kind='button', link=link, joint=prefix + '_stick_click', channel=prefix + '_stick',
                                         point=cap.tolist(), push=[0., 0., -1.], travel_m=js[prefix + '_stick_click']['limit'][1])
    for name, link, joint, channel in (('lt', 'left_trigger', 'left_trigger_hinge', 'left_trigger'),
                                       ('rt', 'right_trigger', 'right_trigger_hinge', 'right_trigger')):
        j = js[joint]
        axis = j['axis'] / np.linalg.norm(j['axis'])
        origin = offset[link]
        t = tri(link)
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        area = np.linalg.norm(n, axis=1)
        ok = area > 1e-12
        centres, n, area = t[ok].mean(1), n[ok] / area[ok, None], area[ok]
        flip = np.einsum('ij,ij->i', n, centres - t.reshape(-1, 3).mean(0)) < 0
        n[flip] *= -1
        velocity = np.cross(axis, centres - origin)
        lever = np.linalg.norm(velocity, axis=1)
        vhat = velocity / np.maximum(lever[:, None], 1e-12)
        face = (np.einsum('ij,ij->i', n, vhat) < -.7) & (lever > np.percentile(lever, 60))
        point = np.average(centres[face], axis=0, weights=area[face])
        push = np.average(vhat[face], axis=0, weights=area[face])
        controls[name] = dict(kind='trigger', link=link, joint=joint, channel=channel, point=point.tolist(),
                              push=(push / np.linalg.norm(push)).tolist(), hinge=origin.tolist(), axis=axis.tolist(),
                              travel_rad=j['limit'][1], lever_m=float(np.linalg.norm(np.cross(axis, point - origin))))
    # Positive rotation about a stick joint axis tips the cap toward axis x z; the signal of
    # that joint is positive there (checked on the station: +0.2 rad -> +0.498, 2026-11-12).
    axes = {}
    for name, prefix in (('ls', 'left'), ('rs', 'right')):
        axes[name] = {}
        for joint, signal in ((prefix + '_stick_x', prefix + '_x'), (prefix + '_stick_y', prefix + '_y')):
            tip = np.cross(js[joint]['axis'], [0., 0., 1.])
            component = 'x' if abs(tip[0]) > abs(tip[1]) else 'y'
            axes[name][component] = [signal, int(np.sign(tip[0 if component == 'x' else 1]))]
    shell = tri('controller_body').reshape(-1, 3)
    files['mobility.urdf'] = hashlib.sha256(urdf.read_bytes()).hexdigest()
    return dict(schema='dexhand-autopilot-controls-v1', frame='gamepad controller_body', units='m;rad',
                source=dict(package='xbox_gamepad_physics==0.4.3', resource='robotwin/full_controls', sha256=files,
                            rules='2026-11-11 intent-IK feasibility study scripts/geom.py control_states'),
                body_bounds=[shell.min(0).tolist(), shell.max(0).tolist()], controls=controls,
                signal_axes=axes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('resources')
    parser.add_argument('output')
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(derive(args.resources), indent=1) + '\n')


if __name__ == '__main__':
    main()
