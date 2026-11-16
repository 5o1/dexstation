#!/usr/bin/env python3
"""Offline clearance between the G20 hand and the gamepad (numpy only; never used at runtime).

Hand: collision-mesh vertices of every link, voxel-downsampled. Gamepad: the CoACD convex
pieces of the RoboTwin resource (the scene loads the exact SDF meshes; the hulls enclose them
with at most ~2.4 mm excess, so clearances here are conservative). Per point the clearance is
the largest plane distance of the nearest convex piece (negative inside).
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from derive_autopilot_controls import joints, link_offsets
from derive_autopilot_pads import load_stl


def voxel(points, size):
    key = np.floor(points / size).astype(np.int64)
    _, index = np.unique(key, axis=0, return_index=True)
    return points[np.sort(index)]


class HandPoints:
    def __init__(self, urdf, size=.003):
        root = ET.parse(urdf).getroot()
        self.links = {}
        for link in root.findall('link'):
            mesh = link.find('collision/geometry/mesh')
            if mesh is not None:
                tri = load_stl((Path(urdf).parent / mesh.get('filename')).resolve())
                self.links[link.get('name')] = voxel(np.concatenate([tri.reshape(-1, 3), tri.mean(1)]), size)

    def world(self, hand, finger_q, palm):
        """{link: (N, 3) world points} for finger joint vectors {finger: q}."""
        result = {'hand_base_link': self.links['hand_base_link'] @ palm[:3, :3].T + palm[:3, 3]}
        for finger, chain in hand.fingers.items():
            for link, pose in chain.link_frames(finger_q[finger], palm).items():
                if link in self.links:
                    result[link] = self.links[link] @ pose[:3, :3].T + pose[:3, 3]
        return result


def load_obj_parts(path):
    """Convex pieces of a CoACD OBJ (one per 'o'/'g' group) as triangle arrays."""
    parts, vertices, current = [], [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith(('o ', 'g ')):
            if current:
                parts.append(current)
            current = []
        elif line.startswith('v '):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith('f '):
            if current is None:
                current = []
            current.append([int(t.split('/')[0]) - 1 for t in line.split()[1:4]])
    if current:
        parts.append(current)
    vertices = np.asarray(vertices)
    return [vertices[np.asarray(f)] for f in parts]


class PadBody:
    """Gamepad as the scene's CoACD convex pieces (controls at rest); conservative by at most ~2.4 mm."""

    def __init__(self, resources):
        root = Path(resources)
        offsets = link_offsets(joints(root / 'mobility.urdf'))
        normals, offsets_d, owner, names = [], [], [], []
        for element in ET.parse(root / 'mobility.urdf').getroot().findall('link'):
            link = element.get('name')
            for mesh in element.findall('collision/geometry/mesh'):
                for tri in load_obj_parts(root / mesh.get('filename')):
                    tri = tri + offsets[link]
                    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    area = np.linalg.norm(n, axis=1)
                    keep = area > 1e-14
                    n, p0 = n[keep] / area[keep, None], tri[keep, 0]
                    d = np.einsum('ij,ij->i', n, p0)
                    centre = tri.reshape(-1, 3).mean(0)
                    flip = n @ centre - d > 0
                    n[flip] *= -1
                    d[flip] *= -1
                    normals.append(n)
                    offsets_d.append(d)
                    owner.append(np.full(len(n), len(names)))
                    names.append(link)
        self.normals, self.d = np.concatenate(normals), np.concatenate(offsets_d)
        self.owner, self.names = np.concatenate(owner), names
        self.starts = np.flatnonzero(np.r_[True, np.diff(self.owner) != 0])
        allp = np.concatenate([self.normals * self.d[:, None]])
        self.lo, self.hi = np.array([-.09, -.07, -.045]), np.array([.09, .07, .045])

    def nearest_plane(self, points_local):
        """Clearance, outward normal (gamepad frame) and hull link of each point's binding plane."""
        plane = points_local @ self.normals.T - self.d
        per_hull = np.maximum.reduceat(plane, self.starts, axis=1)
        hull = per_hull.argmin(1)
        ends = np.r_[self.starts[1:], len(self.d)]
        normals = np.empty((len(points_local), 3))
        for i, h in enumerate(hull):
            j = self.starts[h] + int(plane[i, self.starts[h]:ends[h]].argmax())
            normals[i] = self.normals[j]
        return per_hull[np.arange(len(hull)), hull], normals, [self.names[h] for h in hull]

    def clearance(self, points_local, chunk=256):
        """Signed distance lower bound (m) to the nearest convex piece, and that piece's link."""
        result = np.full(len(points_local), .05)
        nearest = np.full(len(points_local), None, dtype=object)
        near = np.flatnonzero(np.all((points_local > self.lo - .05) & (points_local < self.hi + .05), axis=1))
        for start in range(0, len(near), chunk):
            index = near[start:start + chunk]
            plane = points_local[index] @ self.normals.T - self.d
            per_hull = np.maximum.reduceat(plane, self.starts, axis=1)
            k = per_hull.argmin(1)
            result[index] = per_hull[np.arange(len(index)), k]
            nearest[index] = [self.names[i] for i in k]
        return result, nearest


class ExactPad:
    """The exact closed SDF meshes the scene loads (controls at rest): winding-number inside test plus the
    distance to dense surface samples. Slower than PadBody; used to confirm PadBody's conservative hits."""

    FILES = {'controller_body': ['shell_0', 'shell_front_plate_0'], 'a': ['a_0'], 'b': ['b_0'], 'x': ['x_0'], 'y': ['y_0'],
             'dpad': ['dpad_0'], 'left_shoulder': ['left_shoulder_0'], 'right_shoulder': ['right_shoulder_0'],
             'left_stick': ['left_stick_0', 'left_stick_1'], 'right_stick': ['right_stick_0', 'right_stick_1'],
             'left_trigger': ['left_trigger_0'], 'right_trigger': ['right_trigger_0'], 'view': ['view_0'],
             'menu': ['menu_0'], 'guide': ['guide_0'], 'pair': ['pair_0']}

    def __init__(self, resources):
        from derive_autopilot_controls import load_obj
        root = Path(resources)
        offsets = link_offsets(joints(root / 'mobility.urdf'))
        self.parts = {link: np.concatenate([load_obj(root / f'collision/sdf/{n}.obj') for n in names]) + offsets[link]
                      for link, names in self.FILES.items()}
        self.samples = {link: voxel(np.concatenate([t.reshape(-1, 3), t.mean(1), (t[:, 0] + t[:, 1]) / 2,
                                                    (t[:, 1] + t[:, 2]) / 2, (t[:, 2] + t[:, 0]) / 2]), .0007)
                        for link, t in self.parts.items()}

    @staticmethod
    def winding(tri, points):
        a, b, c = (tri[None, :, i, :] - points[:, None, :] for i in range(3))
        la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
        det = np.einsum('pti,pti->pt', a, np.cross(b, c))
        den = la * lb * lc + np.einsum('pti,pti->pt', a, b) * lc + np.einsum('pti,pti->pt', b, c) * la + \
            np.einsum('pti,pti->pt', c, a) * lb
        return np.abs((2 * np.arctan2(det, den)).sum(1) / (4 * np.pi))

    def clearance(self, points_local):
        """Signed clearance (m, negative inside) and nearest link, per point."""
        best = np.full(len(points_local), np.inf)
        owner = np.full(len(points_local), None, dtype=object)
        for link, tri in self.parts.items():
            d = np.linalg.norm(points_local[:, None, :] - self.samples[link][None], axis=2).min(1)
            near = d < .015
            inside = np.zeros(len(points_local), bool)
            if near.any():
                inside[near] = self.winding(tri, points_local[near]) > .5
            signed = np.where(inside, -d, d)
            better = signed < best
            best[better], owner[better] = signed[better], link
        return best, owner


def hand_clearance(pad_body, hand_points, hand, finger_q, palm, pad_pose, exclude=()):
    """Minimum clearance per hand link (m) and the gamepad link it is nearest to."""
    inverse = np.linalg.inv(pad_pose)
    report = {}
    for link, points in hand_points.world(hand, finger_q, palm).items():
        if link in exclude:
            continue
        local = points @ inverse[:3, :3].T + inverse[:3, 3]
        clearance, nearest = pad_body.clearance(local)
        k = int(clearance.argmin())
        report[link] = (float(clearance[k]), nearest[k])
    return report
