"""Owned MANUS geometry and generic live overlay presentation."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np

# A: MANUS skeleton. C: the G20 frames the retarget tracks. The retarget target vectors (formerly
# layer B) are not drawn: ghost tubes from each G20 origin cluttered the view (2026-11-13).
DEFAULT_LAYERS = dict(enabled=True, A=True, C=True)
def anatomical_geometry(*, retarget='manus_anatomical'):
    """Owned public input geometry; optional providers supply other methods."""
    if retarget != 'manus_anatomical':
        raise ValueError(f'Unavailable retarget geometry: {retarget}')
    from dexstation.inputs.retarget_anatomy import AnatomicalGeometry
    return AnatomicalGeometry()



def _anatomical_fingerprint_resources(*, retarget='manus_anatomical'):
    from dexstation.inputs.retarget_anatomy import ASSETS
    return (Path(__file__).with_name('retarget_anatomy.py'),
            ASSETS/'anatomical.json', ASSETS/'g20_left.urdf', ASSETS/'g20_right.urdf')


anatomical_geometry.fingerprint_resources = _anatomical_fingerprint_resources


def pack_nodes(msg):
    return [dict(node_id=int(n.node_id), parent_node_id=int(n.parent_node_id),
                 chain_type=n.chain_type, joint_type=n.joint_type,
                 position=[float(n.pose.position.x), float(n.pose.position.y), float(n.pose.position.z)])
            for n in msg.raw_nodes]


def unpack_nodes(nodes):
    return [SimpleNamespace(**{k: n[k] for k in ('node_id', 'parent_node_id', 'chain_type', 'joint_type')},
            pose=SimpleNamespace(position=SimpleNamespace(**dict(zip('xyz', n['position']))))) for n in nodes]


@lru_cache(maxsize=2)
def fixed_frames(side):
    """Fixed frames (e.g. ``index_tip``) of the retarget G20 model: child -> (parent link, offset).

    The retarget model adds fingertip and pad frames that the simulated G20 description lacks;
    their parents (``*_distal``) are links of both models.
    """
    import xml.etree.ElementTree as ET
    path = Path(__file__).with_name('retarget_assets')/f'g20_{side}.urdf'
    frames = {}
    for joint in ET.parse(path).getroot().iter('joint'):
        if joint.get('type') != 'fixed':
            continue
        origin = joint.find('origin')
        rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
        if any(abs(v) > 1e-12 for v in rpy):
            continue
        xyz = np.array([float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()])
        frames[joint.find('child').get('link')] = (joint.find('parent').get('link'), xyz)
    return frames


def link_position(links, name, side=None):
    if name in links:
        return np.asarray(links[name].pose.p)
    frame = fixed_frames(side).get(name) if side in ('left', 'right') else None
    if frame is None or frame[0] not in links:
        raise KeyError(f'glove overlay frame {name!r} is neither a simulated link nor a fixed retarget frame')
    parent, offset = frame
    transform = np.asarray(links[parent].pose.to_transformation_matrix())
    return transform[:3, :3] @ offset + transform[:3, 3]


def world_geometry(sample, links, side=None):
    base = np.asarray(links['hand_base_link'].pose.to_transformation_matrix())
    points = np.asarray(sample['points']) @ base[:3, :3].T + base[:3, 3]
    positions = {name: link_position(links, name, side) for name in set(sample['origins']+sample['tasks'])}
    origins = np.array([positions[n] for n in sample['origins']])
    actual = np.array([positions[n] for n in sample['tasks']])
    targets = origins + np.asarray(sample['vectors']) @ base[:3, :3].T
    return dict(points=points, edges=sample['edges'], origins=origins, targets=targets, actual=actual)


# Okabe-Ito-inspired hues adopted in the console art guide. RGB values are sRGB.
FINGER_COLORS = dict(thumb='#ffb000', index='#56b4e9', middle='#3ddc84',
                     ring='#e86bd0', pinky='#ff6f3c', hand='#e6e3dc')


@lru_cache(maxsize=32)
def finger_color(finger, fraction=1.):
    color = FINGER_COLORS.get(finger, FINGER_COLORS['hand'])
    rgb = np.array([int(color[i:i+2], 16)/255 for i in (1, 3, 5)])
    return np.r_[rgb*(.7+.3*fraction), 1.]


@lru_cache(maxsize=8)
def skeleton_styles(labels, edges, tasks):
    """Topology and palette are fixed across frames; only positions change."""
    parents = {a for a, b in edges}
    nodes = []
    for i, (finger, joint) in enumerate(labels):
        fraction = {'MCP': 0., 'PIP': 1/3, 'IP': 2/3, 'DIP': 1.}.get(joint, 1.)
        nodes.append((finger_color(finger, fraction), .006 if finger == 'hand' else .0045,
                      'tip' if finger != 'hand' and i not in parents else 'ring'))
    counts = {finger: sum(n.startswith(finger+'_') for n in tasks) for finger in FINGER_COLORS}
    seen, bones = {}, []
    for name in tasks:
        finger = name.split('_')[0]
        rank = seen.get(finger, 0)
        seen[finger] = rank+1
        bones.append((finger_color(finger, rank/max(counts.get(finger, 1)-1, 1)),
                      'tip' if rank == counts.get(finger, 0)-1 else 'ring'))
    return nodes, bones


def segment_transforms(segments, radius):
    data = np.asarray(segments, dtype=float)
    starts = data[:, 0]
    directions = data[:, 1]-starts
    lengths = np.linalg.norm(directions, axis=1)
    x = directions/np.maximum(lengths[:, None], 1e-12)
    rotations = np.column_stack([1+x[:, 0], np.zeros(len(x)), -x[:, 2], x[:, 1]])
    opposite = rotations[:, 0] < 1e-8
    rotations[opposite] = [0., 0., 1., 0.]
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1)[:, None], 1e-12)
    return starts, rotations, np.column_stack([lengths, np.full((len(x), 2), radius)])


class GloveOverlay:
    """Pooled emissive balls/tubes with unlit, always-visible joint contours.

    Meshes obey scene depth; the viewer line pass preserves their outlines even
    behind the robot. No physics objects or second render/compositing pipeline.
    """
    def __init__(self, backend):
        self.backend = backend
        self.objects = {}
        self.models = {}
        self.links = {}
        self.visible = set()
        self.render_scene = backend.scene.render_system._internal_scene
        import sapien
        self.context = sapien.render.SapienRenderer()._internal_context
        self.sphere = self.context.create_uvsphere_mesh(16, 8)
        # Unit tube along +X, with rounded ends supplied by the joint spheres.
        angle = np.linspace(0, 2*np.pi, 17)[:-1]
        yz = np.column_stack([np.cos(angle), np.sin(angle)])
        vertices = np.concatenate([np.column_stack([np.full(16, x), yz]) for x in (0., 1.)])
        normals = np.tile(np.column_stack([np.zeros(16), yz]), (2, 1))
        faces = []
        for i in range(16):
            j = (i+1) % 16
            faces.extend([(i, j, i+16), (j, j+16, i+16)])
        self.tube = self.context.create_mesh_from_array(vertices.astype('float32'),
                      np.asarray(faces, dtype='uint32'), normals.astype('float32'))
        theta = np.linspace(0, 2*np.pi, 25)
        circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(25)])
        self.rings = np.concatenate([np.stack([np.roll(circle, axis, axis=1)[:-1],
                       np.roll(circle, axis, axis=1)[1:]], axis=1) for axis in range(3)]).reshape(-1, 3)
        # Tips add an octahedral contour, distinct from round intermediate joints.
        tips = np.r_[np.eye(3), -np.eye(3)]
        self.tip_outline = np.array([(a, b) for i, a in enumerate(tips)
                          for b in tips[i+1:] if np.dot(a, b) == 0]).reshape(-1, 3)
        for side, robot in (('left', backend.robot), ('right', backend.right_robot)):
            if robot is not None:
                prefix = 'right_' if side == 'right' else ''
                self.links[side] = {l.name.removeprefix(prefix): l for l in robot.get_links()}

    def hide(self):
        for key in self.visible:
            self.objects[key].set_scale([0., 0., 0.])
        self.visible.clear()

    def _object(self, key, shape, color, position, scale, rotation=(1., 0., 0., 0.), ghost=False):
        if key in self.objects:
            obj = self.objects[key]
            obj.set_position(position)
            if shape in ('tube', 'line'):
                obj.set_rotation(rotation)
            obj.set_scale(scale)
            self.visible.add(key)
            return
        model_key = (shape, tuple(color), ghost)
        if model_key not in self.models:
            if shape in ('ring', 'tip', 'line'):
                vertices = self.rings if shape == 'ring' else self.tip_outline if shape == 'tip' else np.array([[0., 0., 0.], [1., 0., 0.]])
                rgba = np.array(color)
                if ghost:
                    rgba[3] = .4
                self.models[model_key] = self.context.create_line_set(vertices.astype('float32'),
                        np.tile(rgba, (len(vertices), 1)).astype('float32'))
            else:
                material = self.context.create_material(
                    np.r_[np.asarray(color[:3])*.65, 1.], color, .5, .24, .08)
                self.models[model_key] = self.context.create_model(
                    [self.sphere if shape == 'ball' else self.tube], [material])
        if key not in self.objects:
            if shape in ('ring', 'tip', 'line'):
                obj = self.render_scene.add_line_set(self.models[model_key])
                obj.line_width = 2.
            else:
                obj = self.render_scene.add_object(self.models[model_key])
                obj.cast_shadow = False
                if ghost:
                    obj.transparency = .6
            self.objects[key] = obj
        obj = self.objects[key]
        obj.set_position(position)
        obj.set_rotation(rotation)
        obj.set_scale(scale)
        self.visible.add(key)

    def update(self, samples, layers=None):
        previous = self.visible
        self.visible = set()
        layers = dict(DEFAULT_LAYERS, **(layers or {}))
        if layers['enabled']:
            for side, sample in samples.items():
                if side not in self.links:
                    continue
                g = world_geometry(sample, self.links[side], side)
                # Labels travel with points: MANUS message order is not assumed.
                labels = tuple(tuple(label) for label in sample['labels'])
                nodes, bones = skeleton_styles(labels, tuple(map(tuple, sample['edges'])), tuple(sample['tasks']))
                if layers['A']:
                    for i, p in enumerate(g['points']):
                        color, radius, shape = nodes[i]
                        self._object((side, 'A', 'ball', labels[i]), 'ball', color, p, [radius]*3)
                        self._object((side, 'A', 'edge', labels[i]), shape, color, p, [radius*1.12]*3)
                    segments = [(g['points'][a], g['points'][b]) for a, b in g['edges']]
                    for i, (p, q, scale) in enumerate(zip(*segment_transforms(segments, .002))):
                        a, b = g['edges'][i]
                        color = nodes[b][0]
                        edge = (labels[a], labels[b])
                        self._object((side, 'A', 'tube', edge), 'tube', color, p, scale, q)
                        # Unlit centerline remains legible when the tube is occluded.
                        self._object((side, 'A', 'line', edge), 'line', color, p, [scale[0], 1., 1.], q)
                if layers['C']:
                    for i, (color, shape) in enumerate(bones):
                        self._object((side, 'C', 'ring', i), shape, color, g['actual'][i], [.0055]*3)
        for key in previous-self.visible:
            self.objects[key].set_scale([0., 0., 0.])


def update_live_overlay(backend, state):
    overlay = getattr(backend, 'glove_overlay', None)
    samples = state.get('glove_overlay', {})
    if overlay is None and samples:
        overlay = backend.glove_overlay = GloveOverlay(backend)
    if overlay:
        try:
            overlay.update(samples, state.get('glove_layers'))
        except Exception as exc:   # a drawing aid must never stop the render workers
            overlay.hide()
            message = f'glove overlay disabled for this frame: {type(exc).__name__}: {exc}'
            if getattr(backend, 'glove_overlay_error', None) != message:
                backend.glove_overlay_error = message
                import sys
                print(message, file=sys.stderr, flush=True)
            return dict(error=message)
    return {}
