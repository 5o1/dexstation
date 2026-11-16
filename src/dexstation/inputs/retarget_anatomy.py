"""MANUS ROS labels -> anatomical nodes, palm frame and G20 correspondences.

SDK strings name bones, not anatomical joint acronyms. All positions are metres
in the same world frame. No optimizer, ROS or device dependency in this module.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from dexstation.model.kinematics import SerialChain

ASSETS = Path(__file__).with_name('retarget_assets')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
RAW_TO_ANATOMICAL = dict(MCP='metacarpal_base', PIP='mcp', IP='pip', DIP='dip', TIP='tip')
THUMB_RAW_TO_ANATOMICAL = dict(MCP='cmc', PIP='mcp', DIP='ip', TIP='tip')


class SkeletonError(ValueError):
    pass


def load_config(path=None):
    return json.loads(Path(path or ASSETS/'anatomical.json').read_text())


@dataclass(frozen=True)
class Pair:
    finger: str
    origin: str
    task: str
    robot_origin: str
    robot_task: str
    distal: bool = False


PAIRS = tuple([
    Pair('thumb', 'cmc', 'mcp', 'thumb_metacarpals', 'thumb_proximal'),
    Pair('thumb', 'mcp', 'ip', 'thumb_proximal', 'thumb_distal'),
    Pair('thumb', 'ip', 'tip', 'thumb_distal', 'thumb_tip', True),
] + [Pair(f, a, b, f+'_'+ra, f+'_'+rb, b == 'tip')
     for f in FINGERS[1:]
     for a, b, ra, rb in [('mcp', 'pip', 'proximal', 'middle'),
                          ('pip', 'dip', 'middle', 'distal'),
                          ('dip', 'tip', 'distal', 'tip')]])


def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-8:
        raise SkeletonError('degenerate skeleton direction')
    return vector/length


def relabel_nodes(raw_nodes, ranges=None):
    """Validate full 25-node world skeleton and expected parent topology.

    Bounds cover the supplied measured skeleton with margin, not exact identity
    tests. A differently scaled/local-space or missing skeleton is rejected.
    """
    ranges = ranges or load_config()['segment_ranges_m']
    points, identities, ids = {}, {}, set()
    for n in raw_nodes:
        finger = n.chain_type.lower()
        mapping = THUMB_RAW_TO_ANATOMICAL if finger == 'thumb' else RAW_TO_ANATOMICAL
        label = ('hand', 'wrist') if finger == 'hand' else (finger, mapping.get(n.joint_type))
        if label != ('hand', 'wrist') and (finger not in FINGERS or label[1] is None):
            raise SkeletonError(f'unknown MANUS label: {finger}/{n.joint_type}')
        if label in points or n.node_id in ids:
            raise SkeletonError(f'duplicate MANUS node: {label}')
        p = n.pose.position
        point = np.array([p.x, p.y, p.z], dtype=float)
        if not np.isfinite(point).all():
            raise SkeletonError(f'nonfinite MANUS node: {label}')
        points[label] = point
        identities[label] = (n.node_id, n.parent_node_id)
        ids.add(n.node_id)
    expected = {('hand', 'wrist')}
    for f in FINGERS:
        expected.update((f, j) for j in (THUMB_RAW_TO_ANATOMICAL if f == 'thumb' else RAW_TO_ANATOMICAL).values())
    if points.keys() != expected:
        raise SkeletonError(f'expected 25 anatomical nodes; missing={sorted(expected-points.keys())}')
    for f in FINGERS:
        joints = ['wrist', 'cmc', 'mcp', 'ip', 'tip'] if f == 'thumb' else ['wrist', 'metacarpal_base', 'mcp', 'pip', 'dip', 'tip']
        kinds = ['wrist_to_metacarpal', 'thumb_metacarpal', 'thumb_proximal', 'thumb_distal'] if f == 'thumb' else ['wrist_to_metacarpal', 'metacarpal', 'proximal', 'intermediate', 'distal']
        for a, b, kind in zip(joints, joints[1:], kinds):
            la, lb = ('hand', a) if a == 'wrist' else (f, a), (f, b)
            if identities[lb][1] != identities[la][0]:
                raise SkeletonError(f'wrong parent: {la}->{lb}')
            length = float(np.linalg.norm(points[lb]-points[la]))
            lo, hi = ranges[kind]
            if not lo <= length <= hi:
                raise SkeletonError(f'{f}/{kind}: {length:.6f} m outside [{lo}, {hi}]')
    return points


def palm_basis(points, side):
    """Right-handed G20 frame: +Z forward, Y across palm; side preserves chirality.

    Uses wrist and all four anatomical MCPs (raw PIP), never raw MCP roots.
    """
    if side not in ('left', 'right'):
        raise SkeletonError(f'invalid hand side: {side}')
    knuckles = np.array([points[f, 'mcp'] for f in FINGERS[1:]])
    forward = unit(knuckles.mean(0)-points['hand', 'wrist'])
    across = knuckles[0]-knuckles[-1]
    if side == 'left':
        across = -across
    across = unit(across-np.dot(across, forward)*forward)
    return np.stack([unit(np.cross(across, forward)), across, forward], axis=1)


def human_hand_length(points):
    """Measured skeletal hand length: wrist->middle MCP plus all three phalanges.

    Sum of rigid segment lengths is stable when the finger bends.
    """
    labels = [('hand', 'wrist')]+[('middle', j) for j in ('mcp', 'pip', 'dip', 'tip')]
    return float(sum(np.linalg.norm(points[b]-points[a]) for a, b in zip(labels, labels[1:])))


@lru_cache(maxsize=2)
def robot_geometry(side):
    if side not in ('left', 'right'):
        raise ValueError('invalid side')
    path = ASSETS/f'g20_{side}.urdf'
    positions = {'hand_base_link': np.zeros(3)}
    for f in FINGERS:
        chain = SerialChain(path, f+'_tip')
        positions.update({k: v[0, :3, 3] for k, v in chain.link_poses(np.zeros(len(chain.active))).items()})
    lengths = np.array([np.linalg.norm(positions[p.robot_task]-positions[p.robot_origin]) for p in PAIRS])
    if np.any(lengths < .005):
        raise ValueError('robot correspondence contains a segment shorter than 5 mm')
    hand_length = np.linalg.norm(positions['middle_proximal'])+lengths[6:9].sum()
    return lengths, float(hand_length)


def anatomical_targets(raw_nodes, side, config=None):
    """Shared live/overlay/offline reference. No temporal state or compensation."""
    config = config or load_config()
    raw_nodes = list(raw_nodes)
    points = relabel_nodes(raw_nodes, config['segment_ranges_m'])
    basis = palm_basis(points, side)
    wrist = points['hand', 'wrist']
    local = {k: (v-wrist) @ basis for k, v in points.items()}
    lengths, robot_length = robot_geometry(side)
    directions = np.array([unit(local[p.finger, p.task]-local[p.finger, p.origin]) for p in PAIRS])
    tips = np.array([local[f, 'tip'] for f in FINGERS])
    # Native DexPilot ordering: inter-tip pairs, then wrist->tips.
    pilot_points = np.vstack([np.zeros(3), tips])
    pilot_pairs = [(j, i) for i in range(1, 5) for j in range(i+1, 6)]+[(0, i) for i in range(1, 6)]
    scale = robot_length/human_hand_length(points)
    pilot_vectors = np.array([pilot_points[b]-pilot_points[a] for a, b in pilot_pairs])*scale
    labels = [('hand', 'wrist') if n.chain_type.lower() == 'hand' else
              (n.chain_type.lower(), (THUMB_RAW_TO_ANATOMICAL if n.chain_type.lower() == 'thumb' else RAW_TO_ANATOMICAL)[n.joint_type]) for n in raw_nodes]
    id_index = {n.node_id: i for i, n in enumerate(raw_nodes)}
    edges = [(id_index[n.parent_node_id], i) for i, n in enumerate(raw_nodes) if n.parent_node_id in id_index and n.parent_node_id != n.node_id]
    return dict(points=np.array([local[k] for k in labels]).tolist(), labels=labels, edges=edges,
                vectors=(directions*lengths[:, None]).tolist(), directions=directions,
                origins=[p.robot_origin for p in PAIRS], tasks=[p.robot_task for p in PAIRS],
                pilot_vectors=pilot_vectors, human_length=human_hand_length(points),
                robot_length=robot_length, tips=tips)


class AnatomicalGeometry:
    """Overlay adapter; same target preparation as the running node."""
    def prepare(self, nodes, side):
        value = anatomical_targets(nodes, side)
        return {k: value[k] for k in ('points', 'labels', 'edges', 'vectors', 'origins', 'tasks')}
