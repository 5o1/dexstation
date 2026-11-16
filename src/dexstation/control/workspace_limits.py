"""Backend-independent workspace geometry, checks and user-selected hold protocol.

Only construction loads meshes (the existing sim extra supplies trimesh).
Control-cycle math uses numpy. Surface sampling is deterministic: triangle
barycentric grids have <=5 mm edges, then one existing surface point per
2.5 mm voxel is retained. Every surface location is within 5+sqrt(3)*2.5 mm
of a retained point (<10 mm). No runtime import/read of the research prototype.

Reviewer rulings 2/3: true signed Euclidean obstacle distances, conservative
cluster/grid screening, followed by exact distances on retained original points.
Mesh solids retain their triangles; no nonconvex AABB substitution. Robot pairs
use exact Euclidean nearest neighbours between actual surface samples.
"""
import copy
import itertools
import hashlib
import os
import time
import json
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from dexstation.model.constants import LOW_DEG, HIGH_DEG
from dexstation.contracts.limits import AngularViolation, LimitReport, SpatialViolation
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES
from dexstation.model.kinematics import SerialChain, _origin
from dexstation.model.spec import pose_to_matrix, compose as compose_pose


def _vector(value, n):
    q = np.asarray(value, float)
    if q.shape != (n,) or not np.isfinite(q).all():
        raise ValueError(f'expected {n} finite joint positions')
    return q


def _transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def surface_points(triangles):
    """Deterministic <10 mm surface coverage; retained points stay on the mesh."""
    triangles = np.asarray(triangles, float)
    edges = np.stack([triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 1],
                      triangles[:, 0]-triangles[:, 2]], 1)
    counts = np.maximum(1, np.ceil(np.linalg.norm(edges, axis=2).max(1)/.005).astype(int))
    samples = []
    for n in np.unique(counts):
        tri = triangles[counts == n]
        uv = np.array([(i/n, j/n) for i in range(n+1) for j in range(n+1-i)])
        # Bound temporary memory for finely tessellated imported meshes.
        for start in range(0, len(tri), max(1, 100000//len(uv))):
            part = tri[start:start+max(1, 100000//len(uv))]
            samples.append((part[:, None, 0] + uv[None, :, :1]*(part[:, None, 1]-part[:, None, 0])
                            + uv[None, :, 1:]*(part[:, None, 2]-part[:, None, 0])).reshape(-1, 3))
    points = np.concatenate(samples)
    _, indices = np.unique(np.floor(points/.0025).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(indices)]


def _link_points(urdf):
    import trimesh
    result = {}
    for link in ET.parse(urdf).getroot().findall('link'):
        parts = []
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            mesh = geometry.find('mesh')
            if mesh is not None:
                path = Path(mesh.get('filename'))
                path = path if path.is_absolute() else Path(urdf).parent/path
                model = trimesh.load(path, force='mesh')
                tri = np.asarray(model.triangles)*np.array([float(v) for v in mesh.get('scale', '1 1 1').split()])
            elif geometry.find('box') is not None:
                tri = np.asarray(trimesh.creation.box(extents=[float(v) for v in geometry.find('box').get('size').split()]).triangles)
            else:
                raise ValueError('unsupported robot collision geometry')
            parts.append(surface_points(_transform(tri, _origin(collision))))
        if parts:
            result[link.get('name')] = np.concatenate(parts)
    return result


def _triangle_closest_exact(triangles, points):
    """Point/triangle projections without a fixed, dimensioned region tolerance.

    Test the plane projection with oriented edge areas, and compare the three
    exact segment projections. In particular, tiny triangles are not snapped
    to an edge merely because a fourth-power dot product is below a tolerance.
    """
    triangles, points = np.asarray(triangles, float), np.asarray(points, float)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normal = np.cross(b-a, c-a)
    n2 = np.einsum('ij,ij->i', normal, normal)
    factor = np.divide(np.einsum('ij,ij->i', points-a, normal), n2,
                       out=np.zeros(len(points)), where=n2 > 0)
    projection = points-factor[:, None]*normal
    inside = n2 > 0
    for u, v in ((a,b), (b,c), (c,a)):
        inside &= np.einsum('ij,ij->i', np.cross(v-u, projection-u), normal) >= 0
    result = projection.copy()
    best = np.where(inside, np.einsum('ij,ij->i', points-projection, points-projection), np.inf)
    for u, v in ((a,b), (b,c), (c,a)):
        edge = v-u
        squared = np.einsum('ij,ij->i', edge, edge)
        t = np.divide(np.einsum('ij,ij->i', points-u, edge), squared,
                      out=np.zeros(len(points)), where=squared > 0)
        closest = u+np.clip(t, 0, 1)[:, None]*edge
        distance = np.einsum('ij,ij->i', points-closest, points-closest)
        take = distance < best
        result[take], best[take] = closest[take], distance[take]
    return result


def _mesh_unsigned_exact(mesh, points):
    from trimesh.proximity import nearby_faces
    distances = np.empty(len(points))
    for start in range(0, len(points), 256):
        p = points[start:start+256]
        candidates = nearby_faces(mesh, p)
        rows = np.repeat(np.arange(len(p)), [len(c) for c in candidates])
        faces = np.concatenate(candidates)
        best = np.full(len(p), np.inf)
        for begin in range(0, len(rows), 20000):
            r, f = rows[begin:begin+20000], faces[begin:begin+20000]
            nearest = _triangle_closest_exact(mesh.triangles[f], p[r])
            delta = p[r]-nearest
            np.minimum.at(best, r, np.einsum('ij,ij->i', delta, delta))
        distances[start:start+len(p)] = np.sqrt(best)
    return distances

@dataclass
class Obstacle:
    name: str
    normals: np.ndarray
    offsets: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    transform: np.ndarray | None = None
    half: np.ndarray | None = None
    triangles: np.ndarray | None = None
    mesh: object = None
    is_convex_solid: bool = False

    @classmethod
    def box(cls, name, pose, half):
        half = np.asarray(half, float)
        if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
            raise ValueError('box half extents must be positive finite metres')
        transform = pose_to_matrix(pose)
        normals = np.r_[transform[:3, :3].T, -transform[:3, :3].T]
        offsets = normals @ transform[:3, 3] + np.r_[half, half]
        vertices = _transform(np.array(list(itertools.product(*zip(-half, half)))), transform)
        return cls(name, normals, offsets, vertices.min(0), vertices.max(0), transform, half)

    @classmethod
    def convex(cls, name, triangles):
        # Historical name retained for static-check callers. The original mesh
        # is preserved even when it is nonconvex; signed queries use the solid.
        import trimesh
        tri = np.asarray(triangles, float)
        model = trimesh.Trimesh(vertices=tri.reshape(-1, 3),
                               faces=np.arange(tri.size//3).reshape(-1, 3), process=True)
        if not model.is_watertight:
            raise ValueError(f'Obstacle is not a closed solid: {name}')
        model.fix_normals()
        normals = np.asarray(model.face_normals)
        offsets = np.einsum('ij,ij->i', normals, np.asarray(model.triangles)[:, 0])
        vertices = np.asarray(model.vertices)
        residual = max(float((vertices @ normals[i:i+256].T-offsets[i:i+256]).max())
                       for i in range(0, len(normals), 256))
        return cls(name, normals, offsets, vertices.min(0), vertices.max(0),
                   triangles=np.asarray(model.triangles), mesh=model, is_convex_solid=residual <= 1e-12)

    def signed_distances(self, points):
        """One true signed Euclidean distance per point (inside negative)."""
        points = np.asarray(points, float)
        if not len(points):
            return np.empty(0)
        if self.half is not None:
            local = (points-self.transform[:3, 3]) @ self.transform[:3, :3]
            excess = np.abs(local)-self.half
            return np.linalg.norm(np.maximum(excess, 0.), axis=1)+np.minimum(excess.max(1), 0.)
        if self.is_convex_solid:
            # Inside a convex solid, the nearest boundary distance is the
            # minimum supporting-plane slack. Outside, only triangle distance
            # is exact at edges and corners.
            signed = np.empty(len(points))
            for start in range(0, len(points), 256):
                signed[start:start+256] = (points[start:start+256] @ self.normals.T-self.offsets).max(1)
            outside = signed > 0
            signed[outside] = _mesh_unsigned_exact(self.mesh, points[outside])
            return signed
        unsigned = _mesh_unsigned_exact(self.mesh, points)
        return np.where(self.mesh.contains(points), -unsigned, unsigned)

    def distance(self, points):
        return float(self.signed_distances(points).min())

    def fingerprint(self, digest):
        digest.update(self.name.encode())
        for value in (self.transform, self.half, self.triangles):
            if value is not None:
                digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())


def _fixtures(spec, names=None):
    import trimesh
    obstacles = [Obstacle.box('table', spec.table.pose, spec.table.half_extents)]
    for fixture in spec.fixtures:
        if names is not None and fixture.name not in names:
            continue
        if names is None and not fixture.name.startswith(('station_table_frame', 'station_column', 'station_floor')):
            continue
        if not fixture.collision:
            continue
        if fixture.type == 'box':
            obstacles.append(Obstacle.box(fixture.name, fixture.pose, fixture.half_extents))
        else:
            model = trimesh.load(fixture.mesh, force='mesh')
            model.merge_vertices()
            for i, component in enumerate(model.split(only_watertight=False)):
                if len(component.faces) >= 4:
                    obstacles.append(Obstacle.convex(f'{fixture.name}#{i}', _transform(np.asarray(component.triangles), pose_to_matrix(fixture.pose))))
    return obstacles

@dataclass
class PointClusters:
    points: np.ndarray
    centres: np.ndarray
    radii: np.ndarray
    labels: np.ndarray
    centre: np.ndarray
    radius: float

    @classmethod
    def build(cls, points, voxel_m=.025):
        points = np.asarray(points, float)
        _, labels = np.unique(np.floor(points/voxel_m).astype(np.int64), axis=0, return_inverse=True)
        count = np.bincount(labels)
        centres = np.column_stack([np.bincount(labels, weights=points[:, i])/count for i in range(3)])
        radii = np.zeros(len(count))
        np.maximum.at(radii, labels, np.linalg.norm(points-centres[labels], axis=1))
        # Outward rounding prevents a last-bit undersized sphere.
        radii = np.nextafter(radii, np.inf)
        centre = (points.min(0)+points.max(0))/2
        radius = float(np.nextafter(np.linalg.norm(points-centre, axis=1).max(), np.inf))
        return cls(points, centres, radii, labels, centre, radius)

    def point_indices(self, cluster_mask):
        return np.flatnonzero(cluster_mask[self.labels])


class DistanceField:
    """Clipped minimum of signed obstacle distances on a regular world grid.

    min of 1-Lipschitz functions, followed by clipping, is 1-Lipschitz. For
    trilinear weights, sum(w*|vertex-p|) <= sqrt(sum(t*(h-t))) <= h*sqrt(3)/2.
    Grid arithmetic gets an additional 1e-12 m outward numerical allowance.
    Outside the sampled domain queries return -inf (no screening).
    """
    def __init__(self, obstacles, lo, hi, cache_dir, spacing=.01):
        self.spacing = float(spacing)
        self.error = self.spacing*np.sqrt(3)/2+1e-12
        self.lo = np.floor(np.asarray(lo)/spacing)*spacing
        self.shape = tuple((np.ceil((np.asarray(hi)-self.lo)/spacing).astype(int)+1).tolist())
        self.band = .125
        self.values = None
        if not obstacles:
            return
        digest = hashlib.sha256(b'wp5-euclidean-field-v2-exact-triangles-float64')
        digest.update(np.r_[self.lo, self.shape, spacing, self.band].tobytes())
        for obstacle in obstacles:
            obstacle.fingerprint(digest)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir/(digest.hexdigest()+'.npy')
        if self.path.exists():
            self.values = np.load(self.path, mmap_mode='r', allow_pickle=False)
            if self.values.shape != self.shape or self.values.dtype != np.float64:
                raise ValueError('distance-field cache shape/dtype mismatch')
            return
        temporary = cache_dir/(digest.hexdigest()+f'.{os.getpid()}.tmp.npy')
        grid = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float64, shape=self.shape)
        grid[:] = self.band
        for obstacle in obstacles:
            start = np.maximum(0, np.floor((obstacle.lo-self.band-self.lo)/spacing).astype(int))
            stop = np.minimum(self.shape, np.ceil((obstacle.hi+self.band-self.lo)/spacing).astype(int)+1)
            if np.any(stop <= start):
                continue
            # At most one x slab is materialized; no full-domain point array.
            yz = np.array(list(itertools.product(range(start[1], stop[1]), range(start[2], stop[2]))))
            for x in range(start[0], stop[0]):
                points = self.lo+spacing*np.column_stack([np.full(len(yz), x), yz])
                # Euclidean AABB distance is a valid lower bound outside it.
                delta = np.maximum(np.maximum(obstacle.lo-points, points-obstacle.hi), 0.)
                near = np.linalg.norm(delta, axis=1) < self.band
                if near.any():
                    distances = np.clip(obstacle.signed_distances(points[near]), -self.band, self.band)
                    y, z = yz[near].T
                    grid[x, y, z] = np.minimum(grid[x, y, z], distances)
        grid.flush()
        del grid
        os.replace(temporary, self.path)
        self.values = np.load(self.path, mmap_mode='r', allow_pickle=False)

    def query(self, points):
        points = np.asarray(points, float)
        if self.values is None:
            return np.full(len(points), np.inf)
        coords = (points-self.lo)/self.spacing
        indices = np.floor(coords).astype(np.int64)
        good = np.all((indices >= 0) & (indices < np.asarray(self.shape)-1), axis=1)
        result = np.full(len(points), -np.inf)
        if not good.any():
            return result
        idx = indices[good]
        fraction = coords[good]-idx
        value = np.zeros(len(idx))
        for corner in itertools.product((0, 1), repeat=3):
            offset = np.asarray(corner)
            weight = np.prod(np.where(offset, fraction, 1-fraction), axis=1)
            value += weight*self.values[tuple((idx+offset).T)]
        result[good] = value
        return result


@dataclass
class PlacedSurface:
    side: str
    name: str
    kind: str
    clusters: PointClusters
    pose: np.ndarray
    centres: np.ndarray
    centre: np.ndarray

    def transform_points(self, indices):
        return _transform(self.clusters.points[indices], self.pose)


def _centre_displacement(current, previous):
    """Rigid-cluster centre displacement; radii are fixed in link coordinates."""
    return np.linalg.norm(current-previous, axis=-1)


class TemporalBounds:
    """Conservative environment bounds with a periodic full-query anchor.

    lower stores interpolated centre distance minus cluster radius. Its one
    interpolation error is subtracted only in the decision, while every reused
    bound loses the full displacement from the position where it was last used.
    No decision is reused without evaluating the current centre positions.
    """
    def __init__(self, anchor_every=30, jump_m=.05):
        self.anchor_every, self.jump_m = anchor_every, jump_m
        self.clear()

    def clear(self):
        self.centres = self.lower = None
        self.cycle = 0
        self.forced = True

    def screen(self, centres, radii, field, margin):
        forced = self.centres is None or self.centres.shape != centres.shape or self.cycle % self.anchor_every == 0
        delta = None if forced else _centre_displacement(centres, self.centres)
        if delta is not None and np.any(delta > self.jump_m):
            forced = True
        if forced:
            self.lower = field.query(centres)-radii
            queried = len(centres)
            self.cycle = 0
        else:
            lower = self.lower-delta
            query = lower <= margin+field.error
            if query.any():
                lower[query] = field.query(centres[query])-radii[query]
            self.lower = lower
            queried = int(query.sum())
        self.centres = centres.copy()
        self.cycle += 1
        self.forced = forced
        return self.lower-field.error <= margin, queried


class TemporalPairBounds:
    """Each pair loses both current centre displacements before reuse."""
    def __init__(self):
        self.left = self.right = self.lower = None

    def screen(self, left, right, cap):
        a, b = left.clusters, right.clusters
        if self.lower is None or self.lower.shape != (len(a.centres), len(b.centres)):
            self.lower = np.linalg.norm(left.centres[:, None]-right.centres[None], axis=2)-a.radii[:, None]-b.radii[None]
        else:
            da = _centre_displacement(left.centres, self.left)
            db = _centre_displacement(right.centres, self.right)
            self.lower = self.lower-da[:, None]-db[None]
            i, j = np.nonzero(self.lower <= cap+1e-12)
            if len(i):
                self.lower[i, j] = np.linalg.norm(left.centres[i]-right.centres[j], axis=1)-a.radii[i]-b.radii[j]
        self.left, self.right = left.centres.copy(), right.centres.copy()
        return self.lower <= cap+1e-12


def _cluster_pair_distance(left, right, cap, metrics, temporal=None):
    from scipy.spatial import cKDTree
    a, b = left.clusters, right.clusters
    if np.linalg.norm(left.centre-right.centre)-a.radius-b.radius > cap:
        return cap
    # Spheres at both levels enclose every retained original point. A point
    # closer than cap must lie in a pair of clusters not excluded here.
    possible_a = np.zeros(len(a.centres), bool)
    possible_b = np.zeros(len(b.centres), bool)
    if temporal is not None:
        near = temporal.screen(left, right, cap)
        possible_a, possible_b = near.any(1), near.any(0)
    else:
        for start in range(0, len(a.centres), 128):
            distances = np.linalg.norm(left.centres[start:start+128, None]-right.centres[None], axis=2)
            near = distances-a.radii[start:start+128, None]-b.radii[None] <= cap
            possible_a[start:start+128] = near.any(1)
            possible_b |= near.any(0)
    ia, ib = a.point_indices(possible_a), b.point_indices(possible_b)
    if not len(ia) or not len(ib):
        return cap
    pa, pb = left.transform_points(ia), right.transform_points(ib)
    metrics['transformed_points'] += len(pa)+len(pb)
    metrics['exact_points'] += len(pa)+len(pb)
    # eps=0, one worker: exact query and no implicit parallelism.
    if len(pa) > len(pb):
        pa, pb = pb, pa
    distances = cKDTree(pb).query(pa, k=1, eps=0, workers=1)[0]
    return min(cap, float(distances.min()))


class RobotGeometry:
    def __init__(self, spec, side):
        self.arm = getattr(spec, side+'_arm')
        self.hand = getattr(spec, side+'_hand')
        self.chain = SerialChain(self.arm.urdf, self.arm.tip_link)
        self.base = pose_to_matrix(self.arm.base_pose)
        self.mount = pose_to_matrix(self.hand.mount)
        self.arm_points = {name: points for name, points in _link_points(self.arm.urdf).items()
                           if name != self.arm.base_link}
        self.hand_points = _link_points(self.hand.urdf)
        self.hand_chains = [SerialChain(self.hand.urdf, finger+'_distal')
                            for finger in ('thumb', 'index', 'middle', 'ring', 'pinky')]
        self.joint_links = {f'J{i+1}': joint['child'] for i, joint in enumerate(self.chain.joints) if joint['type'] != 'fixed'}
        self.joint_links['flange'] = self.arm.tip_link
        # Intersection of the J1/J2 lines in the base frame, from URDF origins.
        first, second = self.chain.joints[:2]
        t1 = first['origin']
        t2 = t1 @ second['origin']
        p1, p2 = t1[:3, 3], t2[:3, 3]
        a1, a2 = t1[:3, :3] @ first['axis'], t2[:3, :3] @ second['axis']
        factors = np.linalg.lstsq(np.c_[a1, -a2], p2-p1, rcond=None)[0]
        if np.linalg.norm(p1+factors[0]*a1-p2-factors[1]*a2) > 1e-6:
            raise ValueError('J1/J2 axes do not intersect')
        self.shoulder = _transform(p1+factors[0]*a1, self.base)
        self.arm_joints = self._compile(self.chain)
        seen = set()
        self.hand_joints = [joint for chain in self.hand_chains for joint in self._compile(chain, seen)]
        self.clustered = {name: ('arm', PointClusters.build(points)) for name, points in self.arm_points.items()}
        self.clustered.update({name: ('hand', PointClusters.build(points)) for name, points in self.hand_points.items()})


    @staticmethod
    def _compile(chain, seen=None):
        seen = set() if seen is None else seen
        result, parent = [], chain.root_link
        for joint in chain.joints:
            if joint['name'] not in seen:
                axis = joint['axis']/np.linalg.norm(joint['axis'])
                k = np.array([[0.,-axis[2],axis[1]], [axis[2],0.,-axis[0]], [-axis[1],axis[0],0.]])
                result.append((joint, parent, k, k@k))
                seen.add(joint['name'])
            parent = joint['child']
        return result

    @staticmethod
    def _evaluate(joints, values, root, poses, *, expand_mimic=True):
        for joint, parent, k, k2 in joints:
            pose = poses[parent] @ joint['origin']
            kind = joint['type']
            if kind != 'fixed':
                if joint['mimic'] and expand_mimic:
                    source, multiplier, offset = joint['mimic']
                    value = multiplier*values[source]+offset
                    values[joint['name']] = value
                else:
                    value = values[joint['name']]
                if kind in ('revolute', 'continuous'):
                    pose[:3, :3] = pose[:3, :3] @ (np.eye(3)+np.sin(value)*k+(1-np.cos(value))*k2)
                elif kind == 'prismatic':
                    axis = joint['axis']/np.linalg.norm(joint['axis'])
                    pose[:3, 3] += pose[:3, :3] @ (axis*value)
                else:
                    raise ValueError(f'unsupported joint type: {kind}')
            poses[joint['child']] = pose
        return poses

    def frames(self, arm_q, hand_q):
        arm = dict(zip(self.chain.active, _vector(arm_q, 7)))
        frames = self._evaluate(self.arm_joints, arm, self.base, {self.arm.base_link:self.base})
        palm = frames[self.arm.tip_link] @ self.mount
        hand = dict(zip(G20_JOINT_NAMES, _vector(hand_q, 21)))
        hand_frames = self._evaluate(self.hand_joints, hand, palm, {self.hand.root_link:palm}, expand_mimic=False)
        return frames, hand_frames

    def placed(self, side, frames, hand_frames):
        result = []
        for name, (kind, clusters) in self.clustered.items():
            pose = frames[name] if kind == 'arm' else hand_frames[name]
            result.append(PlacedSurface(side, name, kind, clusters, pose,
                _transform(clusters.centres, pose), _transform(clusters.centre, pose)))
        return result

    def world(self, arm_q, hand_q):
        # Full point transformation retained for static checks and the test oracle.
        frames, hand_frames = self.frames(arm_q, hand_q)
        parts = {name: ('arm', _transform(points, frames[name])) for name, points in self.arm_points.items()}
        parts.update({name: ('hand', _transform(points, hand_frames[name])) for name, points in self.hand_points.items()})
        return parts, frames


def _azimuth_range(shoulder, table):
    transform = pose_to_matrix(table.pose)
    x, y, z = table.half_extents
    corners = _transform(np.array([(a*x, b*y, z) for a in (-1, 1) for b in (-1, 1)]), transform)
    # An overhead shoulder admits all azimuths; test the projected polygon.
    xy = corners[[0, 2, 3, 1], :2]
    edge = np.roll(xy, -1, axis=0)-xy
    delta = shoulder[:2]-xy
    cross = edge[:, 0]*delta[:, 1]-edge[:, 1]*delta[:, 0]
    if np.all(cross >= -1e-12) or np.all(cross <= 1e-12):
        return (-180., 180.)
    az = np.degrees(np.arctan2(corners[:, 1]-shoulder[1], corners[:, 0]-shoulder[0]))
    # Unwrap around the direction of the table centre, including rotated tables.
    centre = np.degrees(np.arctan2(transform[1, 3]-shoulder[1], transform[0, 3]-shoulder[0]))
    az = centre+(az-centre+180)%360-180
    return float(az.min()), float(az.max())


def _pair_distance(a, b, cap):
    # Crop only points that can be within cap; far pairs need no dense product.
    lo, hi = b.min(0)-cap, b.max(0)+cap
    a = a[np.all((a >= lo) & (a <= hi), axis=1)]
    if not len(a):
        return cap
    lo, hi = a.min(0)-cap, a.max(0)+cap
    b = b[np.all((b >= lo) & (b <= hi), axis=1)]
    if not len(b):
        return cap
    best = cap*cap
    for start in range(0, len(a), 128):
        delta = a[start:start+128, None]-b[None]
        best = min(best, float(np.einsum('ijk,ijk->ij', delta, delta).min()))
    return best**.5


class WorkspaceLimits:
    def __init__(self, parameters, scene, *, cache_dir=None):
        self.parameters = copy.deepcopy(parameters)
        self.scene = scene
        self.margin = parameters['spatial']['margin_m']
        angular = parameters['angular']
        for layer in ('spatial', 'angular'):
            if parameters[layer]['enabled'] and self._has_null(parameters[layer]):
                raise ValueError(f'limits {layer} not configured: null parameter')
        self.robots = {side: RobotGeometry(scene.spec, side) for side in ('left', 'right')
                       if getattr(scene.spec, side+'_arm') is not None}
        self.obstacles = _fixtures(scene.spec)
        self.angular_range = {side: _azimuth_range(robot.shoulder, scene.spec.table) for side, robot in self.robots.items()}
        self.keep_out = []
        self.angular_enabled = angular['enabled']
        self.spatial_enabled = parameters['spatial']['enabled']
        if cache_dir is None:
            from dexstation.config.site import workspace_root
            cache_dir = workspace_root()/'cache/workspace_limits'
        self.cache_dir = Path(cache_dir)
        # A grid domain covering all reachable arm and finger surfaces. Queries
        # outside it still fall through to exact checks, never a blind pass.
        low, high = [], []
        for robot in self.robots.values():
            arm_reach = sum(np.linalg.norm(j['origin'][:3, 3]) for j in robot.chain.joints)
            arm_reach += max(np.linalg.norm(p, axis=1).max() for p in robot.arm_points.values())
            finger_reach = max(sum(np.linalg.norm(j['origin'][:3, 3]) for j in c.joints) for c in robot.hand_chains)
            finger_reach += max(np.linalg.norm(p, axis=1).max() for p in robot.hand_points.values())
            reach = arm_reach+np.linalg.norm(robot.mount[:3, 3])+finger_reach+.05
            low.append(robot.base[:3, 3]-reach)
            high.append(robot.base[:3, 3]+reach)
        self.field_lo, self.field_hi = np.min(low, axis=0), np.max(high, axis=0)
        self._rebuild_field()

    def _rebuild_field(self):
        obstacles = self.obstacles+self.keep_out if self.spatial_enabled else []
        self.field = DistanceField(obstacles, self.field_lo, self.field_hi, self.cache_dir)
        self._field_ids = tuple(id(o) for o in self.obstacles+self.keep_out)
        self.clear_temporal_cache()

    def clear_temporal_cache(self):
        self._temporal = TemporalBounds(anchor_every=30, jump_m=.05)
        self._temporal_pairs = {}

    def snapshot(self):
        # Geometry/configuration belong to RuntimeScene; acceleration has no
        # logical session state and is deliberately absent from recordings.
        return {}

    def restore(self, state=None):
        """Restore always discards acceleration history; decisions are unchanged."""
        self.clear_temporal_cache()


    @staticmethod
    def _has_null(value):
        if isinstance(value, dict):
            return any(WorkspaceLimits._has_null(v) for v in value.values())
        if isinstance(value, list):
            return any(WorkspaceLimits._has_null(v) for v in value)
        return value is None

    def with_task_keep_out(self, keep_out):
        from dexstation.profiles.compose import _frames
        frames = _frames(self.scene.workbench)
        result = copy.copy(self)
        result.keep_out = list(self.keep_out)
        for zone in keep_out:
            if zone['shape'] != 'box':
                raise ValueError('only box keep-outs are supported')
            pose = compose_pose(frames[zone['frame']], zone['pose'])
            result.keep_out.append(Obstacle.box('keep_out:'+zone['name'], pose, zone['half_extents_m']))
        result._rebuild_field()
        return result

    def _angular(self, side, frames):
        if not self.angular_enabled:
            return []
        p = self.parameters['angular']
        robot = self.robots[side]
        lo, hi = self.angular_range[side]
        result = []
        for joint in p['points']:
            v = frames[robot.joint_links[joint]][:3, 3]-robot.shoulder
            radius, horizontal = np.linalg.norm(v), np.linalg.norm(v[:2])
            if radius < p['exempt']['r_min_m']:
                continue
            az = float(np.degrees(np.arctan2(v[1], v[0])))
            elevation = float(np.degrees(np.arctan2(v[2], horizontal)))
            test_az = (lo+hi)/2+(az-(lo+hi)/2+180)%360-180
            if horizontal >= p['exempt']['h_min_m'] and hi-lo < 360 and not lo+p['azimuth']['margin_deg'] <= test_az <= hi-p['azimuth']['margin_deg']:
                result.append(AngularViolation(side, joint, az, elevation, 'azimuth'))
            if elevation > p['elevation']['max_up_deg']-p['elevation']['margin_deg']:
                result.append(AngularViolation(side, joint, az, elevation, 'elevation_up'))
        return result

    def _check(self, arm_q, hand_q, *, profile, temporal=True):
        metrics = dict(fk_s=0., angular_joint_s=0., cluster_screen_s=0., point_screen_s=0., exact_s=0., dual_s=0.,
                       transformed_points=0, transformed_centres=0, exact_points=0, temporal_skipped_centres=0, field_queried_centres=0, temporal_anchor=0)
        clock = time.perf_counter
        if tuple(id(o) for o in self.obstacles+self.keep_out) != self._field_ids:
            self._rebuild_field()
        spatial, angular, joint, bodies = [], [], {}, {}
        for side, robot in self.robots.items():
            t = clock() if profile else 0.
            q = _vector(arm_q[side], 7)
            frames, hand_frames = robot.frames(q, hand_q[side])
            if profile:
                metrics['fk_s'] += clock()-t
                t = clock()
            bounds = self.parameters['joint']['controller_soft_limits'][side]
            joint[side] = ([True]*7 if self._has_null(bounds) else
                           ((q >= np.radians(bounds['lower_deg'])) & (q <= np.radians(bounds['upper_deg']))).tolist())
            angular.extend(self._angular(side, frames))
            if profile:
                metrics['angular_joint_s'] += clock()-t
            if self.spatial_enabled:
                t = clock() if profile else 0.
                bodies[side] = robot.placed(side, frames, hand_frames)
                if profile:
                    metrics['cluster_screen_s'] += clock()-t
        if self.spatial_enabled:
            t = clock() if profile else 0.
            surfaces = [s for side in bodies.values() for s in side]
            sizes = [len(s.centres) for s in surfaces]
            centres = np.concatenate([s.centres for s in surfaces])
            radii = np.concatenate([s.clusters.radii for s in surfaces])
            if temporal:
                possible, queried = self._temporal.screen(centres, radii, self.field, self.margin)
                if self._temporal.forced:
                    self._temporal_pairs.clear()
                metrics['temporal_skipped_centres'] = len(centres)-queried
                metrics['temporal_anchor'] = int(self._temporal.forced)
            else:
                values = self.field.query(centres)
                possible = values-radii-self.field.error <= self.margin
                queried = len(centres)
            metrics['field_queried_centres'] = queried
            metrics['transformed_centres'] = len(centres)
            if profile:
                metrics['cluster_screen_s'] += clock()-t
            start = 0
            for surface, size in zip(surfaces, sizes):
                keep = possible[start:start+size]
                start += size
                if not keep.any():
                    continue
                t = clock() if profile else 0.
                indices = surface.clusters.point_indices(keep)
                points = surface.transform_points(indices)
                metrics['transformed_points'] += len(points)
                retained = self.field.query(points)-self.field.error <= self.margin
                points = points[retained]
                if profile:
                    metrics['point_screen_s'] += clock()-t
                if not len(points):
                    continue
                t = clock() if profile else 0.
                low, high = points.min(0), points.max(0)
                robot = self.robots[surface.side]
                for obstacle in self.obstacles+self.keep_out:
                    if surface.name == robot.chain.joints[0]['child'] and obstacle.name.split('#')[0] == 'station_column_mount_'+surface.side:
                        continue
                    if np.any(low > obstacle.hi+self.margin) or np.any(high < obstacle.lo-self.margin):
                        continue
                    near = np.all((points >= obstacle.lo-self.margin) & (points <= obstacle.hi+self.margin), axis=1)
                    if not near.any():
                        continue
                    metrics['exact_points'] += int(near.sum())
                    distance = obstacle.distance(points[near])
                    if distance < self.margin:
                        spatial.append(SpatialViolation(surface.side, surface.name, obstacle.name, distance))
                if profile:
                    metrics['exact_s'] += clock()-t
            if len(bodies) == 2:
                t = clock() if profile else 0.
                # Vectorized link-sphere screening before any cluster products.
                left, right = bodies['left'], bodies['right']
                gap = np.linalg.norm(np.array([s.centre for s in left])[:, None]-np.array([s.centre for s in right])[None], axis=2)
                gap -= np.array([s.clusters.radius for s in left])[:, None]
                gap -= np.array([s.clusters.radius for s in right])[None]
                for i, j in np.argwhere(gap <= self.margin):
                    a, b = left[i], right[j]
                    pair_cache = (self._temporal_pairs.setdefault((a.name, b.name), TemporalPairBounds())
                                  if temporal else None)
                    distance = _cluster_pair_distance(a, b, self.margin, metrics, pair_cache)
                    if distance < self.margin:
                        spatial.append(SpatialViolation('left', a.name, 'right:'+b.name, distance))
                if profile:
                    metrics['dual_s'] += clock()-t
        report = LimitReport(not spatial and not angular and all(all(v) for v in joint.values()),
                             spatial, angular, joint, dict(self.angular_range))
        return (report, metrics) if profile else report

    def check(self, arm_q, hand_q):
        return self._check(arm_q, hand_q, profile=False)

    def check_uncached(self, arm_q, hand_q):
        """Independent no-history path for validation; leaves temporal state alone."""
        return self._check(arm_q, hand_q, profile=False, temporal=False)

    def check_profiled(self, arm_q, hand_q, *, temporal=True):
        """Explicit diagnostic entry point; normal checks emit no telemetry."""
        return self._check(arm_q, hand_q, profile=True, temporal=temporal)


def load_limits(path, scene, *, cache_dir=None):
    from dexstation.profiles.compose import _schema
    parameters = json.loads(Path(path).read_text())
    _schema(parameters, 'limits')
    return WorkspaceLimits(parameters, scene, cache_dir=cache_dir)


@dataclass
class ReleaseDecision:
    arm_q: dict
    hand_q: dict
    arm_release: dict
    hand_release: dict
    reasons: dict
    report: LimitReport


def decide_release(limits, previous_arm_q, previous_hand_q, arm_q, hand_q):
    """Pure whole-side attribution followed by the two single-side experiments.

    The caller owns the previous legal state. Return reasons contain only
    serializable strings; report preserves the original candidate violations.
    """
    sides = tuple(previous_arm_q)
    arms = {s: _vector(arm_q[s], 7).copy() for s in sides}
    hands = {s: _vector(hand_q[s], 21).copy() for s in sides}
    old_a = {s: _vector(previous_arm_q[s], 7).copy() for s in sides}
    old_h = {s: _vector(previous_hand_q[s], 21).copy() for s in sides}
    arm_release, hand_release = dict.fromkeys(sides, True), dict.fromkeys(sides, True)
    reasons = {s: [] for s in sides}
    original = limits.check(arms, hands)

    def cross(v):
        return v.obstacle.startswith(('left:', 'right:'))

    for side in sides:
        robot = limits.robots[side]
        arm_bad = any(v.side == side for v in original.angular) or not all(original.joint.get(side, [True]*7))
        arm_bad |= any(v.side == side and not cross(v) and v.link in robot.arm_points for v in original.spatial)
        if arm_bad:
            arm_release[side] = hand_release[side] = False
            arms[side], hands[side] = old_a[side], old_h[side]
            reasons[side].append('arm_limit')
            continue
        if any(v.side == side and not cross(v) and v.link in robot.hand_points for v in original.spatial):
            # Distinguish a finger-only obstruction from the hand being carried
            # into a fixture by the new arm target (old fingers still collide).
            probe_h = dict(hands, **{side: old_h[side]})
            probe = limits.check(arms, probe_h)
            still_bad = any(v.side == side and not cross(v) for v in probe.spatial)
            if still_bad:
                arm_release[side] = hand_release[side] = False
                arms[side], hands[side] = old_a[side], old_h[side]
                reasons[side].append('arm_carries_hand_into_obstacle')
            else:
                hand_release[side] = False
                hands[side] = old_h[side]
                reasons[side].append('hand_limit')
    if any(cross(v) for v in limits.check(arms, hands).spatial):
        reject = []
        for side in sides:
            probe_a, probe_h = dict(old_a), dict(old_h)
            probe_a[side], probe_h[side] = arms[side], hands[side]
            if any(cross(v) for v in limits.check(probe_a, probe_h).spatial):
                reject.append(side)
        for side in (reject or sides):
            arm_release[side] = hand_release[side] = False
            arms[side], hands[side] = old_a[side], old_h[side]
            reasons[side].append('dual_limit')
    return ReleaseDecision({s: q.tolist() for s, q in arms.items()}, {s: q.tolist() for s, q in hands.items()}, arm_release, hand_release, reasons, original)


def _object_meshes(obj):
    """Selected collision geometry in the object root frame; no engine launch."""
    import trimesh
    from dexstation.config.paths import asset_path
    geometry = obj.geometry
    if 'box' in geometry:
        return [np.asarray(trimesh.creation.box(extents=2*np.asarray(geometry['box']['half_extents'])).triangles)]
    if 'mesh' in geometry:
        model = trimesh.load(asset_path(geometry['mesh']), force='mesh')
        return [np.asarray(model.triangles)]
    variant_links = None
    if obj.type == 'xbox_gamepad_physics':
        from xbox_gamepad_physics.resources import query_resource_path
        manifest_path = query_resource_path('robotwin', geometry['profile'])
        manifest = json.loads(manifest_path.read_text())
        urdf = query_resource_path('robotwin', geometry['profile'], manifest['entrypoint'])
        if geometry['collision_mesh'] == 'sdf':
            variant_links = manifest['collision_variants']['sdf']['links']
    else:
        urdf = Path(asset_path(geometry['urdf']))
    root = ET.parse(urdf).getroot()
    by_child = {joint.find('child').get('link'):joint for joint in root.findall('joint')}
    positions = obj.extra.get('initial_joint_positions', {})
    pieces = []
    for link in root.findall('link'):
        name = link.get('name')
        if name in by_child:
            chain = SerialChain(urdf, name)
            values = {j['name']:positions.get(j['name'], 0.) for j in chain.joints if j['type'] != 'fixed'}
            poses = RobotGeometry._evaluate(RobotGeometry._compile(chain), values, np.eye(4),
                {chain.root_link:np.eye(4)}, expand_mimic=False)
            transform = poses[name]
        else:
            transform = np.eye(4)
        if variant_links is not None:
            # Same replacement of URDF CoACD records as the pinned SAPIEN SDF
            # loader, but only reading triangle geometry for initial checking.
            for relative in variant_links.get(name, []):
                model = trimesh.load(Path(urdf).parent/relative, force='mesh')
                pieces.append(_transform(np.asarray(model.triangles), transform))
        else:
            for collision in link.findall('collision'):
                mesh = collision.find('geometry/mesh')
                if obj.type == 'xbox_gamepad_physics' and mesh is not None:
                    from dexstation.model.fixture_collision import convex_objects
                    path = Path(urdf).parent/mesh.get('filename')
                    scale = np.asarray([float(v) for v in mesh.get('scale', '1 1 1').split()])
                    for vertices,faces in convex_objects(path):
                        pieces.append(_transform(vertices[faces]*scale,transform @ _origin(collision)))
                else:
                    pieces.append(_transform(_collision_triangles(collision, Path(urdf).parent), transform))
    return pieces


def _initial_poses(scene, obj):
    from dexstation.profiles.compose import _frames, _sample_pose
    frames = _frames(scene.workbench)
    source = next(o for o in scene.task['objects'] if o['name'] == obj.name)
    nominal = copy.deepcopy(source)
    nominal.pop('randomize', None)
    poses = [tuple(obj.initial_pose_world), _sample_pose(nominal, frames, np.random.default_rng(0))]
    randomize = source.get('randomize') or {}
    xy = randomize.get('xy_m', [[0, 0], [0, 0]])
    yaw = randomize.get('yaw_deg', [0, 0])
    for x, y, angle in itertools.product(xy[0], xy[1], yaw):
        corner = copy.deepcopy(source)
        corner['randomize'] = dict(xy_m=[[x, x], [y, y]], yaw_deg=[angle, angle])
        poses.append(_sample_pose(corner, frames, np.random.default_rng(0)))
    return list(dict.fromkeys(poses))



class _InitialShape:
    """Exact closed-solid Boolean/gap queries, used only at scene validation.

    Manifold3d is already supplied by the pinned gamepad build/teleop extras.
    Mesh64 preserves metre coordinates; this does not alter runtime collision
    shapes or perform convex-hull/AABB substitutions.
    """
    def __init__(self, triangles):
        import trimesh
        self.triangles = np.asarray(triangles, float)
        self.vertices = self.triangles.reshape(-1, 3)
        self.lo, self.hi = self.vertices.min(0), self.vertices.max(0)
        self.mesh = trimesh.Trimesh(vertices=self.vertices,
            faces=np.arange(len(self.vertices)).reshape(-1, 3), process=True)
        self.mesh.fix_normals()
        self._solid = None

    def solid(self):
        if self._solid is None:
            import manifold3d
            if not self.mesh.is_watertight:
                raise ValueError('Initial collision geometry must be a closed solid')
            mesh = manifold3d.Mesh64(np.asarray(self.mesh.vertices, dtype=np.float64),
                                     np.asarray(self.mesh.faces, dtype=np.uint64))
            self._solid = manifold3d.Manifold(mesh)
            if self._solid.status() != manifold3d.Error.NoError:
                raise ValueError(f'Invalid initial collision solid: {self._solid.status()}')
        return self._solid

    def overlaps(self, other):
        if np.any(self.lo >= other.hi) or np.any(other.lo >= self.hi):
            return False  # disjoint or face contact
        intersection = self.solid() ^ other.solid()
        return not intersection.is_empty() and intersection.volume() > 0.

    def gap(self, other, maximum):
        if self.overlaps(other):
            return 0.
        return float(self.solid().min_gap(other.solid(), float(maximum)))


def _box_triangles(pose, half):
    import trimesh
    return _transform(np.asarray(trimesh.creation.box(extents=2*np.asarray(half)).triangles), pose_to_matrix(pose))


def _initial_environment(scene):
    """Workbench fixtures only; task static projections remain task objects."""
    import trimesh
    task_names = {o.name for o in scene.objects}
    parts = [('table', _InitialShape(_box_triangles(scene.spec.table.pose, scene.spec.table.half_extents)))]
    for fixture in scene.spec.fixtures:
        if fixture.name in task_names or not fixture.collision:
            continue
        if fixture.type == 'box':
            triangles = _box_triangles(fixture.pose, fixture.half_extents)
        else:
            # SAPIEN uses fixture.mesh, not the MJCF convex approximation.
            mesh = trimesh.load(fixture.mesh, force='mesh')
            triangles = _transform(np.asarray(mesh.triangles), pose_to_matrix(fixture.pose))
        parts.append((fixture.name, _InitialShape(triangles)))
    return parts


def _initial_robots(scene):
    """Initial collision solids from the repository's composed arm/hand URDF."""
    import tempfile
    from dexstation.model.robot_description import compose_robot_urdf
    result = []
    with tempfile.TemporaryDirectory(prefix='wp5-initial-robot-') as directory:
        for side in ('left', 'right'):
            arm = getattr(scene.spec, side+'_arm')
            if arm is None:
                continue
            hand = getattr(scene.spec, side+'_hand')
            path = compose_robot_urdf(scene.spec, Path(directory)/(side+'.urdf'), side)
            root = ET.parse(path).getroot()
            prefix = 'right_' if side == 'right' else ''
            positions = dict(zip(arm.joint_names, arm.home_q))
            positions.update(zip([prefix+n for n in G20_JOINT_NAMES], hand.home_q))
            base = pose_to_matrix(arm.base_pose)
            for link in root.findall('link'):
                if not link.findall('collision'):
                    continue
                name = link.get('name')
                if name == arm.base_link:
                    local = np.eye(4)
                else:
                    chain = SerialChain(path, name)
                    # Profile initial state, like observed hand_q, explicitly
                    # supplies every physical joint including mimic children.
                    poses = RobotGeometry._evaluate(RobotGeometry._compile(chain), dict(positions),
                        np.eye(4), {chain.root_link:np.eye(4)}, expand_mimic=False)
                    local = poses[name]
                for collision in link.findall('collision'):
                    triangles = _collision_triangles(collision, path.parent)
                    result.append((side+':'+name, _InitialShape(_transform(triangles, base @ local))))
    return result


def _collision_triangles(collision, directory):
    import trimesh
    geometry = collision.find('geometry')
    if geometry.find('mesh') is not None:
        node = geometry.find('mesh')
        path = Path(node.get('filename'))
        path = path if path.is_absolute() else Path(directory)/path
        scale = np.asarray([float(v) for v in node.get('scale', '1 1 1').split()])
        triangles = np.asarray(trimesh.load(path, force='mesh').triangles)*scale
    elif geometry.find('box') is not None:
        triangles = np.asarray(trimesh.creation.box(extents=[float(v) for v in geometry.find('box').get('size').split()]).triangles)
    elif geometry.find('sphere') is not None:
        raise ValueError('Initial sphere collision geometry requires an analytic solid adapter')
    else:
        raise ValueError('Unsupported initial collision geometry')
    return _transform(triangles, _origin(collision))


def validate_initial_state(scene):
    """§5.1.7: whole-geometry checks at nominal/sampled/random-corner poses.

    Closed-solid intersection catches containment and edge/face crossings, not
    just sampled surface points. Existing collision category switches are
    respected; visual-only task pieces never become collision obstacles.
    Table height and task keep-outs apply to every task piece. Contact is
    allowed, penetration is not. No hardware or physics engine is started.
    """
    violations = []
    if not scene.objects:
        return dict(checked=True, ok=True, violations=[])
    try:
        from dexstation.profiles.compose import _frames
        frames = _frames(scene.workbench)
        parameters = json.loads(Path(scene.limits_path).read_text())
        margin = parameters['spatial']['margin_m']
        zones = []
        for zone in scene.task['keep_out']:
            if zone['shape'] != 'box' or margin is None:
                raise ValueError('Initial keep-outs require a box and configured spatial margin')
            pose = compose_pose(frames[zone['frame']], zone['pose'])
            zones.append(('keep_out:'+zone['name'], _InitialShape(_box_triangles(pose, zone['half_extents_m']))))
        fixed = _initial_environment(scene)
        robots = _initial_robots(scene) if any(o.collision['robots'] for o in scene.objects) else []
        # Task static bodies are projected into spec.fixtures by WP2. Avoid
        # double-counting them as environment or overriding object-pair masks.
        static = []
        for obj in scene.objects:
            if obj.type == 'static' and obj.collision['objects']:
                for triangles in _object_meshes(obj):
                    static.append((obj.name, _InitialShape(_transform(triangles, pose_to_matrix(obj.initial_pose_world)))))
        table = pose_to_matrix(scene.spec.table.pose)
        up = table[:3, 2]
        top = table[:3, 3]+up*scene.spec.table.half_extents[2]
        for obj in scene.objects:
            try:
                pieces = _object_meshes(obj)
                for pose in _initial_poses(scene, obj):
                    for index, triangles in enumerate(pieces):
                        triangles = _transform(triangles, pose_to_matrix(pose))
                        shape = _InitialShape(triangles)
                        prefix = dict(object=obj.name, pose=list(pose), piece=index)
                        if ((shape.vertices-top) @ up).min() < -1e-12:
                            violations.append(dict(prefix, reason='below_table_top'))
                        checks = []
                        if obj.collision['environment']:
                            checks.extend((name, solid, 'fixed_penetration') for name,solid in fixed)
                        if obj.collision['objects']:
                            checks.extend((name,solid,'fixed_penetration') for name,solid in static if name != obj.name)
                        if obj.collision['robots']:
                            checks.extend((name,solid,'initial_robot_overlap') for name,solid in robots)
                        for name,solid,reason in checks:
                            if shape.overlaps(solid):
                                violations.append(dict(prefix, reason=reason, obstacle=name))
                                break
                        for name,solid in zones:
                            if np.linalg.norm(np.maximum(np.maximum(shape.lo-solid.hi, solid.lo-shape.hi), 0.)) >= margin:
                                continue
                            if shape.gap(solid, margin) < margin-1e-12:
                                violations.append(dict(prefix, reason='keep_out', obstacle=name))
            except (ValueError, KeyError, OSError, ImportError) as error:
                violations.append(dict(object=obj.name, reason='geometry_validation_failed', detail=str(error)))
    except (ValueError, KeyError, OSError, ImportError) as error:
        violations.append(dict(reason='geometry_validation_failed', detail=str(error)))
    return dict(checked=True, ok=not violations, violations=violations)
