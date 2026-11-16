"""Portable workbench + task exports and CPU-only offline checks (§9.3).

The online adapters are not registered here. Collision pairs implement reviewer
ruling 1; engine masks are a serialization of that explicit relation.
"""
from __future__ import annotations

from dexstation.contracts.versions import SCENE_FORMAT, SCENE_BUNDLE_FORMAT, SOLID_METRIC
from dataclasses import asdict
from dexstation.backends.object_types import object_type
import copy
import hashlib
import heapq
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np

from dexstation.contracts.profiles import RuntimeScene
from dexstation.recording.episode import FINGERS, SIDES, tip_pose
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES, apply_mimic_joint_positions
from dexstation.model.kinematics import _origin, axis_angle_to_matrix
from dexstation.config.paths import resolve_workspace_path
from dexstation.model.robot_description import compose_robot_urdf
from dexstation.model.spec import matrix_to_pose, pose_to_matrix

POSE_CONVENTION = 'world_from_frame:[x,y,z,qw,qx,qy,qz];m;rad'


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


COACD_VERSION = '1.0.14'
COACD_PARAMETERS = dict(threshold=.00025, real_metric=True, max_convex_hull=-1,
    preprocess_mode='off', preprocess_resolution=50, resolution=5000,
    mcts_nodes=25, mcts_iterations=200, mcts_max_depth=4, pca=False,
    merge=False, decimate=False, max_ch_vertex=256, extrude=False,
    extrude_margin=.01, apx_mode='ch', seed=0)
COACD_THRESHOLDS = (.00025, .0005, .001, .0015)


class _MeshSolid:
    def __init__(self, mesh):
        self.mesh = mesh

    def query(self, points, surface_distance=None):
        import trimesh
        distance = trimesh.proximity.closest_point(self.mesh, points)[1] if surface_distance is None else surface_distance
        inside = self.mesh.contains(points)
        signed = np.where(inside, -distance, distance)
        return np.maximum(signed, 0.), signed


class _HullUnionSolid:
    """Union membership by convex halfspaces; distance by all hull triangles.

    For an outside point the nearest point on the union is the nearest point
    on any hull. Internal faces need no boolean surface reconstruction.
    """
    def __init__(self, hulls):
        import trimesh
        from scipy.spatial import ConvexHull
        from rtree.index import Index
        self.mesh = trimesh.util.concatenate(hulls)
        self.planes = [ConvexHull(h.vertices).equations for h in hulls]
        self.boxes = Index(properties=__import__('rtree').index.Property(dimension=3))
        for index,hull in enumerate(hulls):
            self.boxes.insert(index, tuple(hull.bounds.ravel()))

    def query(self, points, surface_distance=None):
        import trimesh
        signed = np.full(len(points), np.inf)
        for index,point in enumerate(points):
            for hull in self.boxes.intersection(tuple(np.r_[point, point])):
                planes = self.planes[hull]
                residual = float(np.max(planes[:,:3]@point + planes[:,3]))
                if residual <= 0:
                    # An inscribed ball in any containing convex hull gives a
                    # valid interior-clearance bound for its entire union.
                    signed[index] = min(signed[index], residual)
        outside = np.isinf(signed)
        if np.any(outside):
            signed[outside] = (trimesh.proximity.closest_point(self.mesh, points[outside])[1]
                               if surface_distance is None else surface_distance[outside])
        return np.maximum(signed, 0.), signed


def _solid_surface_max(triangles, target, *, threshold=.002):
    """Continuous maximum external distance, certified by triangle bounds.

    f(c)+radius is the usual 1-Lipschitz bound. If c has certified interior
    clearance d, max(0,radius-d) is tighter: the ball B(c,d) is in the solid.
    No sampling-only decision and no boolean union surface enter this check.
    """
    import trimesh
    triangles = np.asarray(triangles)
    def evaluate(patches):
        centres=patches.mean(axis=1)
        values=np.zeros(len(patches)); upper=np.zeros(len(patches))
        for start in range(0,len(patches),128):
            patch=patches[start:start+128]; centre=centres[start:start+128]
            _,distance,faces=trimesh.proximity.closest_point(target.mesh,centre)
            boundary=target.mesh.triangles[faces]
            # Distance to a single target triangle is convex. Its maximum over
            # this patch is bounded by its three vertex distances. A target
            # boundary triangle belongs to the target solid, so this is also
            # an upper bound on distance to that solid, even if it is nonconvex.
            face_bound=np.zeros(len(patch))
            for vertex in range(3):
                near=trimesh.triangles.closest_point(boundary,patch[:,vertex,:])
                face_bound=np.maximum(face_bound,np.linalg.norm(near-patch[:,vertex,:],axis=1))
            unresolved=face_bound>threshold
            local=np.zeros(len(patch)); bounds=face_bound.copy()
            if np.any(unresolved):
                v,sign=target.query(centre[unresolved],surface_distance=distance[unresolved])
                radius=np.linalg.norm(patch[unresolved]-centre[unresolved,None,:],axis=2).max(axis=1)
                bounds[unresolved]=np.minimum(bounds[unresolved],np.maximum(0.,sign+radius))
                local[unresolved]=v
            values[start:start+len(patch)]=local;upper[start:start+len(patch)]=bounds
        return centres,values,upper
    centres,values,bounds=evaluate(triangles)
    heap=[(-float(bound),i,tri) for i,(bound,tri) in enumerate(zip(bounds,triangles))]
    heapq.heapify(heap)
    lower,sequence=float(values.max()),len(heap)
    witness=centres[int(values.argmax())].tolist()
    while -heap[0][0]>threshold and lower<=threshold and -heap[0][0]-lower>1e-10:
        parents=[]
        while heap and len(parents)<64 and -heap[0][0]>threshold:
            parents.append(heapq.heappop(heap)[2])
        parents=np.asarray(parents)
        edges=np.linalg.norm(parents-np.roll(parents,-1,axis=1),axis=2).argmax(axis=1)
        ordered=np.asarray([tri[[(int(e)+j)%3 for j in range(3)]] for tri,e in zip(parents,edges)])
        a,b,c=ordered[:,0],ordered[:,1],ordered[:,2];mid=(a+b)/2
        children=np.stack((np.stack((a,mid,c),axis=1),np.stack((mid,b,c),axis=1)),axis=1).reshape(-1,3,3)
        centres,values,bounds=evaluate(children)
        for child,centre,value,bound in zip(children,centres,values,bounds):
            if value>lower:
                lower,witness=float(value),centre.tolist()
            heapq.heappush(heap,(-float(bound),sequence,child));sequence+=1
    return dict(lower_m=lower,upper_m=max(lower,-heap[0][0]),witness_m=witness,
        method='1-Lipschitz branch-and-bound with interior-ball and target-triangle certificates',threshold_m=threshold)


def solid_collision_check(original, hulls):
    """Ruling 4's two directed solid errors; boolean union is volume only."""
    import trimesh
    if not original.is_watertight:
        raise ValueError('solid distance requires a watertight original mesh')
    hull_solid = _HullUnionSolid(hulls)
    outward = _solid_surface_max(hull_solid.mesh.triangles, _MeshSolid(original))
    coverage = _solid_surface_max(original.triangles, hull_solid)
    union = trimesh.boolean.union(hulls, engine='manifold')
    volume = abs(float(original.volume)); ratio = abs(float(union.volume))/volume
    return dict(metric=SOLID_METRIC, original_watertight=bool(original.is_watertight),
        original_contains_method=type(original.ray).__module__, outward=outward, coverage_gap=coverage,
        source_volume_m3=volume, decomposition_union_volume_m3=abs(float(union.volume)),
        decomposition_volume_ratio=ratio, hull_sum_volume_ratio=sum(abs(float(h.volume)) for h in hulls)/volume,
        ok=bool(outward['upper_m'] <= .002 and coverage['upper_m'] <= .002 and ratio <= 1.05))


def _max_surface_distance(surface, original, *, threshold=.002):
    """Bounds on max(surface -> original), including every surface triangle.

    Distance to a closed set is 1-Lipschitz. A centroid distance plus its
    triangle's covering radius bounds the entire patch, unlike point sampling.
    Stop on a certified threshold result, or a 0.5 mm numerical bracket.
    """
    import trimesh
    query = (trimesh.proximity.closest_point if importlib.util.find_spec('rtree')
             else trimesh.proximity.closest_point_naive)
    def distances(points):
        return np.concatenate([query(original, points[i:i+128])[1]
                               for i in range(0, len(points), 128)])
    triangles = surface.triangles
    centres = triangles.mean(axis=1)
    values = distances(centres)
    radii = np.linalg.norm(triangles-centres[:, None, :], axis=2).max(axis=1)
    heap = [(-float(d+r), i, tri) for i,(tri,d,r) in enumerate(zip(triangles,values,radii))]
    heapq.heapify(heap)
    sequence, lower = len(heap), float(values.max())
    while -heap[0][0] > threshold and lower <= threshold and -heap[0][0]-lower > .0005:
        _, _, tri = heapq.heappop(heap)
        a,b,c = tri
        ab,bc,ca = (a+b)/2, (b+c)/2, (c+a)/2
        children = np.array([[a,ab,ca], [ab,b,bc], [ca,bc,c], [ab,bc,ca]])
        for child,d in zip(children, distances(children.mean(axis=1))):
            radius = np.linalg.norm(child-child.mean(axis=0), axis=1).max()
            lower = max(lower, float(d))
            heapq.heappush(heap, (-float(d+radius), sequence, child))
            sequence += 1
    return dict(lower_m=lower, upper_m=-heap[0][0],
        method='1-Lipschitz triangle branch-and-bound on entire surface',
        reporting_bracket_m=.0005, threshold_m=threshold)


def convex_mesh_check(path):
    """Volume ratio and maximum hull-surface -> original-surface distance.

    A closed mesh whose every oriented face supports every vertex is convex:
    its boundary equals its convex hull (zero Hausdorff distance in exact
    arithmetic). Report floating plane residuals as well. For a nonconvex mesh,
    maximize distance by triangle branch-and-bound; distance is 1-Lipschitz,
    giving an upper bound from each patch's centroid and covering radius.
    """
    import trimesh
    mesh = trimesh.load(path, force='mesh')
    hull = mesh.convex_hull
    volume = abs(float(mesh.volume))
    ratio = float(hull.volume) / volume if volume > 0 else float('inf')
    eps = 128 * np.finfo(float).eps * max(1., float(np.max(mesh.extents)))
    normals = mesh.face_normals
    offsets = np.einsum('ij,ij->i', normals, mesh.triangles[:, 0])
    support = float(max(np.max(mesh.vertices[i:i+128] @ normals.T - offsets)
                        for i in range(0, len(mesh.vertices), 128)))
    convex = bool(mesh.is_watertight and mesh.is_winding_consistent and support <= eps)
    if convex:
        points = np.vstack((hull.vertices, hull.triangles_center))
        residual = float(max(np.max(np.abs(np.max(points[i:i+128] @ normals.T - offsets, axis=1)))
                             for i in range(0, len(points), 128)))
        lower, upper = 0., residual + eps
        method = 'closed supporting-plane convexity certificate; exact distance zero; upper includes floating residual'
    elif ratio <= 1.01:
        lower = 0.
        upper = float(np.linalg.norm(hull.vertices-mesh.vertices[0],axis=1).max())
        method = 'conservative diameter bound; volume branch already certifies initial OR classification'
    else:
        bounds = _max_surface_distance(hull, mesh)
        lower, upper, method = bounds['lower_m'], bounds['upper_m'], bounds['method']
    return dict(mesh_volume_m3=volume, hull_volume_m3=float(hull.volume), volume_ratio=ratio,
        convex_support_residual_m=support, convex_certificate=convex,
        hull_to_original_max_distance_lower_m=lower, hull_to_original_max_distance_upper_m=upper,
        distance_method=method, ok=bool(ratio <= 1.01 or upper <= .002))


def vendor_convex_fixture(fixture):
    """Reviewer ruling 2 applies only to meshes declared by the vendor manifest.

    Return None for other fixtures; their original construction policy owns
    their representation. Never apply this ruling to unlisted task supports.
    """
    source = Path(fixture.mesh)
    manifest = next((p/'collision_manifest.json' for p in source.parents
                     if (p/'collision_manifest.json').is_file()), None)
    if manifest is None:
        return None
    record = json.loads(manifest.read_text())
    part = next((part for part in record.get('parts', []) if part.get('file') == source.name), None)
    if part is None:
        return None
    if _hash(source) != part['sha256']:
        raise ValueError(f'vendor collision mesh differs from manifest: {source}')
    check = convex_mesh_check(source)
    if not check['ok']:
        raise ValueError(f'convexity acceptance gate failed: {fixture.name}: {check}')
    return dict(fixture=fixture.name, mesh=str(source), manifest_path=str(manifest),
        manifest_sha256=_hash(manifest), method=record['method'], part=part, check=check)


def _candidate_folder(threshold):
    return f'threshold_{threshold:.5f}'


def _read_candidate_cache(cache_root, fixture, parameters):
    if cache_root is None:
        return None
    root = Path(cache_root)/fixture.name
    for path in (root/_candidate_folder(parameters['threshold'])/'collision_check.json', root/'collision_check.json'):
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        if (record.get('source_sha256') == _hash(Path(fixture.mesh)) and
            record.get('coacd_version') == COACD_VERSION and record.get('parameters') == parameters and
            record.get('hull_files') and
            all(_hash(path.parent/h['file']) == h['sha256'] for h in record['hull_files'])):
            return path,record
    return None


def task_collision_candidate(fixture, output, threshold, *, cache_root=None):
    """One fixed-grid candidate. Failure is data for the constrained optimizer."""
    import trimesh
    start = time.perf_counter()
    if threshold not in COACD_THRESHOLDS:
        raise ValueError('threshold must belong to the reviewer-approved fixed grid')
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    parameters = dict(COACD_PARAMETERS,threshold=threshold)
    cached = _read_candidate_cache(cache_root,fixture,parameters)
    original = trimesh.load(fixture.mesh,force='mesh')
    record = dict(fixture=fixture.name,source_mesh=str(fixture.mesh),source_sha256=_hash(Path(fixture.mesh)),
        method='coacd',coacd_version=COACD_VERSION,parameters=parameters,seed=parameters['seed'],hull_files=[])
    if cached is not None:
        path,previous = cached
        for item in previous['hull_files']:
            target=output/Path(item['file']).name
            if target.resolve() != (path.parent/item['file']).resolve():
                shutil.copyfile(path.parent/item['file'],target)
            record['hull_files'].append(dict(file=target.name,sha256=_hash(target)))
        record.update(reused_from=str(path),coacd_elapsed_s=previous.get('coacd_elapsed_s'),
            hull_count=len(record['hull_files']))
        if previous.get('check',{}).get('metric') == SOLID_METRIC:
            record.update(check=previous['check'],metrics_elapsed_s=previous.get('metrics_elapsed_s'),
                elapsed_s=previous.get('elapsed_s'),reuse_elapsed_s=time.perf_counter()-start)
            _write(output/'collision_check.json',record)
            return record
        record['previous_surface_check'] = previous.get('check')
        hulls=[trimesh.load(output/h['file'],force='mesh',process=False) for h in record['hull_files']]
    else:
        import coacd
        version=importlib.metadata.version('coacd')
        if version != COACD_VERSION:
            raise ValueError(f'reproducible task export requires coacd=={COACD_VERSION}, got {version}')
        _write(output/'collision_check.json',dict(record,status='decomposing'))
        solver_start=time.perf_counter()
        parts=coacd.run_coacd(coacd.Mesh(original.vertices,original.faces),**parameters)
        record['coacd_elapsed_s']=time.perf_counter()-solver_start
        hulls=[trimesh.Trimesh(vertices=v,faces=f,process=False) for v,f in parts]
        for index,hull in enumerate(hulls):
            path=output/f'hull_{index:03d}.obj'
            path.write_text('\n'.join(['o hull_'+str(index)]+
                ['v '+' '.join(format(float(v),'.17g') for v in point) for point in hull.vertices]+
                ['f '+' '.join(str(int(i)+1) for i in face) for face in hull.faces])+'\n')
            record['hull_files'].append(dict(file=path.name,sha256=_hash(path)))
        record['hull_count']=len(hulls)
    _write(output/'collision_check.json',dict(record,status='checking'))
    metric_start=time.perf_counter()
    record['check']=solid_collision_check(original,hulls)
    record['metrics_elapsed_s']=time.perf_counter()-metric_start
    record['elapsed_s']=time.perf_counter()-start
    _write(output/'collision_check.json',record)
    return record


def task_fixture_collision(fixture, output, *, cache_root=None):
    """Ruling 4: minimize hull count subject to all three solid accuracy gates.

    Evaluate every approved candidate, including failures. Never enlarge the
    grid or tune other parameters after a failure. SAPIEN never calls here.
    """
    import trimesh
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    original=trimesh.load(fixture.mesh,force='mesh')
    convex_check=convex_mesh_check(fixture.mesh)
    record=dict(fixture=fixture.name,source_mesh=str(fixture.mesh),source_sha256=_hash(Path(fixture.mesh)),
        convex_check=convex_check,candidates=[],selected=None,hull_files=[])
    if convex_check['ok']:
        if convex_check['convex_certificate']:
            zero=dict(lower_m=0.,upper_m=convex_check['hull_to_original_max_distance_upper_m'],
                      method='closed supporting-plane convexity certificate')
            check=dict(metric=SOLID_METRIC,original_watertight=bool(original.is_watertight),
                original_contains_method=type(original.ray).__module__,outward=dict(zero),coverage_gap=dict(zero),
                decomposition_volume_ratio=convex_check['volume_ratio'],
                ok=bool(zero['upper_m'] <= .002 and convex_check['volume_ratio'] <= 1.05))
        else:
            check=solid_collision_check(original,[original.convex_hull])
        if check['ok']:
            target=output/('source'+Path(fixture.mesh).suffix)
            shutil.copyfile(fixture.mesh,target)
            record.update(method='source_convex',coacd_version=None,parameters=None,seed=None,hull_count=1,
                hull_files=[dict(file=target.name,sha256=_hash(target))],check=check)
            _write(output/'collision_check.json',record)
            return record
    results=[]
    for threshold in COACD_THRESHOLDS:
        folder=output/_candidate_folder(threshold)
        result=task_collision_candidate(fixture,folder,threshold,cache_root=cache_root)
        results.append((folder,result))
    feasible=[(folder,r) for folder,r in results if r['check']['ok']]
    # Hull count is the objective; finer threshold breaks equal-count ties.
    selected=min(feasible,key=lambda pair:(pair[1]['hull_count'],pair[1]['parameters']['threshold'])) if feasible else None
    for folder,result in results:
        summary={k:v for k,v in result.items() if k!='hull_files'}
        summary.update(selected=selected is not None and folder==selected[0],
            record_file=(folder/'collision_check.json').relative_to(output).as_posix())
        record['candidates'].append(summary)
    if selected is None:
        _write(output/'collision_check.json',record)
        raise ValueError(f'task decomposition acceptance gate failed: {fixture.name}: all approved candidates infeasible')
    folder,best=selected
    for key in ('method','coacd_version','parameters','seed','hull_count','check'):
        record[key]=best[key]
    record['selected']=best['parameters']['threshold']
    record['hull_files']=[dict(file=(folder/h['file']).relative_to(output).as_posix(),sha256=h['sha256']) for h in best['hull_files']]
    _write(output/'collision_check.json',record)
    return record


def gamepad_collision_mesh(obj):
    """Transition compatibility requested by reviewer; remove legacy at merge."""
    value = obj.geometry.get('collision_mesh')
    if value is None and isinstance(obj.collision, str):
        value = obj.collision
    if value is None:
        raise ValueError(f'{obj.name}: geometry.collision_mesh is required')
    return value


def collision_flags(obj):
    if isinstance(obj.collision, str):
        # Legacy fixed gamepad policy: robot + table, excluding all fixtures.
        return dict(robots=True, environment=True, objects=True)
    return {k: obj.collision[k] for k in ('robots', 'environment', 'objects')}


def allowed_collision_pairs(objects, fixtures=()):
    """Pure unordered pair relation, before any engine-specific encoding.

    Node names have namespaces so task names cannot alias station geometry.
    Fixture-gamepad exclusion is the existing policy, including task supports.
    """
    pairs = {tuple(sorted(('robot:left', 'robot:right'))),
             tuple(sorted(('robot:left', 'environment:table'))),
             tuple(sorted(('robot:right', 'environment:table')))}
    environment = ['environment:table'] + ['environment:' + f.name for f in fixtures if f.collision]
    for side, node in itertools.product(SIDES, environment):
        pairs.add(tuple(sorted(('robot:' + side, node))))
    for obj in objects:
        flags = collision_flags(obj)
        node = 'object:' + obj.name
        if flags['robots']:
            pairs.update(tuple(sorted((node, 'robot:' + side))) for side in SIDES)
        if flags['environment']:
            targets = environment[:1] if object_type(obj.type).environment_table_only else environment
            pairs.update(tuple(sorted((node, target))) for target in targets)
    for a, b in itertools.combinations(objects, 2):
        if collision_flags(a)['objects'] and collision_flags(b)['objects']:
            pairs.add(tuple(sorted(('object:' + a.name, 'object:' + b.name))))
    return frozenset(pairs)


def collision_masks(objects, fixtures=()):
    """Encode the relation; each task object gets an engine-local bit.

    Unique object bits also preserve gamepad's disabled internal collisions.
    """
    nodes = {'robot:left': 2, 'robot:right': 4, 'environment:table': 16}
    nodes.update({'environment:' + f.name: 8 for f in fixtures if f.collision})
    for index, obj in enumerate(objects):
        nodes['object:' + obj.name] = 1 << (5 + index)
    masks = {node: [bit, 0] for node, bit in nodes.items()}
    for a, b in allowed_collision_pairs(objects, fixtures):
        masks[a][1] |= nodes[b]
        masks[b][1] |= nodes[a]
    return masks


def configure_mujoco_collision(parent, scene, object_bodies):
    masks = collision_masks(scene.objects, scene.spec.fixtures)
    body_nodes = {body: 'object:' + name for name, bodies in object_bodies.items() for body in bodies}
    for geom in parent.geoms:
        if not (geom.contype or geom.conaffinity):
            continue
        body = geom.parent.name
        node = body_nodes.get(body)
        if node is None and body.startswith(('left_', 'right_')):
            node = 'robot:' + body.split('_', 1)[0]
        if node is None:
            node = 'environment:table' if geom.name == 'table' else next(
                ('environment:' + f.name for f in scene.spec.fixtures
                 if geom.name == f.name or geom.name.startswith(f.name + '_convex_')), None)
        if node is not None:
            geom.contype, geom.conaffinity = masks[node]


def add_mujoco_objects(parent, mj, objects, *, attachment=None):
    """Construction only; native primitive/URDF objects, no online task API."""
    return {obj.name: object_type(obj.type).invoke('mujoco_build', globals(),
        parent, mj, obj, attachment=attachment) for obj in objects}




def _add_mujoco_articulation(parent, mj, obj, *, attachment=None):
    geometry = obj.geometry
    source = resolve_workspace_path(geometry['urdf'])
    child = mj.MjSpec.from_file(source)
    frame = parent.worldbody.add_frame(pos=obj.initial_pose_world[:3], quat=obj.initial_pose_world[3:])
    parent.attach(child, prefix=obj.name + '_', frame=frame)
    return [b.name for b in parent.bodies if b.name.startswith(obj.name + '_')]


def _add_mujoco_primitive(parent, mj, obj, *, attachment=None):
    geometry = obj.geometry
    body = parent.worldbody.add_body(name=obj.name, pos=obj.initial_pose_world[:3],
                                    quat=obj.initial_pose_world[3:])
    if object_type(obj.type).mass:
        body.add_freejoint(name=obj.name + '_free')
    shape, parameters = next(iter(geometry.items()))
    types = dict(box=mj.mjtGeom.mjGEOM_BOX, sphere=mj.mjtGeom.mjGEOM_SPHERE,
                 cylinder=mj.mjtGeom.mjGEOM_CYLINDER, capsule=mj.mjtGeom.mjGEOM_CAPSULE,
                 mesh=mj.mjtGeom.mjGEOM_MESH)
    geom = dict(name=obj.name + '_collision', type=types[shape], contype=1, conaffinity=1)
    if shape == 'box':
        geom['size'] = parameters['half_extents']
    elif shape == 'sphere':
        geom['size'] = [parameters['radius_m'], 0, 0]
    elif shape in ('cylinder', 'capsule'):
        geom['size'] = [parameters['radius_m'], parameters['half_length_m'], 0]
    else:
        parent.add_mesh(name=obj.name + '_mesh', file=resolve_workspace_path(parameters['file']))
        geom['meshname'] = obj.name + '_mesh'
    if object_type(obj.type).mass:
        geom['mass'] = obj.inertial['mass_kg']
    if obj.material:
        # MuJoCo has a single Coulomb coefficient, unlike PhysX's pair.
        geom['friction'] = [obj.material['dynamic_friction'], .005, .0001]
    body.add_geom(**geom)
    return [obj.name]


class _Assets:
    def __init__(self, root):
        self.root, self.records = root, []
        self.copied = {}

    def copy(self, source, area):
        source = Path(resolve_workspace_path(str(source)))
        if any(p.is_symlink() for p in (source, *source.parents)):
            raise ValueError(f'asset must be a regular-file clone: {source}')
        source = source.resolve(strict=True)
        if source in self.copied:
            return self.copied[source]
        folder = self.root / area / hashlib.sha256(str(source.parent).encode()).hexdigest()[:16]
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / source.name
        shutil.copyfile(source, target)
        self.copied[source] = target
        self.records.append(dict(source=str(source), path=target.relative_to(self.root).as_posix(), sha256=_hash(target)))
        if source.suffix.lower() == '.glb' and source.with_suffix('.obj').is_file():
            self.copy(source.with_suffix('.obj'), area)
        if source.suffix.lower() in ('.obj', '.mtl'):
            for line in source.read_text().splitlines():
                parts = line.split()
                if parts and parts[0] in ('mtllib', 'map_Kd', 'map_Ks', 'map_Bump', 'bump'):
                    self.copy(source.parent / parts[-1], area)
        return target

    def urdf(self, source, target, area):
        source, target = Path(source), Path(target)
        tree = ET.parse(source)
        for mesh in [*tree.getroot().iter('mesh'), *tree.getroot().iter('texture')]:
            filename = Path(mesh.get('filename'))
            asset = self.copy(filename if filename.is_absolute() else source.parent / filename, area)
            mesh.set('filename', os.path.relpath(asset, target.parent))
        target.parent.mkdir(parents=True, exist_ok=True)
        tree.write(target, encoding='unicode')
        return target


def link_poses(urdf, q, base_pose):
    """Independent URDF FK, before either engine compiles the description."""
    root = ET.parse(urdf).getroot()
    joints = root.findall('joint')
    children = {j.find('child').get('link') for j in joints}
    base = next(link.get('name') for link in root.findall('link') if link.get('name') not in children)
    poses = {base: pose_to_matrix(base_pose)}
    pending = list(joints)
    while pending:
        ready = [j for j in pending if j.find('parent').get('link') in poses]
        if not ready:
            raise ValueError('URDF is not a rooted tree')
        for joint in ready:
            pose = poses[joint.find('parent').get('link')] @ _origin(joint)
            value = q.get(joint.get('name'), 0.)
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ') if joint.find('axis') is not None else np.array([1., 0, 0])
            motion = np.eye(4)
            if joint.get('type') in ('revolute', 'continuous'):
                motion[:3, :3] = axis_angle_to_matrix(axis, np.array([value]))[0]
            elif joint.get('type') == 'prismatic':
                motion[:3, 3] = value * axis / np.linalg.norm(axis)
            poses[joint.find('child').get('link')] = pose @ motion
            pending.remove(joint)
    return {name: matrix_to_pose(pose) for name, pose in poses.items()}


def _tip_offsets(spec):
    result = {}
    for side in SIDES:
        hand = Path(getattr(spec, side + '_hand').urdf)
        root = ET.parse(hand).getroot()
        result[side] = {}
        for finger in FINGERS:
            collision = root.find(f"link[@name='{finger}_distal']/collision")
            mesh = collision.find('geometry/mesh')
            if mesh is not None:
                import trimesh
                vertices = np.asarray(trimesh.load(hand.parent / mesh.get('filename'), force='mesh').vertices).copy()
                vertices *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                origin = collision.find('origin')
                if origin is not None:
                    vertices = vertices @ _origin(collision)[:3, :3].T + _origin(collision)[:3, 3]
                height = float(vertices[:, 2].max())
            else:
                # Primitive fixtures used in isolated contract tests.
                box = collision.find('geometry/box')
                height = float(box.get('size').split()[2]) / 2 + _origin(collision)[2, 3]
            result[side][finger] = [0., 0., height]
    return result


def _fk_reference(scene, robots, offsets):
    samples = []
    for index in range(3):
        record = dict(name=('home' if index == 0 else f'probe_{index}'), q={}, palm_pose_world={}, fingertip_pose_world={})
        for side in SIDES:
            robot = robots[side]
            root = ET.parse(robot['path']).getroot()
            joints = {j.get('name'): j for j in root.findall('joint') if j.get('type') != 'fixed'}
            q = dict(robot['home_q'])
            if index:
                for i, (name, joint) in enumerate(joints.items()):
                    if joint.find('mimic') is not None:
                        continue
                    limit = joint.find('limit')
                    q[name] += .05 * np.sin((i + 1) * index)
                    if limit is not None:
                        q[name] = float(np.clip(q[name], float(limit.get('lower')), float(limit.get('upper'))))
                local = 'right_' if side == 'right' else ''
                values = apply_mimic_joint_positions(side, [q[local + n] for n in G20_JOINT_NAMES])
                q.update(zip([local + n for n in G20_JOINT_NAMES], values))
            poses = link_poses(robot['path'], q, robot['base_pose'])
            record['q'][side] = q
            record['palm_pose_world'][side] = poses[robot['palm_link']]
            record['fingertip_pose_world'][side] = {finger: tip_pose(poses[link], offsets[side][finger]).tolist()
                for finger, link in robot['tip_links'].items()}
        samples.append(record)
    return dict(pose_convention=POSE_CONVENTION, source='independent composed-URDF FK', samples=samples)




def export_scene(scene: RuntimeScene, output, *, format, control_hz, resource_root=None, task_collision_cache=None):
    """Export a composed WP0 RuntimeScene. No hardware or online backend starts."""
    if format not in ('sapien', 'mjcf'):
        raise ValueError('format must be sapien or mjcf')
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError('control_hz must be explicitly supplied and positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = _Assets(output)
    convex_fixtures = []
    task_fixtures, fixture_meshes = [], {}
    bench_fixtures = {f['name'] for f in scene.workbench.get('geometry', {}).get('fixtures', [])}
    for fixture in scene.spec.fixtures:
        if fixture.type == 'mesh' and fixture.collision and fixture.convex_mesh is None:
            evidence = vendor_convex_fixture(fixture)
            if evidence is not None:
                evidence['bundled_manifest'] = assets.copy(evidence['manifest_path'], 'provenance').relative_to(output).as_posix()
                convex_fixtures.append(evidence)
            elif format == 'mjcf' and fixture.name not in bench_fixtures:
                folder = output/'objects/task_collision'/fixture.name
                record = task_fixture_collision(fixture, folder, cache_root=task_collision_cache)
                fixture_meshes[fixture.name] = [folder/h['file'] for h in record['hull_files']]
                record['directory'] = folder.relative_to(output).as_posix()
                task_fixtures.append(record)
    robots = {}
    for side in SIDES:
        arm, hand = getattr(scene.spec, side + '_arm'), getattr(scene.spec, side + '_hand')
        source = compose_robot_urdf(scene.spec, output / f'{side}_composed.urdf', side)
        target = assets.urdf(source, output / 'robots' / (side + '.urdf'), 'robots/meshes')
        source.unlink()
        prefix = 'right_' if side == 'right' else ''
        home = dict(zip(arm.joint_names, arm.home_q))
        home.update(zip([prefix + n for n in G20_JOINT_NAMES], hand.home_q))
        robots[side] = dict(path=str(target), urdf=target.relative_to(output).as_posix(), base_pose=list(arm.base_pose),
            home_q=home, palm_link=prefix + hand.root_link, tip_links={f: prefix + f + '_distal' for f in FINGERS},
            arm_joint_names=list(arm.joint_names), hand_mount=list(hand.mount), mount_status=hand.mount_status)
    offsets = _tip_offsets(scene.spec)
    fk = _fk_reference(scene, robots, offsets)
    _write(output / 'fk_check.json', fk)
    limits = json.loads(scene.limits_path.read_text())
    _write(output / 'limits.json', limits)
    objects = [asdict(o) for o in scene.objects]
    for obj in objects:
        geometry = obj['geometry']
        if 'urdf' in geometry:
            source = Path(resolve_workspace_path(geometry['urdf']))
            target = output / 'objects' / obj['name'] / 'object.urdf'
            assets.urdf(source, target, 'objects/meshes')
            geometry['urdf'] = target.relative_to(output).as_posix()
        if 'mesh' in geometry:
            geometry['mesh']['file'] = assets.copy(geometry['mesh']['file'], 'objects/meshes').relative_to(output).as_posix()
    spec = asdict(scene.spec)
    spec.pop('source_dir', None)
    for side in SIDES:
        for kind in ('arm', 'hand'):
            definition = spec[side + '_' + kind]
            source = Path(definition['urdf'])
            target = output / 'robots' / (side + '_' + kind + '_source.urdf')
            assets.urdf(source, target, 'robots/meshes')
            definition['urdf'] = target.relative_to(output).as_posix()
    for fixture in spec['fixtures']:
        for key in ('mesh', 'convex_mesh'):
            if fixture.get(key):
                fixture[key] = assets.copy(fixture[key], 'objects/fixtures').relative_to(output).as_posix()
    calibration_files = {}
    for name, value in scene.workbench.get('calibration', {}).items():
        if isinstance(value, str):
            source = Path(value)
            if not source.is_absolute():
                source = scene.limits_path.parent / source
            calibration_files[name] = json.loads(source.read_text())
    description = dict(format=SCENE_FORMAT, world_frame=scene.spec.world_frame, robots=robots,
        table=spec['table'], fixtures=spec['fixtures'], objects=objects, cameras=[asdict(c) for c in scene.cameras],
        physics=scene.workbench['physics'], fingertip_offsets=offsets)
    for robot in robots.values():
        robot.pop('path')
    resource_handlers = {}
    for obj in scene.objects:
        entry = object_type(obj.type)
        if 'export_resources' in entry.operations:
            resource_handlers.setdefault(entry.name, entry)
    for entry in resource_handlers.values():
        entry.invoke('export_resources', globals(), scene, resource_root, output, assets, description)
    _write(output / 'scene.json', description)
    _write(output / 'profiles.json', dict(workbench=scene.workbench, task=scene.task, frozen=scene.frozen))
    if format == 'mjcf':
        from dexstation.backends.mujoco_export import CollectionMujocoBackend
        backend = CollectionMujocoBackend(scene.spec, work_dir=output / 'build', resource_root=resource_root,
                                         export_only=True, runtime_scene=scene, fixture_meshes=fixture_meshes)
        tree = ET.parse(output / 'build/scene.xml')
        for element in tree.getroot().findall('asset/*[@file]'):
            asset = assets.copy(element.get('file'), 'objects/mjcf_assets')
            element.set('file', asset.relative_to(output).as_posix())
        tree.write(output / 'scene.mjcf.xml', encoding='unicode')
        inventory = mujoco_inventory(backend.model, backend.object_bodies)
        _write(output / 'inertia_corrections.json', json.loads((output / 'build/inertia_corrections.json').read_text()))
        shutil.rmtree(output / 'build')
    else:
        inventory = sapien_inventory(description, output)
    from dexstation.backends.mujoco_export import ARM_GAINS, HAND_GAINS
    bundle = dict(format=SCENE_BUNDLE_FORMAT, export_format=format, identity=scene.identity,
        calibration=scene.calibration, calibration_files=calibration_files,
        robots=robots, limits=limits, objects=objects, control_hz=control_hz,
        physics=scene.workbench['physics'], fingertip_offsets=offsets, pose_convention=POSE_CONVENTION,
        drives=dict(arm=ARM_GAINS, hand=HAND_GAINS,
            source='existing vendor simulation presets; not real robot K/D; SAPIEN hand rotor inertia 1e-4 adaptation'),
        equivalence='kinematics, installation, limits and object poses only; no dynamic equivalence',
        collision_pairs=sorted(map(list, allowed_collision_pairs(scene.objects, scene.spec.fixtures))),
        inventory=inventory, provenance=assets.records, spec=spec,
        task_keep_out=copy.deepcopy(scene.task.get('keep_out', [])), vendor_convex_fixtures=convex_fixtures,
        task_fixture_collision=task_fixtures)
    bundle['files'] = {p.relative_to(output).as_posix(): _hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    _write(output / 'bundle.json', bundle)
    return bundle


def mujoco_inventory(model, object_bodies):
    return dict(joints=[model.joint(i).name for i in range(model.njnt)],
        actuators=[model.actuator(i).name for i in range(model.nu)],
        bodies=[model.body(i).name for i in range(model.nbody)],
        geoms=[model.geom(i).name for i in range(model.ngeom)],
        meshes=[model.mesh(i).name for i in range(model.nmesh)],
        objects={name: list(bodies) for name, bodies in object_bodies.items()})


def _inventory_primitive(obj, description, root):
    return [], [obj['name']], []


def _inventory_urdf(source):
    tree = ET.parse(source).getroot()
    names = [j.get('name') for j in tree.findall('joint') if j.get('type') != 'fixed']
    return names, [link.get('name') for link in tree.findall('link')]


def _inventory_articulation(obj, description, root):
    names, links = _inventory_urdf(root / obj['geometry']['urdf'])
    return names, links, []




def sapien_inventory(description, root):
    joints = []
    for side, robot in description['robots'].items():
        joints += [side + ':' + j.get('name') for j in ET.parse(root / robot['urdf']).getroot().findall('joint') if j.get('type') != 'fixed']
    objects = {}
    gamepad_drives = []
    for obj in description['objects']:
        names, links, drives = object_type(obj['type']).invoke('inventory', globals(), obj, description, root)
        joints += [obj['name'] + ':' + name for name in names]
        gamepad_drives += [obj['name'] + ':' + name for name in drives]
        objects[obj['name']] = links
    return dict(joints=joints, actuators=[j for j in joints if j.split(':')[0] in SIDES] + gamepad_drives, objects=objects)


def _pose_error(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    position = float(np.linalg.norm(actual[:3] - reference[:3]))
    # atan2 of the relative rotation is stable near identity and ignores q/-q.
    rotation = pose_to_matrix(actual.tolist())[:3, :3].T @ pose_to_matrix(reference.tolist())[:3, :3]
    sine = np.linalg.norm([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]) / 2
    return position, float(np.arctan2(sine, (np.trace(rotation) - 1) / 2))


def _limit_check(root, bundle, reference):
    """Use WP5 when available; unconfigured layers remain explicitly disabled.

    FakeLimits is suitable for unit tests only, never evidence of geometric
    validation. Null placeholders are preserved in limits.json and the report.
    """
    limits = bundle['limits']
    status = {}
    for layer in ('spatial', 'angular'):
        status[layer] = 'disabled' if not limits[layer]['enabled'] else 'enabled'
    status['joint'] = 'disabled' if any(v is None for side in limits['joint']['controller_soft_limits'].values() for v in side.values()) else 'enabled'
    if all(value == 'disabled' for value in status.values()):
        return dict(ok=True, status=status, evaluation='disabled by supplied limits; no computed safety clearance', parameters=limits)
    # A configured layer must use the shared evaluator. Never silently substitute
    # a fake or introduce a second limit algorithm with different semantics.
    from dexstation.control.workspace_limits import load_limits
    from dexstation.model.spec import spec_from_dict
    description = json.loads((root / 'scene.json').read_text())
    spec_data = copy.deepcopy(bundle['spec'])
    for side in SIDES:
        for kind in ('arm', 'hand'):
            spec_data[side + '_' + kind]['urdf'] = str(root / spec_data[side + '_' + kind]['urdf'])
    spec = spec_from_dict(spec_data)
    evaluator = load_limits(root / 'limits.json', spec)
    evaluator = evaluator.with_task_keep_out(bundle['task_keep_out'])
    arm_q, hand_q = {}, {}
    for side in SIDES:
        arm = spec_data[side + '_arm']
        prefix = 'right_' if side == 'right' else ''
        arm_q[side] = [reference['q'][side][name] for name in arm['joint_names']]
        hand_q[side] = [reference['q'][side][prefix + name] for name in G20_JOINT_NAMES]
    return dict(ok=(report := evaluator.check(arm_q, hand_q)).ok, status=status,
                evaluation='shared WP5 evaluator', report=asdict(report))




def _load_export_articulation(scene, obj, root, desc, work_dir, sapien, physics_scene, physics_urdf):
    pose = sapien.Pose(obj.initial_pose_world[:3], obj.initial_pose_world[3:])
    loader = scene.create_urdf_loader()
    loader.fix_root_link = obj.extra['fixed_root']
    source = physics_urdf(root / obj.geometry['urdf'], work_dir / (obj.name + '.urdf'))
    actor = loader.load(str(source))
    actor.set_root_pose(pose)
    return actor, None


def _load_export_primitive(scene, obj, root, desc, work_dir, sapien, physics_scene, physics_urdf):
    pose = sapien.Pose(obj.initial_pose_world[:3], obj.initial_pose_world[3:])
    builder = scene.create_actor_builder()
    shape, parameters = next(iter(obj.geometry.items()))
    material = sapien.physx.PhysxMaterial(**obj.material) if obj.material else None
    kw = dict(material=material) if material else {}
    if object_type(obj.type).mass:
        if shape == 'box':
            volume = 8 * np.prod(parameters['half_extents'])
        elif shape == 'sphere':
            volume = 4 / 3 * np.pi * parameters['radius_m'] ** 3
        elif shape in ('cylinder', 'capsule'):
            radius, length = parameters['radius_m'], parameters['half_length_m']
            volume = 2 * np.pi * radius ** 2 * length
            if shape == 'capsule':
                volume += 4 / 3 * np.pi * radius ** 3
        else:
            import trimesh
            volume = trimesh.load(root / parameters['file'], force='mesh').convex_hull.volume
        kw['density'] = obj.inertial['mass_kg'] / volume
    if shape == 'box':
        builder.add_box_collision(half_size=parameters['half_extents'], **kw)
    elif shape == 'sphere':
        builder.add_sphere_collision(radius=parameters['radius_m'], **kw)
    elif shape in ('cylinder', 'capsule'):
        getattr(builder, 'add_' + shape + '_collision')(radius=parameters['radius_m'], half_length=parameters['half_length_m'], **kw)
    elif shape == 'mesh':
        builder.add_convex_collision_from_file(str(root / parameters['file']), **kw)
    if object_type(obj.type).mass and 'inertia_diag' in obj.inertial:
        builder.set_mass_and_inertia(obj.inertial['mass_kg'], sapien.Pose(), obj.inertial['inertia_diag'])
    actor = builder.build_static(name=obj.name) if object_type(obj.type).static else builder.build(name=obj.name)
    actor.set_pose(pose)
    return actor, None


def load_sapien_scene(root, *, work_dir):
    """Load the portable scene with CPU PhysX, no renderer/ROS/CAN/network."""
    import sapien
    from dexstation.backends.sapien_physics import _physics_urdf, _PhysicsScene, CollectionSapienBackend, ARM_GAINS, HAND_GAINS
    from dexstation.model.fixture_collision import add_sapien_fixture_collision, configure_sapien_shapes, sapien_shapes
    from dexstation.contracts.profiles import ObjectSpec
    from dexstation.model.spec import FixtureSpec
    root, work_dir = Path(root), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    desc = json.loads((root / 'scene.json').read_text())
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(desc['physics']['sapien']['timestep_s'])
    robots, objects, actors = {}, {}, {}
    table = scene.create_actor_builder()
    table.add_box_collision(half_size=desc['table']['half_extents'])
    actors['environment:table'] = table.build_static(name='table')
    actors['environment:table'].set_pose(sapien.Pose(desc['table']['pose'][:3], desc['table']['pose'][3:]))
    fixtures = []
    for item in desc['fixtures']:
        item = dict(item)
        for key in ('mesh', 'convex_mesh'):
            if item.get(key):
                item[key] = str(root / item[key])
        fixture = FixtureSpec(**item)
        fixtures.append(fixture)
        if not fixture.collision:
            continue
        builder = scene.create_actor_builder()
        add_sapien_fixture_collision(builder, fixture)
        actor = builder.build_static(name=fixture.name)
        actor.set_pose(sapien.Pose(fixture.pose[:3], fixture.pose[3:]))
        actors['environment:' + fixture.name] = actor
    for side, definition in desc['robots'].items():
        source = root / definition['urdf']
        physics = _physics_urdf(source, work_dir / (side + '.urdf'))
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(physics))
        robot.set_root_pose(sapien.Pose(definition['base_pose'][:3], definition['base_pose'][3:]))
        for joint in robot.get_active_joints():
            gains = ARM_GAINS if joint.name in definition['arm_joint_names'] else HAND_GAINS
            CollectionSapienBackend._configure(joint, gains)
            joint.set_drive_target(definition['home_q'][joint.name])
        robot.set_qpos([definition['home_q'][joint.name] for joint in robot.get_active_joints()])
        robot.set_qvel(np.zeros(len(robot.get_active_joints())))
        robots[side] = robot
    gamepad = None
    object_specs = [ObjectSpec(**obj) for obj in desc['objects']]
    for obj in object_specs:
        actor, asset = object_type(obj.type).invoke('sapien_export_load', globals(),
            scene, obj, root, desc, work_dir, sapien, _PhysicsScene(scene, work_dir), _physics_urdf)
        objects[obj.name] = actor
        if asset is not None:
            gamepad = asset
    masks = collision_masks(object_specs, fixtures)
    for node, actor in actors.items():
        configure_sapien_shapes(actor)
        for shape in sapien_shapes(actor):
            groups = list(shape.get_collision_groups())
            groups[:2] = masks[node]
            shape.set_collision_groups(groups)
    for side, robot in robots.items():
        for link in robot.get_links():
            configure_sapien_shapes(link)
            for shape in link.get_collision_shapes():
                groups = list(shape.get_collision_groups())
                groups[:2] = masks['robot:' + side]
                groups[2] |= 0x00010000 << SIDES.index(side)
                shape.set_collision_groups(groups)
    for name, actor in objects.items():
        links = actor.get_links() if hasattr(actor, 'get_links') else [actor]
        for link in links:
            configure_sapien_shapes(link)
            for shape in sapien_shapes(link):
                groups = list(shape.get_collision_groups())
                groups[:2] = masks['object:' + name]
                shape.set_collision_groups(groups)
    return scene, robots, objects, gamepad


def _sapien_actual_inventory(robots, objects, gamepad_names):
    joints = [side + ':' + j.name for side, robot in robots.items() for j in robot.get_active_joints()]
    task_joints = [name + ':' + j.name for name, obj in objects.items() if hasattr(obj, 'get_active_joints') for j in obj.get_active_joints()]
    gamepad_joints = [name + ':' + j.name for name, obj in objects.items()
        if name in gamepad_names
        for j in obj.get_active_joints()]
    return dict(joints=joints + task_joints, actuators=list(joints) + gamepad_joints, objects={name:
        [link.name for link in obj.get_links()] if hasattr(obj, 'get_links') else [name]
        for name, obj in objects.items()})


def _inventory_matches(actual, expected):
    # Engine traversal order is not part of §9.3's name/count contract.
    return actual.keys() == expected.keys() and all(
        ({name: sorted(bodies) for name, bodies in actual[key].items()} ==
         {name: sorted(bodies) for name, bodies in expected[key].items()})
        if key == 'objects' else sorted(actual[key]) == sorted(expected[key])
        for key in expected)


def validate_export(root, *, position_atol_m, rotation_atol_rad, work_dir):
    """No stepping trajectory: FK plus static contact detection at home only.

    Tolerances are explicit measured numerical budgets, not hidden defaults.
    Penetration means signed distance < 0 without changing the acceptance gate.
    """
    for value in (position_atol_m, rotation_atol_rad):
        if not np.isfinite(value) or value < 0:
            raise ValueError('FK tolerances must be finite and non-negative')
    root = Path(root).resolve()
    bundle = json.loads((root / 'bundle.json').read_text())
    reference = json.loads((root / 'fk_check.json').read_text())
    files_ok = all(_hash(root / name) == digest for name, digest in bundle['files'].items())
    max_position, max_rotation = 0., 0.
    contacts, object_errors = [], {}
    collision_export = dict(ok=True, hulls={}, minimum_count_ok=True)
    if bundle['export_format'] == 'mjcf':
        import mujoco as mj
        model = mj.MjModel.from_xml_path(str(root / 'scene.mjcf.xml'))
        data = mj.MjData(model)
        expected = bundle['inventory']
        inventory = mujoco_inventory(model, expected['objects'])
        for record in bundle.get('task_fixture_collision', []):
            prefix = record['fixture']+'_convex_'
            loaded = sum(name.startswith(prefix) for name in inventory['geoms'])
            collision_export['hulls'][record['fixture']] = dict(expected=record['hull_count'],loaded=loaded)
            collision_export['ok'] &= loaded == record['hull_count']
            if record['method'] == 'coacd':
                candidates = record['candidates']
                feasible = [c['hull_count'] for c in candidates if c['check']['ok']]
                grid = sorted(c['parameters']['threshold'] for c in candidates)
                optimum = bool(feasible and record['hull_count'] == min(feasible) and grid == list(COACD_THRESHOLDS))
                collision_export['minimum_count_ok'] &= optimum
                collision_export['ok'] &= optimum
        # Object body presence is checked independently against loaded names.
        object_inventory_ok = all(body in inventory['bodies'] for bodies in expected['objects'].values() for body in bodies)
        for sample in reversed(reference['samples']):
            mj.mj_resetData(model, data)
            for side, q in sample['q'].items():
                for name, value in q.items():
                    data.qpos[model.joint(side + '_' + name).qposadr[0]] = value
            mj.mj_forward(model, data)
            for side, robot in bundle['robots'].items():
                names = {'palm': robot['palm_link'], **robot['tip_links']}
                for key, name in names.items():
                    body = model.body(side + '_' + name).id
                    pose = np.r_[data.xpos[body], data.xquat[body]]
                    if key != 'palm':
                        pose = tip_pose(pose, bundle['fingertip_offsets'][side][key])
                    target = sample['palm_pose_world'][side] if key == 'palm' else sample['fingertip_pose_world'][side][key]
                    p, r = _pose_error(pose, target)
                    max_position, max_rotation = max(max_position, p), max(max_rotation, r)
        # Reverse iteration leaves data at the required home pose.
        for contact in data.contact:
            if contact.dist < 0:
                contacts.append(dict(geom1=model.geom(contact.geom[0]).name, geom2=model.geom(contact.geom[1]).name, distance_m=float(contact.dist)))
        for obj in bundle['objects']:
            body = expected['objects'][obj['name']][0]
            i = model.body(body).id
            object_errors[obj['name']] = _pose_error(np.r_[data.xpos[i], data.xquat[i]], obj['initial_pose_world'])
        timestep = float(model.opt.timestep)
    else:
        import sapien
        from dexstation.backends.sapien_physics import _world_pose
        scene, robots, objects, gamepad = load_sapien_scene(root, work_dir=work_dir)
        inventory = _sapien_actual_inventory(robots, objects,
            {obj['name'] for obj in bundle['objects'] if object_type(obj['type']).legacy_gamepad})
        object_inventory_ok = True
        for sample in reversed(reference['samples']):
            for side, robot in robots.items():
                robot.set_qpos([sample['q'][side][j.name] for j in robot.get_active_joints()])
                for joint in robot.get_active_joints():
                    joint.set_drive_target(sample['q'][side][joint.name])
                links = {link.name: link for link in robot.get_links()}
                definition = bundle['robots'][side]
                actual = {'palm': _world_pose(links[definition['palm_link']])}
                actual.update({finger: tip_pose(_world_pose(links[name]), bundle['fingertip_offsets'][side][finger])
                    for finger, name in definition['tip_links'].items()})
                for key, pose in actual.items():
                    target = sample['palm_pose_world'][side] if key == 'palm' else sample['fingertip_pose_world'][side][key]
                    p, r = _pose_error(pose, target)
                    max_position, max_rotation = max(max_position, p), max(max_rotation, r)
        # PhysX contact generation needs one scene step. Freeze current generalized
        # positions with zero dt is not supported; inspect contacts at the first
        # simulation solve, and report the actual timestep in the evidence.
        scene.step()
        for contact in scene.get_contacts():
            for point in contact.points:
                if point.separation < 0:
                    contacts.append(dict(body1=contact.bodies[0].entity.name, body2=contact.bodies[1].entity.name, distance_m=float(point.separation)))
        for obj in bundle['objects']:
            actor = objects[obj['name']]
            pose = actor.get_root_pose() if hasattr(actor, 'get_root_pose') else actor.get_pose()
            object_errors[obj['name']] = _pose_error([*pose.p, *pose.q], obj['initial_pose_world'])
        timestep = float(scene.get_timestep())
    expected = bundle['inventory']
    inventory_ok = object_inventory_ok and _inventory_matches(inventory, expected)
    limit_report = _limit_check(root, bundle, reference['samples'][0])
    fk_ok = max_position <= position_atol_m and max_rotation <= rotation_atol_rad
    poses_ok = all(p <= position_atol_m and r <= rotation_atol_rad for p, r in object_errors.values())
    return dict(ok=files_ok and inventory_ok and fk_ok and poses_ok and not contacts and limit_report['ok'] and collision_export['ok'],
        files_ok=files_ok, inventory_ok=inventory_ok, inventory=inventory,
        fk=dict(ok=fk_ok, max_position_error_m=max_position, max_rotation_error_rad=max_rotation,
            position_atol_m=position_atol_m, rotation_atol_rad=rotation_atol_rad),
        object_pose_errors=object_errors, object_poses_ok=poses_ok,
        home=dict(ok=not contacts, penetrations=contacts), limits=limit_report, timestep_s=timestep,
        engine=bundle['export_format'], collision_export=collision_export)
