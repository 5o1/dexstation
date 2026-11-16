"""Operator-only cameras and a nonoverlapping atlas, shared by live/offline views.

Task cameras use object-local vectors and never enter the collection camera
list or episode camera schema.
"""
import math


TILE_WIDTH, TILE_HEIGHT = 960, 540
AUXILIARY_FPS = 20
# Only the core camera is fixed here. Task camera IDs/slots come from data.
AUXILIARY_VIEWS = ('table',)
ATLAS_WIDTH, ATLAS_HEIGHT = TILE_WIDTH * 2, TILE_HEIGHT * 3
FOVY = math.radians(50)
NEAR = .001


def camera_declarations(backend=None, cfg=None):
    payload = (cfg or {}).get('runtime_scene')
    runtime = getattr(backend, 'runtime_scene', None)
    task = payload['task'] if payload is not None else runtime.task if runtime is not None else {}
    return sorted(task.get('auxiliary_cameras', []), key=lambda c: c['atlas_slot'])


def available_views(backend=None, cfg=None):
    return tuple(c['id'] for c in camera_declarations(backend, cfg)) + AUXILIARY_VIEWS


def selected_views(values, available=AUXILIARY_VIEWS):
    """Validate demand and retain the declaration's fixed atlas order."""
    if not isinstance(values, list) or any(v not in available for v in values):
        raise ValueError('expected a list of auxiliary camera names')
    return [name for name in available if name in values]


# World-fixed table view: straight down from above the table, image top toward
# the gamepad (+x), shifted toward the robot so the area behind the arms stays
# in frame. Values are the operator_view.yaml `overview` camera the user first
# approved; that config section stays the source when present.
TABLE_VIEW = dict(position=(0.55, 0.6202, 2.60), target=(0.55, 0.6202, 0.767), up=(1., 0., 0.))


def table_vectors(cfg=None):
    import numpy as np
    source = dict(TABLE_VIEW)
    source.update({k: v for k, v in ((cfg or {}).get('overview') or {}).items() if k in TABLE_VIEW})
    return tuple(np.asarray(source[k], dtype=float) for k in ('position', 'target', 'up'))


def camera_vectors(transform, camera):
    """Transform declarative object-local position, target and up vectors."""
    import numpy as np
    position, target, up = (camera[k] for k in ('position', 'target', 'up'))
    def world(point):
        return (transform @ np.r_[point, 1.])[:3]
    return world(position), world(target), transform[:3, :3] @ up


def auxiliary_options(cfg=None):
    values = dict(width=ATLAS_WIDTH, height=ATLAS_HEIGHT, fps=AUXILIARY_FPS,
                  framing_margin=1.12, background_color='#3a4048')
    values.update((cfg or {}).get('auxiliary', {}))
    for key in ('width', 'height', 'fps'):
        if type(values[key]) is not int:
            raise ValueError('auxiliary dimensions and fps must be integers')
    if values['width'] < ATLAS_WIDTH or values['height'] < ATLAS_HEIGHT or values['width'] % 64 or values['height'] * 32 != values['width'] * 27 or not 1 <= values['fps'] <= 60:
        raise ValueError('auxiliary stream requires an even 32:27 atlas >=1920x1620, width divisible by 64, and 1..60 fps')
    margin = values['framing_margin']
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin <= 1:
        raise ValueError('auxiliary framing_margin must be finite and greater than 1')
    color = values['background_color']
    if not isinstance(color, str) or len(color) != 7 or color[0] != '#' or any(c not in '0123456789abcdefABCDEF' for c in color[1:]):
        raise ValueError('auxiliary background_color must be #RRGGBB')
    return values


def view_basis(position, target, up):
    """Match look_at: image right, image up, optical forward."""
    import numpy as np
    forward = np.asarray(target) - np.asarray(position)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(right, forward), forward))


def fit_camera(vertices, vectors, width, height, margin=1.12):
    """Exact tangent fit of every world-space vertex, with fixed direction/up.

    At depth z relative to the target, |x-cx| <= tan_x * (d+z).
    The two opposing support planes give the feasible interval for cx (and cy).
    Its first nonempty distance is the exact minimum; no sphere/AABB proxy.
    Target depth is the midpoint of the mesh's depth range. The transverse
    center bisects opposing tangent supports, maximizing clearance at d_min.
    """
    import numpy as np
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('framing requires finite, nonempty mesh vertices (N,3)')
    basis = view_basis(*vectors)
    # Subtract a nearby origin to retain precision under world translations.
    origin = points[0]
    local = (points - origin) @ basis
    z0 = (local[:, 2].min() + local[:, 2].max()) / 2
    z = local[:, 2] - z0
    ty = math.tan(FOVY / 2)
    tangents = (ty * width / height, ty)
    center, distances = [], []
    for axis, tangent in enumerate(tangents):
        upper = np.max(local[:, axis] - tangent*z)
        lower = np.min(local[:, axis] + tangent*z)
        center.append((upper + lower) / 2)
        distances.append((upper - lower) / (2*tangent))
    d_min = float(max(*distances, NEAR - z.min()))
    target = origin + basis @ np.array([*center, z0])
    distance = d_min * margin
    position = target - basis[:, 2]*distance
    return (position, target, vectors[2]), dict(d_min=d_min, distance=distance,
        margin=margin, **projection_bounds(points, (position, target, vectors[2]), width, height))


def projection_bounds(vertices, vectors, width, height):
    """Continuous pixel-edge coordinates, before occlusion/rasterization."""
    import numpy as np
    local = (np.asarray(vertices) - vectors[0]) @ view_basis(*vectors)
    fy = height / (2*math.tan(FOVY/2))
    uv = local[:, :2] / local[:, 2:3] * (fy, -fy) + (width/2, height/2)
    low, high = uv.min(axis=0), uv.max(axis=0)
    return dict(margins_px=dict(left=float(low[0]), right=float(width-high[0]),
        top=float(low[1]), bottom=float(height-high[1])),
        min_depth=float(local[:, 2].min()), max_depth=float(local[:, 2].max()))


class ObjectRenderMesh:
    """Read the actual attached render meshes once; apply current link poses."""
    def __init__(self, task_object):
        import numpy as np
        from sapien.render import RenderBodyComponent
        self.parts = []
        for link in task_object.links():
            entity = getattr(link, 'entity', link)
            body = entity.find_component_by_type(RenderBodyComponent)
            if body is None:
                continue
            for shape in body.render_shapes:
                pose = shape.local_pose.to_transformation_matrix()
                for part in shape.parts:
                    points = np.asarray(part.vertices, dtype=float) * np.asarray(shape.scale)
                    self.parts.append((entity, points @ pose[:3, :3].T + pose[:3, 3]))
        if not self.parts:
            raise ValueError('auxiliary framing requires attached object render meshes')

    def world_vertices(self):
        import numpy as np
        parts = []
        for entity, points in self.parts:
            pose = entity.pose.to_transformation_matrix()
            parts.append(points @ pose[:3, :3].T + pose[:3, 3])
        return np.concatenate(parts)


def background_entity_ids(backend):
    """Environmental surfaces, excluding the arms, hands and working fixture."""
    return (backend.table.per_scene_id, *(f.per_scene_id for f in backend.fixtures
        if f.name.startswith(('station_floor', 'station_backdrop', 'station_table_frame'))
        or f.name.endswith('table_apron')))


def auxiliary_rgb(camera, cfg=None, background_ids=()):
    """Composite auxiliary background surfaces; never mutate the render scene.

    SAPIEN Segmentation channel 1 is entity ID; zero denotes no surface.
    Working fixtures, robots, task objects and overlays retain their rendered pixels.
    This function is used only for auxiliary output, never camera_3/recording.
    """
    import numpy as np
    from dexstation.cameras.image import color_to_rgb8
    rgb = color_to_rgb8(camera.get_picture('Color'))
    empty = np.isin(np.asarray(camera.get_picture('Segmentation'))[..., 1], (0, *background_ids))
    color = auxiliary_options(cfg)['background_color']
    rgb[empty] = [int(color[i:i+2], 16) for i in (1, 3, 5)]
    return rgb


class AuxiliaryFraming:
    """Shared live/snapshot framing in each output's actual aspect ratio."""
    def __init__(self, backend, cfg=None):
        self.backend = backend
        self.declarations = {c['id']: c for c in camera_declarations(backend, cfg)}
        self.margin = (cfg or {}).get('auxiliary', {}).get('framing_margin')
        self.table = table_vectors(cfg)
        self.meshes = {}

    def vectors(self, name, width, height):
        if name == 'table':
            return self.table
        from dexstation.model.spec import pose_to_matrix
        camera = self.declarations[name]
        obj = self.backend.objects[camera['object']]
        # Use the same composed root pose as the original fixed-root camera
        # path. Mesh vertices still use each link's current rendered pose.
        transform = pose_to_matrix(obj.definition.initial_pose_world)
        vectors = camera_vectors(transform, camera)
        if not camera['fit_render_meshes']:
            return vectors
        if camera['object'] not in self.meshes:
            self.meshes[camera['object']] = ObjectRenderMesh(obj)
        mesh = self.meshes[camera['object']]
        margin = self.margin if self.margin is not None else camera['framing_margin']
        return fit_camera(mesh.world_vertices(), vectors, width, height, margin)[0]


def frame_capacity(width, height):
    return width * height * 3 + 65536


def scaled_camera_definition(definition, width, height):
    from dataclasses import replace
    intr = definition.intrinsics
    sx, sy = width / intr.width, height / intr.height
    return replace(definition, intrinsics=replace(intr, width=width, height=height,
        fx=intr.fx*sx, fy=intr.fy*sy, cx=intr.cx*sx, cy=intr.cy*sy))


def atlas_tiles(cfg=None, backend=None):
    options = auxiliary_options(cfg)
    width, height = options['width'], options['height']
    # Keep the 2x3 wire format and Table top's slot 5. Undeclared slots stay
    # black; pane weights never move camera pixels within the atlas.
    slots = [(c['id'], c['atlas_slot']) for c in camera_declarations(backend, cfg)] + [('table', 5)]
    return {name: dict(x=(slot % 2)*width//2, y=(slot // 2)*height//3,
                       width=width//2, height=height//3)
            for name, slot in slots}


class AuxiliaryCameras:
    """Lazy camera creation; disabled cameras never take a picture."""
    def __init__(self, backend, cfg=None):
        self.backend = backend
        self.cfg = cfg or {}
        self.options = auxiliary_options(self.cfg)
        self.framing = AuxiliaryFraming(backend, cfg)
        self.background_ids = background_entity_ids(backend)
        self.cameras = {}

    def capture(self, names):
        import numpy as np
        from xbox_gamepad_physics.examples.scene import look_at
        atlas = np.zeros((self.options['height'], self.options['width'], 3), dtype=np.uint8)
        tiles = atlas_tiles(self.cfg, self.backend)
        selected = selected_views(names, available_views(self.backend, self.cfg))
        for name in selected:
            tile = tiles[name]
            if name not in self.cameras:
                camera = self.backend.scene.add_camera('aux_' + name, tile['width'], tile['height'],
                                                       FOVY, NEAR, 20.)
                self.cameras[name] = camera
            camera = self.cameras[name]
            camera.entity.set_pose(look_at(*self.framing.vectors(name, tile['width'], tile['height'])))
        if selected:
            # Publish all camera creations/poses once before taking pictures.
            self.backend.scene.update_render()
        for name in selected:
            camera = self.cameras[name]
            camera.take_picture()
            tile = tiles[name]
            atlas[tile['y']:tile['y'] + tile['height'], tile['x']:tile['x'] + tile['width']] = auxiliary_rgb(camera, self.cfg, self.background_ids)
        return atlas
