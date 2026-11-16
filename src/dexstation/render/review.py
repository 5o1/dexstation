"""Camera and visual helpers shared by the maintained rendering pipeline."""
import math

import numpy as np

from dexstation.model.spec import optical_to_sapien_pose, compose


def camera_world_optical(camera_spec, backend):
    if camera_spec.parent == backend.spec.world_frame:
        return camera_spec.pose_optical
    owner, link = camera_spec.parent.split(":", 1)
    side, kind = owner.split("_", 1)
    if kind == "hand" and side == "right":
        link = "right_" + link
    return compose(backend.link_pose(link), camera_spec.pose_optical)


def create_spec_camera(backend, camera_spec):
    """Use explicit K and optical centre, never a look-at/FOV approximation."""
    import sapien
    intr = camera_spec.intrinsics
    fovy = 2*math.atan(intr.height/(2*intr.fy))
    if camera_spec.parent == backend.spec.world_frame:
        camera = backend.scene.add_camera(camera_spec.name, intr.width, intr.height, fovy, .001, 20.)
        pose = optical_to_sapien_pose(camera_spec.pose_optical)
        camera.entity.set_pose(sapien.Pose(pose[:3],pose[3:]))
    else:
        pose = optical_to_sapien_pose(camera_spec.pose_optical)
        camera = backend.scene.add_mounted_camera(camera_spec.name,
                    backend.camera_parent_entity(camera_spec.parent),
                    sapien.Pose(pose[:3],pose[3:]), intr.width,intr.height,fovy,.001,20.)
    camera.set_perspective_parameters(.001,20.,intr.fx,intr.fy,intr.cx,intr.cy,0.)
    return camera


def mask_iou(real, simulated):
    union = np.count_nonzero(real | simulated)
    return float(np.count_nonzero(real & simulated) / union) if union else None


def robot_only_review_visuals(backend):
    """Hide current-epoch fixtures for old-epoch robot-only image comparison.

    Home views are captured before this. No collision/PhysX component is touched.
    Current table/camera bodies must not occlude an old recorded robot pose.
    """
    from sapien.render import RenderBodyComponent
    task_actors = [link for obj in getattr(backend, 'objects', {}).values() for link in obj.links()]
    if backend.gamepad is not None and not task_actors:
        task_actors = backend.gamepad.articulation.get_links()
    for actor in [backend.table, *backend.fixtures, *task_actors]:
        # ActorBuilder returns Entity; articulation links expose .entity.
        entity=getattr(actor,'entity',actor)
        body=entity.find_component_by_type(RenderBodyComponent)
        if body is not None:body.visibility=0.
    camera_meshes={f.mesh for f in backend.spec.fixtures if f.name.startswith('camera_')}
    for robot in (backend.robot,backend.right_robot):
        if robot is None:continue
        for link in robot.get_links():
            body=link.entity.find_component_by_type(RenderBodyComponent)
            if body is None:continue
            shapes=body.render_shapes
            if not any(getattr(shape,'filename',None) in camera_meshes for shape in shapes):continue
            replacement=RenderBodyComponent()
            for shape in shapes:
                if getattr(shape,'filename',None) not in camera_meshes:replacement.attach(shape.clone())
            link.entity.remove_component(body);link.entity.add_component(replacement)
