"""Task-owned object operations; optional asset imports remain inside consumers."""
import os
from pathlib import Path
import shutil

from dexstation.profiles.task_objects import TaskObject, collision_mesh
from dexstation.profiles.export import _hash, _inventory_urdf, gamepad_collision_mesh


def _project_gamepad(obj, pose, data):
    if 'gamepad' in data:
        raise ValueError('WorkcellSpec supports one legacy gamepad projection')
    geometry = obj['geometry']
    data['gamepad'] = dict(release=geometry['release'], profile=geometry['profile'],
        collision=geometry['collision_mesh'], root_pose=list(pose), fixed_root=geometry['fixed_root'])


def _load_gamepad(scene, obj, *, render=False, physics_scene=None, work_dir=None, collisions=True):
    import sapien
    pose = sapien.Pose(list(obj.initial_pose_world[:3]), list(obj.initial_pose_world[3:]))
    from xbox_gamepad_physics.robotwin import load_into_scene
    cfg = dict(obj.extra, **obj.geometry)
    pad = load_into_scene(physics_scene or scene, pose,
        fixed_root=cfg['fixed_root'], profile=cfg['profile'],
        collision=collision_mesh(obj), attach_visual=render)
    return TaskObject(obj, pad.articulation, gamepad=pad)


def _add_mujoco_gamepad(parent, mj, obj, *, attachment=None):
    if attachment is None:
        raise ValueError('gamepad ObjectSpec requires spec.gamepad')
    return [b.name for b in parent.bodies if b.name.startswith(attachment.prefix)]


def _export_gamepad_resources(scene, resource_root, output, assets, description):
    if resource_root is None:
        raise ValueError('gamepad export requires explicit resource_root')
    source = Path(resource_root)
    target = output / 'objects/gamepad_resources'
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('gamepad resources must be regular-file clones')
        if path.is_file():
            dest = target / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            assets.records.append(dict(source=str(path), path=dest.relative_to(output).as_posix(), sha256=_hash(dest)))
    description['gamepad_resources'] = 'objects/gamepad_resources'


def _inventory_gamepad(obj, description, root):
    names, links = _inventory_urdf(root / description['gamepad_resources'] /
        'robotwin' / obj['extra']['profile'] / 'mobility.urdf')
    return names, links, names


def _load_export_gamepad(scene, obj, root, desc, work_dir, sapien, physics_scene, physics_urdf):
    pose = sapien.Pose(obj.initial_pose_world[:3], obj.initial_pose_world[3:])
    from xbox_gamepad_physics.robotwin import load_into_scene
    previous = os.environ.get('XBOX_GAMEPAD_PHYSICS_RESOURCES')
    os.environ['XBOX_GAMEPAD_PHYSICS_RESOURCES'] = str(root / desc['gamepad_resources'])
    try:
        gamepad = load_into_scene(physics_scene, pose,
            fixed_root=obj.extra.get('fixed_root', True), profile=obj.extra['profile'],
            attach_visual=False, collision=gamepad_collision_mesh(obj))
    finally:
        if previous is None:
            os.environ.pop('XBOX_GAMEPAD_PHYSICS_RESOURCES', None)
        else:
            os.environ['XBOX_GAMEPAD_PHYSICS_RESOURCES'] = previous
    return gamepad.articulation, gamepad


