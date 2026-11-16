#!/usr/bin/env python3
"""Close-up renders of both G20 hands with the scene appearance, for colour review.

Uses the scene's own lighting, as the operator views do. Cameras sit in each hand's base frame (palm normal +x, fingers +z), so the
views match photographs regardless of arm pose. Visual output only.
"""
import argparse
from pathlib import Path

import numpy as np

from dexstation.config.defaults import DEFAULT_TASK, DEFAULT_WORKBENCH


VIEWS = {'palm': (0.48, 0.0), 'back': (-0.48, 0.0), 'side_a': (0.0, 0.48), 'side_b': (0.0, -0.48)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default=DEFAULT_TASK)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=720)
    args = parser.parse_args()
    import sapien
    from PIL import Image
    from dexstation.profiles.compose import compose, load_task, load_workbench
    from dexstation.backends.sapien_render import SapienBackend
    from xbox_gamepad_physics.examples.scene import look_at
    scene = compose(load_workbench(Path(__file__).resolve().parents[1] /
        'src/dexstation/examples/ar5_g20' / DEFAULT_WORKBENCH), load_task(args.task), 'sapien', seed=0)
    backend = SapienBackend(scene, render=True)
    scene = backend.scene
    camera = scene.add_camera('g20_review', args.size, args.size, np.radians(32), 0.01, 10.)
    args.output.mkdir(parents=True, exist_ok=True)
    tiles = []
    for robot, link_name in ((backend.robot, 'hand_base_link'), (backend.right_robot, 'right_hand_base_link')):
        link = next(l for l in robot.get_links() if l.name == link_name)
        pose = link.entity.pose.to_transformation_matrix()
        world = lambda p: (pose @ np.r_[p, 1.])[:3]
        target = world((0., 0., 0.12))
        row = []
        for name, (dx, dy) in VIEWS.items():
            camera.entity.set_pose(look_at(world((dx, dy, 0.12)), target, pose[:3, 2]))
            scene.update_render()
            camera.take_picture()
            rgb = (np.clip(camera.get_picture('Color')[..., :3], 0, 1) * 255).astype(np.uint8)
            image = Image.fromarray(rgb)
            image.save(args.output / f'{link_name}_{name}.png')
            row.append(image)
        tiles.append(row)
    sheet = Image.new('RGB', (args.size * len(VIEWS), args.size * 2), (0, 0, 0))
    for r, row in enumerate(tiles):
        for c, image in enumerate(row):
            sheet.paste(image, (c * args.size, r * args.size))
    sheet.save(args.output / 'g20_review.png')
    print(args.output / 'g20_review.png')


if __name__ == '__main__':
    main()
