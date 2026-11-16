"""Maintained MJCF visual conversion behavior, independent of report inputs."""
import json
import pytest


def test_mujoco_uses_shipped_obj_and_palette_for_glb_visuals(tmp_path):
    import xml.etree.ElementTree as ET
    from dexstation.backends.mujoco_export import _mujoco_visuals
    visual_dir = tmp_path / 'pkg/meshes/visual/left'
    visual_dir.mkdir(parents=True)
    (visual_dir / 'index_distal_rubber_black.obj').write_text('v 0 0 0\n')
    (tmp_path / 'pkg/materials.json').write_text(json.dumps({'rubber_black': {'hex': '#1e1f21'}}))
    root = ET.fromstring(f'<robot><link name="index_distal"><visual name="index_distal_rubber_black"><geometry>'
                         f'<mesh filename="{visual_dir}/index_distal_rubber_black.glb"/></geometry></visual></link></robot>')
    _mujoco_visuals(root)
    assert root.find('.//mesh').get('filename').endswith('.obj')
    assert root.find('.//material/color').get('rgba') == '0.1176 0.1216 0.1294 1'
    root = ET.fromstring(f'<robot><link name="x"><visual><geometry><mesh filename="{visual_dir}/missing.glb"/></geometry></visual></link></robot>')
    with pytest.raises(ValueError):
        _mujoco_visuals(root)
