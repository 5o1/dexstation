import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parents[1]
module = importlib.util.spec_from_file_location('derive_ar5', ROOT / 'scripts/derive_ar5_08_arms.py')
derive_ar5 = importlib.util.module_from_spec(module)
module.loader.exec_module(derive_ar5)

SOURCE = """<robot name="arm">
  <link name="AR5_5_08L_W4C4A6_base"><visual><geometry><mesh filename="meshes/AR5_5_08L_W4C4A6_base.STL"/></geometry></visual></link>
  <link name="AR5_5_08L_W4C4A6_link1"/>
  <joint name="AR5_5_08L_W4C4A6_joint_1" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/><parent link="AR5_5_08L_W4C4A6_base"/><child link="AR5_5_08L_W4C4A6_link1"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""


def test_derive_renames_by_side_and_keeps_kinematics(tmp_path):
    source = tmp_path / 'arm.urdf'
    source.write_text(SOURCE)
    root = derive_ar5.derive(source, 'left', '../../src/left').getroot()
    assert [l.get('name') for l in root.findall('link')] == ['AR5_left_5_08L_W4C4A6_base', 'AR5_left_5_08L_W4C4A6_link1']
    joint = root.find('joint')
    assert joint.get('name') == 'AR5_left_5_08L_W4C4A6_joint_1'
    assert (joint.find('parent').get('link'), joint.find('child').get('link')) == (
        'AR5_left_5_08L_W4C4A6_base', 'AR5_left_5_08L_W4C4A6_link1')
    original = ET.fromstring(SOURCE).find('joint')
    for tag in ('origin', 'axis', 'limit'):
        assert joint.find(tag).attrib == original.find(tag).attrib
    assert root.find('.//mesh').get('filename') == '../../src/left/meshes/AR5_5_08L_W4C4A6_base.STL'


def test_manifest_pins_both_variants():
    entry = derive_ar5.resource(ROOT / 'references/external_resources.json')
    paths = {f['path'] for f in entry['files']}
    for side, tag in (('left', '08L'), ('right', '08R')):
        assert f'{side}/arm.urdf' in paths
        assert any(p.startswith(f'{side}/meshes/') for p in paths)
    assert entry['derived'] == f"assets/derived/ar5_08_{entry['revision'][:8]}"
    assert all(len(f['sha256']) == 64 and f['bytes'] > 0 for f in entry['files'])
