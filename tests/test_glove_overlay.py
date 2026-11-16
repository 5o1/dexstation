"""Golden original-retarget outputs, step capture, no-data and clean recording."""
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
import pytest
from dexstation.inputs import glove_overlay as go









def test_no_glove_never_constructs_render_objects():
    backend = N()
    for _ in range(10):
        assert go.update_live_overlay(backend, {}) == {}
    assert not hasattr(backend, 'glove_overlay')




def test_segment_transforms_follow_endpoints():
    segments=[([1.,2.,3.],[1.,4.,3.]),([0.,0.,0.],[-2.,0.,0.])]
    for position, rotation, scale, (start, end) in zip(*go.segment_transforms(segments, .002), segments):
        w,x,y,z=rotation
        matrix=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        np.testing.assert_allclose(position,start)
        np.testing.assert_allclose(matrix@np.array([scale[0],0.,0.])+position,end,atol=1e-10)
        np.testing.assert_allclose(scale[1:], .002)




def test_browser_defaults_restore_and_persist_layer_controls():
    import subprocess
    if shutil.which('node') is None:
        pytest.skip('existing Node required')
    page=(Path(go.__file__).parents[1]/'server/web/index.html').read_text()
    block=page[page.index('let gloveLayers='):page.index('const d=config.defaults')]
    script="""import assert from 'node:assert/strict';
import vm from 'node:vm';
for(const saved of [null,{enabled:false,A:false,B:true,C:true}]){
 const controls=[],writes=[];let stored=saved;
 const ctx={localStorage:{getItem:()=>JSON.stringify(stored),setItem:(k,v)=>{stored=JSON.parse(v)}},
 document:{createElement:()=>({style:{},append(...args){this.children=args;}})},
 $:()=>({append:label=>controls.push(label)}),videoSession:'running',
 action:fn=>fn(),api:async(path,value)=>writes.push({path,value})};
 vm.createContext(ctx);vm.runInContext(BLOCK,ctx);
 assert.equal(controls.length,3);
 assert.deepEqual(controls.map(l=>l.children[0].checked),saved?[false,false,true]:[true,true,true]);
 const master=controls[0].children[0];master.checked=!master.checked;master.onchange();
 assert.equal(stored.enabled,master.checked);assert.equal(writes[0].path,'/api/v1/glove-overlay');
 assert.deepEqual(Object.keys(writes[0].value.layers),['enabled','A','C']);
}
""".replace('BLOCK',json.dumps(block))
    # Text nodes are just labels in this minimal DOM.
    script=script.replace("document:{createElement", "document:{createTextNode:s=>s,createElement")
    result=subprocess.run(['node','--input-type=module','-e',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr




def test_fingertip_frames_resolve_from_distal_links():
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    transform = np.eye(4); transform[:3,:3] = rotation; transform[:3,3] = [.3,.2,.7]
    link = N(pose=N(p=transform[:3,3], to_transformation_matrix=lambda: transform))
    links = {'hand_base_link': link, 'index_distal': link}
    sample = dict(points=[[0,0,0]], edges=[], origins=['hand_base_link'], tasks=['index_tip'], vectors=[[0,0,.1]])
    for side in ('left', 'right'):
        parent, offset = go.fixed_frames(side)['index_tip']
        assert parent == 'index_distal'
        g = go.world_geometry(sample, links, side)
        np.testing.assert_allclose(g['actual'][0], rotation @ offset + transform[:3,3])
    with pytest.raises(KeyError):
        go.world_geometry(dict(sample, tasks=['index_nowhere']), links, 'left')


def test_anatomical_overlays_only_name_resolvable_frames():
    import xml.etree.ElementTree as ET
    from dexstation.inputs.retarget_anatomy import PAIRS, FINGERS
    for side in ('left', 'right'):
        root = ET.parse(Path(go.__file__).with_name('retarget_assets')/f'g20_{side}.urdf').getroot()
        moving = {'hand_base_link'} | {j.find('child').get('link') for j in root.iter('joint') if j.get('type') != 'fixed'}
        fixed = go.fixed_frames(side)
        for name in {p.robot_origin for p in PAIRS} | {p.robot_task for p in PAIRS}:
            assert name in moving or fixed.get(name, ('',))[0] in moving, (side, name)


def test_overlay_failure_does_not_stop_rendering(capsys):
    class Broken:
        hidden = False
        def update(self, samples, layers):
            raise KeyError('middle_tip')
        def hide(self):
            self.hidden = True
    backend = N(glove_overlay=Broken())
    for _ in range(2):
        result = go.update_live_overlay(backend, {'glove_overlay': {'left': {}}})
        assert 'middle_tip' in result['error'] and backend.glove_overlay.hidden
    assert capsys.readouterr().err.count('glove overlay disabled') == 1


def test_public_geometry_uses_owned_anatomical_targets():
    from test_retarget_anatomical import measured_nodes
    from dexstation.inputs.retarget_anatomy import anatomical_targets
    geometry = go.anatomical_geometry()
    for side in ('left', 'right'):
        nodes = measured_nodes(side, .2)
        actual = geometry.prepare(nodes, side)
        expected = anatomical_targets(nodes, side)
        for key in ('points', 'labels', 'edges', 'vectors', 'origins', 'tasks'):
            assert actual[key] == expected[key]
    with pytest.raises(ValueError, match='Unavailable retarget geometry'):
        go.anatomical_geometry(retarget='unknown')


def test_ros_overlay_uses_injected_geometry_provider():
    from dexstation.inputs.replay import Ros2Source
    from test_retarget_anatomical import measured_nodes
    source = Ros2Source.__new__(Ros2Source)
    source._lock = threading.Lock()
    source.latest_gloves = {'left': go.pack_nodes(N(raw_nodes=measured_nodes('left')))}
    source.glove_geometry = None
    source.retarget = 'external_method'
    calls = []
    class Geometry:
        def prepare(self, nodes, side):
            return {'side': side, 'count': len(nodes)}
    def provider(*, retarget):
        calls.append(retarget)
        return Geometry()
    source.geometry_provider = provider
    source.freeze_gloves()
    source.freeze_gloves()
    assert calls == ['external_method']
    assert source.glove_samples == {'left': {'side': 'left', 'count': 25}}
    assert source.glove_error is None


def test_snapshot_fingerprints_only_selected_provider_resources(tmp_path):
    import hashlib
    from dexstation.render.scene_snapshot import provider_fingerprint
    public = provider_fingerprint()
    assert public['provider'] == 'dexstation.inputs.glove_overlay.anatomical_geometry'
    assert public['retarget'] == 'manus_anatomical'
    assert len(public['resources']) == 4
    assert all('station' not in Path(path).parts for path in public['resources'])
    assert all(public['resources'].values())
    resource = tmp_path/'provider.json'
    resource.write_text('{"setting": 1}')
    calls = []
    def external_provider(*, retarget):
        raise AssertionError('Fingerprinting must not construct geometry')
    def resources(*, retarget):
        calls.append(retarget)
        return (resource,)
    external_provider.fingerprint_resources = resources
    before = provider_fingerprint(external_provider, retarget='external')
    assert calls == ['external']
    assert before['resources'] == {str(resource): hashlib.sha256(resource.read_bytes()).hexdigest()}
    resource.write_text('{"setting": 2}')
    after = provider_fingerprint(external_provider, retarget='external')
    assert before['resources'] != after['resources']
    assert before['source_sha256'] == after['source_sha256']
