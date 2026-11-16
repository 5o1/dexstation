"""Measured-shape controls, both chiralities. No ROS/device/optimizer imports."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
import pytest
from dexstation.inputs.retarget_anatomy import *
from dexstation.inputs.retarget_runtime import *


def measured_nodes(side='right',bend=0.):
    nodes=[]
    def add(f,j,p,parent):
        i=len(nodes);nodes.append(N(node_id=i,parent_node_id=parent,chain_type=f,joint_type=j,pose=N(position=N(**dict(zip('xyz',p))),orientation=N(x=0.,y=0.,z=0.,w=1.))));return i
    add('Hand','Root',[0,0,0],0)
    sign=1 if side=='right' else -1
    for f,y in zip(('Thumb','Index','Middle','Ring','Pinky'),(.026,.023,.008,-.008,-.023)):
        p=np.array([0,y*sign,.014]);parent=add(f,'MCP',p,0)
        js=('PIP','DIP','TIP') if f=='Thumb' else ('PIP','IP','DIP','TIP')
        lengths=(.042,.03,.025) if f=='Thumb' else (.070,.040,.022,.019)
        for k,(j,L) in enumerate(zip(js,lengths)):
            theta=0 if k==0 else bend*k
            p=p+np.array([-np.sin(theta),0,np.cos(theta)])*L
            parent=add(f,j,p,parent)
    return nodes


@pytest.mark.parametrize('side',['left','right'])
def test_measured_shape_mapping_direction_and_no_degenerate_robot_pairs(side):
    nodes=measured_nodes(side,.35);points=relabel_nodes(nodes);ref=anatomical_targets(nodes,side)
    assert len(points)==25 and len(PAIRS)==15
    assert np.min(robot_geometry(side)[0])>.005
    np.testing.assert_allclose(palm_basis(points,side).T@palm_basis(points,side),np.eye(3),atol=1e-12)
    assert len(ref['vectors'])==15
    # Old vector ignores fingertip. New targets must respond to distal direction.
    before=np.array(ref['vectors']);nodes[9].pose.position.x+=.004
    after=np.array(anatomical_targets(nodes,side)['vectors'])
    assert not np.allclose(before,after)


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('damage',['missing','short','parent','nan','units'])
def test_rejects_bad_hand(side,damage):
    nodes=measured_nodes(side)
    if damage=='missing':nodes.pop()
    if damage=='short':nodes[7].pose.position=nodes[6].pose.position
    if damage=='parent':nodes[7].parent_node_id=0
    if damage=='nan':nodes[7].pose.position.x=float('nan')
    if damage=='units':
        for n in nodes:n.pose.position.z*=1000
    with pytest.raises(SkeletonError):anatomical_targets(nodes,side)


@pytest.mark.parametrize('side',['left','right'])
def test_motor_format_and_range(side):
    import xml.etree.ElementTree as ET
    joints=[j for j in ET.parse(ASSETS/f'g20_{side}.urdf').getroot().findall('joint') if j.get('type')!='fixed']
    for fraction in (0,.2,.5,1):
        q=[float(j.find('limit').get('lower'))+fraction*(float(j.find('limit').get('upper'))-float(j.find('limit').get('lower'))) for j in joints]
        fields=command_fields(motor_values(q,[j.get('name') for j in joints],side))
        assert fields['name']==[f'joint_{i}' for i in range(20)]
        assert all(0<=v<=255 for v in fields['position'])
        assert fields['velocity']==[255.]*20 and fields['effort']==[0.]*20


def test_filter_rate_and_composed_gradient():
    prev=np.zeros(20);cfg=dict(time_constant_s=0,max_motor_units_per_s=100)
    np.testing.assert_allclose(filter_motors(np.ones(20)*255,prev,.01,cfg),1)
    def square(x,g):
        if g.size:g[:]=2*x
        return float(x@x)
    fn=combine_objectives(square,square,.25);g=np.zeros(3);x=np.ones(3)
    assert fn(x,g)==pytest.approx(3.75);np.testing.assert_allclose(g,2.5)


def test_lossless_recording(tmp_path):
    from dexstation.inputs.glove_recording import GloveRecording, read_glove_records, protocol_segment
    rec=GloveRecording(tmp_path,'test');cdr=bytes(range(256));msg=dict(raw_nodes=[],ergonomics=[],raw_sensor=[])
    rec.append('/manus_glove_0',cdr,msg,'open',wall_ns=10,monotonic_ns=20);rec.close()
    row=next(read_glove_records(rec.path));assert base64.b64decode(row['cdr_base64'])==cdr and row['message']==msg
    assert protocol_segment(0)=='open' and protocol_segment(8)=='fist' and protocol_segment(1000) is None
    from dexstation.inputs.glove_recording import PROMPTS, PROTOCOL, accepts_side
    assert all(name in PROMPTS for name,_ in PROTOCOL)
    assert accepts_side('Right','right') and accepts_side(' left ','left')
    assert not accepts_side('Left','right') and accepts_side('Left',None)
