from helpers import bench_data, bench_spec
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import sys
from types import SimpleNamespace

from dexstation.model.spec import load_spec, spec_from_dict, spec_to_dict, pose_to_matrix, optical_to_sapien_pose, compose
from dexstation.model.appearance import load_appearance, match_material
from dexstation.render.review import create_spec_camera, mask_iou, camera_world_optical, robot_only_review_visuals
from dexstation.backends.sapien_render import SapienBackend
from dexstation.control.commands import ArmCommand

ROOT = Path(__file__).parents[1]


def camera():
    return dict(name="camera_1",serial="test-camera",model="D435",parent="world",
                pose_optical=[1,2,3,1,0,0,0],
                intrinsics=dict(width=640,height=480,fx=500.,fy=501.,cx=321.,cy=239.,distortion=[0]*5,model="plumb_bob"),
                status="estimated",source="unit test fixture")


def test_bench_roundtrip_with_camera(tmp_path):
    v4=bench_spec()
    assert v4.right_arm and len(v4.right_hand.home_q)==21
    data=spec_to_dict(v4)
    data["cameras"]=[camera()]
    parsed=spec_from_dict(data)
    assert spec_from_dict(spec_to_dict(parsed))==parsed


@pytest.mark.parametrize("mutate", [
    lambda d:d["cameras"][0]["intrinsics"].update(fx=0),
    lambda d:d["cameras"][0]["intrinsics"].update(width=640.5),
    lambda d:d["cameras"][0].update(status="calibrated"),
    lambda d:d["cameras"].append(copy.deepcopy(d["cameras"][0])),
    lambda d:d["right_hand"].update(home_q=[0]*20),
    lambda d:d.pop("right_hand"),
    lambda d:d["fixtures"][0].update(collision="false"),
    lambda d:d["fixtures"][0].update(half_extents=[1,-1,1]),
    lambda d:d["fixtures"][0].update(unexpected=1),
])
def test_v4_rejects_invalid_fields(mutate):
    data=bench_data(); data["cameras"]=[camera()]
    mutate(data)
    with pytest.raises(ValueError): spec_from_dict(data)


def test_optical_basis_and_offcentre_projection():
    pose=[1,2,3,1,0,0,0]
    transformed=pose_to_matrix(optical_to_sapien_pose(pose))
    np.testing.assert_allclose(transformed[:3,3],pose[:3])
    np.testing.assert_allclose(transformed[:3,:3]@[1,0,0],[0,0,1],atol=1e-14)
    np.testing.assert_allclose(transformed[:3,:3]@[0,1,0],[-1,0,0],atol=1e-14)
    np.testing.assert_allclose(transformed[:3,:3]@[0,0,1],[0,-1,0],atol=1e-14)
    rotated=[.2,-.1,.8,.5,.5,.5,.5]
    expected=pose_to_matrix(rotated) @ np.linalg.inv(pose_to_matrix(pose)) @ transformed
    np.testing.assert_allclose(pose_to_matrix(optical_to_sapien_pose(rotated)),expected,atol=1e-14)


def test_appearance_parsing_first_match_and_rejection(tmp_path):
    table=load_appearance(ROOT/"src/dexstation/examples/ar5_g20/appearance/station_v1.json")
    assert match_material(table,"table")["name"]=="table_green"
    assert match_material(table,"AR5_5_07R_W4C4A2_base")["name"]=="ar5_base_gray"
    assert match_material(table,"AR5_5_07L_W4C4A2_link1")["name"]=="ar5_warm_white"
    assert match_material(table,"right_index_distal")["name"]=="g20_dark"
    assert match_material(table,"unmatched") is None
    table["materials"][0]["base_color"]=[1.1,0,0,1]
    path=tmp_path/"bad.json";path.write_text(json.dumps(table))
    with pytest.raises(ValueError,match="base_color"):load_appearance(path)


def test_calibrated_camera_passes_full_intrinsics_and_optical_pose(monkeypatch):
    data=spec_to_dict(bench_spec());data["cameras"]=[camera()]
    definition=spec_from_dict(data).cameras[0]
    calls={}
    rendered=SimpleNamespace(
        set_perspective_parameters=lambda *args:calls.update(intrinsics=args),
        entity=SimpleNamespace(set_pose=lambda pose:calls.update(pose=pose)))
    backend=SimpleNamespace(spec=SimpleNamespace(world_frame="world"),
                            scene=SimpleNamespace(add_camera=lambda *args:rendered))
    monkeypatch.setitem(sys.modules,"sapien",SimpleNamespace(Pose=lambda p,q:[*p,*q]))
    assert create_spec_camera(backend,definition) is rendered
    assert calls["intrinsics"]==(.001,20.,500.,501.,321.,239.,0.)
    np.testing.assert_allclose(pose_to_matrix(calls["pose"]),pose_to_matrix(optical_to_sapien_pose([1,2,3,1,0,0,0])))
    assert mask_iou(np.array([True,True,False]),np.array([False,True,True])) == 1/3
    assert mask_iou(np.zeros(3,bool),np.zeros(3,bool)) is None


@pytest.mark.parametrize("parent,resolved",[("left_hand:hand_base_link","hand_base_link"),
                                           ("right_hand:hand_base_link","right_hand_base_link"),
                                           ("left_arm:AR5_link","AR5_link"),
                                           ("right_arm:AR5_right_link","AR5_right_link")])
def test_eye_in_hand_parent_follows_link(parent,resolved):
    from dataclasses import replace
    data=spec_to_dict(bench_spec());data["cameras"]=[camera()]
    definition=replace(spec_from_dict(data).cameras[0],parent=parent)
    root_pose=[.2,.3,.8,.5,.5,.5,.5]
    seen=[]
    def pose(name):
        seen.append(name)
        return root_pose
    backend=SimpleNamespace(spec=SimpleNamespace(world_frame="world"),link_pose=pose)
    np.testing.assert_allclose(camera_world_optical(definition,backend),compose(root_pose,definition.pose_optical))
    assert seen == [resolved]
    root_pose[0]+=.1
    np.testing.assert_allclose(camera_world_optical(definition,backend),compose(root_pose,definition.pose_optical))


def test_mounted_camera_attaches_component_without_moving_hand(monkeypatch):
    from dataclasses import replace
    data=spec_to_dict(bench_spec());data["cameras"]=[camera()]
    definition=replace(spec_from_dict(data).cameras[0],parent="left_hand:hand_base_link")
    mount=object();calls={}
    def mounted(name,entity,pose,*parameters):
        calls.update(entity=entity,pose=pose,parameters=parameters)
        return SimpleNamespace(set_perspective_parameters=lambda *args:None)
    backend=SimpleNamespace(spec=SimpleNamespace(world_frame="world"),
        camera_parent_entity=lambda parent:mount,scene=SimpleNamespace(add_mounted_camera=mounted))
    monkeypatch.setitem(sys.modules,"sapien",SimpleNamespace(Pose=lambda p,q:[*p,*q]))
    create_spec_camera(backend,definition)
    assert calls["entity"] is mount
    np.testing.assert_allclose(calls["pose"],optical_to_sapien_pose(definition.pose_optical))


def test_right_live_commands_remain_rejected():
    backend=object.__new__(SapienBackend)
    with pytest.raises(ValueError,match="only for the left"):
        backend.apply([ArmCommand("right",(0.,)*7,0.)])


def test_old_epoch_review_hides_entity_and_link_visuals_only(monkeypatch):
    bodies=[SimpleNamespace(visibility=1.) for _ in range(3)]
    entities=[SimpleNamespace(find_component_by_type=lambda kind,body=body:body) for body in bodies]
    monkeypatch.setitem(sys.modules,"sapien.render",SimpleNamespace(RenderBodyComponent=object))
    backend=SimpleNamespace(table=entities[0],fixtures=[entities[1]],
        gamepad=SimpleNamespace(articulation=SimpleNamespace(get_links=lambda:[SimpleNamespace(entity=entities[2])])),
        spec=SimpleNamespace(fixtures=[]),robot=None,right_robot=None)
    robot_only_review_visuals(backend)
    assert [b.visibility for b in bodies] == [0.,0.,0.]
