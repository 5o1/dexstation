"""Geometry, independent panes and disabled-camera scheduling."""
from helpers import gamepad_camera_task
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from dexstation.render.auxiliary_views import AUXILIARY_VIEWS, ATLAS_WIDTH, ATLAS_HEIGHT, TABLE_VIEW, AuxiliaryCameras, atlas_tiles, camera_vectors, selected_views, frame_capacity


def test_gamepad_relative_views_follow_full_scene_pose():
    base=np.eye(4)
    moved=np.array([[0.,-1.,0.,.65],[1.,0.,0.,.62],[0.,0.,1.,.99],[0.,0.,0.,1.]])
    for camera in gamepad_camera_task()['auxiliary_cameras']:
        old=camera_vectors(base,camera);new=camera_vectors(moved,camera)
        for i in (0,1):assert np.allclose(new[i],moved[:3,:3]@old[i]+moved[:3,3])
        assert np.allclose(new[2],moved[:3,:3]@old[2])


def test_fixed_tiles_do_not_overlap_and_fit_existing_mmap():
    cfg={'runtime_scene':{'task':gamepad_camera_task()}}
    tiles=atlas_tiles(cfg);coverage=np.zeros((ATLAS_HEIGHT,ATLAS_WIDTH),np.uint8)
    for t in tiles.values():coverage[t['y']:t['y']+t['height'],t['x']:t['x']+t['width']]+=1
    assert len(tiles)==5 and coverage.max()==1
    assert not coverage[1080:1620,:960].any()  # Removed camera's slot remains black.
    assert np.count_nonzero(coverage)==ATLAS_WIDTH*ATLAS_HEIGHT*5//6
    assert frame_capacity(ATLAS_WIDTH,ATLAS_HEIGHT)==ATLAS_WIDTH*ATLAS_HEIGHT*3+65536
    for width,height in ((1920,1620),(2560,2160)):
        coverage=np.zeros((height,width),np.uint8)
        for t in atlas_tiles(dict(cfg,auxiliary=dict(width=width,height=height))).values():
            coverage[t['y']:t['y']+t['height'],t['x']:t['x']+t['width']]+=1
        assert coverage.max()==1 and np.count_nonzero(coverage)==width*height*5//6
        assert all(t['width']*9==t['height']*16 for t in atlas_tiles(dict(cfg,auxiliary=dict(width=width,height=height))).values())
    assert selected_views(['right','face','right'],list(tiles))==['face','right']
    for value in ('face',None,['bogus'],['front']):
        with pytest.raises(ValueError):selected_views(value)


@pytest.mark.parametrize('size',[(1920,1620),(2560,2160)])
def test_closed_cameras_never_take_picture(monkeypatch,size):
    # Test scheduling with real pixel assembly and fake camera calls, no renderer needed.
    scene_module=SimpleNamespace(look_at=lambda *args:args)
    monkeypatch.setitem(sys.modules,'xbox_gamepad_physics.examples.scene',scene_module)
    from dexstation.cameras import image as collection_camera
    monkeypatch.setattr(collection_camera,'color_to_rgb8',lambda value:value)
    calls=[]
    class Camera:
        entity=SimpleNamespace(set_pose=lambda pose:None)
        def __init__(self,name,w,h):self.name,self.w,self.h=name,w,h
        def take_picture(self):calls.append(self.name)
        def get_picture(self,kind):
            return np.ones((self.h,self.w,4 if kind=='Segmentation' else 3),np.uint8)
    scene=SimpleNamespace(add_camera=lambda name,w,h,*args:Camera(name,w,h),update_render=lambda:None)
    backend=SimpleNamespace(scene=scene,spec=SimpleNamespace(gamepad=SimpleNamespace(root_pose=(0,0,0,1,0,0,0))))
    from helpers import fake_mesh
    fake_mesh(backend,monkeypatch)
    cameras=AuxiliaryCameras(backend,{'auxiliary':dict(width=size[0],height=size[1])})
    assert not cameras.capture([]).any() and calls==[]
    cameras.capture(['face','right']);cameras.capture(['right'])
    assert calls==['aux_face','aux_right','aux_right']
    assert set(cameras.cameras)=={'face','right'}




def test_table_view_is_world_fixed_top_down_and_configurable():
    import numpy as np
    from dexstation.render.auxiliary_views import AuxiliaryFraming, table_vectors, atlas_tiles
    assert AUXILIARY_VIEWS[-1] == 'table'
    position, target, up = table_vectors()
    assert np.allclose(position - target, np.asarray(TABLE_VIEW['position']) - TABLE_VIEW['target']) and np.allclose(up, [1, 0, 0])
    custom = table_vectors(dict(overview=dict(position=[0, 0, 3], target=[0, 0, 1], width=480)))
    assert np.allclose(custom[0], [0, 0, 3]) and np.allclose(custom[1], [0, 0, 1])
    framing = AuxiliaryFraming.__new__(AuxiliaryFraming)
    framing.table = table_vectors()
    assert framing.vectors('table', 960, 540) is framing.table
    tiles = atlas_tiles()
    assert len({(t['x'], t['y']) for t in tiles.values()}) == 1


def test_empty_hides_gamepad_cameras_and_keeps_table(monkeypatch):
    from dexstation.render.auxiliary_views import AuxiliaryFraming, available_views
    backend=SimpleNamespace(spec=SimpleNamespace(gamepad=None))
    assert available_views(backend)==('table',)
    framing=AuxiliaryFraming(backend)
    assert np.allclose(framing.vectors('table',960,540)[0],TABLE_VIEW['position'])


def test_overview_is_not_a_declared_or_rendered_camera():
    from dexstation.render.auxiliary_views import available_views
    from helpers import camera_backend
    backend = camera_backend(SimpleNamespace())
    assert available_views(backend) == ('face', 'back', 'left', 'right', 'table')
    assert 'overview' not in atlas_tiles(backend=backend)
    with pytest.raises(ValueError):
        selected_views(['overview'], available_views(backend))
