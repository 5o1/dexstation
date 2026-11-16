"""Tangent optimality, real render-mesh transforms and output isolation."""
from helpers import fake_mesh, camera_backend, gamepad_camera_task
from dexstation.model.spec import pose_to_matrix
import math
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from dexstation.render import auxiliary_views as aux




@pytest.mark.parametrize('name', ['face','left','right'])
@pytest.mark.parametrize('size', [(1280,720),(720,1280)])
def test_every_vertex_fits_and_tangent_minimum_is_tight(name,size):
    rng=np.random.default_rng(742)
    points=rng.uniform([-0.09,-0.055,-.022],[.08,.06,.035],(500,3))
    # Offset + rotate the geometry and preserve each old camera's direction/up.
    transform=np.array([[0,-1,0,.71],[1,0,0,.63],[0,0,1,.98],[0,0,0,1.]])
    points=points@transform[:3,:3].T+transform[:3,3]
    original=aux.camera_vectors(transform,next(c for c in gamepad_camera_task()['auxiliary_cameras'] if c['id']==name))
    vectors,info=aux.fit_camera(points,original,*size)
    assert np.allclose(aux.view_basis(*vectors),aux.view_basis(*original))
    assert np.array_equal(vectors[2],original[2])
    assert info['distance']==pytest.approx(info['d_min']*1.12)
    assert min(info['margins_px'].values())>0
    assert info['min_depth']>aux.NEAR
    tight,tight_info=aux.fit_camera(points,original,*size,margin=1)
    assert min(tight_info['margins_px'].values())==pytest.approx(0,abs=1e-9)
    # A smaller distance makes opposing support intervals disjoint: moving the
    # target transversely cannot restore full framing.
    basis=aux.view_basis(*tight)
    local=(points-tight[1])@basis
    d=info['d_min']*(1-1e-5)
    ty=math.tan(aux.FOVY/2)
    gaps=[]
    for axis,t in enumerate((ty*size[0]/size[1],ty)):
        gaps.append(np.max(local[:,axis]-t*(d+local[:,2]))-
                    np.min(local[:,axis]+t*(d+local[:,2])))
    assert max(gaps)>0


def test_mesh_uses_all_vertices_scale_shape_and_current_entity_pose(monkeypatch):
    backend=NS();shape,entity=fake_mesh(backend,monkeypatch)
    shape.scale=np.array([2.,3.,4.])
    shape_pose=np.eye(4);shape_pose[:3,3]=[1,2,3]
    shape.local_pose=NS(to_transformation_matrix=lambda:shape_pose)
    mesh=aux.ObjectRenderMesh(backend.objects['gamepad'])
    expected=shape.parts[0].vertices*shape.scale+[1,2,3]
    np.testing.assert_allclose(mesh.world_vertices(),expected)
    moved=np.array([[0,-1,0,5],[1,0,0,6],[0,0,1,7],[0,0,0,1.]])
    entity.pose=NS(to_transformation_matrix=lambda:moved)
    np.testing.assert_allclose(mesh.world_vertices(),expected@moved[:3,:3].T+[5,6,7])


def test_background_replaces_only_empty_pixels_and_does_not_mutate_color(monkeypatch):
    from dexstation.cameras import image as collection_camera
    monkeypatch.setattr(collection_camera,'color_to_rgb8',lambda value:value[...,:3].copy())
    color=np.full((2,3,4),255,np.uint8)
    seg=np.zeros((2,3,4),np.uint32);seg[0,:,1]=[3,8,17]
    camera=NS(get_picture=lambda kind:color if kind=='Color' else seg)
    result=aux.auxiliary_rgb(camera)
    assert (result[0]==255).all()
    assert (result[1]==[58,64,72]).all()
    assert (color==255).all()
    result=aux.auxiliary_rgb(camera,{'auxiliary':{'background_color':'#123456'}})
    assert (result[1]==[18,52,86]).all()
    result=aux.auxiliary_rgb(camera,background_ids=[8])
    assert (result[0,1]==[58,64,72]).all()
    assert (result[0,[0,2]]==255).all()


def test_untouched_poses_do_not_read_mesh():
    name = 'back'
    backend=NS(spec=NS(gamepad=NS(root_pose=(.5,.6,.8,1,0,0,0))))
    camera_backend(backend)
    framing=aux.AuxiliaryFraming(backend)
    assert all(np.array_equal(a,b) for a,b in zip(framing.vectors(name,640,540),
        aux.camera_vectors(pose_to_matrix(backend.objects['gamepad'].definition.initial_pose_world),framing.declarations[name])))


@pytest.mark.parametrize('values',[{'framing_margin':1},{'framing_margin':float('nan')},
    {'framing_margin':True},{'background_color':'3a4048'},{'background_color':'#zz0000'},
    {'width':1952,'height':1647},{'width':1920,'height':1080}])
def test_invalid_framing_configuration(values):
    with pytest.raises(ValueError):aux.auxiliary_options({'auxiliary':values})


def test_live_output_is_close_and_background_changes_without_touching_scene(monkeypatch):
    """Behavioral regression: old code renders white and leaves >100 px borders."""
    from dexstation.cameras import image as collection_camera
    monkeypatch.setattr(collection_camera,'color_to_rgb8',lambda value:value[...,:3].copy())
    monkeypatch.setitem(sys.modules,'xbox_gamepad_physics.examples.scene',NS(look_at=lambda *vectors:vectors))
    created=[]
    class Camera:
        def __init__(self,name,w,h):
            self.name,self.w,self.h=name,w,h
            self.entity=NS(set_pose=lambda vectors:setattr(self,'vectors',vectors))
            created.append(self)
        def take_picture(self):pass
        def get_picture(self,kind):
            if kind=='Color':return np.full((self.h,self.w,4),255,np.uint8)
            seg=np.ones((self.h,self.w,4),np.uint32);seg[:,0,1]=0
            return seg
    backend=NS(scene=NS(add_camera=lambda *args:Camera(*args[:3]),update_render=lambda:None),
        spec=NS(gamepad=NS(root_pose=(0,0,0,1,0,0,0))))
    shape,_=fake_mesh(backend,monkeypatch)
    cameras=aux.AuxiliaryCameras(backend)
    atlas=cameras.capture(['face','back'])
    assert np.array_equal(atlas[0,0],[58,64,72])
    assert np.array_equal(atlas[0,1],[255,255,255])
    p,t,up=created[0].vectors
    f=(t-p)/np.linalg.norm(t-p);r=np.cross(f,up);r/=np.linalg.norm(r);u=np.cross(r,f)
    xyz=(shape.parts[0].vertices-p)@np.column_stack([r,u,f])
    xy=xyz[:,:2]/xyz[:,2:3]
    half_y=math.tan(math.radians(25));half_x=half_y*960/540
    assert max(np.ptp(xy[:,0])/(2*half_x),np.ptp(xy[:,1])/(2*half_y))>.85


def test_background_surface_ids_leave_working_fixture_and_column_visible():
    backend=NS(table=NS(per_scene_id=1),fixtures=[NS(name=n,per_scene_id=i+2) for i,n in enumerate([
        'station_floor','station_backdrop_green','station_table_frame_panel','v3_table_apron',
        'v3_visual_pad_back','station_column','camera_3_body'])])
    assert aux.background_entity_ids(backend)==(1,2,3,4,5)


def test_snapshot_auxiliary_aspect_does_not_change_camera_3_request(monkeypatch):
    from dexstation.render.scene_snapshot import snapshot_cameras
    from dexstation.render import review as render_review
    from dexstation.model.spec import IntrinsicsSpec, CameraSpec
    monkeypatch.setitem(sys.modules,'xbox_gamepad_physics.examples.scene',NS(look_at=lambda *v:v))
    made=[]
    def create(name,w,h,*args):
        camera=NS(name=name,width=w,height=h,entity=NS(set_pose=lambda p:None))
        made.append(camera);return camera
    backend=NS(scene=NS(add_camera=create),spec=NS(gamepad=NS(root_pose=(0,0,0,1,0,0,0)),
        cameras=[CameraSpec('camera_3','','','world',(0,0,0,1,0,0,0),IntrinsicsSpec(1280,720,800,800,640,360,(),'pinhole'),'','')]))
    fake_mesh(backend,monkeypatch)
    monkeypatch.setattr(render_review,'create_spec_camera',lambda b,d:create(d.name,d.intrinsics.width,d.intrinsics.height))
    cameras=snapshot_cameras(backend,1280,1080)
    assert set(cameras)=={'camera_3',*('aux_'+n for n in aux.available_views(backend))}
    assert (cameras['camera_3'].width,cameras['camera_3'].height)==(1280,1080)
    assert all((c.width,c.height)==(1920,1080) for n,c in cameras.items() if n.startswith('aux_'))


def test_empty_snapshot_never_builds_gamepad_anchored_cameras(monkeypatch):
    from dexstation.render.scene_snapshot import snapshot_cameras
    from dexstation.render import review as render_review
    from dexstation.model.spec import IntrinsicsSpec, CameraSpec
    made=[]
    monkeypatch.setitem(sys.modules,'xbox_gamepad_physics.examples.scene',NS(look_at=lambda *v:v))
    def create(name,w,h,*args):
        made.append(name)
        return NS(name=name,entity=NS(set_pose=lambda pose:None))
    backend=NS(spec=NS(gamepad=None,cameras=[CameraSpec('camera_3','','','world',
        (0,0,0,1,0,0,0),IntrinsicsSpec(1280,720,800,800,640,360,(),'pinhole'),'','')]),
        scene=NS(add_camera=create))
    monkeypatch.setattr(render_review,'create_spec_camera',lambda b,c:create(c.name,1280,720))
    cameras=snapshot_cameras(backend,1280,720)
    assert set(cameras)=={'camera_3','aux_table'}
    assert made==['snapshot_camera_3','snapshot_aux_table']


def test_frozen_camera_vectors_fov_and_near_are_identical(monkeypatch):
    """Frozen outputs from 2ccdaba, independent of the new declarations/math."""
    import json
    from pathlib import Path
    frozen = json.loads((Path(__file__).parent / 'fixtures/auxiliary_cameras_2ccdaba.json').read_text())
    assert aux.FOVY == frozen['fovy'] and aux.NEAR == frozen['near']
    points = np.random.default_rng(frozen['mesh_seed']).uniform(
        frozen['mesh_low'], frozen['mesh_high'], (frozen['mesh_size'], 3))
    for case in frozen['cases']:
        transform = np.asarray(case['transform'])
        backend = camera_backend(NS())
        backend.objects['gamepad'].definition.initial_pose_world = case['root_pose']
        framing = aux.AuxiliaryFraming(backend)
        vertices = points @ transform[:3,:3].T + transform[:3,3]
        framing.meshes['gamepad'] = NS(world_vertices=lambda:vertices)
        for name, expected in case['vectors'].items():
            declaration = framing.declarations[name]
            vectors = aux.camera_vectors(transform, declaration)
            if declaration['fit_render_meshes']:
                vectors = aux.fit_camera(vertices, vectors, case['width'], case['height'], declaration['framing_margin'])[0]
            from helpers import assert_frozen_numeric
            assert_frozen_numeric(np.asarray(vectors).tolist(), expected)
            assert_frozen_numeric(np.asarray(framing.vectors(name,case['width'],case['height'])).tolist(), expected)
