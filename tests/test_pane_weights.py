"""Pane metadata sorting, station overrides, schema errors and lazy discovery."""
import json
from pathlib import Path

import pytest

from dexstation.runtime import plugins
from dexstation.config import site
from dexstation.profiles.compose import load_task


def test_weight_order_ties_and_unweighted():
    panes = [dict(id='z', title='Z'), dict(id='b', title='B', weight=10),
             dict(id='a', title='A', weight=10), dict(id='late', title='Late', weight=1000),
             dict(id='early', title='Early', weight=-1), dict(id='unweighted', title='Unweighted')]
    assert [p['id'] for p in plugins.ordered_panes(panes)] == ['early','a','b','late','unweighted','z']
    assert [p['id'] for p in plugins.ordered_panes(panes, {'z':0, 'b':-2})] == ['b','early','z','a','late','unweighted']
    assert 'weight' not in panes[0]  # Overrides do not mutate plugin declarations.


def test_default_panes_and_site_override(tmp_path, monkeypatch):
    config = tmp_path / 'site.json'
    config.write_text(json.dumps({'web':{'pane_weights':{'gamepad_signals':5,'status':-1}}}))
    monkeypatch.setenv('DEXSTATION_SITE_CONFIG',str(config))
    weights = site.load_site(reload=True)['web']['pane_weights']
    panes = plugins.task_panes(load_task('xbox_gamepad_v3_3'), weights)
    assert [p['id'] for p in panes] == ['status','gamepad_signals','face','back','left','right','table','autopilot','arm_takeover']
    assert [p['id'] for p in plugins.task_panes(load_task('empty'))] == ['table','arm_takeover','status']
    assert [p['id'] for p in plugins.task_panes(load_task('tabletop_cube_grasp'))] == ['table','arm_takeover','status']


@pytest.mark.parametrize('weights,match', [({'unknown_pane':1},'unknown_pane'),
    ({'status':True},'integer'),({'status':1.5},'integer'),({'status':'10'},'integer')])
def test_bad_pane_override_fails_startup(tmp_path, monkeypatch, weights, match):
    from dexstation.server.app import make_app, SessionManager
    config = tmp_path / 'site.json'
    config.write_text(json.dumps({'web':{'pane_weights':weights}}))
    monkeypatch.setenv('DEXSTATION_SITE_CONFIG',str(config))
    site._load_site.cache_clear()
    with pytest.raises(ValueError,match=match):
        make_app(SessionManager(tmp_path/'sessions'))


def test_camera_schema_and_reference_validation(tmp_path):
    task = load_task('xbox_gamepad_v3_3')
    from dexstation.profiles.compose import _schema
    for key, value, message in [('object','missing','Unknown auxiliary camera object'),
        ('atlas_slot',5,'outside range'),('framing_margin',1,'outside range')]:
        data=json.loads(json.dumps(task));data['auxiliary_cameras'][0][key]=value
        with pytest.raises(ValueError,match=message):_schema(data,'task')
    data=json.loads(json.dumps(task));data['panes'][0]['camera']='missing'
    with pytest.raises(ValueError,match='Unknown pane camera'):_schema(data,'task')
