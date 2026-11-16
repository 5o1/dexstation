from helpers import bench_data, bench_spec
"""Deterministic startup race: manager timing and Session share one journal."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexstation.recording.session import Session
from dexstation.server.app import SessionManager
from dexstation.backends.sapien_physics import CollectionSapienBackend
from dexstation.model.spec import load_spec


@pytest.mark.parametrize('timing_phase',['before_journal_creation','before_first_control'])
def test_startup_timing_neither_creates_nor_overwrites_session_journal(tmp_path,monkeypatch,timing_phase):
    import dexstation.recording.session as module
    manager=SessionManager(tmp_path/'sessions')
    manager.output=tmp_path/'session';manager.started_monotonic_s=10.
    if timing_phase=='before_journal_creation':manager.timing('first_linkerta_data_s',10.1)
    def offsets(spec):
        # Session has claimed its directory but is still loading metadata.
        assert manager.output.is_dir()
        manager.flush_timings()
        assert not (manager.output/'events.jsonl').exists()
        return {}
    monkeypatch.setattr(module,'fingertip_offsets',offsets)
    monkeypatch.setattr(module.importlib.metadata,'version',lambda name:'fixture')
    spec=bench_spec()
    backend=CollectionSapienBackend.__new__(CollectionSapienBackend)
    backend.environment_collisions=False;backend.gamepad_collisions=False
    backend.scene=SimpleNamespace(get_timestep=lambda:.001)
    session=Session(manager.output,spec,backend,{'kind':'fixture'})
    try:
        if timing_phase=='before_first_control':manager.timing('first_linkerta_data_s',10.1)
        manager.flush_timings()
        assert not manager.timing_pending
        session.control({'action':'note','text':'first control'})
        manager.timing('first_frame_s',10.2)
        session.control({'action':'note','text':'second control'})
        manager.flush_timings()
        rows=[json.loads(line) for line in (manager.output/'events.jsonl').read_text().splitlines()]
        assert [r['action'] for r in rows]==['session/timing','note','session/timing','note']
        assert [r.get('name') for r in rows if r['action']=='session/timing']==['first_linkerta_data_s','first_frame_s']
        assert [r['text'] for r in rows if r['action']=='note']==['first control','second control']
    finally:session.close()


def test_manager_dispatches_page_startup_glove_settings_to_controller(tmp_path):
    import queue
    from dexstation.runtime.controller import CollectionController, CollectionOptions
    manager=SessionManager(tmp_path/'sessions')
    manager.worker=SimpleNamespace(is_alive=lambda:True)
    manager.controls=queue.Queue()
    controller=CollectionController(CollectionOptions('sapien','udp',tmp_path/'output',1.))
    layers={'enabled':True,'A':False,'C':True}
    manager.control({'action':'glove-overlay','layers':layers,'request_id':'page-startup'})
    controller.control(manager.controls.get_nowait())
    assert controller.glove_layers==layers
    assert controller.control_result['accepted'] and controller.control_result['request_id']=='page-startup'
    with pytest.raises(ValueError,match='expected start'):
        manager.control({'action':'unrecognized'})
