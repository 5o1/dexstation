"""Shared countdown, one pending capture, frozen state and bounded publication."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import time
import pytest
from dexstation.render.snapshots import SceneSnapshots, SnapshotConflict


def test_countdown_cancel_concurrent_and_frozen_capture(tmp_path,monkeypatch):
    async def run():
        import dexstation.render.snapshots as module
        now=[time.time_ns()]
        sleeping=asyncio.Queue();release=asyncio.Event()
        async def sleep(delay):
            sleeping.put_nowait(delay)
            await release.wait()
            now[0]+=round(delay*1e9)
        monkeypatch.setattr(module,'time',SimpleNamespace(time_ns=lambda:now[0],
            monotonic_ns=lambda:now[0],strftime=time.strftime,localtime=time.localtime))
        monkeypatch.setattr(module,'asyncio',SimpleNamespace(sleep=sleep,
            create_task=asyncio.create_task,gather=asyncio.gather,CancelledError=asyncio.CancelledError))
        reads=[];captures=[]
        def reader(run):
            reads.append(now[0])
            return dict(step=17,applied_ns=now[0],arm_q=[[1]*7,[2]*7],hand_q=[[3]*21,[4]*21],
                        snapshot_inputs={'left_hand':dict(raw={'position':[42]*20},received_monotonic_ns=now[0])})
        async def capture(run,folder,identifier):
            captures.append(json.loads((folder/'state.json').read_text()))
            for name in ['camera_3','aux_face','aux_back','aux_left','aux_right','aux_table']:
                (folder/(name+'.png')).write_bytes(b'fake independent PNG')
            return dict(camera_count=6,cameras={})
        service=SceneSnapshots(tmp_path/'snapshots',reader=reader,capture=capture)
        service.start(tmp_path,delay_s=.06,label='reference')
        assert service.status()['state']=='countdown' and service.status()['remaining_s']==.06
        assert await asyncio.wait_for(sleeping.get(),2)==.06
        with pytest.raises(SnapshotConflict):service.start(tmp_path,delay_s=0)
        await service.cancel()
        assert not reads and service.status()['state']=='cancelled'
        service.start(tmp_path,delay_s=.04,label='left / stick')
        before=now[0]
        assert await asyncio.wait_for(sleeping.get(),2)==.04
        assert not reads
        release.set();await service.task
        assert reads==[before+40_000_000]
        assert captures[0]['step']==17 and captures[0]['hand_q'][1]==[4]*21
        result=service.status();assert result['state']=='saved'
        target=Path(result['path']);assert len(list(target.glob('*.png')))==6
        assert result['bytes']<=service.max_bytes
        inputs=json.loads((target/'inputs.json').read_text());assert inputs['left_hand']['raw']['position']==[42]*20 and inputs['left_hand']['age_s']>=0
        assert len(service.list())==1
        assert service.file(result['name']+'/meta.json')==target/'meta.json'
        for name in ('../outside','/etc/passwd','.pending/id'):
            with pytest.raises(FileNotFoundError):service.file(name)
        await service.close()
    asyncio.run(run())


def test_failed_or_oversized_snapshot_is_not_published(tmp_path):
    async def run():
        async def capture(run,folder,identifier):
            (folder/'too-large.png').write_bytes(b'x'*3000)
            return {}
        service=SceneSnapshots(tmp_path/'snapshots',max_bytes=2000,
            reader=lambda run:dict(applied_ns=time.monotonic_ns()),capture=capture)
        service.start(tmp_path,delay_s=0);await service.task
        assert service.status()['state']=='failed' and not service.list()
        assert not list(service.root.glob('*/.pending-*'))
        service.reader=lambda run:dict(applied_ns=time.monotonic_ns()-2_000_000_000)
        service.start(tmp_path,delay_s=0);await service.task
        assert service.status()['state']=='failed' and 'older' in service.status()['error']
        for value in (True,-1,float('nan'),3601):
            with pytest.raises(ValueError):service.start(tmp_path,delay_s=value)
    asyncio.run(run())


def test_cannot_cancel_capture_after_state_has_been_frozen(tmp_path):
    async def run():
        captured=asyncio.Event();release=asyncio.Event()
        async def capture(run,folder,identifier):
            captured.set();await release.wait();return {}
        service=SceneSnapshots(tmp_path,reader=lambda run:dict(applied_ns=time.monotonic_ns()),capture=capture)
        service.start(tmp_path,delay_s=0);await captured.wait()
        with pytest.raises(SnapshotConflict):await service.cancel()
        with pytest.raises(SnapshotConflict):service.start(tmp_path)
        release.set();await service.task
    asyncio.run(run())
