from helpers import bench_data, bench_spec
"""WP4 disk contracts, integrity controls and interrupted-journal recovery."""
from helpers import metadata, writer_for, observations, append
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dexstation.contracts.backend import FakeBackend
from dexstation.contracts.recording import EpisodeWriterProtocol, MemoryEpisodeWriter, WORKBENCH_ROW_KEYS
from dexstation.recording.episode import EpisodeWriter, read_episode
from dexstation.recording.integrity import check_integrity, check_and_store, recover_episode

EXAMPLES = Path(__file__).resolve().parents[1]/'src/dexstation/contracts/examples'










def saved(path, task='empty'):
    writer=writer_for(path,task)
    for obs in observations(task):append(writer,obs)
    writer.close()
    return path


def update_json(path, change):
    data=json.loads(path.read_text());change(data);path.write_text(json.dumps(data)+'\n')


def update_rows(path, change):
    data=[json.loads(line) for line in path.read_text().splitlines()]
    change(data);path.write_text(''.join(json.dumps(row)+'\n' for row in data))


@pytest.mark.parametrize('task',['empty','xbox_gamepad_v3_3'])
def test_disk_matches_memory_protocol(tmp_path,task):
    writer=writer_for(tmp_path/'ep',task)
    memory=MemoryEpisodeWriter(task_id=task,task_format=writer.task_format)
    assert isinstance(writer,EpisodeWriterProtocol)
    for obs in observations(task):
        append(writer,obs);append(memory,obs)
    writer.event('workbench','takeover',{'side':'left'})
    memory.event('workbench','takeover',{'side':'left'})
    if task!='empty':writer.event('task','reset',{'seed':17});memory.event('task','reset',{'seed':17})
    writer.note('sample');writer.close();memory.close()
    result=read_episode(writer.path)
    expected=json.loads(json.dumps(memory.episode.joined(),default=lambda a:a.tolist()))
    assert result.joined()==expected
    assert result.workbench.events[0]==memory.episode.workbench.events[0]
    assert result.notes==['sample']
    assert set(result.workbench.rows[0])==set(WORKBENCH_ROW_KEYS)
    if task=='empty':
        assert result.task is None and not (writer.path/'task').exists()
        assert [layer['name'] for layer in result.meta['layers']]==['workbench']
    else:
        for actual, original in zip(result.task.rows, observations(task)):
            assert np.asarray(actual['gamepad_signal'],np.float64).tobytes()==original.task['gamepad_signal'].tobytes()
            assert actual['gamepad_mask']==original.task['gamepad_mask'].tolist()
            assert np.asarray(list(actual['control_joint_positions'].values())).tobytes()==np.asarray(list(original.task['control_joint_positions'].values())).tobytes()
        assert result.task.events==memory.episode.task.events
    assert not (writer.path/'meta.json').exists()
    assert not (writer.path/'steps.jsonl').exists()


@pytest.mark.parametrize('task,damage',[
    (task,damage)
    for task in ('empty','xbox_gamepad_v3_3')
    for damage in ('none','truncated','corrupt','manifest_count','workbench_count','step_gap',
                   'clock_equal','receiver_count','row_duration','empty','task_count','task_time','task_order')
    if task!='empty' or not damage.startswith('task_')
])
def test_layered_integrity_positive_and_negative_controls(tmp_path,task,damage):
    path=saved(tmp_path/'ep',task)
    steps=path/'workbench/steps.jsonl'
    if damage=='truncated':steps.write_bytes(steps.read_bytes()[:-25])
    elif damage=='corrupt':steps.write_bytes(b'broken\n'+steps.read_bytes())
    elif damage=='manifest_count':update_json(path/'episode.json',lambda m:m.update(rows=3))
    elif damage=='workbench_count':update_rows(steps,lambda r:r.pop())
    elif damage=='step_gap':update_rows(steps,lambda r:r[1].update(step=2))
    elif damage=='clock_equal':update_rows(steps,lambda r:r[1].update(sim_time_s=r[0]['sim_time_s']))
    elif damage=='receiver_count':update_json(path/'workbench/meta.json',lambda m:m.update(received_input_messages=1))
    elif damage=='row_duration':update_json(path/'workbench/meta.json',lambda m:m.update(recording_end_sim_time_s=10.))
    elif damage=='empty':
        steps.write_text('');update_json(path/'episode.json',lambda m:m.update(rows=0))
        if task!='empty':(path/'task'/task/'steps.jsonl').write_text('')
    elif damage.startswith('task_'):
        target=path/'task'/task/'steps.jsonl'
        if damage=='task_count':update_rows(target,lambda r:r.pop())
        elif damage=='task_time':update_rows(target,lambda r:r[1].update(sim_time_s=.041))
        else:update_rows(target,lambda r:r.reverse())
    result=check_integrity(path)
    assert result['passed'] is (damage=='none'),result
    if damage in {'truncated','corrupt','manifest_count','workbench_count','step_gap','clock_equal','task_count','task_time','task_order'}:
        with pytest.raises(ValueError):read_episode(path)


def test_layered_arm_clamps_roundtrip_and_invalid_mask(tmp_path):
    writer=writer_for(tmp_path/'ep')
    flags={'left':None,'right':{'absolute':[True]+[False]*6,'startup':[False,True]+[False]*5}}
    append(writer,next(observations()),clamps=flags);writer.close()
    result=read_episode(writer.path)
    assert result.workbench.rows[0]['arm_clamps']==flags
    assert result.joined()[0]['workbench']['arm_clamps']==flags
    update_rows(writer.path/'workbench/steps.jsonl',lambda r:r[0]['arm_clamps']['right']['absolute'].__setitem__(0,1))
    with pytest.raises(ValueError,match='clamp'):read_episode(writer.path)


def test_crash_between_layers_recovers_only_common_prefix(tmp_path):
    path=tmp_path/'crashed'
    script='''
import os
from pathlib import Path
from dexstation.contracts.backend import FakeBackend
from dexstation.recording.episode import EpisodeWriter
writer=EpisodeWriter(Path(__import__('sys').argv[1]),{'scene_spec_text':'{}'},task_id='cube',task_format='cube-rec-v1')
backend=FakeBackend()
for i in range(2):
    backend.step(.02)
    writer.append(backend.observe(),{},applied_monotonic_ns=0,input_sequences={},arm_clamps={'left':None,'right':None})
class BrokenStream:
    def write(self,text):
        stream=writer.streams['task_original']
        stream.write(text[:len(text)//2]);stream.flush()
        os._exit(23)
writer.streams['task_original']=writer.streams['task']
writer.streams['task']=BrokenStream()
backend.step(.02)
writer.append(backend.observe(),{},applied_monotonic_ns=0,input_sequences={},arm_clamps={'left':None,'right':None})
'''
    process=subprocess.run([sys.executable,'-c',script,str(path)],check=False)
    assert process.returncode==23
    before={str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='truncated|interrupted|counts'):read_episode(path)
    assert before=={str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}
    assert not check_integrity(path)['passed']
    result=recover_episode(path)
    assert result.meta['status']=='interrupted'
    assert len(result.joined())==len(result.workbench.rows)==len(result.task.rows)==2
    audit=json.loads((path/'integrity.json').read_text())['recovery']
    assert audit['recovered_rows']==2
    assert audit['journals']['workbench/steps.jsonl']['retained_rows']==2
    check_and_store(path)
    assert json.loads((path/'integrity.json').read_text())['recovery']==audit


@pytest.mark.parametrize('damage',['time','step','count','events','commands'])
def test_recovery_stops_at_first_invalid_aligned_row(tmp_path,damage):
    path=saved(tmp_path/'ep','xbox_gamepad_v3_3')
    update_json(path/'episode.json',lambda m:m.update(status='recording'))
    target=path/'task/xbox_gamepad_v3_3/steps.jsonl'
    if damage=='time':update_rows(target,lambda r:r[1].update(sim_time_s=.041))
    elif damage=='step':update_rows(target,lambda r:r[1].update(step=5))
    elif damage=='count':update_rows(target,lambda r:r.pop())
    else:
        file=path/('workbench/events.jsonl' if damage=='events' else 'workbench/commands.jsonl')
        with file.open('ab') as stream:stream.write(b'{"incomplete":')
    episode=recover_episode(path)
    assert len(episode.joined())==(2 if damage in {'events','commands'} else 1)
    assert episode.meta['status']=='interrupted'


def test_empty_interruption_and_completed_corruption_require_explicit_recovery(tmp_path):
    path=saved(tmp_path/'ep')
    update_json(path/'episode.json',lambda m:m.update(status='recording',rows=0))
    with pytest.raises(ValueError,match='interrupted'):read_episode(path)
    result=recover_episode(path)
    assert result.task is None and len(result.workbench.rows)==2
    path=saved(tmp_path/'completed')
    steps=path/'workbench/steps.jsonl';steps.write_bytes(steps.read_bytes()[:-25])
    with pytest.raises(ValueError,match='truncated'):read_episode(path)
    result=recover_episode(path)
    assert result.meta['status']=='interrupted' and len(result.workbench.rows)==1
    assert json.loads((path/'integrity.json').read_text())['recovery']['original_status']=='complete'


def test_joined_asserts_clocks_after_in_memory_edit(tmp_path):
    episode=read_episode(saved(tmp_path/'ep','xbox_gamepad_v3_3'))
    episode.task.rows[1]['sim_time_s']+=.001
    with pytest.raises(ValueError,match='times differ'):episode.joined()


def test_task_namespace_and_preencoding_are_atomic_at_validation(tmp_path):
    writer=writer_for(tmp_path/'ep','xbox_gamepad_v3_3')
    obs=next(observations('xbox_gamepad_v3_3'))
    obs.task['step']=999
    with pytest.raises(ValueError,match='clock'):append(writer,obs)
    del obs.task['step'];obs.task['bad']=float('nan')
    with pytest.raises(ValueError):append(writer,obs)
    assert writer.stream.tell()==0 and writer.meta['rows']==0
    writer.close('interrupted')
    writer=writer_for(tmp_path/'empty')
    with pytest.raises(ValueError,match='Empty'):append(writer,next(observations('xbox_gamepad_v3_3')))
    with pytest.raises(ValueError,match='absent'):writer.event('task','reset',{})
    writer.close()
    with pytest.raises(ValueError,match='closed'):append(writer,next(observations()))


def test_frozen_profiles_and_manifest_identity(tmp_path):
    path=saved(tmp_path/'ep','xbox_gamepad_v3_3')
    episode=read_episode(path)
    assert episode.meta['identity']==metadata('xbox_gamepad_v3_3')['identity']
    update_json(path/'task/xbox_gamepad_v3_3/meta.json',lambda m:m.update(task_profile_text='changed'))
    with pytest.raises(ValueError,match='hash'):read_episode(path)


def test_initial_states_are_separate(tmp_path):
    meta=metadata('xbox_gamepad_v3_3')
    meta['motion_initial_state']={'workbench':{'arm':'wb-only'},'task':{'button':'task-only'}}
    writer=EpisodeWriter(tmp_path/'ep',meta,task_format='xbox-gamepad-rec-v1')
    for obs in observations('xbox_gamepad_v3_3'):append(writer,obs)
    writer.close()
    episode=read_episode(writer.path)
    assert episode.workbench.meta['initial_state']=={'arm':'wb-only'}
    assert episode.task.meta['initial_state']=={'button':'task-only'}
    assert 'task-only' not in (writer.path/'workbench/meta.json').read_text()


def test_manifest_matches_episode_schema(tmp_path):
    from helpers import validate_instance, load_schema
    for task in ('empty','xbox_gamepad_v3_3'):
        path=saved(tmp_path/task,task)
        validate_instance(json.loads((path/'episode.json').read_text()),load_schema('episode'))


def test_receive_sequence_gap_roundtrips(tmp_path):
    from dexstation.inputs.replay import command_event
    from dexstation.control.commands import ArmCommand
    writer=writer_for(tmp_path/'ep')
    events=[command_event(ArmCommand('left',(0.,)*7,0.),i) for i in (0,2,3)]
    writer.append_inputs(events)
    writer.meta.update(received_input_messages=3,receive_start_sequence=0,receive_end_sequence=4)
    for obs in observations():append(writer,obs)
    writer.close()
    result=check_integrity(writer.path)
    assert not result['passed'] and 'gap' in result['errors'][0]


def test_session_uses_runtime_scene_and_fake_backend(tmp_path,monkeypatch):
    from dexstation.recording.session import Session
    from dexstation.contracts.profiles import RuntimeScene
    from dexstation.model.spec import spec_from_dict
    from dataclasses import replace
    from concurrent.futures import Future
    root=Path(__file__).resolve().parents[1]
    spec=spec_from_dict(bench_data())
    spec=replace(spec,gamepad=None)
    task=json.loads((EXAMPLES/'empty.json').read_text())
    scene=RuntimeScene(spec,{},task,(),(),(),(),{},Path('limits'),Path('takeover'),{},metadata()['identity'],
                       {'workbench_text':'{}','task_text':json.dumps(task)})
    backend=FakeBackend()
    monkeypatch.setattr('dexstation.recording.session.fingertip_offsets',lambda spec:{})
    session=Session(tmp_path/'session',scene,backend,{'kind':'synthetic'},fps=50)
    # The real controller owns independent receiver counters. This fake
    # controller provides them without starting ROS or a hardware process.
    session.meta.update(session=metadata()['session'])
    class Pool:
        def submit(self,function,path):
            future=Future();future.set_result(function(path));return future
        def shutdown(self,**kwargs):pass
    session.save_pool=Pool()
    session.control({'action':'start'})
    session.writer.meta.update(received_input_messages=0,receive_start_sequence=0,receive_end_sequence=0)
    for _ in range(2):
        backend.step(.02)
        snapshot=session.record(backend.observe(),[])
        json.dumps(snapshot)
        assert snapshot['overlay'] is None
    session.control({'action':'stop'});session.wait_saved()
    assert session.status()['state']=='saved'
    episode=read_episode(session.output/'episode-000000')
    assert episode.task is None
    assert episode.workbench.meta['initial_state']['robot']['sim_time_s']==0.
    assert episode.workbench.meta['workbench_profile_text']=='{}'
    session.close()


def test_layered_integrity_cli(tmp_path,monkeypatch):
    source=saved(tmp_path/'ep','xbox_gamepad_v3_3')
    output=tmp_path/'integrity.json'
    script=Path(__file__).resolve().parents[1]/'scripts/check_collection.py'
    import runpy
    monkeypatch.setattr(sys,'argv',[str(script),'integrity',str(source),'--output',str(output)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script),run_name='__main__')
    assert result.value.code==0
    assert json.loads(output.read_text())['checks']['layer_alignment']


def test_gamepad_asset_and_solver_metadata_belong_to_task_layer(tmp_path):
    meta=metadata('xbox_gamepad_v3_3')
    meta.update(asset_versions={'robot':'test-robot','gamepad':'0.4.3'},gamepad_solver={'position_iterations':8})
    writer=EpisodeWriter(tmp_path/'ep',meta,task_format='xbox-gamepad-rec-v1')
    for obs in observations('xbox_gamepad_v3_3'):append(writer,obs)
    writer.close()
    episode=read_episode(writer.path)
    assert episode.workbench.meta['asset_versions']=={'robot':'test-robot'}
    assert 'gamepad_solver' not in episode.workbench.meta
    assert episode.task.meta['asset_versions']=={'gamepad':'0.4.3'}
    assert episode.task.meta['gamepad_solver']=={'position_iterations':8}


def test_joined_keeps_identically_named_task_channels_separate(tmp_path):
    writer=writer_for(tmp_path/'ep','xbox_gamepad_v3_3')
    obs=next(observations('xbox_gamepad_v3_3'));obs.task['commands']={'custom':'task-only'}
    append(writer,obs,commands={'robot':'workbench-only'});writer.close()
    episode=read_episode(writer.path)
    assert episode.joined()[0]['task']['commands']=={'custom':'task-only'}
    assert episode.joined()[0]['workbench']['commands']=={'robot':'workbench-only'}
    _,robot_rows=episode
    assert robot_rows[0]['commands']=={'robot':'workbench-only'}


def test_recover_cli_keeps_the_original_and_refuses_a_second_backup(tmp_path):
    from dexstation.recording.integrity import main
    path=saved(tmp_path/'ep','xbox_gamepad_v3_3')
    update_json(path/'episode.json',lambda m:m.update(status='recording'))
    update_rows(path/'task/xbox_gamepad_v3_3/steps.jsonl',lambda r:r.pop())
    before={str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}
    assert main([str(path)])==0
    backup=path.with_name('ep.before-recover')
    assert before=={str(p.relative_to(backup)):p.read_bytes() for p in backup.rglob('*') if p.is_file()}
    assert read_episode(path).meta['status']=='interrupted'
    with pytest.raises(SystemExit):main([str(path)])
