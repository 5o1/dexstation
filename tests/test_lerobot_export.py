"""Numeric LeRobot layer separation; optional local data-tools dependency."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from dexstation.control.commands import ArmCommand, HandCommand
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES
from dexstation.recording.lerobot_export import export_episode
from helpers import writer_for, observations, append


@pytest.mark.parametrize('task',['empty','xbox_gamepad_v3_3'])
def test_lerobot_uses_workbench_actions_and_task_namespaces(tmp_path,task):
    pq=pytest.importorskip('pyarrow.parquet')
    writer=writer_for(tmp_path/'source',task)
    commands={}
    for side in ('left','right'):
        commands[side+'_arm']=asdict(ArmCommand(side,(.2,)*7,0.))
        commands[side+'_hand']=asdict(HandCommand(side,dict.fromkeys(G20_JOINT_NAMES,.1),0.,(127.,)*20))
    for obs in observations(task):append(writer,obs,commands=commands)
    writer.close()
    result=export_episode(writer.path,tmp_path/'dataset')
    assert result['rows']==2 and result['features']['action']==[54]
    table=pq.read_table(tmp_path/'dataset/data/chunk-000/file-000.parquet')
    assert table['observation.state'].to_pylist()==[[0.]*56]*2
    assert table['action'].to_pylist()==np.asarray([[.2]*14+[127.]*40]*2,np.float32).tolist()
    keys=[key for key in table.column_names if key.startswith('observation.task.')]
    if task=='empty':assert keys==[]
    else:
        assert set(keys)=={'observation.task.gamepad_signal','observation.task.gamepad_mask','observation.task.control_joint_positions'}
        assert table['observation.task.gamepad_signal'].to_pylist()[0]==(np.arange(32)/31).tolist()
        info=json.loads((tmp_path/'dataset/meta/info.json').read_text())
        assert info['features']['observation.task.control_joint_positions']['names']==['axis','button']
    expected='Empty table' if task=='empty' else 'Xbox gamepad on fixture v3.3'
    assert pq.read_table(tmp_path/'dataset/meta/tasks.parquet')['__index_level_0__'].to_pylist()==[expected]
