import json
from pathlib import Path

import numpy as np
import pytest

from dexstation.contracts.takeover import ArmState
from dexstation.control.takeover import ArmTakeover, load_takeover

PARAMS = dict(match_tolerance_deg=2., match_dwell_s=.2, deviation_budget_deg=10.)


def test_initial_waiting_both_sides_and_snapshot_roundtrip():
    original = ArmTakeover(PARAMS)
    assert {a['state'] for a in original.snapshot()['arms'].values()} == {'waiting'}
    original.update(1., 'left', np.zeros(7), np.zeros(7), True)
    original.update(1.2, 'left', None, np.zeros(7), True)
    restored = ArmTakeover(PARAMS)
    restored.restore(json.loads(json.dumps(original.snapshot())))
    assert restored.snapshot() == original.snapshot()
    assert 'STOP' not in ArmState.__members__


def test_matching_dwell_following_deviation_hold_and_rematch():
    t = ArmTakeover(PARAMS)
    zero = np.zeros(7)
    update = lambda time, target=zero, ok=True: t.update(time, 'left', target, zero, ok)
    assert update(0).state is ArmState.MATCHING
    assert not update(.1).release
    assert update(.21).release
    assert update(.22, np.radians([11, 0, 0, 0, 0, 0, 0])).state is ArmState.HOLD
    assert update(.23).state is ArmState.MATCHING
    assert update(.44).release
    assert update(.45, ok=False).state is ArmState.HOLD
    assert update(.46).state is ArmState.MATCHING
    assert update(.67).release
    decision = update(.68, None)
    assert decision.state is ArmState.WAITING and not decision.release
    assert update(.69).state is ArmState.MATCHING


def test_tolerance_must_hold_continuously_and_sides_independent():
    t = ArmTakeover(PARAMS)
    z = [0.]*7
    t.update(0, 'left', z, z, True)
    t.update(.15, 'left', np.radians([3]*7), z, True)
    assert not t.update(.3, 'left', z, z, True).release
    assert t.update(.51, 'left', z, z, True).release
    assert t.snapshot()['arms']['right']['state'] == 'waiting'
    with pytest.raises(ValueError, match='monotonic'):
        t.update(.1, 'left', z, z, True)


@pytest.mark.parametrize('key', list(PARAMS))
def test_null_refuses_enablement(key):
    with pytest.raises(ValueError, match='null'):
        ArmTakeover(dict(PARAMS, **{key: None}))


def test_workbench_configuration_enables_takeover():
    takeover = load_takeover(Path(__file__).resolve().parents[1]/'src/dexstation/examples/ar5_g20/data/takeover.json')
    assert {a['state'] for a in takeover.snapshot()['arms'].values()} == {'waiting'}


@pytest.mark.parametrize('starting_state', ['waiting','matching','following','hold'])
def test_missing_input_returns_waiting_from_each_state(starting_state):
    t=ArmTakeover(PARAMS);zero=np.zeros(7)
    if starting_state=='matching':t.update(0.,'left',np.radians([5.]*7),zero,True)
    elif starting_state in ('following','hold'):
        t.update(0.,'left',zero,zero,True);t.update(.21,'left',zero,zero,True)
        if starting_state=='hold':t.update(.22,'left',zero,zero,False)
    assert t.snapshot()['arms']['left']['state']==starting_state
    result=t.update(1.,'left',None,zero,False)
    assert result.state is ArmState.WAITING and not result.release
    assert t.snapshot()['arms']['right']['state']=='waiting'


@pytest.mark.parametrize('starting_state',['waiting','matching'])
def test_limit_enters_hold_before_following(starting_state):
    t=ArmTakeover(PARAMS);zero=np.zeros(7)
    if starting_state=='matching':t.update(0.,'left',np.radians([5.]*7),zero,True)
    result=t.update(.1,'left',zero,zero,False)
    assert result.state is ArmState.HOLD and not result.release and result.reason=='limit'


def test_zero_dwell_still_exposes_matching_after_hold():
    t=ArmTakeover(dict(PARAMS,match_dwell_s=0.));zero=np.zeros(7)
    assert t.update(0.,'left',zero,zero,True).state is ArmState.FOLLOWING
    assert t.update(.01,'left',zero,zero,False).state is ArmState.HOLD
    assert t.update(.02,'left',zero,zero,True).state is ArmState.MATCHING
    assert t.update(.03,'left',zero,zero,True).state is ArmState.FOLLOWING
