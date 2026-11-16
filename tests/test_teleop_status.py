"""Display telemetry only: effective targets, sampled duration and optionality."""
from dataclasses import asdict
import math
from types import SimpleNamespace

import pytest

from dexstation.runtime.controller import CollectionController, CollectionOptions
from dexstation.control.commands import ArmCommand, HandCommand
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES, G20_JOINT_LIMITS
from dexstation.control.joint_motion import JointMotion
from dexstation.control.teleop_status import TeleopMonitor, TELEOP_THRESHOLDS
from helpers import observation


def hand_bounds():
    return {s: [[G20_JOINT_LIMITS[s][n][i] for n in G20_JOINT_NAMES] for i in (0, 1)]
            for s in ('left', 'right')}


def test_signed_error_duration_recovery_and_missing_channel():
    m = TeleopMonitor();obs = observation()
    command = ArmCommand('right', (1.,)*7, 0.)
    commands = {'right_arm': asdict(command)}
    targets = {'right_arm': [math.radians(11.)]*7}
    def update(now):return m.update(obs, commands, targets, hand_bounds(), now)['arms']['right']
    assert update(1.)['error_deg'] == pytest.approx([-11.]*7)
    assert update(1.5)['error_duration_s'] == pytest.approx([.5]*7)
    assert update(1.51)['error_duration_s'] == pytest.approx([.51]*7)
    obs.arm_q[1, 2] = math.radians(2.)
    row = update(1.6)
    assert row['error_duration_s'][2] == 0.
    assert row['error_duration_s'][1] == pytest.approx(.6)
    obs.arm_q[1, 2] = 0
    assert update(1.7)['error_duration_s'][2] == 0.
    assert m.update(obs, {}, {}, hand_bounds(), 2.)['arms'] == {}
    assert update(3.)['error_duration_s'] == [0.]*7


def test_speed_limited_and_filtered_effective_target_does_not_false_alarm(tmp_path):
    spec = SimpleNamespace(**{s+'_'+k: SimpleNamespace(home_q=[0.]*n)
                           for s in ('left', 'right') for k, n in (('arm', 7), ('hand', 21))})
    motion = JointMotion(spec, cache=tmp_path)
    command = ArmCommand('right', (math.radians(100), 0, 0, 0, 0, 0, 0), 0.)
    motion.accept([command]);monitor = TeleopMonitor();obs = observation()
    for tick in range(31):
        now = tick/30
        applied = motion.advance_to(now)[0]
        obs.arm_q[1] = applied.positions
        row = monitor.update(obs, {'right_arm': asdict(command)}, {'right_arm': applied.positions}, hand_bounds(), now)
        assert row['arms']['right']['error_deg'] == pytest.approx([0.]*7)
        assert row['arms']['right']['error_duration_s'] == [0.]*7
    # The raw command is still far ahead, but is not what the drive is chasing.
    assert math.degrees(command.positions[0]-applied.positions[0]) > 70


def test_hand_max_error_duration_and_endpoint_flags():
    monitor = TeleopMonitor();obs = observation();bounds = hand_bounds()
    target = [(lo+hi)/2 for lo, hi in zip(*bounds['left'])]
    obs.hand_q[0] = target
    desired = dict(zip(G20_JOINT_NAMES, target))
    command = asdict(HandCommand('left', desired, 0.))
    def update(now):return monitor.update(obs, {'left_hand': command}, {'left_hand': target}, bounds, now)['hands']['left']['fingers']
    assert all(not f['limited'] and f['max_error_deg']==0 for f in update(0.).values())
    obs.hand_q[0, 5] -= math.radians(16.)
    assert update(1.)['index']['error_duration_s'] == 0
    assert update(1.51)['index']['error_duration_s'] == pytest.approx(.51)
    assert update(1.6)['index']['max_error_deg'] == pytest.approx(16.)
    assert update(1.6)['thumb']['error_duration_s'] == 0
    command['joint_positions']['thumb_ip'] = bounds['left'][1][4]
    assert update(1.7)['thumb']['limited']
    obs.hand_q[0, 5] = target[5]
    assert update(2.)['index']['error_duration_s'] == 0.


def test_telemetry_optional_and_status_does_not_advance_timers(tmp_path):
    c = CollectionController(CollectionOptions('sapien', 'udp', tmp_path, 1.))
    assert 'teleop' not in c.status()
    c.teleop_monitor.update(observation(), {}, {}, hand_bounds(), 0.)
    assert c.status()['teleop']['thresholds'] == TELEOP_THRESHOLDS
    first = c.status()['teleop']
    assert c.status()['teleop'] == first
    assert 'teleop' not in c.last_row  # Recording rows/schema stay untouched.


def test_hand_following_uses_backend_target_after_clipping():
    m = TeleopMonitor();obs = observation();bounds = hand_bounds()
    effective = [(lo+hi)/2 for lo, hi in zip(*bounds['right'])]
    desired = dict(zip(G20_JOINT_NAMES, effective));desired['thumb_ip'] = 99.
    effective[4] = bounds['right'][1][4];obs.hand_q[1] = effective
    for now in (0., 1.):
        row = m.update(obs, {'right_hand': asdict(HandCommand('right', desired, 0.))},
                       {'right_hand': effective}, bounds, now)['hands']['right']['fingers']['thumb']
        assert row['max_error_deg'] == row['error_duration_s'] == 0.
        assert row['limited']


def test_arm_input_status_preserves_measured_availability_and_reason():
    from dexstation.control.teleop_status import arm_input_status
    health=dict(left_arm=dict(input_available=False,input_reason='can9 down',linkerta={'joint_age_s':[None]*7}),
                right_arm=dict(input_available=True,input_reason=None))
    status=arm_input_status(health)
    assert status['left']['reason']=='can9 down' and not status['left']['available']
    assert status['right']['available'] and status['right']['reason'] is None
    assert arm_input_status({})=={}
