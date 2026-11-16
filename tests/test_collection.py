"""Collection contracts: separate input/target/measurement, unit and IO boundaries."""
from helpers import observation
import json
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest

from dexstation.inputs.arm_mapping import ArmMapper
from dexstation.inputs.replay import Ros2Source, CollectionReplay
from dexstation.control.commands import arm_command_from_packet, HandCommand, ArmCommand
from dexstation.examples.ar5_g20.mapping import G20_JOINT_NAMES
from dexstation.recording.episode import Observation, EpisodeWriter, read_episode
from dexstation.contracts.recording import EPISODE_FORMAT
from dexstation.control.protocol import ArmCommandPacket, encode, decode
from dexstation.model.contact import has_robot_contact


def gamepad_writer(path, meta):
    return EpisodeWriter(path, meta, task_id='xbox_gamepad_v3_3', task_format='xbox-gamepad-rec-v1',
                         task_meta={'task_profile':{'title':'Xbox gamepad on fixture v3.3'}})




@pytest.mark.parametrize("samples,expected", [([], False), ([(.02, True)], False),
    ([(0., True)], True), ([(-1e-9, True)], True), ([(-.01, False)], False),
    ([(.001, True), (-.01, False)], False), ([(.001, False), (0., True)], True)])
def test_shared_geometric_contact(samples, expected):
    assert has_robot_contact(iter(samples)) is expected


def test_nonfinite_contact_distance():
    with pytest.raises(ValueError):
        has_robot_contact([(float("nan"), True)])


def test_units_and_legacy_packet():
    mapper = ArmMapper({s: [0.] * 7 for s in ("left", "right")}, limit_around_initial=False)
    old = b'{"type":"arm_command","side":"left","position":[1,2,3,4,5,6,7],"stamp":1}'
    assert arm_command_from_packet(decode(old)).positions == (1,2,3,4,5,6,7)
    new = decode(encode(ArmCommandPacket("left", [0.] * 7, 1., "master-deg")))
    c = arm_command_from_packet(new, mapper)
    assert np.allclose(c.positions[:3], np.deg2rad([90,90,-90]))
    assert c.raw_master_deg == (0.,) * 7
    with pytest.raises(ValueError, match="startup"):
        arm_command_from_packet(new)
    with pytest.raises(ValueError):
        decode(old.replace(b'"stamp":1', b'"stamp":NaN'))


def test_subscriber_only_and_stale():
    callbacks = {}
    class Node:
        def create_subscription(self, typ, topic, callback, qos):
            callbacks[topic] = callback
        def create_publisher(self, *a):
            raise AssertionError("subscriber must never publish")
        def destroy_node(self):
            pass
    clock = [10.]
    mapper = ArmMapper({s: [0.] * 7 for s in ("left", "right")})
    source = Ros2Source(mapper, node=Node(), joint_state_type=object, clock=lambda: clock[0])
    assert len(callbacks) == 4
    msg = SimpleNamespace(position=[0.] * 7, name=[])
    callbacks["/right_arm_joint_control"](msg)
    assert len(source.poll(10.)) == 1
    assert source.poll(11.) == []
    callbacks["/cb_left_hand_control_cmd"](SimpleNamespace(position=[0.] * 3, name=[]))
    assert source.decode_errors == 1
    source.close()


def test_missing_ros_is_friendly(monkeypatch):
    import dexstation.inputs.replay as module
    def missing(name):
        raise ImportError(name)
    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="ROS Humble"):
        Ros2Source(None)


def test_udp_master_uses_the_same_mapper():
    import socket
    from dexstation.inputs.udp import UdpSource
    mapper = ArmMapper({s: [0.] * 7 for s in ("left", "right")})
    source = UdpSource("127.0.0.1", 0, stale_after_s=.25, arm_mapper=mapper)
    packet = ArmCommandPacket("right", [0.] * 7, 123., "master-deg")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(encode(packet), source.bound_address)
        commands = source.poll(10.)
        assert commands == [mapper.command("right", [0.] * 7, 123.)]
        assert source.decode_errors == 0
    finally:
        source.close()


def test_episode_roundtrip_and_invalid_pose(tmp_path):
    writer = gamepad_writer(tmp_path / "episode", {"scene_spec_text": "{}", "fps":50})
    obs = observation()
    writer.append(obs, {}, {"values": [0.] * 32})
    writer.note("operator note")
    writer.close()
    meta, rows = read_episode(tmp_path / "episode")
    assert meta["format"] == EPISODE_FORMAT
    assert meta["operator_notes"][0]["text"] == "operator note"
    assert read_episode(tmp_path / "episode").task.rows[0]["control_joint_positions"]["button"] == .001
    obs.palm_pose_world[0,3] = 0.
    with pytest.raises(ValueError, match="quaternion"):
        obs.validate()
    path = tmp_path / "episode/workbench/meta.json"
    meta = json.loads(path.read_text())
    meta["scene_spec_text"] = "changed"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="hash"):
        read_episode(path.parent.parent)


def test_replay_final_target_has_no_master_offset(tmp_path):
    names = [f"{s}_arm_joint_{j}" for s in ("left","right") for j in range(1,8)] + [f"{s}_hand_motor_{j}" for s in ("left","right") for j in range(20)]
    a = np.zeros((3,54)); a[:,14:] = 127.
    path = tmp_path / "replay.npz"
    np.savez(path, action=a, timestamp=[0.,.1,.2], names=names,
             units=["rad"]*14 + ["motor_0_255"]*40, episode_index=[0,0,0])
    source = CollectionReplay(path, loop=True)
    commands = source.poll(10.)
    assert len(commands) == 4
    assert commands[0].positions == (0.,) * 7
    assert commands[0].raw_master_deg is None
    assert commands[1].motor_positions == (127.,) * 20
    source.poll(10.31)
    assert source.cycles == 1


def test_own_episode_replay_preserves_commands_not_measurements(tmp_path):
    commands = {}
    for side in ("left", "right"):
        commands[side+"_hand"] = asdict(HandCommand(side, dict.fromkeys(G20_JOINT_NAMES, .1), 1., (127.,)*20))
        commands[side+"_arm"] = asdict(ArmCommand(side, (.2,)*7, 1.))
    path = tmp_path / "own"
    writer = gamepad_writer(path, {"scene_spec_text": "{}", "fps": 50})
    for t in (.02, .04):
        writer.append(observation(t), commands, {})
    writer.close()
    source = CollectionReplay(path)
    replayed = source.poll(100.)
    assert len(replayed) == 4
    assert replayed[0].joint_positions["thumb_mcp"] == .1  # actual measurement in fixture is zero
    assert replayed[1].positions == (.2,)*7
    assert replayed[1].raw_master_deg is None


def test_operator_notes_do_not_leak_between_episodes(tmp_path):
    shared_meta = {"scene_spec_text": "{}", "operator_notes": []}
    for index, note in enumerate(("first", "second")):
        writer = gamepad_writer(tmp_path / str(index), shared_meta)
        writer.note(note)
        writer.append(observation(), {}, {})
        writer.close()
    assert shared_meta["operator_notes"] == []
    for index, note in enumerate(("first", "second")):
        meta, _ = read_episode(tmp_path / str(index))
        assert [entry["text"] for entry in meta["operator_notes"]] == [note]


def test_own_replay_uses_control_grid_not_quantized_observation_times(tmp_path):
    # A 1 ms physics grid emits .033, .066, .100 for 30 Hz control. Scheduling
    # actions by these post-step observation times would delay the third target.
    path = tmp_path / "quantized"
    writer = gamepad_writer(path, {"scene_spec_text": "{}", "fps": 30})
    for i, t in enumerate((.033, .066, .100)):
        commands = {}
        for side in ("left", "right"):
            commands[side+"_hand"] = asdict(HandCommand(side, dict.fromkeys(G20_JOINT_NAMES, i*.1), 1., (127.,)*20))
            commands[side+"_arm"] = asdict(ArmCommand(side, (.2,)*7, 1.))
        writer.append(observation(t), commands, {})
    writer.close()
    source = CollectionReplay(path)
    for i in range(3):
        commands = source.poll(100. + i/30)
        assert len(commands) == 4
        assert commands[0].joint_positions["thumb_mcp"] == i*.1


def test_raw_input_events_roundtrip_separate_from_protocol_values(tmp_path):
    from dexstation.recording.episode import read_input_events
    callbacks = {}
    class Node:
        def create_subscription(self, typ, topic, callback, qos):
            assert qos >= 100
            callbacks[topic] = callback
        def destroy_node(self):
            pass
    source = Ros2Source(ArmMapper({s:[0.]*7 for s in ("left","right")}), node=Node(), joint_state_type=object)
    raw = [127.125, -0.] + [0.] * 18
    message = SimpleNamespace(position=raw, name=[], velocity=[], effort=[],
                              header=SimpleNamespace(stamp=SimpleNamespace(sec=123, nanosec=456), frame_id="sample"))
    for _ in range(3):
        callbacks["/cb_left_hand_control_cmd"](message)
    assert len(source.poll(source.clock())) == 1  # latest control, all three events retained
    from dexstation.inputs.replay import snapshot_input_events
    latest=snapshot_input_events(source)
    assert latest['left_hand']['sequence']==2 and latest['left_hand']['raw']['position']==raw
    latest['left_hand']['raw']['position'][0]=999
    assert snapshot_input_events(source)['left_hand']['raw']['position'][0]==127.125
    events=source.drain_events()
    assert len(events)==3 and source.drain_events()==[]
    assert events[0]["raw"]["position"] == raw
    assert events[0]["converted_command"]["motor_positions"][0] == 127.
    path=tmp_path/"v3"
    writer=gamepad_writer(path,{"scene_spec_text":"{}"})
    writer.append_inputs(events)
    writer.append(observation(), {}, {}, applied_monotonic_ns=99, input_sequences={"left_hand":2})
    writer.close()
    assert read_input_events(path)==json.loads(json.dumps(events))
    assert np.asarray(read_input_events(path)[0]["raw"]["position"],np.float64).tobytes()==np.asarray(raw,np.float64).tobytes()
    meta, rows=read_episode(path)
    assert meta["input_messages"]==3 and rows[0]["input_sequences"]=={"left_hand":2}
    with (path/"workbench/commands.jsonl").open("a") as f:
        f.write(json.dumps(events[-1])+"\n")
    with pytest.raises(ValueError,match="length"):
        read_episode(path)
    source.close()


def test_archived_formats_are_not_readable(tmp_path):
    path=tmp_path/'old';path.mkdir()
    (path/'meta.json').write_text(json.dumps({'schema_version':4}))
    (path/'steps.jsonl').write_text('')
    with pytest.raises(ValueError,match='layered'):
        read_episode(path)



def test_missing_glove_interface_fails_before_ros_context_or_node(monkeypatch):
    import dexstation.inputs.replay as module
    def no_context():raise AssertionError('missing interface must fail before ROS context creation')
    rclpy=SimpleNamespace(ok=no_context,init=no_context,create_node=no_context)
    def import_module(name):
        if name=='rclpy':return rclpy
        if name=='sensor_msgs.msg':return SimpleNamespace(JointState=object)
        raise ModuleNotFoundError("No module named 'manus_ros2_msgs'")
    monkeypatch.setattr(module.importlib,'import_module',import_module)
    with pytest.raises(RuntimeError,match='manus_ros2_msgs.*independently installed ROS and Manus message packages'):
        module.Ros2Source(None)
