"""No devices, ROS processes or vcan: fake buses and monotonic time only."""
import errno
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as N

import pytest

from dexstation.inputs.linkerta_reader import LinkertaReader, FRAME, QUERY_ID, interface_state, status_key, SocketCanBus

NEGATION = [1, -1, 1, -1, 1, -1, 1] + [1]*7
OFFSETS = [0.]*7 + [-15.] + [0.]*6


class FakeBus:
    def __init__(self):
        self.frames, self.sent, self.error, self.closed = [], [], None, False
    def send(self, can_id, data):
        self.sent.append((can_id, data))
        if self.error:raise self.error
    def recv(self):return self.frames.pop(0) if self.frames else None
    def close(self):self.closed = True


def frame(can_id, values):
    return can_id, b''.join(struct.pack('>h', round(v*10)) for v in values)


def rig(up=True):
    now, bus = [10.], FakeBus()
    state = dict(name='fakecan', up=up, operstate='up' if up else 'down', flags=1 if up else 0,
                 reason=None if up else 'fakecan down')
    reader = LinkertaReader(channel='fakecan', negation=NEGATION, offset_deg=OFFSETS,
                            bus_factory=lambda _:bus, inspect=lambda _:dict(state), clock=lambda:now[0])
    return reader, now, bus, state


@pytest.mark.parametrize('mode', ['down', 'send_failure', 'silent'])
def test_accident_never_publishes_without_measurement(mode):
    reader, now, bus, _ = rig(mode != 'down')
    if mode == 'send_failure':bus.error = OSError(errno.ENETDOWN, 'Network is down')
    for tick in range(101):
        now[0] = 10 + tick/10
        output, status = reader.step()
        assert output == {}
        assert all(not row['publishing'] and row['reason'] for row in status['sides'].values())
        assert all(math.isnan(v) for v in reader.positions)
        assert all(age is None for row in status['sides'].values() for age in row['joint_age_s'])
    if mode == 'down':assert not bus.sent and 'down' in status['sides']['left']['reason']
    if mode == 'send_failure':assert 'Network is down' in status['io_error']


def test_six_measured_joints_seventh_then_expiry():
    reader, now, bus, _ = rig()
    # At the per-joint measurement boundary, exactly six accepted measurements.
    # Wire replies carry 4+3 joints; truncated frames are rejected separately.
    reader.positions[:6] = [0.]*6
    reader.received[:6] = [now[0]]*6
    assert reader.step()[0] == {}
    bus.frames = [frame(0x66, [0, 0, 0])]
    assert reader.step()[0] == {'left': [0.]*7}
    now[0] += .29
    bus.frames = [frame(0x65, [0]*4)]
    reader.step()
    now[0] += .02
    output, status = reader.step()
    assert output == {} and 'left J5' in status['sides']['left']['reason']
    assert status['sides']['right']['publishing'] is False


def test_vendor_wire_protocol_mapping_sign_offset_and_degrees():
    reader, _, bus, _ = rig()
    values = [-170., -90., 0., 90., 179.9, 180., 360., -15., 22.2, -33.3, 44.4, 55.5, -66.6, 77.7]
    for fid, start, end in [(0x65,0,4),(0x66,4,7),(0x67,7,11),(0x68,11,14)]:
        bus.frames.append(frame(fid, values[start:end]))
    output, status = reader.step()
    assert bus.sent == [(QUERY_ID, b'')] and QUERY_ID == 0x04
    # Vendor: signed int16 big-endian / 10, wrap [-180,180), sign + offset.
    expected = [round((v+180)%360-180, 1)*n+o for v,n,o in zip(values,NEGATION,OFFSETS)]
    assert output['left'] + output['right'] == pytest.approx(expected)
    assert status['sides']['left']['joint_age_s'] == [0.]*7
    assert len(FRAME.pack(0x04, 0, b'\0'*8)) == 16


def test_rejected_jump_does_not_refresh_any_measurement_evidence():
    reader, now, bus, _ = rig()
    bus.frames = [frame(0x65, [-170, 0, 0, 0]), frame(0x66, [0]*3)]
    first, _ = reader.step()  # first -170 accepted, no fictional zero baseline
    assert first['left'][0] == -170
    original = reader.received[0]
    for elapsed in (.1, .2, .31):
        now[0] = 10+elapsed
        bus.frames = [frame(0x65, [60, 0, 0, 0]), frame(0x66, [0]*3)]
        output, status = reader.step()
        assert reader.positions[0] == -170 and reader.received[0] == original
        if elapsed < .3:assert output['left'] == first['left']
        else:assert output == {}
    left = status['sides']['left']
    assert left['rejected_counts'][0] == 3
    assert left['reason'] == 'left J1 reply rejected: jump 230.0 deg'
    bus.frames = [frame(0x65, [-160, 0, 0, 0])]
    assert reader.step()[0]['left'][0] == -160

    reader,now,bus,_=rig()
    bus.frames=[frame(0x65,[-170,0,0,0]),frame(0x66,[0]*3)]
    assert 'left' in reader.step()[0]
    now[0]+=.31
    # Both packets arrive, but one joint's >=180-degree jump is rejected.
    bus.frames=[frame(0x65,[60,1,1,1]),frame(0x66,[1]*3)]
    output,status=reader.step()
    assert output=={}
    assert sum(age <= .30 for age in status['sides']['left']['joint_age_s'])==6
    bus.frames=[frame(0x65,[-160,1,1,1])]
    assert 'left' in reader.step()[0]


def test_frames_cannot_invent_missing_bytes_or_accept_can_flags():
    reader, _, bus, _ = rig()
    bus.frames = [(0x65, b'\0'*6), (0x66, b'\0'*5), (0x80000065,b'\0'*8), (0x40000065,b'\0'*8), (0x64,b'\0'*8)]
    assert reader.step()[0] == {} and reader.received == [None]*14


def test_down_invalidates_cache_before_recovery_and_status_transition_key():
    reader, now, bus, state = rig()
    bus.frames = [frame(0x65,[0]*4),frame(0x66,[0]*3)]
    assert 'left' in reader.step()[0]
    state.update(up=False, operstate='down', reason='fakecan down')
    _, status = reader.step()
    assert status['sides']['left']['joint_age_s'] == [0.]*7
    key = status_key(status)
    now[0] += 1
    assert status_key(reader.step()[1]) == key
    state.update(up=True,operstate='up',reason=None)
    assert reader.step()[0] == {}  # old pre-DOWN measurements never resume


def test_sysfs_requires_flag_and_operstate(tmp_path):
    path = tmp_path/'renumbered';path.mkdir()
    for operstate, flags, expected in [('down','0x0',False),('up','0x0',False),('unknown','0x1',True),('up','0x1',True)]:
        (path/'operstate').write_text(operstate);(path/'flags').write_text(flags)
        assert interface_state(path.name,tmp_path)['up'] is expected
    assert not interface_state('missing',tmp_path)['up']


def test_socketcan_adapter_uses_exact_native_frame_with_fake_socket(monkeypatch):
    import dexstation.inputs.linkerta_reader as module
    calls=[]
    class Socket:
        def bind(self,address):calls.append(address)
        def setblocking(self,value):assert value is False
        def send(self,data):calls.append(data);return len(data)
        def recv(self,size):return FRAME.pack(0x65,8,struct.pack('>hhhh',-1700,0,0,0))
        def close(self):calls.append('closed')
    monkeypatch.setattr(module.socket,'socket',lambda *args:Socket())
    bus=SocketCanBus('fakecan');bus.send(0x04,b'')
    assert calls[1] == FRAME.pack(0x04,0,b'\0'*8)
    assert bus.recv() == frame(0x65,[-170,0,0,0])
    bus.close()


def test_ros_adapter_silent_then_measured_qos_names_and_low_rate_status(monkeypatch):
    reader, now, bus, state = rig(False)
    spec=importlib.util.spec_from_file_location('reader_node',Path(__file__).parents[1]/'scripts/linkerta_reader_node.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    published={};qoses={}
    class Message:
        def __init__(self):self.header=N(stamp=None)
    class Node:
        def declare_parameter(self,name,default):return N(value=default)
        def create_publisher(self,typ,topic,qos):
            published[topic]=[];qoses[topic]=qos
            return N(publish=published[topic].append)
        def create_timer(self,period,callback):self.tick=callback
        def get_clock(self):return N(now=lambda:N(to_msg=lambda:'stamp'))
        def get_logger(self):return N(error=lambda msg:None)
        def destroy_node(self):pass
    node=Node()
    def spin(_):
        for t in [0,.01,.49,.5]:now[0]=10+t;node.tick()
        assert published['/left_arm_joint_control']==[]
        assert len(published['/linkerta/status']) == 2
        state.update(up=True,operstate='up',reason=None)
        bus.frames=[frame(0x65,[0]*4),frame(0x66,[0]*3)]
        node.tick()
    monkeypatch.setattr(module,'LinkertaReader',lambda **kw:reader)
    monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    monkeypatch.setattr('dexstation.config.site.resolve_can_interface',lambda *a,**kw:'fakecan')
    monkeypatch.setitem(sys.modules,'rclpy',N(init=lambda:None,create_node=lambda *a,**kw:node,spin=spin,shutdown=lambda:None))
    monkeypatch.setitem(sys.modules,'rclpy.qos',N(QoSProfile=lambda **kw:kw,ReliabilityPolicy=N(BEST_EFFORT='best'),HistoryPolicy=N(KEEP_LAST='last')))
    monkeypatch.setitem(sys.modules,'sensor_msgs.msg',N(JointState=Message))
    monkeypatch.setitem(sys.modules,'std_msgs.msg',N(String=Message))
    module.main()
    left=published['/left_arm_joint_control']
    assert len(left)==1 and left[0].name==[f'joint{i}' for i in range(61,68)]
    assert qoses['/left_arm_joint_control']==dict(depth=1,reliability='best',history='last')
    assert json.loads(published['/linkerta/status'][-1].data)['sides']['left']['publishing']
