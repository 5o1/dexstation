#!/usr/bin/env python3
"""Offline oracle runner/freezer. Never opens a CAN socket or imports ROS.

Run with PYTHONPATH=src and --old /tmp/rw2/old. Tests use the frozen data
and these fake adapters; they do not need the old implementation.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SEED = 4354083
NEGATION = [1, -1, 1, -1, 1, -1, 1] + [1] * 7
OFFSETS = [0.] * 7 + [-15.] + [0.] * 6


def encode(value):
    """Lossless typed encoding, including float bits and container order."""
    if isinstance(value, float):
        return ['float64', struct.pack('>d', value).hex()]
    if isinstance(value, bytes):
        return ['bytes', value.hex()]
    if isinstance(value, tuple):
        return ['tuple', [encode(v) for v in value]]
    if isinstance(value, list):
        return ['list', [encode(v) for v in value]]
    if isinstance(value, dict):
        return ['dict', [[encode(k), encode(v)] for k, v in value.items()]]
    if isinstance(value, set):
        return ['set', sorted([encode(v) for v in value])]
    if value is None or isinstance(value, (str, bool, int)):
        return [type(value).__name__, value]
    raise TypeError(type(value).__name__)


def outcome(call):
    try:
        return ['return', encode(call())]
    except Exception as exc:
        return ['raise', type(exc).__name__, str(exc)]


class Clock:
    def __init__(self, now=0., stride=0.):
        self.now, self.stride, self.calls = now, stride, 0

    def __call__(self):
        now = self.now
        self.now += self.stride
        self.calls += 1
        return now


class FakeBus:
    def __init__(self):
        self.frames, self.sent = [], []
        self.send_error = self.recv_error = self.close_error = None
        self.recv_calls = self.close_calls = 0

    def send(self, can_id, data):
        self.sent.append((can_id, data))
        if self.send_error:
            raise OSError(self.send_error)

    def recv(self):
        self.recv_calls += 1
        if self.recv_error:
            raise OSError(self.recv_error)
        return self.frames.pop(0) if self.frames else None

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise OSError(self.close_error)


def state(reader, clock, bus, opens):
    return dict(positions=reader.positions, received=reader.received,
                rejected=reader.rejected, rejection_reason=reader.rejection_reason,
                interface=reader.interface, io_error=reader.io_error,
                started=reader.started, clock=(clock.now, clock.calls),
                bus_open=reader.bus is not None, opens=opens[0], sent=bus.sent,
                recv_calls=bus.recv_calls, close_calls=bus.close_calls,
                queued=bus.frames)


def run_reader(module, case):
    clock = Clock(case.get('start', 0.), case.get('stride', 0.))
    bus, opens = FakeBus(), [0]
    interface = dict(name='fakecan', up=True, operstate='up', flags=1, reason=None)
    opening_error = [None]

    def factory(channel):
        assert channel == 'fakecan'
        opens[0] += 1
        if opening_error[0]:
            raise OSError(opening_error[0])
        return bus

    reader = module.LinkertaReader(
        channel='fakecan', negation=case.get('negation', NEGATION),
        offset_deg=case.get('offset_deg', OFFSETS),
        max_age_s=case.get('max_age_s', .3), bus_factory=factory,
        inspect=lambda _: dict(interface), clock=clock)
    records = []
    for event in case['events']:
        clock.now = event.get('time', clock.now)
        if 'interface' in event:
            interface.update(event['interface'])
        for key in ('send_error', 'recv_error', 'close_error'):
            setattr(bus, key, event.get(key))
        opening_error[0] = event.get('open_error')
        for can_id, payload in event.get('frames', []):
            bus.frames.append((can_id, bytes.fromhex(payload)))
        if 'positions' in event:
            reader.positions = [math.nan if v is None else v for v in event['positions']]
        action = event.get('action', 'step')
        if action == 'accept':
            can_id, payload = event['frame']
            result = outcome(lambda: reader.accept_frame(can_id, bytes.fromhex(payload)))
        else:
            result = outcome(getattr(reader, action))
        # Snapshot is intentionally separate: clock calls must also be equivalent.
        snapshot = reader.snapshot()
        records.append([result, encode(snapshot), encode(module.status_key(snapshot[1])),
                        encode(state(reader, clock, bus, opens))])
    return records


def run_socket(module, case):
    calls = []
    incoming = list(case.get('incoming', []))

    class Socket:
        def bind(self, address):
            calls.append(('bind', address))
            if case.get('bind_error'):
                raise OSError(case['bind_error'])

        def setblocking(self, value):
            calls.append(('blocking', value))
            if case.get('blocking_error'):
                raise OSError(case['blocking_error'])

        def send(self, data):
            calls.append(('send', data))
            if case.get('send_error'):
                raise OSError(case['send_error'])
            return case.get('write_size', len(data))

        def recv(self, size):
            calls.append(('recv', size))
            item = incoming.pop(0)
            if item == 'blocked':
                raise BlockingIOError()
            if item == 'error':
                raise OSError('read failed')
            return bytes.fromhex(item)

        def close(self):
            calls.append(('close',))

    def factory(*args):
        calls.append(('socket', args))
        if case.get('open_error'):
            raise OSError(case['open_error'])
        return Socket()

    def exercise():
        bus = module.SocketCanBus('fakecan')
        results = []
        for can_id, payload in case.get('send', []):
            results.append(outcome(lambda: bus.send(can_id, bytes.fromhex(payload))))
        for _ in range(len(incoming)):
            results.append(outcome(bus.recv))
        results.append(outcome(bus.close))
        return results

    with patch.object(module.socket, 'socket', factory):
        result = outcome(exercise)
    return [result, encode(calls)]


def run_publishers(module, case):
    graph = {side: [SimpleNamespace(node_name=name, endpoint_gid=bytes.fromhex(gid))
                    for name, gid in rows] for side, rows in case['graph'].items()}
    node = SimpleNamespace(get_publishers_info_by_topic=lambda topic: graph[topic])
    return outcome(lambda: module.publisher_evidence(node, {'left': 'left', 'right': 'right'}))


def run_validation(module, case):
    args = dict(channel='fakecan', negation=NEGATION, offset_deg=OFFSETS,
                inspect=lambda _: {}, clock=lambda: 0.)
    args.update(case['args'])
    return outcome(lambda: module.LinkertaReader(**args) and None)


def run_discovery(module, case):
    calls = []
    clock = Clock(stride=.1)
    graph = case['graph']
    node = SimpleNamespace(
        get_publishers_info_by_topic=lambda topic: [
            SimpleNamespace(node_name=name, endpoint_gid=bytes.fromhex(gid))
            for name, gid in graph['left' if 'left' in topic else 'right']],
        destroy_node=lambda: calls.append('destroy'))

    def create(name, **kwargs):
        calls.append(('create', name, kwargs))
        return node

    def spin(node, **kwargs):
        calls.append(('spin', kwargs))
        if case.get('spin_error'):
            raise RuntimeError('fake graph failure')

    fake_ros = SimpleNamespace(init=lambda: calls.append('init'), create_node=create,
                               spin_once=spin, shutdown=lambda: calls.append('shutdown'))
    with patch.dict('sys.modules', rclpy=fake_ros), patch.object(module.time, 'monotonic', clock):
        result = outcome(module.inspect_ros_publishers)
    return [result, encode(calls), encode((clock.now, clock.calls))]


def run_adapter(module, case):
    """Execute only injected ROS stand-ins, never the real ROS implementation."""
    path = (Path(module.__file__).parent / 'linkerta_reader_node.py'
            if module.__name__.endswith('_linkerta_oracle') else
            ROOT / 'scripts/linkerta_reader_node.py')
    spec = importlib.util.spec_from_file_location('fake_linkerta_node', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    calls, now, callbacks = [], [0.], []

    class Message:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None)

    class Node:
        def declare_parameter(self, name, default):
            calls.append(('parameter', name, default))
            return SimpleNamespace(value=case.get('parameters', {}).get(name, default))

        def create_publisher(self, typ, topic, qos):
            calls.append(('publisher', topic, qos))

            def publish(message):
                values = dict(vars(message), header=dict(vars(message.header)))
                calls.append(('publish', topic, values))
            return SimpleNamespace(publish=publish)

        def create_timer(self, period, callback):
            calls.append(('timer', period))
            callbacks.append(callback)

        def get_clock(self):
            return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: 'fake stamp'))

        def get_logger(self):
            return SimpleNamespace(error=lambda message: calls.append(('error', message)))

        def destroy_node(self):
            calls.append('destroy')

    def step():
        ready = now[0] >= 1.
        io_error = 'fakecan CAN I/O failed: test' if now[0] == 1.01 else None
        rows = {arm: dict(publishing=ready and io_error is None,
                         reason=io_error or (None if ready else f'no reply from {arm} J1 for {now[0]:.2f} s'),
                         joint_age_s=[0. if ready else None] * 7, rejected_counts=[0] * 7,
                         rejection_reasons=[None] * 7) for arm in ('left', 'right')}
        return ({arm: [0.] * 7 for arm in rows if rows[arm]['publishing']},
                dict(interface=dict(up=True), io_error=io_error, sides=rows))

    def reader(**kwargs):
        calls.append(('reader', kwargs))
        return SimpleNamespace(step=step, close=lambda: calls.append('close'))

    def create(name, **kwargs):
        calls.append(('node', name, kwargs))
        return Node()

    def spin(node):
        for instant in (0., .01, .49, .5, 1., 1.01, 1.51):
            now[0] = instant
            callbacks[0]()
        raise KeyboardInterrupt()

    ros = SimpleNamespace(init=lambda: calls.append('init'), create_node=create,
                          spin=spin, shutdown=lambda: calls.append('shutdown'))
    qos = SimpleNamespace(QoSProfile=lambda **kwargs: kwargs,
                          ReliabilityPolicy=SimpleNamespace(BEST_EFFORT='best'),
                          HistoryPolicy=SimpleNamespace(KEEP_LAST='last'))
    import dexstation.config.site as site
    with patch.dict('sys.modules', {'rclpy': ros, 'rclpy.qos': qos,
                    'sensor_msgs.msg': SimpleNamespace(JointState=Message),
                    'std_msgs.msg': SimpleNamespace(String=Message)}), \
            patch.object(site, 'resolve_can_interface', lambda role: 'fakecan'), \
            patch.object(adapter, 'LinkertaReader', reader), \
            patch.object(adapter, 'status_key', module.status_key), \
            patch.object(adapter.time, 'monotonic', lambda: now[0]):
        result = outcome(adapter.main)
    return [result, encode(calls)]


def run_foreign_receiver(module, case):
    """Exercise unchanged receiver validity decisions with old/new graph evidence."""
    from dexstation.inputs import replay as collection_sources, linkerta_reader
    from dexstation.control.commands import ArmCommand
    callbacks, now = {}, [10.]
    own = SimpleNamespace(node_name=module.READER_NODE, endpoint_gid=b'own')
    foreign = SimpleNamespace(node_name='linkerta_node', endpoint_gid=b'foreign')
    graph = {'left': [own], 'right': [own]}

    class Node:
        def create_subscription(self, typ, topic, callback, qos):
            callbacks[topic] = callback

        def get_publishers_info_by_topic(self, topic):
            return graph['left' if 'left' in topic else 'right']

        def destroy_node(self):
            pass

    # Mapping is deliberately a deterministic stand-in, outside the reader scope.
    mapper = SimpleNamespace(command=lambda arm, raw, stamp:
                             ArmCommand(arm, tuple(raw), stamp, tuple(raw)))
    source = collection_sources.Ros2Source(mapper, node=Node(), joint_state_type=object,
                                          clock=lambda: now[0])
    trials = [(10., [own], b'own', [0.] * 7),
              (10.01, [own, foreign], b'foreign', []),
              (10.02, [own, foreign], b'own', [1.] * 7),
              (11.03, [own], b'own', [2.] * 7),
              (11.04, [own], b'unknown', []),
              (12.05, [own], b'own', [3.] * 6),
              (12.06, [own], b'own', [4.] * 7),
              (12.07, [], b'own', [5.] * 7)]
    records = []
    evidence = module.publisher_evidence
    with patch.object(linkerta_reader, 'publisher_evidence', evidence), \
            patch.object(collection_sources.time, 'monotonic_ns', lambda: int(now[0] * 1e9)), \
            patch.object(collection_sources.time, 'time_ns', lambda: 1000000000000 + int(now[0] * 1e9)):
        for instant, publishers, gid, values in trials:
            now[0], graph['left'] = instant, publishers
            message = SimpleNamespace(name=[], position=values)
            result = outcome(lambda: callbacks['/left_arm_joint_control'](
                message, SimpleNamespace(publisher_gid=gid)))
            records.append([result, encode([asdict(command) for command in source.poll(now[0])]),
                            encode(source.health.snapshot()),
                            encode(collection_sources.snapshot_input_events(source)),
                            encode(source.drain_events())])
        source.close()
    return records


RUNNERS = dict(reader=run_reader, socket=run_socket, publishers=run_publishers,
               validation=run_validation, discovery=run_discovery, adapter=run_adapter,
               foreign_receiver=run_foreign_receiver)


def execute(module, case):
    return RUNNERS[case['kind']](module, case)


def wire(can_id, values, trailer=b''):
    return [can_id, (b''.join(struct.pack('>h', v) for v in values) + trailer).hex()]


def corpus():
    cases = []
    add = lambda name, kind, **values: cases.append(dict(name=name, kind=kind, **values))
    replies = [(0x65, 4), (0x66, 3), (0x67, 4), (0x68, 3)]
    limits = [-32768, -32767, -3601, -3600, -1801, -1800, -1799,
              -1, 0, 1, 1799, 1800, 1801, 3599, 3600, 32766, 32767]
    for can_id, count in replies:
        add(f'boundaries-{can_id:x}', 'reader', events=[
            dict(action='accept', frame=wire(can_id, [value] * count),
                 positions=[None] * 14) for value in limits])
        add(f'lengths-flags-{can_id:x}', 'reader', events=[
            dict(action='accept', frame=[flag | can_id, ('80' * length)])
            for flag in (0, 0x80000000, 0x40000000, 0x20000000)
            for length in range(10)])
    complete = [wire(can_id, [0] * count) for can_id, count in replies]
    add('freshness-exact-and-neighbors', 'reader', events=[
        dict(time=0., frames=complete),
        *[dict(time=t) for t in (math.nextafter(.3, 0.), .3, math.nextafter(.3, 1.), 1., -.1)]])
    add('jumps-interleaving-recovery', 'reader', start=10., events=[
        dict(frames=[wire(0x65, [-1700, 0, 0, 0]), complete[1]]),
        dict(time=10.1, frames=[wire(0x65, [600, 1800, 1799, -1]), complete[1]]),
        dict(time=10.31, frames=[wire(0x65, [600, 0, 0, 0]), complete[1], complete[2]]),
        dict(time=10.32, frames=[wire(0x65, [-1600, 0, 0, 0]), complete[3]]),
        dict(time=10.4, interface=dict(up=False, operstate='down', reason='fakecan down')),
        dict(time=11., interface=dict(up=True, operstate='up', reason=None)),
        dict(time=11.1, frames=list(reversed(complete))), dict(action='close')])
    add('io-failures-and-reopen', 'reader', events=[
        dict(frames=complete), dict(send_error='send failed'), dict(open_error='open failed'),
        dict(recv_error='read failed'), dict(frames=complete),
        dict(interface=dict(up=False, reason='fakecan down'), close_error='close failed'),
        dict(interface=dict(up=True, reason=None)), dict(action='close'), dict(action='close')])
    add('bounded-flood', 'reader', events=[
        dict(frames=[[0x123, 'ab']] * 256 + complete), {},
        dict(frames=[[0x65, '00']] * 257 + complete), {}, {}])
    add('advancing-clock', 'reader', start=2., stride=.0001, events=[
        dict(frames=complete), dict(frames=list(reversed(complete))),
        dict(time=2.4), dict(frames=complete)])
    add('custom-affine-configuration', 'reader', negation=[-0., .5, -2., 1, 0, 3., -1] * 2,
        offset_deg=[1., -0., 2., -15., 0., .1, 100.] * 2,
        events=[dict(frames=[wire(can_id, [-1] * count) for can_id, count in replies])])
    rng = random.Random(SEED)
    for batch in range(8):
        events, now = [], 0.
        for tick in range(128):
            now += rng.choice([0., .001, .1, .3, .31])
            frames = []
            for _ in range(rng.randrange(1, 7)):
                can_id, count = rng.choice(replies)
                if rng.random() < .2:
                    can_id |= rng.choice([0x80000000, 0x40000000, 0x20000000])
                payload = bytes(rng.randrange(256) for _ in range(rng.randrange(10)))
                if rng.random() < .7:
                    payload = b''.join(struct.pack('>h', rng.randrange(-32768, 32768)) for _ in range(count))
                frames.append([can_id, payload.hex()])
            event = dict(time=now, frames=frames)
            if tick % 19 == 0:
                event['interface'] = dict(up=False, operstate='down', reason='fakecan down')
            elif tick % 19 == 1:
                event['interface'] = dict(up=True, operstate='up', reason=None)
            events.append(event)
        add(f'seeded-stream-{batch}', 'reader', events=events)
    for key, value in [('max_age_s', 0), ('max_age_s', -1), ('max_age_s', math.inf),
                       ('max_age_s', math.nan), ('negation', []), ('offset_deg', [0.] * 13),
                       ('negation', [math.nan] * 14), ('offset_deg', [math.inf] * 14)]:
        add(f'validation-{len(cases)}', 'validation', args={key: value})
    for error in ('open_error', 'bind_error', 'blocking_error', 'send_error'):
        add(f'socket-{error}', 'socket', **{error: error}, send=[[4, '']])
    for size in (0, 15, 16):
        add(f'socket-write-{size}', 'socket', write_size=size,
            send=[[4, ''], [0x65, '00017fff8000ffff'], [4, '00' * 9]])
    add('socket-receive-boundaries', 'socket', incoming=[
        'blocked', 'error', '', '00' * 15, '00' * 17,
        *[struct.pack('=IB3x8s', 0x65, length, b'\x80\0' * 4).hex() for length in range(11)],
        *[struct.pack('=IB3x8s', flag | 0x65, 8, b'\0' * 8).hex()
          for flag in (0x80000000, 0x40000000, 0x20000000)]])
    own = ['dexhand_linkerta_reader', '010203']
    foreign = ['linkerta_node', 'aabbcc']
    graphs = [dict(left=[], right=[]), dict(left=[own], right=[foreign]),
              dict(left=[foreign, own, foreign, ['other', 'ff']], right=[own, own]),
              dict(left=[['dexhand_linkerta_reader', ''], own], right=[own])]
    for i, graph in enumerate(graphs):
        add(f'publishers-{i}', 'publishers', graph=graph)
        add(f'discovery-{i}', 'discovery', graph=graph)
    add('discovery-cleanup-on-error', 'discovery', graph=graphs[0], spin_error=True)
    for i, parameters in enumerate([
            {}, dict(arm_channel='can2'), dict(publish_rate_hz=0.),
            dict(status_rate_hz=-1.), dict(arm_baudrate=math.inf),
            dict(status_rate_hz=math.nan), dict(publish_rate_hz=50., status_rate_hz=4.)]):
        add(f'adapter-{i}', 'adapter', parameters=parameters)
    add('foreign-receiver-validity', 'foreign_receiver')
    return cases


def load_old(directory):
    spec = importlib.util.spec_from_file_location(
        'dexstation._linkerta_oracle', Path(directory) / 'linkerta_reader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exhaustive(old, new):
    """Every signed int16 value at every joint slot, fresh and stateful."""
    digest = hashlib.sha256()
    count = 0
    for can_id, joints in ((0x65, 4), (0x66, 3), (0x67, 4), (0x68, 3)):
        readers = [m.LinkertaReader(channel='fakecan', negation=NEGATION,
                   offset_deg=OFFSETS, inspect=lambda _: {}, clock=lambda: 0.) for m in (old, new)]
        for fresh in (True, False):
            for raw in range(-32768, 32768):
                data = struct.pack('>h', raw) * joints
                for reader in readers:
                    if fresh:
                        reader.positions = [math.nan] * 14
                    reader.accept_frame(can_id, data)
                a, b = readers
                left = (struct.pack('>14d', *a.positions), a.received, a.rejected, a.rejection_reason)
                right = (struct.pack('>14d', *b.positions), b.received, b.rejected, b.rejection_reason)
                assert left == right, (can_id, fresh, raw)
                digest.update(left[0])
                count += 1
    return dict(frames=count, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--exhaustive', action='store_true')
    args = parser.parse_args()
    from dexstation.inputs import linkerta_reader as new
    old = load_old(args.old)
    cases, records = corpus(), []
    for case in cases:
        expected = execute(old, case)
        actual = execute(new, case)
        assert actual == expected, case['name']
        records.append(dict(case=case, expected=expected))
    payload = dict(baseline='4dbb5373c7ffcf79cb580db00179828deb91f077', seed=SEED,
                   oracle_sha256=hashlib.sha256((args.old/'linkerta_reader.py').read_bytes()).hexdigest(),
                   records=records)
    data = json.dumps(payload, separators=(',', ':'), allow_nan=True).encode()
    if args.freeze:
        args.freeze.parent.mkdir(parents=True, exist_ok=True)
        args.freeze.write_bytes(gzip.compress(data, mtime=0))
    print(json.dumps(dict(cases=len(cases), reader_events=sum(len(c.get('events', [])) for c in cases),
                         wire_frames=sum(len(e.get('frames', [])) + ('frame' in e)
                                         for c in cases for e in c.get('events', [])),
                         seed=SEED, fixture_sha256=hashlib.sha256(gzip.compress(data, mtime=0)).hexdigest())))
    if args.exhaustive:
        print(json.dumps(exhaustive(old, new)))


if __name__ == '__main__':
    main()
