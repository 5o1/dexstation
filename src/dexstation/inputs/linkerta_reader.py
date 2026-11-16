"""Project-owned LinkerTA input reader based on the public CAN contract.

Protocol reference: linker-bot/linkerbot-teleop-arm at 4354083b27fd69f9d042d78edb6924e67f7f03fc
(Apache-2.0). See linkerta_assets/PROVENANCE.md for source lines and our
validity specification. Construction does not open a device or configure CAN.
"""
import json
import math
from pathlib import Path
import socket
import struct
import subprocess
import time
from typing import Protocol

from dexstation.model.constants import SIDES


# Linux classic CAN envelope and the public LinkerArm.h measurement identifiers.
FRAME = struct.Struct('=IB3x8s')
QUERY_ID = 0x04
REPLIES = {0x65: (0, 4), 0x66: (4, 3), 0x67: (7, 4), 0x68: (11, 3)}
READER_NODE = 'dexhand_linkerta_reader'


class Bus(Protocol):
    def send(self, can_id: int, data: bytes): ...
    def recv(self) -> tuple[int, bytes] | None: ...
    def close(self): ...


class SocketCanBus:
    """Classic SocketCAN transport; interface setup belongs to the operator."""

    def __init__(self, channel):
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self.sock.bind((channel,))
            self.sock.setblocking(False)
        except OSError:
            self.close()
            raise

    def send(self, can_id, data):
        envelope = FRAME.pack(can_id, len(data), data.ljust(8, bytes([0])))
        written = self.sock.send(envelope)
        if written != FRAME.size:
            raise OSError('short CAN write')

    def recv(self):
        try:
            envelope = self.sock.recv(FRAME.size)
        except BlockingIOError:
            return None
        if len(envelope) == FRAME.size:
            identifier, dlc, payload = FRAME.unpack(envelope)
            if dlc <= 8:
                return identifier, payload[:dlc]
        return None

    def close(self):
        self.sock.close()


def interface_state(channel, root=Path('/sys/class/net')):
    """Read administrative and operational state without changing the interface."""
    directory = root / channel
    try:
        operational = (directory / 'operstate').read_text().strip()
        bits = int((directory / 'flags').read_text().strip(), 0)
        usable = bool(bits & 1) and operational in ('up', 'unknown')
    except (OSError, ValueError) as error:
        return dict(name=channel, operstate=None, flags=None, up=False,
                    reason=f'{channel} unavailable: {error}')
    return dict(name=channel, operstate=operational, flags=bits, up=usable,
                reason=None if usable else f'{channel} down')


def _angle_degrees(data, offset):
    """Public signed BE /10, wrap and tenths; retain Python float operation order."""
    integer, = struct.unpack_from('!h', data, offset)
    degrees = integer / 10.0
    shifted = degrees + 180.0
    positive = shifted % 360.0
    return positive - 180.0


class LinkertaReader:
    def __init__(self, *, channel, negation, offset_deg, max_age_s=.30,
                 bus_factory=SocketCanBus, inspect=interface_state, clock=time.monotonic):
        if not (math.isfinite(max_age_s) and max_age_s > 0):
            raise ValueError('max_age_s must be finite and positive')
        if len(negation) != 14 or len(offset_deg) != 14:
            raise ValueError('negation and offset_deg need 14 finite values')
        if not all(map(math.isfinite, (*negation, *offset_deg))):
            raise ValueError('negation and offset_deg need 14 finite values')
        self.channel = channel
        self.negation = list(negation)
        self.offset_deg = list(offset_deg)
        self.max_age_s = max_age_s
        self.clock = clock
        self.bus_factory = bus_factory
        self.inspect = inspect
        self.bus = None
        self.positions = [math.nan for _ in range(14)]
        self.received = [None for _ in range(14)]
        self.rejected = [0 for _ in range(14)]
        self.rejection_reason = [None for _ in range(14)]
        self.interface = self.inspect(self.channel)
        self.io_error = None
        self.started = self.clock()

    def accept_frame(self, can_id, data):
        """Refresh only complete, accepted replies; fault trailer bytes are unused."""
        group = REPLIES.get(can_id)
        if group is None:
            return
        start, width = group
        length = len(data)
        if length < width * 2 or length > 8:
            return
        arrival = self.clock()
        for slot in range(width):
            index = start + slot
            angle = _angle_degrees(data, slot * 2)
            previous = self.positions[index]
            distance = abs(angle - previous)
            if math.isfinite(previous) and distance >= 180.0:
                self.rejected[index] += 1
                arm = SIDES[index // 7]
                joint_number = 1 + index % 7
                self.rejection_reason[index] = (
                    f'{arm} J{joint_number} reply rejected: jump {distance:.1f} deg')
            else:
                self.positions[index] = round(angle, 1)
                self.received[index] = arrival
                self.rejection_reason[index] = None

    def step(self):
        self.interface = self.inspect(self.channel)
        if self.interface['up']:
            try:
                if self.bus is None:
                    self.bus = self.bus_factory(self.channel)
                self.bus.send(QUERY_ID, bytes())
                self.io_error = None
                for _ in range(256):
                    incoming = self.bus.recv()
                    if incoming is None:
                        break
                    self.accept_frame(*incoming)
            except OSError as error:
                self.io_error = f'{self.channel} CAN I/O failed: {error}'
                self.close()
        else:
            self.close()
            # Historical timestamps remain diagnostic; DOWN invalidates positions.
            self.positions = [math.nan for _ in range(14)]
        return self.snapshot()

    def snapshot(self):
        sampled_at = self.clock()
        side_status, output = {}, {}
        for arm_index, arm in enumerate(SIDES):
            first = 7 * arm_index
            joints = list(range(first, first + 7))
            ages = [None if self.received[index] is None else
                    max(0.0, sampled_at - self.received[index]) for index in joints]
            failure = self.interface.get('reason') or self.io_error
            for joint_number, index in enumerate(joints, start=1):
                age = ages[joint_number - 1]
                invalid = age is None or age > self.max_age_s or not math.isfinite(self.positions[index])
                if failure is None and invalid:
                    origin = self.received[index]
                    if origin is None:
                        origin = self.started
                    failure = self.rejection_reason[index] or (
                        f'no reply from {arm} J{joint_number} for {sampled_at-origin:.2f} s')
            enabled = failure is None
            side_status[arm] = dict(
                publishing=enabled, reason=failure, joint_age_s=ages,
                rejected_counts=[self.rejected[index] for index in joints],
                rejection_reasons=[self.rejection_reason[index] for index in joints])
            if enabled:
                output[arm] = [self.positions[index] * self.negation[index] +
                               self.offset_deg[index] for index in joints]
        return output, dict(interface=dict(self.interface), io_error=self.io_error, sides=side_status)

    def close(self):
        current = self.bus
        if current is not None:
            current.close()
            self.bus = None


def status_key(status):
    """Stable transition key: elapsed silence and counters alone are not events."""
    sides = []
    for arm, details in status['sides'].items():
        reason = details['reason']
        stable_reason = reason.split(' for ')[0] if reason else None
        sides.append((arm, details['publishing'], stable_reason,
                      tuple(details['rejection_reasons'])))
    return tuple(status['interface'].items()), status['io_error'], tuple(sides)


def publisher_evidence(node, topics):
    """Project graph policy: separate own endpoint GIDs from foreign node names."""
    evidence = {}
    for arm, topic in topics.items():
        strangers, own_gids = set(), set()
        for endpoint in node.get_publishers_info_by_topic(topic):
            if endpoint.node_name == READER_NODE:
                own_gids.add(bytes(endpoint.endpoint_gid))
            else:
                strangers.add(endpoint.node_name)
        evidence[arm] = dict(foreign=sorted(strangers), allowed=own_gids)
    return evidence


def inspect_ros_publishers():
    """Discover graph evidence without constructing a CAN reader."""
    import rclpy
    rclpy.init()
    observer = rclpy.create_node('dexhand_linkerta_preflight', enable_rosout=False,
                                start_parameter_services=False)
    try:
        deadline = time.monotonic() + .6
        while time.monotonic() < deadline:
            rclpy.spin_once(observer, timeout_sec=.05)
        topics = {arm: f'/{arm}_arm_joint_control' for arm in SIDES}
        rows = publisher_evidence(observer, topics)
        names = set()
        for row in rows.values():
            names.update(row['foreign'])
        return sorted(names)
    finally:
        observer.destroy_node()
        rclpy.shutdown()


def preflight_linkerta():
    """Require the configured input role without opening its bus."""
    return CanSystem().require()


class CanPreflightError(RuntimeError):
    def __init__(self, can):
        self.can = can
        super().__init__(can['reason'])


class CanSystem:
    """One role only. System calls and sysfs root are injectable for offline tests."""
    def __init__(self, *, run=None, sysfs_root=Path('/sys/class/net'), resolve=None):
        from dexstation.config.site import resolve_can_interface
        self.run = run or subprocess.run
        self.root = Path(sysfs_root)
        self.resolve = resolve or resolve_can_interface

    def check(self, role='linkerta'):
        if role != 'linkerta':
            raise ValueError('only the linkerta CAN role is supported')
        result = dict(role=role, iface=None, state='missing', fix='configure', ready=False)
        try:
            iface = self.resolve(role, sysfs_root=self.root)
            result['iface'] = iface
            state = interface_state(iface, self.root)
            result.update(state=state['operstate'] or 'missing', flags=state['flags'], fix='bring_up')
            if (self.root/iface/'type').read_text().strip() != '280':
                raise ValueError('role does not resolve to a CAN interface')
            if not state['up'] or state['operstate'] != 'up':
                result['reason'] = state['reason'] or f'{iface} is not operationally up'
                return result
            reply = self.run(['ip', '-details', '-json', 'link', 'show', 'dev', iface],
                             capture_output=True, text=True, timeout=5, check=True)
            rows = json.loads(reply.stdout)
            bitrate = rows[0]['linkinfo']['info_data']['bittiming']['bitrate']
            result['bitrate'] = bitrate
            result['ready'] = bitrate == 1000000
            result['fix'] = None if result['ready'] else 'manual_bitrate'
            result['reason'] = None if result['ready'] else f'{iface} is up with bitrate {bitrate}; manual setup required'
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
            result.update(reason=f'CAN preflight failed: {exc}')
        return result

    def require(self, role='linkerta'):
        state = self.check(role)
        if not state['ready']:
            raise CanPreflightError(state)
        return state
