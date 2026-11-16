"""Non-blocking UDP targets shared with the collection receiver."""

import json
import ipaddress
import socket

from dexstation.control.commands import ArmCommand, HandCommand, arm_command_from_packet, hand_command_from_packet, teleop_hand_packet
from dexstation.control.protocol import ArmCommandPacket, HandCommandPacket, decode


def _key(command):
    return ("hand" if isinstance(command, HandCommand) else "arm", command.side)


class UdpSource:
    """Hold the newest valid command per kind and side until it becomes stale."""

    def __init__(self, bind_host, port, *, stale_after_s, allow_source=None, arm_mapper=None):
        if stale_after_s < 0:
            raise ValueError("stale_after_s must be non-negative")
        self.stale_after_s = float(stale_after_s)
        self.arm_mapper = arm_mapper
        self.allow_source = None if allow_source is None else frozenset(
            str(ipaddress.IPv4Address(ip)) for ip in allow_source)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((bind_host, port))
        self.socket.setblocking(False)
        self.bound_address = self.socket.getsockname()
        self.port = self.bound_address[1]
        self.decode_errors = 0
        self.rejected_packets = 0
        self.received_packets = 0
        self._newest = {}
        self.event_sink = None

    def poll(self, now):
        while True:
            try:
                data, address = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            if self.allow_source is not None and address[0] not in self.allow_source:
                self.rejected_packets += 1
                continue
            self.received_packets += 1
            try:
                packet = decode(data)
                if isinstance(packet, HandCommandPacket):
                    command = hand_command_from_packet(teleop_hand_packet(packet))
                elif isinstance(packet, ArmCommandPacket):
                    command = arm_command_from_packet(packet, self.arm_mapper)
                else:
                    continue
            except (TypeError, ValueError) as exc:
                self.decode_errors += 1
                if self.event_sink:
                    self.event_sink(data, None, str(exc))
                continue
            if self.event_sink:
                self.event_sink(data, command, None)
            self._newest[_key(command)] = (float(now), command)
        stale = [key for key, (received, _command) in self._newest.items()
                 if now - received > self.stale_after_s]
        for key in stale:
            del self._newest[key]
        return [item[1] for item in self._newest.values()]

    def close(self):
        self.socket.close()
