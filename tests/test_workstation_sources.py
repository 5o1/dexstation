import json
import socket
import pytest

from dexstation.examples.ar5_g20 import mapping as g20_mapping
from dexstation.control.commands import ArmCommand, HandCommand
from dexstation.control.protocol import ArmCommandPacket, HandCommandPacket, encode
from dexstation.inputs.udp import UdpSource


def _hand(stamp=1.0):
    return HandCommand("left", {name: 0.1 for name in g20_mapping.G20_JOINT_NAMES}, stamp)


def test_udp_drains_newest_holds_and_expires():
    source = UdpSource("127.0.0.1", 0, stale_after_s=0.2)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for stamp, value in [(1.0, 255), (2.0, 0)]:
            packet = HandCommandPacket("left", "motor", [], [value] * 20, [], [], stamp)
            sender.sendto(encode(packet), source.bound_address)
        sender.sendto(b"bad", source.bound_address)
        commands = source.poll(10.0)
        assert len(commands) == 1 and commands[0].stamp == 2.0
        assert source.decode_errors == 1
        assert source.poll(10.19) == commands
        assert source.poll(10.21) == []
    finally:
        sender.close()
        source.close()








@pytest.mark.parametrize("allowed,accepted", [(None, True), (["127.0.0.1"], True),
                                              (["198.51.100.34"], False),
                                              (["198.51.100.34", "127.0.0.1"], True)])
def test_udp_allowlist_counts_rejections_before_decoding(allowed, accepted):
    source = UdpSource("127.0.0.1", 0, stale_after_s=.2, allow_source=allowed)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        try:
            sender.sendto(encode(HandCommandPacket("left", "motor", [], [128]*20, [], [], 1)),
                          source.bound_address)
            sender.sendto(b"invalid JSON", source.bound_address)
            assert bool(source.poll(10)) == accepted
            assert source.received_packets == (2 if accepted else 0)
            assert source.rejected_packets == (0 if accepted else 2)
            assert source.decode_errors == (1 if accepted else 0)
            assert source.poll(10.21) == []
        finally:
            source.close()


def test_udp_invalid_allowlist_does_not_open_socket():
    with pytest.raises(ValueError):
        UdpSource("127.0.0.1", 0, stale_after_s=.2, allow_source=["not-an-ip"])
