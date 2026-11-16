#!/usr/bin/env python3
"""ROS example for the project-owned LinkerTA measurement reader.

Starting main sends CAN measurement queries. Topic, QoS and numeric defaults
are project configuration; see linkerta_assets/PROVENANCE.md and ruling RW2-1.
"""
import json
import math
import time

from dexstation.inputs.linkerta_reader import LinkertaReader, status_key, READER_NODE


def main():
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from dexstation.config.site import resolve_can_interface

    rclpy.init()
    node = rclpy.create_node(READER_NODE, enable_rosout=False)

    def setting(key, fallback):
        return node.declare_parameter(key, fallback).value

    role = setting('arm_channel', 'linkerta')
    if role != 'linkerta':
        raise ValueError('arm_channel must be the linkerta role; set its interface override in station.can_roles')
    channel = resolve_can_interface('linkerta')
    bitrate = setting('arm_baudrate', 1000000)
    frequency = setting('publish_rate_hz', 100.)
    status_frequency = setting('status_rate_hz', 2.)
    age_limit = setting('max_age_s', .30)
    signs = setting('negation', [1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1])
    offsets = setting('offset_deg', [0.] * 7 + [-15.] + [0.] * 6)
    if not all(math.isfinite(value) and value > 0 for value in (frequency, status_frequency, bitrate)):
        raise ValueError('rates and bitrate must be finite and positive')
    reader = LinkertaReader(channel=channel, negation=signs, offset_deg=offsets, max_age_s=age_limit)
    # RW2-1: match the collection subscriber's latest-value configuration.
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    arm_publishers = {}
    for arm in ('left', 'right'):
        arm_publishers[arm] = node.create_publisher(JointState, f'/{arm}_arm_joint_control', qos)
    health_publisher = node.create_publisher(String, '/linkerta/status', 1)
    previous_key = None
    previous_status_time = -math.inf

    def publish_tick():
        nonlocal previous_key, previous_status_time
        measured, health = reader.step()
        for arm, degrees in measured.items():
            packet = JointState()
            start = 61 if arm == 'left' else 51
            packet.name = [f'joint{number}' for number in range(start, start + 7)]
            packet.position = degrees
            packet.header.stamp = node.get_clock().now().to_msg()
            arm_publishers[arm].publish(packet)
        instant = time.monotonic()
        key = status_key(health)
        transition = key != previous_key
        if transition or instant - previous_status_time >= 1 / status_frequency:
            packet = String()
            packet.data = json.dumps(dict(health, configured_bitrate=bitrate), allow_nan=False)
            health_publisher.publish(packet)
            if transition and health['io_error']:
                node.get_logger().error(health['io_error'])
            previous_key = key
            previous_status_time = instant

    node.create_timer(1 / frequency, publish_tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    import sys
    if '--inspect-publishers' in sys.argv:
        from dexstation.inputs.linkerta_reader import inspect_ros_publishers
        print(json.dumps(inspect_ros_publishers()))
    else:
        main()
