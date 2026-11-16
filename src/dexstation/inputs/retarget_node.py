"""ROS-only MANUS anatomical retarget node. Contains no device/SDK/CAN imports."""
import argparse
import os
import time
from dexstation.inputs.retarget_anatomy import SkeletonError, load_config
from dexstation.inputs.retarget_runtime import AnatomicalRetargeter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--method', choices=('manus_anatomical',), default='manus_anatomical')
    args, ros_args = parser.parse_known_args(argv)
    parser_domain = os.environ.get('ROS_DOMAIN_ID')
    if parser_domain is not None and not parser_domain.isdecimal():
        raise SystemExit('ROS_DOMAIN_ID must be a nonnegative integer')
    import rclpy
    from sensor_msgs.msg import JointState
    from manus_ros2_msgs.msg import ManusGlove
    config = load_config(args.config)
    rclpy.init(args=ros_args)
    node = rclpy.create_node('manus_anatomical_retarget', enable_rosout=False)
    engine_type = AnatomicalRetargeter
    engines = {s: engine_type(s, config) for s in ('left', 'right')}
    publishers = {s: node.create_publisher(JointState, f'/cb_{s}_hand_control_cmd', 10) for s in engines}
    latest, consumed, last_time, errors = {}, {}, {}, {}
    def receive(msg):
        side = msg.side.strip().lower()
        if side in engines:
            latest[side] = (time.monotonic(), msg)
    subscriptions = [node.create_subscription(ManusGlove, '/manus_glove_'+str(i), receive, 10) for i in range(2)]
    def tick():
        now = time.monotonic()
        for side in engines:
            value = latest.get(side)
            if value is None or consumed.get(side) == value[0] or now-value[0] > config['stale_after_s']:
                continue
            consumed[side] = value[0]
            dt = min(now-last_time.get(side, now-1/config['publish_hz']), config['stale_after_s'])
            try:
                fields, _, _ = engines[side].step(value[1].raw_nodes, dt)
            except (SkeletonError, ValueError, RuntimeError) as exc:
                # Reject this hand's frame; never re-publish its previous command.
                if str(exc) != errors.get(side):
                    node.get_logger().error(f'{side}: no output: {exc}')
                errors[side] = str(exc)
                continue
            last_time[side] = now
            errors.pop(side, None)
            message = JointState()
            message.header.stamp = node.get_clock().now().to_msg()
            for field, data in fields.items():
                setattr(message, field, data)
            publishers[side].publish(message)
    timer = node.create_timer(1/config['publish_hz'], tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
