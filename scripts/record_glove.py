#!/usr/bin/env python3
"""Read-only MANUS recorder. Starts no inputs and creates no ROS publishers."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dexstation.config.site import data_root
from dexstation.inputs.glove_recording import GloveRecording, PROMPTS, PROTOCOL, accepts_side, protocol_segment, guide_status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--output-root', type=Path, default=data_root() / 'glove_raw')
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--protocol', action='store_true', help='guide all labeled AK actions; overrides --seconds')
    p.add_argument('--hand', choices=('left', 'right'))
    p.add_argument('--countdown', type=float, default=5.)
    p.add_argument('--json-progress', action='store_true', help='emit machine-readable progress on stdout')
    args = p.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not math.isfinite(args.countdown) or args.countdown < 0:
        p.error('seconds must be positive and countdown nonnegative, both finite')
    if os.environ.get('ROS_DOMAIN_ID', '88') != '88':
        p.error('recording requires ROS_DOMAIN_ID=88')
    os.environ.update(ROS_DOMAIN_ID='88', ROS_LOCALHOST_ONLY='1')
    cancelled = False
    def interrupt(*_):
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    import rclpy
    from rclpy.serialization import deserialize_message
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.convert import message_to_ordereddict
    from manus_ros2_msgs.msg import ManusGlove
    rclpy.init()
    # rclpy may install its own SIGINT handler; restore our graceful finalizer.
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    node = rclpy.create_node('glove_raw_recorder', enable_rosout=False, start_parameter_services=False)
    label = f'{args.label}_{args.hand}' if args.hand else args.label
    recording = GloveRecording(args.output_root, label, metadata=dict(
        hand=args.hand or 'both', protocol=PROTOCOL if args.protocol else [(args.label, args.seconds)],
        ros_fields=ManusGlove.get_fields_and_field_types(), lossless_encoding='ROS2 CDR serialized bytes, base64'))
    duration = sum(d for _, d in PROTOCOL) if args.protocol else args.seconds
    side_name = dict(left='Left hand', right='Right hand').get(args.hand, 'Both hands')
    begin = None
    last_text = None
    def progress(phase, elapsed=0, remaining=None, error=None):
        nonlocal last_text
        state = guide_status(phase, elapsed, hand=args.hand, remaining=remaining)
        if not args.protocol:
            state.update(step=1 if phase in ('recording','completed') else 0, total_steps=1,
                         prompt=args.label, next_prompt=None, remaining_s=max(0,duration-elapsed) if phase=='recording' else 0)
        state.update(count=recording.meta['count'], output_dir=str(recording.path), error=error)
        if args.json_progress:
            print('GLOVE_PROGRESS '+json.dumps(state, ensure_ascii=False), flush=True)
        else:
            text = (f'{side_name} · starts in {math.ceil(remaining)} s' if phase == 'countdown' else
                    f'[{state["step"]}/{state["total_steps"]}] {side_name} {state["prompt"]} · {state["remaining_s"]:.0f} s · next: {state["next_prompt"] or "—"}' if phase == 'recording' else
                    f'{phase}: {recording.meta["count"]} messages: {recording.path}')
            key = (phase, state['step'], math.ceil(remaining) if phase == 'countdown' else None)
            if key != last_text:
                print(text, flush=True)
                last_text = key
    def callback(topic):
        def receive(cdr, info):
            mono_ns, wall_ns = time.monotonic_ns(), time.time_ns()
            segment = protocol_segment(mono_ns/1e9-begin) if args.protocol else args.label
            if segment is None or cancelled:
                return
            msg = deserialize_message(cdr, ManusGlove)
            if accepts_side(msg.side, args.hand):
                recording.append(topic, cdr, message_to_ordereddict(msg), segment,
                    wall_ns=wall_ns, monotonic_ns=mono_ns, source_ns=info.source_timestamp, received_ns=info.received_timestamp)
        return receive
    completed, error = False, None
    try:
        deadline = time.monotonic()+(args.countdown if args.protocol else 0)
        while not cancelled and time.monotonic() < deadline:
            progress('countdown', remaining=max(0, deadline-time.monotonic()))
            time.sleep(.1)
        if not cancelled:
            begin = time.monotonic()
            qos = QoSProfile(depth=512, reliability=ReliabilityPolicy.BEST_EFFORT)
            subscriptions = [node.create_subscription(ManusGlove, topic, callback(topic), qos, raw=True)
                             for topic in ('/manus_glove_0', '/manus_glove_1')]
            last_progress = -1.
            while not cancelled and time.monotonic()-begin < duration:
                elapsed = time.monotonic()-begin
                if elapsed-last_progress >= .1:
                    progress('recording', elapsed)
                    last_progress = elapsed
                rclpy.spin_once(node, timeout_sec=.05)
            completed = not cancelled and recording.meta['count'] > 0
            if not cancelled and not completed:
                error = f'{side_name} has no data'
    except Exception as exc:
        error = str(exc)
    finally:
        recording.close(complete=completed)
        node.destroy_node()
        rclpy.shutdown()
    progress('cancelled' if cancelled else 'failed' if error else 'completed', error=error)
    return 2 if error else 0


if __name__ == '__main__':
    sys.exit(main())
