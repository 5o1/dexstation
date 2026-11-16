"""Subscriber-only ROS input and timed dual-arm/dual-hand replay."""
import importlib
import copy
from contextlib import nullcontext
import json
import math
from dataclasses import asdict
import queue
import threading
import time
from pathlib import Path

import numpy as np

from dexstation.runtime.health import InputHealth
from dexstation.control.commands import ArmCommand, HandCommand, hand_command_from_packet, teleop_hand_packet
from dexstation.control.protocol import HandCommandPacket
from dexstation.model.constants import SIDES
from dexstation.recording.episode import read_episode


def ros_command_topics(prefix=""):
    return {side + "_" + kind: prefix.rstrip("/") + topic
            for side in SIDES
            for kind, topic in (("hand", f"/cb_{side}_hand_control_cmd"),
                                ("arm", f"/{side}_arm_joint_control"))}


def snapshot_input_events(source):
    """Copy at most the four latest raw command events; never drain the recorder."""
    with getattr(source, '_lock', nullcontext()):
        return copy.deepcopy(getattr(source, 'latest_events', {}))


class RosInputHealth(InputHealth):
    def __init__(self, source, clock):
        super().__init__(clock)
        self.source = source

    def snapshot(self):
        result = super().snapshot()
        for side, state in self.source.arm_status().items():
            result[side+'_arm'].update(input_available=state['available'], input_reason=state['reason'],
                                      linkerta=state['linkerta'], warning=not state['available'])
        return result


class Ros2Source:
    """Continuous ROS reception; lossless event FIFO and separate latest targets.

    A single executor thread serializes callbacks in receive order. Only the
    recorder drains events; control polling never consumes or discards them.
    """
    def __init__(self, mapper, *, node=None, joint_state_type=None, spin_once=None,
                 stale_after_s=.25, clock=time.monotonic, packet_sink=None,
                 topic_prefix="", qos_depth=128, reliability=None, retarget='manus_anatomical', status_type=None, geometry_provider=None):
        if qos_depth < 100:
            raise ValueError("ROS command QoS must buffer at least 100 messages per channel")
        self.health = RosInputHealth(self, clock)
        self.mapper, self.clock = mapper, clock
        self.packet_sink, self.stale_after_s = packet_sink, stale_after_s
        self.decode_errors = self.received_packets = 0
        self.latest, self.polled_sequences = {}, {}
        self.latest_events = {}
        self.latest_gloves = {}
        self.linkerta_status = None
        self.linkerta_received = None
        self.arm_errors = {}
        self.glove_received = {}
        self.glove_samples = {}
        self.glove_error = None
        self.glove_geometry = None
        self.geometry_provider = geometry_provider
        self.retarget = retarget
        self._lock, self._events = threading.Lock(), queue.Queue()
        self._rclpy = self._executor = self._thread = None
        self._owns_context, self._closing = False, False
        self._thread_error = None
        self.spin_once = spin_once or (lambda: None)  # injected unit-test node only
        if node is None:
            try:
                self._rclpy = importlib.import_module("rclpy")
                joint_state_type = importlib.import_module("sensor_msgs.msg").JointState
                glove_type = importlib.import_module('manus_ros2_msgs.msg').ManusGlove
                status_type = importlib.import_module('std_msgs.msg').String
            except ImportError as exc:
                raise RuntimeError("ROS2 input needs ROS Humble rclpy, sensor_msgs and manus_ros2_msgs; source the independently installed ROS and Manus message packages") from exc
            from dexstation.inputs.linkerta_reader import preflight_linkerta
            preflight_linkerta()
            self._owns_context = not self._rclpy.ok()
            if self._owns_context:
                self._rclpy.init()
            node = self._rclpy.create_node("gamepad_collection_subscriber", enable_rosout=False,
                                           start_parameter_services=False)
        self.node = node
        # Real rclpy nodes always supply graph introspection and MessageInfo.
        # Minimal injected nodes keep the older offline decoder test interface.
        self.publisher_checks = hasattr(node, 'get_publishers_info_by_topic')
        self.publisher_topics = {s:ros_command_topics(topic_prefix)[s+'_arm'] for s in SIDES}
        self.publisher_checked = -math.inf
        self.publisher_rows = {}
        self.arm_publisher_errors = {}
        self.subscriptions = []
        # Station publishers: retarget hand commands are RELIABLE, LinkerTA arm
        # commands are BEST_EFFORT. A RELIABLE subscription never matches a
        # BEST_EFFORT publisher, so each channel follows its publisher.
        self.reliability = {"hand": "reliable", "arm": "best_effort", **(reliability or {})}
        for key, topic in ros_command_topics(topic_prefix).items():
            side, kind = key.split("_")
            qos = qos_depth
            if self._rclpy is not None:
                qos_module = importlib.import_module("rclpy.qos")
                qos = qos_module.QoSProfile(depth=qos_depth, reliability=getattr(
                    qos_module.ReliabilityPolicy, self.reliability[kind].upper()))
            def callback(s, k):
                if k == 'arm' and self.publisher_checks:
                    def arm(msg, info):
                        self._receive_arm(s, msg, info)
                    return arm
                return lambda msg: self._receive(s, k, msg)
            self.subscriptions.append(node.create_subscription(joint_state_type, topic, callback(side, kind), qos))
        if self._rclpy is not None:
            for topic in ('/manus_glove_0', '/manus_glove_1'):
                self.subscriptions.append(node.create_subscription(
                    glove_type, topic_prefix.rstrip('/')+topic, self._receive_glove, 1))
        if status_type is not None:
            self.subscriptions.append(node.create_subscription(
                status_type, topic_prefix.rstrip('/')+'/linkerta/status', self._receive_linkerta_status, 1))
        if self._rclpy is not None:
            executor_type = importlib.import_module("rclpy.executors").SingleThreadedExecutor
            self._executor = executor_type(context=node.context)
            self._executor.add_node(node)
            self._thread = threading.Thread(target=self._spin, name="collection-ros-receiver")
            self._thread.start()

    def _refresh_publishers(self, force=False):
        if not self.publisher_checks or (not force and self.clock()-self.publisher_checked < 1.):
            return
        from dexstation.inputs.linkerta_reader import publisher_evidence
        rows = publisher_evidence(self.node, self.publisher_topics)
        with self._lock:
            self.publisher_rows, self.publisher_checked = rows, self.clock()
            for side, row in rows.items():
                reason = (f'foreign publisher on {self.publisher_topics[side]}: '+', '.join(row['foreign'])
                          if row['foreign'] else None if row['allowed'] else
                          f'no confirmed LinkerTA publisher on {self.publisher_topics[side]}')
                if reason:
                    self.arm_publisher_errors[side] = reason
                    self.latest.pop(('arm', side), None)
                    self.latest_events.pop(side+'_arm', None)
                else:
                    self.arm_publisher_errors.pop(side, None)

    def _receive_arm(self, side, msg, info):
        gid = bytes(info.publisher_gid)
        self._refresh_publishers()
        if gid not in self.publisher_rows.get(side, {}).get('allowed', set()):
            self._refresh_publishers(force=True)
        with self._lock:
            row = self.publisher_rows.get(side, {})
            reason = self.arm_publisher_errors.get(side)
            if gid not in row.get('allowed', set()):
                reason = reason or f'unconfirmed publisher on {self.publisher_topics[side]}'
            if reason:
                self.arm_publisher_errors[side] = reason
                self.latest.pop(('arm', side), None)
                self.latest_events.pop(side+'_arm', None)
                return
            self._receive_locked(side, 'arm', msg)

    def _receive_linkerta_status(self, msg):
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict) or not isinstance(value.get('sides'), dict):
                return
            if any(not isinstance(value['sides'].get(s), dict) for s in SIDES):
                return
        except (ValueError, TypeError, AttributeError):
            return
        with self._lock:
            self.linkerta_status, self.linkerta_received = value, self.clock()

    def arm_status(self):
        self._refresh_publishers()
        now = self.clock()
        with self._lock:
            result = {}
            for side in SIDES:
                latest = self.latest.get(('arm', side))
                available = latest is not None and now-latest[0] <= self.stale_after_s
                status = self.linkerta_status
                status_fresh = self.linkerta_received is not None and now-self.linkerta_received <= 1.5
                detail = dict(status['sides'][side], interface=status.get('interface'),
                              status_age_s=now-self.linkerta_received) if status else None
                reason = None if available else (
                    self.arm_publisher_errors.get(side) or
                    (status['sides'][side].get('reason') if status and status_fresh else None)
                    or self.arm_errors.get(side) or ('no fresh arm message' if status_fresh else
                        'LinkerTA status stale' if status else 'no CAN data'))
                result[side] = dict(available=available, reason=reason, linkerta=detail)
            return result

    def _receive_glove(self, msg):
        from dexstation.inputs.glove_overlay import pack_nodes
        side = (msg.side or '').strip().lower()
        side = next((s for s in SIDES if side.startswith(s)), None)
        if side:
            nodes = pack_nodes(msg)
            with self._lock:
                self.latest_gloves[side] = nodes
                self.glove_received[side] = self.clock()

    def glove_status(self):
        now = self.clock()
        with self._lock:
            received = dict(self.glove_received)
        return {side: dict(state='ready' if side in received and now-received[side] <= .25 else 'starting',
                           age_s=now-received[side] if side in received else None)
                for side in SIDES}

    def freeze_gloves(self):
        from dexstation.inputs.glove_overlay import anatomical_geometry, unpack_nodes
        with self._lock:
            gloves = dict(self.latest_gloves)
        self.glove_samples = {}
        if not gloves:
            return
        try:
            if self.glove_geometry is None:
                provider = getattr(self, 'geometry_provider', None) or anatomical_geometry
                self.glove_geometry = provider(retarget=getattr(self, 'retarget', 'manus_anatomical'))
            errors = []
            for side, nodes in gloves.items():
                try:
                    value = self.glove_geometry.prepare(unpack_nodes(nodes), side)
                    if value is not None:
                        self.glove_samples[side] = value
                except ValueError as exc:
                    errors.append(f'{side}: {exc}')
            self.glove_error = '; '.join(errors) or None
        except (OSError, ValueError, KeyError, ImportError) as exc:
            self.glove_error = str(exc)

    def _spin(self):
        try:
            self._executor.spin()
        except Exception as exc:
            if not self._closing:
                self._thread_error = exc

    def _receive(self, side, kind, msg):
        with self._lock:
            self._receive_locked(side, kind, msg)

    def _receive_locked(self, side, kind, msg):
        from dexstation.control.protocol import ArmCommandPacket, encode, decode
        received_mono, received_wall = time.monotonic_ns(), time.time_ns()
        received_clock = self.clock()
        header = getattr(msg, "header", None)
        sec = int(header.stamp.sec) if header else received_wall // 10**9
        nanosec = int(header.stamp.nanosec) if header else received_wall % 10**9
        stamp = sec + nanosec * 1e-9
        raw = dict(name=list(msg.name), position=list(msg.position),
                   velocity=list(getattr(msg, "velocity", [])), effort=list(getattr(msg, "effort", [])))
        event = dict(sequence=self.received_packets, channel=side + "_" + kind,
                     received_monotonic_ns=received_mono, received_wall_time_ns=received_wall,
                     header=dict(stamp=dict(sec=sec, nanosec=nanosec),
                                 frame_id=getattr(header, "frame_id", "")),
                     raw=raw, converted_command=None, error=None)
        self.received_packets += 1
        packet = command = None
        try:
            if kind == "hand":
                if len(raw["position"]) != 20:
                    raise ValueError("expected 20 hand motors")
                # Convert a separate copy. Preserve original binary64 values in raw.
                packet = teleop_hand_packet(decode(encode(HandCommandPacket(side, "motor", raw["name"],
                                      raw["position"], raw["velocity"], raw["effort"], stamp))))
                command = hand_command_from_packet(packet)
            else:
                if len(raw['position']) != 7 or not all(math.isfinite(v) for v in raw['position']):
                    raise ValueError('expected 7 finite arm positions')
                command = self.mapper.command(side, raw["position"], stamp)
                packet = ArmCommandPacket(side, raw["position"], stamp, "master-deg")
            event["converted_command"] = asdict(command)
        except (KeyError, TypeError, ValueError) as exc:
            self.decode_errors += 1
            event["error"] = str(exc)
        if kind == 'arm':
            if event['error']:
                self.arm_errors[side] = event['error']
            else:
                self.arm_errors.pop(side, None)
        self.health.receive(event["channel"], error=event["error"] is not None)
        self._events.put_nowait(event)
        self.latest_events[event['channel']] = event
        if command is not None:
            self.latest[(kind, side)] = (received_clock, command, event["sequence"])
            if self.packet_sink:
                self.packet_sink(packet)

    def boundary(self):
        with self._lock:
            return self.drain_events(), self.received_packets

    def drain_events(self):
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def poll(self, now):
        self._refresh_publishers()
        self.freeze_gloves()
        if self._thread_error:
            raise RuntimeError("ROS receiver failed") from self._thread_error
        if self._rclpy is None:
            self.spin_once()
        with self._lock:
            fresh = [(k, v) for k, v in self.latest.items() if now - v[0] <= self.stale_after_s]
            self.polled_sequences = {side + "_" + kind: v[2] for (kind, side), v in fresh}
            return [v[1] for k, v in fresh]

    def close(self):
        self._closing = True
        if self._executor:
            self._executor.shutdown(timeout_sec=5.)
            self._thread.join(timeout=5.)
            if self._thread.is_alive():
                raise RuntimeError("ROS receiver did not stop")
        self.node.destroy_node()
        if self._owns_context:
            self._rclpy.shutdown()


class CollectionReplay:
    """Replay final slave targets directly. Optional loops preserve original rate."""
    def __init__(self, path, *, loop=False, stateful=None):
        path = Path(path)
        self.loop = loop
        self.decode_errors = 0
        self.cycles = 0
        self.start = None
        self.last_index = None
        self.events = []
        self._input_events = []
        self.latest_events = {}
        self.received_packets = 0
        self.health = InputHealth()
        self.uniform_grid = path.is_dir()
        self.motion_journal = None
        self._journal_index = 0
        self.emit_all = bool(stateful)
        self._emitted_index = -1
        if path.is_dir():
            meta, rows = read_episode(path)
            enabled=bool(meta.get('motion_initial_state')) if stateful is None else stateful
            if enabled:
                if not meta.get('motion_initial_state'):
                    raise ValueError('This episode has no motion state snapshot; use legacy target replay')
                if loop:raise ValueError('Stateful replay cannot loop; use legacy for target-only looping')
                from dexstation.recording.episode import read_input_events
                self.motion_journal=read_input_events(path,meta=meta)
                self._record_origin_ns=meta['motion_initial_state']['input_clock_origin_ns']
                self._record_start_s=meta['recording_start_sim_time_s']
                self.received_packets=meta.get('receive_start_sequence',0)
                self.start=0.
            fps = float(meta["fps"])
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("episode control fps must be positive")
            # Commands belong to the control grid; observations are post-step
            # samples on the engine's quantized clock, not action event times.
            self.times = np.arange(len(rows), dtype=float) / fps
            for row in rows:
                commands = []
                for side in SIDES:
                    h = row["commands"][side + "_hand"]
                    a = row["commands"][side + "_arm"]
                    if h is not None:
                        commands.append(HandCommand(side, dict(h["joint_positions"]), h["stamp"],
                                                    None if h["motor_positions"] is None else tuple(h["motor_positions"])))
                    if a is not None:
                        commands.append(ArmCommand(side, tuple(a["positions"]), a["stamp"],
                                                   None if a["raw_master_deg"] is None else tuple(a["raw_master_deg"]),
                                                   a.get("clamp_flags")))
                self.events.append(commands)
        else:
            with np.load(path, allow_pickle=False) as z:
                action = z["action"]
                self.times = np.asarray(z["timestamp"], float)
                names, units = z["names"].tolist(), z["units"].tolist()
                if len(set(z["episode_index"].tolist())) != 1:
                    raise ValueError("replay NPZ must contain exactly one source episode")
                for row, stamp in zip(action, self.times):
                    commands = []
                    for side in SIDES:
                        ai = [names.index(f"{side}_arm_joint_{j}") for j in range(1, 8)]
                        hi = [names.index(f"{side}_hand_motor_{j}") for j in range(20)]
                        if any(units[i] != "rad" for i in ai) or any(units[i] != "motor_0_255" for i in hi):
                            raise ValueError("replay units must be slave rad / motor_0_255")
                        if not np.isfinite(row).all() or np.any(row[hi] < 0) or np.any(row[hi] > 255):
                            raise ValueError("invalid replay action")
                        commands.append(ArmCommand(side, tuple(row[ai]), float(stamp)))
                        commands.append(hand_command_from_packet(HandCommandPacket(
                            side, "motor", [], row[hi].tolist(), [], [], float(stamp))))
                    self.events.append(commands)
        if len(self.times) < 2 or not np.isfinite(self.times).all() or not np.all(np.diff(self.times) > 0):
            raise ValueError("replay timestamps must increase and contain at least two rows")
        self.times -= self.times[0]
        self.cycle_s = float(self.times[-1] + np.median(np.diff(self.times)))
        if self.emit_all:self.start=0.

    def poll(self, now):
        if self.motion_journal is not None:
            return self._poll_motion_journal(float(now))
        if self.start is None:
            self.start = float(now)
        elapsed = now - self.start
        self.cycles = int(elapsed // self.cycle_s) if self.loop else 0
        phase = elapsed % self.cycle_s if self.loop else elapsed
        query = phase + (1e-12 if self.uniform_grid else 0.)
        index = min(len(self.events) - 1, max(0, np.searchsorted(self.times, query, side="right") - 1))
        if self.emit_all:
            flat=self.cycles*len(self.events)+index
            commands=[]
            for number in range(self._emitted_index+1,flat+1):
                cycle,i=divmod(number,len(self.events))
                for command in self.events[i]:
                    event=command_event(command,self.received_packets)
                    event['scheduled_sim_time_s']=cycle*self.cycle_s+float(self.times[i])
                    self._input_events.append(event);self.latest_events[event['channel']]=event
                    self.received_packets+=1;commands.append(command)
                    self.health.receive(event['channel'])
            self._emitted_index=flat
            return commands
        marker = (self.cycles, index)
        if marker == self.last_index:
            return []
        self.last_index = marker
        commands = self.events[index]
        for command in commands:
            event = command_event(command, self.received_packets)
            self._input_events.append(event)
            self.latest_events[event['channel']] = event
            self.received_packets += 1
            self.health.receive(command.side + ("_hand" if isinstance(command, HandCommand) else "_arm"))
        return commands

    def _poll_motion_journal(self,elapsed):
        from dexstation.control.commands import command_from_event
        commands=[]
        while self._journal_index<len(self.motion_journal):
            original=self.motion_journal[self._journal_index]
            when=original.get('scheduled_sim_time_s')
            if when is None:when=(original['received_monotonic_ns']-self._record_origin_ns)/1e9
            if when-self._record_start_s>elapsed+1e-10:break
            self._journal_index+=1
            event=dict(original,sequence=self.received_packets,received_monotonic_ns=time.monotonic_ns(),
                       scheduled_sim_time_s=when)
            self._input_events.append(event);self.received_packets+=1
            self.latest_events[event['channel']]=event
            command=command_from_event(event)
            if command is not None and not event.get('realtime_discarded'):
                commands.append(command);self.health.receive(event['channel'])
        return commands

    def boundary(self):
        return self.drain_events(), self.received_packets

    def drain_events(self):
        events, self._input_events = self._input_events, []
        return events

    def close(self):
        pass


def command_event(command, sequence):
    wall = time.time_ns()
    channel = command.side + ('_hand' if isinstance(command, HandCommand) else '_arm')
    return dict(sequence=sequence, channel=channel, received_monotonic_ns=time.monotonic_ns(),
                received_wall_time_ns=wall, header=dict(stamp=dict(sec=wall//10**9, nanosec=wall%10**9), frame_id=''),
                raw=asdict(command), converted_command=asdict(command), error=None)


class CollectionUdpSource:
    """Existing UDP decoder/control behavior plus receive journal and health."""
    def __init__(self, *args, continuous=False, **kwargs):
        from dexstation.inputs.udp import UdpSource
        self.source = UdpSource(*args, **kwargs)
        self.source.event_sink = self._receive
        self.health = InputHealth()
        self._events = []
        self.latest_events = {}
        self.received_packets = 0
        self._lock=threading.RLock()
        self._stop=threading.Event()
        self._thread=None;self._thread_error=None;self._latest=[]
        if continuous:
            def receive():
                try:
                    while not self._stop.is_set():
                        with self._lock:self._latest=self.source.poll(time.monotonic())
                        self._stop.wait(.001)
                except Exception as exc:self._thread_error=exc
            self._thread=threading.Thread(target=receive,name='collection-udp-receiver')
            self._thread.start()

    @property
    def decode_errors(self):
        return self.source.decode_errors

    def _receive(self, data, command, error):
        if command is not None:
            event = command_event(command, self.received_packets)
        else:
            wall = time.time_ns()
            event = dict(sequence=self.received_packets, channel='unknown', received_monotonic_ns=time.monotonic_ns(),
                         received_wall_time_ns=wall, header=dict(stamp=dict(sec=wall//10**9, nanosec=wall%10**9)),
                         converted_command=None)
        if command is None:
            try:
                raw = json.loads(data)
                kind = 'arm' if raw.get('type')=='arm_command' else 'hand' if raw.get('type')=='command' else None
                side = raw.get('side') if kind=='arm' else raw.get('hand')
                if kind and side in SIDES:event['channel']=side+'_'+kind
            except (ValueError, AttributeError):pass
        event.update(raw=dict(datagram_hex=data.hex()), error=error)
        self.received_packets += 1
        self._events.append(event)
        if event['channel'] in ('left_arm','right_arm','left_hand','right_hand'):
            self.latest_events[event['channel']] = event
        self.health.receive(event['channel'], error=error is not None)

    def poll(self, now):
        if self._thread_error:raise RuntimeError('UDP receiver failed') from self._thread_error
        with self._lock:
            return list(self._latest) if self._thread is not None else self.source.poll(now)

    def drain_events(self):
        with self._lock:
            events, self._events = self._events, []
            return events

    def boundary(self):
        with self._lock:return self.drain_events(), self.received_packets

    def close(self):
        self._stop.set()
        if self._thread is not None:self._thread.join(timeout=5.)
        self.source.close()
