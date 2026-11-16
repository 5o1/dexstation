"""Lazy built-in source adapters; construction is authorized by the controller."""
from dexstation.inputs.replay import CollectionReplay, CollectionUdpSource, Ros2Source
from dexstation.config.site import site_value


def ros2(context):
    a = context.options
    return Ros2Source(context.mapper, topic_prefix=a.ros_topic_prefix, retarget=a.retarget,
        geometry_provider=getattr(context, 'geometry_provider', None))


def udp(context):
    a = context.options
    return CollectionUdpSource(a.udp_host, a.udp_port, stale_after_s=.25,
        allow_source=site_value('udp.allow_sources', []) or None, arm_mapper=context.mapper, continuous=True)


def replay(context):
    a = context.options
    source = CollectionReplay(a.replay, loop=a.loop_replay,
        stateful=False if context.replay_episode is not None else bool(context.profile))
    if context.replay_episode is not None and context.initial:
        if a.loop_replay:
            raise ValueError('Stateful replay cannot loop')
        from dexstation.recording.episode import read_input_events
        source.motion_journal = read_input_events(a.replay, meta=context.replay_meta)
        source._record_origin_ns = context.initial['input_clock_origin_ns']
        source._record_start_s = context.replay_meta['recording_start_sim_time_s']
        source.received_packets = context.replay_meta.get('receive_start_sequence', 0)
        source.start = 0.
        source.emit_all = True
    return source
