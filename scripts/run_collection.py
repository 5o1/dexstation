#!/usr/bin/env python3
"""Dual-hand/dual-arm simulation collection. Never constructs a hardware backend."""
import argparse
import json
import queue
import sys
import threading
from pathlib import Path

from dexstation.config.defaults import DEFAULT_TASK, DEFAULT_WORKBENCH
from dexstation.inputs.registry import source_names, source_capabilities, validate_source
from dexstation.backends.registry import backend_names, get_backend
from dexstation.config.site import site_value, data_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=backend_names(online=True), required=True)
    p.add_argument("--scene-variant", choices=("full", "free-space"), default="full",
                   help="free-space disables environment collision in code, leaving v4.4 unchanged")
    p.add_argument("--mujoco-workers", type=int, default=1,
                   help="native CPU worker threads for MuJoCo; changes no solver parameters")
    p.add_argument("--workbench", default=DEFAULT_WORKBENCH)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--seed", type=int)
    p.add_argument("--source", choices=source_names(), required=True)
    p.add_argument("--autopilot-tasks", type=Path, help="IK autopilot task set (src/dexstation/task_plugins/xbox_gamepad_v3_3/data/autopilot/*.json)")
    p.add_argument("--replay", type=Path)
    p.add_argument("--ros-topic-prefix", default="", help="optional ROS subscription topic prefix")
    p.add_argument("--loop-replay", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--resource-root", type=Path)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--rate-hz", type=int, default=30)
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--record", action="store_true", help="start recording immediately; otherwise keyboard/browser start")
    p.add_argument("--note", default="")
    p.add_argument("--diagnostics", action="store_true", help="bounded diagnostic capture for this session only")
    p.add_argument("--view", action="store_true")
    p.add_argument("--host", default=site_value("operator.bind_host", "127.0.0.1"))
    p.add_argument("--port", type=int, default=site_value("operator.port", 8780))
    p.add_argument("--udp-host", default=site_value("udp.bind_host", "127.0.0.1"))
    p.add_argument("--udp-port", type=int, default=site_value("udp.port", 50200))
    p.add_argument("--arm-startup", choices=("spec", "first_command", "none"), default="spec",
                   help="arm startup clamp: around scene home_q, around the first master command, or none (sim only)")
    p.add_argument('--motion-profile', default=site_value('collection.motion_profile','auto'),
                   help='auto: station for live input; recorded profile for episodes; legacy for older replays')
    a = p.parse_args()
    if a.output is None:
        import time
        from uuid import uuid4
        from dexstation.profiles.compose import load_task
        a.output=data_root()/"sessions"/load_task(a.task)['id']/("session-"+time.strftime("%Y%m%d-%H%M%S")+"-"+uuid4().hex[:6])
    if a.duration <= 0 or a.rate_hz <= 0:
        p.error("duration and rate must be positive")
    if a.mujoco_workers < 1 or (not get_backend(a.backend).supports_workers and a.mujoco_workers != 1):
        p.error("--mujoco-workers must be positive and applies only to MuJoCo")
    if source_capabilities(a.source).replay and a.replay is None:
        p.error("--replay is required")
    if source_capabilities(a.source).task_set_option and getattr(a, source_capabilities(a.source).task_set_option) is None:
        p.error("--" + source_capabilities(a.source).task_set_option.replace("_", "-") + " is required")
    try:
        validate_source(a)
    except ValueError as exc:
        p.error(str(exc))
    from dexstation.runtime.controller import CollectionController, CollectionOptions
    controls = queue.Queue()
    if sys.stdin.isatty():
        def keyboard():
            for line in sys.stdin:
                text = line.strip()
                controls.put({"action": text if text in {"start", "stop"} else "note", "text": text})
        threading.Thread(target=keyboard, daemon=True).start()
    try:
        result = CollectionController(CollectionOptions(**vars(a)), controls=controls).run()
        print(json.dumps(result))
    except (RuntimeError, ValueError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
