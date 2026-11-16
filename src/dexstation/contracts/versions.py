"""Identifiers emitted by the existing file formats and public interfaces.

Values describe independent existing contracts; this table does not establish a
new compatibility policy or change the versions serialized by any producer.
"""

EPISODE_FORMAT = "dexhand-episode-v1"
WORKBENCH_FORMAT = "dexhand-workbench-rec-v1"
TASK_SCHEMA = "dexhand-task-v1"
WORKBENCH_SCHEMA = "dexhand-workbench-v1"
SCENE_FORMAT = "dexhand-scene-v1"
SCENE_BUNDLE_FORMAT = "dexhand-scene-bundle-v1"
AUTOPILOT_TASKS_SCHEMA = "dexhand-autopilot-tasks-v1"
AUTOPILOT_CONTROLS_SCHEMA = "dexhand-autopilot-controls-v1"
AUTOPILOT_FINGER_PADS_SCHEMA = "dexhand-autopilot-finger-pads-v1"
AUTOPILOT_PLAY_POSES_SCHEMA = "dexhand-autopilot-play-poses-v2"
MASTER_RANGE_SCHEMA = "dexhand-master-range-reference-v1"
REAL_ROBOT_CALIBRATION_SCHEMA = "dexhand-real-robot-calibration-v2"
TELEOP_CALIBRATION_SCHEMA = "dexhand-teleop-calibration-v2"
TAKEOVER_SCHEMA = "dexhand-takeover-v1"
WORKSPACE_LIMITS_SCHEMA = "dexhand-workspace-limits-v1"
GAMEPAD_RECORDING_FORMAT = "xbox-gamepad-rec-v1"
CUBE_RECORDING_FORMAT = "tabletop-cube-grasp-rec-v1"
GLOVE_RECORDING_FORMAT = "manus-cdr-jsonl-v1"
SDL3_LAYOUT_ID = "sdl3-gamepad-core-v1"
XINPUT_LAYOUT_ID = "xinput-normalized-v1"
SOLID_METRIC = "solid-v1"
ARM_MAPPING_VERSION = "rokae-target-map-snapshot-20261014-v1"
STATION_BASELINE_VERSION = "station-factory-baseline-v2"
STATION_APPEARANCE_V1 = "station-v1"
STATION_APPEARANCE_V2 = "station-v2"

GAMEPAD_CHANNEL_KIND = "gamepad_sdl3_core_v1"
LEROBOT_CODEBASE_VERSION = "v3.0"
WORKBENCH_MANIFEST_VERSION = 1
MOTION_PROFILE_VERSION = 2
MOTION_SNAPSHOT_VERSION = 2
CONTROLLER_SNAPSHOT_VERSION = 2
SAPIEN_SNAPSHOT_VERSION = 2
MUJOCO_SNAPSHOT_VERSION = 2
TAKEOVER_SNAPSHOT_VERSION = 1

API_VERSION = 1
API_PREFIX = "/api/v1"

# Include format, signal-layout, mapping, appearance and interface identifiers in
# one inventory, including versions carried only by declarative JSON assets.
VERSIONS = {
    "episode": EPISODE_FORMAT,
    "workbench_recording": WORKBENCH_FORMAT,
    "task": TASK_SCHEMA,
    "workbench": WORKBENCH_SCHEMA,
    "scene": SCENE_FORMAT,
    "scene_bundle": SCENE_BUNDLE_FORMAT,
    "autopilot_tasks": AUTOPILOT_TASKS_SCHEMA,
    "autopilot_controls": AUTOPILOT_CONTROLS_SCHEMA,
    "autopilot_finger_pads": AUTOPILOT_FINGER_PADS_SCHEMA,
    "autopilot_play_poses": AUTOPILOT_PLAY_POSES_SCHEMA,
    "master_range": MASTER_RANGE_SCHEMA,
    "real_robot_calibration": REAL_ROBOT_CALIBRATION_SCHEMA,
    "teleop_calibration": TELEOP_CALIBRATION_SCHEMA,
    "takeover": TAKEOVER_SCHEMA,
    "workspace_limits": WORKSPACE_LIMITS_SCHEMA,
    "gamepad_recording": GAMEPAD_RECORDING_FORMAT,
    "cube_recording": CUBE_RECORDING_FORMAT,
    "glove_recording": GLOVE_RECORDING_FORMAT,
    "sdl3_layout": SDL3_LAYOUT_ID,
    "xinput_layout": XINPUT_LAYOUT_ID,
    "solid_metric": SOLID_METRIC,
    "arm_mapping": ARM_MAPPING_VERSION,
    "station_baseline": STATION_BASELINE_VERSION,
    "station_appearance_v1": STATION_APPEARANCE_V1,
    "station_appearance_v2": STATION_APPEARANCE_V2,
    "gamepad_channel": GAMEPAD_CHANNEL_KIND,
    "lerobot_codebase": LEROBOT_CODEBASE_VERSION,
    "workbench_manifest": WORKBENCH_MANIFEST_VERSION,
    "motion_profile": MOTION_PROFILE_VERSION,
    "motion_snapshot": MOTION_SNAPSHOT_VERSION,
    "controller_snapshot": CONTROLLER_SNAPSHOT_VERSION,
    "sapien_snapshot": SAPIEN_SNAPSHOT_VERSION,
    "mujoco_snapshot": MUJOCO_SNAPSHOT_VERSION,
    "takeover_snapshot": TAKEOVER_SNAPSHOT_VERSION,
    "task_plugin_api": API_VERSION,
    "http_api": API_PREFIX,
}
