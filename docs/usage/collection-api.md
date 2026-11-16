# Collection API

Run requests on the collection station's loopback address. The CLI is `python scripts/collection_ctl.py`. Read status first, act only on explicit instructions, and verify the actual result afterward.

Remote writes default to disabled on every service start. Only an on-site person may enable them in the web UI. Assistants must not copy cookies, impersonate the web page, call the toggle endpoint, or add enable-remote-control during restart to bypass this switch. GET status/config/list/SSE requests are readable. Writes use JSON and `X-Gamepad-VLA-Client`; retain the existing web-cookie and Origin protocol.

```bash
python scripts/collection_ctl.py status --json
python scripts/collection_ctl.py record-start
python scripts/collection_ctl.py record-stop --wait --json
python scripts/collection_ctl.py session-stop
```

Stop recording with idempotent `record-stop --wait`. Wait for integrity results for the returned episode_id; do not reuse a previous episode's passed result. If a spoken "stop" is ambiguous, prefer stopping recording, which does not mean emergency stop or power-off. Voiding and exporting require an explicit episode ID. Preserve raw data and do not use force to bypass input health without authorization.

## Registered Routes

Source: `make_app` in [`collection_server.py`](../../src/dexstation/server/app.py), including its direct `app.router.add_*` calls, both final registration loops, and static resources. Request fields below come from the handlers and, for session start, `SessionManager.start`. Path and query fields are labeled; other fields are JSON body keys. "None" means no route-specific fields are consumed. All POST requests still require a JSON object, including `{}` when there are no fields.

Every `add_get` also registers HEAD through aiohttp's default behavior. Static resources register GET and HEAD too. No OPTIONS route is explicitly registered. WebSocket paths use GET for the handshake; HEAD is registered but does not establish a WebSocket.

| Method | Path | Purpose | Request fields actually read |
|---|---|---|---|
| GET, HEAD | `/` | Collection web page; establishes the browser session cookie | None |
| GET, HEAD | `/video` | Proxy the active simulation's main WebSocket video stream | None; binary stream from the worker |
| GET, HEAD | `/aux-video` | Proxy auxiliary WebSocket video and forward text selection messages | WebSocket text payload forwarded unchanged to the worker |
| GET, HEAD | `/auxiliary-panel.mjs` | Auxiliary-view panel module | None |
| GET, HEAD | `/teleop-table.mjs` | Teleoperation status-table module | None |
| GET, HEAD | `/camera-panel.mjs` | Camera-panel module | None |
| GET, HEAD | `/task-cells.mjs` | Task-cell mounting module | None |
| GET, HEAD | `/api/v1/tasks/{task_id}/cells` | Task cell/action configuration, resolving module/CSS URLs | Path: `task_id` |
| GET, HEAD | `/tasks/{task_id}/web/{file:.+}` | Serve an existing file only from the task's web directory | Path: `task_id`, `file` |
| POST | `/api/v1/task/action` | Dispatch an action between physics steps; forbidden actions return 409 during recording | `action` (string), `params` (object, default `{}`) |
| POST | `/api/v1/can/bring-up` | Bring up the configured LinkerTA CAN role through the installed helper | `role` (must be `linkerta`), `password` (optional) |
| POST | `/api/v1/glove-recording` | Start the guided raw-glove recording in a ready ROS2 session | `hand` (`left` or `right`), `label` (default `subject01`) |
| POST | `/api/v1/glove-recording/cancel` | Cancel raw-glove recording and retain incomplete data | None |
| POST | `/api/v1/snapshot` | Schedule a native scene snapshot | `delay_s` (default 30, configurable; 0..3600), `label` (default empty, at most 120 characters) |
| POST | `/api/v1/snapshot/cancel` | Cancel a pending countdown idempotently; capture already underway returns a conflict | None |
| GET, HEAD | `/api/v1/snapshots` | List native scene snapshots | None |
| GET, HEAD | `/api/v1/snapshots/{name:.+}` | Snapshot metadata/file list, or raw PNG/JSON download when the name ends in `.png`/`.json` | Path: `name`, including nested `<snapshot>/<file>` |
| GET, HEAD | `/api/v1/camera/status` | Camera state, devices, and quality information | Query: optional `camera_name` |
| GET, HEAD | `/api/v1/camera/preview.jpg` | Fresh JPEG preview; stale/not started returns 503 | Query: optional `camera_name` |
| GET, HEAD | `/api/v1/camera/photos` | List calibration photos | Query: optional `camera_name` |
| GET, HEAD | `/api/v1/camera/photos/{name}` | Download an original photo file | Path: `name`; query: optional `camera_name` |
| GET, HEAD | `/api/v1/camera/record/state` | Return upstream recording state/quality report unchanged | None |
| POST | `/api/v1/camera/{action:start\|stop\|snapshot\|record/start\|record/stop}` | Camera-set preview start/stop, PNG + same-frame intrinsics snapshot, or ego-tool recording start/stop with preview restoration as needed | Path: `action`; only `snapshot` reads body `label` (default empty) and query `camera_name`; other actions read no body/query fields |
| GET, HEAD | `/overlay/{filename}` | Static collection overlay, styles, fonts, and related assets | Path: `filename` |
| GET, HEAD | `/web/{filename}` | Static operator-web assets | Path: `filename` |
| GET, HEAD | `/api/v1/config` | Available backend/source/scene choices, site defaults, workbench/tasks, and snapshot settings | None |
| GET, HEAD | `/api/v1/status` | Session, recording, input health, disk, safety, streaming, snapshots, input_chain, and startup service_code | None |
| GET, HEAD | `/api/v1/events` | SSE status snapshots; read full status after reconnecting | None |
| GET, HEAD | `/api/v1/episodes` | List recordings under the current output root, optionally filtered by task | Query: optional `task` |
| GET, HEAD | `/api/v1/episodes/{id}` | Episode meta, integrity, and voided status | Path: `id` |
| POST | `/api/v1/retarget/switch` | Switch only the session-owned ROS retarget process | `method` (one of `config.retarget_methods`) |
| POST | `/api/v1/signal-source/switch` | Switch driving signal in a running ROS2 session | `source` (`teleop` or `real_robot`) |
| POST | `/api/v1/session/start` | Start a profile-based simulation or replay session | `backend`, `source`, `workbench`, `task`, `seed`, `scene_variant`, `replay`, `episode`, `duration_s`/`duration`, `rate_hz`, `loop_replay`, `ros_topic_prefix`, `arm_startup`, `motion_profile`, `retarget`, `autopilot_tasks`, `diagnostics`; see below |
| POST | `/api/v1/session/stop` | Stop this service's simulation and registered inputs; safe to repeat | None |
| POST | `/api/v1/recording/start` | Begin an episode after input-health checks | `note` (optional), `force` (boolean, default false; explicit authorization required) |
| POST | `/api/v1/recording/stop` | Return episode_id immediately; save and check asynchronously | None |
| POST | `/api/v1/notes` | Append a note | `text` (string) |
| POST | `/api/v1/episodes/{id}/void` | Mark an episode void while preserving original data; reject active recordings | Path: `id`; body: `reason` (default empty) |
| POST | `/api/v1/episodes/{id}/export` | Export an explicitly identified episode to LeRobot | Path: `id` |
| POST | `/api/v1/remote-control` | Toggle remote writes from the on-site browser session only; assistants must not call it | `enabled` (boolean) |
| POST | `/api/v1/view` | Select the fixed main camera; auxiliary views remain independent | `view` (must be `camera_3`) |
| POST | `/api/v1/glove-overlay` | Set display layers | `layers`: object with boolean `enabled`, `A` (glove skeleton), `C` (tracked G20 links); legacy `B` is ignored |

The camera action resource matches these five concrete POST paths: `/api/v1/camera/start`, `/api/v1/camera/stop`, `/api/v1/camera/snapshot`, `/api/v1/camera/record/start`, and `/api/v1/camera/record/stop`. Camera status, preview, photos, individual photo, and snapshot accept `?camera_name=<name>`. If omitted, CameraPanel selects the first started camera. Start/stop/record operate on the camera set and do not consume `camera_name`. These routes retain write permissions and do not drive hands or arms. LinkerTA queries and other real-device operations still require separate explicit authorization.

## Session Start Fields

Read `/api/v1/config` for effective defaults, which merge site configuration and server launch values. These defaults are not request fields. Its `estop_hint` is the expanded, resolved stop-script path from `site.estop_script()`, including private `station.estop_script` configuration, followed by ` (does not cut motor power)`; it is not a fixed tilde-path string.

| Field | Meaning and accepted values |
|---|---|
| `backend` | Online backend: `sapien` only. MuJoCo is an offline export/build path, not an online collection choice |
| `source` | `ros2`, `udp`, `replay`, or `ik_autopilot` |
| `workbench` | Workbench ID or profile path; code fallback `ar5_08_g20_bench_bimanual` |
| `task` | Task ID; code fallback `empty`, overridable by site/server defaults |
| `seed` | Optional nonnegative integer, or null; controls task randomization |
| `scene_variant` | Default `full`; `free-space` is a CLI/API debug variant with recording disabled |
| `replay` | Replay input path; required for `source=replay` unless `episode` supplies it. Use a layered episode directory for stateful replay |
| `episode` | Existing episode ID. The manager derives replay path, source, backend, scene variant, task, rate, and duration from the recording and disables looping |
| `duration_s`, `duration` | Positive duration in seconds, or null for unlimited; the handler renames `duration_s` to `duration`, replacing `duration` if both are supplied |
| `rate_hz` | Positive integer observation/recording rate |
| `loop_replay` | Replay-loop option, converted to a boolean by the manager; stateful replay still follows its restoration rules |
| `ros_topic_prefix` | ROS subscription prefix; default empty |
| `arm_startup` | `spec`, `first_command`, or `none`; falls back to site `collection.arm_startup`, then `spec` |
| `motion_profile` | `auto`, `station`, `legacy`, or a supported profile; default from site `collection.motion_profile`, then `auto` |
| `retarget` | Method from `config.retarget_methods`; default `manus_dex` |
| `autopilot_tasks` | For `ik_autopilot`, a configured task-set path listed by `config.autopilot_tasks`; manager rejects unknown paths |
| `diagnostics` | Boolean, default false; enables bounded diagnostics for this session only |

`output_root`, `resource_root`, UDP bind host/port, video host/port, `view`, and `realtime` are not body-controlled session options. The manager selects them through server/site configuration or internally. `--spec` and legacy scene files are retired.

## Status, Permissions, and Results

`service_code.git_commit` is the web service's startup Git commit; it does not change with a disk checkout. After deployment it must match local/deployed HEAD. `input_chain.timings` includes observed first_frame_s, first_manus_data_s, first_retarget_data_s, and first_linkerta_data_s. Missing fields mean not yet observed; do not infer first glove data from frequency.

`glove_health.left/right` contains message state and age_s, with null for unknown values. input_health reports frequency, age, counts, and decode_errors for four command inputs. Waiting UI state does not block rendering; recording still rejects unhealthy inputs through the existing gate.

POST middleware requires `Content-Type: application/json`, a nonempty `X-Gamepad-VLA-Client`, and either the local browser session cookie or locally enabled remote control. If present, Origin must match the request scheme and host. Only the local browser session can call `/api/v1/remote-control`. Read the returned message and state for ownership conflicts (409), disabled remote writes (403), unhealthy inputs (409), and service/input startup failures (503). Invalid parameters can return 400. Sending a request does not establish success.

Diagnostics default to off. Explicit capture is bounded per file by 900 seconds, 120000 records/unit, or 32 MiB; ordinary status/statistics update in memory. Stop ends only registered owned processes and does not stop manual input chains. Software Stop is neither an emergency stop nor power-off.
