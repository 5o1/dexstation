# Simulation collection

The collection pipeline connects an input source, mapping, a physics backend, and layered recording. SAPIEN is the online simulation backend; MuJoCo supports offline export. The maintained renderer uses a separate twin, shared-memory snapshots, and NVENC video. Physics and recording contracts are unchanged.

## Configuration and startup

Copy `configs/site.example.json` to a local file and set `DEXSTATION_SITE_CONFIG` to its path. Set `DEXSTATION_WORKSPACE` to your external resource directory and `DEXSTATION_DATA` to your recording directory. Local configuration is never committed. `simulation.python` and `simulation.resources` select an existing simulation environment and prepared gamepad resources. Only use a rendering GPU you are authorized to use.

```bash
dexstation-server --host 127.0.0.1 --port 8780 --enable-remote-control
```

Open the local URL, select the workbench `ar5_08_g20_bench_bimanual`, task `empty`, and an input. Start simulation, then record explicitly. The other included tasks are `tabletop_cube_grasp` and `xbox_gamepad_v3_3`. The gamepad task requires its separately downloaded release resources. Inputs include UDP, replay, ROS2, and task-owned sources. See [anatomical input](anatomical-retarget.md) for LinkerTA and Manus.

Python entries prefer these environment names. Deprecated aliases remain accepted, with the first nonempty value winning from left to right and one deprecation warning per process:

| Preferred | Deprecated aliases |
| --- | --- |
| `DEXSTATION_HOME` | `DEXHAND_HOME` |
| `DEXSTATION_WORKSPACE` | `DEXHAND_WORKSPACE`, `GAMEPAD_VLA_WORKSPACE` |
| `DEXSTATION_DATA` | `DEXHAND_DATA`, `GAMEPAD_VLA_DATA` |
| `DEXSTATION_SITE_CONFIG` | `DEXHAND_SITE_CONFIG`, `GAMEPAD_VLA_SITE_CONFIG` |
| `DEXSTATION_SIM_PYTHON` | `DEXHAND_SIM_PYTHON`, `GAMEPAD_VLA_SIM_PYTHON` |
| `DEXSTATION_RENDER_DEVICE` | `DEXHAND_RENDER_DEVICE`, `GAMEPAD_VLA_RENDER_DEVICE` |
| `DEXSTATION_REMOTE_CODE` | `DEXHAND_REMOTE_CODE`, `GAMEPAD_VLA_REMOTE_CODE` |
| `DEXSTATION_COMPUTE_PROFILE` | `DEXHAND_COMPUTE_PROFILE`, `GAMEPAD_VLA_COMPUTE_PROFILE` |
| `DEXSTATION_CONDA` | `DEXHAND_CONDA`, `GAMEPAD_VLA_CONDA` |

Site/default fallback follows environment selection. Resolved site, workspace, and render settings propagate to children under supported spellings. The local remote-control switch, Origin checks, and client header gate writes. Session stop cleans up registered processes only. Real-robot mode and device services require a separate private extension and are not included in this release.

## Recording and recovery

Layered episodes contain `episode.json`, `workbench/`, and task-owned layers. Use `python scripts/check_collection.py integrity EPISODE --output RESULT.json` to check integrity. Stop recording normally before stopping a session. `dexstation-recover EPISODE` copies the original to `EPISODE.before-recover` and trims an interrupted recording to its valid common prefix; it does not certify completeness.

LeRobot export uses the existing numeric v3 interface. The 54-dimensional action requires known master-arm targets and raw motor messages for both hands. Unknown values are never fabricated. Per-frame diagnostics are off by default. Snapshots preserve the completed physics state rather than capturing a browser composite. See the [API](collection-api.md) and [plugin guide](../dev/plugins.md).
