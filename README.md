# dexstation

dexstation is a Python simulation collection runtime and local web console for dexterous manipulation. It connects input plugins, workbench and task profiles, physics, video, and layered episode recording. This initial release includes simulation and inputs. Real-robot mode, device drivers, and station services are not included.

## Install

Use an isolated environment. The pinned simulation wheel requires **CPython 3.10 on Linux x86_64, glibc 2.35 or newer**. Install from this source snapshot:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install '.[web,sim]'
```

The `sim` extra pins SAPIEN `3.0.3+concon.physx5.6.1.a3` to the custom wheel in [xbox_gamepad_physics v0.4.3](https://github.com/5o1/xbox_gamepad_physics/releases/tag/v0.4.3), with SHA256 in `pyproject.toml`, and `xbox-gamepad-physics==0.4.3`. Keep these pins; a stock SAPIEN wheel does not satisfy this contract. Other platforms can use the base package for analysis; they have not been validated for simulation.

## Two reference examples

**LinkerTA + Manus input:** the public input example uses `manus_anatomical` and the maintained G20 reference models. Independently install compatible ROS2 `rclpy`, `sensor_msgs`, and `manus_ros2_msgs` packages. See the [input tutorial](docs/usage/anatomical-retarget.md) and [file-level provenance](src/dexstation/inputs/linkerta_assets/PROVENANCE.md). Hardware input startup requires explicit authorization; tests use constructed messages.

**Simulated AR5 0.8 + G20 bench:** select workbench `ar5_08_g20_bench_bimanual` with task `empty`, `tabletop_cube_grasp`, or `xbox_gamepad_v3_3`. Reference calibration, poses, gains, and limits ship unchanged and require validation for your setup. External vendor assets are downloaded by users, not redistributed:

```bash
export DEXSTATION_WORKSPACE="$PWD/workspace"
export DEXSTATION_DATA="$PWD/data"
python scripts/fetch_resources.py --workspace "$DEXSTATION_WORKSPACE"
cp configs/site.example.json site.local.json
export DEXSTATION_SITE_CONFIG="$PWD/site.local.json"
dexstation-server --host 127.0.0.1 --port 8780 --enable-remote-control
```

Fetching verifies pinned files; it does not extract archives, derive models, or install environments. Prepare the gamepad release resources with its upstream build tools and configure `simulation.resources`; `scripts/derive_ar5_08_arms.py` prepares the paired arm description. See [resource notes](references/robot_assets.md) and [collection usage](docs/usage/collection.md). `linker-sim-assets` has **no license file published upstream** at the recorded pins; users must arrange the right to use its assets. AR5 remains on that source. Fresh-machine rendering and physical calibration have not been validated by this release assembly.

## Interfaces and data

The [architecture](docs/architecture.md), [plugin guide](docs/dev/plugins.md), and [HTTP API](docs/usage/collection-api.md) describe the implemented interfaces. Plugins use the five stable entry-point groups: `dexstation.sources`, `dexstation.backends`, `dexstation.object_types`, `dexstation.channel_types`, and `dexstation.tasks`. Public extension types are exported by `dexstation.api`.

The API remains `/api/v1`. Episodes keep `dexhand-episode-v1`; workbench recordings keep `dexhand-workbench-rec-v1`; gamepad recordings keep `xbox-gamepad-rec-v1`; raw glove recordings keep `manus-cdr-jsonl-v1`. These identifiers are data compatibility contracts. `dexstation-recover` repairs interrupted recordings while retaining a backup. Optional LeRobot v3 export requires known input targets.

Run hardware-free checks with `PYTHONPATH=src python -m pytest tests`. See [CONTRIBUTING.md](CONTRIBUTING.md), [MIT license](LICENSE), and [third-party notices](THIRD_PARTY_NOTICES.md).
