# Sources and Third-Party Notices

This file attributes copied code, model descriptions, and external resources. Public SDK interfaces and mathematical references used for project-authored rewrites are identified separately from copied files. Upstream projects and downloaded assets retain their own licenses.

## Adapted Code and Maintained Copies Distributed Here

Paths below identify distributed files. Provenance is recorded at file level; offsets, endpoints, poses, limits, gains, and landmark translations are project configuration. A dated snapshot or file hash is not a Git commit.

| Current path in this repository | Upstream repository URL | Pinned revision or snapshot | Upstream license | Source and adaptation |
|---|---|---|---|---|
| `src/dexstation/inputs/retarget_motor_clone.py` (four HandCore functions/utilities) | https://github.com/linker-bot/linkerhand-telop-ros2 | `12300b1e45b263a253cc88fdb7635f95587d0741` | MIT, Copyright (c) 2024 Linker Robotics; full text below and in module header | Exact `trans_to_motor_left/right` from `src/linkerhand_retarget/linkerhand_retarget/linkerhand/handcore.py:236-265`; exact `scale_value` and `is_within_range` from `linkerhand/utils.py:147-152`. Slot adapter rewritten by this project from public URDF joint order and command indices. |
| `src/dexstation/inputs/retarget_assets/g20_left.urdf`, `src/dexstation/inputs/retarget_assets/g20_right.urdf` | https://github.com/linker-bot/linkerhand-telop-ros2 | `12300b1e45b263a253cc88fdb7635f95587d0741` | MIT, Copyright (c) 2024 Linker Robotics; full text below and in each XML header | Public `src/linkerhand_retarget/linkerhand_retarget/assets/robots/hands/linker_hand/g20_{left,right}/linkerhand_g20_{left,right}.urdf`; visual/collision elements removed and XML reserialized. Nine tip/pad landmark frames are our calibration configuration used by retarget objectives. |
| `src/dexstation/inputs/retarget_assets/motor_mapping.json` | https://github.com/linker-bot/linkerhand-telop-ros2 | `12300b1e45b263a253cc88fdb7635f95587d0741` | MIT, Copyright (c) 2024 Linker Robotics; full text below | Public `src/linkerhand_retarget/linkerhand_retarget/config/hand_config.yml` G20 arrays converted to JSON, plus our left-thumb upper value 255 for HandCore encoding. |
| `src/dexstation/examples/ar5_g20/mapping.py` | https://github.com/linker-bot/linkerhand-telop-ros2 | `12300b1e45b263a253cc88fdb7635f95587d0741` | Project implementation MIT; public data basis MIT, Copyright (c) 2024 Linker Robotics | Independently implemented float encoder/decoder from public HandCore contracts, G20 URDF joint order/limits/mimics and `hand_config.yml` arrays. Endpoint directions, teleop thumb reversal and default pose are our configuration. |
| `src/dexstation/inputs/arm_mapping.py`, `src/dexstation/model/station_math.py` | Public interface/equation references in [provenance.json](src/dexstation/examples/ar5_g20/motion_assets/provenance.json) | LinkerTA `109766554f56728da458aac6d74780d30378d1b1`; xCoreSDK `061c2fb11f360f1585aae2776a4bb713b9e7c49c`; libfranka `e88d08f64cbf842b8f53c0b551963684064038d6` | Project MIT implementation; referenced SDKs Apache-2.0 | Rewritten Python/numpy implementation based on public degree/radian interfaces and the first-order RC equation. No SDK implementation code copied. Offsets, limits, startup policies and ramp parameters are project configuration. |
| `tests/fixtures/motion/sdk_filter_reference.json`, `tests/fixtures/motion/sdk_frequency_reference.json` | Historical numerical observations, not copied SDK code | xCoreSDK 0.5.1; library SHA256 `9d991a21e08363680436a9b042863b8729269256d904cc85f952a8408033a741` | Generated numerical test data; SDK license not established for the historical binary | Scalar call results and frequency-response samples, used only in tests. No SDK binaries distributed. |
| `src/dexstation/examples/ar5_g20/model.py` | not recorded for the vendor source | `station_20261014`; upstream Git commit not recorded | not recorded for the vendor snapshot | The same project's `assets.py` and `assets/usd/right/config.yaml`; only joint-order constants remain, without Isaac conversion code. |
| `src/dexstation/inputs/linkerta_reader.py`, `scripts/linkerta_reader_node.py`, `src/dexstation/inputs/linkerta_assets/linkerta.yaml` | https://github.com/linker-bot/linkerbot-teleop-arm (renamed from `linkerhand-tele-arm`) | Public protocol reference `4354083b27fd69f9d042d78edb6924e67f7f03fc` | Project MIT implementations; public reference declares Apache-2.0 in `src/linkerta/package.xml` | Reader and ROS adapter independently rewritten from the public protocol and our validity specification; signs, offsets, rates, timeouts, topics and QoS are our configuration (RW2-1). [File-level provenance](src/dexstation/inputs/linkerta_assets/PROVENANCE.md), [source hashes](src/dexstation/inputs/linkerta_assets/sources.json), [full Apache-2.0 text](src/dexstation/inputs/linkerta_assets/APACHE-2.0.txt) and [project attribution NOTICE](src/dexstation/inputs/linkerta_assets/NOTICE). No upstream LICENSE/NOTICE file exists at this pin; no modified station implementation is used as the rewrite basis. |
| `src/dexstation/render/web/app.js` | https://github.com/5o1/xbox_gamepad_physics | `v0.3.1` | MIT; copyright and full text below | Adapt `teleop/web/app.js`, retaining video packets and WebCodecs decoding for the operator UI and telemetry |
| `src/dexstation/task_plugins/xbox_gamepad_v3_3/web/` | https://github.com/sevaht/gamepad-overlay | `31236cd6a029828c8b01966ddc1a92c31e5cd852` | The Unlicense (`UPSTREAM_LICENSE`); project adapter/assets license MIT (`ASSET_PROJECT_LICENSE`) | Copied through the user's read-only xbox_gamepad_physics checkout; retain [upstream.json](src/dexstation/task_plugins/xbox_gamepad_v3_3/web/upstream.json) and license texts. The copied renderer's thresholds and signal adaptation were unchanged at import; recorded patches retain continuous input values/exact-change rendering and later D-pad/Guide/stick-highlight fixes |
| `src/dexstation/task_plugins/xbox_gamepad_v3_3/signals.py` | https://github.com/5o1/resnake_gym and https://github.com/5o1/xbox_gamepad_physics | Source integration commits not recorded; resource-package pin is separately `v0.4.3` | not recorded for the signal-contract source modules | Channel names/order correspond to `xinput-normalized-v1` and `sdl3-gamepad-core-v1` for protocol compatibility; module header names the sources. |

Motion configuration is ours and remains unchanged in `motion_assets/station.json`. [File-level provenance](src/dexstation/examples/ar5_g20/motion_assets/provenance.json), the [configuration ledger](src/dexstation/examples/ar5_g20/motion_assets/source_ledger.json), and the empty [vendor inventory](src/dexstation/examples/ar5_g20/motion_assets/vendor/provenance.json) describe the current distribution. No vendor MJCF/YAML copies or station C++ math are distributed.


### Linker Robotics MIT Notice

MIT License

Copyright (c) 2024 Linker Robotics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


### xbox_gamepad_physics License

MIT License

Copyright (c) 2026 5o1

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Downloaded Resources and External References

Models, gamepad resources, and external calibration solvers are fetched from upstream rather than bundled with this repository or Python distribution. The following paths are inventory destinations **outside the source tree**, relative to the configured workspace, unless a repository path is explicitly identified. File hashes, sizes, and license-file locations remain in [external_resources.json](references/external_resources.json) and the maintained workbench snapshot `src/dexstation/examples/ar5_g20/data/sources/external_resources.json`. See [robot assets](references/robot_assets.md).

| Item | Upstream repository URL | Pinned revision | Upstream license recorded | Current repository reference / external destination |
|---|---|---|---|---|
| G20-corresponding public L25 geometry | https://github.com/linker-bot/linkerhand-urdf | `075cc7d42cc1e756bdcbece0fc069a0779fc5237` | Apache-2.0; downloaded LICENSE | `references/external_resources.json` → `external/linkerhand_urdf_075cc7d4` |
| AR5 0.8R source (old scene) | https://github.com/linker-bot/linker-sim-assets | `e6bcf44ff0203534e14500cb836f57e9f9492292` | no license file published upstream | `references/external_resources.json` → `assets/ar5_08r_linker_sim_e6bcf44f` |
| AR5 0.8L/0.8R sources and derived arms | https://github.com/linker-bot/linker-sim-assets | `d0224b8da58802fe7cb2e35dcc9771693c4543b6` | no license file published upstream | `references/external_resources.json` → `assets/ar5_08_linker_sim_d0224b8d`, `assets/derived/ar5_08_d0224b8d` |
| Vendor workbench geometry and baseline | https://github.com/linker-bot/linker-sim-assets | `d0224b8da58802fe7cb2e35dcc9771693c4543b6` | no license file published upstream | Repository: `references/vendor_bench_baseline.json`, `src/dexstation/examples/ar5_g20/data/sources/baseline.json`, composed `workbench.json`, and `src/dexstation/task_plugins/xbox_gamepad_v3_3/data/fixtures.json`; external: `assets/linker_sim_bench_d0224b8d` |
| RealSense D435 body description | https://github.com/realsenseai/realsense-ros | `7a87097c499e3b0b69518d54a8e7a9ac1924e87c` | Apache-2.0; retain downloaded LICENSE and NOTICE.md | `references/external_resources.json` → `external/online_20261015/realsense_d435_d455` |
| Gamepad wheel/resources and custom SAPIEN release | https://github.com/5o1/xbox_gamepad_physics | `v0.4.3` | not recorded as one license identifier in the resource inventory; use each release asset's own licenses and third-party notices | `references/external_resources.json` → `assets/xbox_gamepad_physics_v0.4.3/release`; the web-code MIT notice does not replace other asset licenses |
| G20 PBR visuals | https://github.com/5o1/g20_pbr_description | `v0.1.1`; geometry corresponds to linkerhand-urdf L25 at `49f2602` | Apache-2.0; upstream geometry Copyright 2025 LinkerHand,Ltd.; retain archive NOTICE | `references/external_resources.json` → `assets/g20_pbr_description_v0.1.1`, unpacked root `g20_pbr_description/` |
| Retargeting library reference | https://github.com/dexsuite/dex-retargeting | Installed version `0.5.0`; Git commit not recorded | not recorded in the provenance record | `references/anatomical_retarget_sources.json`, `installed_library`; recorded optimizer.py, seq_retarget.py, and retargeting_config.py hashes, no bundled library |

Calibration reference data retains its project records; external tools keep their own licenses. Workbench internal-copy provenance is in [PROVENANCE.md](src/dexstation/examples/ar5_g20/data/PROVENANCE.md), with snapshot `6db85471a9ec6e401037ae8a2f56d50e4df05324` (`codex/wp2`). It records local calibration, baseline, override, base-scene, and resource-inventory copies; these internal snapshot identities do not supply missing vendor license fields.

## Offline Web Fonts

| Current repository path | Upstream repository URL | Pinned revision | License |
|---|---|---|---|
| `src/dexstation/server/web/overlay/fonts/IBMPlexSansCondensed-Regular.woff2` and `src/dexstation/task_plugins/xbox_gamepad_v3_3/web/fonts/IBMPlexSansCondensed-Regular.woff2` | https://github.com/IBM/plex | not recorded | SIL Open Font License 1.1; full text in `IBM-Plex-LICENSE.txt` beside each copy |
| `src/dexstation/server/web/overlay/fonts/JetBrainsMono-Regular.woff2` and `src/dexstation/task_plugins/xbox_gamepad_v3_3/web/fonts/JetBrainsMono-Regular.woff2` | https://github.com/JetBrains/JetBrainsMono | not recorded | SIL Open Font License 1.1; full text in `JetBrainsMono-OFL.txt` beside each copy |

Original font files are distributed with the web UI and wheel, without glyph changes. The page makes no external font requests.
