# Profile provenance

Source repository snapshot: `6db85471a9ec6e401037ae8a2f56d50e4df05324` (`codex/wp2`).
All files below are maintained, byte-identical regular-file clones within this worktree. No symlinks or runtime reads of an upstream workstation are used.

| Repository source | Local clone | SHA256 |
|---|---|---|
| `configs/workcell/teleop_calibration.json` | `calibration/teleop.json` | `9c02e12e83215e7e25c794b5820be851daaef95cb41edb568c3d03134b576608` |
| `configs/workcell/real_robot_calibration.json` | `calibration/real_robot.json` | `2f9d66c5b384f0e3216fe25a94e1e68f19e7a75be7e65bb32ad988970795d4c3` |
| `references/vendor_bench_baseline.json` | `sources/baseline.json` | `a6c1f448e0b0c520b652b2cd3d03c16aead9e977784d9a77f2df180c255d9043` |
| `configs/workcell/vendor_bench_overrides.json` | `sources/overrides.json` | `3f3ef47992f2c152def529471ba1c20dab0ed15221ca51b8b935ceaebe45eebd` |
| `configs/workcell/placement_trials/gamepad_fixture_v3_3.json` | `sources/base_scene.json` | `c5d24d987bce98e2450d5f309c95fd3061b02af01c7a17238dd65c7ed98e6c89` |
| `references/external_resources.json` | `sources/external_resources.json` | `0cbae0409c4c3188dc942bc6a08e36cde44305cbce9ba8c9d8955586818f4c67` |

Generation: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/vendor_bench_scene.py compose --workbench`.
Geometry comes from the pinned vendor baseline and recorded overrides; the workbench keeps 50 station fixtures. The 69 v3.3 task pieces are emitted as static task objects in `configs/tasks/xbox_gamepad_v3_3/fixtures.json`.
Reviewer rulings 1-3 move the right-hand pregrasp to task initial_robot_state, separate collision_mesh from collision participants, and preserve the original 13 colliding proxies / 56 visual-only pieces. No geometry, actuator, mapping or collision-policy edits are made.
The two legacy calibration files remain byte-identical compatibility copies for readers owned by other work packages. Update canonical facts here; coordinate those readers/copies during integration.
Ruling 5 includes referenced bytes and pinned external asset metadata in the canonical profile digest; limits.json / takeover.json are null references until supplied by WP5. Regenerate after changing referenced configuration.
External URDF/mesh assets are not copied, opened or hashed by this package; their pinned versions/SHAs remain in sources/external_resources.json and the existing fixture provenance at the maintained reference configuration
This package has not validated physical geometry, rendering, contact, hardware connectivity, or real-time replay. Those session-start checks belong to WP3/WP5/WP8a per reviewer ruling 4.

## Calibration reference

`calibration/master_linkerta_range.json` contains reference measurements used in master-mapping design, not loaded by `workbench.json`. Raw recordings and maintenance records remain private. Numerical reference values are unchanged and require validation for another setup.
