# External resources

The [resource inventory](external_resources.json) and maintained example inventory pin download URLs, revisions, sizes, and SHA256 values. Vendor payloads stay in the configured workspace and are not redistributed. Run `python scripts/fetch_resources.py --workspace WORKSPACE` to verify or download them; this does not extract or derive resources.

The simulated bench uses AR5 0.8 left/right arms and workbench references from `linker-sim-assets` at `d0224b8da58802fe7cb2e35dcc9771693c4543b6`. The old paired-right variant stays recorded at `e6bcf44ff0203534e14500cb836f57e9f9492292`. There is **no license file published upstream** at these recorded pins. Users must arrange asset use rights. AR5 remains on this source.

G20 corresponding geometry comes from `linkerhand-urdf`; PBR visuals use `g20_pbr_description` v0.1.1. Retain their Apache-2.0 notices. The D435 shell comes from `realsense-ros` and supplies visual geometry only. Gamepad resources and the custom SAPIEN wheel use the pinned xbox_gamepad_physics v0.4.3 release and its own notices. See [third-party notices](../THIRD_PARTY_NOTICES.md).

`scripts/derive_ar5_08_arms.py` derives the paired arm descriptions after resource acquisition. Gamepad release tools prepare SDF/visual resources separately. Set `simulation.resources` to the prepared resource root. Reference configuration values ship unchanged and require setup-specific calibration; geometry correspondence does not establish physical protocol or contact behavior.
