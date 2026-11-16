# Anatomical MANUS Retargeting and Recording

The public input example selects `manus_anatomical`. It uses the owned anatomical vector and DexPilot objectives, with the packaged G20 reference models and motor conversion. Input and output remain the bimanual AR5/G20 v1 contract; hand output uses 20 motor values in the existing ROS `JointState` payload.

The ROS node subscribes to `/manus_glove_0` and `/manus_glove_1` and publishes `/cb_left_hand_control_cmd` and `/cb_right_hand_control_cmd`. Prepare independently installed compatible `rclpy`, `sensor_msgs`, and `manus_ros2_msgs` packages, and configure `ROS_DOMAIN_ID` for your environment. The node contains no device SDK or CAN driver.

```bash
PYTHONPATH=src python -m dexstation.inputs.retarget_node --method manus_anatomical
```

Settings live in `src/dexstation/inputs/retarget_assets/anatomical.json`. Targets use meters and a right-handed palm frame, preserve chirality, reject malformed skeletons, and apply the existing temporal output filter. These reference settings need lab-specific validation before physical use.

Raw recording helpers preserve every received CDR payload and decoded field, with receive times and DDS source times recorded separately. `GloveRecording` stores `messages.jsonl` and completeness metadata; interrupted recordings remain incomplete. No received messages is a recording failure. Received-message preservation does not guarantee that DDS drops no frames.

The optional web recording job requires an injected command factory for your independently installed ROS environment. This release includes `manus_anatomical`; additional device integrations are separate extensions.
