# Contributing

Use a private virtual environment and run `PYTHONPATH=src python -m pytest tests`. Ordinary tests must use fake devices and must not launch hardware nodes or send CAN frames. Physical execution requires separate explicit authorization. Do not bypass the local remote-control switch; stop only registered processes.

Preserve data schemas, format identifiers, arm mapping, actuator gains, collision policies, reference values, and the pinned SAPIEN/gamepad contracts. Control changes require primary-source research and measured validation. Keep downloaded assets, local configuration, data, and run outputs outside source control. Never mutate a shared environment. Reused resources must be maintained ordinary files with file-level provenance.
