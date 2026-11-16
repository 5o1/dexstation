# Architecture

`dexstation.api` exposes the extension types and `Services` composition contract. The public runtime can run without a device extension. Configuration combines code defaults, a local JSON file validated against the packaged site schema, and supported environment overrides. Registries discover the five `dexstation.*` entry-point groups and isolate plugin discovery failures.

## Collection flow

Input sources produce observations and command targets. Mapping and control apply the existing calibration, workspace limits, takeover policy, and motion profiles. The composer loads a workbench plus a task into `RuntimeScene`; profiles carry immutable identities and resource digests. The AR5/G20 workbench is the supplied output example, while tasks own their objects, evaluation, recording channels, and optional web panes.

`runtime.controller` coordinates the source, backend, task runtime, and episode writer. `contracts.backend` includes the hardware-free fake backend used in tests. SAPIEN is the online physics backend; MuJoCo is an offline model export. Backend and object registries keep task-specific optional dependencies out of generic empty/cube startup.

The rendering twin consumes physics snapshots through the existing shared-memory pipeline. Operator streaming uses NVENC and browser WebCodecs. Task-defined auxiliary panes render independently. Snapshots preserve a completed simulation state, camera intrinsics, and extrinsics; display-only glove overlays do not modify control targets or episode physics.

## Server and service composition

`server.app` owns the aiohttp application, `/api/v1`, SSE, and `SessionManager`. The CLI calls the same server entry point. Configuration returns installed capabilities; the web UI uses those capabilities to expose controls. Writes require the local remote-control switch and existing request authorization checks. Session stop releases only processes registered by that session.

`config.service_defaults` supplies generic services, including the public input catalog. A configured `module:callable` bootstrap can supply extension implementations through `contracts.services.Services`. Camera presentation, health, command factories, and device services are injected rather than imported from a private implementation. Real-robot execution is not part of the public distribution.

## Data and resources

Recordings retain the layered episode, workbench, task, raw glove, and signal format identifiers. Integrity checks validate the recorded common prefix; recovery first retains the original episode. Export never invents missing master targets or raw motor inputs.

Packaged examples include reference configuration and file-level provenance. Vendor payloads remain external and are acquired by the user from pinned manifests. Paths resolve through the configured workspace or packaged resources. Calibration values are reference values; release assembly does not claim physical calibration or measured rendering acceptance.
