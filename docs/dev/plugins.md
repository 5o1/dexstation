# Extension registries

Built-ins register in code, so a source checkout works without package installation.
Installed extensions expose **entry objects**, not registration functions, in these
Python entry point groups:

| Group | Entry object | Registry module |
| --- | --- | --- |
| `dexstation.sources` | `SourceEntry` | `dexstation.inputs.registry` |
| `dexstation.backends` | `BackendEntry` | `dexstation.backends.registry` |
| `dexstation.object_types` | `ObjectType` | `dexstation.backends.object_types` |
| `dexstation.channel_types` | `ChannelType` | `dexstation.backends.channel_types` |
| `dexstation.tasks` | `TaskPlugin` | `dexstation.runtime.plugins` |

These group names will survive the package rename. Discovery runs once per group
per process, on first use. Every successful load is logged. Import, validation and
duplicate-name failures are logged and appear in `/api/v1/config` as
`plugin_status.errors` entries with `group`, `name` and `reason`. Other plugins and
built-ins remain usable. The field is absent when there are no discovery failures,
which preserves the built-in response. Restart the process after installing plugins.
Entry-point imports must not connect to devices or start threads or processes.

The REST client discovers plugins only for the selected `session-start` command
(including its help). Software stop, status, recording controls and other control
commands import only the standard library and code defaults; they never discover
or import plugins, so broken or slow extensions cannot delay the local stop.

## Sources

`SourceEntry(name, factory, capabilities=SourceCapabilities(), validate=None,
api_order=100)` uses a `factory(SourceContext)` callable. The context provides
`options`, `spec`, `mapper`, `scene`, `profile`, `replay_episode`, `replay_meta` and
`initial`. The optional `validate(options)` raises `ValueError` on invalid options.
CLI order is registration order; API order sorts by `api_order` and keeps ties in
registration order. The original built-in orders are preserved.

`SourceCapabilities` declares:

- `clock`: `wall` (monotonic time) or `simulation` (last observed simulation time).
- `hardware_lease`: acquire the session lease before calling the factory.
- `teleop_calibration`: apply the scene's existing teleop offsets to the mapper.
- `motion_profile`: the profile selected by `auto`, normally `legacy` or `station`.
- `real_robot_mirror`: wrap the source in the existing calibrated mirror switch.
- `task_declared`: require the source ID in the task's `sources` list.
- `latest_input`: use the existing latest-event selection at each native step.
- `replay`: restore replay metadata and use the existing simulation frame clock,
  including its motion-period offset; replay inputs retain their own loop policy.
- `ros_topics`, `input_chain`, `health_gate`, `glove_recording`: opt into the
  corresponding existing ROS topic status, managed chain, recording health gate
  and glove capture behavior.
- `flattened_observation`: adapt a layered observation to the existing flat source
  callback format. `task_set_option` identifies an existing task-set option field.

A source supplies `poll(now) -> commands`, `drain_events() -> events`,
`boundary() -> (events, received_count)`, `close()`, `health.snapshot()`,
`decode_errors` and `received_packets`. Commands and events use the existing input
contracts. Optional observation, contact and status hooks retain their current
interfaces. These capabilities do not authorize physical execution; hardware use
still requires separate explicit authorization and the local control switch.

Minimal package:

```toml
[project.entry-points."dexstation.sources"]
lab_input = "lab_extension:source_entry"
```

```python
from dexstation.runtime.health import InputHealth
from dexstation.inputs.registry import SourceEntry, SourceCapabilities

class LabInput:
    def __init__(self):
        self.health = InputHealth()
        self.decode_errors = self.received_packets = 0

    def poll(self, now):
        return []

    def drain_events(self):
        return []

    def boundary(self):
        return [], self.received_packets

    def close(self):
        pass

source_entry = SourceEntry(
    "lab_input", lambda context: LabInput(),
    SourceCapabilities(clock="simulation", motion_profile="legacy"),
)
```

## Backends

`BackendEntry(name, factory, versions={}, imports=(), online=True, version='',
check=None, supports_workers=False, timestep=..., motion_adapter=None)` constructs
with `factory(scene, **kwargs)` and implements the collection backend contract.
`self_check()` calls the optional zero-argument `check` (raise on failure), imports
only declared modules, checks exact distribution versions and reports
`name`, `available`, `reason`, and `check='import+version'`. Optional task asset
versions can be checked without importing their packages. `version` is recorded
in scene identity. `timestep(backend)` reads the legacy timestep;
`motion_adapter` selects an existing motion-profile numerical adapter.

The online list drives CLI choices, composition and the API. The page displays
available entries from that list. SAPIEN is the sole built-in online engine;
MuJoCo remains export-only. Dependency constraints live in each backend entry.

## Objects and channels

`ObjectType(name, schema={}, static=False, mass=False, legacy_gamepad=False,
environment_table_only=False, operations={})` extends task validation with its
schema and selects operations. External entries supply callables; built-ins may
use local handler names. Operation signatures are:

- `compose(obj_dict, pose, spec_data)` projects compatible legacy scene fields.
- `sapien_load(scene, obj, render=False, physics_scene=None, work_dir=None,
  collisions=True)` returns a `TaskObject`.
- `mujoco_build(parent, mj, obj, attachment=None)` returns body names for export.
- `sapien_export_load(scene, obj, root, description, work_dir, sapien,
  physics_scene, physics_urdf)` returns `(actor, optional_gamepad_handle)`.
- `inventory(obj_dict, description, root)` returns joint, link and asset-drive names.
- Optional `export_resources(scene, resource_root, output, assets, description)`
  adds task-owned export resources.

Unsupported operations raise explicit errors. Flags preserve existing static,
mass and legacy gamepad projection/collision behavior; extensions should normally
leave the compatibility flags false. Primitive geometry dispatch remains inside
primitive handlers.

`ChannelType(name, metadata=None, sapien_observer=None)` supplies either
`(shape, dtype, unit)` metadata or `metadata(raw_channel)`, and
`sapien_observer(backend, task_object, channel_spec, raw_channel) -> value`.
Absent metadata requires explicit shape, dtype and unit. Absent readers remain
unsupported by the online engine, as with today's `contact_force` and `plugin`
channels. Gamepad object/channel entries register from the task plugin's `gamepad_types`; the
`xbox_gamepad_physics` package loads only when gamepad assets are used.

## Task plugins

Installed task packages expose a `TaskPlugin` entry object in **`dexstation.tasks`**:

```toml
[project.entry-points."dexstation.tasks"]
lab_task = "lab_task:entry"
```

Use the task ID as the entry-point name so a failed load can be associated with
that task's session start. `dexstation.runtime.plugins` provides
`register_task(entry)`, `task_plugins() -> dict[id, TaskPlugin]`, and
`task_plugin(id, required=False) -> TaskPlugin | None`. Declarative tasks can
continue to live under `configs/tasks` without a Python plugin.

```python
from pathlib import Path
from dexstation.runtime.plugins import TaskPlugin

# hooks is a Python module or object with the optional hooks described below.
entry = TaskPlugin(
    id="lab_task", package=hooks,
    data_directory=Path(__file__).parent / "data",
    web_directory=Path(__file__).parent / "web",
)
```

`TaskPlugin(id, package, data_directory, web_directory, sources=(), activate=None,
api_version=1, modules={})` declares:

- `package`: an importable package name or a hook object. Its optional
  `API_VERSION` must equal 1; the entry's `api_version` must also equal 1.
- `data_directory`: a directory containing `task.json` and its relative data
  files. The profile ID must match the entry ID.
- `web_directory`: the directory served by `/tasks/<id>/web/<file>`. The server
  preserves the existing regular-path containment check.
- `sources`: a tuple of R2 `SourceEntry` objects. Their factories and validators
  can be lazy callables. Source names must be unique across registries/tasks.
- `activate() -> None`: an optional registration hook called once when this task
  profile is loaded for composition. In a fresh renderer/export/replay process,
  an unresolved object/channel kind also activates the registered task whose
  profile declares that kind. Register its `ObjectType` and `ChannelType`
  entries here, using the R2 interfaces above. This must not construct a scene,
  import optional physics assets, or touch hardware.
- `modules`: optional mappings from profile module strings to module names or
  objects. Without an explicit mapping, a profile module must be a descendant of
  `package`. No Python import or web path is inferred from the task ID.

External entries and their package hooks are checked during discovery. Failures
use the same log and `plugin_status.errors` reporting as the other entry groups;
other tasks remain usable. Activation failures are also reported when that task
is first selected. Built-in manifests register in code and supply lazy
source metadata before external source discovery, preserving the original CLI
and API ordering. Merely listing sources does not import the gamepad task
implementation. No built-in entry points are advertised by this distribution;
that would duplicate code registrations in installed checkouts.

The implemented Python hooks are:

| Hook | Signature | Consumer and timing |
| --- | --- | --- |
| `ACTIONS` | `dict[str, Callable[[ActionContext, dict], object]]` | `TaskRuntime` resolves each declared action's `plugin` symbol at session start; invokes it on an accepted action. Existing named callable attributes also remain supported. Recording/action policy is unchanged. |
| `cell_state` | `(task_status: dict) -> dict[str, JSONValue]` | `TaskRuntime.status()` passes a copy of task identity/channels/actions; merges validated results by cell ID. Existing error isolation and journal behavior apply. |
| Declared source | `SourceEntry.factory(SourceContext) -> source` and optional `validate(options) -> None` | R2 source registry validates the options and constructs the source. `TaskRuntime` resolves the task profile's source `module` through its registered entry before construction. The source protocol and capability values are unchanged. |
| Object/channel registration | `activate() -> None` | Profile loading calls this before validating object/channel kinds. Register using `register_object_type(ObjectType(...))` and `register_channel_type(ChannelType(...))`. Object handlers can be lazy callables. |
| `task_sets` | `(data_directory: Path, source: str | None) -> list[str]` | Config defaults/listing and session-start validation ask the task entry for its task sets. The gamepad hook reads the existing `sources[].task_sets` glob from its profile. Returned paths are strings, matching the existing config API. |
| `evaluate` | `(episode: Path, **kwargs) -> object` | Offline consumers call `task_plugin(id).evaluate(episode, **kwargs)`. It resolves `task.json`'s `evaluation.offline` module and invokes its `evaluate` function. The gamepad implementation accepts `tasks_path=None` and returns the existing `(summary, results, false_events)` tuple. |

The game's declared `autopilot` module exposes `make_source(context)` and its
`evaluate` module exposes `evaluate(episode, tasks_path=None)`. Their current full
module strings in task.json are unchanged. General `observe`, `observations`, and
`make_source(scene, options)` hooks beyond the registered source above are **not
implemented**. A source's existing `observe(observation)` callback is still part
of the source protocol; it does not implement a general task observation hook.
The autopilot's optional `contact_reader(backend) -> callable` adapts the existing
force-sign/object convention for that source; the controller otherwise passes
`backend.contact_force` unchanged. Its optional `status_fields() -> dict` source
hook supplies the existing top-level `autopilot` status payload; the core retains
the nullable API key and has no autopilot-specific callback dispatch.

Task cells retain `mount(root, api)` and optional `unmount()`. Module subscribers
receive `status.task`; `api.status()` returns the full status. The mounter also
provides generic session-form extensions:

- `api.config`: the `/api/v1/config` response, including core `panes` and
  `pane_source_capabilities` for grid presentation.
- `api.addSessionControl(node)`: mounts a task-owned node in the session form;
  the mounter applies the cell's `visible_when` rule and removes it on teardown.
- `api.sessionOptions(read: () -> dict)`: contributes values to the session-start
  body. The page awaits cell loading before reading these values. Later cells
  overwrite matching keys; the page's standard session fields take precedence.

The gamepad selector uses these APIs to retain its label, filename display,
default, selected path and `ik_autopilot` visibility rule. Its overlay pane hosts
the existing vendored renderer in an iframe to isolate its window globals;
canonical signals are sent from the task channels, without browser-device input.

R3-1 keeps generic URDF `Chain` and `_rotation` in `kinematics.py`. The task IK
module imports them from the core. `Chain` consumes explicit URDF paths and has
no gamepad assets or `load_asset` dependency. Task measurements and `load_asset`
remain in the gamepad package.


### Grid panes and auxiliary cameras

Task profiles may declare `panes` alongside below-grid `cells`. Both web
locations use the same `mount(root, api)` / `unmount()` interface and status,
action, error-isolation, and session-control lifecycle. `GET
/api/v1/tasks/<id>/cells` returns `cells`, combined core/task `panes`, and
`actions`; module and CSS URLs use the task's web directory. A pane has `id`,
`title`, optional integer `weight`, and exactly one of `camera` or `module`.
Core information panes are supplied by the console. Web pane modules receive a
root that fills their grid rectangle; use its available height rather than a
fixed pixel height. The console overlays a title and provides internal overflow
for text content. The gamepad SVG preserves `xMidYMid meet` inside its iframe.

```json
{
  "auxiliary_cameras": [{
    "id": "inspection", "object": "fixture",
    "position": [0, 0.005, 0.34], "target": [0, 0.005, 0], "up": [0, 1, 0],
    "fit_render_meshes": true, "framing_margin": 1.12, "atlas_slot": 0
  }],
  "panes": [{"id": "inspection", "title": "Inspection", "weight": 10,
             "camera": "inspection"}],
  "cells": []
}
```

Camera vectors are in the named object's composed root frame. Framing reads all
attached render mesh parts, preserving shape scale, shape-local pose, current
link poses, FOV, and near plane. The existing fixed-root gamepad camera results
are preserved. This initial interface anchors unfitted vectors to the composed
root; it does not introduce a moving-root camera control policy. Mesh fitting
is generic and needs no gamepad hook. `fit_render_meshes: false` retains the
specified vectors. A configured auxiliary `framing_margin` overrides the task's
margin, preserving existing render settings.

The atlas stays 2×3 at 1920×1620 by default, with the same mmap/NVENC and
WebSocket protocol. Task camera slots are integers 0–4; Table top reserves slot
5. Missing slots remain black. The gamepad task uses slots 0–3; slot 4 is now
unused. Slots are independent of pane weights. Auxiliary images are operator
outputs, outside the calibrated collection-camera list. Task declarations are
stored with the task profile as recording provenance; observation/camera/action
formats are unchanged.

`plugins.CORE_PANES` owns Table top (60), Arm takeover (90), and Status (100).
Task weights belong in `task.json`: the gamepad task owns its four videos
(10/20/30/40), Gamepad signals (50), and Autopilot (85). Order is ascending
weight, then ID; missing weights sort last, by ID. Private `web.pane_weights`
(default `{}`) overrides known IDs across installed and declarative tasks;
unknown IDs fail startup without importing task implementation packages.

`visible_when` is an exact-match map over effective UI status. In addition to
`source`, it can use `source_capabilities.teleop_calibration`. The console obtains
that capability from the source registry, so new live master-arm sources need no
name-based special case. Before session start, the selected source is effective;
starting/running/stopping sessions use their own source. Autopilot is a grid pane
with `visible_when: {"source": "ik_autopilot"}`. Arm takeover uses the capability
rule. The gamepad task has no below-grid cells; other tasks may retain them.

The toolbar persists camera/web-pane choices in `collection.auxiliaryViews`.
Unknown saved IDs are ignored. On first migration, new web panes default to
checked; `collection.paneSelectionVersion` distinguishes this migration from an
operator's later explicit unchecked choice. Task changes preserve saved choices.

## Injected auxiliary services

`make_app(manager=None, *, services=None, ...)` and
`CollectionController(options, *, services=None, ...)` accept a `Services` bundle
from `dexstation.contracts.services`. `SessionManager(..., services=None)` passes
that same composition into spawned controller workers. Construct bundles from
module-level factories so the spawn start method can serialize them. This adds
no entry-point group; the five groups above retain their names and collision
rules.

The public contracts are:

| Service | Interface |
| --- | --- |
| Managed input | `preflight()`, `launch(manual=False)`, `status()`, `stop(include_manual=False)`, `switch_retarget(method)`; `retarget` and `manual` state |
| Method catalog | `Services.retarget_methods` contains accepted IDs in display order; `defaults` selects a method |
| Camera | `status(camera_name=None)`, `start()`, `stop()`, `preview_jpeg(camera_name=None)`, `snapshot(label='', camera_name=None)`, `photos()`, `photo(name)`, `record_start()`, `record_stop()`, `record_state()` |
| CAN preflight | `check(role)` and `require(role)`, both non-mutating |
| Optional CAN bring-up | `bring_up(role, password=None)`, enabled only by an explicit composition's `can_bring_up` flag |
| Mirror switching | `wrap_source(source, context)` returns the source or an adapter with `switch(mode)` and `real_status()` |
| Overlay geometry | `geometry_provider(retarget=method)`; optional `fingerprint_resources(retarget=method)` declares reproducibility inputs |
| Glove capture | `glove_recorder_factory(root)` constructs the generic job lifecycle with an explicitly supplied command factory |

Public defaults select UDP for the console and `manus_anatomical` for the input
example. The camera returns `available: false` and `mode: "unavailable"`; the CAN
bring-up route remains registered and returns HTTP 409 with
`error_code: "bring_up_unavailable"`. A public input adapter can use an injected
`CanSystem` without obtaining privileged setup. Configure its independently
installed ROS environment through `input_example.ros_setup`, `ros_environment`,
`python`, `log_root`, and `ros_domain_id`. No mirror checkout is read or executed
by the public adapter. Glove recording needs an explicit launch command service.

A site may select exactly one explicit bootstrap with
`services.bootstrap: "extension.module:create_services"`. The callable returns
`Services`; construction must not start devices or processes. An optional
module-level `shell_exports()` supplies extension shell settings. The public
configuration loader and CLI have no station import. Explicitly supplied
`services=` takes precedence over configured service selection.

Private extensions can inject input catalogs, device services, camera presentation,
and source wrappers through the same bundle. The public runtime has no private
implementation dependency. Construction alone must not start hardware.

UI fragments and camera presentation travel in the bundle and use existing web
routes. Public UI renders generic camera state; USB details and privileged CAN
dialogs belong to the station composition. This preserves the station's API
config fields and the route inventory. The bimanual AR5/G20 v1 data contract,
format IDs, units, reference mapping and control policy remain unchanged.
