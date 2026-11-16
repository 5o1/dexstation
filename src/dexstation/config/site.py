"""Generic validated configuration and runtime paths."""
import argparse
from copy import deepcopy
from functools import lru_cache
import json
from importlib import import_module
import logging
import os
from pathlib import Path
import shlex

from dexstation.contracts.validation import _validate
from dexstation.config.defaults import SITE_DEFAULTS
from dexstation.config.environment import environment, environment_exports, publish_environment


def repository_root():
    return Path(__file__).resolve().parents[3]


def home():
    """Root of the independent dexhand projects (not the user's HOME)."""
    return Path(environment("HOME") or "~/dexhand").expanduser().resolve()


def default_path(new_default, legacy_default, *, directory=None):
    """Use the new layout only when its directory exists; never create it."""
    new = Path(new_default).expanduser()
    directory = Path(directory).expanduser() if directory is not None else new
    return (new if directory.is_dir() else Path(legacy_default).expanduser()).resolve()



def site_config_path():
    explicit = environment("SITE_CONFIG")
    if explicit:
        return Path(explicit).expanduser().resolve()
    # The new layout is detected under home() only. A legacy workspace (for example
    # ~/gamepad_vla) may contain code/.local or data/ for unrelated reasons.
    new = home() / "workstation/code/.local/station.json"
    if new.parent.is_dir():
        return new.resolve()
    return repository_root() / "configs/site.local.json"


def _unknown_keys(value, schema, prefix=''):
    """Collect unknown fields while permitting schema-defined dynamic mappings."""
    if not isinstance(value, dict):
        return []
    result = []
    for key, child in value.items():
        path = f'{prefix}.{key}' if prefix else key
        child_schema = schema.get('properties', {}).get(key)
        if child_schema is None:
            additional = schema.get('additionalProperties', True)
            if isinstance(additional, dict):
                child_schema = additional
            else:
                result.append(path)
                continue
        result.extend(_unknown_keys(child, child_schema, path))
    return result


@lru_cache(maxsize=16)
def _load_site(local, explicit):
    return read_site(local, explicit, SITE_DEFAULTS)


def read_site(local, explicit, defaults):
    """Read a validated site with caller-supplied defaults, without global mutation."""
    values = deepcopy(defaults)
    if explicit or local.exists():
        override = json.loads(local.read_text())
        if not isinstance(override, dict):
            raise ValueError('site: expected object')

        def merge(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = value
        merge(values, override)
    schema = json.loads((Path(__file__).parents[1] / 'contracts/schemas/site.schema.json').read_text())
    _validate(values, schema, label='site')
    if values['web']['pane_weights']:
        from dexstation.runtime.plugins import known_pane_ids
        for pane_id in values['web']['pane_weights']:
            if pane_id not in known_pane_ids():
                raise ValueError(f'Unknown web.pane_weights pane id: {pane_id}')
    unknown = _unknown_keys(values, schema)
    if unknown:
        logging.getLogger(__name__).warning('Unknown site configuration keys: %s', ', '.join(sorted(unknown)))
    return values


def load_site(*, reload=False):
    """Load and validate each selected private file once; reload clears the cache.

    Return a copy so callers cannot mutate the cached effective configuration.
    The example file is documentation and is never read here.
    """
    if reload:
        _load_site.cache_clear()
    explicit = bool(environment("SITE_CONFIG"))
    return deepcopy(_load_site(site_config_path(), explicit))


def reload_site():
    """Explicitly reread private settings, for tests or deliberate reconfiguration."""
    return load_site(reload=True)


def site_value(key, default=None, *, required=False):
    value = load_site()
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            value = default
            break
        value = value[part]
    if required and (value is None or value == ""):
        raise ValueError(f"Set {key} in configs/site.local.json or DEXSTATION_SITE_CONFIG (deprecated aliases are also supported)")
    return value


def workspace_root(*, legacy_default="~/.local/share/dexstation"):
    value = environment("WORKSPACE") or site_value("workspace")
    return Path(value).expanduser().resolve() if value else default_path(home() / "workstation", legacy_default)


def data_root():
    value = environment("DATA") or site_value("data_root")
    return Path(value).expanduser().resolve() if value else default_path(home() / "workstation/data", "~/gamepad_vla_data")




def resolve_can_interface(role='linkerta', *, interface=None, sysfs_root=Path('/sys/class/net'), roles=None):
    """Resolve one role by its USB path, with an optional explicit iface override.

    USB paths are suffixes of /sys/devices/.../usbN/...; match full components,
    never a substring (1-1 must not match 1-12). Ambiguity is an error.
    """
    if interface:
        if '/' in interface or not interface or interface in ('.', '..'):
            raise ValueError('invalid CAN interface override')
        return interface
    roles = site_value('station.can_roles', {}) if roles is None else roles
    config = roles.get(role, {})
    if config.get('interface'):
        return resolve_can_interface(role, interface=config['interface'])
    usb_path = config.get('usb_path', '').strip('/')
    if not usb_path:
        raise ValueError(f'Configure station.can_roles.{role}.usb_path')
    suffix = tuple(Path(usb_path).parts)
    matches = []
    for path in Path(sysfs_root).iterdir():
        if (path/'device').exists() and (path/'type').read_text().strip() == '280':
            resolved = (path/'device').resolve().parts
            if resolved[-len(suffix):] == suffix:
                matches.append(path.name)
    if len(matches) != 1:
        raise ValueError(f'CAN role {role}: expected one USB match, found {len(matches)}')
    return matches[0]




def simulation_python():
    """Interpreter for simulation and rendering workers."""
    value = environment("SIM_PYTHON") or site_value("simulation.python")
    return Path(value).expanduser() if value else workspace_root() / "envs/sapien-3.0.3concon-xgp0.4.3-py310/bin/python"


def gamepad_resources():
    value = os.environ.get("XBOX_GAMEPAD_PHYSICS_RESOURCES") or site_value("simulation.resources")
    return Path(value).expanduser().resolve() if value else workspace_root() / "assets/xbox_gamepad_physics_v0.4.3/resources"


def configure_runtime():
    """Keep explicit environment overrides; otherwise use private site settings."""
    config = environment("SITE_CONFIG")
    if config or site_config_path().exists():
        publish_environment("SITE_CONFIG", site_config_path())
    # Renderer/gateway children change cwd to their run directory. Old consumers
    # still read GAMEPAD_VLA_*; publish the effective values under all names.
    publish_environment("WORKSPACE", workspace_root())
    value = site_value("compute.cuda_visible_devices")
    if value:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(value))
    render = environment("RENDER_DEVICE") or site_value("compute.render_device")
    if render:
        publish_environment("RENDER_DEVICE", render)
    os.environ["XBOX_GAMEPAD_PHYSICS_RESOURCES"] = str(gamepad_resources())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--get")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    if args.get:
        value = site_value(args.get, required=True)
        print(json.dumps(value) if isinstance(value, (dict, list)) else value)
    elif args.shell:
        values = environment_exports({
            "HOME": str(home()),
            "WORKSPACE": str(workspace_root()),
            "DATA": str(data_root()),
            "REMOTE_CODE": environment("REMOTE_CODE") or site_value("compute.remote_code") or str(default_path(home() / "workstation/code", workspace_root() / "code/gamepad_vla")),
            "COMPUTE_PROFILE": environment("COMPUTE_PROFILE") or site_value("compute.ssh_profile", ""),
            "CONDA": environment("CONDA") or site_value("conda_executable", "conda"),
        })
        # The selected composition may add shell exports without core knowing
        # any station-specific variable or configuration field.
        target = site_value('services.bootstrap', '')
        if target:
            module, separator, symbol = target.partition(':')
            if not separator:
                raise ValueError('services.bootstrap must be module:callable')
            extension = getattr(import_module(module), 'shell_exports', None)
            if extension is not None:
                values.update(extension())
        for key, value in values.items():
            print(f"export {key}={shlex.quote(str(value))}")
    else:
        parser.error("choose --get KEY or --shell")


if __name__ == "__main__":
    main()


def hardware_lock_path():
    """Use the station's shared inode, independent of service port/data root."""
    return default_path(home() / 'workstation/locks/station-hardware.lock',
                        workspace_root() / 'locks/station-hardware.lock',
                        directory=home() / 'workstation')
