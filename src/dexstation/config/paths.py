"""Resolve workspace asset references without changing scene geometry."""
from pathlib import Path

from dexstation.config.site import workspace_root, repository_root


_PACKAGE = Path(__file__).resolve().parents[1]
_RESOURCE_PATHS = {
    'configs/appearance/': 'examples/ar5_g20/appearance/',
    'configs/workbenches/ar5_08_g20_bench_bimanual/': 'examples/ar5_g20/data/',
    'configs/render/': 'render/data/',
    'configs/inputs/': 'inputs/linkerta_assets/',
    'motion_assets/': 'examples/ar5_g20/motion_assets/',
}


def package_path(value):
    """Resolve unchanged built-in resource references inside the installed package."""
    value = str(value)
    if value.startswith('configs/tasks/'):
        task, _, relative = value.removeprefix('configs/tasks/').partition('/')
        return _PACKAGE / 'task_plugins' / task / 'data' / relative
    for prefix, destination in _RESOURCE_PATHS.items():
        if value == prefix.rstrip('/') or value.startswith(prefix):
            return _PACKAGE / destination / value[len(prefix):]
    return repository_root() / value


def resolve_workspace_path(value):
    """Leave ordinary and historical paths intact; resolve workspace:// paths."""
    if value.startswith("workspace://"):
        return str(workspace_root() / value.removeprefix("workspace://"))
    return value


def asset_path(value):
    path = Path(resolve_workspace_path(str(value)))
    if not path.is_absolute():
        path = package_path(str(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f'task asset must be a regular-file clone: {path}')
    return str(path)
