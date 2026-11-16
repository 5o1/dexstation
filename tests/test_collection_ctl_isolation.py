"""The software-stop client works with only the standard library and defaults."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from helpers import ROOT
from stop_helpers import api_stub, stop_checkout


@pytest.mark.parametrize('plugin', ['raising', 'sleeping'])
@pytest.mark.parametrize('client_tree', ['full', 'isolated'])
def test_control_commands_do_not_discover_plugins(tmp_path, stop_checkout, plugin, client_tree):
    """Use system Python, real client HTTP, and hostile entry points in a child."""
    checkout = ROOT if client_tree == 'full' else stop_checkout
    module_name = 'r2_' + plugin + '_plugin'
    marker = tmp_path / 'plugin-imported'
    code = "raise ImportError('missing plugin dependency')" if plugin == 'raising' else 'import time; time.sleep(30)'
    (tmp_path / (module_name + '.py')).write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).touch()\n{code}\n')
    runner = tmp_path / 'probe.py'
    runner.write_text('''
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

checkout, plugin_name, url = sys.argv[1:]
sys.path.insert(0, str(Path(checkout) / 'src'))
discovery_calls = []
def entry_points(*, group):
    discovery_calls.append(group)
    return [importlib.metadata.EntryPoint(name='hostile', value=plugin_name + ':entry', group=group)]
importlib.metadata.entry_points = entry_points
spec = importlib.util.spec_from_file_location('collection_ctl_probe', Path(checkout) / 'scripts/collection_ctl.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
# All control parsers, including their help, must remain independent of plugins.
commands = [
    ['status'], ['session-stop'], ['record-start'], ['record-stop'],
    ['note', 'test'], ['episodes'], ['void', 'episode', '--reason', 'test'], ['export', 'episode'],
    ['snapshot'], ['snapshot-cancel'], ['snapshots'],
    ['snapshot-get', 'snapshot', '--output', 'unused'],
]
for args in commands:
    assert client.parser().parse_args(args).command == args[0]
for args in [[], *commands]:
    try:
        client.parser().parse_args([args[0], '--help'] if args else ['--help'])
    except SystemExit as exc:
        assert exc.code == 0
result = client.main(['session-stop', '--url', url, '--json'])
project_modules = sorted(name for name in sys.modules if name == 'dexstation' or name.startswith('dexstation.'))
assert result == 0, result
assert discovery_calls == [], discovery_calls
assert not any(name.startswith('r2_') and name.endswith('_plugin') for name in sys.modules)
assert project_modules == ['dexstation', 'dexstation.config', 'dexstation.config.defaults'], project_modules
print('PROBE_RESULT=' + json.dumps(dict(result=result, discovery_calls=discovery_calls, project_modules=project_modules)))
''')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with api_stub('ok') as (url, requests):
        begin = time.monotonic()
        result = subprocess.run(['/usr/bin/python3', '-S', str(runner), str(checkout), module_name, url],
            env=env, capture_output=True, text=True, timeout=3)
        elapsed = time.monotonic() - begin
    assert result.returncode == 0, (result.stdout, result.stderr)
    probe = json.loads(next(line.removeprefix('PROBE_RESULT=') for line in result.stdout.splitlines()
                            if line.startswith('PROBE_RESULT=')))
    assert probe['discovery_calls'] == []
    assert probe['result'] == 0 and requests == ['/api/v1/session/stop']
    assert not marker.exists()
    assert elapsed < 3


def test_stop_checkout_has_exact_minimal_dependency_files(stop_checkout):
    package = stop_checkout / 'src/dexstation'
    assert {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()} == {
        '__init__.py', 'config/__init__.py', 'config/defaults.py',
    }


def test_client_import_closure_is_standard_library_plus_registry_metadata():
    """Include lazy imports; factories/task implementations cannot enter this set."""
    package = ROOT / 'src/dexstation'
    allowed = {'dexstation', 'dexstation.config.defaults',
        'dexstation.inputs.registry', 'dexstation.backends.registry',
        'dexstation.runtime.plugins'}
    files = [(ROOT / 'scripts/collection_ctl.py', '')]
    files.extend((package / (name.removeprefix('dexstation.').replace('.', '/') + '.py'), 'dexstation')
                 for name in sorted(allowed - {'dexstation'}))
    files.append((package / '__init__.py', 'dexstation'))
    edges = set()
    for path, context in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = importlib.util.resolve_name('.' * node.level + (node.module or ''), context) if node.level else node.module
                names = [name]
            else:
                continue
            for name in names:
                assert name.split('.')[0] in sys.stdlib_module_names or name in allowed, (path, name)
                if name in allowed:
                    edges.add(name)
    assert edges == allowed - {'dexstation'}
