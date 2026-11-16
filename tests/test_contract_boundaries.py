"""Keep contracts independent and inventory serialized format/interface IDs."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

from dexstation.contracts import versions

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/dexstation'
FORMAT_ID = re.compile(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)+-v[0-9]+')


def implementation_imports(package):
    violations = []
    for path in sorted(package.rglob('*.py')):
        parts = path.relative_to(package.parent.parent).with_suffix('').parts
        module = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
        context = module if path.name == '__init__.py' else module.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ''
                if node.level:
                    name = importlib.util.resolve_name('.' * node.level + name, context)
                names = [name, *(name + '.' + item.name for item in node.names)]
            else:
                continue
            for name in names:
                if name.startswith('dexstation') and not (
                        name == 'dexstation.contracts' or name.startswith('dexstation.contracts.')):
                    violations.append((str(path), node.lineno, name))
    return violations


def test_contracts_import_no_implementation_modules():
    assert implementation_imports(PACKAGE / 'contracts') == []
    # Also exercise lazy imports in ordinary module initialization in a fresh
    # interpreter, independent of whichever implementation tests ran earlier.
    probe = '''
import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob('*.py')):
    importlib.import_module('dexstation.contracts.' + path.stem)
assert not [name for name in sys.modules if name.startswith('dexstation.')
            and name != 'dexstation.contracts'
            and not name.startswith('dexstation.contracts.')]
'''
    subprocess.run([sys.executable, '-c', probe, str(PACKAGE / 'contracts')], check=True)


def test_contract_import_guard_includes_lazy_and_type_only_imports(tmp_path):
    package = tmp_path / 'dexstation/contracts'
    package.mkdir(parents=True)
    (package / 'sample.py').write_text('''
from .backend import CollectionBackend
if TYPE_CHECKING:
    from dexstation.model.spec import WorkcellSpec
def lazy():
    from dexstation.recording import session
''')
    assert {name for _, _, name in implementation_imports(package)} == {
        'dexstation.model.spec', 'dexstation.model.spec.WorkcellSpec',
        'dexstation.recording', 'dexstation.recording.session'}


def scalar_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from scalar_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from scalar_strings(child)


def format_literals(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.suffix == '.py':
            values = (node.value for node in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
                      if isinstance(node, ast.Constant) and isinstance(node.value, str))
        elif path.suffix == '.json':
            values = scalar_strings(json.loads(path.read_text(encoding='utf-8')))
        elif path.suffix in {'.mjs', '.js', '.html', '.yaml', '.yml'}:
            values = (match[1] for match in re.findall(
                r'''(["'])([a-z][a-z0-9-]*-v[0-9]+)\1''', path.read_text(encoding='utf-8')))
        else:
            continue
        for value in values:
            if FORMAT_ID.fullmatch(value):
                result.setdefault(value, set()).add(str(path.relative_to(root)))
    return result


def test_all_format_literals_are_in_the_version_table():
    inventory = {**format_literals(PACKAGE), **format_literals(ROOT / 'configs')}
    registered = set(versions.VERSIONS.values())
    assert not inventory.keys() - registered, {
        value: sorted(inventory[value]) for value in sorted(inventory.keys() - registered)}
    assert versions.API_VERSION == versions.VERSIONS['task_plugin_api'] == 1
    assert versions.API_PREFIX == versions.VERSIONS['http_api'] == '/api/v1'


def test_format_inventory_finds_an_unregistered_json_identifier(tmp_path):
    (tmp_path / 'new.json').write_text('{"format": "new-lab-recording-v9"}')
    assert format_literals(tmp_path) == {'new-lab-recording-v9': {'new.json'}}
    assert 'new-lab-recording-v9' not in versions.VERSIONS.values()
