"""Static import-layer ratchet, including lazy and type-checking imports."""
import ast
import difflib
import importlib.util
import json

from helpers import ROOT, GOLDEN_ROOT, assert_golden, updating_golden


RANK = {name: rank for rank, names in enumerate((
    ('contracts',), ('config',), ('model',), ('profiles',), ('control',),
    ('backends', 'inputs', 'recording', 'render', 'cameras'),
    ('runtime',), ('tasks',), ('server',), ('cli',))) for name in names}
POLICY = GOLDEN_ROOT / 'import_layers.json'


def import_graph(package):
    """Resolve import statements against on-disk modules without importing them."""
    modules = {}
    for path in sorted(package.rglob('*.py')):
        parts = list(path.relative_to(package.parent).with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        modules['.'.join(parts)] = path
    edges = set()
    for source, path in modules.items():
        context = source if path.name == '__init__.py' else source.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'), filename=str(path))):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    base = importlib.util.resolve_name('.' * node.level + base, context)
                targets = [base, *(base + '.' + alias.name for alias in node.names)]
            else:
                continue
            edges.update((source, target) for target in targets if target in modules and target != source)
    return modules, edges


def policy_and_reverse_edges():
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    layers = policy['module_layers']
    modules, edges = import_graph(ROOT / 'src/dexstation')
    missing = sorted(modules.keys() - layers.keys())
    assert not missing, f'Modules need an explicit layer in {POLICY}: {missing}'
    removed = sorted(layers.keys() - modules.keys())
    assert not removed, f'Delete removed modules from {POLICY}: {removed}'
    invalid = {module: layer for module, layer in layers.items() if layer not in RANK}
    assert not invalid, f'Unknown layer names: {invalid}'
    reverse = {edge for edge in edges if RANK[layers[edge[0]]] < RANK[layers[edge[1]]]}
    return policy, reverse


def edge_diff(allowed, actual):
    return ''.join(difflib.unified_diff(
        [f'{source} -> {target}\n' for source, target in sorted(allowed)],
        [f'{source} -> {target}\n' for source, target in sorted(actual)],
        fromfile='allowlist', tofile='actual reverse imports'))


def test_import_layers():
    policy, reverse = policy_and_reverse_edges()
    allowed = {tuple(edge) for edge in policy['reverse_allowlist']}
    assert len(allowed) == len(policy['reverse_allowlist']), 'Delete duplicate allowlist entries'
    assert not reverse - allowed, 'New reverse imports are forbidden, even in update mode:\n' + edge_diff(allowed, reverse)
    if updating_golden():
        policy['reverse_allowlist'] = [list(edge) for edge in sorted(reverse)]
        assert_golden('import_layers.json', policy)


def test_import_allowlist_only_shrinks():
    policy, reverse = policy_and_reverse_edges()
    allowed = {tuple(edge) for edge in policy['reverse_allowlist']}
    # Update mode can remove stale entries, but test_import_layers prevents additions.
    if updating_golden():
        return
    assert not allowed - reverse, 'Delete these obsolete reverse imports from the allowlist:\n' + edge_diff(allowed, reverse)


def test_import_graph_resolves_nested_and_relative_imports(tmp_path):
    package = tmp_path / 'sample'
    (package / 'nested').mkdir(parents=True)
    for name in ('__init__.py', 'a.py', 'b.py', 'nested/__init__.py', 'nested/c.py'):
        (package / name).touch()
    (package / 'a.py').write_text('''
from . import b
from .b import VALUE
import sample.nested.c as c
if TYPE_CHECKING:
    from .nested.c import SomeType
def lazy():
    from .nested import c
''', encoding='utf-8')
    (package / 'nested/c.py').write_text('from ..a import VALUE\n', encoding='utf-8')
    modules, edges = import_graph(package)
    assert set(modules) == {'sample', 'sample.a', 'sample.b', 'sample.nested', 'sample.nested.c'}
    assert edges == {('sample.a', target) for target in (
        'sample', 'sample.b', 'sample.nested', 'sample.nested.c')} | {('sample.nested.c', 'sample.a')}
