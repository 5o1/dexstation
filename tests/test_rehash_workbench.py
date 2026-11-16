"""Manifest maintenance detects byte drift and preserves maintained profile text."""
import copy
import json
from pathlib import Path
import re
import runpy
import shutil

from dexstation.profiles.compose import load_workbench

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'src/dexstation/examples/ar5_g20/data'
rehash = runpy.run_path(str(ROOT / 'scripts/rehash_workbench.py'))['main']


def test_check_committed_bench_and_detect_takeover_drift(tmp_path, capsys):
    assert rehash([str(BENCH), '--check']) == 0
    directory = tmp_path / 'bench'
    shutil.copytree(BENCH, directory)
    path = directory / 'workbench.json'
    original = path.read_bytes()
    with (directory / 'takeover.json').open('a') as stream:
        stream.write('\n')
    assert rehash([str(directory), '--check']) == 1
    assert 'manifest.sha256' in capsys.readouterr().out
    assert path.read_bytes() == original


def test_repair_only_hash_fields_and_keep_formatting(tmp_path, capsys):
    directory = tmp_path / 'bench'
    shutil.copytree(BENCH, directory)
    path = directory / 'workbench.json'
    original = path.read_text().replace('\n', '\r\n')
    path.write_bytes(original.encode())
    before = json.loads(original)
    with (directory / 'sources/baseline.json').open('a') as stream:
        stream.write(' ')
    assert rehash([str(directory), '--check']) == 1
    assert 'manifest.components.baseline.sha256' in capsys.readouterr().out
    assert rehash([str(directory)]) == 0
    after_text = path.read_bytes().decode()
    after = json.loads(after_text)
    expected = copy.deepcopy(before)
    expected['manifest']['sha256'] = after['manifest']['sha256']
    expected['manifest']['components']['baseline']['sha256'] = after['manifest']['components']['baseline']['sha256']
    assert after == expected
    hide_hashes = lambda text: re.sub(r'("sha256"\s*:\s*")[0-9a-f]{64}', r'\1HASH', text)
    assert hide_hashes(after_text) == hide_hashes(original)
    assert rehash([str(directory), '--check']) == 0
    load_workbench(directory)
