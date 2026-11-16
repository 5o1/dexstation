"""Exact baseline replay with fake CAN, sysfs-independent clocks and graph stubs."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from dexstation.inputs import linkerta_reader


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'linkerta_equivalence', ROOT / 'scripts/research/linkerta_equivalence.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
with gzip.open(ROOT / 'tests/fixtures/linkerta_equivalence/4dbb537.json.gz', 'rt') as stream:
    FIXTURE = json.load(stream)


@pytest.mark.parametrize('record', FIXTURE['records'], ids=lambda r: r['case']['name'])
def test_baseline_exact(record):
    assert replay.execute(linkerta_reader, record['case']) == record['expected']
