"""Shared fixtures isolate tests from workstation environment settings."""
import os

import pytest


@pytest.fixture(autouse=True)
def isolated_workstation_environment(monkeypatch):
    """Each test supplies its own workstation settings when it needs them."""
    for name in tuple(os.environ):
        if name.startswith(('DEXSTATION_', 'DEXHAND_', 'GAMEPAD_VLA_')) or name == 'CUDA_VISIBLE_DEVICES':
            monkeypatch.delenv(name)


@pytest.fixture
def isolated_station_paths(tmp_path, monkeypatch):
    monkeypatch.setenv('DEXSTATION_HOME', str(tmp_path/'dexhand'))
    monkeypatch.setenv('DEXSTATION_WORKSPACE', str(tmp_path/'workspace'))



@pytest.fixture(autouse=True)
def valid_bench_for_runtime_loads(monkeypatch, request):
    """Runtime API tests validate a rehashed copy; one dedicated test checks Git."""
    if request.module.__name__ == 'test_rehash_workbench':
        return
    from pathlib import Path
    from dexstation.profiles import compose as profiles
    from helpers import ROOT, bench_directory
    original = profiles.load_workbench
    committed = ROOT / 'src/dexstation/examples/ar5_g20/data'

    def load(path):
        candidate = Path(path).absolute()
        if candidate in (committed, committed / 'workbench.json'):
            path = bench_directory()
        return original(path)

    monkeypatch.setattr(profiles, 'load_workbench', load)
