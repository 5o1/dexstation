from helpers import bench_data
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile

import pytest

from dexstation.config import site
from dexstation.runtime.provenance import code_provenance

from dexstation.config.site import configure_runtime, site_value, workspace_root
from dexstation.model.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_path_environment(monkeypatch):
    # configure_runtime writes both names; register every key for restoration.
    for suffix in ('HOME', 'WORKSPACE', 'DATA', 'SITE_CONFIG', 'SIM_PYTHON', 'RENDER_DEVICE'):
        monkeypatch.setenv('DEXSTATION_' + suffix, '')
        if suffix != 'HOME':
            monkeypatch.setenv('GAMEPAD_VLA_' + suffix, '')




def test_runtime_resolves_config_before_workers_change_directory(tmp_path, monkeypatch):
    (tmp_path / "site.json").write_text(json.dumps({"workspace": str(tmp_path / "workspace")}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GAMEPAD_VLA_SITE_CONFIG", "site.json")
    monkeypatch.delenv("GAMEPAD_VLA_WORKSPACE", raising=False)
    monkeypatch.delenv("XBOX_GAMEPAD_PHYSICS_RESOURCES", raising=False)
    configure_runtime()
    (tmp_path / "run").mkdir()
    monkeypatch.chdir(tmp_path / "run")
    assert site_value("workspace") == str(tmp_path / "workspace")
    assert workspace_root() == tmp_path / "workspace"


def test_simulation_resources_follow_site_and_explicit_override(tmp_path, monkeypatch):
    from dexstation.config.site import simulation_python
    config = tmp_path / "site.json"
    config.write_text(json.dumps({"simulation": {
        "python": str(tmp_path / "new environment/bin/python"),
        "resources": str(tmp_path / "new resources"),
    }}))
    monkeypatch.setenv("GAMEPAD_VLA_SITE_CONFIG", str(config))
    monkeypatch.delenv("XBOX_GAMEPAD_PHYSICS_RESOURCES", raising=False)
    monkeypatch.delenv("GAMEPAD_VLA_SIM_PYTHON", raising=False)
    configure_runtime()
    assert simulation_python() == tmp_path / "new environment/bin/python"
    assert os.environ['XBOX_GAMEPAD_PHYSICS_RESOURCES'] == str(tmp_path / "new resources")
    monkeypatch.setenv("GAMEPAD_VLA_SIM_PYTHON", str(tmp_path / "rollback/bin/python"))
    monkeypatch.setenv("XBOX_GAMEPAD_PHYSICS_RESOURCES", str(tmp_path / "rollback resources"))
    configure_runtime()
    assert simulation_python() == tmp_path / "rollback/bin/python"
    assert os.environ['XBOX_GAMEPAD_PHYSICS_RESOURCES'] == str(tmp_path / "rollback resources")




def test_public_check_finds_password_in_copied_chinese_document(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "copied.json").write_text(json.dumps({"excerpt": "管理员" + "密码" + "TEST_ONLY"}, ensure_ascii=False))
    find = runpy.run_path(str(ROOT / "scripts/check_public_tree.py"))["findings"]
    matches = list(find(tmp_path))
    assert any(category == "documented password" for _, _, category in matches)


def test_default_path_requires_a_directory(tmp_path):
    new, old = tmp_path / 'new', tmp_path / 'old'
    assert site.default_path(new, old) == old
    new.write_text('a file is not a layout directory')
    assert site.default_path(new, old) == old
    new.unlink(); new.mkdir()
    assert site.default_path(new, old) == new
    assert site.default_path(new / 'script.sh', old, directory=new) == new / 'script.sh'










def test_hardware_lock_path_shares_station_inode_across_layouts(tmp_path, monkeypatch):
    from dexstation.config.site import hardware_lock_path
    home=tmp_path/'dexhand'
    workspace=tmp_path/'legacy-workspace'
    config=tmp_path/'site.json';config.write_text('{}')
    monkeypatch.setenv('DEXSTATION_HOME',str(home))
    monkeypatch.setenv('DEXSTATION_SITE_CONFIG',str(config))
    monkeypatch.setenv('DEXSTATION_WORKSPACE',str(workspace))
    assert hardware_lock_path()==workspace/'locks/station-hardware.lock'
    (home/'workstation').mkdir(parents=True)
    assert hardware_lock_path()==home/'workstation/locks/station-hardware.lock'


def test_default_hardware_lock_path_follows_home_without_environment_overrides(tmp_path, monkeypatch):
    from dexstation.config.site import hardware_lock_path
    for name in tuple(os.environ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    path=hardware_lock_path()
    assert path.is_relative_to(tmp_path)
    assert path==tmp_path/'.local/share/dexstation/locks/station-hardware.lock'








@pytest.mark.parametrize(('override', 'key'), [
    ({'workspace': 4}, 'workspace'),
    ({'camera': {'width': True}}, 'camera.width'),
    ({'collection': {'duration': 'forever'}}, 'collection.duration'),
    ({'collection': {'idle_timeout_s': 'later'}}, 'collection.idle_timeout_s'),
    ({'udp': {'allow_sources': [3]}}, 'udp.allow_sources'),
    ({'station': {'can_roles': {'other_role': {'usb_path': 7}}}}, 'station.can_roles.other_role.usb_path'),
    ({'camera': {'serials': {'extra_camera': 123}}}, 'camera.serials.extra_camera'),
    ({'camera': {'usb_ports': {'extra_camera': 123}}}, 'camera.usb_ports.extra_camera'),
    ({'camera': {'usb_ports': {'extra_camera': '2-1.3:1.0'}}}, 'camera.usb_ports.extra_camera'),
])
def test_site_validation_names_wrong_type_key(tmp_path, monkeypatch, override, key):
    path = tmp_path / 'invalid-site.json'
    path.write_text(json.dumps(override))
    monkeypatch.setenv('DEXSTATION_SITE_CONFIG', str(path))
    with pytest.raises(ValueError, match=key):
        site.reload_site()


def test_site_unknown_keys_warn_once_and_cache_requires_explicit_reload(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'cached-site.json'
    path.write_text(json.dumps({'workspace': 'first', 'new_key': True,
                                'collection': {'another_new_key': 'retained'}}))
    monkeypatch.setenv('DEXSTATION_SITE_CONFIG', str(path))
    with caplog.at_level('WARNING', logger=site.__name__):
        values = site.reload_site()
        assert values['new_key'] is True
        assert site.load_site() == values
        warnings = [record.message for record in caplog.records
                    if record.name == site.__name__ and 'Unknown site configuration keys' in record.message]
        assert len(warnings) == 1
        assert 'new_key' in warnings[0] and 'collection.another_new_key' in warnings[0]
    values['workspace'] = 'caller-mutated'
    path.write_text('{"workspace":"second"}')
    assert site.site_value('workspace') == 'first'
    assert site.reload_site()['workspace'] == 'second'




@pytest.mark.parametrize('key', [
    'snapshots.delay_s', 'collection.idle_timeout_s',
])
def test_site_accepts_fractional_timing_and_reader_rates(tmp_path, monkeypatch, key):
    section, field = key.split('.')
    path = tmp_path / 'fractional-site.json'
    path.write_text(json.dumps({section: {field: 1.5}}))
    monkeypatch.setenv('DEXSTATION_SITE_CONFIG', str(path))
    assert site.reload_site()[section][field] == 1.5
