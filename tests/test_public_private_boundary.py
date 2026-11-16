"""Fresh-process acceptance for the standalone public runtime."""
import json
from pathlib import Path
from helpers import ROOT


def _probe(script, directory, *args):
    import os
    import subprocess
    import sys
    environment = {key:value for key,value in os.environ.items()
                   if not key.startswith(('DEXSTATION_', 'DEXHAND_', 'GAMEPAD_VLA_'))}
    environment['DEXSTATION_TEST_PUBLIC_ROOT'] = str(ROOT)
    environment['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + environment.get('PYTHONPATH', '')
    return subprocess.run([sys.executable, str(Path(__file__).resolve().parent/script), str(directory), *args],
                          cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)


def test_public_app_and_sessions_in_fresh_process_without_station(tmp_path):
    result = _probe('public_boundary_probe.py', tmp_path/'public')
    assert result.returncode == 0, result.stdout + result.stderr
    import json
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence['private_import_attempts'] == []
    assert evidence['tasks'] == ['empty', 'tabletop_cube_grasp']
    assert evidence['replay_steps'] == 2
