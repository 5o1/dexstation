"""Run every maintained JavaScript test through the standard pytest command."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


NODE = shutil.which('node')
SCRIPTS = sorted(Path(__file__).parent.glob('*.mjs'))


@pytest.mark.skipif(NODE is None, reason='Node.js is not installed')
@pytest.mark.parametrize('script', SCRIPTS, ids=lambda path: path.name)
def test_node_script(script):
    env = dict(os.environ)
    # The optional review branch targets a separate live browser/service workflow.
    env.pop('TELEOP_REVIEW_URL', None)
    result = subprocess.run([NODE, str(script)], cwd=script.parent.parent,
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
