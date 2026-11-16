"""The bench calibration has one maintained location, including untracked files."""
import os
from pathlib import Path

from helpers import ROOT


def test_single_bench_calibration_copy():
    expected = {f'src/dexstation/examples/ar5_g20/data/calibration/{name}.json'
                for name in ('teleop', 'real_robot')}
    actual = set()
    for directory, children, files in os.walk(ROOT):
        if '.git' in children:
            children.remove('.git')
        for name in files:
            if name.endswith('.json') and name.startswith(('teleop', 'real_robot')):
                actual.add((Path(directory) / name).relative_to(ROOT).as_posix())
    assert actual == expected, f'Calibration files must have a single bench copy: {sorted(actual ^ expected)}'
