"""Frozen f07aa4c numerical results; no compiler, old source, or device required."""
import dataclasses
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexstation.inputs import arm_mapping
from dexstation.examples.ar5_g20 import mapping as g20_mapping
from dexstation.inputs import retarget_motor_clone
from dexstation.model import station_math

MODULES = dict(arm=arm_mapping, g20=g20_mapping, slots=retarget_motor_clone, math=station_math)
FIXTURES = Path(__file__).parent / 'fixtures/rewrite_equivalence'


def encode(value):
    if dataclasses.is_dataclass(value):
        return {'dataclass': type(value).__name__, 'value': encode(dataclasses.asdict(value))}
    if isinstance(value, np.ndarray):
        return {'array': value.tolist(), 'dtype': value.dtype.str, 'shape': list(value.shape)}
    if isinstance(value, np.generic):
        return {'scalar': value.item(), 'dtype': value.dtype.str}
    if isinstance(value, tuple):
        return {'tuple': [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return {'type': type(value).__name__, 'value': value}


def evaluate(case, modules=MODULES):
    module = modules[case['module']]
    args = case.get('args', [])
    kwargs = case.get('kwargs', {})
    try:
        if case['module'] == 'math':
            result = getattr(module.StationMath(), case['function'])(*args, **kwargs)
        elif case['module'] == 'arm':
            kwargs = dict(kwargs)
            if isinstance(kwargs.get('station_math'), bool):
                kwargs['station_math'] = modules['math'].StationMath() if kwargs['station_math'] else None
            mapper = module.ArmMapper(**kwargs)
            result = [mapper.command(*command) for command in args]
            if case.get('metadata'):
                result = [result, mapper.metadata()]
        elif case['function'].startswith('trans_to_motor_'):
            result = getattr(module, case['function'])(SimpleNamespace(**kwargs), *args)
        else:
            result = getattr(module, case['function'])(*args, **kwargs)
        return encode(result)
    except Exception as error:
        return {'exception': type(error).__name__, 'message': str(error)}


def assert_exact(actual, expected, path='output'):
    """Check containers, scalar types, dtypes and shapes as well as exact values."""
    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        if 'array' in expected:
            assert actual['dtype'] == expected['dtype'], path
            assert actual['shape'] == expected['shape'], path
            a = np.asarray(actual['array'], dtype=actual['dtype'])
            b = np.asarray(expected['array'], dtype=expected['dtype'])
            assert np.array_equal(a, b, equal_nan=True), path
            # Also catch signed-zero and NaN-payload changes in ndarray outputs.
            assert a.tobytes() == b.tobytes(), path
        else:
            for key in expected:
                assert_exact(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_exact(a, b, f'{path}[{index}]')
    elif isinstance(expected, float):
        assert np.array_equal(np.asarray(actual), np.asarray(expected), equal_nan=True), path
        assert np.float64(actual).tobytes() == np.float64(expected).tobytes(), path
    else:
        assert actual == expected, path


def records():
    with gzip.open(FIXTURES / 'f07aa4c.json.gz', 'rt') as stream:
        return json.load(stream)['records']


@pytest.mark.parametrize('group', ['g20', 'slots', 'arm', 'math'])
def test_frozen_rewrite_equivalence(group):
    for record in records():
        if record['case']['module'] == group:
            assert_exact(evaluate(record['case']), record['expected'], record['case']['id'])
