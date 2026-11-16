"""R4c-1 rounding tolerance and must-fail structural/numerical controls."""
import json
import math
import numpy as np
import pytest
from helpers import assert_frozen_numeric


def test_float_roundoff_is_allowed_but_larger_differences_are_rejected():
    expected = .6492810601169936
    assert_frozen_numeric({'camera': [expected + 2e-13]}, {'camera': [expected]})
    with pytest.raises(AssertionError):
        assert_frozen_numeric(expected + 2e-12, expected)
    # Absolute tolerance alone cannot hide a relative error in small values.
    with pytest.raises(AssertionError):
        assert_frozen_numeric(1e-5 + 1e-13, 1e-5)


@pytest.mark.parametrize('value', [0., .6492810601169936, 1., 1000., 1920.])
def test_one_billionth_float_perturbation_must_fail(value):
    with pytest.raises(AssertionError):
        assert_frozen_numeric({'camera': [value + 1e-9]}, {'camera': [value]})


@pytest.mark.parametrize('actual,expected', [
    ({'a': 1.}, {'b': 1.}), ([1.], [1., 2.]), ((1.,), [1.]),
    (1, 1.), (True, 1), (2, 1), ('1.0', '1'),
    (np.zeros((1, 2)), np.zeros((2, 1))),
    (np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float64)),
])
def test_nonfloat_structure_and_values_must_fail(actual, expected):
    with pytest.raises(AssertionError):
        assert_frozen_numeric(actual, expected)


def test_scene_golden_comparison_rejects_one_billionth(tmp_path, monkeypatch):
    import helpers
    monkeypatch.setattr(helpers, 'GOLDEN_ROOT', tmp_path)
    (tmp_path/'scene.json').write_text(json.dumps({'pose': [.65], 'count': 1}))
    helpers.assert_golden('scene.json', {'pose': [math.nextafter(.65, math.inf)], 'count': 1}, numeric=True)
    with pytest.raises(AssertionError):
        helpers.assert_golden('scene.json', {'pose': [.65 + 1e-9], 'count': 1}, numeric=True)
    with pytest.raises(AssertionError):
        helpers.assert_golden('scene.json', {'pose': [.65], 'count': 1.}, numeric=True)
