from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexstation.backends.sapien_render import PHYSICS_DT, SapienBackend, validate_scene_injection
from dexstation.model.spec import dump_spec, load_spec










@pytest.mark.parametrize("scene,gamepad", [(object(), None), (None, object()),
    (SimpleNamespace(get_timestep=lambda: 1 / 60), object()),
    (SimpleNamespace(get_timestep=lambda: 1 / 239), object())])
def test_constructor_rejects_bad_injection_without_sapien(scene, gamepad):
    with pytest.raises(ValueError, match="together|timestep"):
        SapienBackend(None, scene=scene, gamepad=gamepad)


@pytest.mark.parametrize("timestep", [PHYSICS_DT, PHYSICS_DT + 0.5e-12,
                                      PHYSICS_DT - 0.5e-12, float(np.float32(PHYSICS_DT))])
def test_injection_accepts_matching_timestep(timestep):
    validate_scene_injection(SimpleNamespace(get_timestep=lambda: timestep), object())
    validate_scene_injection(None, None)
