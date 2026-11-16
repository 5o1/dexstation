"""Environment compatibility shared by configuration and worker processes."""
import logging
import os


ENVIRONMENT_NAMES = {
    'HOME': ('DEXSTATION_HOME', 'DEXHAND_HOME'),
    'WORKSPACE': ('DEXSTATION_WORKSPACE', 'DEXHAND_WORKSPACE', 'GAMEPAD_VLA_WORKSPACE'),
    'DATA': ('DEXSTATION_DATA', 'DEXHAND_DATA', 'GAMEPAD_VLA_DATA'),
    'SITE_CONFIG': ('DEXSTATION_SITE_CONFIG', 'DEXHAND_SITE_CONFIG', 'GAMEPAD_VLA_SITE_CONFIG'),
    'SIM_PYTHON': ('DEXSTATION_SIM_PYTHON', 'DEXHAND_SIM_PYTHON', 'GAMEPAD_VLA_SIM_PYTHON'),
    'RENDER_DEVICE': ('DEXSTATION_RENDER_DEVICE', 'DEXHAND_RENDER_DEVICE', 'GAMEPAD_VLA_RENDER_DEVICE'),
    'REMOTE_CODE': ('DEXSTATION_REMOTE_CODE', 'DEXHAND_REMOTE_CODE', 'GAMEPAD_VLA_REMOTE_CODE'),
    'COMPUTE_PROFILE': ('DEXSTATION_COMPUTE_PROFILE', 'DEXHAND_COMPUTE_PROFILE', 'GAMEPAD_VLA_COMPUTE_PROFILE'),
    'CONDA': ('DEXSTATION_CONDA', 'DEXHAND_CONDA', 'GAMEPAD_VLA_CONDA'),
}
_warned = False


def environment(key):
    """Return the first nonempty value; warn once per process about aliases."""
    global _warned
    names = ENVIRONMENT_NAMES[key]
    for index, name in enumerate(names):
        value = os.environ.get(name)
        if value:
            if index and not _warned:
                logging.getLogger(__name__).warning('%s is deprecated; use %s', name, names[0])
                _warned = True
            return value
    return None


def environment_exports(values):
    """Publish effective values under preferred names and deprecated aliases."""
    return {name: str(value) for key, value in values.items() for name in ENVIRONMENT_NAMES[key]}


def publish_environment(key, value):
    os.environ.update(environment_exports({key: value}))
