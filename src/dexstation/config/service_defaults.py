"""Public defaults and explicit, site-selected service composition."""
from importlib import import_module
from functools import partial

from dexstation.contracts.services import DisabledCamera, Services


def public_services(*, can_system=None):
    from dexstation.inputs.supervisor import InputChain
    from dexstation.inputs.glove_overlay import anatomical_geometry
    from dexstation.inputs.glove_recording import WebGloveRecording
    from dexstation.inputs.linkerta_reader import CanSystem
    from dexstation.runtime.health import detect_safety
    can_system = can_system or CanSystem()
    return Services(input_chain_factory=partial(InputChain, can_system=can_system), camera_factory=DisabledCamera,
                    glove_recorder_factory=WebGloveRecording, can_system=can_system,
                    geometry_provider=anatomical_geometry, safety_check=detect_safety,
                    web_presentation={'retarget_labels': {'manus_anatomical': 'vector · anatomical'}})


def configured_services():
    """Load only the callable explicitly selected by site config; no discovery."""
    from dexstation.config.site import site_value
    target = site_value('services.bootstrap', '')
    if not target:
        return public_services()
    module, separator, symbol = target.partition(':')
    if not separator or not module or not symbol:
        raise ValueError('services.bootstrap must be module:callable')
    services = getattr(import_module(module), symbol)()
    if not isinstance(services, Services):
        raise TypeError('services.bootstrap must return Services')
    return services
