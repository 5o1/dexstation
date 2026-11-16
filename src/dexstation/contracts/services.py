"""Injected auxiliary services; implementations do not belong to the web server."""
from dataclasses import dataclass, field
from typing import Callable, Protocol


class InputConflict(RuntimeError):
    pass


class ForeignPublisherError(InputConflict):
    def __init__(self, names, detail=''):
        self.input = dict(role='linkerta', foreign_publishers=sorted(set(names)), fix='stop_foreign')
        super().__init__('Another LinkerTA publisher is running: '+', '.join(names)+(' '+detail if detail else ''))


class CameraConflict(RuntimeError):
    pass


class InputService(Protocol):
    retarget: str
    manual: bool
    def preflight(self): ...
    def launch(self, *, manual=False): ...
    def status(self) -> dict: ...
    def stop(self, *, include_manual=False): ...
    def switch_retarget(self, method: str): ...


class CanService(Protocol):
    """Non-mutating preflight. Privileged bring-up is an optional capability."""
    def check(self, role='linkerta') -> dict: ...
    def require(self, role='linkerta') -> dict: ...


class CanBringUpService(CanService, Protocol):
    def bring_up(self, role='linkerta', password=None) -> dict: ...


class CameraService(Protocol):
    def status(self, camera_name=None) -> dict: ...
    def start(self): ...
    def stop(self): ...
    def preview_jpeg(self, camera_name=None): ...
    def snapshot(self, label='', camera_name=None): ...
    def photos(self, camera_name=None): ...
    def photo(self, name, camera_name=None): ...
    def record_start(self): ...
    def record_stop(self): ...
    def record_state(self): ...


class MirrorSource(Protocol):
    def switch(self, mode: str): ...
    def real_status(self) -> dict: ...


class GeometryProvider(Protocol):
    def __call__(self, *, retarget: str): ...


def unchanged_source(source, context):
    return source


@dataclass
class Services:
    """One explicit composition, also passed to spawned controllers.

    Factories are ordinary callables, not plugin discovery groups. Source wrapping
    may add mode switching and status without replacing a registry entry.
    """
    input_chain_factory: Callable
    camera_factory: Callable
    glove_recorder_factory: Callable
    can_system: CanService
    geometry_provider: GeometryProvider
    wrap_source: Callable = unchanged_source
    retarget_methods: tuple[str, ...] = ('manus_anatomical',)
    retarget_labels: dict = field(default_factory=lambda: {'manus_anatomical': 'vector · anatomical'})
    signal_sources: tuple[dict, ...] = ({'value': 'teleop', 'label': 'Teleop'},)
    defaults: dict = field(default_factory=dict)
    estop_hint: str = ''
    can_bring_up: bool = False
    shell_exports: Callable | None = None
    data_root: Callable | None = None
    auxiliary_root: Callable | None = None
    web_presentation: dict = field(default_factory=dict)
    web_fragments: dict = field(default_factory=dict)
    camera_presentation: str = ''
    safety_check: Callable | None = None


class DisabledCamera:
    """No device access; unavailable differs explicitly from an idle camera."""
    def __init__(self, *args, **kwargs):
        pass

    def status(self, camera_name=None):
        return dict(available=False, mode='unavailable', error='Camera service not available',
                    cameras=[], selected_camera=None)

    def start(self):
        raise CameraConflict('Camera service not available')

    def stop(self):
        return self.status()

    def preview_jpeg(self, camera_name=None):
        raise CameraConflict('Camera service not available')

    def snapshot(self, label='', camera_name=None):
        raise CameraConflict('Camera service not available')

    def photos(self, camera_name=None):
        return []

    def photo(self, name, camera_name=None):
        raise FileNotFoundError(name)

    def record_start(self):
        raise CameraConflict('Camera service not available')

    def record_stop(self):
        return self.record_state()

    def record_state(self):
        return dict(available=False, cameras={})
