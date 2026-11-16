"""Python action and browser module interfaces (§8.2–8.3, §15.2).

Browser protocols describe the JavaScript surface; no mount implementation lives
here. unmount is optional. Cell payloads are status.task.cell_state[cell_id].
"""

from typing import Callable, Protocol, TypedDict, runtime_checkable
from collections.abc import Awaitable

import numpy as np

from dexstation.contracts.backend import StepObservation

from dexstation.contracts.versions import API_VERSION
BUILTIN_CELLS = ("values", "lamps", "banner", "controls")
BUILTIN_ACTIONS = ("resample_objects", "restore_initial")


@runtime_checkable
class ActionContext(Protocol):
    def set_object_state(self, name: str, pose, twist=None) -> None: ...
    def state(self) -> StepObservation: ...
    def rng(self) -> np.random.Generator: ...


class TaskAction(Protocol):
    def __call__(self, ctx: ActionContext, params: dict) -> object: ...


class ActionReply(TypedDict):
    accepted: bool


class ActionSuccess(ActionReply):
    result: object


class ActionFailure(ActionReply):
    error: str


class CellAPI(Protocol):
    def subscribe(self, fn: Callable[[dict], None]) -> None: ...
    def action(self, name: str, params: dict) -> Awaitable[ActionSuccess | ActionFailure]: ...
    def banner(self, text: str | None) -> None: ...
    def speak(self, text: str) -> None: ...
    def status(self) -> dict: ...


class CellModule(Protocol):
    def mount(self, root: object, api: CellAPI) -> None: ...


class UnmountableCellModule(CellModule, Protocol):
    def unmount(self) -> None: ...
