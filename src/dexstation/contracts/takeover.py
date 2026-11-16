"""Arm-only takeover types (§6.3); the fake has no state machine.

Reviewer ruling 1: master_target_rad=None means no fresh input. Production
implementations return WAITING/release=False; freshness is owned by the caller.
Stop belongs to the session, which discards this object; it is not an ArmState.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class ArmState(str, Enum):
    WAITING = "waiting"
    MATCHING = "matching"
    FOLLOWING = "following"
    HOLD = "hold"


@dataclass
class ArmDecision:
    state: ArmState
    release: bool
    deviation_deg: list[float]
    reason: str | None


@runtime_checkable
class ArmTakeover(Protocol):
    def update(self, t: float, side: str, master_target_rad: list[float] | tuple[float, ...] | None, current_reference_rad, limit_ok: bool) -> ArmDecision: ...
    def snapshot(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


class FakeTakeover:
    """Always following/released, even if limit_ok is false; tests only.

    Empty deviations mean unmeasured, not measured zero. Snapshot has no state.
    """

    def update(self, t: float, side: str, master_target_rad: list[float] | tuple[float, ...] | None, current_reference_rad, limit_ok: bool) -> ArmDecision:
        return ArmDecision(ArmState.FOLLOWING, True, [], None)

    def snapshot(self) -> dict:
        return {}

    def restore(self, state: dict) -> None:
        pass
