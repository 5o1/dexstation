"""Limit result types and protocol (§10.1); no FK or limit computation."""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass
class SpatialViolation:
    side: str
    link: str
    obstacle: str
    distance_m: float


@dataclass
class AngularViolation:
    side: str
    joint: str
    azimuth_deg: float
    elevation_deg: float
    reason: Literal["azimuth", "elevation_up"]


@dataclass
class LimitReport:
    ok: bool
    spatial: list[SpatialViolation]
    angular: list[AngularViolation]
    joint: dict[str, list[bool]]
    angular_range: dict[str, tuple[float, float]]


@runtime_checkable
class WorkspaceLimits(Protocol):
    def with_task_keep_out(self, keep_out: list) -> "WorkspaceLimits": ...
    def check(self, arm_q: dict, hand_q: dict) -> LimitReport: ...


class FakeLimits:
    """Always ok; ignores inputs/keep-outs and exposes no geometric ranges.

    Empty angular_range follows reviewer ruling 1; it is not a computed range.
    """

    def with_task_keep_out(self, keep_out: list) -> "FakeLimits":
        return self

    def check(self, arm_q: dict, hand_q: dict) -> LimitReport:
        return LimitReport(ok=True, spatial=[], angular=[], joint={}, angular_range={})
