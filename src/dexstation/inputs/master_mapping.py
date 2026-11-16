"""LinkerTA -> AR5 pose mapping, independent of ArmMapper's command path.

All FK is in the arm base frame at ``arm.tip_link``. Angles at the public
boundary are degrees; solver angles are radians and URDF velocities use rad/s.
The caller owns the last reachable output and passes it back on every cycle.
This module neither operates hardware nor applies the WP5 safety/takeover layer.
"""
from dataclasses import dataclass
import math
import heapq
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np

from dexstation.inputs.arm_mapping import OFFSETS
from dexstation.model.kinematics import SerialChain


@dataclass(frozen=True)
class MappingParams:
    s_p: float = 0.01
    s_R: float = 0.05
    s_e: float = 0.05
    delta_position_m: float = 0.0005
    delta_orientation_rad: float = math.radians(0.1)
    epsilon_position_m: float = 0.002
    epsilon_orientation_rad: float = math.radians(1.)
    elbow_band_m: float = 0.002
    elbow_certificate_m: float = 0.001
    sweep_limit_m: float = 0.02
    arm_speed_deg_s: float = 40.
    sweep_step_s: float = 0.01
    sweep_coarse_intervals: int = 8
    sweep_refinement_factor: int = 10
    sweep_refined_sample_cap: int = 16380
    branch_joint_threshold_deg: float = 90.
    margin_deg: float = 3.
    max_iterations: int = 24
    max_elbow_iterations: int = 8
    damping: float = 0.001
    max_step_rad: float = 0.35
    step_tolerance_rad: float = 1e-9
    elbow_tie_m: float = 1e-10
    line_search_steps: int = 6
    sweep_cache_capacity: int = 128


@dataclass
class MappingResult:
    q_deg: tuple[float, ...] | None
    reachable: bool
    hold: bool
    intended_pose: np.ndarray
    achieved_pose: np.ndarray | None
    errors: dict[str, float | None]
    cost_terms: dict[str, float | None]
    active_limits: tuple[str, ...]
    seed_used: str
    iterations: int
    solve_ms: float
    reason: str
    candidate_q_deg: tuple[float, ...]
    candidate_pose: np.ndarray
    candidate_errors: dict[str, float]
    candidate_cost_terms: dict[str, float]
    certificate: str
    seeds_count: int
    best_hand_errors: dict[str, float]
    sweep_m: float | None
    candidate_sweep_m: float
    sweep_method: str
    duration_s: float
    branch_switch: bool
    hold_reason: str | None
    q_now_deg: tuple[float, ...]


def _hat(v):
    v = np.asarray(v)
    out = np.zeros((*v.shape[:-1], 3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def _cross(a, b):
    return np.stack((a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]), axis=-1)


def rotation_log(rotation):
    """Principal SO(3) logarithm, including rotations arbitrarily close to pi."""
    r = np.asarray(rotation, float)
    vee = np.stack((r[..., 2, 1] - r[..., 1, 2],
                    r[..., 0, 2] - r[..., 2, 0],
                    r[..., 1, 0] - r[..., 0, 1]), axis=-1) / 2
    sine = np.linalg.norm(vee, axis=-1)
    cosine = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1., 1.)
    theta = np.arctan2(sine, cosine)
    out = vee * np.divide(theta, sine, out=np.ones_like(theta), where=sine > 1e-12)[..., None]
    flat_r, flat_o = r.reshape(-1, 3, 3), out.reshape(-1, 3)
    for i in np.flatnonzero((cosine < -0.999999).reshape(-1)):
        _, vectors = np.linalg.eigh((flat_r[i] + flat_r[i].T) / 2)
        axis = vectors[:, -1]
        if axis @ vee.reshape(-1, 3)[i] < 0:
            axis = -axis
        flat_o[i] = theta.reshape(-1)[i] * axis
    return out


class MasterPoseMapper:
    """Arm-only IK: pose/sweep feasibility, hand band, elbow band, request, time.

    ``errors`` and ``cost_terms`` describe the actual output (None on a failed
    first cycle). ``candidate_*`` and ``active_limits`` always describe the
    chosen solver candidate, so hold diagnostics do not hide the failed solve.
    ``full_search=True`` bypasses certificates for offline comparison. Exact
    certificates preserve the full-search solution; warm certificates are the
    explicitly bounded approximation from reviewer ruling 6. Costs are returned,
    never printed. Execution time is seconds; squared joint costs are seconds².
    """

    def __init__(self, scene, side, params=None):
        if side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        self.params = params or MappingParams()
        p = self.params
        positive = (p.s_p, p.s_R, p.s_e, p.damping, p.max_step_rad, p.step_tolerance_rad,
                    p.arm_speed_deg_s, p.sweep_step_s)
        nonnegative = (p.delta_position_m, p.delta_orientation_rad, p.epsilon_position_m,
                       p.epsilon_orientation_rad, p.margin_deg, p.elbow_tie_m,
                       p.elbow_band_m, p.elbow_certificate_m, p.sweep_limit_m)
        if (not all(np.isfinite(v) and v > 0 for v in positive)
                or not all(np.isfinite(v) and v >= 0 for v in nonnegative)
                or p.max_iterations < 0 or p.max_elbow_iterations < 0 or p.line_search_steps < 1
                or p.sweep_coarse_intervals < 1 or p.sweep_refinement_factor < 2
                or not isinstance(p.sweep_refined_sample_cap, int) or p.sweep_refined_sample_cap < 0
                or p.sweep_cache_capacity < 0):
            raise ValueError('invalid mapping parameters')
        self.side = side
        arm = getattr(scene.spec, side + '_arm')
        self.chain = SerialChain(arm.urdf, arm.tip_link)
        if self.chain.root_link != arm.base_link or len(self.chain.active) != 7:
            raise ValueError('expected a seven-joint chain rooted at the arm base')
        self.joints = self.chain.joints
        if any(j['type'] not in ('fixed', 'revolute') or j['mimic'] for j in self.joints):
            raise ValueError('AR5 mapping requires independent revolute joints')
        lower, upper = self.chain.active_limits
        self.lower, self.upper = lower + math.radians(p.margin_deg), upper - math.radians(p.margin_deg)
        if np.any(self.lower >= self.upper):
            raise ValueError('margin leaves an empty joint range')
        by_name = {j.get('name'): j for j in ET.parse(arm.urdf).getroot().findall('joint')}
        self.velocity_rad_s = np.array([float(by_name[n].find('limit').get('velocity'))
                                        for n in self.chain.active])
        if not np.all(np.isfinite(self.velocity_rad_s) & (self.velocity_rad_s > 0)):
            raise ValueError('URDF must provide positive velocity limits')
        self.offset_deg = np.array(OFFSETS[side]) + np.asarray(scene.calibration['teleop'][side])
        self._identity7 = np.eye(7)
        self.arm_speed_rad_s = math.radians(p.arm_speed_deg_s)
        lever, acceleration = np.zeros(7), 0.
        point_levers, point_acceleration = [], []
        depth = 0
        for joint in self.joints:
            length = np.linalg.norm(joint['origin'][:3, 3])
            lever[:depth] += length
            # Even with constant joint rates, upstream motion rotates each
            # downstream axis: |Omega_dot| <= omega^2*d*(d-1)/2. Include this
            # angular-acceleration term as well as |Omega|^2 <= (d*omega)^2.
            acceleration += length * self.arm_speed_rad_s**2 * (depth**2 + depth*(depth-1)/2)
            if joint['type'] != 'fixed':
                if depth in (3, 5):
                    point_levers.append(lever.copy())
                    point_acceleration.append(acceleration)
                depth += 1
        self._point_levers = np.array([*point_levers, lever])
        self._point_acceleration_bounds_m_s2 = np.array([*point_acceleration, acceleration])
        self.sweep_curvature_bound_m_s2 = float(np.max(self._point_acceleration_bounds_m_s2))
        constant, origins, axes = np.eye(4), [], []
        for joint in self.joints:
            constant = constant @ joint['origin']
            if joint['type'] != 'fixed':
                origins.append(constant.copy())
                axes.append(joint['axis'] / np.linalg.norm(joint['axis']))
                constant = np.eye(4)
        origins = np.array(origins)
        self._kin_R, self._kin_p = origins[:, :3, :3], origins[:, :3, 3]
        self._kin_axis = np.array(axes)
        self._kin_skew = _hat(self._kin_axis)
        self._kin_skew2 = self._kin_skew @ self._kin_skew
        self._kin_tail_R, self._kin_tail_p = constant[:3, :3], constant[:3, 3]
        self._identity3, self._identity4 = np.eye(3), np.eye(4)
        self._identity_origins = np.array_equal(self._kin_R, np.broadcast_to(self._identity3, (7, 3, 3)))
        self._zero_p5 = not np.any(self._kin_p[5])
        self._zero_p6 = not np.any(self._kin_p[6])
        self._line_alphas = .5 ** np.arange(p.line_search_steps)
        self._elbow_alphas = np.r_[0., self._line_alphas]
        self._refinement_fractions = np.linspace(0., 1., p.sweep_refinement_factor+1)
        # This is certificate provenance only. q_now is always an explicit input.
        self._last_copy_deg = None
        self._sweep_memo = {}

    @staticmethod
    def _vector(value):
        q = np.asarray(value, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('expected seven finite angles in degrees')
        return q

    def _kinematics(self, q_rad):
        """All seven links and all configurations, with three prefix-scan stages.

        Only NumPy 1.21 APIs are used. Each stage reads its entire right-hand
        side before assignment; the matrix product is associative. There is no
        per-configuration or per-joint Python loop in the evaluation kernel.
        """
        q = np.atleast_2d(q_rad)
        rotation = (self._identity3 + np.sin(q)[..., None, None] * self._kin_skew
                    + (1 - np.cos(q))[..., None, None] * self._kin_skew2)
        if not self._identity_origins:
            rotation = self._kin_R @ rotation
        position = np.broadcast_to(self._kin_p, (len(q), 7, 3)).copy()
        for stride in (1, 2, 4):
            position[:, stride:] = (position[:, :-stride]
                + (rotation[:, :-stride] @ position[:, stride:, :, None])[..., 0])
            rotation[:, stride:] = rotation[:, :-stride] @ rotation[:, stride:]
        palm = position[:, -1] + rotation[:, -1] @ self._kin_tail_p
        return rotation, position, palm

    def frames(self, q_rad, *, jacobian=False, points=False):
        """Batched FK and geometric Jacobians; q_rad is (N,7)."""
        rotation, position, palm = self._kinematics(q_rad)
        pose = np.broadcast_to(self._identity4, (len(position), 4, 4)).copy()
        pose[:, :3, :3] = rotation[:, -1] @ self._kin_tail_R
        pose[:, :3, 3] = palm
        elbow, wrist = position[:, 3], position[:, 5]
        if not jacobian:
            return (pose, elbow, wrist) if points else (pose, elbow)
        axes = (rotation @ self._kin_axis[None, :, :, None])[..., 0]
        jp = _cross(axes, palm[:, None] - position).transpose(0, 2, 1)
        je = _cross(axes, elbow[:, None] - position).transpose(0, 2, 1)
        je[:, :, 3:] = 0.
        result = (pose, elbow, jp, axes.transpose(0, 2, 1), je)
        return (*result, wrist) if points else result

    def executed_path(self, q_now_rad, target_rad, times_s):
        """40 deg/s per-joint motion; shorter joint moves finish first."""
        delta = np.asarray(target_rad) - q_now_rad
        travel = self.arm_speed_rad_s * np.asarray(times_s)[:, None]
        return q_now_rad + np.sign(delta) * np.minimum(abs(delta), travel)

    def _sweep_points(self, q):
        # Point-only seven-link product: no all-link prefix arrays or unused
        # endpoint rotation. Constants still come from the parsed URDF.
        q = np.atleast_2d(q)
        local = self._identity3 + np.sin(q)[..., None, None] * self._kin_skew + (1 - np.cos(q))[..., None, None] * self._kin_skew2
        if not self._identity_origins:
            local = self._kin_R @ local
        r0 = local[:, 0]
        r1 = r0 @ local[:, 1]
        r2 = r1 @ local[:, 2]
        r3 = r2 @ local[:, 3]
        r4 = r3 @ local[:, 4]
        r5 = r4 @ local[:, 5]
        r6 = r5 @ local[:, 6]
        p1 = self._kin_p[0] + r0 @ self._kin_p[1]
        p2 = p1 + r1 @ self._kin_p[2]
        elbow = p2 + r2 @ self._kin_p[3]
        p4 = elbow + r3 @ self._kin_p[4]
        wrist = p4 if self._zero_p5 else p4 + r4 @ self._kin_p[5]
        p6 = wrist if self._zero_p6 else wrist + r5 @ self._kin_p[6]
        palm = p6 + r6 @ self._kin_tail_p
        return np.stack((elbow, wrist, palm), axis=1)


    @staticmethod
    def _point_deviations(points, start, end):
        segment = end - start
        length2 = np.sum(segment * segment, axis=1)
        fraction = np.divide(np.sum((points - start) * segment, axis=2), length2,
                             out=np.zeros(points.shape[:2]), where=length2 > 0)
        nearest = start + np.clip(fraction, 0., 1.)[..., None] * segment
        return np.max(np.linalg.norm(points - nearest, axis=2), axis=1)

    @classmethod
    def _point_sweep(cls, points):
        return float(np.max(cls._point_deviations(points, points[0], points[-1])))

    def _interval_error(self, gap_s):
        """Point-chord error between arrivals, using the full acceleration bound.

        B includes both rotating-axis angular acceleration and centripetal
        acceleration, summed over ancestor link vectors for each tracked point.
        """
        return self.sweep_curvature_bound_m_s2 * np.asarray(gap_s) ** 2 / 8

    @staticmethod
    def _unique_times(values):
        """Sorted unique finite float time coordinates, without masked-array dispatch.

        NumPy 2.5's general unique path lazily imports numpy.ma. These grids
        contain only finite float64 values, so sorting and adjacent comparison
        give exactly the same coordinates without that first-use import.
        """
        values = np.sort(values)
        if not len(values):
            return values
        keep = np.empty(len(values), dtype=bool)
        keep[0] = True
        keep[1:] = values[1:] != values[:-1]
        return values[keep]

    def _certify_intervals(self, q_now, target, times, points):
        """Refine only uncertified intervals, bounded by an integer sample cap.

        Every deviation uses the original whole-path endpoint segments. Joint
        arrival times are present in the base grid, so the full point-
        acceleration bound is valid on each interval and all descendants.
        Heap order is largest upper
        bound first, then increasing start/end time; no clock controls work.
        """
        start, end = points[0], points[-1]
        deviations = self._point_deviations(points, start, end)
        lower = float(np.max(deviations))
        error = self._interval_error(np.diff(times))
        initial_error = float(np.max(error)) if len(error) else 0.
        upper = np.maximum(deviations[:-1], deviations[1:]) + error
        limit = self.params.sweep_limit_m
        certified = float(np.max(upper[upper <= limit])) if np.any(upper <= limit) else lower
        heap = [(-float(upper[i]), float(times[i]), float(times[i+1]),
                 float(deviations[i]), float(deviations[i+1]))
                for i in np.flatnonzero(upper > limit)]
        heapq.heapify(heap)
        added = 0
        method = 'local_upper_bound'
        while heap and lower <= limit:
            needed = self.params.sweep_refinement_factor - 1
            count = min(len(heap), (self.params.sweep_refined_sample_cap-added)//needed)
            if count == 0:
                method = 'local_refinement_cap'
                break
            # Deterministic breadth-first batches: highest current bound,
            # then time. Children are queued only after this whole batch.
            batch = np.array([heapq.heappop(heap) for _ in range(count)])
            left, right = batch[:, 1], batch[:, 2]
            grid = left[:, None] + (right-left)[:, None] * self._refinement_fractions
            grid[:, -1] = right
            path = self.executed_path(q_now, target, grid[:, 1:-1].reshape(-1))
            values = self._point_deviations(self._sweep_points(path), start, end)
            added += len(values)
            lower = max(lower, float(np.max(values)))
            d = np.concatenate((batch[:, 3:4], values.reshape(count, needed), batch[:, 4:5]), axis=1)
            bounds = np.maximum(d[:, :-1], d[:, 1:]) + self._interval_error(np.diff(grid, axis=1))
            if np.any(bounds <= limit):
                certified = max(certified, float(np.max(bounds[bounds <= limit])))
            for i, j in zip(*np.nonzero(bounds > limit)):
                heapq.heappush(heap, (-float(bounds[i,j]), float(grid[i,j]), float(grid[i,j+1]),
                                     float(d[i,j]), float(d[i,j+1])))
        if lower > limit:
            method, result_upper = 'local_sample_rejection', lower
        else:
            result_upper = max(certified, -heap[0][0] if heap else lower)
        return dict(sweep_m=float(result_upper), sampled_sweep_m=lower, method=method,
                    samples=len(times)+added, base_samples=len(times), refined_samples=added,
                    duration_s=float(times[-1]), discretization_bound_m=float(result_upper-lower),
                    coarse_error_bound_m=initial_error, uncertified_intervals=len(heap),
                    certificate_complete=not heap and lower <= limit)

    def sweep_bounds(self, q_now_rad, candidates_rad):
        """Conservative path-sweep upper bounds, using endpoint axis radii.

        Radius to axis j is independent of upstream joints and q_j. Its change
        from downstream joint k is at most R_k * abs(dq_k). Backward induction
        bounds it throughout the endpoint joint box by min(link-length bound,
        (start_radius + end_radius + downstream total variation)/2). For any
        intermediate point, distances to the path's two endpoints are bounded
        by partial arc lengths; the smaller is at most half the total length.
        The straight segment includes both endpoints, so its distance is no
        larger. All configurations/points are processed in the same FK batch.
        """
        candidates = np.atleast_2d(candidates_rad)
        rotation, origins, palm = self._kinematics(np.concatenate((np.asarray(q_now_rad)[None], candidates)))
        points = np.stack((origins[:, 3], origins[:, 5], palm), axis=1)
        axes = (rotation @ self._kin_axis[None, :, :, None])[..., 0]
        radii = np.linalg.norm(_cross(axes[:, None], points[:, :, None] - origins[:, None]), axis=-1)
        start, end = radii[0], radii[1:]
        delta = abs(candidates - q_now_rad)
        variation = np.zeros((len(candidates), 3))
        for j in range(6, -1, -1):
            radius = np.minimum(self._point_levers[:, j],
                np.maximum(np.maximum(start[:, j], end[:, :, j]),
                           (start[:, j] + end[:, :, j] + variation) / 2))
            variation += radius * delta[:, j, None]
        # Roundoff allowance for FK/norm products, not a physical threshold.
        return np.max(variation, axis=1) / 2 + 1e-12

    def sweeps(self, q_now_rad, candidates_rad, *, step_s=None, force_dense=False):
        """Reuse only bit-identical paths and the immutable default sampling policy."""
        if step_s is not None or force_dense or not self.params.sweep_cache_capacity:
            return self._sweeps_uncached(q_now_rad, candidates_rad, step_s=step_s,
                                         force_dense=force_dense)
        candidates = np.atleast_2d(candidates_rad)
        origin = np.asarray(q_now_rad).tobytes()
        keys = [(origin, q.tobytes()) for q in candidates]
        missing = [i for i, key in enumerate(keys) if key not in self._sweep_memo]
        values = {} if not missing else dict(zip(missing,
            self._sweeps_uncached(q_now_rad, candidates[missing])))
        # Snapshot hits before FIFO insertion can evict a later row's entry.
        available = {i: self._sweep_memo[key] for i, key in enumerate(keys) if i not in values}
        results = []
        for i, key in enumerate(keys):
            value = values[i] if i in values else available[i]
            results.append(dict(value))
            if key not in self._sweep_memo:
                if len(self._sweep_memo) >= self.params.sweep_cache_capacity:
                    self._sweep_memo.pop(next(iter(self._sweep_memo)))
                self._sweep_memo[key] = dict(value)
        return results

    def _sweeps_uncached(self, q_now_rad, candidates_rad, *, step_s=None, force_dense=False):
        """Sample only elbow, wrist center, and arm end frame; no finger data.

        An arc-length upper bound can certify feasibility without sampling.
        Otherwise, coarse samples can prove rejection; survivors get a grid
        with <= step_s gaps, joint-arrival times, endpoints, and coarse samples.
        ``sweep_m`` is marked as an upper bound, sampled value, or rejection
        lower bound by ``method``. It is never an unmarked zero approximation.
        """
        candidates = np.atleast_2d(candidates_rad)
        step_s = self.params.sweep_step_s if step_s is None else step_s
        if not math.isfinite(step_s) or step_s <= 0:
            raise ValueError('sweep step must be positive')
        delta = abs(candidates - q_now_rad)
        duration = self._tau(candidates, q_now_rad)
        # Point-to-start and point-to-end distances are bounded by the two
        # partial arc lengths. Their minimum is at most half the whole length.
        bounds = np.max(delta @ self._point_levers.T, axis=1) / 2
        results = [dict(sweep_m=float(bound), sampled_sweep_m=None, method='arc_length_bound', samples=0,
                        duration_s=float(t), discretization_bound_m=0.)
                   for bound, t in zip(bounds, duration)]
        pending = list(range(len(candidates))) if force_dense else list(np.flatnonzero(bounds > self.params.sweep_limit_m))
        if not pending:
            return results
        if not force_dense:
            tighter = self.sweep_bounds(q_now_rad, candidates[pending])
            # Preserve the old sampler's decision too: sampled maximum is no
            # greater than this bound, and its largest error bar is B*h^2/8.
            error = self.sweep_curvature_bound_m_s2 * step_s ** 2 / 8
            sample = []
            for i, bound in zip(pending, tighter):
                if bound + error <= self.params.sweep_limit_m:
                    results[i] = dict(sweep_m=float(bound + error), sampled_sweep_m=None,
                        method='axis_radius_bound', samples=0, duration_s=float(duration[i]),
                        analytical_bound_m=float(bound), discretization_bound_m=float(error))
                else:
                    sample.append(i)
            pending = sample
            if not pending:
                return results
        times = {}
        paths = []
        for i in pending:
            coarse = np.linspace(0., duration[i], self.params.sweep_coarse_intervals + 1)
            arrivals = delta[i] / self.arm_speed_rad_s
            times[i] = self._unique_times(np.r_[coarse, arrivals])
            paths.append(self.executed_path(q_now_rad, candidates[i], times[i]))
        points = self._sweep_points(np.concatenate(paths))
        dense, offset = [], 0
        for i, path in zip(pending, paths):
            value = self._point_sweep(points[offset:offset + len(path)])
            offset += len(path)
            results[i] = dict(sweep_m=value, sampled_sweep_m=value, method='coarse_rejection', samples=len(path),
                              duration_s=float(duration[i]), discretization_bound_m=None)
            if force_dense or value <= self.params.sweep_limit_m:
                dense.append(i)
        if dense:
            paths = []
            for i in dense:
                grid = np.arange(int(math.floor(duration[i] / step_s)) + 1) * step_s
                times[i] = self._unique_times(np.r_[grid, times[i], duration[i]])
                paths.append(self.executed_path(q_now_rad, candidates[i], times[i]))
            points = self._sweep_points(np.concatenate(paths))
            offset = 0
            for i, path in zip(dense, paths):
                point_rows = points[offset:offset + len(path)]
                offset += len(path)
                if force_dense:
                    # Offline ground-truth audit: complete fixed grid, no cap.
                    value = self._point_sweep(point_rows)
                    error_bound = float(np.max(self._interval_error(np.diff(times[i])))) if len(path)>1 else 0.
                    results[i] = dict(sweep_m=value+error_bound, sampled_sweep_m=value,
                        method='sampled_upper_bound', samples=len(path), duration_s=float(duration[i]),
                        discretization_bound_m=error_bound)
                else:
                    results[i] = self._certify_intervals(q_now_rad, candidates[i], times[i], point_rows)
        return results

    def _residual(self, q, target, elbow, *, jacobian=False):
        frames = self.frames(q, jacobian=jacobian)
        pose, e = frames[:2]
        omega = rotation_log(pose[:, :3, :3] @ target[:3, :3].T)
        residual = np.concatenate(((pose[:, :3, 3] - target[:3, 3]) / self.params.s_p,
                                   omega / self.params.s_R), axis=1)
        er = (e - elbow) / self.params.s_e
        if not jacobian:
            return residual, er
        theta = np.linalg.norm(omega, axis=1)
        # Inverse left SO(3) Jacobian. The series is stable at zero.
        coefficient = np.full_like(theta, 1 / 12)
        large = theta > 1e-5
        coefficient[large] = ((1 - theta[large] / (2 * np.tan(theta[large] / 2)))
                               / theta[large] ** 2)
        skew = _hat(omega)
        inverse = np.eye(3) - skew / 2 + coefficient[:, None, None] * (skew @ skew)
        j = np.concatenate((frames[2] / self.params.s_p,
                            (inverse @ frames[3]) / self.params.s_R), axis=1)
        return residual, er, j, frames[4] / self.params.s_e

    def _wrap(self, q):
        middle = (self.lower + self.upper) / 2
        return np.clip(middle + (q - middle + np.pi) % (2 * np.pi) - np.pi,
                       self.lower, self.upper)

    def _seeds(self, request, previous):
        seeds = []
        if previous is not None:
            seeds.append(('previous', previous.copy()))
        seeds.append(('request_wrapped', self._wrap(request)))
        for a in (-1, 1):
            for b in (-1, 1):
                equivalent = request.copy()
                equivalent[2] += a * np.pi
                equivalent[3] *= -1
                equivalent[4] += b * np.pi
                seeds.append((f'equivalent_{a:+d}_{b:+d}', self._wrap(equivalent)))
        turns = int(round(((self.lower[6] + self.upper[6]) / 2 - request[6]) / (2 * np.pi)))
        for k in (turns - 1, turns, turns + 1):
            shifted = self._wrap(request)
            shifted[6] = request[6] + 2 * np.pi * k
            if self.lower[6] <= shifted[6] <= self.upper[6]:
                seeds.append((f'J7_turns_{k:+d}', shifted))
        unique = []
        for name, seed in seeds:
            if not any(np.max(abs(seed - existing)) < 1e-12 for _, existing in unique):
                unique.append((name, seed))
        return [name for name, _ in unique], np.array([q for _, q in unique])

    def _warm_ok(self, q, r, er, previous):
        """Physical approximation bounds; sweep feasibility is checked separately."""
        p = self.params
        return (np.linalg.norm(r[:3]) * p.s_p <= p.delta_position_m
                and np.linalg.norm(r[3:]) * p.s_R <= p.delta_orientation_rad
                and np.linalg.norm(er) * p.s_e <= p.elbow_certificate_m)

    def _hand_solve(self, seeds, target, elbow, *, certificate_previous=None, iterations=None):
        q = seeds.copy()
        p = self.params
        limit = p.max_iterations if iterations is None else iterations
        iterations = 0
        for _ in range(limit):
            r, er, j, _ = self._residual(q, target, elbow, jacobian=True)
            if certificate_previous is not None and self._warm_ok(q[0], r[0], er[0], certificate_previous):
                break
            gradient = (j.transpose(0, 2, 1) @ r[..., None])[..., 0]
            fixed = (((q <= self.lower + 1e-12) & (gradient > 0))
                     | ((q >= self.upper - 1e-12) & (gradient < 0)))
            j = j * (~fixed)[:, None, :]
            normal = j.transpose(0, 2, 1) @ j + p.damping ** 2 * self._identity7
            rhs = -(j.transpose(0, 2, 1) @ r[..., None])
            step = np.linalg.solve(normal, rhs)[..., 0]
            step *= np.minimum(1., p.max_step_rad / np.maximum(np.max(abs(step), axis=1), 1e-30))[:, None]
            if np.max(abs(step)) <= p.step_tolerance_rad:
                break
            value = np.sum(r * r, axis=1)
            trials = np.clip(q[:, None] + self._line_alphas[None, :, None] * step[:, None], self.lower, self.upper)
            rr, ee = self._residual(trials.reshape(-1, 7), target, elbow)
            scores = np.sum(rr * rr, axis=1).reshape(len(q), -1)
            improve = scores < value[:, None]
            accepted = np.any(improve, axis=1) & (np.max(abs(step), axis=1) > p.step_tolerance_rad)
            first = np.argmax(improve, axis=1)
            best = q.copy()
            best[accepted] = trials[np.arange(len(q))[accepted], first[accepted]]
            iterations += 1
            change = np.max(abs(best - q))
            q = best
            if change <= p.step_tolerance_rad:
                break
        return q, iterations

    def _in_band(self, r, best_r):
        return ((np.linalg.norm(r[:, :3], axis=1) * self.params.s_p
                 <= np.linalg.norm(best_r[:3]) * self.params.s_p + self.params.delta_position_m + 1e-14)
                & (np.linalg.norm(r[:, 3:], axis=1) * self.params.s_R
                   <= np.linalg.norm(best_r[3:]) * self.params.s_R + self.params.delta_orientation_rad + 1e-14))

    def _elbow_solve(self, q, target, elbow, best_r, *, iterations=None):
        p = self.params
        limit = p.max_elbow_iterations if iterations is None else iterations
        iterations = 0
        initial_r, initial_er = self._residual(q, target, elbow)
        eligible = self._in_band(initial_r, best_r)
        elbow_norm = np.linalg.norm(initial_er, axis=1)
        if not np.any(eligible & (elbow_norm > p.elbow_tie_m / p.s_e)):
            return q, iterations
        active = eligible & (elbow_norm > 0.)
        # Ineligible rows never move in the original algorithm. Zero-elbow rows
        # cannot improve. Removing these rows preserves every candidate's math.
        output, q = q.copy(), q[active].copy()
        for _ in range(limit):
            r, er, j, je = self._residual(q, target, elbow, jacobian=True)
            eligible = self._in_band(r, best_r)
            if not np.any(eligible & (np.linalg.norm(er, axis=1) > p.elbow_tie_m / p.s_e)):
                break
            # Project the elbow GN step onto the hand Jacobian's null space.
            _, singular, vt = np.linalg.svd(j, full_matrices=True)
            rank = np.sum(singular > 1e-9, axis=1)
            null = vt.transpose(0, 2, 1) * (np.arange(7)[None, None, :] >= rank[:, None, None])
            a = je @ null
            step_null = np.linalg.solve(a.transpose(0, 2, 1) @ a + p.damping ** 2 * self._identity7,
                                        -(a.transpose(0, 2, 1) @ er[..., None]))
            step = (null @ step_null)[..., 0]
            step[~eligible] = 0.
            step *= np.minimum(1., p.max_step_rad / np.maximum(np.max(abs(step), axis=1), 1e-30))[:, None]
            # Evaluate all line-search rows in one FK batch. Original q is first,
            # so argmin preserves strict improvement and the old tie ordering.
            trials = np.clip(q[:, None, :] + self._elbow_alphas[None, :, None] * step[:, None, :], self.lower, self.upper)
            rr, ee = self._residual(trials.reshape(-1, 7), target, elbow)
            values = np.sum(ee * ee, axis=1).reshape(len(q), -1)
            allowed = self._in_band(rr, best_r).reshape(len(q), -1) & eligible[:, None]
            allowed[:, 0] = True
            best_index = np.argmin(np.where(allowed, values, np.inf), axis=1)
            best = trials[np.arange(len(q)), best_index]
            iterations += 1
            change = np.max(abs(q - best))
            q = best
            if change <= p.step_tolerance_rad:
                break
        output[active] = q
        return output, iterations

    def _tau(self, q, q_now):
        return np.max(abs(q - q_now), axis=-1) / self.arm_speed_rad_s

    def _request_distance(self, q, request):
        difference = np.array(q - request, copy=True)
        # The master J7 is continuous; AR5 output constraints remain finite.
        difference[..., 6] = (difference[..., 6] + np.pi) % (2 * np.pi) - np.pi
        return np.sum((difference / self.velocity_rad_s) ** 2, axis=-1)

    def _select(self, candidates, target, elbow, q_now, request, sweep_cache):
        """Lazy feasible anchors preserve the original physical-band order."""
        r, er = self._residual(candidates, target, elbow)
        pose_ok = ((np.linalg.norm(r[:, :3], axis=1) * self.params.s_p <= self.params.epsilon_position_m)
                   & (np.linalg.norm(r[:, 3:], axis=1) * self.params.s_R <= self.params.epsilon_orientation_rad))
        scores = np.sum(r * r, axis=1)
        indices = np.flatnonzero(pose_ok)

        def first_feasible(order):
            for i in order:
                key = candidates[i].tobytes()
                if key not in sweep_cache:
                    sweep_cache[key] = self.sweeps(q_now, candidates[i:i + 1])[0]
                if sweep_cache[key]['sweep_m'] <= self.params.sweep_limit_m:
                    return int(i)
            return None

        anchor = first_feasible(indices[np.argsort(scores[indices], kind='stable')])
        if anchor is None:
            selected = int(np.argmin(np.where(pose_ok, scores, np.inf))) if np.any(pose_ok) else int(np.argmin(scores))
            return selected, False, 'sweep' if np.any(pose_ok) else 'unreachable', r[selected].copy()
        best_r = r[anchor].copy()
        eligible = pose_ok & self._in_band(r, best_r)
        elbow_errors = np.linalg.norm(er, axis=1) * self.params.s_e
        indices = np.flatnonzero(eligible)
        elbow_anchor = first_feasible(indices[np.argsort(elbow_errors[indices], kind='stable')])
        eligible &= elbow_errors <= elbow_errors[elbow_anchor] + self.params.elbow_band_m
        indices = np.flatnonzero(eligible)
        order = np.lexsort((self._tau(candidates[indices], q_now), self._request_distance(candidates[indices], request)))
        selected = first_feasible(indices[order])
        return selected, True, None, best_r

    @staticmethod
    def _candidate_pool(parts):
        labels = [label for names, _ in parts for label in names]
        rows = np.concatenate([np.asarray(values) for _, values in parts])
        same = np.max(abs(rows[:, None] - rows[None, :]), axis=2) < 1e-12
        kept = []
        for i in range(len(rows)):
            if not kept or not np.any(same[i, kept]):
                kept.append(i)
        return [labels[i] for i in kept], rows[kept].copy()


    def _full_solve(self, request, previous, q_now, target, elbow, sweep_cache, warm):
        names, seeds = self._seeds(request, previous)
        solved, hi = self._hand_solve(seeds, target, elbow)
        parts = [([n + ':hand' for n in names], solved), ([n + ':seed' for n in names], seeds)]
        if warm is not None:
            parts.append((['previous:warm_iterate'], warm[None]))
        labels, pool = self._candidate_pool(parts)
        _, _, _, best_r = self._select(pool, target, elbow, q_now, request, sweep_cache)
        refined, ei = self._elbow_solve(pool, target, elbow, best_r)
        labels, pool = self._candidate_pool([(labels, pool), ([n + ':elbow' for n in labels], refined)])
        selected, feasible, reason, best_r = self._select(pool, target, elbow, q_now, request, sweep_cache)
        return pool[selected], labels[selected], feasible, reason, best_r, hi, ei, len(seeds)

    def _measure(self, q, target, elbow, previous):
        r, er = self._residual(q[None], target, elbow)
        errors = dict(position_m=float(np.linalg.norm(r[0, :3]) * self.params.s_p),
                      orientation_rad=float(np.linalg.norm(r[0, 3:]) * self.params.s_R),
                      elbow_m=float(np.linalg.norm(er[0]) * self.params.s_e))
        costs = dict(position=float(r[0, :3] @ r[0, :3]),
                     orientation=float(r[0, 3:] @ r[0, 3:]), elbow=float(er[0] @ er[0]),
                     motion_s2=0. if previous is None else float(np.sum(((q - previous) / self.velocity_rad_s) ** 2)))
        return errors, costs

    def command(self, master_deg, prev_output_deg, q_now_deg, *, full_search=False):
        """Complete this query in the same cycle; only exact/warm shortcuts apply."""
        started = perf_counter()
        result = self._command_impl(master_deg, prev_output_deg, q_now_deg, full_search=full_search)
        result.solve_ms = (perf_counter()-started)*1000
        return result

    def _command_impl(self, master_deg, prev_output_deg, q_now_deg, *, full_search=False):
        started = perf_counter()
        request_deg = self._vector(master_deg) + self.offset_deg
        request = np.deg2rad(request_deg)
        previous_deg = None if prev_output_deg is None else self._vector(prev_output_deg)
        previous = None if previous_deg is None else np.deg2rad(previous_deg)
        q_now_deg = self._vector(q_now_deg)
        q_now = np.deg2rad(q_now_deg)
        if previous is not None and np.any((previous < self.lower - 1e-10) | (previous > self.upper + 1e-10)):
            raise ValueError('previous output must be inside the mapper box constraints')
        target_batch, elbow_batch = self.frames(request[None])
        target, elbow = target_batch[0], elbow_batch[0]
        candidate, certificate, seed_used = None, 'full', ''
        hand_iterations = elbow_iterations = seeds_count = 0
        feasible, hold_reason = False, None
        sweep_cache = {}
        previous_copy = (self._last_copy_deg is not None and previous_deg is not None
                         and np.max(abs(np.deg2rad(previous_deg - self._last_copy_deg))) <= 1e-9)
        if (not full_search and np.all((request >= self.lower) & (request <= self.upper))
                and previous_copy):
            sweep = self.sweeps(q_now, request[None])[0]
            sweep_cache[request.tobytes()] = sweep
            if sweep['sweep_m'] <= self.params.sweep_limit_m:
                # Pose, elbow, and request-distance objectives attain zero.
                candidate, certificate, seed_used = request.copy(), 'exact', 'exact_certificate'
                best_r, feasible = np.zeros(6), True
        warm = None
        if candidate is None and previous is not None:
            warm, hi = self._hand_solve(previous[None], target, elbow, certificate_previous=previous)
            wr, we = self._residual(warm, target, elbow)
            if not self._warm_ok(warm[0], wr[0], we[0], previous):
                warm, ei = self._elbow_solve(warm, target, elbow, wr[0].copy())
                elbow_iterations += ei
                wr, we = self._residual(warm, target, elbow)
            hand_iterations += hi
            seeds_count += 1
            warm = warm[0]
            if not full_search and self._warm_ok(warm, wr[0], we[0], previous):
                sweep = self.sweeps(q_now, warm[None])[0]
                sweep_cache[warm.tobytes()] = sweep
                if sweep['sweep_m'] <= self.params.sweep_limit_m:
                    candidate, certificate, seed_used = warm, 'warm', 'warm_certificate'
                    feasible = True
            if candidate is not None:
                best_r = np.zeros(6)  # nonnegative global lower bound, not a measured full solve
        if candidate is None:
            candidate, seed_used, feasible, hold_reason, best_r, hi, ei, count = self._full_solve(
                request, previous, q_now, target, elbow, sweep_cache, warm)
            hand_iterations += hi
            elbow_iterations += ei
            seeds_count += count
        if candidate.tobytes() not in sweep_cache:
            sweep_cache[candidate.tobytes()] = self.sweeps(q_now, candidate[None])[0]
        candidate_sweep = sweep_cache[candidate.tobytes()]
        candidate_frames = (target_batch, elbow_batch) if certificate == 'exact' else self.frames(candidate[None])
        candidate_pose = candidate_frames[0][0]
        def measure_frames(q, frames):
            pose, e = frames[0][0], frames[1][0]
            pe = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
            re = float(np.linalg.norm(rotation_log(pose[:3, :3] @ target[:3, :3].T)))
            ee = float(np.linalg.norm(e - elbow))
            errors = dict(position_m=pe, orientation_rad=re, elbow_m=ee)
            costs = dict(position=(pe / self.params.s_p) ** 2, orientation=(re / self.params.s_R) ** 2,
                         elbow=(ee / self.params.s_e) ** 2, tau_s=float(self._tau(q, q_now)),
                         request_distance_s2=float(self._request_distance(q, request)),
                         motion_s2=0. if previous is None else float(np.sum(((q - previous) / self.velocity_rad_s) ** 2)))
            return errors, costs
        candidate_errors, candidate_costs = measure_frames(candidate, candidate_frames)
        active = tuple(f'J{i + 1}:{bound}' for i in range(7) for bound, value in
                       (('lower', self.lower[i]), ('upper', self.upper[i])) if abs(candidate[i] - value) <= 1e-7)
        reachable = feasible
        output = candidate if reachable else previous
        if output is None:
            achieved, errors = None, dict(position_m=None, orientation_rad=None, elbow_m=None)
            costs = dict(position=None, orientation=None, elbow=None, motion_s2=None, tau_s=None, request_distance_s2=None)
        elif reachable:
            achieved, errors, costs = candidate_pose, candidate_errors, candidate_costs
        else:
            held_frames = self.frames(output[None])
            achieved = held_frames[0][0]
            errors, costs = measure_frames(output, held_frames)
        reason = '' if reachable else (f'{self.side.upper()} {hold_reason.upper()}: position={candidate_errors["position_m"]:.6g} m, '
                                       f'orientation={candidate_errors["orientation_rad"]:.6g} rad; '
                                       f'sweep={candidate_sweep["sweep_m"]:.6g} m; '
                                       f'active={",".join(active) or "none (bounded local solve)"}; '
                                       + ('no previous reachable output' if previous is None else 'holding previous reachable output'))
        output_deg = (None if output is None else tuple(request_deg if certificate == 'exact' else np.rad2deg(output))
                      if reachable else tuple(previous_deg))
        output_sweep = (None if output is None else candidate_sweep if reachable else self.sweeps(q_now, output[None])[0])
        if costs['position'] is not None:
            costs['sweep_m'] = output_sweep['sweep_m']
        candidate_costs['sweep_m'] = candidate_sweep['sweep_m']
        branch_origin = q_now if previous is None else previous
        branch = bool(reachable and np.max(abs(np.rad2deg(candidate[[2, 4]] - branch_origin[[2, 4]])))
                      >= self.params.branch_joint_threshold_deg)
        if not full_search:
            self._last_copy_deg = (request_deg.copy() if reachable and
                np.max(abs(np.deg2rad(output_deg) - request)) <= 1e-9 else None)
        return MappingResult(output_deg, bool(reachable), not reachable,
                             target, achieved, errors, costs, active, seed_used,
                             hand_iterations + elbow_iterations, (perf_counter() - started) * 1000, reason,
                             tuple(np.rad2deg(candidate)), candidate_pose, candidate_errors, candidate_costs,
                             certificate, seeds_count,
                             dict(position_m=float(np.linalg.norm(best_r[:3]) * self.params.s_p),
                                  orientation_rad=float(np.linalg.norm(best_r[3:]) * self.params.s_R)),
                             None if output_sweep is None else output_sweep['sweep_m'], candidate_sweep['sweep_m'],
                             'none' if output_sweep is None else output_sweep['method'],
                             float(self._tau(candidate, q_now)), branch, hold_reason, tuple(q_now_deg))
