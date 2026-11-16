"""Arm-only pose mapping, spatial feasibility, and same-cycle full-search decisions."""
from helpers import bench_directory
from functools import lru_cache
from dataclasses import replace
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from dexstation.inputs.arm_mapping import ArmMapper
from dexstation.model.kinematics import Chain
from dexstation.control.joint_motion import load_profile
from dexstation.inputs.master_mapping import MappingParams, MasterPoseMapper, rotation_log
from dexstation.profiles.compose import compose, load_task, load_workbench

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def scene():
    if not os.environ.get('DEXSTATION_WORKSPACE'):
        pytest.skip('WP12 requires an explicitly configured DEXSTATION_WORKSPACE')
    value = compose(load_workbench(bench_directory()),
                    load_task('empty'), 'sapien', seed=0)
    if not all(Path(getattr(value.spec, s + '_arm').urdf).is_file() for s in ('left', 'right')):
        pytest.skip('AR5 URDFs are absent from the configured workspace')
    return value



def call_request(mapper, request, previous, q_now, *, full_search=False):
    return mapper.command(np.asarray(request) - mapper.offset_deg, previous, q_now, full_search=full_search)


def advance(mapper, q_now, target, dt=.01):
    if target is None:
        return np.array(q_now, copy=True)
    return np.rad2deg(mapper.executed_path(np.deg2rad(q_now), np.deg2rad(target), np.array([dt]))[0])


def initial_copy(scene, mapper, raw):
    copier = ArmMapper({s: getattr(scene.spec, s + '_arm').home_q for s in ('left', 'right')},
                       startup='none', joint_offset_deg=scene.calibration['teleop'])
    return np.rad2deg(np.clip(copier.command(mapper.side, raw, 0.).positions, mapper.lower, mapper.upper))


def test_parameters_and_arm_only_scope(scene):
    p = MappingParams()
    assert (p.s_p, p.s_R, p.s_e) == (.01, .05, .05)
    assert (p.delta_position_m, p.delta_orientation_rad) == (.0005, math.radians(.1))
    assert (p.epsilon_position_m, p.epsilon_orientation_rad) == (.002, math.radians(1))
    assert (p.elbow_band_m, p.elbow_certificate_m, p.sweep_limit_m, p.margin_deg) == (.002, .001, .02, 3.)
    assert p.arm_speed_deg_s == load_profile('station')['arm']['max_speed_deg_s'] == 40.
    arm_only = SimpleNamespace(spec=SimpleNamespace(right_arm=scene.spec.right_arm), calibration=scene.calibration)
    mapper = MasterPoseMapper(arm_only, 'right')
    assert mapper.offset_deg[0] == pytest.approx(-90 + 19.15)
    with pytest.raises(TypeError):
        mapper.command(np.zeros(7), None)  # q_now must never be inferred.


@pytest.mark.parametrize('side', ['left', 'right'])
def test_fk_points_and_analytic_jacobians(scene, side):
    mapper = MasterPoseMapper(scene, side)
    arm = getattr(scene.spec, side + '_arm')
    original = Chain(arm.urdf, arm.tip_link)
    q = np.deg2rad([10, 35, 110, 40, -25, 10, 15.])
    pose, motion = original.frames(q, np.eye(4))
    actual, elbow, wrist = mapper.frames(q[None], points=True)
    np.testing.assert_allclose(actual[0], pose, atol=1e-14)
    np.testing.assert_allclose(elbow[0], motion[3][1], atol=1e-14)
    np.testing.assert_allclose(wrist[0], motion[5][1], atol=1e-14)
    target, target_elbow = mapper.frames((q + .2)[None])
    r, er, jac, ej = mapper._residual(q[None], target[0], target_elbow[0], jacobian=True)
    for column in range(7):
        shifted = q.copy()
        shifted[column] += 1e-7
        r2, e2 = mapper._residual(shifted[None], target[0], target_elbow[0])
        np.testing.assert_allclose((r2 - r)[0] / 1e-7, jac[0, :, column], atol=5e-6)
        np.testing.assert_allclose((e2 - er)[0] / 1e-7, ej[0, :, column], atol=2e-6)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_batched_fk_matches_original_on_random_configurations(scene, side, record_property):
    m = MasterPoseMapper(scene, side)
    arm = getattr(scene.spec, side + '_arm')
    original = Chain(arm.urdf, arm.tip_link)
    q = np.random.default_rng(731).uniform(-2*np.pi, 2*np.pi, (512, 7))
    poses, elbow, wrist = m.frames(q, points=True)
    np.testing.assert_allclose(m._sweep_points(q), np.stack((elbow, wrist, poses[:, :3, 3]), axis=1),
                               atol=1e-12, rtol=0)
    maximum_position = maximum_rotation = 0.
    for i, configuration in enumerate(q):
        reference, motion = original.frames(configuration, np.eye(4))
        pe = float(np.linalg.norm(poses[i, :3, 3] - reference[:3, 3]))
        re = float(np.linalg.norm(rotation_log(poses[i, :3, :3] @ reference[:3, :3].T)))
        maximum_position, maximum_rotation = max(maximum_position, pe), max(maximum_rotation, re)
        assert pe <= 1e-12 and re <= 1e-12
        np.testing.assert_allclose(elbow[i], motion[3][1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(wrist[i], motion[5][1], atol=1e-12, rtol=0)
    record_property('position_error_m', maximum_position)
    record_property('orientation_error_rad', maximum_rotation)


@pytest.mark.parametrize('angle', [0., 1e-10, .8, math.pi - 1e-8, math.pi])
def test_rotation_log(angle):
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    assert np.linalg.norm(rotation_log(rotation)) == pytest.approx(angle, abs=1e-12)


def test_rate_path_stops_shorter_joints_first(scene):
    m = MasterPoseMapper(scene, 'right')
    start = np.deg2rad([0, 70, 100, 40, -10, 0, 0.])
    goal = np.deg2rad([0, 70, -80, -40, 170, 0, 0.])
    path = np.rad2deg(m.executed_path(start, goal, np.array([0., 1., 2., 4.5])))
    np.testing.assert_allclose(path[1, [2, 3, 4]], [60, 0, 30], atol=1e-12)
    np.testing.assert_allclose(path[2, [2, 3, 4]], [20, -40, 70], atol=1e-12)
    np.testing.assert_allclose(path[-1], np.rad2deg(goal), atol=1e-12)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_full_point_acceleration_bound_covers_random_rates(scene, side, record_property):
    m = MasterPoseMapper(scene, side)
    # Closed-form contract for this pinned AR5 geometry. This detects omitting
    # a term even if random accelerations happen to stay below the old bound.
    w2 = m.arm_speed_rad_s**2
    elbow = (.1745 + 5*.332 + 12*.0105)*w2
    wrist = elbow + 22*math.hypot(.0105,.278)*w2
    expected = np.array([elbow,wrist,wrist+70*.09*w2])
    np.testing.assert_allclose(m._point_acceleration_bounds_m_s2,expected,atol=1e-12,rtol=0)
    rng = np.random.default_rng(1401 if side == 'left' else 1402)
    count, h = 2000, 1e-4
    configurations = rng.uniform(m.lower, m.upper, (count, 7))
    signs = rng.choice([-1., 1.], (count, 7))
    maxima = np.zeros(3)
    for start in range(0, count, 512):
        q = configurations[start:start+512]
        velocity = signs[start:start+512] * m.arm_speed_rad_s
        points = m._sweep_points(np.concatenate((q-h*velocity, q, q+h*velocity)))
        before, center, after = np.split(points, 3)
        acceleration = np.linalg.norm((after-2*center+before)/(h*h), axis=-1)
        maxima = np.maximum(maxima, np.max(acceleration, axis=0))
        assert np.all(acceleration <= m._point_acceleration_bounds_m_s2[None] + 1e-5)
    assert m.sweep_curvature_bound_m_s2 == max(m._point_acceleration_bounds_m_s2)
    record_property('configurations_and_sign_patterns', count)
    record_property('finite_difference_step_s', h)
    record_property('max_point_acceleration_m_s2', maxima.tolist())
    record_property('point_acceleration_bounds_m_s2', m._point_acceleration_bounds_m_s2.tolist())


def test_straight_roll_is_allowed_and_bent_roll_is_rejected(scene, record_property):
    m = MasterPoseMapper(scene, 'right')
    a = np.deg2rad([0, 70, 100, 0, -10, 0, 0.])
    b = a.copy()
    b[[2, 4]] += np.deg2rad([-180, 180])
    straight = m.sweeps(a, b[None])[0]
    dense = m.sweeps(a, b[None], step_s=.001, force_dense=True)[0]
    assert straight['duration_s'] == pytest.approx(4.5)
    assert dense['sampled_sweep_m'] == pytest.approx(.0105, abs=1e-12)
    assert straight['sweep_m'] <= m.params.sweep_limit_m
    assert dense['sampled_sweep_m'] <= straight['sweep_m']
    a[3], b[3] = np.deg2rad(40), np.deg2rad(-40)
    bent = m.sweeps(a, b[None])[0]
    assert bent['sweep_m'] > m.params.sweep_limit_m
    assert bent['method'] == 'coarse_rejection'
    record_property('straight_roll_sweep_m', straight['sweep_m'])
    record_property('bent_roll_rejection_lower_bound_m', bent['sweep_m'])


@pytest.mark.parametrize('side', ['left', 'right'])
def test_axis_radius_bound_covers_dense_random_paths(scene, side, record_property):
    m = MasterPoseMapper(scene, side)
    rng = np.random.default_rng(1031)
    margins = []
    for _ in range(24):
        start = rng.uniform(m.lower, m.upper)
        goal = np.clip(start + rng.uniform(-np.pi, np.pi, 7), m.lower, m.upper)
        bound = m.sweep_bounds(start, goal[None])[0]
        dense = m.sweeps(start, goal[None], step_s=.001, force_dense=True)[0]['sampled_sweep_m']
        assert bound + 1e-12 >= dense
        margins.append(float(bound - dense))
    # A half-sized (incorrect) bound is detectably unsafe for a straight roll.
    start = np.deg2rad([0., 70., 100., 0., -10., 0., 0.])
    goal = start.copy()
    goal[[2, 4]] += np.deg2rad([-180., 180.])
    bound = m.sweep_bounds(start, goal[None])[0]
    dense = m.sweeps(start, goal[None], step_s=.001, force_dense=True)[0]['sampled_sweep_m']
    assert .5 * bound < dense <= bound
    record_property('minimum_random_bound_margin_m', min(margins))


def borderline_path():
    return np.deg2rad([19.19543515950331, 82.31311234851168, 174.75202348110275,
        .8442181675570805, -71.04238531698498, -.5429117208839896, 2.143795418385307]), np.deg2rad([
        19.51570883968362, 82.2, -134.37586132897204, .8382183702492363,
        -121.96288390573721, -.15392720991693667, 2.1])


def test_local_refinement_certifies_borderline_path_and_honors_cap(scene, record_property):
    start, goal = borderline_path()
    m = MasterPoseMapper(scene, 'right')
    result = m.sweeps(start, goal[None])[0]
    dense = m.sweeps(start, goal[None], step_s=.001, force_dense=True)[0]
    assert result['method'] == 'local_upper_bound' and result['certificate_complete']
    assert dense['sampled_sweep_m'] <= result['sweep_m'] <= .02
    assert 0 < result['refined_samples'] <= m.params.sweep_refined_sample_cap
    assert result['samples'] == result['base_samples'] + result['refined_samples']
    capped = MasterPoseMapper(scene, 'right', replace(MappingParams(), sweep_refined_sample_cap=0))
    rejected = capped.sweeps(start, goal[None])[0]
    assert rejected['method'] == 'local_refinement_cap' and not rejected['certificate_complete']
    assert rejected['refined_samples'] == 0 and rejected['sweep_m'] > .02
    assert .02-dense['sampled_sweep_m'] <= rejected['coarse_error_bound_m']
    record_property('local_refinement', result)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_local_sweep_certificate_is_sound_against_dense_random_paths(scene, side, record_property):
    m = MasterPoseMapper(scene, side)
    rng = np.random.default_rng(1201)
    counts = dict(accepted=0, rejected=0, local=0)
    for i in range(40):
        start = rng.uniform(m.lower, m.upper)
        goal = np.clip(start + rng.uniform(-np.pi, np.pi, 7), m.lower, m.upper)
        if i < 20:
            start = np.deg2rad([0., 70., rng.uniform(140,174), rng.uniform(-.3,.3),
                               rng.uniform(-174,-140), rng.uniform(-1,1), 0.])
            goal = start.copy()
            goal[[2,4]] = np.deg2rad([rng.uniform(-174,-140), rng.uniform(140,174)])
        result = m.sweeps(start, goal[None])[0]
        dense = m.sweeps(start, goal[None], step_s=.001, force_dense=True)[0]
        accepted = result['sweep_m'] <= .02
        counts['accepted' if accepted else 'rejected'] += 1
        counts['local'] += int(result['method'].startswith('local'))
        if accepted:
            assert dense['sampled_sweep_m'] <= result['sweep_m'] + 1e-12
            assert dense['sampled_sweep_m'] <= .02 + 1e-12
        if 'refined_samples' in result:
            assert result['refined_samples'] <= m.params.sweep_refined_sample_cap
    assert counts['accepted'] and counts['rejected'] and counts['local']
    record_property('certificate_path_counts', counts)


def test_interval_error_term_prevents_hidden_peak_acceptance(scene):
    m = MasterPoseMapper(scene, 'right')
    # A C2 curve with |p''| <= 2000 m/s^2, maximum between the base samples.
    # This tests the interpolation certificate independently of AR5 FK.
    m.sweep_curvature_bound_m_s2 = 2000.
    start = np.zeros(7)
    goal = start.copy()
    goal[0] = m.arm_speed_rad_s * .02
    def points(q):
        x = q[:,0] / (m.arm_speed_rad_s*.02)
        z = (.02/1.51)*np.sin(np.pi*x)**2*(1+x)
        p = np.zeros((len(q),3,3));p[:,:,2] = z[:,None]
        return p
    times = np.linspace(0.,.02,9)
    base = points(m.executed_path(start,goal,times))
    fine = points(m.executed_path(start,goal,np.linspace(0.,.02,21)))
    assert m._point_sweep(base) < .02 < m._point_sweep(fine)
    with patch.object(m, '_sweep_points', points):
        result = m._certify_intervals(start,goal,times,base)
    assert result['sweep_m'] > .02 and not result['certificate_complete']


def test_time_grid_deduplication_matches_numpy_unique():
    rng = np.random.default_rng(1301)
    for duration in np.r_[0., .01, 8.8251, rng.uniform(0., 9., 128)]:
        times = np.r_[np.linspace(0.,duration,9),rng.uniform(0.,duration,7),
                      np.arange(int(duration/.01)+1)*.01,duration]
        actual = MasterPoseMapper._unique_times(times)
        expected = np.unique(times)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


def test_lazy_selection_preserves_feasible_band_anchors(scene):
    m = MasterPoseMapper(scene, 'right')
    # Physical bands are anchored to feasible rows, not the lowest-error
    # sweep-blocked row. Exercise selection against eager feasibility.
    rng = np.random.default_rng(1041)
    for _ in range(16):
        q = rng.uniform(m.lower, m.upper)
        candidates = np.clip(q + rng.normal(0., .003, (12, 7)), m.lower, m.upper)
        target, elbow = m.frames(q[None])
        now = q + rng.uniform(-.02, .02, 7)
        cache = {}
        selected, feasible, reason, best_r = m._select(candidates, target[0], elbow[0], now, q, cache)
        r, er = m._residual(candidates, target[0], elbow[0])
        pose_ok = ((np.linalg.norm(r[:, :3], axis=1) * m.params.s_p <= m.params.epsilon_position_m)
                   & (np.linalg.norm(r[:, 3:], axis=1) * m.params.s_R <= m.params.epsilon_orientation_rad))
        mask = np.array([ok and s['sweep_m'] <= m.params.sweep_limit_m
                         for ok, s in zip(pose_ok, m.sweeps(now, candidates))])
        scores = np.sum(r*r, axis=1)
        if not np.any(mask):
            assert not feasible and reason == ('sweep' if np.any(pose_ok) else 'unreachable')
            continue
        anchor = r[np.argmin(np.where(mask, scores, np.inf))]
        eligible = mask & m._in_band(r, anchor)
        ee = np.linalg.norm(er, axis=1) * m.params.s_e
        eligible &= ee <= min(ee[eligible]) + m.params.elbow_band_m
        indices = np.flatnonzero(eligible)
        order = np.lexsort((m._tau(candidates[indices], now), m._request_distance(candidates[indices], q)))
        assert feasible and reason is None and selected == indices[order[0]]
        np.testing.assert_array_equal(best_r, anchor)
    # Deterministic counterexamples to anchoring a band on an infeasible row.
    candidates = np.zeros((4, 7))
    candidates[:, 0] = np.arange(4) * .1
    for blocked, position, elbows, hand_anchor in [
            (0, [0., .0004, .0007, .00075], [0., .004, 0., .001], 1),
            (1, [0., .0001, .0002, .0003], [.01, 0., .004, .005], 0)]:
        r, er = np.zeros((4, 6)), np.zeros((4, 3))
        r[:, 0], er[:, 0] = np.array(position)/m.params.s_p, np.array(elbows)/m.params.s_e
        def sweeps(now, rows):
            return [dict(sweep_m=.03 if round(q[0]/.1) == blocked else 0.) for q in rows]
        with patch.object(m, '_residual', return_value=(r, er)), patch.object(m, 'sweeps', sweeps):
            selected, feasible, reason, best_r = m._select(candidates, np.eye(4), np.zeros(3),
                candidates[3], candidates[3], {})
        assert feasible and reason is None and selected == 3
        np.testing.assert_array_equal(best_r, r[hand_anchor])


def test_short_path_bound_covers_densely_sampled_sweep(scene):
    m = MasterPoseMapper(scene, 'right')
    q = np.deg2rad([0, 40, 30, 60, 10, 5, 0.])
    goal = q + np.deg2rad([.4, -.3, .2, -.1, .15, -.2, .4])
    bound = m.sweeps(q, goal[None])[0]
    dense = m.sweeps(q, goal[None], step_s=.0001, force_dense=True)[0]
    assert bound['method'] == 'arc_length_bound' and bound['samples'] == 0
    assert dense['sweep_m'] <= bound['sweep_m'] <= .02


def test_sweep_cache_matches_uncached_and_uses_current_state(scene):
    m = MasterPoseMapper(scene, 'right')
    start = np.deg2rad([0., 70., 100., 0., -10., 0., 0.])
    goal = start.copy()
    goal[[2, 4]] += np.deg2rad([-180., 180.])
    first = m.sweeps(start, goal[None])[0]
    assert first == m._sweeps_uncached(start, goal[None])[0]
    assert m.sweeps(start, goal[None])[0] == first
    second = m.sweeps(goal, goal[None])[0]
    assert second == m._sweeps_uncached(goal, goal[None])[0]
    assert second['sweep_m'] == 0. and first['sweep_m'] > 0.
    assert len(m._sweep_memo) <= m.params.sweep_cache_capacity
    tiny = MasterPoseMapper(scene, 'right', replace(MappingParams(), sweep_cache_capacity=2))
    old = np.array([start, goal])
    tiny.sweeps(start, old)
    mixed = np.array([start + .001, start, goal])
    # Inserting the miss first evicts a hit that is needed later in this batch.
    assert tiny.sweeps(start, mixed) == tiny._sweeps_uncached(start, mixed)
    assert len(tiny._sweep_memo) == 2


def test_cached_and_uncached_commands_agree(scene):
    # Synthetic requests exercise command decisions without historical station logs.
    start = np.array([0., 40., 30., 60., 10., 5., 0.])
    requests = [start + np.sin(i / 4) * np.array([.2, .1, -.1, .1, .2, -.1, .3])
                for i in range(12)]
    records = []
    for capacity in (128, 0):
        m = MasterPoseMapper(scene, 'right', replace(MappingParams(), sweep_cache_capacity=capacity))
        q_now = start.copy()
        previous = start.copy()
        sequence = []
        for request in requests:
            q_now = advance(m, q_now, previous)
            result = call_request(m, request, previous, q_now)
            assert result.hold or result.sweep_m <= m.params.sweep_limit_m
            if result.hold:
                assert result.hold_reason in ('unreachable', 'sweep')
            else:
                assert result.errors['position_m'] <= result.best_hand_errors['position_m'] + m.params.delta_position_m + 1e-12
                assert result.errors['orientation_rad'] <= result.best_hand_errors['orientation_rad'] + m.params.delta_orientation_rad + 1e-12
                assert result.errors['position_m'] <= m.params.epsilon_position_m + 1e-12
                assert result.errors['orientation_rad'] <= m.params.epsilon_orientation_rad + 1e-12
            sequence.append((result.q_deg, result.hold, result.hold_reason, result.sweep_m))
            previous = result.q_deg
        records.append(sequence)
    assert records[0] == records[1]


def test_feasible_straight_branch_switch_is_automatic(scene):
    m = MasterPoseMapper(scene, 'right')
    q_now = np.array([0., 70., 100., 0., -10., 0., 0.])
    target = np.array([0., 70., -80., 0., 170., 0., 0.])
    previous = call_request(m, q_now, q_now, q_now).q_deg
    result = call_request(m, target, previous, q_now)
    assert result.reachable and not result.hold and result.branch_switch
    np.testing.assert_allclose(result.q_deg, target, atol=math.degrees(1e-9), rtol=0)
    assert result.duration_s == pytest.approx(4.5)
    dense = m.sweeps(np.deg2rad(q_now), np.deg2rad(target)[None], step_s=.001, force_dense=True)[0]
    assert dense['sampled_sweep_m'] == pytest.approx(.0105, abs=1e-12)
    assert dense['sampled_sweep_m'] <= result.sweep_m <= m.params.sweep_limit_m


def test_reachable_pose_with_no_accepted_path_holds_for_sweep(scene):
    m = MasterPoseMapper(scene, 'right')
    q_now = np.array([0., 40., 30., 60., 10., 5., 0.])
    request = np.array([90., -20., -120., 120., 110., -30., 40.])
    result = call_request(m, request, q_now, q_now, full_search=True)
    assert result.hold and not result.reachable and result.hold_reason == 'sweep'
    np.testing.assert_array_equal(result.q_deg, q_now)
    assert result.candidate_errors['position_m'] <= .002
    assert result.candidate_errors['orientation_rad'] <= math.radians(1)
    assert result.candidate_sweep_m > .02 and 'SWEEP' in result.reason
    cold = call_request(MasterPoseMapper(scene, 'right'), request, q_now, q_now)
    assert cold.hold == result.hold and cold.hold_reason == result.hold_reason
    assert cold.iterations > 0


def test_q_now_changes_feasibility_independently_of_previous_output(scene):
    m = MasterPoseMapper(scene, 'right')
    request = np.array([0., 40., -70., -40., 155., 10., 15.])
    far_now = np.array([0., 40., 110., 40., -25., 10., 15.])
    near = call_request(m, request, request, request)
    far = call_request(m, request, request, far_now)
    assert near.reachable
    assert far.q_now_deg == tuple(far_now)
    assert far.hold or np.max(abs(np.deg2rad(far.q_deg) - np.deg2rad(request))) > 1e-9
    if far.reachable:
        assert far.sweep_m <= .02


@pytest.mark.parametrize('side', ['left', 'right'])
def test_continuous_copy_and_exact_certificate(scene, side, record_property):
    m = MasterPoseMapper(scene, side)
    previous = np.array([0., 40., 30., 60., 10., 5., 0.])
    q_now = previous.copy()
    previous = call_request(m, previous, previous, q_now).q_deg
    maximum = 0.
    for t in np.linspace(0., 1., 12):
        request = np.array([2*t, 40+t, 30+3*t, 60-2*t, 10-t, 5+t, 2*t])
        result = call_request(m, request, previous, q_now)
        reference = call_request(m, request, previous, q_now, full_search=True)
        assert result.reachable == reference.reachable and result.hold == reference.hold
        assert result.certificate == 'exact' and result.seeds_count == 0
        difference = np.max(abs(np.deg2rad(result.q_deg) - np.deg2rad(reference.q_deg)))
        maximum = max(maximum, float(difference))
        assert difference <= 1e-9
        np.testing.assert_allclose(result.q_deg, request, atol=math.degrees(1e-9), rtol=0)
        q_now = advance(m, q_now, result.q_deg)
        previous = result.q_deg
    record_property('max_exact_certificate_difference_rad', maximum)


def test_coaxial_master_split_and_repeated_near_singular_motion(scene, record_property):
    m = MasterPoseMapper(scene, 'right')
    q_now = np.array([0., 0., 30., 40., 10., 0., 0.])
    previous = call_request(m, q_now, q_now, q_now).q_deg
    single = np.array([.1, 0., 30., 40., 10., 0., 0.])
    first = call_request(m, single, previous, q_now)
    reference = call_request(m, single, previous, q_now, full_search=True)
    assert first.certificate == 'exact'
    np.testing.assert_allclose(np.deg2rad(first.q_deg), np.deg2rad(reference.q_deg), atol=1e-9, rtol=0)
    maximum = 0.
    q_now = np.array([0., 0., 31., 40., 10., 0., 0.])
    previous = call_request(m, q_now, q_now, q_now).q_deg
    for phase in np.linspace(0., 6 * math.pi, 121):
        request = np.array([2*math.sin(phase), .05*math.sin(2*phase), 30+math.cos(phase), 40., 10., 0., 0.])
        result = call_request(m, request, previous, q_now)
        reference = call_request(m, request, previous, q_now, full_search=True)
        assert result.reachable and reference.reachable
        difference = float(np.max(abs(np.deg2rad(result.q_deg) - np.deg2rad(reference.q_deg))))
        maximum = max(maximum, difference)
        assert difference <= 1e-9
        np.testing.assert_allclose(np.array(result.q_deg)[[0, 2]], request[[0, 2]], atol=math.degrees(1e-9), rtol=0)
        previous = result.q_deg
        q_now = advance(m, q_now, previous)
    record_property('coaxial_max_difference_rad', maximum)


def test_j7_370_counterexample_is_an_accepted_bounded_approximation(scene, record_property):
    m = MasterPoseMapper(scene, 'right')
    previous = np.array([0., 40., 30., 60., 10., 5., 0.])
    q_now = previous.copy()
    request = np.array([.1, 40.1, 30.1, 60.1, 10.1, 5.1, 370.1])
    result = call_request(m, request, previous, q_now)
    reference = call_request(m, request, previous, q_now, full_search=True)
    # Warm certificates are compared with the accepted physical bounds.
    assert result.certificate in ('warm', 'full')
    assert result.reachable == reference.reachable and result.hold == reference.hold
    assert result.errors['position_m'] <= reference.best_hand_errors['position_m'] + m.params.delta_position_m
    assert result.errors['orientation_rad'] <= reference.best_hand_errors['orientation_rad'] + m.params.delta_orientation_rad
    assert result.errors['elbow_m'] <= reference.errors['elbow_m'] + m.params.elbow_certificate_m
    assert result.sweep_m <= m.params.sweep_limit_m
    assert result.errors['position_m'] <= result.best_hand_errors['position_m'] + m.params.delta_position_m + 1e-12
    assert result.errors['orientation_rad'] <= result.best_hand_errors['orientation_rad'] + m.params.delta_orientation_rad + 1e-12
    assert result.errors['position_m'] <= m.params.epsilon_position_m + 1e-12
    assert result.errors['orientation_rad'] <= m.params.epsilon_orientation_rad + 1e-12
    record_property('warm_joint_difference_rad', float(np.max(abs(np.deg2rad(result.q_deg) - np.deg2rad(reference.q_deg)))))


def test_iteration_limits_do_not_disable_solver_progress(scene):
    m = MasterPoseMapper(scene, 'right')
    q = np.deg2rad([0., 40., 30., 60., 10., 5., 0.])
    goal = q.copy()
    goal[0] += .1
    target, elbow = m.frames(goal[None])
    initial, _ = m._residual(q[None], target[0], elbow[0])
    one, count = m._hand_solve(q[None], target[0], elbow[0], iterations=1)
    one_error, _ = m._residual(one, target[0], elbow[0])
    assert count == 1 and np.sum(one_error ** 2) < np.sum(initial ** 2)
    full, count = m._hand_solve(q[None], target[0], elbow[0])
    full_error, elbow_error = m._residual(full, target[0], elbow[0])
    assert count > 0 and np.sum(full_error ** 2) < np.sum(initial ** 2)
    refined, count = m._elbow_solve(full, target[0], elbow[0], full_error[0], iterations=1)
    _, refined_elbow = m._residual(refined, target[0], elbow[0])
    assert count == 1 and np.linalg.norm(refined_elbow) < np.linalg.norm(elbow_error)


def test_nonidentity_equivalent_seed_group(scene):
    m = MasterPoseMapper(scene, 'right')
    q = np.deg2rad([10, 35, 110, 40, -25, 10, 15.])
    equivalent = q.copy()
    equivalent[[2, 4]] += np.pi
    equivalent[3] *= -1
    poses, elbows = m.frames(np.array([q, equivalent]))
    assert np.linalg.norm(poses[0, :3, 3] - poses[1, :3, 3]) == pytest.approx(.014364846019678, abs=1e-12)
    assert np.linalg.norm(elbows[0] - elbows[1]) == pytest.approx(.021, abs=1e-12)
    assert any(name.startswith('equivalent_') for name in m._seeds(q, None)[0])


@pytest.mark.parametrize('turns', [-3, -1, 1, 4])
def test_j7_full_turns(scene, turns):
    m = MasterPoseMapper(scene, 'right')
    request = np.array([0., 40., 30., 60., 10., 5., 10.])
    q_now = request.copy()
    ordinary = call_request(m, request, None, q_now)
    request[6] += turns * 360
    turned = call_request(m, request, None, q_now)
    assert ordinary.reachable and turned.reachable
    np.testing.assert_allclose(np.deg2rad(turned.q_deg), np.deg2rad(ordinary.q_deg), atol=1e-9, rtol=0)


def test_unreachable_hold_first_cycle_and_recovery(scene):
    m = MasterPoseMapper(scene, 'right')
    q_now = np.array([0., 40., 30., 60., 10., 0., 0.])
    prior = call_request(m, q_now, q_now, q_now)
    request = np.array([0., 0., 0., 0., 0., 90., 0.])
    held = call_request(m, request, prior.q_deg, q_now, full_search=True)
    assert held.hold and not held.reachable and held.hold_reason == 'unreachable'
    np.testing.assert_array_equal(held.q_deg, prior.q_deg)
    assert 'J6:upper' in held.active_limits and 'UNREACHABLE' in held.reason
    first = call_request(m, request, None, q_now)
    assert first.q_deg is None and first.hold and first.hold_reason == 'unreachable'
    assert call_request(m, q_now, held.q_deg, q_now).reachable
