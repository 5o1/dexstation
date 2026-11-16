"""Composition of installed dex-retargeting objectives; no in-house optimizer.

Direction errors use native VectorOptimizer FK/Jacobians/autograd/NLopt. Native
DexPilotOptimizer supplies fingertip projection, weights and hysteresis.
"""
import json
from types import SimpleNamespace, MethodType
import xml.etree.ElementTree as ET
import numpy as np
from dexstation.inputs.retarget_anatomy import ASSETS, FINGERS, PAIRS, anatomical_targets, load_config, robot_geometry
from dexstation.inputs.retarget_motor_clone import RIGHT_SLOT_FROM_JOINT, logical_qpos_from_dex, trans_to_motor_left, trans_to_motor_right


def motor_context(side):
    if side not in ('left', 'right'):
        raise ValueError('invalid hand side')
    mapping = json.loads((ASSETS/'motor_mapping.json').read_text())
    joints = [j for j in ET.parse(ASSETS/f'g20_{side}.urdf').getroot().findall('joint') if j.get('type') != 'fixed']
    limits = [(float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))) for j in joints]
    # Original config uses the YAML string "None"; normalize only that spelling.
    def values(prefix):
        return [None if v == 'None' else v for v in mapping[f'{prefix}_{side}_g20']]
    suffix = 'l' if side == 'left' else 'r'
    return SimpleNamespace(**{f'hand_numjoints_{suffix}': 20,
        f'sourcedataindex_{suffix}': values('commandsourcedataindex'),
        f'urdfdataindex_{suffix}': values('urdfdataindex'),
        f'dataminvalue_{suffix}': values('commandlower'),
        f'datamaxvalue_{suffix}': values('commandupper'),
        f'hand_lower_limits_{suffix}': [v[0] for v in limits],
        f'hand_upper_limits_{suffix}': [v[1] for v in limits]})


def motor_values(qpos, joint_names, side):
    qpos = np.asarray(qpos)
    if not np.isfinite(qpos).all():
        raise ValueError('nonfinite retarget output')
    logical = logical_qpos_from_dex(qpos, joint_names, side)
    fn = trans_to_motor_left if side == 'left' else trans_to_motor_right
    result = fn(motor_context(side), logical)
    if len(result) != 20 or not all(0 <= v <= 255 for v in result):
        raise ValueError('invalid motor command')
    return result


def command_fields(positions):
    """Exact original handretarget JointState payload, timestamp set by ROS node."""
    if len(positions) != 20 or not np.isfinite(positions).all() or any(not 0 <= v <= 255 for v in positions):
        raise ValueError('expected 20 motor values in [0,255]')
    return dict(name=[f'joint_{i}' for i in range(20)], position=[float(v) for v in positions],
                velocity=[255.]*20, effort=[0.]*20)


def filter_motors(raw, previous, dt, config):
    """Only temporal output stage: continuous-time RC plus rate limit in motors/s."""
    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(raw).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('invalid output or sample interval')
    if previous is None:
        return raw.copy()
    tau = config['time_constant_s']
    rate = config['max_motor_units_per_s']
    if tau < 0 or rate <= 0:
        raise ValueError('invalid output filter configuration')
    alpha = -np.expm1(-dt/tau) if tau else 1.
    return previous+np.clip(alpha*(raw-previous), -rate*dt, rate*dt)


def combine_objectives(direction, pinch, weight):
    """Sum native scalar objectives and gradients without changing their solver."""
    def objective(x, grad):
        if grad.size:
            a, b = np.empty_like(grad), np.empty_like(grad)
            value = direction(x, a)+weight*pinch(x, b)
            grad[:] = a+weight*b
            return value
        return direction(x, grad)+weight*pinch(x, grad)
    return objective


class AnatomicalRetargeter:
    def __init__(self, side, config=None):
        # Imports remain lazy: anatomy/recording/unit tests need only NumPy.
        import torch
        from dex_retargeting.retargeting_config import RetargetingConfig
        from dex_retargeting.optimizer import DexPilotOptimizer
        self.side, self.config = side, config or load_config()
        path = ASSETS/f'g20_{side}.urdf'
        joints = [j.get('name') for j in ET.parse(path).getroot().findall('joint') if j.get('type') != 'fixed' and j.find('mimic') is None]
        seq = RetargetingConfig.from_dict(dict(type='vector', urdf_path=str(path),
            target_joint_names=joints, target_origin_link_names=[p.robot_origin for p in PAIRS],
            target_task_link_names=[p.robot_task for p in PAIRS],
            target_link_human_indices=np.zeros((2, len(PAIRS)), dtype=int),
            scaling_factor=1., normal_delta=0., low_pass_alpha=-1.)).build()
        vector = seq.optimizer
        lengths, robot_length = robot_geometry(side)
        weights = torch.tensor([self.config['distal_weight'] if p.distal else 1. for p in PAIRS])
        lengths_t = torch.tensor(lengths)
        beta = self.config['direction_huber_delta']
        # For rigid segments |v_robot|=L: |v_robot-L*u_human|/L is
        # exactly the chord distance between unit directions. Weight distal 1/4.
        def direction_loss(distances, _zeros):
            scaled = distances/lengths_t
            loss = torch.nn.functional.smooth_l1_loss(scaled, torch.zeros_like(scaled), beta=beta, reduction='none')
            return (loss*weights).sum()/weights.sum()
        vector.huber_loss = direction_loss
        pilot = DexPilotOptimizer(vector.robot, joints, [f+'_tip' for f in FINGERS], 'hand_base_link',
            norm_delta=0., scaling=1.,
            project_dist=self.config['pinch_project_hand_fraction']*robot_length,
            escape_dist=self.config['pinch_escape_hand_fraction']*robot_length,
            eta1=self.config['pinch_contact_robot_fraction']*robot_length,
            eta2=self.config['pinch_secondary_robot_fraction']*robot_length)
        if vector.adaptor is not None:
            pilot.set_kinematic_adaptor(vector.adaptor)
        # Dimensionless pinch residuals, keep all native projection weights.
        pilot.huber_loss = lambda a, b: torch.nn.functional.smooth_l1_loss(a/robot_length, b/robot_length, beta=.1, reduction='none')
        original = vector.get_objective_function
        def factory(_self, target, fixed, last):
            return combine_objectives(original(target[:len(PAIRS)], fixed, last),
                pilot.get_objective_function(target[len(PAIRS):], fixed, last), self.config['dexpilot_weight'])
        vector.get_objective_function = MethodType(factory, vector)
        vector.opt.set_maxeval(self.config['maxeval'])
        # Open URDF zero pose, within limits; no fixed JSON/posture targets.
        seq.set_qpos(np.zeros(vector.robot.dof))
        self.seq, self.pilot = seq, pilot
        self.previous = None

    @property
    def joint_names(self):
        return self.seq.joint_names

    def solve(self, nodes):
        reference = anatomical_targets(nodes, self.side, self.config)
        target = np.vstack([reference['vectors'], reference['pilot_vectors']]).astype(np.float32)
        qpos = self.seq.retarget(target)
        if self.seq.optimizer.opt.last_optimize_result() < 0:
            raise RuntimeError('dex-retargeting optimizer failed')
        if not np.isfinite(qpos).all():
            raise ValueError('optimizer returned nonfinite qpos')
        return qpos, reference

    def step(self, nodes, dt):
        qpos, reference = self.solve(nodes)
        raw = motor_values(qpos, self.joint_names, self.side)
        self.previous = filter_motors(raw, self.previous, dt, self.config['output_filter'])
        return command_fields(np.rint(self.previous).tolist()), qpos, reference
