"""Public MIT HandCore functions and a project-authored URDF slot adapter.

Copied functions: linker-bot/linkerhand-telop-ros2
commit 12300b1e45b263a253cc88fdb7635f95587d0741,
src/linkerhand_retarget/linkerhand_retarget/linkerhand/handcore.py:236-265
and linkerhand/utils.py:147-152. Only methods are exposed as free functions.
The slot adapter is ours, derived from public command indices and URDF names.

MIT License

Copyright (c) 2024 Linker Robotics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from __future__ import annotations
import numpy as np

RIGHT_SLOT_FROM_JOINT = {'thumb_cmc_pitch': 18,
 'index_mcp_pitch': 1,
 'middle_mcp_pitch': 9,
 'ring_mcp_pitch': 13,
 'pinky_mcp_pitch': 5,
 'thumb_cmc_roll': 16,
 'index_mcp_roll': 0,
 'middle_mcp_roll': 8,
 'ring_mcp_roll': 12,
 'pinky_mcp_roll': 4,
 'thumb_cmc_yaw': 17,
 'thumb_mcp': 19,
 'index_pip': 2,
 'middle_pip': 10,
 'ring_pip': 14,
 'pinky_pip': 6,
 'thumb_ip': 20,
 'index_dip': 3,
 'middle_dip': 11,
 'ring_dip': 15,
 'pinky_dip': 7}

LEFT_SLOT_FROM_JOINT = dict(RIGHT_SLOT_FROM_JOINT)

def logical_qpos_from_dex(robot_qpos: np.ndarray, joint_names: List[str], hand: str) -> np.ndarray:
    """Place supplied joints into command-source order; unknown names are ignored."""
    slots = RIGHT_SLOT_FROM_JOINT if hand == "right" else LEFT_SLOT_FROM_JOINT
    result = np.zeros(21, dtype=np.float64)
    for name, value in zip(joint_names, robot_qpos):
        if name in slots:
            result[slots[name]] = float(value)
    # Public thumb IP mimic; other supplied passive slots retain their values.
    result[20] = (1.03 if hand == "right" else 1.02) * result[19]
    return result

def trans_to_motor_left(self, temp_l):
        jointpositions_l = [255.0] * self.hand_numjoints_l
        for i in range(self.hand_numjoints_l):
            if self.sourcedataindex_l[i] is not None:
                val_l = temp_l[self.sourcedataindex_l[i]]
                val_l = is_within_range(val_l,
                                        self.hand_lower_limits_l[self.urdfdataindex_l[i]],
                                        self.hand_upper_limits_l[self.urdfdataindex_l[i]])
                jointpositions_l[i] = int(scale_value(val_l,
                                                      self.hand_lower_limits_l[self.urdfdataindex_l[i]],
                                                      self.hand_upper_limits_l[self.urdfdataindex_l[i]],
                                                      self.dataminvalue_l[i],
                                                      self.datamaxvalue_l[i]))
        return jointpositions_l

def trans_to_motor_right(self, temp_r):
        jointpositions_r = [255.0] * self.hand_numjoints_r
        for i in range(self.hand_numjoints_r):
            if self.sourcedataindex_r[i] is not None:
                val_r = temp_r[self.sourcedataindex_r[i]]
                val_r = is_within_range(val_r,
                                        self.hand_lower_limits_r[self.urdfdataindex_r[i]],
                                        self.hand_upper_limits_r[self.urdfdataindex_r[i]])

                jointpositions_r[i] = int(scale_value(val_r,
                                                      self.hand_lower_limits_r[self.urdfdataindex_r[i]],
                                                      self.hand_upper_limits_r[self.urdfdataindex_r[i]],
                                                      self.dataminvalue_r[i],
                                                      self.datamaxvalue_r[i]))
        return jointpositions_r

def scale_value(original_value, a_min, a_max, b_min, b_max):
    return (original_value - a_min) * (b_max - b_min) / (a_max - a_min) + b_min

def is_within_range(value, min_value, max_value):
    return min(max_value, max(min_value, value))
