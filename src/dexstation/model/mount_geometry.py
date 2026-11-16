"""Derive J1 corrections from arm mount orientations (wxyz quaternions)."""

import numpy as np


def _rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError('mount orientation needs four finite, non-zero quaternion values')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def j1_offset_deg(from_wxyz, to_wxyz, *, tilt_tolerance_deg=0.01) -> float:
    """Return phi with R(to) = R(from) @ Rz(phi), wrapped to (-180, 180].

    Normalize wxyz inputs. Reject relative z-axis tilt above tilt_tolerance_deg.
    """
    if not np.isfinite(tilt_tolerance_deg) or tilt_tolerance_deg < 0:
        raise ValueError('tilt_tolerance_deg must be finite and non-negative')
    rel = _rotation(from_wxyz).T @ _rotation(to_wxyz)
    tilt = float(np.degrees(np.arccos(np.clip(rel[2, 2], -1., 1.))))
    if tilt > tilt_tolerance_deg:
        raise ValueError(f'mount z-axis tilt {tilt:.6f} deg exceeds {tilt_tolerance_deg} deg; recalibrate')
    phi = float(np.degrees(np.arctan2(rel[1, 0], rel[0, 0])))
    wrapped = (phi + 180.) % 360. - 180.
    return 180. if wrapped == -180. else wrapped
