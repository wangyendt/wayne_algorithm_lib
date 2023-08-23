# author: wangye(Wayne)
# license: Apache Licence
# file: tools.py
# time: 2023-08-09-11:12:10
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import numpy as np
import qmt
from typing import *


def quaternion_decompose(quaternion: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    quaternion_decompose

    This function is used to decompose a quaternion into a rotation angle around
    vertical axis (heading) and a rotation angle around horizontal axis (inclination).

    .. math::

        \begin{array}{rcl}
            \theta &=& 2\arccos{q_w} \\
            \theta_h &=& 2\arctan{|\frac{q_z}{q_w}|} \\
            \theta_i &=& 2\arccos{\sqrt{q_w^2+q_z^2}}
        \end{array}

    Parameters
    ----------
    quaternion : np.ndarray
        shape: (4,) or (1, 4) or (N, 4)

    Returns
    -------
    angle : np.ndarray
        the angle corresponds to the quaternion
    angle_heading : np.ndarray
        the angle around vertical axis
    angle_inclination : np.ndarray
        the angle around horizontal axis

    Examples
    --------
    >>> from pywayne.ahrs.tools import quaternion_decompose
    >>> q = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0])
    >>> print(quaternion_decompose(q))
    (array([1.57079633]), array([0.]), array([1.57079633]))
    """
    if type(quaternion) is list:
        quaternion = np.array(quaternion)
    if len(quaternion.shape) == 1:
        assert quaternion.shape[0] == 4
        quaternion = quaternion[None, :]
    else:
        assert len(quaternion.shape) == 2 and quaternion.shape[1] == 4
    assert np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-6)
    angle_all = 2 * np.arccos(np.abs(quaternion[:, 0]))
    angle_heading = 2 * np.arctan2(np.abs(quaternion[:, 3]), np.abs(quaternion[:, 0]))
    angle_inclination = 2 * np.arccos(np.sqrt(quaternion[:, 0] ** 2 + quaternion[:, 3] ** 2))
    return angle_all, angle_heading, angle_inclination


def quaternion_roll_pitch_compensate(quaternion: np.ndarray) -> np.ndarray:
    r"""
    quaternion_roll_pitch_compensate

    This function is used to calculate compensation rotation to make the orientation's pitch & roll to zero.

    Here is how to conduct the formula,

    .. math::

        \begin{array}{rcl}
            \textbf{R}_{cur}        &=& (\textbf{R}_z\textbf{R}_y\textbf{R}_x)\textbf{R}_{identity} \\
                                    &=& \textbf{R}_z\textbf{R}_y\textbf{R}_x \\
            \textbf{R}_{corr}       &=& \textbf{R}_y\textbf{R}_x=\textbf{R}_z^{-1}\textbf{R}_{cur} \\
            \textbf{R}_{corrected}  &=& \textbf{R}_{ori}\textbf{R}_{corr} \\
                                    &=& \textbf{R}_{ori}\textbf{R}_z^{-1}\textbf{R}_{cur}
        \end{array}


    Here is how to use the compensation rotation,

    .. math::

        \begin{array}{rcl}
            \textbf{R}_{cur}\textbf{R}_{corr}=\textbf{R}_{z}\textbf{R}_y\textbf{R}_x(\textbf{R}_y\textbf{R}_x)^{-1}=\textbf{R}_z
        \end{array}

    Parameters
    ----------
    quaternion : np.ndarray
        shape: (4,) or (1, 4) or (4, 1)

    Returns
    -------
    q_yx_inv : np.ndarray
        compensation rotation

    Examples
    --------
    >>> from pywayne.ahrs.tools import quaternion_roll_pitch_compensate
    >>> q = np.array([0.989893, -0.099295, 0.024504, -0.098242])
    >>> print(quaternion_roll_pitch_compensate(q))
    [ 0.99475519  0.10125102 -0.01442838 -0.00146859]
    """
    if type(quaternion) is list:
        quaternion = np.array(quaternion)
    quaternion = quaternion.squeeze()
    quaternion = quaternion / np.linalg.norm(quaternion)
    q_zyx = qmt.quatFromEulerAngles(qmt.eulerAngles(quaternion, 'xyz', intrinsic=False), 'xyz', intrinsic=False)
    assert np.allclose(q_zyx, quaternion, atol=1e-6)
    yaw = qmt.eulerAngles(q_zyx, 'xyz', False)[2]
    q_z = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    q_yx = qmt.qmult(qmt.qinv(q_z), q_zyx)
    q_yx_inv = qmt.qinv(q_yx)
    return q_yx_inv
