# author: wangye(Wayne)
# license: Apache Licence
# file: tools.py
# time: 2023-08-09-11:12:10
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import numpy as np
from typing import *


def quaternion_decompose(quaternion: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    quaternion_decompose

    This function is used to decompose a quaternion into a rotation angle around
    vertical axis (heading) and a rotation angle around horizontal axis (inclination).

    .. math::
        \\theta = 2\\arccos{q_w}

    .. math::
        \\theta_h = 2\\arctan{|\\frac{q_z}{q_w}|}

    .. math::
        \\theta_i = 2\\arccos{\\sqrt{q_w^2+q_z^2}}

    Parameters
    ----------
    quaternion : np.ndarray
        shape: (4,) or (4, 1) or (N, 4)

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
    >>> a = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0])
    >>> print(quaternion_decompose(a))
    (array([1.57079633]), array([0.]), array([1.57079633]))
    """
    if type(quaternion) is list:
        quaternion = np.array(quaternion)
    if len(quaternion.shape) == 1:
        assert quaternion.shape[0] == 4
        quaternion = quaternion[None, :]
    else:
        assert len(quaternion.shape) == 2 and quaternion.shape[1] == 4
    assert np.all(np.linalg.norm(quaternion, axis=1) == 1)
    angle_all = 2 * np.arccos(np.abs(quaternion[:, 0]))
    angle_heading = 2 * np.arctan2(np.abs(quaternion[:, 3]), np.abs(quaternion[:, 0]))
    angle_inclination = 2 * np.arccos(np.sqrt(quaternion[:, 0] ** 2 + quaternion[:, 3] ** 2))
    return angle_all, angle_heading, angle_inclination
