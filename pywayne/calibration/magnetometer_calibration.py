# author: wangye(Wayne)
# license: Apache Licence
# file: magnetometer_calibration.py
# time: 2023-12-05-19:48:43
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import numpy as np
from vqf import VQF
import qmt
from typing import Tuple, Generator, List


class MagnetometerCalibrator:
    """
    This class is designed for calibrating magnetometers. It employs sensor data from accelerometers,
    gyroscopes, and magnetometers to compute calibration parameters.

    Attributes:
        method (str): The method used for calibration, with 'close_form' as the default.
    """

    def __init__(self, method: str = 'close_form'):
        """
        Constructor for the MagnetometerCalibrator class.

        Args:
            method (str): The calibration method to be used. Default is 'close_form'.
        """
        self.method = method

    def _calc_pk(self, ts: np.ndarray, acc: np.ndarray, gyro: np.ndarray, mag: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Calculates the calibration matrix P_k for the magnetometer using sensor data.

        Args:
            ts (np.ndarray): Timestamps of the sensor readings.
            acc (np.ndarray): Accelerometer data.
            gyro (np.ndarray): Gyroscope data.
            mag (np.ndarray): Magnetometer data.

        Yields:
            tuple: A tuple containing the minimum eigenvector x_min and the matrix P_k_2 at each iteration.
        """
        # mag = mag / np.linalg.norm(mag, axis=1, keepdims=True)
        dt = np.mean(np.diff(ts))
        N = ts.shape[0]
        vqf = VQF(gyrTs=dt)
        vqf.setTauAcc(3.0)
        C_i0_b = []
        q0 = np.array([1, 0, 0, 0], dtype=float)
        P_k_2 = np.zeros((15, 15))
        for i in range(N):
            vqf.updateGyr(gyro[i])
            vqf.updateAcc(acc[i])
            q_k = vqf.getQuat6D()
            if i == 0:
                q0 = q_k
            C_i0_b.append(qmt.quatToRotMat(
                qmt.qmult(qmt.qinv(q0), q_k)
            ))
            y_m = mag[i].reshape((3, 1))
            p_k = np.c_[
                np.kron(y_m.T, C_i0_b[-1]),
                -C_i0_b[-1],
                -np.eye(3)
            ]  # (3,15)
            P_k_2 = P_k_2 + p_k.T @ p_k
            e_val, e_vec = np.linalg.eig(P_k_2)  # sp.linalg.eigh(P_k_2)
            e_val = np.abs(e_val)
            e_vec_min = e_vec[:, e_val.argmin()]
            x_min = e_vec_min / np.linalg.norm(e_vec_min[-3:])
            yield x_min if x_min[0] > 0 else -x_min, P_k_2

    def _calc_S_h(self, x_min: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the soft-iron matrix (Sm), hard-iron offset vector (h), and initial magnetic field (m_i0)
        from the minimum eigenvector.

        Args:
            x_min (np.ndarray): Minimum eigenvector obtained from the P_k matrix.

        Returns:
            tuple: A tuple containing the soft-iron matrix (Sm), hard-iron vector (h), and initial magnetic field (m_i0).
        """
        # x_min: (15,)
        vec_Sm_inv = x_min[:9]
        Sm_inv = vec_Sm_inv.reshape((3, 3)).T
        Sm = np.linalg.pinv(Sm_inv)
        h = Sm @ x_min[9:12]
        m_i0 = x_min[-3:]
        return np.real(Sm), np.real(h), np.real(m_i0)

    def process(self, ts: np.ndarray, acc: np.ndarray, gyro: np.ndarray, mag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes the sensor data to compute the calibration parameters for the magnetometer.

        Args:
            ts (np.ndarray): Timestamps of the sensor readings.
            acc (np.ndarray): Accelerometer data.
            gyro (np.ndarray): Gyroscope data.
            mag (np.ndarray): Magnetometer data.

        Returns:
            tuple: A tuple containing the soft-iron matrix (Sm) and the hard-iron vector (h).
        """
        acc = np.ascontiguousarray(acc)
        gyro = np.ascontiguousarray(gyro)
        mag = np.ascontiguousarray(mag)

        Sm = np.eye(3)
        h = np.zeros(3, )

        for x_min, P_k_2 in self._calc_pk(ts, acc, gyro, mag):
            Sm, h, m_i0 = self._calc_S_h(x_min)

        print(f'{Sm=}')
        print(f'{h=}')

        return Sm, h
