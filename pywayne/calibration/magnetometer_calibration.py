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
from typing import Tuple, Generator


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

    @staticmethod
    def _validate_inputs(
        ts: np.ndarray,
        acc: np.ndarray,
        gyro: np.ndarray,
        mag: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ts = np.asarray(ts, dtype=np.float64)
        acc = np.ascontiguousarray(acc, dtype=np.float64)
        gyro = np.ascontiguousarray(gyro, dtype=np.float64)
        mag = np.ascontiguousarray(mag, dtype=np.float64)

        if ts.ndim != 1 or ts.size < 2:
            raise ValueError("ts must be a 1D array with at least two samples")
        expected_shape = (ts.size, 3)
        for name, values in (("acc", acc), ("gyro", gyro), ("mag", mag)):
            if values.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}")
        if not all(np.all(np.isfinite(values)) for values in (ts, acc, gyro, mag)):
            raise ValueError("sensor inputs must contain only finite values")

        dt = float(np.mean(np.diff(ts)))
        if dt <= 0.0:
            raise ValueError("timestamps must have a positive mean sampling interval")
        return ts, acc, gyro, mag

    @staticmethod
    def _smallest_eigenvector(matrix: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        vector = eigenvectors[:, np.abs(eigenvalues).argmin()]
        normalizer = np.linalg.norm(vector[-3:])
        if normalizer == 0.0:
            raise ValueError("magnetometer calibration is degenerate")
        vector = vector / normalizer
        return vector if vector[0] > 0 else -vector

    @staticmethod
    def _build_pk_matrices(
        ts: np.ndarray,
        acc: np.ndarray,
        gyro: np.ndarray,
        mag: np.ndarray,
    ) -> np.ndarray:
        """Build all per-sample P_k matrices using compiled/batched operations."""
        dt = float(np.mean(np.diff(ts)))
        vqf = VQF(gyrTs=dt)
        vqf.setTauAcc(3.0)
        quaternions = vqf.updateBatch(gyro, acc)["quat6D"]

        relative_quaternions = qmt.qmult(qmt.qinv(quaternions[0]), quaternions)
        rotations = qmt.quatToRotMat(relative_quaternions)
        magnetic_blocks = np.concatenate(
            [mag[:, index, None, None] * rotations for index in range(3)],
            axis=2,
        )
        identity_blocks = np.broadcast_to(-np.eye(3), rotations.shape)
        return np.concatenate(
            [magnetic_blocks, -rotations, identity_blocks],
            axis=2,
        )

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
        ts, acc, gyro, mag = self._validate_inputs(ts, acc, gyro, mag)
        p_matrices = self._build_pk_matrices(ts, acc, gyro, mag)
        accumulated = np.zeros((15, 15), dtype=np.float64)
        for p_k in p_matrices:
            accumulated = accumulated + p_k.T @ p_k
            yield self._smallest_eigenvector(accumulated), accumulated

    def _calc_pk_final(
        self,
        ts: np.ndarray,
        acc: np.ndarray,
        gyro: np.ndarray,
        mag: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate only the final solution used by process(), without N eig calls."""
        ts, acc, gyro, mag = self._validate_inputs(ts, acc, gyro, mag)
        p_matrices = self._build_pk_matrices(ts, acc, gyro, mag)
        accumulated = np.einsum("nri,nrj->ij", p_matrices, p_matrices)
        return self._smallest_eigenvector(accumulated), accumulated

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
        x_min, _ = self._calc_pk_final(ts, acc, gyro, mag)
        Sm, h, _ = self._calc_S_h(x_min)

        print(f'{Sm=}')
        print(f'{h=}')

        return Sm, h
