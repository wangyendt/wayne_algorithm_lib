校准工具 (calibration)
========================

本模块提供了用于传感器校准的工具函数与类，主要针对磁力计校准问题。通过利用加速度计、陀螺仪和磁力计的数据，模块旨在帮助用户计算并校正传感器的软铁（soft-iron）效应和硬铁（hard-iron）偏差。

MagnetometerCalibrator 类
---------------------------

该类用于对磁力计数据进行校准，具体步骤包括：

- 使用传感器数据逐步累积计算校准矩阵 P_k。
- 根据 P_k 的最小特征向量计算软铁矩阵 Sm、硬铁偏差向量 h，以及初始磁场 m_i0。
- 最终，通过 process 方法整理数据并输出校准参数。

**主要方法**:

- __init__(self, method: str = 'close_form')

  构造函数，初始化校准器，并指定校准方法（默认使用 close_form 方法）。

- _calc_pk(self, ts: np.ndarray, acc: np.ndarray, gyro: np.ndarray, mag: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]

  内部方法，根据传感器数据逐步计算校准矩阵 P_k，并在每一步生成当前的最小特征向量和累积的 P_k 矩阵。

- _calc_S_h(self, x_min: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

  内部方法，根据 _calc_pk 得到的最小特征向量计算软铁矩阵 Sm、硬铁偏差向量 h 和初始磁场 m_i0。

- process(self, ts: np.ndarray, acc: np.ndarray, gyro: np.ndarray, mag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]

  主接口方法，接收时间戳、加速度计、陀螺仪和磁力计数据，经过内部计算后输出校准参数（软铁矩阵 Sm 和硬铁偏差 h）。

**示例**::

   >>> from pywayne.calibration.magnetometer_calibration import MagnetometerCalibrator
   >>> import numpy as np
   >>> # 假设 ts, acc, gyro, mag 为已知的传感器数据数组
   >>> calibrator = MagnetometerCalibrator()
   >>> Sm, h = calibrator.process(ts, acc, gyro, mag)
   >>> print("软铁矩阵:", Sm)
   >>> print("硬铁偏差:", h)

预留内容 - temporal_calibration
------------------------------------

模块中还包含 temporal_calibration.py 文件，用于未来扩展时空校准功能，目前该部分内容尚处于预留阶段。 