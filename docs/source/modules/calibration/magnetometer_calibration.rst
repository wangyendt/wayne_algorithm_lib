磁力计校准 (magnetometer_calibration)
=====================================

.. automodule:: pywayne.calibration.magnetometer_calibration

该类用于对磁力计数据进行校准，具体步骤包括：

- 使用 VQF 批处理传感器数据，并批量构造、累积校准矩阵 P_k。
- 根据 P_k 的最小特征向量计算软铁矩阵 Sm、硬铁偏差向量 h，以及初始磁场 m_i0。
- ``process`` 只对最终累积矩阵求一次特征向量，避免逐样本重复特征分解。

.. autoclass:: pywayne.calibration.magnetometer_calibration.MagnetometerCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

**示例**::

   >>> from pywayne.calibration.magnetometer_calibration import MagnetometerCalibrator
   >>> import numpy as np
   >>> # 假设 ts, acc, gyro, mag 为已知的传感器数据数组
   >>> calibrator = MagnetometerCalibrator()
   >>> Sm, h = calibrator.process(ts, acc, gyro, mag)
   >>> print("软铁矩阵:", Sm)
   >>> print("硬铁偏差:", h)
