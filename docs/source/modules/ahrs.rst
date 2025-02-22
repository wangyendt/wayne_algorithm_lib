姿态估计 (AHRS)
=================

本模块提供了用于姿态估计的工具函数，主要基于四元数数学原理来解算物体的整体旋转角、航向角及倾斜角。模块中主要包含两个函数，分别用于四元数分解和姿态补偿。

quaternion_decompose 函数
-------------------------
该函数用于将给定的四元数分解为整体旋转角、航向角以及倾斜角。

**参数**:
- quaternion: np.ndarray
  可以是形状为 (4,) 的数组，或 (N, 4) 的数组（也支持列表，会自动转换为 numpy 数组）。

**返回值**:
- angle_all: 总旋转角（数组类型）。
- angle_heading: 航向角（垂直轴旋转角）。
- angle_inclination: 倾斜角（水平轴旋转角）。

**示例**::

   >>> from pywayne.ahrs.tools import quaternion_decompose
   >>> import numpy as np
   >>> q = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
   >>> print(quaternion_decompose(q))

quaternion_roll_pitch_compensate 函数
---------------------------------------
该函数用于计算补偿旋转，使得物体的俯仰和横滚归零，从而只保留航向角信息。

**参数**:
- quaternion: np.ndarray
  待处理的四元数，可以为列表或 numpy 数组，函数内部会自动归一化。

**返回值**:
- 补偿旋转四元数，表示需要施加的逆补偿旋转。

**示例**::

   >>> from pywayne.ahrs.tools import quaternion_roll_pitch_compensate
   >>> import numpy as np
   >>> q = np.array([0.989893, -0.099295, 0.024504, -0.098242])
   >>> print(quaternion_roll_pitch_compensate(q)) 