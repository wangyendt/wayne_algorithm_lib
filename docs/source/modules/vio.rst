视觉惯性里程计工具 (VIO)
=============================

本模块提供了一些用于视觉惯性里程计 (Visual Inertial Odometry, VIO) 数据处理与分析的工具函数，主要包含以下功能：

- 将 SE(3) 矩阵转换为 pose 表示
- 将 pose 表示转换为 SE(3) 矩阵
- 使用 3D 可视化工具展示 pose 信息

函数说明
---------

SE3_to_pose
~~~~~~~~~~~
该函数用于将 SE(3) 矩阵转换为 pose 表示。转换后的 pose 包含平移向量 (tx, ty, tz) 以及表示旋转的四元数 (qw, qx, qy, qz)。

**参数**:

- SE3_mat (np.ndarray): 输入的 SE(3) 矩阵。可以是单个 4x4 矩阵或者由多个 SE(3) 组成的数组。

**返回值**:

- np.ndarray: 包含平移和四元数的 pose 表示数组。

**示例**::

   >>> import numpy as np
   >>> from pywayne.vio.tools import SE3_to_pose
   >>> SE3 = np.eye(4)
   >>> pose = SE3_to_pose(SE3)
   >>> print(pose)


pose_to_SE3
~~~~~~~~~~~
该函数用于将 pose 表示转换为 SE(3) 矩阵，用于描述刚体变换。

**参数**:

- pose_mat (np.ndarray): 输入的 pose 数组，形状应为 (7,) 或 (N,7)，包含平移和四元数信息。

**返回值**:

- np.ndarray: 转换后的 SE(3) 矩阵数组。

**示例**::

   >>> from pywayne.vio.tools import pose_to_SE3
   >>> SE3_recon = pose_to_SE3(pose)
   >>> print(SE3_recon)


visualize_pose
~~~~~~~~~~~~~~
该函数用于可视化一组 pose 数据。通过 matplotlib 的 3D 绘图功能，将每个 pose 的平移信息以点的形式显示，并根据四元数计算旋转矩阵来绘制对应的方向箭头。

**参数**:

- poses: 一组 pose 数据，每个 pose 的格式为 (tx, ty, tz, qw, qx, qy, qz)。
- arrow_length_ratio (float): 箭头长度相对于数据范围的比例，默认值为 0.1。

**示例**::

   >>> from pywayne.vio.tools import visualize_pose, SE3_to_pose
   >>> import numpy as np
   >>> # 生成一些随机的 SE(3) 矩阵（实际应用中需保证旋转矩阵正交性）
   >>> SE3_array = np.random.randn(5, 4, 4)
   >>> poses = SE3_to_pose(SE3_array)
   >>> visualize_pose(poses)


模块扩展建议
--------------

如果未来需要更多视觉惯性里程计算法与处理功能，可以考虑在 VIO 模块中增加更多子模块，如状态估计、滤波算法以及数据同步与校准方法，进一步支持多传感器融合算法的开发与调试。 