VIO 工具函数 (tools)
=====================

.. automodule:: pywayne.vio.tools
   :members:
   :undoc-members:

该模块提供了一些用于视觉惯性里程计 (Visual Inertial Odometry, VIO) 数据处理与分析的工具函数，主要包含以下功能：

- 将 SE(3) 矩阵转换为 pose 表示
- 将 pose 表示转换为 SE(3) 矩阵
- 使用 3D 可视化工具展示 pose 信息

函数说明
---------

.. autofunction:: pywayne.vio.tools.SE3_to_pose
   :noindex:

.. autofunction:: pywayne.vio.tools.pose_to_SE3
   :noindex:

.. autofunction:: pywayne.vio.tools.visualize_pose
   :noindex:

**示例**::

   >>> from pywayne.vio.tools import visualize_pose, SE3_to_pose, pose_to_SE3
   >>> import numpy as np
   >>> 
   >>> # SE3 to Pose
   >>> SE3 = np.eye(4)
   >>> pose = SE3_to_pose(SE3)
   >>> print("Pose:", pose)
   >>> 
   >>> # Pose to SE3
   >>> SE3_recon = pose_to_SE3(pose)
   >>> print("Reconstructed SE3:\n", SE3_recon)
   >>> 
   >>> # Visualize Pose
   >>> SE3_array = np.random.randn(5, 4, 4) # Example random data
   >>> # Ensure valid rotation matrices for visualization
   >>> for i in range(SE3_array.shape[0]):
   ...     q = np.random.rand(4)
   ...     q /= np.linalg.norm(q)
   ...     SE3_array[i, :3, :3] = Quaternion(q).to_DCM()
   ...     SE3_array[i, :3, 3] = np.random.rand(3) * 5
   ...     SE3_array[i, 3, :3] = 0
   ...     SE3_array[i, 3, 3] = 1
   >>> poses_to_visualize = SE3_to_pose(SE3_array)
   >>> # visualize_pose(poses_to_visualize) # Uncomment to display plot 