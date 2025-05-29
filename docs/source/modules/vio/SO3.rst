SO3 旋转矩阵工具 (SO3)
=======================

.. automodule:: pywayne.vio.SO3
   :members:
   :undoc-members:

该模块提供了SO(3)旋转矩阵的完整操作工具集，包括旋转矩阵的生成、转换、组合、对数/指数映射以及平均等功能。SO(3)群表示三维空间中的旋转，是机器人学、计算机视觉和SLAM领域的重要数学工具。

数学背景
---------

SO(3)是特殊正交群，表示三维欧几里得空间中的旋转。SO(3)群的元素是3×3的正交矩阵，满足：

- **正交性**: :math:`R^T R = I`
- **行列式为1**: :math:`\det(R) = 1`

相应的李代数so(3)由3×3反对称矩阵组成，通过指数映射连接到李群SO(3)。

主要功能分类
-------------

基础操作函数
~~~~~~~~~~~~~

.. autofunction:: pywayne.vio.SO3.check_SO3
   :noindex:

验证矩阵是否为有效的SO(3)旋转矩阵。

.. autofunction:: pywayne.vio.SO3.SO3_mul
   :noindex:

计算两个旋转矩阵的乘积，实现旋转组合。

.. autofunction:: pywayne.vio.SO3.SO3_diff
   :noindex:

计算两个旋转矩阵之间的相对旋转。

.. autofunction:: pywayne.vio.SO3.SO3_inv
   :noindex:

计算旋转矩阵的逆（转置）。

反对称矩阵操作
~~~~~~~~~~~~~~~

.. autofunction:: pywayne.vio.SO3.SO3_skew
   :noindex:

将3D向量转换为反对称矩阵（skew-symmetric matrix）。

.. autofunction:: pywayne.vio.SO3.SO3_unskew
   :noindex:

从反对称矩阵提取对应的3D向量。

旋转表示转换
~~~~~~~~~~~~~

.. autofunction:: pywayne.vio.SO3.SO3_from_quat
   :noindex:

.. autofunction:: pywayne.vio.SO3.SO3_to_quat
   :noindex:

四元数与旋转矩阵的相互转换，使用Hamilton约定(wxyz)。

.. autofunction:: pywayne.vio.SO3.SO3_from_axis_angle
   :noindex:

.. autofunction:: pywayne.vio.SO3.SO3_to_axis_angle
   :noindex:

轴角表示与旋转矩阵的相互转换。

.. autofunction:: pywayne.vio.SO3.SO3_from_euler
   :noindex:

.. autofunction:: pywayne.vio.SO3.SO3_to_euler
   :noindex:

欧拉角与旋转矩阵的相互转换，支持多种旋转序列。

李群/李代数映射
~~~~~~~~~~~~~~~

.. autofunction:: pywayne.vio.SO3.SO3_Log
   :noindex:

SO(3) → so(3)的对数映射，输出旋转向量(3D)。

.. autofunction:: pywayne.vio.SO3.SO3_log
   :noindex:

SO(3) → so(3)的对数映射，输出反对称矩阵(3×3)。

.. autofunction:: pywayne.vio.SO3.SO3_Exp
   :noindex:

so(3) → SO(3)的指数映射，从旋转向量生成旋转矩阵。

.. autofunction:: pywayne.vio.SO3.SO3_exp
   :noindex:

so(3) → SO(3)的指数映射，从反对称矩阵生成旋转矩阵。

平均与统计
~~~~~~~~~~~

.. autofunction:: pywayne.vio.SO3.SO3_mean
   :noindex:

计算多个旋转矩阵的平均值，使用基于四元数的数值稳定方法。

数学关系
--------

该模块中的函数满足以下重要数学关系：

**往返转换**:
  - :math:`R = \text{SO3\_Exp}(\text{SO3\_Log}(R))`
  - :math:`\text{SO3\_log}(R) = \text{SO3\_skew}(\text{SO3\_Log}(R))`

**群运算**:
  - :math:`\text{SO3\_mul}(R_1, R_2) = R_1 \cdot R_2`
  - :math:`\text{SO3\_inv}(R) = R^T`

**代数操作**:
  - :math:`\text{SO3\_exp}(\omega^\wedge) = \exp(\omega^\wedge)`
  - :math:`\text{SO3\_Log}(R) = \log(R)^\vee`

使用示例
--------

基础操作示例
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pywayne.vio.SO3 import *
   
   # 创建旋转矩阵（绕Z轴90度）
   angle = np.pi/2
   R = np.array([
       [np.cos(angle), -np.sin(angle), 0],
       [np.sin(angle),  np.cos(angle), 0],
       [0,              0,             1]
   ])
   
   # 验证是否为有效的SO(3)矩阵
   print(f"是否为有效SO3: {check_SO3(R)}")
   
   # 计算逆矩阵
   R_inv = SO3_inv(R)
   
   # 验证 R @ R^(-1) = I
   identity_check = SO3_mul(R, R_inv)
   print(f"与单位矩阵的误差: {np.linalg.norm(identity_check - np.eye(3)):.2e}")

反对称矩阵操作
~~~~~~~~~~~~~~~

.. code-block:: python

   # 创建向量并转换为反对称矩阵
   vec = np.array([1, 2, 3])
   skew_mat = SO3_skew(vec)
   print(f"反对称矩阵:\n{skew_mat}")
   
   # 从反对称矩阵恢复向量
   recovered_vec = SO3_unskew(skew_mat)
   print(f"恢复向量: {recovered_vec}")
   print(f"往返误差: {np.linalg.norm(vec - recovered_vec):.2e}")

旋转表示转换
~~~~~~~~~~~~~

.. code-block:: python

   # 四元数转换
   quat = SO3_to_quat(R.reshape(1, 3, 3))
   R_from_quat = SO3_from_quat(quat)
   print(f"四元数转换误差: {np.linalg.norm(R - R_from_quat[0]):.2e}")
   
   # 轴角转换
   axis, angle = SO3_to_axis_angle(R.reshape(1, 3, 3))
   R_from_axis_angle = SO3_from_axis_angle(axis, angle)
   print(f"轴角转换误差: {np.linalg.norm(R - R_from_axis_angle[0]):.2e}")
   
   # 欧拉角转换
   euler = SO3_to_euler(R.reshape(1, 3, 3))
   R_from_euler = SO3_from_euler(euler)
   print(f"欧拉角转换误差: {np.linalg.norm(R - R_from_euler[0]):.2e}")

李群/李代数映射
~~~~~~~~~~~~~~~

.. code-block:: python

   # 对数映射 - 向量形式
   log_vec = SO3_Log(R)
   print(f"对数映射向量: {log_vec}")
   
   # 对数映射 - 矩阵形式  
   log_mat = SO3_log(R)
   print(f"对数映射矩阵:\n{log_mat}")
   
   # 指数映射 - 从向量
   R_from_exp = SO3_Exp(log_vec)
   print(f"指数映射(向量)误差: {np.linalg.norm(R - R_from_exp):.2e}")
   
   # 指数映射 - 从矩阵
   R_from_exp_mat = SO3_exp(log_mat)
   print(f"指数映射(矩阵)误差: {np.linalg.norm(R - R_from_exp_mat):.2e}")

批量处理示例
~~~~~~~~~~~~~

.. code-block:: python

   # 创建多个随机旋转
   n_rotations = 100
   random_axis = np.random.randn(n_rotations, 3)
   random_axis = random_axis / np.linalg.norm(random_axis, axis=1, keepdims=True)
   random_angles = np.random.uniform(0, np.pi, n_rotations)
   
   # 批量生成旋转矩阵
   R_batch = SO3_from_axis_angle(random_axis, random_angles)
   
   # 批量转换为四元数
   quat_batch = SO3_to_quat(R_batch)
   
   # 计算平均旋转
   R_mean = SO3_mean(R_batch)
   print(f"平均旋转矩阵是否有效: {check_SO3(R_mean)}")

性能特点
--------

- **批量处理**: 所有函数都支持批量操作，提高计算效率
- **数值稳定**: 采用数值稳定的算法，避免奇异情况
- **高精度**: 往返转换误差通常在机器精度水平(~1e-16)
- **向量化**: 使用NumPy的einsum等高效操作

应用场景
--------

1. **机器人学**: 关节旋转、工具坐标变换
2. **计算机视觉**: 相机外参、特征匹配
3. **SLAM**: 里程计估计、姿态优化
4. **航空航天**: 姿态控制、导航系统
5. **动画制作**: 3D模型旋转、插值

注意事项
--------

1. **输入格式**: 单个矩阵输入为3×3，批量输入为N×3×3
2. **约定**: 四元数使用Hamilton约定(wxyz)
3. **角度单位**: 所有角度使用弧度制
4. **数值精度**: 建议使用float64以获得最佳精度
5. **奇异情况**: 180度旋转附近可能出现数值不稳定 