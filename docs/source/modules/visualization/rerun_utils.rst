Rerun 工具 (rerun_utils)
========================

.. automodule:: pywayne.visualization.rerun_utils

.. autoclass:: pywayne.visualization.rerun_utils.RerunUtils
   :members:
   :undoc-members:
   :show-inheritance:

   Rerun可视化工具类，提供了一系列静态方法来向Rerun查看器添加各种3D可视化元素。

   **主要功能**:

   - 向Rerun查看器添加3D可视化元素：
     - **点云**: 支持单色或多色点云可视化
     - **轨迹**: 支持3D轨迹线的可视化
     - **相机**: 支持3D相机模型的可视化，可选择性显示图像
     - **平面**: 支持通过中心+法向量或变换矩阵定义的平面
     - **棋盘**: 支持在任意位置和朝向创建标准棋盘格
   - 简化与Rerun SDK的交互，提供直观的API接口
   - 支持静态可视化元素的添加和管理

   **主要方法 (按功能分类)**:

   - **点云可视化**:
     - `add_point_cloud(points_3d, colors, label)`: 添加3D点云
   - **轨迹可视化**:
     - `add_trajectory(traj_endpoints, colors, label)`: 添加3D轨迹线
   - **相机可视化**:
     - `add_camera(camera_pose, image, label)`: 添加3D相机模型
   - **平面可视化**:
     - `add_plane_from_center_and_normal(center, normal, half_size, color, label)`: 通过几何参数添加平面
     - `add_plane_from_Twp(Twp, half_size, color, label)`: 通过变换矩阵添加平面
   - **棋盘可视化**:
     - `add_chessboard_from_Twp(rows, cols, cell_size, Twp, color1, color2, label)`: 通过变换矩阵添加3D棋盘格
   - **内部辅助方法**:
     - `_get_quaternion_from_v1_and_v2(v1, v2)`: 计算两个向量间的旋转四元数

   **示例**::

      >>> import numpy as np
      >>> import rerun as rr
      >>> from pywayne.visualization.rerun_utils import RerunUtils
      >>> 
      >>> # 初始化Rerun
      >>> rr.init('example', spawn=True)
      >>> 
      >>> # 添加红色点云
      >>> points = np.random.rand(100, 3)
      >>> RerunUtils.add_point_cloud(points, colors=[255, 0, 0], label='red_points')
      >>> 
      >>> # 添加轨迹线
      >>> trajectory = np.cumsum(np.random.rand(50, 3) * 0.1, axis=0)
      >>> RerunUtils.add_trajectory(trajectory, colors=[0, 255, 0], label='path')
      >>> 
      >>> # 添加相机
      >>> camera_pose = np.eye(4)
      >>> camera_pose[:3, 3] = [1, 0, 0]  # 设置位置
      >>> RerunUtils.add_camera(camera_pose, label='main_camera')
      >>> 
      >>> # 添加蓝色平面
      >>> center = np.array([0, 0, 0])
      >>> normal = np.array([0, 0, 1])  # 垂直向上
      >>> RerunUtils.add_plane_from_center_and_normal(
      ...     center, normal, half_size=1.0, 
      ...     color=np.array([0, 0, 255]), label='ground'
      ... )
      >>> 
      >>> # 添加标准8x8棋盘
      >>> RerunUtils.add_chessboard_from_Twp(
      ...     rows=8, cols=8, cell_size=0.1,
      ...     Twp=np.eye(4), label='chessboard'
      ... )

点云可视化详细说明
------------------

   **add_point_cloud() 方法**:

   用于在Rerun查看器中添加3D点云数据，支持灵活的颜色配置。

   **参数**:
     - `points_3d` (numpy.ndarray): 3D点坐标，形状为(N, 3)
     - `colors` (可选): 点的颜色配置，支持多种格式
     - `label` (str): 点云标签，默认为'point_cloud'

   **颜色配置选项**:
     - **单色**: `[R, G, B]` - 所有点使用相同颜色
     - **多色**: `(N, 3)` 数组 - 每个点对应一个RGB颜色
     - **默认**: `None` - 使用红色 `[255, 0, 0]`

   **示例**::

      >>> # 单色点云
      >>> points = np.random.rand(1000, 3)
      >>> RerunUtils.add_point_cloud(points, colors=[0, 255, 0])  # 绿色
      >>> 
      >>> # 多色点云
      >>> colors = np.random.randint(0, 255, (1000, 3))
      >>> RerunUtils.add_point_cloud(points, colors=colors, label='rainbow')

轨迹可视化详细说明
------------------

   **add_trajectory() 方法**:

   用于在Rerun查看器中添加3D轨迹线，将一系列点连接成连续的路径。

   **参数**:
     - `traj_endpoints` (numpy.ndarray): 轨迹点坐标，形状为(N, 3)
     - `colors` (可选): 轨迹线颜色
     - `label` (str): 轨迹标签，默认为'trajectory'

   **示例**::

      >>> # 螺旋轨迹
      >>> t = np.linspace(0, 4*np.pi, 100)
      >>> trajectory = np.column_stack([
      ...     np.cos(t), np.sin(t), t/10
      ... ])
      >>> RerunUtils.add_trajectory(trajectory, colors=[255, 255, 0])

相机可视化详细说明
------------------

   **add_camera() 方法**:

   用于在Rerun查看器中添加3D相机模型，显示相机的位置、朝向和可选的图像内容。

   **参数**:
     - `camera_pose` (numpy.ndarray): 相机位姿矩阵，形状为(4, 4)
     - `image` (可选): 相机图像，支持多种格式
     - `label` (str): 相机标签，默认为'camera'

   **图像格式支持**:
     - **numpy数组**: 直接的图像数据，形状为(H, W, 3)
     - **文件路径**: 字符串或Path对象，指向图像文件
     - **None**: 不显示图像，仅显示相机模型

   **示例**::

      >>> # 基本相机
      >>> pose = np.eye(4)
      >>> pose[:3, 3] = [2, 1, 1]  # 设置位置
      >>> RerunUtils.add_camera(pose, label='observer')
      >>> 
      >>> # 带图像的相机
      >>> RerunUtils.add_camera(
      ...     pose, image='path/to/image.jpg', 
      ...     label='rgb_camera'
      ... )

平面可视化详细说明
------------------

   **平面创建方法**:

   提供两种方式创建3D平面：几何参数法和变换矩阵法。

   **方法1: add_plane_from_center_and_normal()**
     - 通过中心点和法向量定义平面
     - 适用于已知几何参数的场景

   **方法2: add_plane_from_Twp()**
     - 通过4x4变换矩阵定义平面
     - 适用于已有SE(3)变换的场景

   **示例**::

      >>> # 方法1: 几何参数
      >>> center = np.array([0, 0, 1])
      >>> normal = np.array([0, 0, 1])  # 水平面
      >>> RerunUtils.add_plane_from_center_and_normal(
      ...     center, normal, half_size=2.0,
      ...     color=np.array([128, 128, 128])
      ... )
      >>> 
      >>> # 方法2: 变换矩阵
      >>> Twp = np.eye(4)
      >>> Twp[:3, 3] = [1, 1, 0]  # 平移
      >>> RerunUtils.add_plane_from_Twp(
      ...     Twp, half_size=1.5,
      ...     color=np.array([255, 192, 128])
      ... )

棋盘可视化详细说明
------------------

   **add_chessboard_from_Twp() 方法**:

   用于通过变换矩阵在3D空间中创建棋盘格模式，常用于相机标定、空间参考或视觉装饰。

   **参数**:
     - `rows` (int): 棋盘行数，默认为8
     - `cols` (int): 棋盘列数，默认为8
     - `cell_size` (float): 单个格子的边长，默认为0.1
     - `Twp` (numpy.ndarray): 棋盘位姿矩阵，默认为单位矩阵
     - `color1` (numpy.ndarray): 第一种颜色（黑格），默认为黑色
     - `color2` (numpy.ndarray): 第二种颜色（白格），默认为白色
     - `label` (str): 棋盘基础标签，默认为'chessboard'

   **特性**:
     - 以指定位置为中心对称分布
     - 格子按照标准棋盘模式交替颜色
     - 每个格子独立标签：`{label}/{row}_{col}`
     - 支持任意位置和朝向

   **示例**::

      >>> # 标准棋盘
      >>> RerunUtils.add_chessboard_from_Twp()
      >>> 
      >>> # 相机标定棋盘
      >>> RerunUtils.add_chessboard_from_Twp(
      ...     rows=9, cols=6, cell_size=0.025,  # 25mm格子
      ...     label='calibration_board'
      ... )
      >>> 
      >>> # 倾斜彩色棋盘
      >>> from pywayne.vio.SE3 import SE3
      >>> Twp_tilted = SE3.SE3_exp(np.array([0, 0, 0, 0, np.pi/4, 0]))
      >>> RerunUtils.add_chessboard_from_Twp(
      ...     rows=6, cols=6, cell_size=0.08,
      ...     Twp=Twp_tilted,
      ...     color1=np.array([255, 0, 0]),    # 红色
      ...     color2=np.array([0, 0, 255]),    # 蓝色
      ...     label='colored_board'
      ... )

--------------------------------------------------

通过上述示例，用户可以快速掌握 RerunUtils 类的主要功能，并将其应用于3D数据可视化、机器人调试、计算机视觉开发等领域。RerunUtils 提供了简洁直观的API，大大简化了与Rerun SDK的交互过程。 