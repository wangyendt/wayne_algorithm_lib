视觉化工具 (visualization)
===========================

本模块主要用于 3D 数据的可视化展示，集成了 Pangolin 查看器和 Rerun 工具。

.. toctree::
   :maxdepth: 1
   :caption: Visualization Submodules:

   visualization/pangolin_utils
   visualization/rerun_utils

PangolinViewer 类 (pangolin_utils)
-------------------------------------

.. py:class:: PangolinViewer(width: int, height: int, run_on_start: bool = False)

   PangolinViewer 类封装了 Pangolin 查看器，在 3D 环境中提供高效实时的数据展示功能。用户可以通过此类初始化一个 3D 可视化窗口，并发布轨迹、点云、图像等数据。
   
   **主要功能**:
   
   - 初始化可视化窗口，指定窗口的宽度和高度。
   - 启动查看器主循环，实现实时数据刷新。
   - 支持发布轨迹、3D 点云、跟踪图像以及其它视觉惯性数据。
   - 提供窗口重置、刷新、关闭等常用操作。
   
   **主要方法**:
   
   - **run(self)**: 启动查看器的主循环，保持视图更新。
   - **close(self)**: 关闭查看器窗口。
   - **join(self)**: 等待查看器进程结束，适用于线程同步。
   - **reset(self)**: 重置查看器状态。
   - **init(self)**: 初始化查看器（例如设置初始视角）。
   - **show(self, delay_time_in_s: float = 0.0)**: 刷新视图，可设置延时以实现动态更新。
   - **should_not_quit(self) -> bool**: 检查查看器是否处于运行状态，用于循环控制。
   - **set_img_resolution(self, width: int, height: int)**: 设置显示图像的分辨率。
   - **publish_traj(self, t_wc: np.ndarray, q_wc: np.ndarray)**: 发布轨迹数据，用于显示实时轨迹信息。
   - **publish_3D_points(self, slam_pts, msckf_pts)**: 发布 3D 点云数据，便于展示不同来源的点云。
   - **publish_track_img(self, img: np.ndarray)**: 发布跟踪图像，用于图像调试和展示。
   - **publish_vio_opt_data(self, vals)**: 发布视觉惯性优化数据，用于 VIO 算法的结果展示。
   - **publish_plane_detection_img(self, img: np.ndarray)**: 发布平面检测结果图像。
   - **publish_plane_triangulate_pts(self, plane_tri_pts)**: 发布平面三角化点数据。
   - **publish_plane_vio_stable_pts(self, plane_vio_stable_pts)**: 发布视觉惯性稳定点数据。
   - **publish_planes_horizontal(self, his_planes_horizontal)**: 发布水平面数据。
   - **publish_planes_vertical(self, planes)**: 发布垂直面数据。
   - **publish_traj_gt(self, q_wc_gt, t_wc_gt)**: 发布地面真实轨迹数据，用于与估计轨迹对比。
   - **get_algorithm_wait_flag(self)**: 获取算法等待标志，用于同步算法状态。
   - **set_visualize_opencv_mat(self)**: 设置使用 OpenCV 格式的图像进行展示。
   - **algorithm_wait(self)**: 在算法中暂停，等待外部触发。
   - **notify_algorithm(self)**: 通知暂停的算法继续执行。
   
   **示例**::
   
      >>> import numpy as np
      >>> from pywayne.visualization import pangolin_utils
      >>> # 创建 Pangolin 查看器，窗口尺寸为 640x480
      >>> viewer = pangolin_utils.PangolinViewer(640, 480)
      >>> viewer.init()  # 初始化查看器
      >>> # 示例：发布随机生成的轨迹数据
      >>> traj = np.random.rand(100, 3)
      >>> viewer.publish_traj(traj, traj)  # 此处示例使用相同数据作为轨迹
      >>> # 循环刷新视图，实时展示数据
      >>> while viewer.should_not_quit():
      ...     viewer.show(0.01)
      >>> viewer.close()

RerunUtils 类
--------------

.. py:class:: RerunUtils

   RerunUtils 类提供了一系列静态方法来向 Rerun 查看器添加各种 3D 可视化元素。该类封装了常用的 3D 可视化操作，简化了与 Rerun SDK 的交互。
   
   **主要功能**:
   
   - 向 Rerun 查看器添加 3D 可视化元素（点云、轨迹、相机、平面、棋盘等）
   - 提供简洁直观的 API 接口
   - 支持静态可视化元素的添加和管理
   
   **主要方法**:
   
   - **add_point_cloud(points_3d, colors, label)**: 添加 3D 点云，支持单色或多色配置
   - **add_trajectory(traj_endpoints, colors, label)**: 添加 3D 轨迹线
   - **add_camera(camera_pose, image, label)**: 添加 3D 相机模型，可选择性显示图像
   - **add_plane_from_center_and_normal(center, normal, half_size, color, label)**: 通过几何参数添加平面
   - **add_plane_from_Twp(Twp, half_size, color, label)**: 通过变换矩阵添加平面
   - **add_chessboard_from_Twp(rows, cols, cell_size, Twp, color1, color2, label)**: 通过变换矩阵添加 3D 棋盘格
   
   **示例**::
   
      >>> import numpy as np
      >>> import rerun as rr
      >>> from pywayne.visualization.rerun_utils import RerunUtils
      >>> # 初始化 Rerun
      >>> rr.init('example', spawn=True)
      >>> # 添加红色点云
      >>> points = np.random.rand(100, 3)
      >>> RerunUtils.add_point_cloud(points, colors=[255, 0, 0])
      >>> # 添加相机
      >>> camera_pose = np.eye(4)
      >>> RerunUtils.add_camera(camera_pose, label='main_camera')

--------------------------------------------------

通过上述示例，用户可以快速掌握 visualization 模块中 PangolinViewer 和 RerunUtils 类的主要功能，并将其应用于机器人、计算机视觉等领域中的 3D 数据实时展示和交互控制。 