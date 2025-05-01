Pangolin 查看器 (pangolin)
============================

.. automodule:: pywayne.visualization.pangolin

.. autoclass:: pywayne.visualization.pangolin.PangolinViewer
   :members:
   :undoc-members:
   :show-inheritance:

   封装了 Pangolin 查看器的 Python 接口，用于在 3D 环境中高效实时地展示数据。

   **主要功能**:

   - 初始化和管理 3D 可视化窗口。
   - 提供丰富的 API 用于发布和清除各种 3D 数据：
     - **点云**: 支持添加带单一颜色、多种颜色或命名颜色的点云。
     - **轨迹**: 支持通过位置+四元数 (多种格式) 或 SE3 矩阵发布轨迹，可选择显示相机模型。
     - **相机**: 支持独立添加和管理相机位姿（通过四元数或 SE3），并设置主跟随相机。
     - **平面**: 支持通过顶点或法线+中心点添加（半透明）平面。
     - **直线**: 支持添加指定起点、终点、颜色和线宽的线段。
   - 支持显示左右两个视图的图像（通过 Numpy 数组或文件路径）。
   - 提供窗口控制（运行、关闭、重置、刷新）和状态查询功能。
   - 支持简单的步进模式控制，用于调试和演示。

   **主要方法 (按功能分类)**:

   - **核心控制**:
     - `run()`: 启动查看器主循环。
     - `close()`: 关闭查看器。
     - `join()`: 等待查看器进程结束。
     - `reset()`: 重置查看器状态。
     - `init()`: 初始化视图（如设置初始视角）。
     - `show(delay_time_in_s)`: 刷新视图并处理事件。
     - `should_not_quit()`: 检查查看器是否应继续运行。
   - **点云 API**:
     - `clear_all_points()`
     - `add_points(points, color, label, point_size)`
     - `add_points_with_colors(points, colors, label, point_size)`
     - `add_points_with_color_name(points, color_name, label, point_size)`
   - **轨迹 API**:
     - `clear_all_trajectories()`
     - `add_trajectory_quat(positions, orientations, color, quat_format, label, line_width, show_cameras, camera_size)`
     - `add_trajectory_se3(poses_se3, color, label, line_width, show_cameras, camera_size)`
   - **相机 API**:
     - `clear_all_cameras()`
     - `set_main_camera(camera_id)`
     - `add_camera_quat(position, orientation, color, quat_format, label, scale, line_width)`
     - `add_camera_se3(pose_se3, color, label, scale, line_width)`
   - **平面 API**:
     - `clear_all_planes()`
     - `add_plane(vertices, color, alpha, label)`
     - `add_plane_normal_center(normal, center, size, color, alpha, label)`
   - **直线 API**:
     - `clear_all_lines()`
     - `add_line(start_point, end_point, color, line_width, label)`
   - **图像 API**:
     - `set_img_resolution(width, height)`
     - `add_image_1(img, image_path)`
     - `add_image_2(img, image_path)`
   - **步进控制 API**:
     - `is_step_mode_active()`
     - `wait_for_step()`
   - **旧版 VIO 相关 (可能弃用或整合)**:
     - `publish_traj(...)`, `publish_3D_points(...)`, `publish_track_img(...)`, etc.

   **示例**::

      >>> import numpy as np
      >>> from pywayne.visualization import pangolin
      >>> from pywayne.visualization.pangolin import Colors # Import Colors class
      >>> import time
      >>> 
      >>> # 创建 Pangolin 查看器
      >>> viewer = pangolin.PangolinViewer(800, 600)
      >>> viewer.init()
      >>> 
      >>> # 准备数据
      >>> # 螺旋线轨迹 (SE3)
      >>> num_points = 100
      >>> theta = np.linspace(0, 3 * 2 * np.pi, num_points)
      >>> radius = 0.5
      >>> height = 1.0
      >>> t_x = radius * np.cos(theta)
      >>> t_y = radius * np.sin(theta)
      >>> t_z = np.linspace(0, height, num_points)
      >>> helix_positions = np.column_stack((t_x, t_y, t_z))
      >>> helix_poses_se3 = []
      >>> from scipy.spatial.transform import Rotation as R
      >>> for i in range(num_points):
      ...     z_axis = np.array([np.sin(theta[i]/2.0), 0, np.cos(theta[i]/2.0)]) # 简单倾斜
      ...     z_axis /= np.linalg.norm(z_axis)
      ...     x_dir = np.array([np.cos(theta[i]), np.sin(theta[i]), 0])
      ...     x_axis = x_dir / np.linalg.norm(x_dir) if np.linalg.norm(x_dir) > 1e-6 else np.array([1.,0.,0.])
      ...     y_axis = np.cross(z_axis, x_axis)
      ...     pose = np.identity(4)
      ...     pose[:3, 0] = x_axis
      ...     pose[:3, 1] = y_axis
      ...     pose[:3, 2] = z_axis
      ...     pose[:3, 3] = helix_positions[i]
      ...     helix_poses_se3.append(pose)
      >>> helix_poses_se3 = np.array(helix_poses_se3, dtype=np.float32)
      >>> 
      >>> # 球形点云 (带颜色)
      >>> sphere_radius = 1.5
      >>> phi, psi = np.linspace(0, np.pi, 20), np.linspace(0, 2 * np.pi, 40)
      >>> phi_grid, psi_grid = np.meshgrid(phi, psi, indexing='ij')
      >>> sphere_x = sphere_radius * np.sin(phi_grid) * np.cos(psi_grid)
      >>> sphere_y = sphere_radius * np.sin(phi_grid) * np.sin(psi_grid)
      >>> sphere_z = sphere_radius * np.cos(phi_grid)
      >>> sphere_points = np.stack([sphere_x.ravel(), sphere_y.ravel(), sphere_z.ravel()], axis=-1).astype(np.float32)
      >>> sphere_colors = np.stack([(np.sin(phi_grid*2).ravel()+1)/2, (np.cos(psi_grid*2).ravel()+1)/2, (np.sin(psi_grid).ravel()+1)/2], axis=-1).astype(np.float32)
      >>> 
      >>> # 可视化循环
      >>> frame = 0
      >>> while viewer.should_not_quit() and frame < num_points:
      ...     viewer.clear_all_trajectories()
      ...     viewer.clear_all_points()
      ...     viewer.clear_all_cameras()
      ... 
      ...     # 添加部分轨迹，带相机模型
      ...     viewer.add_trajectory_se3(helix_poses_se3[:frame+1], color=Colors.CYAN, label="Helix", show_cameras=True, camera_size=0.05)
      ...     # 添加静态相机
      ...     static_pose = np.identity(4, dtype=np.float32)
      ...     static_pose[:3,3] = [0.8, -0.8, 0.2]
      ...     cam_id = viewer.add_camera_se3(static_pose, color=Colors.ORANGE, label="Static Cam", scale=0.1)
      ...     # 让视图跟随这个静态相机
      ...     viewer.set_main_camera(cam_id)
      ... 
      ...     # 添加带颜色的点云
      ...     viewer.add_points_with_colors(sphere_points, sphere_colors, label="Sphere")
      ...     
      ...     # 添加一条直线
      ...     viewer.add_line(np.zeros(3), helix_positions[frame], color=Colors.WHITE, line_width=2.0)
      ... 
      ...     viewer.show(delay_time_in_s=0.05)
      ...     frame += 1
      ...     # time.sleep(0.05) # Use viewer.show delay instead
      >>> 
      >>> print("Viewer loop finished. Press Ctrl+C or close window.")
      >>> viewer.join() # Wait for window to be closed manually
      >>> # viewer.close() # Or close programmatically

--------------------------------------------------

通过上述示例，用户可以快速掌握 visualization 模块中 PangolinViewer 类的主要功能，并将其应用于机器人、计算机视觉等领域中的 3D 数据实时展示和交互控制。 