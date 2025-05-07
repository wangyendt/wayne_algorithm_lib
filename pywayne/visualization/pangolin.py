# author: wangye(Wayne)
# license: Apache Licence
# file: pangolin.py
# time: 2023-11-06-11:14:19
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import sys
import subprocess
import os
import importlib
import numpy as np
import pandas as pd
import time
import cv2
from scipy.spatial.transform import Rotation as R

# 使用类封装颜色常量
class Colors:
    RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    BLUE = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    YELLOW = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    CYAN = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    MAGENTA = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    BLACK = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ORANGE = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    PURPLE = np.array([0.5, 0.0, 0.5], dtype=np.float32)
    GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    BROWN = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    PINK = np.array([1.0, 0.75, 0.8], dtype=np.float32)

# 辅助函数定义移至 __main__ 块


class PangolinViewer:
    def __init__(self, width: int, height: int, run_on_start=False):
        self.width = width
        self.height = height
        self.run_on_start = run_on_start
        self.viewer = self._check_lib_exists()

    # 恢复用户提供的原始版本
    def _check_lib_exists(self):
        lib_path = os.path.join(os.path.dirname(__file__), 'lib')
        sys.path.append(str(lib_path))
        try:
            from pangolin_viewer import PangolinViewer as Viewer
        except ImportError:
            os.makedirs(lib_path, exist_ok=True)
            subprocess.run(['gettool', 'pangolin_viewer', '-b', '-t', str(lib_path)], check=True)
            importlib.invalidate_caches()
            Viewer = importlib.import_module("pangolin_viewer").PangolinViewer
        return Viewer(self.width, self.height, self.run_on_start)

    def run(self):
        self.viewer.run()

    def close(self):
        self.viewer.close()

    def join(self):
        self.viewer.join()

    def reset(self):
        self.viewer.reset()

    def init(self):
        # Calls C++ view_init (bound name of extern_init)
        self.viewer.view_init() 

    def show(self, delay_time_in_s=0.0):
        # Calls C++ show (bound name of extern_run_single_step)
        self.viewer.show(delay_time_in_s) 

    def should_not_quit(self):
        # Calls C++ should_not_quit (bound name of extern_should_not_quit)
        return self.viewer.should_not_quit() 

    def set_img_resolution(self, width: int, height: int):
        self.viewer.set_img_resolution(width, height)

    # --- Re-adding New API Wrappers --- 

    def clear_all_visual_elements(self):
        self.viewer.clear_all_visual_elements()

    # Point Cloud API
    def clear_all_points(self):
        self.viewer.clear_all_points()
        
    def add_points(self, points, color=None, label="", point_size=4.0):
        if color is None:
            color = Colors.RED
        self.viewer.add_points(points.astype(np.float32), color, label, point_size)
    
    def add_points_with_colors(self, points, colors, label="", point_size=4.0):
        self.viewer.add_points_with_colors(points.astype(np.float32), colors.astype(np.float32), label, point_size)
    
    def add_points_with_color_name(self, points, color_name="red", label="", point_size=4.0):
        self.viewer.add_points_with_color_name(points.astype(np.float32), color_name, label, point_size)

    # Trajectory API
    def clear_all_trajectories(self):
        self.viewer.clear_all_trajectories()
    
    def add_trajectory_quat(self, positions, orientations, color=None, quat_format="wxyz", 
                           label="", line_width=1.0, show_cameras=False, camera_size=0.05):
        if color is None:
            color = Colors.GREEN
        self.viewer.add_trajectory_quat(positions.astype(np.float32), orientations.astype(np.float32), 
                                      color, quat_format, label, line_width, show_cameras, camera_size)
    
    def add_trajectory_se3(self, poses_se3, color=None, label="", line_width=1.0, 
                          show_cameras=False, camera_size=0.05):
        if color is None:
            color = Colors.GREEN
        self.viewer.add_trajectory_se3(poses_se3.astype(np.float32), color, label, line_width, 
                                     show_cameras, camera_size)
    
    # Camera API
    def clear_all_cameras(self):
        self.viewer.clear_all_cameras()
    
    def set_main_camera(self, camera_id):
        self.viewer.set_main_camera(camera_id)
    
    def add_camera_quat(self, position, orientation, color=None, quat_format="wxyz",
                       label="", scale=0.1, line_width=1.0):
        if color is None:
            color = Colors.YELLOW
        return self.viewer.add_camera_quat(position.astype(np.float32), orientation.astype(np.float32), 
                                         color, quat_format, label, scale, line_width)
    
    def add_camera_se3(self, pose_se3, color=None, label="", scale=0.1, line_width=1.0):
        if color is None:
            color = Colors.YELLOW
        return self.viewer.add_camera_se3(pose_se3.astype(np.float32), color, label, scale, line_width)

    # Plane API
    def clear_all_planes(self):
        self.viewer.clear_all_planes()
        
    def add_plane(self, vertices, color=None, alpha=0.5, label=""):
        if vertices.shape[0] < 3:
            print("Warning: Plane needs at least 3 vertices. Skipping.")
            return
        if color is None:
            color = Colors.GRAY
        self.viewer.add_plane(vertices.astype(np.float32), color, alpha, label)
        
    def add_plane_normal_center(self, normal, center, size, color=None, alpha=0.5, label=""):
        if color is None:
            color = Colors.GRAY
        self.viewer.add_plane_normal_center(normal.astype(np.float32), center.astype(np.float32), size, color, alpha, label)
        
    # Line API
    def clear_all_lines(self):
        self.viewer.clear_all_lines()
        
    def add_line(self, start_point, end_point, color=None, line_width=1.0, label=""):
        if color is None:
            color = Colors.WHITE
        self.viewer.add_line(start_point.astype(np.float32), end_point.astype(np.float32), color, line_width, label)

    # Image API
    def add_image_1(self, img=None, image_path=None):
        if img is not None:
            # Ensure correct type and shape if needed
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            self.viewer.add_image_1(img)
        elif image_path is not None:
            self.viewer.add_image_1(image_path)
    
    def add_image_2(self, img=None, image_path=None):
        if img is not None:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            self.viewer.add_image_2(img)
        elif image_path is not None:
            self.viewer.add_image_2(image_path)
            
    # --- End of New API Wrappers --- 

    def get_algorithm_wait_flag(self):
        return self.viewer.get_algorithm_wait_flag()

    def set_visualize_opencv_mat(self):
        self.viewer.set_visualize_opencv_mat()

    def algorithm_wait(self):
        self.viewer.algorithm_wait()

    def notify_algorithm(self):
        self.viewer.notify_algorithm()

    # --- Modified Step Control Wrappers --- 

    def is_step_mode_active(self): # 重命名
        return self.viewer.is_step_mode_active()

    def wait_for_step(self): # 重命名
        self.viewer.wait_for_step()


if __name__ == '__main__':
    # 将辅助函数定义移到这里
    def random_walk_3d(steps=1000):
        steps = np.random.uniform(-0.001, 0.001, (steps, 3))
        walk = np.cumsum(steps, axis=0)
        return walk

    def make_traj():
        timestamps = pd.date_range(start='2024-01-01', periods=100, freq='min')
        
        # 椭圆参数
        a = 0.5  # 椭圆长轴
        b = 0.2  # 椭圆短轴

        # 时间步数
        theta = np.linspace(0, 2 * np.pi, 100)

        # 椭圆轨迹
        t_wc_x = a * np.cos(theta)
        t_wc_y = b * np.sin(theta)
        t_wc_z = np.zeros(100)
        t_wc = np.column_stack((t_wc_x, t_wc_y, t_wc_z)).astype(np.float32)

        # 生成四元数数据 q_wc，使其姿态一直朝向运动方向 (xyzw, Hamilton)
        q_wc = []
        for i in range(len(t_wc_x)):
            angle = np.arctan2(t_wc_y[i], t_wc_x[i]) + np.pi / 2 # 朝向切线方向
            # R.from_euler 的默认顺序是 'zyx'
            quat = R.from_euler('z', angle).as_quat() # 输出 xyzw
            q_wc.append(quat)
        q_wc = np.array(q_wc).astype(np.float32)

        # 构建 DataFrame
        data = np.column_stack((t_wc, q_wc))
        df = pd.DataFrame(data, columns=['t_wc_x', 't_wc_y', 't_wc_z', 'q_wc_x', 'q_wc_y', 'q_wc_z', 'q_wc_w'])
        df.insert(0, 'timestamp', timestamps)
        
        return df

    # 生成螺旋线轨迹数据 (用于演示新API)
    def make_helix_traj(num_points=100, radius=0.3, height=1.0, turns=3):
        theta = np.linspace(0, turns * 2 * np.pi, num_points)
        t_x = radius * np.cos(theta)
        t_y = radius * np.sin(theta)
        t_z = np.linspace(0, height, num_points)
        positions = np.column_stack((t_x, t_y, t_z)).astype(np.float32)

        # 姿态：相机朝向Z轴正方向，X轴指向螺旋线外侧
        orientations_se3 = []
        for i in range(num_points):
            z_axis = np.array([0, 0, 1]) # Z轴保持不变
            # X轴指向外侧，大致方向为(cos(theta), sin(theta), 0)
            x_dir = np.array([np.cos(theta[i]), np.sin(theta[i]), 0])
            if np.linalg.norm(x_dir) < 1e-6: # Avoid division by zero at poles if needed
                x_axis = np.array([1.0, 0.0, 0.0]) # Default or handle singularity
            else:
                x_axis = x_dir / np.linalg.norm(x_dir)
            # Y轴通过叉乘得到
            y_axis = np.cross(z_axis, x_axis)
            
            rot_mat = np.eye(4)
            rot_mat[:3, 0] = x_axis
            rot_mat[:3, 1] = y_axis
            rot_mat[:3, 2] = z_axis
            rot_mat[:3, 3] = positions[i]
            orientations_se3.append(rot_mat)
            
        return np.array(orientations_se3, dtype=np.float32)

    print("Pangolin 可视化示例")
    # 生成主轨迹数据 (椭圆)
    main_traj_data = make_traj().values
    main_traj_t = main_traj_data[:, 1:4].astype(np.float32)
    main_traj_q = main_traj_data[:, 4:8].astype(np.float32) # xyzw
    
    # 生成额外的轨迹数据 (螺旋线)
    helix_traj_se3 = make_helix_traj()
    
    # 创建Pangolin查看器 (直接使用当前文件定义的类)
    viewer = PangolinViewer(640, 480, False)
    viewer.set_img_resolution(752, 480)
    viewer.init()
    
    # 创建一些测试图像
    img = np.zeros((480, 752, 3), dtype=np.uint8)
    cv2.putText(img, "Pangolin Viewer", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    # 准备点云数据
    print("准备点云数据...")
    cube_size = 0.5
    cube_points = np.array([[x, y, z] for x in [-cube_size, cube_size] for y in [-cube_size, cube_size] for z in [-cube_size, cube_size]], dtype=np.float32)
    theta, phi = np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 10)
    radius = 1.0
    sphere_points = np.array([[radius * np.sin(p) * np.cos(t), radius * np.sin(p) * np.sin(t), radius * np.cos(p)] for t in theta for p in phi], dtype=np.float32)
    sphere_colors = np.array([[(np.cos(t) + 1) / 2, (np.sin(p) + 1) / 2, (np.sin(t) + 1) / 2] for t in theta for p in phi], dtype=np.float32)
    grid_size, grid_step = 3, 0.2
    grid_points = np.array([[x, -1.5, z] for x in np.arange(-grid_size/2, grid_size/2+grid_step, grid_step) for z in np.arange(-grid_size/2, grid_size/2+grid_step, grid_step)], dtype=np.float32)
    line_points = np.array([[0, 0, 0], [0, 2, 0]], dtype=np.float32)
    random_points_1 = np.random.rand(20, 3).astype(np.float32) * 0.5 + np.array([1.5, 0, 1.0], dtype=np.float32)
    random_points_2 = np.random.rand(50, 3).astype(np.float32) * 0.5 + np.array([-1.5, 0, 1.0], dtype=np.float32)
    
    # 运行可视化循环
    print("开始可视化循环...")
    idx = 0
    frame_count = 0
    cube_extra_points, line_extra_points, random_extra_points = [], [], []
    
    # 用于演示主相机切换
    follow_helix_end = False
    helix_end_cam_id = None
    static_cam_id = None

    while viewer.should_not_quit():
        frame_count += 1
        # 更新主轨迹索引（循环）
        main_idx = frame_count % len(main_traj_t)
        
        # 1. 发布主轨迹的当前位姿 (使用旧API，视图默认跟随它)
        # current_t = main_traj_t[main_idx]
        # current_q_xyzw = main_traj_q[main_idx]
        # current_q_wxyz = np.array([current_q_xyzw[3], current_q_xyzw[0], current_q_xyzw[1], current_q_xyzw[2]], dtype=np.float32)
        # viewer.publish_traj(t_wc=current_t, q_wc=current_q_wxyz)
        # 注意：如果设置了主相机，publish_traj仍然会画出绿色相机，但视图不再跟随它
        
        # 2. 清除上一帧的额外轨迹、独立相机、点云、平面和直线
        # viewer.clear_all_trajectories()
        # viewer.clear_all_cameras()
        # viewer.clear_all_points()
        # viewer.clear_all_planes()
        # viewer.clear_all_lines()
        viewer.clear_all_visual_elements()
        
        # 3. 添加额外的轨迹 (使用新API)
        # 只显示螺旋线的一部分，模拟动态增长
        helix_idx = frame_count % (len(helix_traj_se3) + 50) # 比轨迹长一点，让它显示完后消失一会
        current_helix_pose_se3 = None
        if helix_idx < len(helix_traj_se3):
            # 使用 SE3 添加螺旋线轨迹，不显示相机
            viewer.add_trajectory_se3(
                poses_se3=helix_traj_se3[:helix_idx+1], 
                color=Colors.CYAN, 
                label="helix_se3", 
                line_width=2.0, 
                show_cameras=False, # 不在这里显示相机模型
                camera_size=0.04
            )
            current_helix_pose_se3 = helix_traj_se3[helix_idx]
        
        # 添加主轨迹的历史路径 (只画线)，使用 quat 添加
        viewer.add_trajectory_quat(
            positions=main_traj_t[:main_idx+1],
            orientations=main_traj_q[:main_idx+1], # xyzw
            color=Colors.GREEN,
            quat_format="xyzw", 
            label="main_history",
            line_width=3.0,
            # show_cameras=True # 可以选择显示历史相机
        )
        
        # 4. 添加独立的相机 (使用新API)
        # 添加一个静态相机
        static_cam_pose = np.identity(4, dtype=np.float32)
        static_cam_pose[0, 3] = 0.5 # x
        static_cam_pose[1, 3] = 0.2 # y
        static_cam_pose[2, 3] = 0.1 # z
        # 让它稍微旋转一下
        static_rot = R.from_euler('y', -np.pi / 6).as_matrix()
        static_cam_pose[:3, :3] = static_rot
        static_cam_id = viewer.add_camera_se3(
            pose_se3=static_cam_pose, 
            color=Colors.ORANGE, 
            label="static_cam", 
            scale=0.15, 
            line_width=2.0
        )

        # 添加螺旋线末端的相机（如果螺旋线正在显示）
        if current_helix_pose_se3 is not None:
            helix_end_cam_id = viewer.add_camera_se3(
                pose_se3=current_helix_pose_se3, 
                color=Colors.MAGENTA, 
                label="helix_end_cam", 
                scale=0.1, 
                line_width=1.5
            )
            
        # 添加主轨迹当前位置的相机（使用quat）
        current_t = main_traj_t[main_idx]
        current_q_xyzw = main_traj_q[main_idx]
        main_current_cam_id = viewer.add_camera_quat(
            position=current_t,
            orientation=current_q_xyzw, # xyzw
            color=Colors.YELLOW,
            quat_format="xyzw",
            label="main_current_cam",
            scale=0.12,
            line_width=2.0
        )

        # 5. 设置主相机 (演示切换)
        if frame_count % 150 < 50: # 前50帧跟随主轨迹当前相机
             viewer.set_main_camera(main_current_cam_id)
             follow_target = "Main Traj Cam"
        elif frame_count % 150 < 100: # 中间50帧跟随螺旋线末端相机
            if helix_end_cam_id is not None:
                 viewer.set_main_camera(helix_end_cam_id)
                 follow_target = "Helix End Cam"
            else: # 如果螺旋线结束了，就跟随静态相机
                viewer.set_main_camera(static_cam_id)
                follow_target = "Static Cam (Helix Ended)"
        else: # 后50帧跟随静态相机
            viewer.set_main_camera(static_cam_id)
            follow_target = "Static Cam"

        # 6. 添加点云
        viewer.add_points(cube_points, Colors.RED, "cube", 6.0)
        viewer.add_points_with_colors(sphere_points, sphere_colors, "sphere", 5.0)
        viewer.add_points(grid_points, Colors.GREEN, "grid", 4.0)
        
        # 6.5 新增：添加平面和直线
        # 添加一个半透明的蓝色矩形平面在XZ平面上
        plane_vertices = np.array([
            [-1.0, 0.0, -1.0],
            [ 1.0, 0.0, -1.0],
            [ 1.0, 0.0,  1.0],
            [-1.0, 0.0,  1.0]
        ], dtype=np.float32)
        viewer.add_plane(plane_vertices, color=Colors.BLUE, alpha=0.4, label="xz_plane")
        
        # 添加一个位于原点下方，法线朝向Y轴正方向，半边长为0.8的绿色半透明平面
        viewer.add_plane_normal_center(
            normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            center=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            size=0.8,
            color=Colors.GREEN,
            alpha=0.6,
            label="y_normal_plane"
        )
        
        # 添加一条从原点指向动态位置的粗黄色直线
        line_end_point = np.array([np.sin(frame_count * 0.05) * 1.5, 
                                   np.cos(frame_count * 0.05) * 1.5, 
                                   0.5], dtype=np.float32)
        viewer.add_line(np.zeros(3, dtype=np.float32), line_end_point, 
                      color=Colors.YELLOW, line_width=3.0, label="dynamic_line")
        
        # 7. 更新和显示图像 (使用新 API)
        img_copy = img.copy() # Start with the base image
        cv2.putText(img_copy, f"Frame: {frame_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img_copy, f"Following: {follow_target}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # 添加第一个图像 (使用Numpy数组)
        viewer.add_image_1(img=img_copy)
        
        # 添加第二个图像 (使用文件路径，测试缩放)
        # viewer.add_image_2(image_path='xxx')
        viewer.add_image_2(img=img_copy)
        
        # 8. 渲染一帧
        viewer.show(delay_time_in_s=0.03)

    # 清理测试图像文件
    # 以下代码已移除，test_img_path变量未定义
