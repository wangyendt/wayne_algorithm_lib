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
            # 修复：应该导入外部的pangolin_viewer，而不是自己
            Viewer = importlib.import_module("pangolin_viewer").PangolinViewer
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
    
    def add_plane_from_Twp(self, Twp, size, color=None, alpha=0.5, label=""):
        """
        通过变换矩阵添加3D平面
        
        使用4x4变换矩阵定义平面的位置和朝向。变换矩阵完全确定了平面在世界坐标系中的位姿。
        
        Args:
            Twp (np.ndarray): 平面在世界坐标系中的位姿矩阵，形状为(4, 4)。
                             这是一个SE(3)变换矩阵，定义了平面的位置和朝向
            size (float): 平面的半尺寸（从中心到边缘的距离）
            color (np.ndarray, optional): 平面颜色，RGB格式，默认为灰色
            alpha (float, optional): 透明度，默认为0.5
            label (str, optional): 平面在Pangolin中的标签，默认为空字符串
        
        Examples:
            >>> # 添加单位矩阵定义的平面（原点处，Z轴向上）
            >>> Twp = np.eye(4)
            >>> viewer.add_plane_from_Twp(Twp, 1.0)
            
            >>> # 添加旋转的平面
            >>> import pywayne.vio.SE3 as SE3
            >>> Twp = SE3.SE3_exp(np.array([1, 2, 3, 0, 0, np.pi/4]))
            >>> viewer.add_plane_from_Twp(Twp, 0.5, color=np.array([0, 1, 0]))
        """
        if color is None:
            color = Colors.GRAY
        
        # 从变换矩阵提取位置和朝向
        center = Twp[:3, 3].astype(np.float32)
        normal = Twp[:3, 2].astype(np.float32)  # Z轴作为法向量
        
        self.add_plane_normal_center(normal, center, size, color, alpha, label)
        
    def add_chessboard(self, rows=8, cols=8, cell_size=0.1, origin=None, normal=None, 
                       color1=None, color2=None, alpha=0.8, label="chessboard"):
        """
        绘制棋盘
        
        Args:
            rows: 行数，默认8
            cols: 列数，默认8
            cell_size: 单个格子的尺寸，默认0.1
            origin: 棋盘中间点坐标，默认为(0,0,0)
            normal: 棋盘法向量，默认为(0,0,1)，即XY平面
            color1: 第一种颜色（黑色格子），默认为黑色
            color2: 第二种颜色（白色格子），默认为白色
            alpha: 透明度，默认0.8
            label: 标签前缀，默认"chessboard"
        """
        if origin is None:
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            origin = origin.astype(np.float32)
            
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            normal = normal.astype(np.float32)
            normal = normal / np.linalg.norm(normal)  # 归一化
            
        if color1 is None:
            color1 = Colors.BLACK
        if color2 is None:
            color2 = Colors.WHITE
            
        # 构建局部坐标系
        # 找到一个与normal不平行的向量作为参考
        if abs(normal[2]) < 0.9:
            up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            up_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            
        # 构建正交基
        u_axis = np.cross(normal, up_ref)
        u_axis = u_axis / np.linalg.norm(u_axis)  # 归一化u轴
        v_axis = np.cross(normal, u_axis)  # v轴
        
        # 绘制每个格子
        for row in range(rows):
            for col in range(cols):
                # 计算格子中心
                u_offset = (col - cols/2.0 + 0.5) * cell_size
                v_offset = (row - rows/2.0 + 0.5) * cell_size
                cell_center = origin + u_offset * u_axis + v_offset * v_axis
                
                # 计算格子的四个顶点
                half_size = cell_size / 2.0
                vertices = np.array([
                    cell_center - half_size * u_axis - half_size * v_axis,
                    cell_center + half_size * u_axis - half_size * v_axis,
                    cell_center + half_size * u_axis + half_size * v_axis,
                    cell_center - half_size * u_axis + half_size * v_axis
                ], dtype=np.float32)
                
                # 根据行列位置选择颜色（棋盘模式）
                if (row + col) % 2 == 0:
                    color = color1  # 黑色格子
                else:
                    color = color2  # 白色格子
                    
                # 添加格子
                cell_label = f"{label}_r{row}_c{col}"
                self.add_plane(vertices, color=color, alpha=alpha, label=cell_label)
    
    def add_chessboard_from_Twp(self, rows=8, cols=8, cell_size=0.1, Twp=None, 
                               color1=None, color2=None, alpha=0.8, label="chessboard"):
        """
        通过变换矩阵绘制3D棋盘格
        
        使用4x4变换矩阵定义棋盘的位置和朝向。棋盘可以放置在3D空间的任意位置和朝向。
        每个格子都是独立的平面对象，具有层次化的标签命名。
        
        Args:
            rows (int, optional): 棋盘行数，默认为8
            cols (int, optional): 棋盘列数，默认为8
            cell_size (float, optional): 单个格子的边长尺寸，默认为0.1
            Twp (np.ndarray, optional): 棋盘中心在世界坐标系中的位姿矩阵，
                                       形状为(4, 4)，默认为单位矩阵（原点处）
            color1 (np.ndarray, optional): 第一种颜色（通常为黑色格子），
                                         RGB格式，默认为黑色
            color2 (np.ndarray, optional): 第二种颜色（通常为白色格子），
                                         RGB格式，默认为白色
            alpha (float, optional): 透明度，默认为0.8
            label (str, optional): 棋盘的基础标签，默认为'chessboard'。
                                 每个格子的标签为'{label}_r{row}_c{col}'
        
        Examples:
            >>> # 创建标准8x8棋盘
            >>> viewer.add_chessboard_from_Twp()
            
            >>> # 创建倾斜的大棋盘
            >>> import pywayne.vio.SE3 as SE3
            >>> Twp_tilted = SE3.SE3_exp(np.array([0, 0, 0, 0, np.pi/4, 0]))
            >>> viewer.add_chessboard_from_Twp(rows=10, cols=10, cell_size=0.2, 
            ...                               Twp=Twp_tilted, label='big_board')
            
            >>> # 创建红蓝配色的棋盘
            >>> viewer.add_chessboard_from_Twp(color1=np.array([1, 0, 0]),    # 红色
            ...                               color2=np.array([0, 0, 1]))     # 蓝色
        
        Note:
            - 棋盘以Twp指定的位置为中心对称分布
            - 格子按照国际象棋标准进行颜色交替（左上角为color1）
            - 每个格子在Pangolin中有独立的标签，便于单独操作
        """
        if Twp is None:
            Twp = np.eye(4)
        else:
            Twp = Twp.astype(np.float32)
            
        if color1 is None:
            color1 = Colors.BLACK
        if color2 is None:
            color2 = Colors.WHITE
        
        # 提取旋转矩阵和平移向量
        Rwp = Twp[:3, :3]
        twp = Twp[:3, 3]
        
        for row in range(rows):
            for col in range(cols):
                # 计算每个格子在棋盘局部坐标系中的偏移
                col_offset = (col - (cols - 1) / 2) * cell_size
                row_offset = (row - (rows - 1) / 2) * cell_size
                local_translation = np.array([col_offset, row_offset, 0], dtype=np.float32)
                
                # 创建格子的局部变换矩阵（只有平移，没有旋转）
                T_local = np.eye(4, dtype=np.float32)
                T_local[:3, 3] = local_translation
                
                # 计算格子在世界坐标系中的变换矩阵
                Twp_rc = Twp @ T_local
                
                # 选择颜色（棋盘模式）
                color = color1 if (row + col) % 2 == 0 else color2
                
                # 添加格子
                cell_label = f"{label}_r{row}_c{col}"
                self.add_plane_from_Twp(Twp_rc, cell_size/2, color=color, alpha=alpha, label=cell_label)
        
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
    viewer = PangolinViewer(800, 600, False)
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
                show_cameras=False, # 不在这里显示相机
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

        viewer.add_plane_from_Twp(
            Twp=np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0]
            ]),
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
        
        # 6.6 新增：添加棋盘演示        
        # 在XY平面添加棋盘
        viewer.add_chessboard(
            rows=12, 
            cols=12, 
            cell_size=0.08, 
            origin=np.array([0.0, -0.5, 1.0], dtype=np.float32),
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            alpha=0.8,
            label="colored_chessboard"
        )

        viewer.add_chessboard_from_Twp(
            rows=8,
            cols=8,
            cell_size=0.1,
            Twp=np.eye(4),
            color1=np.array([0, 0, 0]),
            color2=np.array([255, 255, 255]),
            alpha=0.8,
            label="chessboard"
        )
        
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
