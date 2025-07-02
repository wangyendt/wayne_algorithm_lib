import numpy as np
import rerun as rr
import cv2
from pathlib import Path
from typing import Union, Optional
from pywayne.tools import wayne_print
from pywayne.vio.SE3 import *
from pywayne.vio.SO3 import *


class RerunUtils:
    """
    Rerun可视化工具类
    
    提供了一系列静态方法来向Rerun查看器添加各种3D可视化元素，包括点云、轨迹、相机、
    平面和棋盘格等。该类封装了常用的3D可视化操作，简化了与Rerun SDK的交互。
    
    Examples:
        基本使用方法：
        >>> import rerun as rr
        >>> rr.init('example', spawn=True)
        >>> RerunUtils.add_point_cloud(points_3d, colors=[255, 0, 0])
        >>> RerunUtils.add_camera(camera_pose, label='main_camera')
    """

    @staticmethod
    def add_point_cloud(points_3d: np.ndarray, colors=None, label: str = 'point_cloud'):
        """
        向Rerun查看器添加3D点云
        
        将3D点云数据可视化为静态对象。支持自定义颜色和标签。
        
        Args:
            points_3d (np.ndarray): 3D点坐标数组，形状为(N, 3)，其中N是点的数量
            colors (Union[list, np.ndarray], optional): 点的颜色。可以是：
                - 单个RGB颜色列表：[R, G, B]，所有点使用相同颜色
                - 颜色数组：形状为(N, 3)，每个点对应一个RGB颜色
                - None：默认使用红色[255, 0, 0]
            label (str, optional): 点云在Rerun中的标签/路径，默认为'point_cloud'
        
        Examples:
            >>> # 添加红色点云
            >>> points = np.random.rand(100, 3)
            >>> RerunUtils.add_point_cloud(points)
            
            >>> # 添加多彩点云
            >>> colors = np.random.randint(0, 255, (100, 3))
            >>> RerunUtils.add_point_cloud(points, colors=colors, label='colored_points')
        """
        if colors is None:
            colors = [255, 0, 0]
        rr.log(
            label,
            rr.Points3D(
                positions=points_3d,
                colors=colors,
                # labels=label,
            ),
            static=True,
        )

    @staticmethod
    def add_trajectory(traj_endpoints: np.ndarray, colors=None, label: str = 'trajectory'):
        """
        向Rerun查看器添加3D轨迹线
        
        将3D轨迹可视化为连续的线段。轨迹以静态对象的形式显示。
        
        Args:
            traj_endpoints (np.ndarray): 轨迹点坐标数组，形状为(N, 3)，其中N是轨迹点数量。
                                       相邻点之间会自动连线形成轨迹
            colors (Union[list, np.ndarray], optional): 轨迹线颜色。可以是：
                - 单个RGB颜色列表：[R, G, B]，整条轨迹使用相同颜色
                - 颜色数组：为轨迹的不同段指定颜色
                - None：默认使用绿色[0, 255, 0]
            label (str, optional): 轨迹在Rerun中的标签/路径，默认为'trajectory'
        
        Examples:
            >>> # 添加绿色轨迹
            >>> trajectory = np.cumsum(np.random.rand(50, 3) * 0.1, axis=0)
            >>> RerunUtils.add_trajectory(trajectory)
            
            >>> # 添加蓝色轨迹
            >>> RerunUtils.add_trajectory(trajectory, colors=[0, 0, 255], label='blue_path')
        """
        if colors is None:
            colors = [0, 255, 0]
        rr.log(
            label,
            rr.LineStrips3D(
                traj_endpoints,
                colors=colors,
                # labels=label,
            ),
            static=True,
        )

    @classmethod
    def add_camera(cls, camera_pose: np.ndarray, image: Optional[Union[np.ndarray, str, Path]] = None, label: str = 'camera'):
        """
        向Rerun查看器添加3D相机
        
        在3D空间中可视化相机的位置、朝向和图像内容。相机以针孔模型显示，
        可选择性地显示相机拍摄的图像。
        
        Args:
            camera_pose (np.ndarray): 相机在世界坐标系中的位姿矩阵，形状为(4, 4)。
                                    这是一个SE(3)变换矩阵，包含旋转和平移信息
            image (Union[np.ndarray, str, Path], optional): 相机图像。可以是：
                - np.ndarray：直接的图像数组，形状为(H, W, 3)
                - str或Path：图像文件路径
                - None：不显示图像
            label (str, optional): 相机在Rerun中的标签/路径，默认为'camera'
        
        Returns:
            str: 返回相机的标签名称
        
        Examples:
            >>> # 添加基本相机
            >>> pose = np.eye(4)
            >>> RerunUtils.add_camera(pose, label='main_camera')
            
            >>> # 添加带图像的相机
            >>> RerunUtils.add_camera(pose, image='path/to/image.jpg', label='rgb_camera')
        """
        if isinstance(image, np.ndarray):
            pass
        elif isinstance(image, str) or isinstance(image, Path):
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = None
        if image is not None:
            img_height, img_width = image.shape[:2]
        else:
            img_height, img_width = 480, 640

        # 计算位置和旋转
        translation = camera_pose[..., :3, 3]
        rotation = SO3.SO3_to_quat(camera_pose[..., :3, :3])[[1,2,3,0]]

        rr.log(
            f'{label}',
            rr.Transform3D(
                translation=translation, 
                rotation=rr.Quaternion(xyzw=rotation),
            ),
            rr.Pinhole(
                width=img_width, height=img_height,
                focal_length=500.0, 
                camera_xyz=rr.ViewCoordinates.RDF, 
                image_plane_distance=0.3,
            ),
        )

        if image is not None:
            rr.log(f'{label}', rr.Image(image))
        return label
    
    @staticmethod
    def _get_quaternion_from_v1_and_v2(v1: np.ndarray, v2: np.ndarray):
        """
        计算将向量v1旋转到向量v2的四元数
        
        使用Rodrigues旋转公式计算最短路径旋转四元数，将单位向量v1旋转到v2的方向。
        这是一个内部辅助方法，主要用于平面法向量的旋转计算。
        
        Args:
            v1 (np.ndarray): 源向量，形状为(3,)
            v2 (np.ndarray): 目标向量，形状为(3,)
        
        Returns:
            np.ndarray: 四元数，格式为[x, y, z, w]，表示从v1到v2的旋转
        
        Raises:
            ValueError: 当输入向量为零向量时抛出异常
        
        Note:
            - 输入向量会自动归一化
            - 使用Rodrigues公式处理一般情况，特殊处理平行和反平行向量
            - 返回的四元数格式为XYZW（Rerun所需格式）
        """
        v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
        if v1_norm == 0.0 or v2_norm == 0.0:
            raise ValueError('v1 or v2 is zero vector')
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        c = np.dot(v1, v2)
        k = np.cross(v1, v2)
        R = np.eye(3)
        if np.isclose(c, 1.0):
            pass
        elif np.isclose(c, -1.0):
            R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        else:
            K = SO3.SO3_skew(k)
            R = np.eye(3) + K + K @ K * (1.0 / (1.0 + c))
        return SO3.SO3_to_quat(R)[[1, 2, 3, 0]] # x, y, z, w
    
    @classmethod
    def add_plane_from_center_and_normal(cls, center: np.ndarray, normal: np.ndarray, half_size: float, color: np.ndarray = np.array([0, 0, 255]), label: str = 'plane'):
        """
        通过中心点和法向量添加3D平面
        
        创建一个以指定点为中心、具有特定法向量的矩形平面。平面朝向由法向量确定。
        
        Args:
            center (np.ndarray): 平面中心点坐标，形状为(3,)
            normal (np.ndarray): 平面法向量，形状为(3,)，会自动归一化
            half_size (float): 平面的半尺寸（从中心到边缘的距离）
            color (np.ndarray, optional): 平面颜色，RGB格式，默认为蓝色[0, 0, 255]
            label (str, optional): 平面在Rerun中的标签/路径，默认为'plane'
        
        Examples:
            >>> # 添加水平平面
            >>> center = np.array([0, 0, 0])
            >>> normal = np.array([0, 0, 1])  # 垂直向上
            >>> RerunUtils.add_plane_from_center_and_normal(center, normal, 1.0)
            
            >>> # 添加倾斜平面
            >>> normal = np.array([1, 1, 1])  # 倾斜法向量
            >>> RerunUtils.add_plane_from_center_and_normal(center, normal, 0.5, 
            ...                                           color=np.array([255, 0, 0]))
        """
        q = cls._get_quaternion_from_v1_and_v2(np.array([0.0, 0.0, 1.0]), normal)
        rr.log(
            label,
            rr.Boxes3D(
                centers=center,
                half_sizes=[half_size, half_size, 0.0],
                quaternions=rr.Quaternion(xyzw=q),
                colors=color,
                fill_mode='solid'
            ),
        )

    @classmethod
    def add_plane_from_Twp(cls, Twp: np.ndarray, half_size: float, color: np.ndarray = np.array([0, 0, 255]), label: str = 'plane'):
        """
        通过变换矩阵添加3D平面
        
        使用4x4变换矩阵定义平面的位置和朝向。变换矩阵完全确定了平面在世界坐标系中的位姿。
        
        Args:
            Twp (np.ndarray): 平面在世界坐标系中的位姿矩阵，形状为(4, 4)。
                             这是一个SE(3)变换矩阵，定义了平面的位置和朝向
            half_size (float): 平面的半尺寸（从中心到边缘的距离）
            color (np.ndarray, optional): 平面颜色，RGB格式，默认为蓝色[0, 0, 255]
            label (str, optional): 平面在Rerun中的标签/路径，默认为'plane'
        
        Examples:
            >>> # 添加单位矩阵定义的平面（原点处，Z轴向上）
            >>> Twp = np.eye(4)
            >>> RerunUtils.add_plane_from_Twp(Twp, 1.0)
            
            >>> # 添加旋转的平面
            >>> Twp = SE3.SE3_exp(np.array([1, 2, 3, 0, 0, np.pi/4]))
            >>> RerunUtils.add_plane_from_Twp(Twp, 0.5, color=np.array([0, 255, 0]))
        """
        Rwp, twp = Twp[:3, :3], Twp[:3, 3]
        rr.log(
            label,
            rr.Boxes3D(
                centers=twp,
                half_sizes=[half_size, half_size, 0.0],
                quaternions=rr.Quaternion(xyzw=SO3.SO3_to_quat(Rwp)[[1, 2, 3, 0]]),
                colors=color,
                fill_mode='solid'
            )
        )

    @classmethod
    def add_chessboard_from_Twp(cls, rows=8, cols=8, cell_size=0.1, Twp: np.ndarray=np.eye(4), color1=None, color2=None, label: str = 'chessboard'):
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
                                         RGB格式，默认为黑色[0, 0, 0]
            color2 (np.ndarray, optional): 第二种颜色（通常为白色格子），
                                         RGB格式，默认为白色[255, 255, 255]
            label (str, optional): 棋盘的基础标签，默认为'chessboard'。
                                 每个格子的标签为'{label}/{row}_{col}'
        
        Examples:
            >>> # 创建标准8x8棋盘
            >>> RerunUtils.add_chessboard_from_Twp()
            
            >>> # 创建倾斜的大棋盘
            >>> Twp_tilted = SE3.SE3_exp(np.array([0, 0, 0, 0, np.pi/4, 0]))
            >>> RerunUtils.add_chessboard_from_Twp(rows=10, cols=10, cell_size=0.2, 
            ...                                   Twp=Twp_tilted, label='big_board')
            
            >>> # 创建红蓝配色的棋盘
            >>> RerunUtils.add_chessboard_from_Twp(color1=np.array([255, 0, 0]),    # 红色
            ...                                   color2=np.array([0, 0, 255]))     # 蓝色
        
        Note:
            - 棋盘以Twp指定的位置为中心对称分布
            - 格子按照国际象棋标准进行颜色交替（左上角为color1）
            - 每个格子在Rerun中有独立的标签，便于单独操作
        """
        Rwp, twp = Twp[:3, :3], Twp[:3, 3]

        if color1 is None:
            color1 = np.array([0, 0, 0])
        if color2 is None:
            color2 = np.array([255, 255, 255])
        
        for row in range(rows):
            for col in range(cols):
                # 计算每个格子在棋盘局部坐标系中的偏移
                col_offset = (col - (cols - 1) / 2) * cell_size  
                row_offset = (row - (rows - 1) / 2) * cell_size
                local_translation = np.array([col_offset, row_offset, 0])
                
                # 创建格子的局部变换矩阵（只有平移，没有旋转）
                T_local = np.eye(4)
                T_local[:3, 3] = local_translation
                
                # 计算格子在世界坐标系中的变换矩阵
                Twp_rc = Twp @ T_local
                
                # 添加格子
                cls.add_plane_from_Twp(Twp_rc, cell_size/2, color1 if (row + col) % 2 == 0 else color2, f'{label}/{row}_{col}')


if __name__ == '__main__':
    rr.init('world', spawn=True)
    # rr.log('world', rr.ViewCoordinates.RDF, static=True)
    def generate_random_walk_points3d(num_points: int):
        points_3d = np.random.rand(num_points, 3) * 0.1
        return np.cumsum(points_3d, axis=0)
    points_3d = generate_random_walk_points3d(100)
    points_3d -= np.mean(points_3d, axis=0)
    
    # 生成图像
    height, width = 480, 640
    
    # 创建第一个图像：蓝色背景，画圆和文字
    img1 = np.zeros((height, width, 3), dtype=np.uint8)
    img1[:, :] = [255, 150, 100]  # 蓝色背景 (BGR格式)
    # 画一个红色圆圈
    cv2.circle(img1, (320, 240), 80, (0, 0, 255), -1)  # 红色圆圈 (BGR格式)
    # 画一个绿色矩形
    cv2.rectangle(img1, (100, 100), (200, 200), (0, 255, 0), 3)  # 绿色矩形 (BGR格式)
    # 添加文字
    cv2.putText(img1, 'Camera 1', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img1, 'Rerun Demo', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    # 转换为RGB格式（供Rerun使用）
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    # 创建第二个图像：绿色背景，画线条和文字
    img2 = np.zeros((height, width, 3), dtype=np.uint8)
    img2[:, :] = [150, 255, 150]  # 绿色背景 (BGR格式)
    # 画对角线
    cv2.line(img2, (0, 0), (width, height), (255, 0, 255), 5)  # 紫色对角线 (BGR格式)
    cv2.line(img2, (width, 0), (0, height), (0, 255, 255), 5)  # 黄色对角线 (BGR格式)
    # 画椭圆
    cv2.ellipse(img2, (320, 240), (120, 60), 45, 0, 360, (0, 0, 255), 3)  # 红色椭圆 (BGR格式)
    # 添加文字
    cv2.putText(img2, 'Camera 2', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img2, 'Generated Image', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # 转换为RGB格式（供Rerun使用）
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 初始化相机
    camera1 = RerunUtils.add_camera(np.eye(4), image=img1, label='camera1')
    camera2 = RerunUtils.add_camera(np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]), image=img2, label='camera2')
    RerunUtils.add_plane_from_center_and_normal(np.array([0.0, 0.0, 0.0]), np.array([0.0, -np.sqrt(2)/2, np.sqrt(2)/2]), 1.0)
    RerunUtils.add_chessboard_from_Twp(8, 8, 0.1, np.eye(4), np.array([0, 0, 0]), np.array([255, 255, 255]))
    
    
    RerunUtils.add_point_cloud(points_3d)
    RerunUtils.add_trajectory(points_3d)


