"""
SE3 工具函数库 - 刚体变换操作

本库提供了完整的SE(3)刚体变换操作，包括：
- 基础变换：构造、验证、相乘、求逆
- 表示转换：四元数、轴角、欧拉角等
- 李群李代数映射：log/Log, exp/Exp

李群李代数函数命名规范：
- SE3_Log: SE(3) → 李代数向量 (6维向量 [ρ, θ])
- SE3_log: SE(3) → 李代数矩阵 (4×4 反对称形式)
- SE3_Exp: 李代数向量 → SE(3)
- SE3_exp: 李代数矩阵 → SE(3)

使用示例：

# 1. 基础变换操作
import numpy as np
from pywayne.vio.SE3 import *

# 构造SE3变换矩阵
R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # 旋转矩阵
t = np.array([1, 2, 3])  # 平移向量
T = SE3_from_Rt(R, t)

# 验证SE3矩阵
is_valid = check_SE3(T)

# 矩阵相乘和求逆
T1 = SE3_from_Rt(np.eye(3), [1, 0, 0])
T2 = SE3_from_Rt(np.eye(3), [0, 1, 0])
T_combined = SE3_mul(T1, T2)
T_inv = SE3_inv(T)

# 2. 李群李代数映射
# 向量形式 (常用)
xi = np.array([0.1, 0.2, 0.3, 0.05, 0.1, 0.15])  # [ρ, θ]
T_from_xi = SE3_Exp(xi)  # 李代数向量 → SE(3)
xi_recovered = SE3_Log(T_from_xi)  # SE(3) → 李代数向量

# 矩阵形式 (理论计算)
xi_hat = SE3_skew(xi)  # 6维向量 → 4×4李代数矩阵
T_from_hat = SE3_exp(xi_hat)  # 李代数矩阵 → SE(3)
xi_hat_recovered = SE3_log(T_from_hat)  # SE(3) → 李代数矩阵

# 3. 批量处理
n = 100
# 批量李代数向量
xi_batch = np.random.randn(n, 6) * 0.1
T_batch = SE3_Exp(xi_batch)  # 批量指数映射
xi_batch_recovered = SE3_Log(T_batch)  # 批量对数映射

# 4. 表示转换
q = np.array([[1, 0, 0, 0]])  # 四元数 (w,x,y,z)
t = np.array([[1, 2, 3]])    # 平移
T_from_qt = SE3_from_quat_trans(q, t)
q_recovered, t_recovered = SE3_to_quat_trans(T_from_qt)

# 5. 平均变换
T_list = [SE3_Exp(np.random.randn(6) * 0.1) for _ in range(10)]
T_matrices = np.array(T_list)
T_mean = SE3_mean(T_matrices)

性能参考 (1000个变换):
- SE3_Exp: ~2.5 ms
- SE3_Log: ~0.8 ms  
- SE3_exp: ~2.5 ms
- SE3_log: ~0.9 ms
"""

import math
import os
import sys
import time

import numpy as np

# 添加当前目录到路径并导入SO3
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from pywayne.tools import wayne_print  # noqa: E402

# 直接导入SO3文件
import SO3  # noqa: E402

def check_SE3(T: np.ndarray) -> bool:
    """
    Check if a matrix is a valid SE(3) transformation matrix.
    Args:
        T: 4x4 transformation matrix
    Returns:
        True if T is a valid SE(3) transformation matrix, False otherwise
    """
    if T.shape != (4, 4):
        return False
    
    # 检查旋转部分是否为有效的SO(3)
    R = T[:3, :3]
    if not SO3.check_SO3(R):
        return False
    
    # 检查底部行是否为 [0, 0, 0, 1]
    if not np.allclose(T[3, :], [0, 0, 0, 1]):
        return False
    
    return True

def SE3_skew(xi: np.ndarray) -> np.ndarray:
    """
    Convert a 6D vector to SE(3) Lie algebra (4x4 skew-symmetric form).
    Args:
        xi: Nx6 vector [ρ, θ] where ρ is translation part, θ is rotation part
    Returns:
        skew_mat: Nx4x4 SE(3) Lie algebra matrix
    """
    # 确保输入是二维的
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
        single_matrix = True
    else:
        single_matrix = False
    
    # 使用einsum构建SE(3)李代数矩阵
    # 创建4x4模板
    se3_template = np.zeros((4, 4, 6))
    
    # 设置旋转部分的模板（SO(3)反对称矩阵部分）
    epsilon = np.array([[[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                       [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                       [[0, -1, 0], [1, 0, 0], [0, 0, 0]]])
    
    # 将epsilon放入4x4模板的左上3x3部分，对应theta部分
    se3_template[:3, :3, 3:] = epsilon
    
    # 设置平移部分的模板（右上角部分）
    se3_template[:3, 3, :3] = np.eye(3)
    
    # 使用einsum计算
    skew_mats = np.einsum('ijk,nk->nij', se3_template, xi)
    
    if single_matrix:
        return skew_mats[0]
    return skew_mats

def SE3_mul(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Multiply two SE(3) transformation matrices.
    Args:
        T1: 4x4 transformation matrix
        T2: 4x4 transformation matrix
    Returns:
        T1 @ T2
    """
    return T1 @ T2

def SE3_diff(T1: np.ndarray, T2: np.ndarray, from_1_to_2: bool = True) -> np.ndarray:
    """
    Compute the difference between two SE(3) transformation matrices.
    Args:
        T1: 4x4 transformation matrix
        T2: 4x4 transformation matrix
        from_1_to_2: if True, compute T1^(-1) @ T2, otherwise compute T2^(-1) @ T1
    Returns:
        SE(3) difference matrix
    """
    if from_1_to_2:
        return np.linalg.inv(T1) @ T2
    else:
        return np.linalg.inv(T2) @ T1

def SE3_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Construct SE(3) matrix from rotation matrix and translation vector.
    Args:
        R: Nx3x3 rotation matrices or 3x3 rotation matrix
        t: Nx3 translation vectors or 3 translation vector
    Returns:
        T: Nx4x4 or 4x4 SE(3) transformation matrices
    """
    # 处理输入维度
    if R.ndim == 2:
        R = R.reshape(1, 3, 3)
        single_matrix = True
    else:
        single_matrix = False
    
    if t.ndim == 1:
        t = t.reshape(1, -1)
    
    N = R.shape[0]
    T = np.zeros((N, 4, 4))
    
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1
    
    if single_matrix:
        return T[0]
    return T

def SE3_to_Rt(T: np.ndarray) -> tuple:
    """
    Extract rotation matrix and translation vector from SE(3) matrix.
    Args:
        T: Nx4x4 or 4x4 SE(3) transformation matrices
    Returns:
        R: Nx3x3 or 3x3 rotation matrices
        t: Nx3 or 3 translation vectors
    """
    if T.ndim == 2:
        return T[:3, :3], T[:3, 3]
    else:
        return T[:, :3, :3], T[:, :3, 3]

def SE3_from_quat_trans(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert quaternion and translation to SE(3) transformation matrix.
    Args:
        q: Nx4 quaternion, wxyz, Hamilton convention
        t: Nx3 translation vector
    Returns:
        T: Nx4x4 SE(3) transformation matrix
    """
    R = SO3.SO3_from_quat(q)
    return SE3_from_Rt(R, t)

def SE3_to_quat_trans(T: np.ndarray) -> tuple:
    """
    Convert SE(3) transformation matrix to quaternion and translation.
    Args:
        T: Nx4x4 SE(3) transformation matrix
    Returns:
        q: Nx4 quaternion, wxyz, Hamilton convention
        t: Nx3 translation vector
    """
    R, t = SE3_to_Rt(T)
    q = SO3.SO3_to_quat(R)
    return q, t

def SE3_from_axis_angle_trans(axis: np.ndarray, angle: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle and translation to SE(3) transformation matrix.
    Args:
        axis: Nx3 rotation axis
        angle: Nx1 rotation angle
        t: Nx3 translation vector
    Returns:
        T: Nx4x4 SE(3) transformation matrix
    """
    R = SO3.SO3_from_axis_angle(axis, angle)
    return SE3_from_Rt(R, t)

def SE3_to_axis_angle_trans(T: np.ndarray) -> tuple:
    """
    Convert SE(3) transformation matrix to axis-angle and translation.
    Args:
        T: Nx4x4 SE(3) transformation matrix
    Returns:
        axis: Nx3 rotation axis
        angle: Nx1 rotation angle
        t: Nx3 translation vector
    """
    R, t = SE3_to_Rt(T)
    axis, angle = SO3.SO3_to_axis_angle(R)
    return axis, angle, t

def SE3_from_euler_trans(euler_angles: np.ndarray, t: np.ndarray, axes: str = 'zyx', intrinsic: bool = True) -> np.ndarray:
    """
    Convert Euler angles and translation to SE(3) transformation matrix.
    Args:
        euler_angles: Nx3 Euler angles, rad
        t: Nx3 translation vector
        axes: Euler angles sequence
        intrinsic: if True, use intrinsic rotation, otherwise use extrinsic rotation
    Returns:
        T: Nx4x4 SE(3) transformation matrix
    """
    R = SO3.SO3_from_euler(euler_angles, axes, intrinsic)
    return SE3_from_Rt(R, t)

def SE3_to_euler_trans(T: np.ndarray, axes: str = 'zyx', intrinsic: bool = True) -> tuple:
    """
    Convert SE(3) transformation matrix to Euler angles and translation.
    Args:
        T: Nx4x4 SE(3) transformation matrix
        axes: Euler angles sequence
        intrinsic: if True, use intrinsic rotation, otherwise use extrinsic rotation
    Returns:
        euler_angles: Nx3 Euler angles, rad
        t: Nx3 translation vector
    """
    R, t = SE3_to_Rt(T)
    euler_angles = SO3.SO3_to_euler(R, axes, intrinsic)
    return euler_angles, t

def SE3_Log(T: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) transformation matrix to Lie algebra vector.
    Args:
        T: Nx4x4 or 4x4 SE(3) transformation matrix
    Returns:
        xi: Nx6 or 6 SE(3) Lie algebra vector [ρ, θ]
    """
    if T.ndim == 2:
        T = T.reshape(1, 4, 4)
        single_vector = True
    else:
        single_vector = False
    
    N = T.shape[0]
    xi = np.zeros((N, 6))
    
    # 批量提取旋转和平移
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    
    # 批量计算旋转的对数映射
    theta_vecs = SO3.SO3_Log(R)  # Nx3
    thetas = np.linalg.norm(theta_vecs, axis=1)  # N,
    
    # 矢量化处理小角度和大角度情况
    small_angle_mask = thetas < 1e-6
    large_angle_mask = ~small_angle_mask
    
    # 对于小角度情况
    if np.any(small_angle_mask):
        small_theta_vecs = theta_vecs[small_angle_mask]
        small_theta_skews = SO3.SO3_skew(small_theta_vecs)  # Mx3x3
        if small_theta_skews.ndim == 2:
            small_theta_skews = small_theta_skews.reshape(1, 3, 3)
        
        # V_inv = I - 0.5 * theta_skew
        V_inv_small = np.eye(3) - 0.5 * small_theta_skews
        
        # 计算平移部分
        small_t = t[small_angle_mask]
        rho_small = np.einsum('nij,nj->ni', V_inv_small, small_t)
        
        xi[small_angle_mask, :3] = rho_small
        xi[small_angle_mask, 3:] = small_theta_vecs
    
    # 对于大角度情况
    if np.any(large_angle_mask):
        large_theta_vecs = theta_vecs[large_angle_mask]
        large_thetas = thetas[large_angle_mask]
        large_theta_skews = SO3.SO3_skew(large_theta_vecs)  # Mx3x3
        if large_theta_skews.ndim == 2:
            large_theta_skews = large_theta_skews.reshape(1, 3, 3)
        
        # 计算V^(-1)矩阵
        cos_thetas = np.cos(large_thetas)
        sin_thetas = np.sin(large_thetas)
        
        # 矢量化计算系数
        coeffs = (2 * sin_thetas - large_thetas * (1 + cos_thetas)) / (2 * large_thetas**2 * sin_thetas)
        
        # V_inv = I - 0.5 * theta_skew + coeff * (theta_skew @ theta_skew)
        I_batch = np.tile(np.eye(3), (len(large_thetas), 1, 1))
        theta_skew_squared = np.einsum('nij,njk->nik', large_theta_skews, large_theta_skews)
        
        V_inv_large = (I_batch - 0.5 * large_theta_skews + 
                      coeffs[:, np.newaxis, np.newaxis] * theta_skew_squared)
        
        # 计算平移部分
        large_t = t[large_angle_mask]
        rho_large = np.einsum('nij,nj->ni', V_inv_large, large_t)
        
        xi[large_angle_mask, :3] = rho_large
        xi[large_angle_mask, 3:] = large_theta_vecs
    
    if single_vector:
        return xi[0]
    return xi

def SE3_log(T: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) transformation matrix to Lie algebra matrix.
    Args:
        T: Nx4x4 or 4x4 SE(3) transformation matrix
    Returns:
        xi_hat: Nx4x4 or 4x4 SE(3) Lie algebra matrix
    """
    xi_vec = SE3_Log(T)
    return SE3_skew(xi_vec)

def SE3_Exp(xi: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) Lie algebra vector to transformation matrix.
    Args:
        xi: Nx6 or 6 SE(3) Lie algebra vector [ρ, θ]
    Returns:
        T: Nx4x4 or 4x4 SE(3) transformation matrix
    """
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
        single_matrix = True
    else:
        single_matrix = False
    
    N = xi.shape[0]
    T = np.zeros((N, 4, 4))
    
    # 分离平移和旋转部分
    rho = xi[:, :3]  # 平移部分
    theta_vecs = xi[:, 3:]  # 旋转部分
    thetas = np.linalg.norm(theta_vecs, axis=1)
    
    # 处理小角度和大角度情况
    small_angle_mask = thetas < 1e-6
    large_angle_mask = ~small_angle_mask
    
    # 对于小角度情况
    if np.any(small_angle_mask):
        small_rho = rho[small_angle_mask]
        small_theta_vecs = theta_vecs[small_angle_mask]
        small_theta_skews = SO3.SO3_skew(small_theta_vecs)
        if small_theta_skews.ndim == 2:
            small_theta_skews = small_theta_skews.reshape(1, 3, 3)
        
        # 小角度近似
        R_small = np.eye(3) + small_theta_skews
        V_small = np.eye(3) + 0.5 * small_theta_skews
        
        # 计算平移
        t_small = np.einsum('nij,nj->ni', V_small, small_rho)
        
        # 构建变换矩阵
        for i, idx in enumerate(np.where(small_angle_mask)[0]):
            T[idx] = SE3_from_Rt(R_small[i], t_small[i])
    
    # 对于大角度情况
    if np.any(large_angle_mask):
        large_rho = rho[large_angle_mask]
        large_theta_vecs = theta_vecs[large_angle_mask]
        large_thetas = thetas[large_angle_mask]
        
        # 使用SO3_Exp计算旋转矩阵
        R_large = SO3.SO3_Exp(large_theta_vecs)
        if R_large.ndim == 2:
            R_large = R_large.reshape(1, 3, 3)
        
        # 计算V矩阵
        large_theta_skews = SO3.SO3_skew(large_theta_vecs)
        if large_theta_skews.ndim == 2:
            large_theta_skews = large_theta_skews.reshape(1, 3, 3)
        
        cos_thetas = np.cos(large_thetas)
        sin_thetas = np.sin(large_thetas)
        
        # V = I + (1-cos(θ))/θ² * θ_skew + (θ-sin(θ))/θ³ * θ_skew²
        I_batch = np.tile(np.eye(3), (len(large_thetas), 1, 1))
        theta_skew_squared = np.einsum('nij,njk->nik', large_theta_skews, large_theta_skews)
        
        coeff1 = (1 - cos_thetas) / (large_thetas**2)
        coeff2 = (large_thetas - sin_thetas) / (large_thetas**3)
        
        V_large = (I_batch + 
                  coeff1[:, np.newaxis, np.newaxis] * large_theta_skews + 
                  coeff2[:, np.newaxis, np.newaxis] * theta_skew_squared)
        
        # 计算平移
        t_large = np.einsum('nij,nj->ni', V_large, large_rho)
        
        # 构建变换矩阵
        for i, idx in enumerate(np.where(large_angle_mask)[0]):
            T[idx] = SE3_from_Rt(R_large[i], t_large[i])
    
    # 设置齐次坐标部分
    T[:, 3, 3] = 1
    
    if single_matrix:
        return T[0]
    return T

def SE3_exp(xi_hat: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) Lie algebra matrix to transformation matrix.
    Args:
        xi_hat: Nx4x4 or 4x4 SE(3) Lie algebra matrix
    Returns:
        T: Nx4x4 or 4x4 SE(3) transformation matrix
    """
    # 从李代数矩阵提取6维向量
    xi_vec = SE3_unskew(xi_hat)
    
    # 使用SE3_Exp转换
    return SE3_Exp(xi_vec)

def SE3_mean(T: np.ndarray) -> np.ndarray:
    """
    Compute the mean of multiple SE(3) transformation matrices using scipy.
    Args:
        T: Nx4x4 transformation matrices
    Returns:
        mean_T: 4x4 mean transformation matrix
    """
    if T.ndim == 2:
        # 单个矩阵，直接返回
        return T
    
    # 基于scipy的方法，分别处理旋转和平移
    R, t = SE3_to_Rt(T)
    
    # 计算旋转的平均
    mean_R = SO3.SO3_mean(R)
    
    # 计算平移的平均
    mean_t = np.mean(t, axis=0)
    
    return SE3_from_Rt(mean_R, mean_t)

def SE3_inv(T: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of SE(3) transformation matrix.
    Args:
        T: Nx4x4 or 4x4 SE(3) transformation matrix
    Returns:
        T_inv: Nx4x4 or 4x4 inverse transformation matrix
    """
    T = np.asarray(T)
    single_transform = T.ndim == 2
    if single_transform:
        if T.shape != (4, 4):
            raise ValueError(f"single SE(3) input must have shape (4, 4), got {T.shape}")
        batch = T[None, ...]
    elif T.ndim == 3 and T.shape[1:] == (4, 4):
        batch = T
    else:
        raise ValueError(f"SE(3) input must have shape (4, 4) or (N, 4, 4), got {T.shape}")

    output_dtype = np.result_type(batch.dtype, np.float64)
    inverse = np.zeros(batch.shape, dtype=output_dtype)
    rotation_transpose = np.swapaxes(batch[:, :3, :3], 1, 2)
    inverse[:, :3, :3] = rotation_transpose
    inverse[:, :3, 3] = -np.einsum(
        "nij,nj->ni",
        rotation_transpose,
        batch[:, :3, 3],
    )
    inverse[:, 3, 3] = 1.0
    return inverse[0] if single_transform else inverse

def SE3_unskew(xi_hat: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) Lie algebra matrix to 6D vector.
    Args:
        xi_hat: Nx4x4 or 4x4 SE(3) Lie algebra matrix
    Returns:
        xi: Nx6 or 6 vector [ρ, θ] where ρ is translation part, θ is rotation part
    """
    # 确保输入是三维的
    if xi_hat.ndim == 2:
        xi_hat = xi_hat.reshape(1, 4, 4)
        single_vector = True
    else:
        single_vector = False
    
    N = xi_hat.shape[0]
    xi = np.zeros((N, 6))
    
    # 提取平移部分（右上角3x1）
    xi[:, :3] = xi_hat[:, :3, 3]
    
    # 提取旋转部分（左上角3x3反对称矩阵转换为向量）
    theta_skew = xi_hat[:, :3, :3]
    xi[:, 3:] = SO3.SO3_unskew(theta_skew)
    
    if single_vector:
        return xi[0]
    return xi

if __name__ == '__main__':
    wayne_print("🔧 SE3 工具函数全面测试", "cyan", bold=True)
    print("=" * 60)
    
    # ================================
    # 1. 测试基础工具函数
    # ================================
    wayne_print("\n📐 1. 基础工具函数测试", "blue", bold=True)
    
    # 测试 SE3_skew
    wayne_print("   [1.1] SE3_skew - SE(3)李代数", "yellow")
    xi1 = np.array([1, 2, 3, 0.1, 0.2, 0.3])  # [ρ, θ]
    skew1 = SE3_skew(xi1)
    print(f"   输入向量: {xi1}")
    print(f"   SE(3)李代数矩阵形状: {skew1.shape}")
    print(f"   矩阵:\n{skew1}")
    
    # 测试批量SE3_skew
    xi2 = np.array([[1, 2, 3, 0.1, 0.2, 0.3], [4, 5, 6, 0.4, 0.5, 0.6]])
    skew2 = SE3_skew(xi2)
    print(f"   批量输入形状: {xi2.shape}")
    print(f"   批量输出形状: {skew2.shape}")
    
    # 测试 SE3_unskew
    wayne_print("   [1.1.1] SE3_unskew - SE(3)李代数转向量", "yellow")
    xi1_recovered = SE3_unskew(skew1)
    xi2_recovered = SE3_unskew(skew2)
    unskew_error1 = np.linalg.norm(xi1 - xi1_recovered)
    unskew_error2 = np.linalg.norm(xi2 - xi2_recovered)
    print(f"   单个向量往返误差: {unskew_error1:.2e}")
    print(f"   批量向量往返误差: {unskew_error2:.2e}")
    print(f"   恢复向量1: {xi1_recovered}")
    print(f"   恢复向量2形状: {xi2_recovered.shape}")
    
    # 测试 check_SE3
    wayne_print("   [1.2] check_SE3 - 验证变换矩阵", "yellow")
    T_identity = np.eye(4)
    print(f"   单位矩阵是否为SE3: {check_SE3(T_identity)}")
    print(f"   李代数矩阵是否为SE3: {check_SE3(skew1)}")
    
    # 创建有效的SE3矩阵进行测试
    angle_30 = math.pi / 6  # 精确的30度
    R_test = np.array([
        [math.cos(angle_30), -math.sin(angle_30), 0],
        [math.sin(angle_30),  math.cos(angle_30), 0],
        [0, 0, 1]
    ])
    t_test = np.array([1, 2, 3])
    T_test = SE3_from_Rt(R_test, t_test)
    print(f"   构造的SE3矩阵是否有效: {check_SE3(T_test)}")
    
    # 测试 SE3_mul
    wayne_print("   [1.3] SE3_mul - 变换矩阵相乘", "yellow")
    T1 = SE3_from_Rt(R_test, t_test)
    T2 = SE3_from_Rt(np.eye(3), np.array([0, 0, 1]))
    T_mul = SE3_mul(T1, T2)
    print(f"   T1 @ T2 是否为有效SE3: {check_SE3(T_mul)}")
    
    # 测试 SE3_inv
    wayne_print("   [1.4] SE3_inv - 变换矩阵求逆", "yellow")
    T_inv = SE3_inv(T_test)
    T_identity_check = SE3_mul(T_test, T_inv)
    identity_error = np.linalg.norm(T_identity_check - np.eye(4))
    print(f"   T @ T^(-1) 与单位矩阵误差: {identity_error:.2e}")
    
    # ================================
    # 2. 测试转换函数
    # ================================
    wayne_print("\n🔄 2. SE(3)表示转换测试", "blue", bold=True)
    
    # 测试四元数+平移转换
    wayne_print("   [2.1] 四元数+平移转换", "yellow")
    q_test = np.array([[0.866, 0, 0, 0.5]])  # 绕Z轴30度
    q_test = q_test / np.linalg.norm(q_test)  # 归一化四元数
    t_test = np.array([[1, 2, 3]])
    T_from_qt = SE3_from_quat_trans(q_test, t_test)
    q_recovered, t_recovered = SE3_to_quat_trans(T_from_qt)
    qt_error = (np.linalg.norm(q_test - q_recovered) + 
               np.linalg.norm(t_test - t_recovered))
    print(f"   四元数+平移往返误差: {qt_error:.2e}")
    
    # 测试轴角+平移转换
    wayne_print("   [2.2] 轴角+平移转换", "yellow")
    axis_test = np.array([[0, 0, 1]])
    angle_test = np.array([math.pi/6])
    T_from_aat = SE3_from_axis_angle_trans(axis_test, angle_test, t_test)
    axis_rec, angle_rec, t_rec = SE3_to_axis_angle_trans(T_from_aat)
    aat_error = (np.linalg.norm(axis_test - axis_rec) + 
                np.linalg.norm(angle_test - angle_rec) + 
                np.linalg.norm(t_test - t_rec))
    print(f"   轴角+平移往返误差: {aat_error:.2e}")
    
    # ================================
    # 3. 测试对数/指数映射
    # ================================
    wayne_print("\n📊 3. 对数/指数映射测试", "blue", bold=True)
    
    # 测试 SE3_Log (向量形式)
    wayne_print("   [3.1] SE3_Log - 对数映射 (向量形式)", "yellow")
    xi_recovered = SE3_Log(T_test)
    print(f"   对数映射向量: {xi_recovered}")
    print(f"   向量形状: {xi_recovered.shape}")
    
    # 测试 SE3_log (矩阵形式)
    wayne_print("   [3.2] SE3_log - 对数映射 (矩阵形式)", "yellow")
    xi_hat = SE3_log(T_test)
    print(f"   李代数矩阵形状: {xi_hat.shape}")
    print(f"   李代数矩阵:\n{xi_hat}")
    
    # 测试 SE3_Exp (向量输入)
    wayne_print("   [3.3] SE3_Exp - 指数映射 (向量输入)", "yellow")
    T_from_exp = SE3_Exp(xi_recovered)
    log_exp_error = np.linalg.norm(T_test - T_from_exp)
    print(f"   Log → Exp 往返误差: {log_exp_error:.2e}")
    print(f"   结果是否为有效SE3: {check_SE3(T_from_exp)}")
    
    # 测试 SE3_exp (矩阵输入)
    wayne_print("   [3.4] SE3_exp - 指数映射 (矩阵输入)", "yellow")
    T_from_exp_mat = SE3_exp(xi_hat)
    if T_from_exp_mat.ndim == 2:  # 单个矩阵
        exp_mat_error = np.linalg.norm(T_test - T_from_exp_mat)
        exp_mat_valid = check_SE3(T_from_exp_mat)
    else:  # 批量矩阵
        exp_mat_error = np.linalg.norm(T_test - T_from_exp_mat[0])
        exp_mat_valid = check_SE3(T_from_exp_mat[0])
    print(f"   log → exp 往返误差: {exp_mat_error:.2e}")
    print(f"   结果是否为有效SE3: {exp_mat_valid}")
    
    # 测试完整往返转换
    wayne_print("   [3.5] 完整往返转换验证", "yellow")
    
    # 测试向量路径：xi → Exp → Log → xi
    test_xi = np.array([0.1, 0.2, 0.3, 0.05, 0.1, 0.15])
    T1 = SE3_Exp(test_xi)
    recovered_xi = SE3_Log(T1)
    T2 = SE3_Exp(recovered_xi)
    xi_path_error = np.linalg.norm(T1 - T2)
    print(f"   向量路径往返误差: {xi_path_error:.2e}")
    
    # 测试矩阵路径：xi_hat → exp → log → xi_hat
    test_xi_hat = SE3_skew(test_xi)
    T3 = SE3_exp(test_xi_hat)
    recovered_xi_hat = SE3_log(T3)
    T4 = SE3_exp(recovered_xi_hat)
    matrix_path_error = np.linalg.norm(T3 - T4)
    print(f"   矩阵路径往返误差: {matrix_path_error:.2e}")
    
    # 验证关系一致性：log(T) = skew(Log(T))
    wayne_print("   [3.6] 函数关系验证", "yellow")
    xi_from_Log = SE3_Log(T_test)
    xi_hat_from_log = SE3_log(T_test)
    xi_hat_from_skew = SE3_skew(xi_from_Log)
    relationship_error = np.linalg.norm(xi_hat_from_log - xi_hat_from_skew)
    print(f"   log(T) vs skew(Log(T)) 误差: {relationship_error:.2e}")
    
    # ================================
    # 4. 测试平均变换
    # ================================
    wayne_print("\n🎯 4. SE(3)平均测试", "blue", bold=True)
    
    # 创建多个SE3矩阵
    angles = [0, math.pi/6, math.pi/4, math.pi/3]
    translations = [[1, 0, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2]]
    T_matrices = []
    
    for angle, trans in zip(angles, translations):
        R = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
            [0,                0,               1]
        ])
        T = SE3_from_Rt(R, np.array(trans))
        T_matrices.append(T)
    
    T_matrices = np.array(T_matrices)
    
    wayne_print("   [4.1] 输入数据", "yellow")
    print(f"   变换矩阵数量: {len(T_matrices)}")
    print(f"   旋转角度 (度): {[math.degrees(a) for a in angles]}")
    
    # 测试平均方法
    wayne_print("   [4.2] SE3平均方法", "yellow")
    mean_T_result = SE3_mean(T_matrices)
    print(f"   结果是否为有效SE3: {check_SE3(mean_T_result)}")
    
    # ================================
    # 5. 性能基准测试
    # ================================
    wayne_print("\n⚡ 5. 性能基准测试", "blue", bold=True)
    
    # 创建测试数据
    n_test = 100
    test_angles = np.linspace(0, 2*math.pi, n_test)
    test_translations = np.random.randn(n_test, 3)
    large_T_matrices = np.array([
        SE3_from_Rt(
            np.array([[math.cos(a), -math.sin(a), 0],
                     [math.sin(a),  math.cos(a), 0],
                     [0,            0,           1]]),
            trans
        ) for a, trans in zip(test_angles, test_translations)
    ])
    
    wayne_print(f"   [5.1] 测试数据量: {n_test} 个SE3矩阵", "yellow")
    
    # 性能测试
    test_cases = [
        ("SE3_to_quat_trans", lambda: SE3_to_quat_trans(large_T_matrices)),
        ("SE3_Log", lambda: SE3_Log(large_T_matrices)),
        ("SE3_log", lambda: SE3_log(large_T_matrices)),
        ("SE3_Exp", lambda: SE3_Exp(np.random.randn(n_test, 6) * 0.5)),
        ("SE3_exp", lambda: SE3_exp(SE3_skew(np.random.randn(n_test, 6) * 0.5))),
        ("SE3_mean", lambda: SE3_mean(large_T_matrices)),
    ]
    
    for test_name, test_func in test_cases:
        start_time = time.time()
        for _ in range(5):
            result = test_func()
        avg_time = (time.time() - start_time) / 5
        print(f"   {test_name:18}: {avg_time*1000:6.2f} ms")
    
    wayne_print("\n🎉 SE(3)测试完成!", "green", bold=True)
    print("=" * 60)
