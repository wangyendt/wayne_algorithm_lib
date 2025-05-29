import numpy as np
import qmt
import math
import time
from pywayne.tools import wayne_print
from scipy.spatial.transform import Rotation

def check_SO3(R: np.ndarray) -> bool:
    """
    Check if a matrix is a valid SO(3) rotation matrix.
    Args:
        R: 3x3 rotation matrix
    Returns:
        True if R is a valid SO(3) rotation matrix, False otherwise
    """
    if R.shape != (3, 3):
        return False
    if not np.allclose(R.T @ R, np.eye(3)):
        return False
    return True

def SO3_skew(vec: np.ndarray) -> np.ndarray:
    """
    Convert a vector to a skew-symmetric matrix.
    Args:
        vec: Nx3 vector or 3-element vector
    Returns:
        skew_mat: Nx3x3 skew-symmetric matrix
    """
    # 确保输入是二维的
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    
    epsilon = np.array([[[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                    [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                    [[0, -1, 0], [1, 0, 0], [0, 0, 0]]])
    skew = np.einsum('ijk, nk -> nij', epsilon, vec)
    
    # 如果原始输入是一维的，返回单个矩阵而不是数组
    if skew.shape[0] == 1:
        return skew[0]
    return skew

def SO3_unskew(skew: np.ndarray) -> np.ndarray:
    """
    Convert a skew-symmetric matrix to a vector.
    Args:
        skew: Nx3x3 or 3x3 skew-symmetric matrix
    Returns:
        vec: Nx3 or 3 vector
    """
    # 确保输入是三维的
    if skew.ndim == 2:
        skew = skew.reshape(1, 3, 3)
        single_vector = True
    else:
        single_vector = False
    
    # 从反对称矩阵中提取向量 [x, y, z]
    # skew = [[ 0, -z,  y],
    #         [ z,  0, -x],
    #         [-y,  x,  0]]
    # 所以 vec = [skew[2,1], skew[0,2], skew[1,0]]
    vec = np.array([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]]).T
    
    if single_vector:
        return vec[0]
    return vec

def SO3_mul(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """
    Multiply two SO(3) rotation matrices.
    Args:
        R1: 3x3 rotation matrix
        R2: 3x3 rotation matrix
    Returns:
        R1 @ R2
    """
    return R1 @ R2

def SO3_diff(R1: np.ndarray, R2: np.ndarray, from_1_to_2: bool = True) -> np.ndarray:
    """
    Compute the difference between two SO(3) rotation matrices.
    Args:
        R1: 3x3 rotation matrix
        R2: 3x3 rotation matrix
        from_1_to_2: if True, compute R2 @ R1.T, otherwise compute R1.T @ R2
    Returns:
        R1.T @ R2 if from_1_to_2 is True, otherwise R1 @ R2.T
    """
    if from_1_to_2:
        return R1.T @ R2
    else:
        return R2.T @ R1
    
def SO3_from_quat(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a SO(3) rotation matrix.
    Args:
        q: Nx4 quaternion, wxyz, Hamilton convention
    Returns:
        R: Nx3x3 rotation matrix
    """
    return qmt.quatToRotMat(q)

def SO3_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert a SO(3) rotation matrix to a quaternion.
    Args:
        R: Nx3x3 rotation matrix
    Returns:
        q: Nx4 quaternion, wxyz, Hamilton convention
    """
    return qmt.quatFromRotMat(R)

def SO3_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert an axis-angle to a SO(3) rotation matrix.
    Args:
        axis: Nx3 axis
        angle: Nx1 angle, rad
    Returns:
        R: Nx3x3 rotation matrix
    """
    return qmt.quatToRotMat(qmt.quatFromAngleAxis(angle, axis))

def SO3_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    Convert a SO(3) rotation matrix to an axis-angle.
    Args:
        R: Nx3x3 rotation matrix
    Returns:
        axis: Nx3 axis
        angle: Nx1 angle
    """
    q = qmt.quatFromRotMat(R)
    angle = qmt.quatAngle(q)
    axis = qmt.quatAxis(q)
    return axis, angle

def SO3_from_euler(euler_angles: np.ndarray, axes: str = 'zyx', intrinsic: bool = True) -> np.ndarray:
    """
    Convert Euler angles to a SO(3) rotation matrix.
    Args:
        euler_angles: Nx3 Euler angles, rad
        axes: Euler angles sequence
        intrinsic: if True, use intrinsic rotation, otherwise use extrinsic rotation
    Returns:
        R: Nx3x3 rotation matrix
    """
    return qmt.quatToRotMat(qmt.quatFromEulerAngles(euler_angles, axes, intrinsic))

def SO3_to_euler(R: np.ndarray, axes: str = 'zyx', intrinsic: bool = True) -> np.ndarray:
    """
    Convert a SO(3) rotation matrix to Euler angles.
    Args:
        R: Nx3x3 rotation matrix
        axes: Euler angles sequence
        intrinsic: if True, use intrinsic rotation, otherwise use extrinsic rotation
    Returns:    
        euler_angles: Nx3 Euler angles, rad
    """
    return qmt.eulerAngles(qmt.quatFromRotMat(R), axes, intrinsic)

def SO3_Log(R: np.ndarray) -> np.ndarray:
    """
    Convert SO(3) rotation matrix to Lie algebra vector (rotation vector).
    Args:
        R: Nx3x3 or 3x3 rotation matrix
    Returns:
        rotvec: Nx3 or 3 rotation vector (Lie algebra vector)
    """
    return qmt.quatToRotVec(qmt.quatFromRotMat(R))

def SO3_log(R: np.ndarray) -> np.ndarray:
    """
    Convert SO(3) rotation matrix to Lie algebra matrix (skew-symmetric matrix).
    Args:
        R: Nx3x3 or 3x3 rotation matrix
    Returns:
        log_map: Nx3x3 or 3x3 skew-symmetric matrix (Lie algebra matrix)
    """
    rotvec = SO3_Log(R)
    return SO3_skew(rotvec)

def SO3_Exp(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert Lie algebra vector (rotation vector) to SO(3) rotation matrix.
    Args:
        rotvec: Nx3 or 3 rotation vector (Lie algebra vector)
    Returns:
        R: Nx3x3 or 3x3 rotation matrix
    """
    if rotvec.ndim == 1:
        rotvec = rotvec.reshape(1, -1)
        single_matrix = True
    else:
        single_matrix = False
    
    # 计算角度
    angles = np.linalg.norm(rotvec, axis=1)
    
    # 处理零向量情况
    zero_mask = angles < 1e-8
    
    # 初始化结果
    N = rotvec.shape[0]
    R = np.zeros((N, 3, 3))
    
    # 零向量对应单位矩阵
    R[zero_mask] = np.eye(3)
    
    # 非零向量情况
    if np.any(~zero_mask):
        non_zero_rotvec = rotvec[~zero_mask]
        non_zero_angles = angles[~zero_mask]
        
        # 归一化轴
        axes = non_zero_rotvec / non_zero_angles[:, np.newaxis]
        
        # 使用qmt转换
        R[~zero_mask] = qmt.quatToRotMat(qmt.quatFromAngleAxis(non_zero_angles, axes))
    
    if single_matrix:
        return R[0]
    return R

def SO3_exp(omega_hat: np.ndarray) -> np.ndarray:
    """
    Convert Lie algebra matrix (skew-symmetric matrix) to SO(3) rotation matrix.
    Args:
        omega_hat: Nx3x3 or 3x3 skew-symmetric matrix (Lie algebra matrix)
    Returns:
        R: Nx3x3 or 3x3 rotation matrix
    """
    # 从反对称矩阵提取旋转向量
    rotvec = SO3_unskew(omega_hat)
    
    # 使用SO3_Exp转换
    return SO3_Exp(rotvec)

def SO3_inv(R: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of SO(3) rotation matrix.
    For SO(3) matrices, the inverse is simply the transpose.
    Args:
        R: Nx3x3 or 3x3 rotation matrices
    Returns:
        R_inv: Nx3x3 or 3x3 inverse rotation matrices
    """
    if R.ndim == 2:
        # 单个3x3矩阵
        return R.T
    else:
        # 批量Nx3x3矩阵
        return np.transpose(R, (0, 2, 1))

def SO3_mean(R: np.ndarray) -> np.ndarray:
    """
    Compute the mean of multiple SO(3) rotation matrices using scipy.
    Args:
        R: Nx3x3 rotation matrices
    Returns:
        mean_R: 3x3 mean rotation matrix
    """
    if R.ndim == 2:
        # 单个矩阵，直接返回
        return R
    
    # 基于scipy的方法
    # 将旋转矩阵转换为Rotation对象
    rotations = Rotation.from_matrix(R)
    
    # 计算平均旋转
    mean_rotation = rotations.mean()
    
    # 转换回旋转矩阵
    return mean_rotation.as_matrix()


if __name__ == '__main__':
    wayne_print("🔧 SO3 工具函数全面测试", "cyan", bold=True)
    print("=" * 60)
    
    # ================================
    # 1. 测试基础工具函数
    # ================================
    wayne_print("\n📐 1. 基础工具函数测试", "blue", bold=True)
    
    # 测试 SO3_skew
    wayne_print("   [1.1] SO3_skew - 反对称矩阵", "yellow")
    vec1 = np.array([1, 2, 3])
    skew1 = SO3_skew(vec1)
    print(f"   输入向量: {vec1}")
    print(f"   反对称矩阵:\n{skew1}")
    
    vec2 = np.array([[1, 2, 3], [4, 5, 6]])
    skew2 = SO3_skew(vec2)
    print(f"   批量输入形状: {vec2.shape}")
    print(f"   批量输出形状: {skew2.shape}")
    
    # 测试 SO3_unskew
    wayne_print("   [1.1.1] SO3_unskew - 反对称矩阵转向量", "yellow")
    vec1_recovered = SO3_unskew(skew1)
    vec2_recovered = SO3_unskew(skew2)
    unskew_error1 = np.linalg.norm(vec1 - vec1_recovered)
    unskew_error2 = np.linalg.norm(vec2 - vec2_recovered)
    print(f"   单个向量往返误差: {unskew_error1:.2e}")
    print(f"   批量向量往返误差: {unskew_error2:.2e}")
    print(f"   恢复向量1: {vec1_recovered}")
    print(f"   恢复向量2形状: {vec2_recovered.shape}")
    
    # 测试 check_SO3
    wayne_print("   [1.2] check_SO3 - 验证旋转矩阵", "yellow")
    print(f"   单位矩阵是否为SO3: {check_SO3(np.eye(3))}")
    print(f"   反对称矩阵是否为SO3: {check_SO3(skew1)}")
    
    # 测试 SO3_mul
    wayne_print("   [1.3] SO3_mul - 旋转矩阵相乘", "yellow")
    R1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # 绕X轴90度
    R2 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # 绕Y轴90度
    R_mul = SO3_mul(R1, R2)
    print(f"   R1 @ R2 是否为有效SO3: {check_SO3(R_mul)}")
    
    # 测试 SO3_diff
    wayne_print("   [1.4] SO3_diff - 旋转差异", "yellow")
    R_diff = SO3_diff(R1, R2)
    print(f"   旋转差异矩阵是否为有效SO3: {check_SO3(R_diff)}")
    
    # 测试 SO3_inv
    wayne_print("   [1.5] SO3_inv - 旋转矩阵求逆", "yellow")
    R_inv = SO3_inv(R1)
    R_identity_check = SO3_mul(R1, R_inv)
    identity_error = np.linalg.norm(R_identity_check - np.eye(3))
    print(f"   R @ R^(-1) 与单位矩阵误差: {identity_error:.2e}")
    print(f"   逆矩阵是否为有效SO3: {check_SO3(R_inv)}")
    
    # 测试批量求逆
    R_batch = np.array([R1, R2])
    R_inv_batch = SO3_inv(R_batch)
    print(f"   批量求逆形状: {R_batch.shape} → {R_inv_batch.shape}")
    
    # ================================
    # 2. 测试转换函数
    # ================================
    wayne_print("\n🔄 2. 旋转表示转换测试", "blue", bold=True)
    
    # 创建测试用的旋转矩阵（绕Z轴45度）
    angle = math.pi / 4
    R_test = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle),  math.cos(angle), 0],
        [0,                0,               1]
    ])
    
    # 测试四元数转换
    wayne_print("   [2.1] 四元数转换", "yellow")
    quat = SO3_to_quat(R_test.reshape(1, 3, 3))
    R_from_quat = SO3_from_quat(quat)
    quat_error = np.linalg.norm(R_test - R_from_quat[0])
    print(f"   原始矩阵 → 四元数 → 矩阵误差: {quat_error:.2e}")
    print(f"   四元数: {quat[0]}")
    
    # 测试轴角转换
    wayne_print("   [2.2] 轴角转换", "yellow")
    axis, angle_out = SO3_to_axis_angle(R_test.reshape(1, 3, 3))
    R_from_axis_angle = SO3_from_axis_angle(axis, angle_out)
    axis_angle_error = np.linalg.norm(R_test - R_from_axis_angle[0])
    print(f"   原始矩阵 → 轴角 → 矩阵误差: {axis_angle_error:.2e}")
    print(f"   轴: {axis[0]}, 角度: {math.degrees(angle_out[0]):.2f}°")
    
    # 测试欧拉角转换
    wayne_print("   [2.3] 欧拉角转换", "yellow")
    euler = SO3_to_euler(R_test.reshape(1, 3, 3))
    R_from_euler = SO3_from_euler(euler)
    euler_error = np.linalg.norm(R_test - R_from_euler[0])
    print(f"   原始矩阵 → 欧拉角 → 矩阵误差: {euler_error:.2e}")
    print(f"   欧拉角 (度): {np.degrees(euler[0])}")
    
    # ================================
    # 3. 测试对数映射
    # ================================
    wayne_print("\n📊 3. 对数/指数映射测试", "blue", bold=True)
    
    # 测试 SO3_Log (返回向量)
    wayne_print("   [3.1] SO3_Log - 对数映射 (向量形式)", "yellow")
    log_vec = SO3_Log(R_test.reshape(1, 3, 3))
    print(f"   对数映射向量: {log_vec[0]}")
    print(f"   向量模长: {np.linalg.norm(log_vec[0]):.4f}")
    
    # 测试 SO3_log (返回反对称矩阵)
    wayne_print("   [3.2] SO3_log - 对数映射 (矩阵形式)", "yellow")
    log_mat = SO3_log(R_test.reshape(1, 3, 3))
    print(f"   对数映射矩阵形状: {log_mat.shape}")
    print(f"   是否为反对称矩阵: {np.allclose(log_mat, -log_mat.T)}")
    
    # 测试 SO3_Exp (向量输入)
    wayne_print("   [3.3] SO3_Exp - 指数映射 (向量输入)", "yellow")
    R_from_exp = SO3_Exp(log_vec)
    exp_error = np.linalg.norm(R_test - R_from_exp[0])
    print(f"   Log → Exp 往返误差: {exp_error:.2e}")
    print(f"   结果是否为有效SO3: {check_SO3(R_from_exp[0])}")
    
    # 测试 SO3_exp (矩阵输入)
    wayne_print("   [3.4] SO3_exp - 指数映射 (矩阵输入)", "yellow")
    R_from_exp_mat = SO3_exp(log_mat)
    if R_from_exp_mat.ndim == 2:  # 单个矩阵
        exp_mat_error = np.linalg.norm(R_test - R_from_exp_mat)
        exp_mat_valid = check_SO3(R_from_exp_mat)
    else:  # 批量矩阵
        exp_mat_error = np.linalg.norm(R_test - R_from_exp_mat[0])
        exp_mat_valid = check_SO3(R_from_exp_mat[0])
    print(f"   log → exp 往返误差: {exp_mat_error:.2e}")
    print(f"   结果是否为有效SO3: {exp_mat_valid}")
    
    # 测试完整往返转换
    wayne_print("   [3.5] 完整往返转换验证", "yellow")
    
    # 测试向量路径：R → Log → Exp → R
    test_rotvec = np.array([0.1, 0.2, 0.3])
    R1 = SO3_Exp(test_rotvec)
    recovered_rotvec = SO3_Log(R1)
    R2 = SO3_Exp(recovered_rotvec)
    rotvec_path_error = np.linalg.norm(R1 - R2)
    print(f"   向量路径往返误差: {rotvec_path_error:.2e}")
    
    # 测试矩阵路径：R → log → exp → R
    test_skew = SO3_skew(test_rotvec)
    R3 = SO3_exp(test_skew)
    recovered_skew = SO3_log(R3)
    R4 = SO3_exp(recovered_skew)
    matrix_path_error = np.linalg.norm(R3 - R4)
    print(f"   矩阵路径往返误差: {matrix_path_error:.2e}")
    
    # ================================
    # 4. 测试平均旋转
    # ================================
    wayne_print("\n🎯 4. 旋转平均测试", "blue", bold=True)
    
    # 创建多个旋转矩阵
    angles = [0, math.pi/6, math.pi/4, math.pi/3, math.pi/2]
    R_matrices = []
    for a in angles:
        R = np.array([
            [math.cos(a), -math.sin(a), 0],
            [math.sin(a),  math.cos(a), 0],
            [0,            0,           1]
        ])
        R_matrices.append(R)
    R_matrices = np.array(R_matrices)
    
    wayne_print("   [4.1] 输入数据", "yellow")
    print(f"   旋转矩阵数量: {len(R_matrices)}")
    print(f"   旋转角度 (度): {[math.degrees(a) for a in angles]}")
    print(f"   期望平均角度: {math.degrees(np.mean(angles)):.1f}°")
    
    # 测试平均方法
    wayne_print("   [4.2] SO3平均方法", "yellow")
    mean_R_result = SO3_mean(R_matrices)
    print(f"   结果是否为有效SO3: {check_SO3(mean_R_result)}")
    
    # 从结果反推角度
    cos_theta = mean_R_result[0, 0]
    sin_theta = mean_R_result[1, 0]
    calculated_angle = math.atan2(sin_theta, cos_theta)
    print(f"   计算得到的角度: {math.degrees(calculated_angle):.1f}°")
    
    # ================================
    # 5. 性能基准测试
    # ================================
    wayne_print("\n⚡ 5. 性能基准测试", "blue", bold=True)
    
    # 创建大量测试数据
    n_test = 1000
    test_angles = np.linspace(0, 2*math.pi, n_test)
    large_R_matrices = np.array([
        [[math.cos(a), -math.sin(a), 0],
         [math.sin(a),  math.cos(a), 0],
         [0,            0,           1]] for a in test_angles
    ])
    
    wayne_print(f"   [5.1] 测试数据量: {n_test} 个旋转矩阵", "yellow")
    
    # 测试各种转换的性能
    test_cases = [
        ("SO3_to_quat", lambda: SO3_to_quat(large_R_matrices)),
        ("SO3_to_euler", lambda: SO3_to_euler(large_R_matrices)),
        ("SO3_Log", lambda: SO3_Log(large_R_matrices)),
        ("SO3_log", lambda: SO3_log(large_R_matrices)),
        ("SO3_Exp", lambda: SO3_Exp(np.random.randn(n_test, 3) * 0.5)),
        ("SO3_exp", lambda: SO3_exp(SO3_skew(np.random.randn(n_test, 3) * 0.5))),
        ("SO3_inv", lambda: SO3_inv(large_R_matrices)),
        ("SO3_mean", lambda: SO3_mean(large_R_matrices))
    ]
    
    for test_name, test_func in test_cases:
        start_time = time.time()
        for _ in range(5):  # 运行5次取平均
            result = test_func()
        avg_time = (time.time() - start_time) / 5
        print(f"   {test_name:15}: {avg_time*1000:6.2f} ms")
    
    # ================================
    # 6. 综合验证测试
    # ================================
    wayne_print("\n✅ 6. 综合验证测试", "blue", bold=True)
    
    # 创建随机旋转进行往返转换测试
    wayne_print("   [6.1] 往返转换精度验证", "yellow")
    
    # 随机轴和角度
    np.random.seed(42)
    random_axis = np.random.randn(5, 3)
    random_axis = random_axis / np.linalg.norm(random_axis, axis=1, keepdims=True)
    random_angles = np.random.uniform(0, math.pi, 5)
    
    # 轴角 → 旋转矩阵 → 轴角
    R_random = SO3_from_axis_angle(random_axis, random_angles)
    recovered_axis, recovered_angles = SO3_to_axis_angle(R_random)
    
    axis_error = np.mean([np.linalg.norm(a1 - a2) for a1, a2 in zip(random_axis, recovered_axis)])
    angle_error = np.mean(np.abs(random_angles - recovered_angles.flatten()))
    
    print(f"   轴向量平均误差: {axis_error:.2e}")
    print(f"   角度平均误差: {math.degrees(angle_error):.2e}°")
    
    # 验证所有生成的矩阵都是有效的SO3
    validity_check = [check_SO3(R) for R in R_random]
    print(f"   生成矩阵SO3有效性: {sum(validity_check)}/{len(validity_check)}")
    
    wayne_print("\n🎉 所有测试完成!", "green", bold=True)
    print("=" * 60)
