# author: wangye(Wayne)
# license: Apache Licence
# file: tools.py
# time: 2024-01-10-20:37:48
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import numpy as np
import qmt
import matplotlib.pyplot as plt


def SE3_to_pose(SE3_mat: np.ndarray) -> np.ndarray:
    """
    将SE3矩阵转换为pose矩阵
    :param SE3_mat: N个SE(3)
    :return: N个pose, tx, ty, tz, qw, qx, qy, qz
    """
    SE3_mat = np.asarray(SE3_mat, dtype=np.float64)
    if SE3_mat.ndim == 2:
        if SE3_mat.shape != (4, 4):
            raise ValueError(f"single SE(3) input must have shape (4, 4), got {SE3_mat.shape}")
        SE3_mat = SE3_mat[None, ...]
    elif SE3_mat.ndim != 3 or SE3_mat.shape[1:] != (4, 4):
        raise ValueError(f"SE(3) input must have shape (4, 4) or (N, 4, 4), got {SE3_mat.shape}")

    quaternions = qmt.quatFromRotMat(SE3_mat[:, :3, :3])
    return np.concatenate([SE3_mat[:, :3, 3], quaternions], axis=1)


def pose_to_SE3(pose_mat: np.ndarray) -> np.ndarray:
    """
    将pose矩阵转换为SE3矩阵
    :param pose_mat: N个pose, shape: (N, 7), tx, ty, tz, qw, qx, qy, qz
    :return: N个SE(3)
    """
    pose_mat = np.asarray(pose_mat, dtype=np.float64)
    if pose_mat.ndim == 1:
        if pose_mat.shape != (7,):
            raise ValueError(f"single pose input must have shape (7,), got {pose_mat.shape}")
        pose_mat = pose_mat[None, ...]
    elif pose_mat.ndim != 2 or pose_mat.shape[1] != 7:
        raise ValueError(f"pose input must have shape (7,) or (N, 7), got {pose_mat.shape}")

    rotations = qmt.quatToRotMat(pose_mat[:, 3:])
    SE3_mat = np.zeros((pose_mat.shape[0], 4, 4), dtype=np.float64)
    SE3_mat[:, :3, :3] = rotations
    SE3_mat[:, :3, 3] = pose_mat[:, :3]
    SE3_mat[:, 3, 3] = 1.0
    return SE3_mat


def visualize_pose(poses, arrow_length_ratio=0.1):
    """
    Initialize the PoseVisualizer with a set of poses and scale factors for arrows.

    :param poses: A list or array of SE(3) poses, each defined as (tx, ty, tz, qw, qx, qy, qz).
    :param arrow_length_ratio: A scalar to determine the length of the arrows relative to the data range.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the scale factor based on the translations
    translations = np.array([pose[:3] for pose in poses])
    max_range = np.array([translations[:, i].max() - translations[:, i].min() for i in range(3)]).max()
    arrow_length = max_range * arrow_length_ratio

    # Plotting each pose
    for pose in poses:
        # Extract the quaternion and translation from the pose
        quaternion = pose[3:]
        translation = pose[:3]

        # Convert quaternion to rotation matrix
        rotation_matrix = qmt.quatToRotMat(quaternion)

        # Draw the position
        ax.scatter(translation[0], translation[1], translation[2], color='k', s=50)

        # Draw the orientation arrows using quiver
        for i, color in enumerate(['r', 'g', 'b']):
            ax.quiver(translation[0], translation[1], translation[2],
                      rotation_matrix[0, i], rotation_matrix[1, i], rotation_matrix[2, i],
                      length=arrow_length, color=color, normalize=True)

    # Setting the aspect ratio to 'equal' so the scale is the same on all axes
    ax.set_aspect('auto')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of SE(3) Poses')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    a = np.random.randn(10, 4, 4)
    for i in range(a.shape[0]):
        b = a[i][:3, :3].squeeze()
        U, S, V = np.linalg.svd(b)
        SO3 = U @ V
        if np.linalg.det(SO3) < 0:
            V[2, :] *= -1
        a[i][:3, :3] = U @ V
        a[i][3] = [0, 0, 0, 1]
    pose = SE3_to_pose(a)
    SE3 = pose_to_SE3(pose)

    visualize_pose(pose)
