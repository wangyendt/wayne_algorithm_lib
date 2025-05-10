相机模型封装
=================

该模块提供了 `CameraModel` 类，用于封装 `camera_models` C++ 库（通过 pybind11 暴露），简化相机模型的加载和使用。

**主要功能**:

- 在初始化时检查 `camera_models` 库，如果不存在则尝试使用 `gettool` 获取。
- 从 YAML 文件加载相机参数。
- 提供访问相机基本属性（如模型类型、名称、图像尺寸）的接口。
- 封装核心的相机几何操作，如从 2D 图像点到 3D 射线的反投影 (`lift_projective`) 和从 3D 空间点到 2D 图像点的投影 (`space_to_plane`)。
- 提供将相机参数导出为 Python 字典的方法。

**主要方法和属性**:

- `__init__(self)`: 初始化 `CameraModel`，检查并导入所需的 `camera_models` 库。
- `load_from_yaml(self, yaml_path)`: 从指定的 YAML 文件加载相机模型。
- `lift_projective(self, p)`: 将 2D 图像点 (Tuple, List, 或 np.ndarray) 提升到 3D projective 射线 (返回 np.ndarray)。
- `space_to_plane(self, P)`: 将 3D 空间点 (Tuple, List, 或 np.ndarray) 投影到 2D 图像平面 (返回 np.ndarray)。
- `get_parameters_as_dict(self)`: 返回包含相机详细参数的字典。
- `model_type` (property): 获取相机模型类型。
- `camera_name` (property): 获取相机名称。
- `image_width` (property): 获取图像宽度。
- `image_height` (property): 获取图像高度。

**示例**::

   >>> from pywayne.cv.camera_model import CameraModel
   >>> from pywayne.cv.tools import write_cv_yaml
   >>> import os
   >>> import numpy as np # 确保导入 numpy
   >>> 
   >>> # 准备一个示例 YAML 文件 (实际应用中应从标定结果获取)
   >>> sample_data = {
   ...     "model_type": "PINHOLE",
   ...     "camera_name": "my_cam",
   ...     "image_width": 1280,
   ...     "image_height": 720,
   ...     "distortion_parameters": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0},
   ...     "projection_parameters": {"fx": 600.0, "fy": 600.0, "cx": 640.0, "cy": 360.0}
   ... }
   >>> yaml_file = "temp_cam_config.yaml"
   >>> _ = write_cv_yaml(yaml_file, sample_data) # write_cv_yaml 返回布尔值
   >>> 
   >>> # 初始化并加载模型
   >>> cam = CameraModel()
   >>> cam.load_from_yaml(yaml_file)
   Loaded camera 'my_cam' (Type: ModelType.PINHOLE) from temp_cam_config.yaml # 示例输出可能略有不同
   >>> print(f"Loaded camera: {cam.camera_name}, Type: {cam.model_type}")
   Loaded camera: my_cam, Type: ModelType.PINHOLE
   >>> 
   >>> # 获取参数
   >>> params = cam.get_parameters_as_dict()
   >>> print(f"Focal Length: fx={params['fx']:.1f}, fy={params['fy']:.1f}")
   Focal Length: fx=600.0, fy=600.0
   >>> 
   >>> # 投影和反投影示例
   >>> image_point_list = [700.0, 400.0] # 使用 list 作为输入
   >>> ray_3d_numpy = cam.lift_projective(image_point_list) # 返回 np.ndarray
   >>> print(f"Lifted ray for {image_point_list}: {ray_3d_numpy} (type: {type(ray_3d_numpy)})")
   Lifted ray for [700.0, 400.0]: [0.09901951 0.06601301 0.99278962] (type: <class 'numpy.ndarray'>)
   >>> 
   >>> world_point_numpy = np.array([0.1, -0.2, 1.5]) # 使用 np.ndarray 作为输入
   >>> projected_point_numpy = cam.space_to_plane(world_point_numpy) # 返回 np.ndarray
   >>> print(f"Projected point for {world_point_numpy}: {projected_point_numpy} (type: {type(projected_point_numpy)})")
   Projected point for [ 0.1 -0.2  1.5]: [680. 280.] (type: <class 'numpy.ndarray'>)
   >>> 
   >>> # 清理临时文件
   >>> os.remove(yaml_file)

.. autoclass:: pywayne.cv.camera_model.CameraModel
   :members:
   :undoc-members:
   :show-inheritance: 