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
- `lift_projective(self, p)`: 将 2D 图像点提升到 3D  projective 射线。
- `space_to_plane(self, P)`: 将 3D 空间点投影到 2D 图像平面。
- `get_parameters_as_dict(self)`: 返回包含相机详细参数的字典。
- `model_type` (property): 获取相机模型类型。
- `camera_name` (property): 获取相机名称。
- `image_width` (property): 获取图像宽度。
- `image_height` (property): 获取图像高度。

**示例**::

   >>> from pywayne.cv.camera_model import CameraModel
   >>> from pywayne.cv.tools import write_cv_yaml
   >>> import os
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
   >>> write_cv_yaml(yaml_file, sample_data)
   True
   >>> 
   >>> # 初始化并加载模型
   >>> cam = CameraModel()
   >>> cam.load_from_yaml(yaml_file)
   >>> print(f"Loaded camera: {cam.camera_name}, Type: {cam.model_type}")
   Loaded camera: my_cam, Type: ModelType.PINHOLE
   >>> 
   >>> # 获取参数
   >>> params = cam.get_parameters_as_dict()
   >>> print(f"Focal Length: fx={params['fx']:.1f}, fy={params['fy']:.1f}")
   Focal Length: fx=600.0, fy=600.0
   >>> 
   >>> # 投影和反投影示例
   >>> image_point = (700.0, 400.0) # 图像上的一个点
   >>> ray_3d = cam.lift_projective(image_point)
   >>> print(f"Lifted ray for {image_point}: ({ray_3d[0]:.3f}, {ray_3d[1]:.3f}, {ray_3d[2]:.3f})")
   Lifted ray for (700.0, 400.0): (0.099, 0.066, 0.993)
   >>> 
   >>> world_point = (0.1, -0.2, 1.5) # 相机坐标系下的一个 3D 点
   >>> projected_point = cam.space_to_plane(world_point)
   >>> print(f"Projected point for {world_point}: ({projected_point[0]:.1f}, {projected_point[1]:.1f})")
   Projected point for (0.1, -0.2, 1.5): (680.0, 280.0)
   >>> 
   >>> # 清理临时文件
   >>> os.remove(yaml_file)

.. autoclass:: pywayne.cv.camera_model.CameraModel
   :members:
   :undoc-members:
   :show-inheritance: 