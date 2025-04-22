计算机视觉 (CV)
=================

本模块提供了与计算机视觉相关的工具与方法，主要包括以下几个部分：

1. AprilTag 角点检测
2. 几何外壳计算（Geometric Hull Calculator）
3. OpenCV YAML 文件读写

AprilTag 角点检测
-------------------

该部分主要由 ApriltagCornerDetector 类提供，该类用于检测图像中的 AprilTag 标签。

**主要功能**:

- 自动检测图像中的 AprilTag 标签。
- 支持输入图像路径或 numpy 数组作为输入。
- 提供直接检测和绘制检测结果的方法。

**主要方法**:

- __init__(self): 初始化检测器，并检测或安装所需的第三方库。

- detect(self, image: Union[str, Path, np.ndarray], show_result: bool = False):
  接收图像（路径或数组），返回检测结果。如果 show_result 为 True，则显示检测结果。

- detect_and_draw(self, image: Union[str, np.ndarray]) -> np.ndarray:
  在原图上绘制检测结果，并返回绘制后的图像。

**示例**::

   >>> from pywayne.cv.apriltag_detector import ApriltagCornerDetector
   >>> detector = ApriltagCornerDetector()
   >>> detections = detector.detect('test.png', show_result=True)

几何外壳计算
---------------

该部分主要由 GeometricHullCalculator 类提供，用于计算给定点集的凸包和凹包，同时也支持随机点生成和可视化操作。

**主要功能**:

- 计算并返回点集的最小边界矩形（MBR）、凸包和凹包。
- 支持使用 OpenCV 和 matplotlib 分别进行结果的可视化展示。
- 提供生成随机点集的辅助方法，方便示例测试和验证算法。

**主要方法**:

- generate_random_points(num_points: int = 100, scale: int = 50, offset: int = 150):
  生成随机点集，用于测试或示例展示。

- get_convex_hull(self):
  返回点集的凸包结果。

- get_concave_hull(self):
  返回点集的凹包结果。

- visualize_opencv(self):
  使用 OpenCV 显示计算结果。

- visualize_matplotlib(self):
  使用 matplotlib 绘制与展示计算结果。

**示例**::

   >>> from pywayne.cv.geometric_hull_calculator import GeometricHullCalculator
   >>> hull_calculator = GeometricHullCalculator(points=GeometricHullCalculator.generate_random_points())
   >>> convex_hull = hull_calculator.get_convex_hull()
   >>> concave_hull = hull_calculator.get_concave_hull()
   >>> hull_calculator.visualize_matplotlib()

OpenCV YAML 文件读写
----------------------

这部分功能位于 `pywayne.cv.tools` 模块中，提供了读取和写入 OpenCV `cv2.FileStorage` 支持的 YAML 格式文件的函数。

**主要功能**:

- `read_cv_yaml(file_path: str)`: 读取 OpenCV YAML 文件并返回一个字典。支持基本数据类型和 NumPy 数组。
- `write_cv_yaml(file_path: str, data: dict)`: 将包含基本数据类型和 NumPy 数组的字典写入 OpenCV YAML 文件。

**示例**::

   >>> import numpy as np
   >>> from pywayne.cv.tools import read_cv_yaml, write_cv_yaml
   >>> 
   >>> # 准备要写入的数据
   >>> data_to_write = {
   ...     "matrix": np.eye(3),
   ...     "integer_value": 10,
   ...     "float_value": 3.14,
   ...     "string_value": "hello_cv",
   ...     "list_value": [1, 2, 3]
   ... }
   >>> 
   >>> # 写入 YAML 文件
   >>> output_file = "cv_config.yaml"
   >>> success = write_cv_yaml(output_file, data_to_write)
   >>> if success:
   ...     print(f"成功写入到 {output_file}")
   >>> 
   >>> # 从 YAML 文件读取
   >>> read_data = read_cv_yaml(output_file)
   >>> if read_data:
   ...     print(f"从 {output_file} 读取的数据:")
   ...     for k, v in read_data.items():
   ...         print(f"  {k}: {v}")


模块组织建议
----------------

由于 CV 模块功能较为丰富，如果未来内容进一步扩展，可以考虑为该模块创建独立的子文件夹（例如 docs/source/cv/），从而将每个子模块的文档分别归档，便于维护和查阅。 