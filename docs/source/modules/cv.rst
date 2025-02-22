计算机视觉 (CV)
=================

本模块提供了与计算机视觉相关的工具与方法，主要包括两个部分：

1. AprilTag 角点检测
2. 几何外壳计算（Geometric Hull Calculator）

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


模块组织建议
----------------

由于 CV 模块功能较为丰富，如果未来内容进一步扩展，可以考虑为该模块创建独立的子文件夹（例如 docs/source/cv/），从而将每个子模块的文档分别归档，便于维护和查阅。 