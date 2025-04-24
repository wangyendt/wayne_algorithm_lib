几何外壳计算
===============

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

.. autoclass:: pywayne.cv.geometric_hull_calculator.GeometricHullCalculator
   :members:
   :undoc-members:
   :show-inheritance: 