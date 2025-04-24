AprilTag 角点检测
===================

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

.. autoclass:: pywayne.cv.apriltag_detector.ApriltagCornerDetector
   :members:
   :undoc-members:
   :show-inheritance: 