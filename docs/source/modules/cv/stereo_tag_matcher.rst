双目 AprilTag 匹配
=====================

.. automodule:: pywayne.cv.stereo_tag_matcher

.. autoclass:: pywayne.cv.stereo_tag_matcher.StereoTagMatcher
   :members:
   :undoc-members:
   :show-inheritance:

   **主要功能**:

   - 读取一对图像（左右视图），可以是文件路径或 NumPy 数组。
   - 在两张图像中分别检测 AprilTag。
   - 识别并匹配两张图像中共同出现的 AprilTag。
   - 返回匹配到的 Tag 的信息，包括它们在各自图像中的中心点和角点坐标（原始坐标，非缩放）。
   - 生成一个将左右视图拼接在一起的图像，并在图像上标注检测到的 Tag：
     - 所有检测到的 Tag 用绿色框标出。
     - 左右视图中都出现的公共 Tag 用黄色框标出。
     - 连接公共 Tag 在左右视图中的中心点（红色线）。
   - 支持将结果图像显示出来。

   **应用场景**:

   - 双目相机标定。
   - 机器人视觉导航中，通过匹配已知 Tag 确定相机位姿。
   - 增强现实应用中，用于识别和跟踪目标。

   **示例**::

      >>> from pywayne.cv.stereo_tag_matcher import StereoTagMatcher
      >>> from pathlib import Path
      >>> 
      >>> # 初始化匹配器，可以使用默认参数或自定义颜色、线宽等
      >>> matcher = StereoTagMatcher(target_height=600)
      >>> 
      >>> # 处理一对图像
      >>> image1_path = Path('path/to/left_image.png')
      >>> image2_path = Path('path/to/right_image.png')
      >>> 
      >>> matched_info, stitched_image = matcher.process_pair(image1_path, image2_path, show=True)
      >>> 
      >>> if matched_info:
      ...     print("匹配到的 Tag 信息:")
      ...     for tag_id, info in matched_info.items():
      ...         print(f"  Tag ID: {tag_id}")
      ...         print(f"    Left Center: {info['cam1_center']}")
      ...         print(f"    Right Center: {info['cam2_center']}")
      >>> 
      >>> # 如果需要保存结果图像
      >>> # import cv2
      >>> # if stitched_image is not None:
      >>> #     cv2.imwrite('stitched_result.png', stitched_image) 