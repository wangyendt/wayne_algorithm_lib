# author: wangye(Wayne)
# license: Apache Licence
# file: apriltag_detector.py
# time: 2024-10-10-15:33:42
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import os
import sys
import subprocess
import importlib
import cv2
import numpy as np
from typing import Union
from pathlib import Path


class ApriltagCornerDetector:
    def __init__(self):
        apriltag_detection = self._check_and_import_lib()
        self.apriltag_36h11_detector = apriltag_detection.TagDetector(apriltag_detection.tag_codes_36h11(), 2)

    def _check_and_import_lib(self):
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
        sys.path.append(str(lib_path))
        try:
            return importlib.import_module("apriltag_detection")
        except ImportError:
            os.makedirs(lib_path, exist_ok=True)
            subprocess.run(['gettool', 'apriltag_detection', '-b', '-t', str(lib_path)], check=True)
            importlib.invalidate_caches()
            return importlib.import_module("apriltag_detection")

    def detect(self, image: Union[str, Path, np.ndarray], show_result: bool = False):
        """
        检测图像中的AprilTag
        
        Args:
            image: 输入图像，可以是图片路径（str/Path）或者numpy数组（np.ndarray）
            show_result: 是否显示结果
            
        Returns:
            detections: 检测结果
        """
        # 如果输入是路径或Path对象，读取图片
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图片: {image}")

        if len(image.shape) == 2:
            image = image[..., None]
        if image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        detections = self.apriltag_36h11_detector.extract_tags(gray_image)
        if show_result:
            color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            self._show(color_image, detections)
        return detections

    def detect_and_draw(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        检测图像中的AprilTag并在原图上绘制结果
        
        Args:
            image: 输入图像，可以是图片路径（str）或者numpy数组（np.ndarray）
            
        Returns:
            np.ndarray: 绘制了检测结果的图像（与输入图像大小相同）
        """
        # 如果输入是路径或Path对象，读取图片
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图片: {image}")

        # 保存原始图像的副本
        result_image = image.copy()
        
        # 检测AprilTag
        detections = self.detect(image)
        
        # 计算绘制元素的尺寸（根据图像尺寸）
        image_size = max(image.shape[0], image.shape[1])
        circle_radius = max(3, int(image_size * 0.005))  # 圆圈半径为图像最大边长的0.5%，最小为3像素
        circle_thickness = max(2, int(image_size * 0.002))  # 线条粗细为图像最大边长的0.2%，最小为2像素
        font_scale = max(0.5, image_size * 0.001)  # 字体大小为图像最大边长的0.1%，最小为0.5
        font_thickness = max(1, int(image_size * 0.002))  # 字体粗细为图像最大边长的0.2%，最小为1像素
        
        # 在图像上绘制检测结果
        for detection in detections:
            # 绘制角点（绿色圆圈）
            corners = np.array(detection.corners, dtype=np.int32)
            for corner in corners:
                cv2.circle(result_image, tuple(corner), circle_radius, (0, 255, 0), circle_thickness)

            # 绘制 ID（绿色文字）
            center = tuple(map(int, detection.center))
            text = str(detection.id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # 计算文字应该放置的位置，使其居中
            text_x = int(center[0] - text_size[0] / 2)
            text_y = int(center[1] + text_size[1] / 2)

            cv2.putText(result_image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
        
        return result_image

    def _show(self, image: np.ndarray, detections):
        # 计算绘制元素的尺寸（根据图像尺寸）
        image_size = max(image.shape[0], image.shape[1])
        circle_radius = max(3, int(image_size * 0.005))  # 圆圈半径为图像最大边长的0.5%，最小为3像素
        circle_thickness = max(2, int(image_size * 0.002))  # 线条粗细为图像最大边长的0.2%，最小为2像素
        polyline_thickness = max(2, int(image_size * 0.004))  # 多边形线条粗细为图像最大边长的0.4%，最小为2像素
        font_scale = max(0.5, image_size * 0.001)  # 字体大小为图像最大边长的0.2%，最小为0.5
        font_thickness = max(1, int(image_size * 0.004))  # 字体粗细为图像最大边长的0.2%，最小为1像素

        # 在图像上绘制检测结果
        for detection in detections:
            # 绘制边框（绿色）
            corners = np.array(detection.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [corners], True, (0, 255, 0), polyline_thickness)  # 绿色 (B, G, R)

            # 绘制角点（红色圆圈）
            for corner in corners:
                cv2.circle(image, tuple(corner[0]), circle_radius, (0, 0, 255), circle_thickness)  # 红色 (B, G, R)

            # 绘制 ID（红色文字）
            center = tuple(map(int, detection.center))
            text = str(detection.id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # 计算文字应该放置的位置，使其居中
            text_x = int(center[0] - text_size[0] / 2)
            text_y = int(center[1] + text_size[1] / 2)

            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

            # 打印检测结果
            print(f"检测到 AprilTag:")
            print(f"  ID: {detection.id}")
            print(f"  汉明距离: {detection.hamming_distance}")
            print(f"  中心: {detection.center}")
            print(f"  角点: {detection.corners}")
            print()

        # 等比例缩放图像
        resized_image = self._resize_image(image, width=800)

        # 显示结果
        cv2.imshow('AprilTag Detection', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def _resize_image(image, width=640):
        """等比例缩放图像到指定宽度"""
        height = int(image.shape[0] * (width / image.shape[1]))
        return cv2.resize(image, (width, height))


if __name__ == '__main__':
    # 创建检测器实例
    detector = ApriltagCornerDetector()

    # 测试从文件读取图片
    test_image_path = 'test.png'
    detections = detector.detect(test_image_path, show_result=True)

    # 测试直接传入图片数组
    test_image = cv2.imread('test.png', cv2.IMREAD_COLOR)
    detections = detector.detect(test_image, show_result=True)

    # 测试检测并绘制结果
    result_image = detector.detect_and_draw(test_image)
    # 显示结果（仅用于测试）
    cv2.imshow('Detection Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
