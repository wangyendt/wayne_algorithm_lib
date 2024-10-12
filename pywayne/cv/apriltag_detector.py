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

    def detect(self, image: np.ndarray, show_result: bool = False):
        if image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        detections = self.apriltag_36h11_detector.extract_tags(gray_image)
        if show_result:
            color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            self._show(color_image, detections)
        return detections

    def _show(self, image: np.ndarray, detections):
        # 在图像上绘制检测结果
        for detection in detections:
            # 绘制边框（绿色）
            corners = np.array(detection.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [corners], True, (0, 255, 0), 8)  # 绿色 (B, G, R)

            # 绘制角点（红色圆圈）
            for corner in corners:
                cv2.circle(image, tuple(corner[0]), 5, (0, 0, 255), -1)  # 红色 (B, G, R)

            # 绘制 ID（红色文字）
            center = tuple(map(int, detection.center))
            text = str(detection.id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            font_thickness = 6
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
    test_image = cv2.imread('test.png', cv2.IMREAD_COLOR)
    detector = ApriltagCornerDetector()
    detections = detector.detect(test_image, show_result=True)
