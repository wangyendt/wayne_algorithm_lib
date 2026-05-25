# author: wangye(Wayne)
# license: Apache Licence
# file: apriltag_detector.py
# time: 2024-10-10-15:33:42
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import os
import cv2
import numpy as np
from typing import Optional, Sequence, Union
from pathlib import Path
from pywayne.cpp_loader import import_cpp_module


class ApriltagCornerDetector:
    _TAG_FAMILY_FACTORIES = {
        "16h5": "tag_codes_16h5",
        "25h7": "tag_codes_25h7",
        "25h9": "tag_codes_25h9",
        "36h9": "tag_codes_36h9",
        "36h11": "tag_codes_36h11",
    }

    def __init__(
        self,
        tag_family: str = "36h11",
        black_border: int = 2,
        preprocess: Optional[Union[str, Sequence[str]]] = None,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size=(8, 8),
    ):
        self.tag_family = self._normalize_tag_family(tag_family)
        self.black_border = black_border
        self.preprocess = preprocess
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        apriltag_detection = self._check_and_import_lib()
        if self.tag_family not in self._TAG_FAMILY_FACTORIES:
            supported = ", ".join(sorted(self._TAG_FAMILY_FACTORIES))
            raise ValueError(f"Unsupported tag_family '{tag_family}'. Supported: {supported}")

        factory = getattr(apriltag_detection, self._TAG_FAMILY_FACTORIES[self.tag_family])
        self.detector = apriltag_detection.TagDetector(factory(), self.black_border)

    def _check_and_import_lib(self):
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
        return import_cpp_module("apriltag_detection", "apriltag_detection", lib_path)

    @staticmethod
    def _normalize_tag_family(tag_family: str) -> str:
        family = str(tag_family).strip().lower().replace("_", "").replace("-", "")
        if family.startswith("tag"):
            family = family[3:]
        return family

    def _normalize_preprocess(self, preprocess):
        preprocess = self.preprocess if preprocess is None else preprocess
        if preprocess is None or preprocess == "" or preprocess == "none":
            return []
        if isinstance(preprocess, str):
            normalized = preprocess.strip().lower().replace("_", "-")
            aliases = {
                "norm-clahe": ["norm", "clahe"],
                "normalize-clahe": ["normalize", "clahe"],
                "hist-eq": ["equalize"],
                "histeq": ["equalize"],
                "hist-equalize": ["equalize"],
                "histogram-equalization": ["equalize"],
                "equalize-hist": ["equalize"],
                "adaptive-histogram-equalization": ["clahe"],
            }
            if normalized in aliases:
                return aliases[normalized]
            return [p for p in normalized.split("-") if p]
        return [str(p).strip().lower().replace("_", "-") for p in preprocess]

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray_image = image
        elif image.ndim == 3 and image.shape[2] == 1:
            gray_image = image[:, :, 0]
        elif image.ndim == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported image shape for AprilTag detection: {image.shape}")

        if gray_image.dtype != np.uint8:
            gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return np.ascontiguousarray(gray_image)

    def _apply_preprocess(self, gray_image: np.ndarray, preprocess=None) -> np.ndarray:
        result = gray_image
        for mode in self._normalize_preprocess(preprocess):
            if mode in {"norm", "normalize"}:
                result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            elif mode in {"clahe"}:
                clahe = cv2.createCLAHE(
                    clipLimit=self.clahe_clip_limit,
                    tileGridSize=self.clahe_tile_grid_size,
                )
                result = clahe.apply(result)
            elif mode in {"equalize", "equalise", "hist", "histogram", "eq"}:
                result = cv2.equalizeHist(result)
            else:
                raise ValueError(f"Unsupported preprocess mode: {mode}")
        return np.ascontiguousarray(result)

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image_path = image
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
        return image

    def detect(self, image: Union[str, Path, np.ndarray], show_result: bool = False, preprocess=None):
        """
        检测图像中的AprilTag
        
        Args:
            image: 输入图像，可以是图片路径（str/Path）或者numpy数组（np.ndarray）
            show_result: 是否显示结果
            preprocess: 可选预处理，支持 None、"norm"、"clahe"、"equalize"、
                "norm-clahe"，或这些模式组成的列表。
            
        Returns:
            detections: 检测结果
        """
        image = self._load_image(image)
        gray_image = self._apply_preprocess(self._to_gray(image), preprocess)
        detections = self.detector.extract_tags(gray_image)
        if show_result:
            color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            self._show(color_image, detections)
        return detections

    def detect_and_draw(self, image: Union[str, Path, np.ndarray], preprocess=None) -> np.ndarray:
        """
        检测图像中的AprilTag并在原图上绘制结果
        
        Args:
            image: 输入图像，可以是图片路径（str）或者numpy数组（np.ndarray）
            preprocess: 传给 detect() 的可选预处理模式
            
        Returns:
            np.ndarray: 绘制了检测结果的图像（与输入图像大小相同）
        """
        image = self._load_image(image)

        # 保存原始图像的副本
        if image.ndim == 2:
            result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            result_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            result_image = image.copy()
        
        # 检测AprilTag
        detections = self.detect(image, preprocess=preprocess)
        
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
