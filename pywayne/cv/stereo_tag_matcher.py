import cv2
import numpy as np
from pathlib import Path
from pywayne.cv.apriltag_detector import ApriltagCornerDetector


class StereoTagMatcher:
    def __init__(
        self,
        target_height=600,
        line_color=(0, 0, 255),
        line_thickness=2,
        box_thickness=2,
        all_tag_color=(0, 255, 0),
        common_tag_color=(0, 255, 255),
    ):
        self.target_height = target_height
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.box_thickness = box_thickness
        self.all_tag_color = all_tag_color
        self.common_tag_color = common_tag_color

        try:
            self.detector = ApriltagCornerDetector()
            print("ApriltagCornerDetector initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize ApriltagCornerDetector: {e}")
            self.detector = None

    def _load_image(self, image_input):
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                print(f"Warning: Could not read image from {image_input}")
            return img
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()  # Return a copy to avoid modifying original array
        else:
            print(f"Warning: Invalid image input type: {type(image_input)}")
            return None

    def _resize_and_scale(self, image, tags_data):
        """Resizes image and scales tag coordinates."""
        h, w = image.shape[:2]
        if h == 0:
            return None, {}
        scale = self.target_height / h
        target_width = int(w * scale)

        resized_image = cv2.resize(
            image, (target_width, self.target_height), interpolation=cv2.INTER_AREA
        )

        scaled_tags_data = {}
        for tag_id, data in tags_data.items():
            scaled_center = (data["center"][0] * scale, data["center"][1] * scale)
            scaled_corners = [(c[0] * scale, c[1] * scale) for c in data["corners"]]
            scaled_tags_data[tag_id] = {
                "center": scaled_center,
                "corners": scaled_corners,
            }

        return resized_image, scaled_tags_data

    def _draw_tag_box(self, image, corners, color, thickness, offset_x=0):
        """Draws a bounding box using tag corners."""
        pts = np.array(corners, dtype=np.int32)
        pts[:, 0] += offset_x  # Apply x-offset for right image
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

    def process_pair(self, image1_input, image2_input, show=False):
        """
        Processes a pair of images (path or np.ndarray) to find common AprilTags,
        returns matched tag info and the annotated stitched image.

        Args:
            image1_input: Path or np.ndarray for the first image (left).
            image2_input: Path or np.ndarray for the second image (right).
            show (bool): If True, display the resulting stitched image.

        Returns:
            tuple: A tuple `(matched_tags_info, stitched_image)` where `matched_tags_info`
                   is a dictionary of matched tag details and `stitched_image` is the
                   annotated numpy array image (or None on error).

        """
        if self.detector is None:
            print("Error: Detector not initialized.")
            return {}, None

        img1 = self._load_image(image1_input)
        img2 = self._load_image(image2_input)

        if img1 is None or img2 is None:
            print("Error loading one or both images.")
            return {}, None

        # --- Detection ---
        detections1 = self.detector.detect(img1)
        detections2 = self.detector.detect(img2)
        tags1_data_orig = {
            det.id: {"center": det.center, "corners": det.corners}
            for det in detections1
        }
        tags2_data_orig = {
            det.id: {"center": det.center, "corners": det.corners}
            for det in detections2
        }
        print(
            f"  Detected {len(tags1_data_orig)} tags in cam1, {len(tags2_data_orig)} tags in cam2."
        )
        # -----------------

        # --- Resize and Scale for Drawing ---
        img1_res, tags1_scaled = self._resize_and_scale(img1, tags1_data_orig)
        img2_res, tags2_scaled = self._resize_and_scale(img2, tags2_data_orig)
        if img1_res is None or img2_res is None:
            print("Error during resizing.")
            return {}, None
        # ----------------------------------

        # --- Stitching ---
        stitched_image = cv2.hconcat([img1_res, img2_res])
        w1_res = img1_res.shape[1]  # Width of the left image after resizing
        # -----------------

        # --- Find Common Tags & Prepare Output Data ---
        common_ids = set(tags1_scaled.keys()) & set(tags2_scaled.keys())
        print(f"  Found {len(common_ids)} common tags: {common_ids}")

        matched_tags_info = {}
        for tag_id in common_ids:
            matched_tags_info[tag_id] = {
                "cam1_center": tags1_data_orig[tag_id]["center"],
                "cam1_corners": tags1_data_orig[tag_id]["corners"],
                "cam2_center": tags2_data_orig[tag_id]["center"],
                "cam2_corners": tags2_data_orig[tag_id]["corners"],
            }
        # -------------------------------------------

        # --- Drawing Annotations ---
        # 1. Draw GREEN boxes for ALL tags
        for tag_id, data in tags1_scaled.items():
            self._draw_tag_box(
                stitched_image, data["corners"], self.all_tag_color, self.box_thickness
            )
        for tag_id, data in tags2_scaled.items():
            self._draw_tag_box(
                stitched_image,
                data["corners"],
                self.all_tag_color,
                self.box_thickness,
                offset_x=w1_res,
            )

        # 2. Draw YELLOW boxes for COMMON tags (over green)
        for tag_id in common_ids:
            self._draw_tag_box(
                stitched_image,
                tags1_scaled[tag_id]["corners"],
                self.common_tag_color,
                self.box_thickness,
            )
            self._draw_tag_box(
                stitched_image,
                tags2_scaled[tag_id]["corners"],
                self.common_tag_color,
                self.box_thickness,
                offset_x=w1_res,
            )

        # 3. Draw RED lines connecting centers of COMMON tags
        for tag_id in common_ids:
            center1_s = tags1_scaled[tag_id]["center"]
            center2_s = tags2_scaled[tag_id]["center"]
            pt1 = (int(center1_s[0]), int(center1_s[1]))
            pt2 = (int(center2_s[0] + w1_res), int(center2_s[1]))
            cv2.line(stitched_image, pt1, pt2, self.line_color, self.line_thickness)
        # -------------------------

        # --- Optional Display ---
        if show:
            display_name = "Stereo Tag Matching Result"
            cv2.namedWindow(display_name, cv2.WINDOW_NORMAL)
            cv2.imshow(display_name, stitched_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # -----------------------

        return matched_tags_info, stitched_image


if __name__ == "__main__":
    # --- Configuration ---
    CAM1_DIR = Path("path/to/cam1")
    CAM2_DIR = Path("path/to/cam2")
    TARGET_HEIGHT = 600  # Fixed height for resizing images
    LINE_COLOR = (0, 0, 255)  # Red in BGR
    LINE_THICKNESS = 2
    BOX_THICKNESS = 4
    ALL_TAG_COLOR = (0, 255, 0)  # Green in BGR
    COMMON_TAG_COLOR = (0, 255, 255)  # Yellow in BGR
    # ---------------------
    # Pass configuration to the main function
    config = {
        "target_height": TARGET_HEIGHT,
        "line_color": LINE_COLOR,
        "line_thickness": LINE_THICKNESS,
        "box_thickness": BOX_THICKNESS,
        "all_tag_color": ALL_TAG_COLOR,
        "common_tag_color": COMMON_TAG_COLOR,
    }
    matcher = StereoTagMatcher(**config)

    path1 = Path(CAM1_DIR)
    path2 = Path(CAM2_DIR)

    for f1, f2 in zip(path1.glob("*.png"), path2.glob("*.png")):
        img1 = cv2.imread(str(f1))
        img2 = cv2.imread(str(f2))
        img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        matcher.process_pair(img1, img2, show=True)
