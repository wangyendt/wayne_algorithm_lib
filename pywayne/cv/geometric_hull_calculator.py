# author: wangye(Wayne)
# license: Apache Licence
# file: geometric_hull_calculator.py
# time: 2024-10-11-14:25:14
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from concave_hull import concave_hull
import alphashape
from shapely.geometry import Polygon, MultiPolygon


class GeometricHullCalculator:
    """
    A class for calculating and visualizing geometric hulls of a set of points.

    This class provides methods to compute minimum bounding rectangles, convex hulls,
    and concave hulls for a given set of 2D points. It also offers visualization
    options using both OpenCV and Matplotlib.

    Attributes:
        points (np.ndarray): The input set of 2D points.
        algorithm (str): The algorithm to use for concave hull calculation ('concave-hull' or 'alphashape').
        use_filtered_pts (bool): Whether to use filtered points for hull calculations.
        box (np.ndarray): The minimum bounding rectangle of the points.
        center (np.ndarray): The center point of the input points.
        filter_radius (float): The radius used for filtering points.
        concave_hull_result (np.ndarray or list): The resulting concave hull.
        convex_hull_points (np.ndarray): The convex hull points.
    """

    def __init__(self, points, algorithm='concave-hull', use_filtered_pts=False):
        """
        Initialize the GeometricHullCalculator with a set of points and options.

        Args:
            points (np.ndarray): The input set of 2D points.
            algorithm (str): The algorithm to use for concave hull calculation.
            use_filtered_pts (bool): Whether to use filtered points for hull calculations.
        """
        self.points = points
        self.algorithm = algorithm
        self.use_filtered_pts = use_filtered_pts
        self.box, self.center, self.filter_radius, self.concave_hull_result, self.convex_hull_points = self._process_points()

    @staticmethod
    def generate_random_points(num_points=100, scale=50, offset=150):
        """
        Generate a set of random 2D points.

        Args:
            num_points (int): The number of points to generate.
            scale (float): The scale factor for the random distribution.
            offset (float): The offset to add to all points.

        Returns:
            np.ndarray: An array of random 2D points.
        """
        return np.random.randn(num_points, 2) * scale + offset

    @staticmethod
    def _compute_mbr(points):
        """
        Compute the Minimum Bounding Rectangle (MBR) of a set of points.

        Args:
            points (np.ndarray): The input set of 2D points.

        Returns:
            np.ndarray: The four corners of the MBR.
        """
        rect = cv2.minAreaRect(points.astype(np.float32))
        return np.intp(cv2.boxPoints(rect))

    @staticmethod
    def _compute_convex_hull(points):
        """
        Compute the Convex Hull of a set of points.

        Args:
            points (np.ndarray): The input set of 2D points.

        Returns:
            np.ndarray: The points forming the Convex Hull.
        """
        hull = ConvexHull(points)
        return points[hull.vertices]

    @staticmethod
    def _filter_points(points, center, radius):
        """
        Filter out points within a certain radius of the center.

        Args:
            points (np.ndarray): The input set of 2D points.
            center (np.ndarray): The center point.
            radius (float): The filter radius.

        Returns:
            np.ndarray: The filtered set of points.
        """
        return np.array([point for point in points if np.linalg.norm(point - center) > radius])

    def _process_points(self):
        """
        Process the input points to compute various geometric properties.

        Returns:
            tuple: Contains the MBR, center, filter radius, concave hull, and convex hull.
        """
        box = self._compute_mbr(self.points)
        edge_lengths = np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1])
        short_edge = min(edge_lengths)
        filter_radius = short_edge * 0.3
        center = np.mean(self.points, axis=0).astype(int)

        filtered_points = self._filter_points(self.points, center, filter_radius) if self.use_filtered_pts else self.points

        if self.algorithm == 'concave-hull':
            concave_hull_result = concave_hull(filtered_points, concavity=2.0, length_threshold=50.0)
        elif self.algorithm == 'alphashape':
            alpha = 0.95 * alphashape.optimizealpha(filtered_points)
            alpha_shape = alphashape.alphashape(filtered_points, alpha)
            concave_hull_result = self._process_alphashape(alpha_shape)
        else:
            raise ValueError("Invalid algorithm specified")

        convex_hull_points = self._compute_convex_hull(self.points)
        return box, center, filter_radius, concave_hull_result, convex_hull_points

    @staticmethod
    def _process_alphashape(alpha_shape):
        """
        Process the result of alphashape algorithm.

        Args:
            alpha_shape (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The alpha shape result.

        Returns:
            np.ndarray or list: The processed concave hull result.
        """
        if isinstance(alpha_shape, MultiPolygon):
            return [np.array(poly.exterior.coords) for poly in alpha_shape.geoms]
        elif isinstance(alpha_shape, Polygon):
            return np.array(alpha_shape.exterior.coords)
        else:
            raise TypeError("Unexpected type for alpha_shape")

    def compute_concave_hull_area(self):
        """
        Compute the area of the concave hull.

        Returns:
            float: The area of the concave hull.
        """
        if self.algorithm == 'alphashape':
            if isinstance(self.concave_hull_result, list):
                polygon = MultiPolygon([Polygon(coords) for coords in self.concave_hull_result])
            else:
                polygon = Polygon(self.concave_hull_result)
        else:
            polygon = Polygon(self.concave_hull_result)
        return polygon.area

    def visualize_opencv(self):
        """
        Visualize the geometric shapes using OpenCV.
        """
        max_x = int(np.max(self.points[:, 0])) + 50  # 宽度，增加一些边距
        max_y = int(np.max(self.points[:, 1])) + 50
        img = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255
        self._draw_points_opencv(img, self.points)
        self._draw_polygon_opencv(img, self.box, color=(0, 0, 255))
        if self.use_filtered_pts:
            self._draw_center_and_circle_opencv(img, self.center, self.filter_radius)
        self._draw_hulls_opencv(img, self.concave_hull_result, self.convex_hull_points)
        img_with_legend = self._add_legend_opencv(img)
        cv2.imshow('Geometric Shapes', img_with_legend)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_matplotlib(self):
        """
        Visualize the geometric shapes using Matplotlib.
        """
        fig, ax = plt.subplots()
        self._draw_points_matplotlib(ax, self.points)
        self._draw_polygon_matplotlib(ax, self.box, color='blue', label='MBR')
        if self.use_filtered_pts:
            self._draw_center_and_circle_matplotlib(ax, self.center, self.filter_radius)
        self._draw_hulls_matplotlib(ax, self.concave_hull_result, self.convex_hull_points)
        ax.legend()
        ax.set_aspect('equal')
        plt.show()

    @staticmethod
    def _draw_points_opencv(img, points, color=(255, 0, 0), radius=3):
        """Draw points on an OpenCV image."""
        for point in points:
            cv2.circle(img, (int(point[0]), int(point[1])), radius, color, -1)

    @staticmethod
    def _draw_polygon_opencv(img, points, color=(0, 255, 0), thickness=2):
        """Draw a polygon on an OpenCV image."""
        if points.shape[0] > 1:
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)

    @staticmethod
    def _draw_center_and_circle_opencv(img, center, radius, color=(0, 255, 255), center_radius=5, thickness=2):
        """Draw center point and filter circle on an OpenCV image."""
        cv2.circle(img, (center[0], center[1]), center_radius, color, -1)
        cv2.circle(img, (center[0], center[1]), int(radius), color, thickness)

    def _draw_hulls_opencv(self, img, concave_hull, convex_hull_points):
        """Draw concave and convex hulls on an OpenCV image."""
        self._draw_polygon_opencv(img, np.array(concave_hull, dtype=np.int32), color=(0, 255, 0))
        self._draw_polygon_opencv(img, convex_hull_points.astype(np.int32), color=(255, 255, 0))

    @staticmethod
    def _add_legend_opencv(img):
        """Add a legend to an OpenCV image."""
        legend_items = [
            ("Points", (255, 0, 0)),
            ("MBR", (0, 0, 255)),
            ("Center & Filter Radius", (0, 255, 255)),
            ("Concave Hull", (0, 255, 0)),
            ("Convex Hull", (255, 255, 0))
        ]

        legend_height = 20 * len(legend_items)
        legend_img = np.ones((legend_height, img.shape[1], 3), dtype=np.uint8) * 255
        for i, (text, color) in enumerate(legend_items):
            cv2.putText(legend_img, text, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return np.vstack((legend_img, img))

    @staticmethod
    def _draw_points_matplotlib(ax, points, color='red'):
        """Draw points on a Matplotlib axes."""
        ax.scatter(points[:, 0], points[:, 1], color=color, label='Points')

    @staticmethod
    def _draw_polygon_matplotlib(ax, points, color='green', label='Polygon'):
        """Draw a polygon on a Matplotlib axes."""
        if points.shape[0] > 1:
            polygon = plt.Polygon(points, closed=True, fill=None, edgecolor=color, label=label)
            ax.add_patch(polygon)

    @staticmethod
    def _draw_center_and_circle_matplotlib(ax, center, radius, color='blue'):
        """Draw center point and filter circle on a Matplotlib axes."""
        ax.scatter(center[0], center[1], color=color, label='Center')
        circle = plt.Circle(center, radius, color=color, fill=False, label='Filter Radius')
        ax.add_patch(circle)

    def _draw_hulls_matplotlib(self, ax, concave_hull, convex_hull_points):
        """Draw concave and convex hulls on a Matplotlib axes."""
        self._draw_polygon_matplotlib(ax, np.array(concave_hull), color='orange', label='Concave Hull')
        self._draw_polygon_matplotlib(ax, convex_hull_points, color='purple', label='Convex Hull')

    def get_mbr(self):
        """Get the Minimum Bounding Rectangle."""
        return self.box

    def get_convex_hull(self):
        """Get the Convex Hull points."""
        return self.convex_hull_points

    def get_concave_hull(self):
        """Get the Concave Hull result."""
        return self.concave_hull_result


if __name__ == "__main__":
    points = GeometricHullCalculator.generate_random_points()
    calculator = GeometricHullCalculator(points, algorithm='concave-hull', use_filtered_pts=False)
    # calculator = GeometricHullCalculator(points, algorithm='alphashape', use_filtered_pts=True)
    print("MBR:", calculator.get_mbr())
    print("Convex Hull:", calculator.get_convex_hull())
    print("Concave Hull:", calculator.get_concave_hull())

    visualization_method = 'matplotlib'  # or 'opencv'
    if visualization_method == 'opencv':
        calculator.visualize_opencv()
    else:
        calculator.visualize_matplotlib()
