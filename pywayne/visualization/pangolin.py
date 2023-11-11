# author: wangye(Wayne)
# license: Apache Licence
# file: pangolin.py
# time: 2023-11-06-11:14:19
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import sys
import subprocess
import os
import importlib
import numpy as np


class PangolinViewer:
    def __init__(self, width: int, height: int, run_on_start=False):
        self.width = width
        self.height = height
        self.run_on_start = run_on_start
        self.viewer = self._check_lib_exists()

    def _check_lib_exists(self):
        lib_path = os.path.join(os.path.dirname(__file__), 'lib')
        sys.path.append(str(lib_path))
        try:
            from pangolin_viewer import PangolinViewer as Viewer
        except ImportError:
            os.makedirs(lib_path, exist_ok=True)
            subprocess.run(['gettool', 'pangolin', '-b', '-t', str(lib_path)], check=True)
            importlib.invalidate_caches()
            Viewer = importlib.import_module("pangolin_viewer").PangolinViewer
        return Viewer(self.width, self.height, self.run_on_start)

    def run(self):
        self.viewer.run()

    def close(self):
        self.viewer.close()

    def join(self):
        self.viewer.join()

    def reset(self):
        self.viewer.reset()

    def init(self):
        self.viewer.view_init()

    def show(self, delay_time_in_s=0.0):
        self.viewer.show(delay_time_in_s)

    def should_not_quit(self):
        return self.viewer.should_not_quit()

    def set_img_resolution(self, width: int, height: int):
        self.viewer.set_img_resolution(width, height)

    def publish_traj(self, t_wc: np.ndarray, q_wc: np.ndarray):
        self.viewer.publish_traj(t_wc=t_wc, q_wc=q_wc)

    def publish_3D_points(self, slam_pts, msckf_pts):
        self.viewer.publish_3D_points(slam_pts, msckf_pts)

    def publish_track_img(self, img: np.ndarray):
        assert img.dtype == np.uint8 and len(img.shape) == 3
        self.viewer.publish_track_img(img)

    def publish_vio_opt_data(self, vals):
        self.viewer.publish_vio_opt_data(vals)

    def publish_plane_detection_img(self, img: np.ndarray):
        assert img.dtype == np.uint8 and len(img.shape) == 3
        self.viewer.publish_plane_detection_img(img)

    def publish_plane_triangulate_pts(self, plane_tri_pts):
        self.viewer.publish_plane_triangulate_pts(plane_tri_pts)

    def publish_plane_vio_stable_pts(self, plane_vio_stable_pts):
        self.viewer.publish_plane_vio_stable_pts(plane_vio_stable_pts)

    def publish_planes_horizontal(self, his_planes_horizontal):
        self.viewer.publish_planes_horizontal(his_planes_horizontal)

    def publish_planes_vertical(self, planes):
        self.viewer.publish_planes_vertical(planes)

    def publish_traj_gt(self, q_wc_gt, t_wc_gt):
        self.viewer.publish_traj_gt(q_wc_gt, t_wc_gt)

    def get_algorithm_wait_flag(self):
        return self.viewer.get_algorithm_wait_flag()

    def set_visualize_opencv_mat(self):
        self.viewer.set_visualize_opencv_mat()

    def algorithm_wait(self):
        self.viewer.algorithm_wait()

    def notify_algorithm(self):
        self.viewer.notify_algorithm()


if __name__ == '__main__':
    import numpy as np


    def random_walk_3d(steps=1000):
        steps = np.random.uniform(-0.001, 0.001, (steps, 3))
        walk = np.cumsum(steps, axis=0)
        return walk


    viewer = PangolinViewer(640, 480, False)  # .viewer
    viewer.set_img_resolution(640, 480)
    viewer.init()
    cur_pos = np.zeros((100, 3))
    while viewer.should_not_quit():
        random_walk = random_walk_3d(100)
        cur_pos = cur_pos + random_walk
        viewer.publish_3D_points(cur_pos, cur_pos)
        viewer.show(delay_time_in_s=0.0)
