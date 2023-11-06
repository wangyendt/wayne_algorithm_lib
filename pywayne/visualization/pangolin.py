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


if __name__ == '__main__':
    import numpy as np


    def random_walk_3d(steps=1000):
        steps = np.random.uniform(-0.001, 0.001, (steps, 3))
        walk = np.cumsum(steps, axis=0)
        return walk


    viewer = PangolinViewer(640, 480, False).viewer
    viewer.set_img_resolution(752, 480)
    viewer.view_init()
    cur_pos = np.zeros((100, 3))
    while viewer.should_quit():
        random_walk = random_walk_3d(100)
        cur_pos = cur_pos + random_walk
        viewer.publish_3D_points(cur_pos, cur_pos)
        viewer.show(delay_time_in_s=0.0)
