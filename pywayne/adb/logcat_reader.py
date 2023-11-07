# author: wangye(Wayne)
# license: Apache Licence
# file: logcat_reader.py
# time: 2023-11-07-11:00:24
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import os
import sys
import subprocess
import importlib


class AdbLogcatReader:
    def __init__(self, backend='cpp'):
        self.backend = backend
        self.adb_logcat = None
        if self.backend == 'cpp':
            self.adb_logcat = self._check_lib_exists()

    def start(self):
        if self.backend == 'cpp':
            self.adb_logcat.clearLogcat()
            self.adb_logcat.startLogcat()
        elif self.backend == 'python':
            subprocess.run(["adb", "logcat", "-c"])
            self.adb_logcat = subprocess.Popen(["adb", "logcat"], stdout=subprocess.PIPE, text=True)

    def read(self):
        if self.backend == 'cpp':
            while True:
                line = self.adb_logcat.readLine()
                if not line: return
                yield line.rstrip()
        elif self.backend == 'python':
            try:
                while True:
                    line = self.adb_logcat.stdout.readline()
                    if not line: return
                    yield line.rstrip()
            except KeyboardInterrupt:
                # Handle Ctrl-C to stop reading the logcat
                print("Stopping logcat capture.")
            finally:
                self.adb_logcat.terminate()
                self.adb_logcat.wait()

    def _check_lib_exists(self):
        lib_path = os.path.join(os.path.dirname(__file__), 'lib')
        sys.path.append(str(lib_path))
        try:
            from adb_logcat import ADBLogcatReader as AdbLogcat
        except ImportError:
            os.makedirs(lib_path, exist_ok=True)
            subprocess.run(['gettool', 'adb_logcat_reader', '-b', '-t', str(lib_path)], check=True)
            importlib.invalidate_caches()
            AdbLogcat = importlib.import_module("adb_logcat").ADBLogcatReader
        return AdbLogcat()


if __name__ == '__main__':
    reader = AdbLogcatReader()
    reader.start()
    for line in reader.read():
        print(line)
