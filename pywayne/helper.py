# author: wangye(Wayne)
# license: Apache Licence
# file: helper.py
# time: 2024-12-20-19:06:36
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import inspect
import os
import time
from functools import reduce
from pathlib import Path
from pywayne.tools import wayne_print, read_yaml_config, write_yaml_config


class Helper:
    def __init__(self, project_root: str = '', config_file_name: str = 'common_info.yaml'):
        if not project_root:
            caller_frame = inspect.stack()[1]
            caller_filename = caller_frame.filename
            self.project_root = os.path.dirname(caller_filename)  # 获取目录路径
        self.project_root = Path(project_root)
        self.config_path = Path(self.project_root) / config_file_name
        if not self.config_path.exists():
            write_yaml_config(self.config_path.name, {})

    def get_proj_root(self):
        return str(self.project_root.absolute())

    def get_config_path(self):
        return str(self.config_path.absolute())

    def set_module_value(self, *keys, value):
        config_path = self.config_path.absolute()
        if not config_path.exists():
            wayne_print(f'Set value failed: config file {config_path} not found, return None', 'yellow')
            return
        write_yaml_config(str(config_path), reduce(
            lambda d, k: {k: d},
            reversed(keys),
            value
        ), update=True)

    def get_module_value(self, *keys, max_waiting_time: float = 0.0, debug: bool = True):
        config_path = self.config_path.absolute()
        if max_waiting_time > 0.0:
            time_start = time.time()
            while time.time() - time_start <= max_waiting_time:
                if not config_path.exists:
                    if debug:
                        wayne_print(f'waiting for config file {config_path}...', 'yellow')
                    time.sleep(1)
                    continue

                try:
                    value = reduce(dict.get, keys, read_yaml_config(str(config_path)))
                    if value is not None:
                        return value
                    if debug:
                        wayne_print(f'waiting for {keys=} [value not found]...', 'yellow')
                    time.sleep(1)
                except Exception as e:
                    if debug:
                        wayne_print(f'waiting for {keys=} [{str(e)}]...', 'yellow')
                    time.sleep(1)
            wayne_print(f'Timeout: could not read value at {keys=}, return None', 'red', True)
            return None

        if not config_path.exists():
            wayne_print(f'Get value failed: config file {config_path} not found, return None', 'yellow')
            return None

        try:
            return reduce(dict.get, keys, read_yaml_config(str(config_path)))
        except Exception as e:
            wayne_print(f'Get value failed: {e}, return None', 'red', True)
            return None

    def delete_module_value(self, *keys):
        config_path = self.config_path.absolute()
        if not config_path.exists():
            wayne_print(f'Delete value failed: config file {config_path} not found, return', 'yellow')
            return

        try:
            config = read_yaml_config(str(config_path))
            parent = reduce(dict.get, keys[:-1], config or {})
            if parent is not None and keys[-1] in parent:
                del parent[keys[-1]]
                write_yaml_config(str(config_path), config, update=False)

        except Exception as e:
            wayne_print(f'Delete value failed: {e}', 'red', True)


if __name__ == '__main__':
    helper = Helper('./')
    helper.set_module_value('a', 'b', value=2)
    helper.set_module_value('a', 'c', value=1)
    helper.set_module_value('d', 'e', 'f', value=3)
    helper.delete_module_value('a', 'c')
    helper.delete_module_value('a')
    helper.delete_module_value('d')
