# author: wangye(Wayne)
# license: Apache Licence
# file: gettool.py
# time: 2023-11-01-22:47:10
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import argparse
import tempfile
import shutil
import subprocess
import os
import yaml


def fetch_tool(tool_name, target_dir=None):
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(["git", "clone", "--sparse", "https://github.com/wangyendt/cpp_tools", temp_dir])
        os.chdir(temp_dir)
        name_to_path_map_yaml_file = 'name_to_path_map.yaml'
        assert name_to_path_map_yaml_file in os.listdir('.')
        with open(name_to_path_map_yaml_file, 'r') as f:
            name_to_path_map = yaml.safe_load(f)
        subprocess.run(["git", "sparse-checkout", "set", name_to_path_map[tool_name]])
        if target_dir is None:
            target_dir = os.path.join(cwd, name_to_path_map[tool_name])
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(name_to_path_map[tool_name], target_dir)
        print(f"Tool {tool_name} has been copied to {target_dir}")
    os.chdir(cwd)


def main():
    parser = argparse.ArgumentParser(description='Tool fetcher.')

    parser.add_argument('name_pos', nargs='?', default=None, help='Name of the tool (positional)')
    parser.add_argument('-n', '--name', default=None, help='Name of the tool')
    parser.add_argument('-U', '--upgrade', action='store_true', help='Upgrade the tool')
    parser.add_argument('-f', '--force', action='store_true', help='Force action')

    args = parser.parse_args()

    # 如果通过 -n 或 --name 提供了名称，则使用它，否则使用位置参数提供的名称
    tool_name = args.name if args.name is not None else args.name_pos

    # 检查是否提供了名称
    if tool_name is None:
        parser.error("the following arguments are required: name")

    fetch_tool(tool_name)

    print(f"Fetching tool: {tool_name}")
    if args.upgrade:
        print("(not implemented yet)")
    if args.force:
        print("(not implemented yet)")


if __name__ == '__main__':
    main()
