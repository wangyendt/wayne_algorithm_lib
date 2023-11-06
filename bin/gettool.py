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


def fetch_tool(tool_name, target_dir='', build=False, clean=False):
    print(f"Fetching tool: {tool_name}")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        subprocess.run(["git", "clone", "--sparse", "https://github.com/wangyendt/cpp_tools", temp_dir])
        name_to_path_map_yaml_file = 'name_to_path_map.yaml'
        assert name_to_path_map_yaml_file in os.listdir('.')
        with open(name_to_path_map_yaml_file, 'r') as f:
            name_to_path_map = yaml.safe_load(f)
        tool_path = name_to_path_map[tool_name]
        if not target_dir:
            target_dir = os.path.join(cwd, tool_path)
        if not build and os.path.exists(target_dir):
            if input(f'{target_dir} already exists, still want to fetch? (Y/N)').lower() != 'y': return
        with open('.gitmodules', 'r') as f:
            tool_is_submodule = any(f'path = {tool_path}' in line for line in f)
        if tool_is_submodule:
            subprocess.run(["git", "submodule", "update", "--init", "--recursive", tool_path])
        else:
            subprocess.run(["git", "sparse-checkout", "set", tool_path])
        if build:
            os.system(f'cd {tool_path} && mkdir -p build && cd build && cmake .. && make -j12')
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(f'{tool_path}/lib/', target_dir)
        else:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            if clean:
                shutil.copytree(os.path.join(tool_path, 'src'), target_dir)
            else:
                shutil.copytree(tool_path, target_dir)
        print(f"Tool {tool_name} has been copied to {target_dir}")
    os.chdir(cwd)


def print_supported_tools():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        subprocess.run(
            ["git", "clone", "--sparse", "https://github.com/wangyendt/cpp_tools", temp_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        name_to_path_map_yaml_file = 'name_to_path_map.yaml'
        assert name_to_path_map_yaml_file in os.listdir('.')
        with open(name_to_path_map_yaml_file, 'r') as f:
            name_to_path_map = yaml.safe_load(f)
            print('Currently supported tools:')
            print(', '.join(name_to_path_map.keys()))


def main():
    parser = argparse.ArgumentParser(description='Tool fetcher.')

    parser.add_argument('name_pos', nargs='?', default=None, help='Name of the tool (positional)')
    parser.add_argument('-n', '--name', default=None, help='Name of the tool')
    parser.add_argument('-t', '--target_path', default='', type=str, help='Target path for the tool')
    parser.add_argument('-b', '--build', action='store_true', help='Whether to build lib')
    parser.add_argument('-U', '--upgrade', action='store_true', help='Upgrade the tool')
    parser.add_argument('-f', '--force', action='store_true', help='Force action')
    parser.add_argument('-c', '--clean', action='store_true', help='Only fetch c++ sources')
    parser.add_argument('-l', '--list', action='store_true', help='List current supported tools')

    args = parser.parse_args()

    if args.list:
        print_supported_tools()
        return

    # 如果通过 -n 或 --name 提供了名称，则使用它，否则使用位置参数提供的名称
    tool_name = args.name if args.name is not None else args.name_pos
    target_path = args.target_path
    build = args.build
    clean = args.clean

    # 检查是否提供了名称
    if tool_name is None:
        parser.error("the following arguments are required: name")

    fetch_tool(tool_name, target_dir=target_path, build=build, clean=clean)

    if args.upgrade:
        print("(not implemented yet)")
    if args.force:
        print("(not implemented yet)")


if __name__ == '__main__':
    main()
