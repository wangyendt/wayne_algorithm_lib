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
from pywayne.tools import wayne_print, read_yaml_config, write_yaml_config


def sparse_clone(url: str, target_dir: str):
    result = subprocess.run(f'git clone --sparse {url} {target_dir}', shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        result = subprocess.run(f'git clone --no-checkout {url} {target_dir}', shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            cwd = os.getcwd()
            os.chdir(target_dir)
            subprocess.run('git sparse-checkout init --cone', shell=True, capture_output=True, text=True)
            os.chdir(cwd)
            wayne_print(f'Cloned repository from {url} to {target_dir} with sparse-checkout initialized.', 'yellow')
        else:
            wayne_print(f'Failed to clone repository from {url} to {target_dir}.', 'red')
            exit(0)
    else:
        wayne_print(f'Successfully cloned repository from {url} to {target_dir} with --sparse option.', 'green')


def fetch_tool(url: str, tool_name, target_dir='', build=False, clean=False):
    print(f"Fetching tool: {tool_name}")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        sparse_clone(url, temp_dir)
        name_to_path_map_yaml_file = 'name_to_path_map.yaml'
        assert name_to_path_map_yaml_file in os.listdir('.'), f'Failed to find {name_to_path_map_yaml_file} in {temp_dir}'
        name_to_path_map = read_yaml_config(name_to_path_map_yaml_file)
        tool_path = name_to_path_map[tool_name]['path']
        if not target_dir:
            target_dir = os.path.join(cwd, tool_path)
        if not build and os.path.exists(target_dir):
            if input(f'{target_dir} already exists, still want to fetch? (Y/N)').lower() != 'y': return
        if os.path.exists('.gitmodules'):
            with open('.gitmodules', 'r') as f:
                tool_is_submodule = any(f'path = {tool_path}' in line for line in f)
        else:
            tool_is_submodule = False
        if tool_is_submodule:
            subprocess.run(["git", "submodule", "update", "--init", "--recursive", tool_path])
        else:
            subprocess.run(["git", "sparse-checkout", "set", tool_path])
        if build:
            if not name_to_path_map[tool_name]['buildable']:
                wayne_print(f'{tool_name} is not buildable, skip building', 'red')
            elif not os.path.exists(f'{tool_path}/CMakeLists.txt'):
                wayne_print(f'{tool_name} does not have a CMakeLists.txt, skip building', 'red')
            else:
                command = f'cd {tool_path} && mkdir -p build && cd build && cmake .. && make -j12'
                os.system(command)
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


def print_supported_tools(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        sparse_clone(url, temp_dir)
        name_to_path_map_yaml_file = 'name_to_path_map.yaml'
        assert name_to_path_map_yaml_file in os.listdir('.'), f"{name_to_path_map_yaml_file} not found in {temp_dir}"
        name_to_path_map = read_yaml_config(name_to_path_map_yaml_file)
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
    parser.add_argument('--get-url', action='store_true', help='get current URL of the tool')
    parser.add_argument('--set-url', type=str, default='', help='set current URL of the tool, e.g. "https://github.com/wangyendt/cpp_tools"')
    parser.add_argument('--reset-url', action='store_true', help='reset url to default: "https://github.com/wangyendt/cpp_tools"')

    args = parser.parse_args()

    exe_dir = os.path.dirname(__file__)
    config_yaml_file = os.path.normpath(f'{exe_dir}/config.yaml')

    url = "https://github.com/wangyendt/cpp_tools"
    if args.set_url:
        url = args.set_url
    if args.reset_url or args.set_url:
        write_yaml_config(config_yaml_file, {'url': url}, use_lock=False)
        return
    if not os.path.exists(config_yaml_file):
        wayne_print(f'config file {config_yaml_file} not found, use default url: {url}', 'yellow')
        write_yaml_config(config_yaml_file, {'url': url}, use_lock=False)
    config = read_yaml_config(config_yaml_file, use_lock=False)
    if config:
        url = config['url']
    else:
        wayne_print(f'read_yaml_config failed: {config_yaml_file}', 'yellow')

    if args.get_url:
        wayne_print(f'current url is: {url}', 'green')
        return

    if args.list:
        print_supported_tools(url)
        return

    # 如果通过 -n 或 --name 提供了名称，则使用它，否则使用位置参数提供的名称
    tool_name = args.name if args.name is not None else args.name_pos
    target_path = args.target_path
    build = args.build
    clean = args.clean

    # 检查是否提供了名称
    if tool_name is None:
        parser.error("the following arguments are required: name")

    fetch_tool(url, tool_name, target_dir=target_path, build=build, clean=clean)

    if args.upgrade:
        wayne_print("(not implemented yet)", "yellow")
    if args.force:
        wayne_print("(not implemented yet)", "yellow")


if __name__ == '__main__':
    main()
