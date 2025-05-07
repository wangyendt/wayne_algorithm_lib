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
    wayne_print(f"Attempting to clone repository from {url} to {target_dir} with --sparse option...", "cyan")
    result_sparse = subprocess.run(f'git clone --sparse --progress {url} "{target_dir}"', shell=True, text=True)
    if result_sparse.returncode != 0:
        wayne_print(f'Sparse clone failed. Attempting clone with --no-checkout for manual sparse-checkout init..._OUTPUT', 'yellow')
        result_no_checkout = subprocess.run(f'git clone --no-checkout --progress {url} "{target_dir}"', shell=True, text=True)
        if result_no_checkout.returncode == 0:
            cwd = os.getcwd()
            try:
                os.chdir(target_dir)
                wayne_print(f'Initializing sparse-checkout in {target_dir}..._OUTPUT', 'cyan')
                init_result = subprocess.run('git sparse-checkout init --cone', shell=True, text=True, capture_output=True)
                if init_result.returncode == 0:
                    wayne_print(f'Cloned repository from {url} to {target_dir} with sparse-checkout initialized.', 'yellow')
                else:
                    wayne_print(f'Failed to initialize sparse-checkout in {target_dir}. Error: {init_result.stderr}', 'red')
                    # exit(0) # Consider if failure here is critical enough to exit
            finally:
                os.chdir(cwd)
        else:
            wayne_print(f'Failed to clone repository from {url} to {target_dir} even with --no-checkout.', 'red')
            exit(0)
    else:
        wayne_print(f'Successfully cloned repository from {url} to {target_dir} with --sparse option.', 'green')


def handle_installation(
    tool_name: str,
    name_to_path_map: dict,
    main_tool_source_dir: str,
    cpp_tools_repo_temp_dir: str,
    global_install_flag_str: str
):
    """
    Handles the installation of a tool by executing its installation script.
    """
    wayne_print(f"Attempting to install {tool_name}...", "cyan")

    tool_config = name_to_path_map.get(tool_name)
    if not tool_config:
        wayne_print(f"Error: Configuration for {tool_name} not found in name_to_path_map.yaml. Cannot install.", "red")
        return

    if not tool_config.get('installable', False):
        wayne_print(f"Info: Tool {tool_name} is marked as not installable. Skipping installation.", "yellow")
        return

    installation_details = tool_config.get('installation')
    if not installation_details:
        wayne_print(f"Error: No 'installation' details found for {tool_name} in name_to_path_map.yaml. Cannot install.", "red")
        return

    install_script_relative_path = installation_details.get('install_script')
    if not install_script_relative_path:
        wayne_print(f"Error: 'install_script' not specified for {tool_name} in installation details. Cannot install.", "red")
        return

    install_script_abs_path = os.path.join(cpp_tools_repo_temp_dir, install_script_relative_path)

    if not os.path.exists(install_script_abs_path):
        wayne_print(f"Error: Installation script '{install_script_abs_path}' not found. Cannot install {tool_name}.", "red")
        return
    
    try:
        os.chmod(install_script_abs_path, 0o755) # Make script executable
    except Exception as e:
        wayne_print(f"Error: Failed to make installation script '{install_script_abs_path}' executable: {e}", "red")
        return

    script_args = [install_script_abs_path, main_tool_source_dir]
    extra_deps = installation_details.get('extra_dependencies', [])

    # Ensure source code for extra dependencies is checked out in the temp repo
    if extra_deps:
        wayne_print(f"Tool {tool_name} has extra dependencies: {', '.join(extra_deps)}. Ensuring they are checked out in {cpp_tools_repo_temp_dir}...", "cyan")
        original_cwd_for_deps_checkout = os.getcwd()
        try:
            os.chdir(cpp_tools_repo_temp_dir) # Change to the temp repo root

            # Get a list of submodule paths from .gitmodules to differentiate handling
            known_submodule_paths = set()
            gitmodules_path = os.path.join(cpp_tools_repo_temp_dir, ".gitmodules")
            if os.path.exists(gitmodules_path):
                try:
                    # subprocess.run is safer than parsing manually if git is available
                    status_result = subprocess.run(["git", "submodule", "status"], capture_output=True, text=True, check=True)
                    for line in status_result.stdout.strip().split('\n'):
                        if line:
                            parts = line.strip().split()
                            if len(parts) > 1:
                                known_submodule_paths.add(parts[1]) # Path is usually the second element
                except Exception as e:
                    wayne_print(f"Warning: Could not parse .gitmodules or run 'git submodule status': {e}", "yellow")
            wayne_print(f"Known submodule paths in {cpp_tools_repo_temp_dir}: {known_submodule_paths if known_submodule_paths else 'None'}", "magenta")

            for dep_name in extra_deps:
                dep_config = name_to_path_map.get(dep_name)
                if not dep_config or not dep_config.get('path'):
                    wayne_print(f"Error: Path for dependency '{dep_name}' not found in config. Cannot ensure checkout.", "red")
                    continue
                
                dep_relative_path = dep_config['path']
                wayne_print(f"Processing dependency '{dep_name}' (path: {dep_relative_path})...", "magenta")

                if dep_relative_path in known_submodule_paths:
                    wayne_print(f"Dependency '{dep_name}' ({dep_relative_path}) is a submodule. Updating...", "blue")
                    # For real-time progress, do not capture output here.
                    # Ensure the command itself is verbose (--progress).
                    dep_checkout_process = subprocess.run(
                        ["git", "submodule", "update", "--init", "--recursive", "--progress", dep_relative_path],
                        text=True, # Keep text=True for consistent output handling if any parsing were needed
                        check=False # We check returncode manually
                    )
                    if dep_checkout_process.returncode == 0:
                        wayne_print(f"Submodule '{dep_name}' ({dep_relative_path}) updated successfully.", "green")
                    else:
                        wayne_print(f"Warning: Failed to update submodule '{dep_name}' ({dep_relative_path}). Return code: {dep_checkout_process.returncode}", "yellow")
                else:
                    wayne_print(f"Dependency '{dep_name}' ({dep_relative_path}) is a direct path. Adding to sparse checkout...", "blue")
                    dep_checkout_result = subprocess.run(
                        ["git", "sparse-checkout", "add", dep_relative_path],
                        capture_output=True, text=True, check=False
                    )
                    if dep_checkout_result.returncode == 0:
                        wayne_print(f"Successfully added/ensured '{dep_relative_path}' for dependency '{dep_name}'.", "green")
                    else:
                        wayne_print(f"Warning: Failed to add '{dep_relative_path}' for '{dep_name}' to sparse checkout. Error: {dep_checkout_result.stderr.strip()}", "yellow")
        finally:
            os.chdir(original_cwd_for_deps_checkout)

    for dep_name in extra_deps:
        dep_config = name_to_path_map.get(dep_name)
        if not dep_config:
            wayne_print(f"Error: Configuration for dependency '{dep_name}' of '{tool_name}' not found. Cannot install.", "red")
            return
        dep_relative_path = dep_config.get('path')
        if not dep_relative_path:
            wayne_print(f"Error: 'path' not specified for dependency '{dep_name}'. Cannot resolve its source directory.", "red")
            return
        dep_abs_src_path = os.path.join(cpp_tools_repo_temp_dir, dep_relative_path)
        script_args.append(dep_abs_src_path)

    script_args.append(global_install_flag_str)

    wayne_print(f"Executing installation script for {tool_name}: {' '.join(script_args)}", "magenta")
    wayne_print(f"Script will be run from CWD: {cpp_tools_repo_temp_dir}", "magenta")

    # Prepare the environment for the script
    script_env = os.environ.copy() # Start with a copy of the current environment
    script_env['NON_INTERACTIVE_INSTALL'] = 'true'
    wayne_print(f"Setting NON_INTERACTIVE_INSTALL=true for script execution.", "blue")

    try:
        # Run the script, allow its output to go to stdout/stderr directly
        # Pass the modified environment using the 'env' parameter
        result = subprocess.run(
            script_args,
            cwd=cpp_tools_repo_temp_dir,
            text=True,
            check=False,
            env=script_env # Pass the custom environment
        )
        if result.returncode == 0:
            wayne_print(f"Successfully installed {tool_name}.", "green")
        else:
            wayne_print(f"Installation script for {tool_name} failed with return code {result.returncode}.", "red")
            wayne_print("Please check the script output above for details.", "red")
    except FileNotFoundError:
        wayne_print(f"Error: Installation script '{install_script_abs_path}' not found or not executable when trying to run.", "red")
    except Exception as e:
        wayne_print(f"An error occurred while running the installation script for {tool_name}: {e}", "red")


def fetch_tool(url: str, tool_name, target_dir='', build=False, clean=False, version=None, install_requested=False, global_install_flag_str="false"):
    print(f"Fetching tool: {tool_name}")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        sparse_clone(url, temp_dir)

        # If installation is requested, ensure the install_scripts directory is checked out early.
        # This needs to happen before any specific tool path might be set via sparse-checkout set,
        # or before submodule operations if those were to clear/reset sparse settings.
        if install_requested:
            wayne_print("Installation requested, ensuring 'install_scripts' directory is checked out from the repo root...", "cyan")
            # We are in temp_dir (repo root) here
            install_scripts_checkout_result = subprocess.run(
                ["git", "sparse-checkout", "add", "install_scripts"],
                capture_output=True, text=True, check=False # check=False to handle errors manually
            )
            if install_scripts_checkout_result.returncode == 0:
                if "Adding paths ..." in install_scripts_checkout_result.stdout or not install_scripts_checkout_result.stdout.strip(): # Check if it actually did something or was already set
                    wayne_print("'install_scripts' directory successfully added to sparse checkout or already present.", "green")
                else:
                    # Sometimes git outputs to stdout even on no-op for 'add' if already present, so minor check for actual change indication.
                    wayne_print(f"'install_scripts' checkout status: {install_scripts_checkout_result.stdout.strip()}", "green")
            else:
                wayne_print(f"Warning: Failed to add 'install_scripts' to sparse checkout. Installation might fail. Error: {install_scripts_checkout_result.stderr.strip()}", "yellow")

        name_to_path_map_yaml_file = 'name_to_path_map.yaml'
        assert name_to_path_map_yaml_file in os.listdir('.'), f'Failed to find {name_to_path_map_yaml_file} in {temp_dir}'
        current_name_to_path_map = read_yaml_config(name_to_path_map_yaml_file)
        tool_path_from_yaml = current_name_to_path_map[tool_name]['path']

        temp_tool_src_path = os.path.join(temp_dir, tool_path_from_yaml)

        if not target_dir:
            target_dir = os.path.join(cwd, tool_path_from_yaml)

        if not build and os.path.exists(target_dir):
            if input(f'{target_dir} already exists, still want to fetch? (Y/N)').lower() != 'y': return
        if os.path.exists('.gitmodules'):
            with open('.gitmodules', 'r') as f:
                tool_is_submodule = any(f'path = {tool_path_from_yaml}' in line for line in f)
        else:
            tool_is_submodule = False
        if tool_is_submodule:
            wayne_print(f"Updating submodule: {tool_path_from_yaml}...", "cyan")
            submodule_update_result = subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive", "--progress", tool_path_from_yaml],
                text=True
            )
            if submodule_update_result.returncode != 0:
                wayne_print(f"Failed to update submodule {tool_path_from_yaml}. Git command return code: {submodule_update_result.returncode}", "red")

            if version:
                wayne_print(f"Checking out version {version} for submodule {tool_name}...", 'yellow')
                try:
                    os.chdir(temp_tool_src_path)
                    checkout_result = subprocess.run(["git", "checkout", version], capture_output=True, text=True, check=True)
                    wayne_print(f"Successfully checked out version {version} for {tool_name}.", 'green')
                except subprocess.CalledProcessError as e:
                    wayne_print(f"Failed to checkout version {version} for {tool_name}: {e.stderr}", 'red')
                finally:
                    os.chdir(temp_dir)
        else:
            subprocess.run(["git", "sparse-checkout", "set", tool_path_from_yaml], check=True)
            if version:
                 wayne_print(f"Warning: --version specified ({version}) but {tool_name} is not a submodule and not built from main repo. Version ignored.", 'yellow')
        if build:
            if not current_name_to_path_map[tool_name]['buildable']:
                wayne_print(f'{tool_name} is not buildable, skip building', 'red')
            elif not os.path.exists(os.path.join(temp_tool_src_path, 'CMakeLists.txt')):
                wayne_print(f'{tool_name} does not have a CMakeLists.txt in {temp_tool_src_path}, skip building', 'red')
            else:
                build_command_dir = temp_tool_src_path
                command = f'cd "{build_command_dir}" && mkdir -p build && cd build && cmake .. && make -j12'
                os.system(command)
                if os.path.exists(target_dir): shutil.rmtree(target_dir)
                shutil.copytree(os.path.join(temp_tool_src_path, 'lib/'), target_dir, dirs_exist_ok=True)
        else:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            if clean:
                src_in_temp = os.path.join(temp_tool_src_path, 'src')
                include_in_temp = os.path.join(temp_tool_src_path, 'include')
                if os.path.exists(src_in_temp) and os.path.isdir(src_in_temp):
                    shutil.copytree(src_in_temp, target_dir, dirs_exist_ok=True)
                    if os.path.exists(include_in_temp) and os.path.isdir(include_in_temp):
                        shutil.copytree(include_in_temp, target_dir, dirs_exist_ok=True)
                    wayne_print(f"Clean copy: Copied 'src' and (if exists) 'include' from {temp_tool_src_path} to {target_dir}", "green")
                else:
                    wayne_print(f"Warning: 'src' directory not found in {temp_tool_src_path}. Performing a full copy instead of clean copy.", "yellow")
                    shutil.copytree(temp_tool_src_path, target_dir, dirs_exist_ok=True)
            else:
                shutil.copytree(temp_tool_src_path, target_dir, dirs_exist_ok=True)
        
        print(f"Tool {tool_name} source has been copied to {target_dir}")

        if install_requested:
            handle_installation(
                tool_name,
                current_name_to_path_map,
                target_dir,
                temp_dir,
                global_install_flag_str
            )
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
    parser.add_argument('-v', '--version', default=None, type=str, help='Specify a version/tag/branch to checkout (only for submodules)')
    parser.add_argument('-U', '--upgrade', action='store_true', help='Upgrade the tool')
    parser.add_argument('-f', '--force', action='store_true', help='Force action')
    parser.add_argument('-c', '--clean', action='store_true', help='Only fetch c++ sources')
    parser.add_argument('-l', '--list', action='store_true', help='List current supported tools')
    parser.add_argument('--get-url', action='store_true', help='get current URL of the tool')
    parser.add_argument('--set-url', type=str, default='', help='set current URL of the tool, e.g. "https://github.com/wangyendt/cpp_tools"')
    parser.add_argument('--reset-url', action='store_true', help='reset url to default: "https://github.com/wangyendt/cpp_tools"')
    parser.add_argument('-i', '--install', action='store_true', help='Install the tool after fetching (if installable)')
    parser.add_argument('--global-install-flag', type=str, default="false", choices=['true', 'false'], help='Flag for sudo make install for installation scripts (default: false)')

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

    tool_name = args.name if args.name is not None else args.name_pos
    target_path = args.target_path
    build = args.build
    clean = args.clean
    version = args.version
    install_requested = args.install
    global_install_flag_str = args.global_install_flag

    if tool_name is None:
        parser.error("the following arguments are required: name")

    fetch_tool(
        url,
        tool_name,
        target_dir=target_path,
        build=build,
        clean=clean,
        version=version,
        install_requested=install_requested,
        global_install_flag_str=global_install_flag_str
    )

    if args.upgrade:
        wayne_print("(not implemented yet)", "yellow")
    if args.force:
        wayne_print("(not implemented yet)", "yellow")


if __name__ == '__main__':
    main()
