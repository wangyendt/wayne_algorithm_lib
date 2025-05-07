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
    final_target_dir = target_dir # User-specified or default, resolved before temp dir ops
    tool_path_in_final_repo = "" # Will be set if target_dir is a repo and tool_name is part of it

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        # Clone the main cpp_tools repo sparsely to get .gitmodules and name_to_path_map.yaml
        sparse_clone(url, "cpp_tools_repo") # Clone into a subdirectory within temp_dir
        cpp_tools_repo_root = os.path.join(temp_dir, "cpp_tools_repo")
        os.chdir(cpp_tools_repo_root) # CD into the cloned repo root

        if install_requested:
            wayne_print("Installation requested, ensuring 'install_scripts' directory is checked out from the repo root...", "cyan")
            install_scripts_checkout_result = subprocess.run(
                ["git", "sparse-checkout", "add", "install_scripts"],
                capture_output=True, text=True, check=False
            )
            if install_scripts_checkout_result.returncode == 0:
                wayne_print("'install_scripts' directory successfully added/ensured.", "green")
            else:
                wayne_print(f"Warning: Failed to add 'install_scripts' to sparse checkout. Error: {install_scripts_checkout_result.stderr.strip()}", "yellow")

        name_to_path_map_yaml_file = 'name_to_path_map.yaml'
        if not os.path.exists(name_to_path_map_yaml_file):
            wayne_print(f"Error: {name_to_path_map_yaml_file} not found in the root of {cpp_tools_repo_root}. Cannot proceed.", "red")
            os.chdir(cwd) # Go back to original CWD before exiting temp_dir context
            return
        current_name_to_path_map = read_yaml_config(name_to_path_map_yaml_file)
        
        if tool_name not in current_name_to_path_map:
            wayne_print(f"Error: Tool '{tool_name}' not found in {name_to_path_map_yaml_file}.", "red")
            os.chdir(cwd)
            return
            
        tool_config = current_name_to_path_map[tool_name]
        tool_path_from_yaml = tool_config['path'] # e.g., third_party/eigen

        if not final_target_dir: # If user didn't specify a target_dir
            final_target_dir = os.path.join(cwd, tool_path_from_yaml) # Default target relative to original CWD
        
        wayne_print(f"Final target directory for {tool_name}: {final_target_dir}", "cyan")

        # Determine if the tool is a submodule
        tool_is_submodule = False
        submodule_url = ""
        gitmodules_path = os.path.join(cpp_tools_repo_root, ".gitmodules")
        if os.path.exists(gitmodules_path):
            try:
                # Using git config to get submodule URL is more reliable
                submodule_url_result = subprocess.run(
                    ["git", "config", "--file", ".gitmodules", f"submodule.{tool_path_from_yaml}.url"],
                    capture_output=True, text=True, check=False, # check=False, we check returncode
                    cwd=cpp_tools_repo_root # ensure command runs in cpp_tools_repo_root where .gitmodules is
                )
                if submodule_url_result.returncode == 0 and submodule_url_result.stdout.strip():
                    submodule_url = submodule_url_result.stdout.strip()
                    tool_is_submodule = True
                    wayne_print(f"Tool '{tool_name}' is a submodule. URL: {submodule_url}", "blue")
                else:
                    wayne_print(f"Tool '{tool_name}' ({tool_path_from_yaml}) not found as a submodule in .gitmodules or URL is empty.", "magenta")
            except Exception as e:
                wayne_print(f"Could not read submodule URL for {tool_path_from_yaml} from .gitmodules: {e}", "yellow")
        
        # --- Logic for fetching the tool --- 
        if tool_is_submodule:
            wayne_print(f"Fetching '{tool_name}' as a full repository from {submodule_url} into {final_target_dir}", "cyan")
            if os.path.exists(final_target_dir):
                # Simplified: remove if exists. Add prompting/force later if needed.
                wayne_print(f"Target directory {final_target_dir} exists. Removing it first.", "yellow")
                shutil.rmtree(final_target_dir)
            
            clone_result = subprocess.run(["git", "clone", "--progress", submodule_url, final_target_dir], text=True)
            if clone_result.returncode != 0:
                wayne_print(f"Failed to clone submodule {tool_name} from {submodule_url} into {final_target_dir}.", "red")
                os.chdir(cwd)
                return
            wayne_print(f"Successfully cloned {tool_name} into {final_target_dir}", "green")

            if version:
                wayne_print(f"Checking out version {version} for {tool_name} in {final_target_dir}...", 'yellow')
                checkout_cwd = os.getcwd() # This is cpp_tools_repo_root
                try:
                    os.chdir(final_target_dir) # CD into the newly cloned repo
                    checkout_result_sub = subprocess.run(["git", "checkout", version], capture_output=True, text=True, check=True)
                    wayne_print(f"Successfully checked out version {version} for {tool_name}.", 'green')
                except subprocess.CalledProcessError as e:
                    wayne_print(f"Failed to checkout version {version} for {tool_name}: {e.stderr.strip()}", 'red')
                finally:
                    os.chdir(checkout_cwd) # CD back to cpp_tools_repo_root
            
            # For submodules cloned as full repos, the build/clean logic (if any) would apply to final_target_dir
            # and install_script also gets final_target_dir as main_tool_source_dir.
            # The current build/copytree part below is for non-submodules or old way.
        
        else: # Tool is not a submodule, or we are fetching it as part of cpp_tools repo (sparse-checkout)
            wayne_print(f"Fetching '{tool_name}' using sparse-checkout from cpp_tools repo.", "cyan")
            # Ensure the specific tool path is checked out in cpp_tools_repo for sparse checkout
            # This command is run in cpp_tools_repo_root
            sparse_set_result = subprocess.run(["git", "sparse-checkout", "set", tool_path_from_yaml], capture_output=True, text=True)
            if sparse_set_result.returncode != 0:
                wayne_print(f"Failed to set sparse-checkout for {tool_path_from_yaml}. Error: {sparse_set_result.stderr.strip()}", "red")
                os.chdir(cwd)
                return
            
            temp_tool_src_path = os.path.join(cpp_tools_repo_root, tool_path_from_yaml)
            if not os.path.exists(temp_tool_src_path):
                 wayne_print(f"Error: Source path {temp_tool_src_path} for '{tool_name}' does not exist after sparse-checkout set. This should not happen.", "red")
                 os.chdir(cwd)
                 return

            # Build or copytree logic for tools fetched via sparse-checkout
            if build:
                if not tool_config.get('buildable', False):
                    wayne_print(f'{tool_name} is not buildable, skip building', 'red')
                    # If not buildable but build flag was set, should we copy source like build=false?
                    # For now, do nothing more here, effectively it won't produce a different target_dir than if build was false and no copy happened.
                    # Consider copying source as a fallback if buildable is false but build is true.
                elif not os.path.exists(os.path.join(temp_tool_src_path, 'CMakeLists.txt')):
                    wayne_print(f'{tool_name} does not have a CMakeLists.txt in {temp_tool_src_path}, skip building', 'red')
                else:
                    # Restored and adapted build logic for non-submodule (sparse-checkout) tools
                    wayne_print(f"Building {tool_name} in {temp_tool_src_path}...", "blue")
                    original_cwd_for_build = os.getcwd() # Should be cpp_tools_repo_root
                    
                    build_dir_in_tool_src = os.path.join(temp_tool_src_path, 'build')
                    if os.path.exists(build_dir_in_tool_src):
                        shutil.rmtree(build_dir_in_tool_src)
                    os.makedirs(build_dir_in_tool_src)
                    
                    try:
                        os.chdir(build_dir_in_tool_src) # CD into .../cv/apriltag_detection/build
                        
                        wayne_print(f"Running CMake in {os.getcwd()} (source: ..)", "magenta")
                        # Basic CMake command, remove capture_output=True to print log directly
                        cmake_process = subprocess.run(["cmake", ".."], text=True, check=False)
                        if cmake_process.returncode != 0:
                            wayne_print(f"CMake failed for {tool_name} with return code {cmake_process.returncode}. Check output above.", "red")
                        else:
                            wayne_print(f"CMake successful for {tool_name}.", "green")
                            wayne_print(f"Running make in {os.getcwd()}", "magenta")
                            num_cores = os.cpu_count() or 1
                            make_cmd = ["make", f"-j{num_cores}"]
                            # Basic make command, remove capture_output=True to print log directly
                            make_process = subprocess.run(make_cmd, text=True, check=False)
                            if make_process.returncode != 0:
                                wayne_print(f"Make failed for {tool_name} with return code {make_process.returncode}. Check output above.", "red")
                            else:
                                wayne_print(f"Make successful for {tool_name}.", "green")
                                compiled_lib_path_in_source_tree = os.path.join(temp_tool_src_path, "lib") 

                                if os.path.exists(compiled_lib_path_in_source_tree) and os.path.isdir(compiled_lib_path_in_source_tree):
                                    if os.path.exists(final_target_dir):
                                        shutil.rmtree(final_target_dir)
                                    # Copy the content of temp_tool_src_path/lib to final_target_dir
                                    # This assumes final_target_dir is meant to BE the lib directory content.
                                    # If final_target_dir is foo/bar and lib is foo/bar/lib, adjust copytree dst.
                                    # For this specific case, apriltag_detection outputs to ${CMAKE_CURRENT_SOURCE_DIR}/lib,
                                    # and if gettool copies this to final_target_dir, final_target_dir becomes the effective 'lib' output.
                                    shutil.copytree(compiled_lib_path_in_source_tree, final_target_dir, dirs_exist_ok=True)
                                    wayne_print(f"Copied compiled library from {compiled_lib_path_in_source_tree} to {final_target_dir}", "green")
                                else:
                                    # Fallback if even source_dir/lib is not found.
                                    wayne_print(f"Build successful, but expected 'lib' directory not found in {compiled_lib_path_in_source_tree} (source tree). \nAttempting to copy entire source tree {temp_tool_src_path} as fallback, assuming it might be header-only or configured in-place.", "yellow")
                                    if os.path.exists(final_target_dir):
                                        shutil.rmtree(final_target_dir)
                                    shutil.copytree(temp_tool_src_path, final_target_dir, dirs_exist_ok=True) 
                                    wayne_print(f"Copied entire source tree from {temp_tool_src_path} to {final_target_dir} as fallback.", "yellow")
                    except Exception as e:
                        wayne_print(f"An error occurred during the build process for {tool_name}: {e}", "red")
                    finally:
                        os.chdir(original_cwd_for_build) # CD back to cpp_tools_repo_root
            else: # Not building (build=False for sparse-checkout tool)
                if os.path.exists(final_target_dir):
                    shutil.rmtree(final_target_dir)
                if clean:
                    # ... (clean copy logic from temp_tool_src_path to final_target_dir) ...
                    src_in_temp = os.path.join(temp_tool_src_path, 'src')
                    include_in_temp = os.path.join(temp_tool_src_path, 'include')
                    if os.path.exists(src_in_temp) and os.path.isdir(src_in_temp):
                        shutil.copytree(src_in_temp, final_target_dir, dirs_exist_ok=True) # Copies to root of final_target_dir
                        if os.path.exists(include_in_temp) and os.path.isdir(include_in_temp):
                             # This would overwrite/merge into final_target_dir if src also copied files there.
                             # Consider copying to final_target_dir/include if that's the desired structure.
                            shutil.copytree(include_in_temp, final_target_dir, dirs_exist_ok=True) 
                        wayne_print(f"Clean copy: Copied 'src' and (if exists) 'include' from {temp_tool_src_path} to {final_target_dir}", "green")
                    else:
                        wayne_print(f"Warning: 'src' directory not found in {temp_tool_src_path}. Performing a full copy instead of clean copy.", "yellow")
                        shutil.copytree(temp_tool_src_path, final_target_dir, dirs_exist_ok=True)
                else: # Not clean, copy everything from temp_tool_src_path
                    shutil.copytree(temp_tool_src_path, final_target_dir, dirs_exist_ok=True)
            wayne_print(f"Tool {tool_name} (sparse-checkout) source has been copied to {final_target_dir}", "green")

        # After successful fetch (either direct clone or sparse-checkout + copy)
        if install_requested:
            handle_installation(
                tool_name,
                current_name_to_path_map,
                final_target_dir, # This is where the main tool's source now resides
                cpp_tools_repo_root, # This is still the root of the cpp_tools clone for dep paths and install scripts
                global_install_flag_str
            )
        
        os.chdir(cwd) # Go back to original CWD before temp_dir is removed
    # --- End of fetch_tool logic, temp_dir is now gone ---
    # os.chdir(cwd) # This was inside, moved it to be before temp_dir cleanup if an error occurs early


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
