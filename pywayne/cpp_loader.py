# author: wangye(Wayne)
# license: Apache Licence

import importlib
import os
import subprocess
import sys
from pathlib import Path


def _local_extension_exists(lib_path: Path, module_name: str) -> bool:
    patterns = (
        f"{module_name}*.so",
        f"{module_name}*.pyd",
    )
    return any(path.is_file() for pattern in patterns for path in lib_path.glob(pattern))


def _is_missing_requested_module(error: ImportError, module_name: str) -> bool:
    missing_name = getattr(error, "name", None)
    if missing_name == module_name:
        return True
    message = str(error)
    return "No module named" in message and module_name in message


def import_cpp_module(module_name: str, tool_name: str, lib_path: os.PathLike):
    """
    Import a pybind module from lib_path, building it with gettool once if needed.

    ImportError can also mean an existing extension failed to load because one of
    its shared-library dependencies is missing or ABI-incompatible. In that case
    we still rebuild once, but keep the original error in the final exception.
    """
    lib_path = Path(lib_path)
    lib_path_str = str(lib_path)
    if lib_path_str not in sys.path:
        sys.path.insert(0, lib_path_str)

    try:
        return importlib.import_module(module_name)
    except ImportError as first_error:
        local_extension_exists = _local_extension_exists(lib_path, module_name)
        missing_requested_module = _is_missing_requested_module(first_error, module_name)

        if local_extension_exists and not missing_requested_module:
            print(
                f"Found local '{module_name}' extension, but it failed to load. "
                "This usually means a runtime library is missing or ABI-incompatible. "
                "Rebuilding once with gettool..."
            )
        else:
            print(f"'{module_name}' module not found. Attempting to build '{tool_name}' with gettool...")

        os.makedirs(lib_path, exist_ok=True)
        try:
            subprocess.run(["gettool", tool_name, "-b", "-t", lib_path_str], check=True)
        except FileNotFoundError as error:
            raise RuntimeError("gettool command not found. Please ensure it is installed and in PATH.") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"gettool failed while building '{tool_name}' for module '{module_name}' "
                f"with return code {error.returncode}."
            ) from error

        importlib.invalidate_caches()
        sys.modules.pop(module_name, None)
        try:
            return importlib.import_module(module_name)
        except ImportError as second_error:
            raise ImportError(
                f"'{module_name}' was built or found, but importing it still failed. "
                "Check bundled runtime libraries and ABI compatibility. "
                f"Original import error: {first_error}; retry import error: {second_error}"
            ) from second_error
