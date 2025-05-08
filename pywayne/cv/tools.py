import cv2
import numpy as np
from typing import Optional, Union, Dict, Any, List
from pywayne.tools import wayne_print # Added wayne_print import


def _parse_cv_node(node: cv2.FileNode) -> Any:
    """Recursively parses a cv2.FileNode."""
    if node is None or node.empty():
        return None
    elif node.isMap():
        # Check if it looks like an OpenCV matrix first
        if hasattr(node, 'keys') and callable(node.keys) and \
           "rows" in node.keys() and "cols" in node.keys() and "dt" in node.keys():
            mat_val = node.mat()
            if mat_val is not None:
                return mat_val
            else:
                wayne_print(f"Node looks like a matrix but failed to read .mat()", color='yellow')
                return None # Failed matrix read
        else:
            # Handle general maps (dictionaries) recursively
            map_data = {}
            keys = []
            # Check if keys method exists and is callable
            node_keys_attr = getattr(node, 'keys', None)
            if node_keys_attr and callable(node_keys_attr):
                try:
                    keys = node_keys_attr()
                except Exception as e:
                     wayne_print(f"Error calling .keys() on map node: {e}", color='red')
                     return None # Cannot process map if keys() fails
            else:
                 wayne_print(f"Cannot retrieve keys from map node. Node object might not support .keys()", color='red')
                 return None # Cannot process map without keys

            if not keys:
                 wayne_print(f"Map node has no keys.", color='yellow')
                 return {} # Empty map

            for key in keys:
                sub_node = node.getNode(key) # Or potentially node[key]
                map_data[key] = _parse_cv_node(sub_node)
            return map_data
    elif node.isSeq():
         seq_data = []
         for i in range(node.size()):
             item_node = node.at(i)
             seq_data.append(_parse_cv_node(item_node))
         return seq_data
    elif node.isInt():
        # Use real() for both int and float, then cast
        return int(node.real())
    elif node.isReal():
        return node.real()
    elif node.isString():
        return node.string()
    elif node.isNone(): # Explicit check for None/Empty type in YAML
         return None
    else:
        # Fallback attempt for types like Flow
        # logger.warning(f"Encountered an unsupported node type. Trying to read as matrix.")
        wayne_print(f"Encountered an unsupported node type. Trying to read as matrix.", color='yellow')
        mat_val = node.mat()
        if mat_val is not None:
            return mat_val
        else:
            # logger.warning(f"Fallback as matrix failed. Returning None for this node.")
            wayne_print(f"Fallback as matrix failed. Returning None for this node.", color='yellow')
            return None


def read_cv_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """
    读取 OpenCV YAML 文件，支持嵌套结构和基本类型。

    Args:
        file_path: YAML 文件路径。

    Returns:
        包含文件内容的字典，如果读取失败则返回 None。
    """
    data = {}
    fs = None # Initialize fs to None
    try:
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            wayne_print(f"Failed to open OpenCV YAML file for reading: {file_path}", color='red')
            return None

        data = {}
        root_node = fs.root()
        if not root_node.isMap():
            wayne_print(f"Root node in {file_path} is not a Map (dictionary). Cannot process.", color='red')
            # fs.release() # release in finally
            return None

        keys = []
        # Check if keys method exists and is callable for the root node
        root_keys_attr = getattr(root_node, 'keys', None)
        if root_keys_attr and callable(root_keys_attr):
            try:
                keys = root_keys_attr()
            except Exception as e:
                wayne_print(f"Error calling .keys() on root node in {file_path}: {e}", color='red')
                # fs.release() # release in finally
                return None
        else:
            wayne_print(f"Cannot retrieve keys from the root FileNode in {file_path}. Root node object might not support .keys() in this version.", color='red')
            # fs.release() # release in finally
            return None

        if not keys:
             wayne_print(f"No keys found under the root node in {file_path}", color='yellow')
             # fs.release() # release in finally
             return {}

        for key in keys:
            node = root_node.getNode(key) # Or potentially root_node[key]
            data[key] = _parse_cv_node(node) # Use the recursive parser

        # fs.release() # release in finally
        return data

    except cv2.error as e:
        wayne_print(f"cv2 error reading OpenCV YAML file {file_path}: {e}", color='red')
        return None
    except Exception as e:
         wayne_print(f"An unexpected error occurred while processing {file_path}: {e}", color='red')
         # traceback.print_exc() # Optional: print full traceback for debugging
         return None
    finally:
         # Ensure fs is released if it was opened, regardless of errors
         if fs is not None and fs.isOpened():
             try:
                 fs.release()
             except Exception as e:
                 wayne_print(f"Error releasing FileStorage for {file_path}: {e}", color='red')

def write_cv_yaml(file_path: str, data: Dict[str, Any]) -> bool:
    """
    将字典写入 OpenCV YAML 文件。

    Args:
        file_path: 要写入的 YAML 文件路径。
        data: 要写入的字典数据。 支持类型: np.ndarray, int, float, str, list (基本类型), dict (基本类型).

    Returns:
        如果写入成功则返回 True，否则返回 False。
    """
    fs = None # Initialize fs to None
    try:
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_WRITE)
        if not fs.isOpened():
            wayne_print(f"Failed to open OpenCV YAML file for writing: {file_path}", color='red')
            return False

        for key, value in data.items():
            _write_cv_node(fs, key, value)

        # fs.release() # release in finally
        return True
    except cv2.error as e:
        wayne_print(f"Error writing OpenCV YAML file {file_path}: {e}", color='red')
        return False
    except Exception as e:
        wayne_print(f"An unexpected error occurred while writing {file_path}: {e}", color='red')
        # traceback.print_exc() # Optional: print full traceback for debugging
        return False
    finally:
        if fs is not None and fs.isOpened():
            try:
                fs.release()
            except Exception as e:
                 wayne_print(f"Error releasing FileStorage for {file_path}: {e}", color='red')


def _write_cv_node(fs: cv2.FileStorage, key: str, value: Any):
    """
    Helper function to write a node (key-value pair) to the OpenCV FileStorage.
    Handles different data types including nested dictionaries and lists.
    """
    if isinstance(value, np.ndarray):
        fs.write(key, value)
    elif isinstance(value, (int, float, str)):
        fs.write(key, value)
    elif isinstance(value, list):
        # For lists, use startWriteStruct with FileNode_SEQ
        # If key is provided (e.g. for top-level list or list in dict), use it.
        # If this is an item within a list, key should be empty string by OpenCV's convention
        fs.startWriteStruct(key, cv2.FileNode_SEQ)
        for item in value:
            # For items in a sequence, the "key" is empty.
            # We recursively call _write_cv_node with an empty key for the item itself.
            # This might need adjustment based on how OpenCV expects unnamed items to be written.
            # OpenCV uses an empty string for unnamed sequence items.
            _write_cv_node(fs, "", item) # Pass empty key for list items
        fs.endWriteStruct()
    elif isinstance(value, dict):
        # For dicts, use startWriteStruct with FileNode_MAP
        fs.startWriteStruct(key, cv2.FileNode_MAP)
        for sub_key, sub_value in value.items():
            _write_cv_node(fs, sub_key, sub_value) # Recursive call for sub-dictionary
        fs.endWriteStruct()
    elif value is None:
        wayne_print(f"Skipping None value for key: {key}", color='yellow')
        # OpenCV FileStorage doesn't have a direct way to write None.
        # We can skip or write an empty string/map. For now, skipping.
        # fs.write(key, "") # Alternative: write empty string or an empty map
    else:
        wayne_print(f"Skipping unsupported type {type(value)} for key: {key}", color='yellow')


if __name__ == '__main__':
    import os
    # 1. Create a sample dictionary with various data types
    sample_data_to_write = {
        "camera_name": "test_camera_01",
        "image_width": 1920,
        "image_height": 1080,
        "fps": 60.5,
        "is_color": True,
        "calibration_matrix": np.array([[1000.0, 0.0, 960.0],
                                        [0.0, 1000.0, 540.0],
                                        [0.0, 0.0, 1.0]], dtype=np.float64),
        "distortion_coeffs": np.array([-0.1, 0.01, 0.001, 0.0005, 0.0]),
        "simple_list": [1, 2, 3, 4, 5],
        "list_of_strings": ["apple", "banana", "cherry"],
        "list_with_mixed_types": [10, "mixed", 20.5, False],
        "nested_info": {
            "sensor_type": "CMOS",
            "serial_number": "SN12345XYZ",
            "calibration_date": "2024-07-30",
            "sub_parameters": {
                "param_A": 123,
                "param_B": "value_b",
                "param_C_list": [0.1, 0.2, 0.3]
            }
        },
        "empty_nested_dict": {},
        "list_of_dictionaries": [
            {"id": 1, "name": "item1", "value": 100},
            {"id": 2, "name": "item2", "value": 200.5},
            {"id": 3, "name": "item3", "tags": ["tagA", "tagB"]}
        ],
        "none_value_test": None, # This will be skipped during writing as per current _write_cv_node
        "transformation_matrix": np.random.rand(4, 4)
    }

    yaml_file_path = "example_cv.yaml"

    # 2. Write the dictionary to a YAML file
    wayne_print(f"\nAttempting to write data to '{yaml_file_path}'...", color='cyan')
    write_success = write_cv_yaml(yaml_file_path, sample_data_to_write)

    if write_success:
        wayne_print(f"Successfully wrote data to '{yaml_file_path}'", color='green')

        # 3. Read the data back from the YAML file
        wayne_print(f"\nAttempting to read data from '{yaml_file_path}'...", color='cyan')
        read_data = read_cv_yaml(yaml_file_path)

        if read_data is not None:
            wayne_print("Successfully read data:", color='green')
            # You can add more detailed checks here if needed
            # For example, comparing read_data with sample_data_to_write
            # (Note: direct comparison might fail for np.arrays without element-wise check)
            for key, value in read_data.items():
                if isinstance(value, np.ndarray):
                    wayne_print(f"  {key} (np.ndarray shape: {value.shape}, dtype: {value.dtype}):\n{value}", color='blue')
                elif isinstance(value, dict):
                    wayne_print(f"  {key} (dict):", color='blue')
                    for sub_key, sub_value in value.items(): # type: ignore
                        wayne_print(f"    {sub_key}: {sub_value}", color='magenta')
                elif isinstance(value, list):
                    wayne_print(f"  {key} (list, {len(value)} items): {value}", color='blue')
                else:
                    wayne_print(f"  {key}: {value} (type: {type(value).__name__})", color='blue')
        else:
            wayne_print(f"Failed to read data from '{yaml_file_path}'", color='red')
    else:
        wayne_print(f"Failed to write data to '{yaml_file_path}'", color='red')

    # 4. Clean up: Remove the created YAML file
    if os.path.exists(yaml_file_path):
        try:
            os.remove(yaml_file_path)
            wayne_print(f"\nSuccessfully removed temporary file '{yaml_file_path}'", color='green')
        except OSError as e:
            wayne_print(f"\nError removing temporary file '{yaml_file_path}': {e}", color='red')
    else:
        wayne_print(f"\nTemporary file '{yaml_file_path}' was not found for cleanup.", color='yellow')

