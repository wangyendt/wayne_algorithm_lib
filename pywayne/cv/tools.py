import cv2
import numpy as np
# import logging # Removed logging import
from typing import Optional, Union, Dict, Any, List
from pywayne.tools import wayne_print # Added wayne_print import

# logger = logging.getLogger(__name__) # Removed logger initialization

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
    try:
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_WRITE)
        if not fs.isOpened():
            # logger.error(f"Failed to open OpenCV YAML file for writing: {file_path}")
            wayne_print(f"Failed to open OpenCV YAML file for writing: {file_path}", color='red')
            return False

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                fs.write(key, value)
            elif isinstance(value, (int, float, str)):
                fs.write(key, value)
            elif isinstance(value, list): # Basic list writing (as sequence)
                fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for item in value:
                     # Note: Writing complex items within list needs specific handling
                     if isinstance(item, (int, float, str)):
                          fs.write("", item) # OpenCV uses empty string for unnamed sequence items
                     else:
                         # logger.warning(f"Skipping unsupported type {type(item)} in list for key '{key}'")
                         wayne_print(f"Skipping unsupported type {type(item)} in list for key '{key}'", color='yellow')
                fs.endWriteStruct()
            elif isinstance(value, dict): # Basic dictionary writing (as map)
                fs.startWriteStruct(key, cv2.FileNode_MAP)
                for sub_key, sub_value in value.items():
                     # Note: Writing complex items within dict needs specific handling
                     if isinstance(sub_value, (int, float, str, np.ndarray)):
                         fs.write(sub_key, sub_value)
                     else:
                          # logger.warning(f"Skipping unsupported type {type(sub_value)} in dict for key '{key}.{sub_key}'")
                          wayne_print(f"Skipping unsupported type {type(sub_value)} in dict for key '{key}.{sub_key}'", color='yellow')
                fs.endWriteStruct()
            elif value is None:
                 # OpenCV FileStorage doesn't have a direct way to write None.
                 # We can skip or write an empty string/map based on desired behavior.
                 # logger.warning(f"Skipping None value for key: {key}")
                 wayne_print(f"Skipping None value for key: {key}", color='yellow')
                 # fs.write(key, "") # Alternative: write empty string
            else:
                # logger.warning(f"Skipping unsupported type {type(value)} for key: {key}")
                wayne_print(f"Skipping unsupported type {type(value)} for key: {key}", color='yellow')

        fs.release()
        return True
    except cv2.error as e:
        # logger.error(f"Error writing OpenCV YAML file {file_path}: {e}")
        wayne_print(f"Error writing OpenCV YAML file {file_path}: {e}", color='red')
        return False
    except Exception as e:
        # logger.error(f"An unexpected error occurred while writing {file_path}: {e}")
        wayne_print(f"An unexpected error occurred while writing {file_path}: {e}", color='red')
        # Ensure fs is released if it was opened
        if 'fs' in locals() and fs.isOpened():
             fs.release()
        return False
