import cv2
import numpy as np
# import logging # Removed logging import
from typing import Optional, Union, Dict, Any
from pywayne.tools import wayne_print # Added wayne_print import

# logger = logging.getLogger(__name__) # Removed logger initialization

def read_cv_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """
    读取 OpenCV YAML 文件。

    Args:
        file_path: YAML 文件路径。

    Returns:
        包含文件内容的字典，如果读取失败则返回 None。
    """
    data = {}
    try:
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            # logger.error(f"Failed to open OpenCV YAML file for reading: {file_path}")
            wayne_print(f"Failed to open OpenCV YAML file for reading: {file_path}", color='red')
            return None

        data = {}
        try:
            root_node = fs.root()
            if not root_node.isMap():
                # logger.error(f"Root node in {file_path} is not a Map (dictionary). Cannot process.")
                wayne_print(f"Root node in {file_path} is not a Map (dictionary). Cannot process.", color='red')
                fs.release()
                return None

            # Attempt to get keys from the root node map
            keys = []
            if hasattr(root_node, 'keys') and callable(root_node.keys):
                keys = root_node.keys()
            else:
                # If root_node.keys() doesn't work, this version might require a different approach
                # We could try iterating root_node directly if it's supported, but it's less common
                # logger.error(f"Cannot retrieve keys from the root FileNode in {file_path}. Root node object might not support .keys() in this version.")
                wayne_print(f"Cannot retrieve keys from the root FileNode in {file_path}. Root node object might not support .keys() in this version.", color='red')
                fs.release()
                return None

            if not keys:
                 # logger.warning(f"No keys found under the root node in {file_path}")
                 wayne_print(f"No keys found under the root node in {file_path}", color='yellow')
                 fs.release()
                 return {}

            for key in keys:
                node = root_node.getNode(key) # Or potentially root_node[key]
                if node is None or node.empty():
                    # logger.warning(f"Could not get valid node for key: {key}")
                    wayne_print(f"Could not get valid node for key: {key}", color='yellow')
                    continue

                # Process the node (same logic as before)
                if node.isMap():
                    # Check if it looks like an OpenCV matrix
                    if hasattr(node, 'keys') and callable(node.keys) and \
                       "rows" in node.keys() and "cols" in node.keys() and "dt" in node.keys():
                        mat_val = node.mat()
                        if mat_val is not None:
                            data[key] = mat_val
                        else:
                            # logger.warning(f"Node for key '{key}' looks like a matrix but failed to read .mat()")
                            wayne_print(f"Node for key '{key}' looks like a matrix but failed to read .mat()", color='yellow')
                            data[key] = None # Failed matrix read
                    else:
                        # Handle non-matrix maps (potentially nested)
                        # For now, log a warning as deep parsing isn't implemented
                        # logger.warning(f"Skipping complex/non-matrix map node for key: {key}. Manual parsing needed.")
                        wayne_print(f"Skipping complex/non-matrix map node for key: {key}. Manual parsing needed.", color='yellow')
                        data[key] = None # Placeholder for complex map
                elif node.isSeq():
                     seq_data = []
                     for i in range(node.size()):
                         item_node = node.at(i)
                         if item_node.isInt():
                             seq_data.append(int(item_node.real()))
                         elif item_node.isReal():
                             seq_data.append(item_node.real())
                         elif item_node.isString():
                             seq_data.append(item_node.string())
                         elif item_node.isMap() or item_node.isSeq():
                              # logger.warning(f"Skipping complex item in sequence for key: {key}")
                              wayne_print(f"Skipping complex item in sequence for key: {key}", color='yellow')
                         # Add more types as needed
                     data[key] = seq_data
                elif node.isInt():
                    data[key] = int(node.real())
                elif node.isReal():
                    data[key] = node.real()
                elif node.isString():
                    data[key] = node.string()
                elif node.isNone():
                     data[key] = None
                else:
                    mat_val = node.mat() # Try reading as matrix as fallback
                    if mat_val is not None:
                        data[key] = mat_val
                    else:
                         # logger.warning(f"Unsupported node type for key: {key}")
                         wayne_print(f"Unsupported node type for key: {key}", color='yellow')
                         data[key] = None

        except AttributeError as e:
             # logger.error(f"Error iterating through nodes (likely using .next()): {e}. This OpenCV version might have different iteration methods.")
             wayne_print(f"Error iterating through nodes (likely using .next()): {e}. This OpenCV version might have different iteration methods.", color='red')
             fs.release()
             return None
        except Exception as e:
             # logger.error(f"Error processing nodes in {file_path}: {e}")
             wayne_print(f"Error processing nodes in {file_path}: {e}", color='red')
             fs.release()
             return None

        fs.release()
        return data
    except cv2.error as e:
        # logger.error(f"Error reading OpenCV YAML file {file_path}: {e}")
        wayne_print(f"Error reading OpenCV YAML file {file_path}: {e}", color='red')
        return None

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
