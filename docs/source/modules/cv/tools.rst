OpenCV YAML 文件读写
======================

这部分功能位于 `pywayne.cv.tools` 模块中，提供了读取和写入 OpenCV `cv2.FileStorage` 支持的 YAML 格式文件的函数。

**主要功能**:

- `read_cv_yaml(file_path: str)`: 读取 OpenCV YAML 文件并返回一个字典。支持基本数据类型、NumPy 数组以及嵌套的字典和列表。
- `write_cv_yaml(file_path: str, data: dict)`: 将字典写入 OpenCV YAML 文件。支持基本数据类型、NumPy 数组，以及嵌套的字典和列表。

**示例**::

   >>> import numpy as np
   >>> from pywayne.cv.tools import read_cv_yaml, write_cv_yaml
   >>> 
   >>> # 准备要写入的数据
   >>> data_to_write = {
   ...     "matrix": np.eye(3),
   ...     "integer_value": 10,
   ...     "float_value": 3.14,
   ...     "string_value": "hello_cv",
   ...     "list_value": [1, 2, 3, {"nested_in_list": "item_in_list_dict"}],
   ...     "nested_dictionary": {
   ...         "sub_item_A": "text_val",
   ...         "sub_item_B": 789,
   ...         "sub_list_in_dict": [True, False, 0.5]
   ...     }
   ... }
   >>> 
   >>> # 写入 YAML 文件
   >>> output_file = "cv_config.yaml"
   >>> success = write_cv_yaml(output_file, data_to_write)
   >>> if success:
   ...     print(f"成功写入到 {output_file}")
   >>> 
   >>> # 从 YAML 文件读取
   >>> read_data = read_cv_yaml(output_file)
   >>> if read_data:
   ...     print(f"从 {output_file} 读取的数据:")
   ...     for k, v in read_data.items():
   ...         print(f"  {k}: {v}")

.. autofunction:: pywayne.cv.tools.read_cv_yaml
.. autofunction:: pywayne.cv.tools.write_cv_yaml 