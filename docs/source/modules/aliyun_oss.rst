阿里云 OSS 文件管理 (aliyun_oss)
================================

本模块封装了与阿里云 OSS 服务交互的核心功能。通过 OssManager 类，用户可以轻松实现文件上传、下载、删除以及目录操作等功能，适用于云存储管理和大规模文件处理场景。

OssManager 类详细说明
------------------------

.. py:class:: OssManager(endpoint: str, bucket_name: str, api_key: Optional[str]=None, api_secret: Optional[str]=None, verbose: bool=True)

   OssManager 类用于管理与阿里云 OSS 的交互，支持文件的上传、下载、删除，以及目录内容的列举与读取。

   **主要功能**：
   
   - 根据指定的 OSS endpoint 和 bucket_name 初始化连接。
   - 上传本地文件、文本或图像到 OSS。
   - 下载单个文件或具有特定前缀的多个文件。
   - 列出所有文件键或根据前缀过滤后的文件键。
   - 删除指定文件或根据前缀删除多个文件。
   - 上传和下载整个目录。
   - 读取 OSS 上存储文件的内容。
   - 检查文件是否存在，获取文件元数据，复制和移动文件等高级操作。

   **方法**:

   - __init__(self, endpoint: str, bucket_name: str, api_key: Optional[str]=None, api_secret: Optional[str]=None, verbose: bool=True)
     
     构造函数，用于初始化 OssManager 实例。
     
     **参数**:
     
     - **endpoint**: OSS 服务的访问地址。
     - **bucket_name**: 桶名称，用于指定存储空间。
     - **api_key**: 用户的 API Key，可选参数。
     - **api_secret**: 用户的 API Secret，可选参数。
     - **verbose**: 是否启用详细输出，默认为 True。

   - _print_info(self, message: str)
     
     内部方法，用于打印提示信息，帮助调试或记录日志。
   
   - _print_warning(self, message: str)
     
     内部方法，用于打印警告信息。
   
   - _check_write_permission(self) -> bool
     
     检查当前 OSS 账号是否具备写权限，返回布尔值。

   - download_file(self, key: str, root_dir: Optional[str]=None, use_basename: bool = False) -> bool
     
     下载 OSS 中指定 key 的文件，并存储到 root_dir 目录中（如果指定）。
     
     **参数**:
     
     - **key**: OSS 中的键值
     - **root_dir**: 下载文件的根目录，默认为当前目录
     - **use_basename**: 是否只使用 key 的文件名部分构建本地路径，默认为 False。
       如果为 True，则 `a/b/c.txt` 下载到 `root_dir/c.txt`；
       如果为 False，则下载到 `root_dir/a/b/c.txt`。
     
     **返回**:
     
     - 布尔值，指示是否成功下载。

   - download_files_with_prefix(self, prefix: str, root_dir: Optional[str]=None, use_basename: bool = False) -> bool
     
     下载 OSS 中所有以指定前缀开头的文件，存储到指定目录中。
     
     **参数**:
     
     - **prefix**: 键值前缀
     - **root_dir**: 下载文件的根目录，默认为当前目录
     - **use_basename**: 是否只使用 key 的文件名部分构建本地路径，与 download_file 相同

   - list_all_keys(self, sort: bool=True) -> List[str]
     
     列出 OSS 桶中所有的文件键，支持按字母排序。
   
   - list_keys_with_prefix(self, prefix: str, sort: bool=True) -> List[str]
     
     根据前缀过滤并列出 OSS 桶中匹配的文件键。

   - upload_file(self, key: str, file_path: str) -> bool
     
     将本地文件上传到 OSS，并指定存储的 key。
     
     **返回**:
     
     - 布尔值，指示上传操作是否成功。

   - upload_text(self, key: str, text: str) -> bool
     
     上传文本内容到 OSS，并存储为指定 key 对应的文件。

   - upload_image(self, key: str, image: np.ndarray) -> bool
     
     上传以 numpy 数组表示的图像到 OSS，适用于图像存储需求。

   - delete_file(self, key: str) -> bool
     
     删除 OSS 中指定 key 的文件。

   - delete_files_with_prefix(self, prefix: str) -> bool
     
     删除 OSS 中所有以指定前缀开头的文件。

   - upload_directory(self, local_path: str, prefix: str="") -> bool
     
     将本地目录上传到 OSS，所有文件存储时会以 prefix 作为路径前缀。
     
   - download_directory(self, prefix: str, local_path: str, use_basename: bool = False) -> bool
     
     下载 OSS 中指定前缀的所有文件，并存储到本地目录中。
     
     **参数**:
     
     - **prefix**: 键值前缀
     - **local_path**: 下载文件的本地目录
     - **use_basename**: 是否只使用 key 的文件名部分构建本地路径，与 download_file 相同

   - list_directory_contents(self, prefix: str, sort: bool=True) -> List[tuple[str, bool]]
     
     列出 OSS 中指定目录（通过前缀指定）的内容，返回每个文件或子目录的键和是否为目录的标识。

   - read_file_content(self, key: str) -> Optional[str]
     
     读取 OSS 上存储的文件内容，并返回字符串形式的内容。
   
   - key_exists(self, key: str) -> bool
     
     检查指定的键值是否存在于 OSS 中。
     
     **参数**:
     
     - **key**: 要检查的 OSS 键值
     
     **返回**:
     
     - 布尔值，表示键值是否存在

   - get_file_metadata(self, key: str) -> Optional[dict]
     
     获取 OSS 中指定文件的元数据信息。
     
     **参数**:
     
     - **key**: OSS 中的键值
     
     **返回**:
     
     - 包含元数据信息的字典，如不存在则返回 None

   - copy_object(self, source_key: str, target_key: str) -> bool
     
     在 OSS 中复制文件。
     
     **参数**:
     
     - **source_key**: 源文件的键值
     - **target_key**: 目标文件的键值
     
     **返回**:
     
     - 布尔值，表示复制操作是否成功

   - move_object(self, source_key: str, target_key: str) -> bool
     
     在 OSS 中移动文件（复制后删除源文件）。
     
     **参数**:
     
     - **source_key**: 源文件的键值
     - **target_key**: 目标文件的键值
     
     **返回**:
     
     - 布尔值，表示移动操作是否成功

使用示例
----------

下面的示例展示了如何使用 OssManager 完成基本的文件上传和下载操作：

.. code-block:: python

   from pywayne.aliyun_oss import OssManager
   # 初始化 OssManager 实例
   oss = OssManager(endpoint="https://oss-cn-xxx.aliyuncs.com", bucket_name="my-bucket", api_key="your_api_key", api_secret="your_api_secret")
   
   # 上传本地文件到 OSS
   success = oss.upload_file(key="data/sample.txt", file_path="./sample.txt")
   if success:
       print("文件上传成功！")
   
   # 列出所有文件键
   keys = oss.list_all_keys()
   print("当前文件列表：", keys)
   
   # 下载指定文件到本地目录
   success = oss.download_file(key="data/sample.txt", root_dir="./downloads")
   if success:
       print("文件下载成功！")
   
   # 读取 OSS 上文件的内容
   content = oss.read_file_content(key="data/sample.txt")
   print("文件内容：", content)
   
   # 检查文件是否存在
   if oss.key_exists("data/sample.txt"):
       print("文件存在")
       
   # 获取文件元数据
   metadata = oss.get_file_metadata("data/sample.txt")
   print("文件大小：", metadata.get("size"), "字节")
   
   # 复制文件
   oss.copy_object("data/sample.txt", "backup/sample.txt")
   
   # 移动文件
   oss.move_object("data/temp.txt", "archive/temp.txt")

模块扩展建议
--------------

如果未来需要实现更复杂的 OSS 操作，如多线程上传、断点续传或更细粒度的权限控制，可以在 OssManager 类的基础上进行扩展，以满足不同应用场景的需求。 