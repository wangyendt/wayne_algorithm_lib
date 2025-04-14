# author: wangye(Wayne)
# license: Apache Licence
# file: aliyun_oss.py
# time: 2025-01-02-15:39:30
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import os
import oss2
import cv2
import numpy as np
from natsort import natsorted
from typing import List, Optional
from pywayne.tools import wayne_print
from pathlib import Path
import shutil


class OssManager:
    def __init__(
            self,
            endpoint: str,
            bucket_name: str,
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            verbose: bool = True
    ):
        """
        初始化 OSS 工具类

        Args:
            endpoint: OSS endpoint
            bucket_name: Bucket 名称
            api_key: API Key（可选）
            api_secret: API Secret（可选）
            verbose: 是否打印操作信息，默认为 True
        """
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.verbose = verbose

        # 检查是否具有写入权限
        self.has_write_permission = bool(api_key and api_secret)

        # 初始化 OSS 客户端
        if self.has_write_permission:
            auth = oss2.Auth(api_key, api_secret)
        else:
            auth = oss2.AnonymousAuth()

        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)

    def _print_info(self, message: str):
        """打印信息"""
        if self.verbose:
            wayne_print(message, 'green')

    def _print_warning(self, message: str):
        """打印警告"""
        if self.verbose:
            wayne_print(message, 'yellow')

    def _check_write_permission(self) -> bool:
        """检查写入权限"""
        if not self.has_write_permission:
            self._print_warning("没有写入权限：未提供 API Key 或 API Secret")
            return False
        return True

    def download_file(self, key: str, root_dir: Optional[str] = None, use_basename: bool = False) -> bool:
        """
        下载指定 key 的文件到本地

        Args:
            key: OSS 中的键值
            root_dir: 下载文件的根目录，默认为当前目录
            use_basename: 是否只使用 key 的文件名部分构建本地路径，默认为 False。
                          如果为 True，则 `a/b/c.txt` 下载到 `root_dir/c.txt`。
                          如果为 False，则下载到 `root_dir/a/b/c.txt`。

        Returns:
            是否下载成功
        """
        try:
            # 确定下载路径
            key_path = Path(key)
            if root_dir:
                base_save_dir = Path(root_dir)
            else:
                # 如果 root_dir 未指定，则使用当前目录
                base_save_dir = Path.cwd()

            if use_basename:
                save_path = base_save_dir / key_path.name
            else:
                save_path = base_save_dir / key_path

            # 创建必要的目录
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 下载文件
            self.bucket.get_object_to_file(key, str(save_path))
            self._print_info(f"成功下载文件：{key} -> {save_path}")
            return True
        except Exception as e:
            self._print_warning(f"下载文件失败：{str(e)}")
            return False

    def download_files_with_prefix(self, prefix: str, root_dir: Optional[str] = None, use_basename: bool = False) -> bool:
        """
        下载指定前缀的所有文件到本地

        Args:
            prefix: 键值前缀
            root_dir: 下载文件的根目录，默认为当前目录
            use_basename: 是否只使用 key 的文件名部分构建本地路径，默认为 False。
                          如果为 True，则 `a/b/c.txt` 下载到 `root_dir/c.txt`。
                          如果为 False，则下载到 `root_dir/a/b/c.txt`。

        Returns:
            是否全部下载成功
        """
        try:
            # 获取所有匹配的键值
            keys = self.list_keys_with_prefix(prefix)
            if not keys:
                self._print_warning(f"未找到前缀为 '{prefix}' 的文件")
                return False

            # 下载所有文件
            success_count = 0
            for key in keys:
                if self.download_file(key, root_dir, use_basename):
                    success_count += 1

            # 检查是否全部下载成功
            if success_count == len(keys):
                self._print_info(f"成功下载所有文件，共 {success_count} 个")
                return True
            else:
                self._print_warning(f"部分文件下载失败：成功 {success_count}/{len(keys)}")
                return False
        except Exception as e:
            self._print_warning(f"批量下载文件失败：{str(e)}")
            return False
        finally:
            return True

    def list_all_keys(self, sort: bool = True) -> List[str]:
        """
        列举 bucket 中所有 key

        Args:
            sort: 是否进行自然排序，默认为 True

        Returns:
            包含所有 key 的列表
        """
        keys = []
        for obj in oss2.ObjectIterator(self.bucket):
            keys.append(obj.key)

        if sort:
            keys = natsorted(keys)

        self._print_info(f"成功获取所有 key，共 {len(keys)} 个")
        return keys

    def list_keys_with_prefix(self, prefix: str, sort: bool = True) -> List[str]:
        """
        列举指定前缀的所有 key

        Args:
            prefix: 键值前缀
            sort: 是否进行自然排序，默认为 True

        Returns:
            包含指定前缀的 key 列表
        """
        keys = []
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
            keys.append(obj.key)

        if sort:
            keys = natsorted(keys)

        self._print_info(f"成功获取前缀为 '{prefix}' 的所有 key，共 {len(keys)} 个")
        return keys

    def upload_file(self, key: str, file_path: str) -> bool:
        """
        上传本地文件

        Args:
            key: OSS 中的键值
            file_path: 本地文件路径

        Returns:
            是否上传成功
        """
        if not self._check_write_permission():
            return False

        if not os.path.exists(file_path):
            self._print_warning(f"文件不存在：{file_path}")
            return False

        try:
            self.bucket.put_object_from_file(key, file_path)
            self._print_info(f"成功上传文件：{key}")
            return True
        except Exception as e:
            self._print_warning(f"上传文件失败：{str(e)}")
            return False

    def upload_text(self, key: str, text: str) -> bool:
        """
        上传文本内容

        Args:
            key: OSS 中的键值
            text: 要上传的文本内容

        Returns:
            是否上传成功
        """
        if not self._check_write_permission():
            return False

        try:
            self.bucket.put_object(key, text.encode('utf-8'))
            self._print_info(f"成功上传文本：{key}")
            return True
        except Exception as e:
            self._print_warning(f"上传文本失败：{str(e)}")
            return False

    def upload_image(self, key: str, image: np.ndarray) -> bool:
        """
        上传图片

        Args:
            key: OSS 中的键值
            image: OpenCV 图片对象 (numpy.ndarray)

        Returns:
            是否上传成功
        """
        if not self._check_write_permission():
            return False

        try:
            # 将图片编码为 JPEG 格式的字节流
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                self._print_warning("图片编码失败")
                return False

            self.bucket.put_object(key, buffer.tobytes())
            self._print_info(f"成功上传图片：{key}")
            return True
        except Exception as e:
            self._print_warning(f"上传图片失败：{str(e)}")
            return False

    def delete_file(self, key: str) -> bool:
        """
        删除指定 key 的文件

        Args:
            key: 要删除的文件的键值

        Returns:
            是否删除成功
        """
        if not self._check_write_permission():
            return False

        try:
            self.bucket.delete_object(key)
            self._print_info(f"成功删除文件：{key}")
            return True
        except Exception as e:
            self._print_warning(f"删除文件失败：{str(e)}")
            return False

    def delete_files_with_prefix(self, prefix: str) -> bool:
        """
        删除指定前缀的所有文件

        Args:
            prefix: 要删除的文件的键值前缀

        Returns:
            是否全部删除成功
        """
        if not self._check_write_permission():
            return False

        try:
            # 获取所有匹配的键值
            keys = self.list_keys_with_prefix(prefix)
            if not keys:
                self._print_warning(f"未找到前缀为 '{prefix}' 的文件")
                return False

            # 批量删除
            self.bucket.batch_delete_objects(keys)
            self._print_info(f"成功删除前缀为 '{prefix}' 的所有文件，共 {len(keys)} 个")
            return True
        except Exception as e:
            self._print_warning(f"批量删除文件失败：{str(e)}")
            return False

    def upload_directory(self, local_path: str, prefix: str = "") -> bool:
        """
        上传整个文件夹到 OSS

        Args:
            local_path: 本地文件夹路径
            prefix: OSS 中的前缀路径，默认为空

        Returns:
            是否全部上传成功
        """
        if not self._check_write_permission():
            return False

        if not os.path.isdir(local_path):
            self._print_warning(f"目录不存在：{local_path}")
            return False

        success = True
        for root, _, files in os.walk(local_path):
            for file in files:
                # 获取文件的完整路径
                file_path = os.path.join(root, file)
                # 计算相对路径
                rel_path = os.path.relpath(file_path, local_path)
                # 构建 OSS key
                key = os.path.join(prefix, rel_path).replace("\\", "/")
                # 上传文件
                if not self.upload_file(key, file_path):
                    success = False

        return success

    def download_directory(self, prefix: str, local_path: str, use_basename: bool = False) -> bool:
        """
        从 OSS 下载整个文件夹

        Args:
            prefix: OSS 中的前缀路径 (e.g., 'data/images/')
            local_path: 下载到本地的目标路径
            use_basename: 是否只使用每个文件的基本名称保存，默认为 False。
                          如果为 True，`prefix/sub/file.txt` 会下载到 `local_path/file.txt`。
                          如果为 False，`a/b/c.txt` (且 prefix 是 `a/b/`) 会下载到 `local_path/a/b/c.txt`。
                          *注意*: 当 `use_basename` 为 True 时，如果文件夹中存在同名文件，
                          它们会被下载到同一个 `local_path` 下，可能导致覆盖。

        Returns:
            是否全部下载成功
        """
        # 确保 prefix 以 '/' 结尾，以便正确列出文件，除非 prefix 为空
        normalized_prefix = prefix
        if normalized_prefix and not normalized_prefix.endswith('/'):
            normalized_prefix += '/'

        # 获取所有匹配的文件 key
        # 使用 normalized_prefix 进行 list 操作
        keys = self.list_keys_with_prefix(normalized_prefix, sort=False) # 不需要排序，因为下载顺序不重要
        if not keys:
            # 检查 prefix 本身是否存在作为一个对象（可能是个空文件伪装的目录标记）
            is_prefix_an_object = self.key_exists(prefix) if prefix and not prefix.endswith('/') else False

            if is_prefix_an_object:
                 # 如果 prefix 本身是个文件，尝试下载它
                 self._print_info(f"Prefix '{prefix}' is an object (file), attempting to download it.")
                 # 创建父目录
                 os.makedirs(local_path, exist_ok=True)
                 return self.download_file(prefix, local_path, use_basename=use_basename)

            else:
                # 如果 prefix 对应的 key 列表为空，且 prefix 本身不是文件，则可能是空目录或不存在
                self._print_warning(f"未找到前缀为 '{normalized_prefix}' 的文件或该前缀是一个空文件夹")
                # 仍然创建本地目录结构
                os.makedirs(local_path, exist_ok=True)
                # 如果 key 列表为空，且 prefix 也不是文件，则认为操作"成功"（没有文件需要下载）
                return True


        # 创建本地目录 (如果 use_basename=True，文件直接放在 local_path)
        os.makedirs(local_path, exist_ok=True)

        success = True
        for key in keys:
            # list_keys_with_prefix 可能会返回目录本身 (e.g., 'data/') 如果它是作为一个 key 存在的
            # download_file 不能下载目录 key，所以跳过
            if key.endswith('/'):
                continue

            # 将 use_basename 传递给 download_file, local_path 作为 root_dir
            if not self.download_file(key, local_path, use_basename=use_basename):
                success = False
                self._print_warning(f"下载文件 {key} 失败") # 添加具体文件失败信息

        if success:
             self._print_info(f"成功下载目录 {prefix} 到 {local_path}" + (" (使用 basename)" if use_basename else ""))
        else:
             self._print_warning(f"下载目录 {prefix} 到 {local_path} 时部分文件失败")


        return success

    def list_directory_contents(self, prefix: str, sort: bool = True) -> List[tuple[str, bool]]:
        """
        列出指定文件夹下的所有文件和子文件夹（不深入子文件夹）

        Args:
            prefix: OSS 中的前缀路径
            sort: 是否排序，默认为 True

        Returns:
            包含 (name, is_directory) 元组的列表，name 是文件或文件夹名，is_directory 表示是否是文件夹
        """
        contents: List[tuple[str, bool]] = []
        normalized_prefix = prefix if prefix.endswith('/') else prefix + '/'

        # 获取所有以该前缀开头的文件
        all_keys = self.list_keys_with_prefix(normalized_prefix)

        # 处理每个 key
        for key in all_keys:
            # 移除前缀，得到相对路径
            relative_path = key[len(normalized_prefix):]
            components = relative_path.split('/')

            # 只处理第一级目录下的内容
            if components and components[0]:
                if len(components) == 1:
                    # 这是一个文件
                    contents.append((components[0], False))
                else:
                    # 这是一个目录
                    dir_name = components[0]
                    # 检查是否已经添加过这个目录
                    if not any(item[0] == dir_name and item[1] for item in contents):
                        contents.append((dir_name, True))

        if sort:
            # 先按类型排序（目录在前），再按名称排序
            contents.sort(key=lambda x: (not x[1], x[0]))

        self._print_info(f"成功获取目录 '{normalized_prefix}' 的内容，共 {len(contents)} 项")
        return contents

    def read_file_content(self, key: str) -> Optional[str]:
        """
        读取 OSS 上指定文件的内容

        Args:
            key: OSS 中的键值

        Returns:
            文件内容字符串，如果读取失败则返回 None
        """
        try:
            # 检查是否为文件夹（通过检查是否以'/'结尾或是否有子文件）
            if key.endswith('/'):
                self._print_warning(f"指定的键值 '{key}' 是一个文件夹")
                return None
                
            # 直接使用 bucket.list_objects 检查是否有子文件
            for _ in oss2.ObjectIterator(self.bucket, prefix=key + '/', delimiter='/', max_keys=1):
                self._print_warning(f"指定的键值 '{key}' 是一个文件夹")
                return None

            # 获取文件对象
            object_stream = self.bucket.get_object(key)
            
            # 读取内容并解码
            content = object_stream.read().decode('utf-8')
            
            self._print_info(f"成功读取文件内容：{key}")
            return content
            
        except oss2.exceptions.NoSuchKey:
            self._print_warning(f"文件不存在：{key}")
            return None
        except Exception as e:
            self._print_warning(f"读取文件失败：{str(e)}")
            return None

    def key_exists(self, key: str) -> bool:
        """
        检查指定的 key 是否存在于 OSS bucket 中

        Args:
            key: 要检查的 OSS 键值

        Returns:
            如果 key 存在则返回 True，否则返回 False
        """
        try:
            exists = self.bucket.object_exists(key)
            if exists:
                self._print_info(f"Key '{key}' 存在")
            else:
                self._print_info(f"Key '{key}' 不存在")
            return exists
        except Exception as e:
            self._print_warning(f"检查 key '{key}' 是否存在时出错：{str(e)}")
            return False

    def get_file_metadata(self, key: str) -> Optional[dict]:
        """
        获取指定 key 文件的元数据

        Args:
            key: OSS 中的键值

        Returns:
            包含元数据（如 content_length, last_modified, etag, content_type 等）的字典，
            如果文件不存在或发生错误则返回 None
        """
        try:
            meta = self.bucket.get_object_meta(key)
            metadata = {
                'content_length': meta.content_length,
                'last_modified': meta.last_modified,
                'etag': meta.etag,
                'content_type': meta.headers.get('Content-Type'),
                # 可以根据需要添加更多元数据字段
                # 'request_id': meta.request_id,
                # 'server_crc': meta.server_crc,
                # 'content_md5': meta.content_md5,
                # 'headers': meta.headers, # 原始 HTTP 头部
            }
            self._print_info(f"成功获取文件 '{key}' 的元数据")
            return metadata
        except oss2.exceptions.NoSuchKey:
            self._print_warning(f"获取元数据失败：文件 '{key}' 不存在")
            return None
        except Exception as e:
            self._print_warning(f"获取文件 '{key}' 元数据时出错：{str(e)}")
            return None

    def copy_object(self, source_key: str, target_key: str) -> bool:
        """
        在同一个 bucket 内复制对象

        Args:
            source_key: 源对象的键值
            target_key: 目标对象的键值

        Returns:
            是否复制成功
        """
        if not self._check_write_permission():
            return False

        try:
            # 检查源文件是否存在
            if not self.key_exists(source_key):
                self._print_warning(f"复制失败：源文件 '{source_key}' 不存在")
                return False
                
            result = self.bucket.copy_object(self.bucket_name, source_key, target_key)
            # 检查复制操作的 HTTP 状态码
            if 200 <= result.status < 300:
                self._print_info(f"成功将 '{source_key}' 复制到 '{target_key}'")
                return True
            else:
                self._print_warning(f"复制文件失败：从 '{source_key}' 到 '{target_key}'，状态码: {result.status}")
                return False
        except Exception as e:
            self._print_warning(f"复制文件时出错：从 '{source_key}' 到 '{target_key}' - {str(e)}")
            return False

    def move_object(self, source_key: str, target_key: str) -> bool:
        """
        在同一个 bucket 内移动/重命名对象（通过复制后删除实现）

        Args:
            source_key: 源对象的键值
            target_key: 目标对象的键值

        Returns:
            是否移动成功
        """
        if not self._check_write_permission():
             return False

        try:
            # 检查源文件是否存在
            if not self.key_exists(source_key):
                self._print_warning(f"移动失败：源文件 '{source_key}' 不存在")
                return False

            # 复制对象
            copy_success = self.copy_object(source_key, target_key)

            if copy_success:
                # 删除源对象
                delete_success = self.delete_file(source_key)
                if delete_success:
                    self._print_info(f"成功将 '{source_key}' 移动到 '{target_key}'")
                    return True
                else:
                    self._print_warning(f"移动文件 '{source_key}' 时删除源文件失败，目标文件 '{target_key}' 可能已创建")
                    # Consider attempting to delete the copied target_key for atomicity? Maybe not.
                    return False
            else:
                self._print_warning(f"移动文件失败：无法将 '{source_key}' 复制到 '{target_key}'")
                return False
        except Exception as e:
            self._print_warning(f"移动文件时出错：从 '{source_key}' 到 '{target_key}' - {str(e)}")
            return False


if __name__ == '__main__':
    # 初始化 OSS 管理器
    manager = OssManager(
        endpoint="xxx",
        bucket_name="xxx",
        api_key="xxx",
        api_secret="xxx"
    )

    # 创建测试文件
    test_file_path = "test.txt"
    test_content = "Hello, World!"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)

    # 创建测试文件夹
    test_dir_path = "test_dir"
    os.makedirs(test_dir_path, exist_ok=True)
    with open(os.path.join(test_dir_path, "file1.txt"), "w", encoding="utf-8") as f:
        f.write("File 1")
    with open(os.path.join(test_dir_path, "file2.txt"), "w", encoding="utf-8") as f:
        f.write("File 2")
    os.makedirs(os.path.join(test_dir_path, "subdir"), exist_ok=True)
    with open(os.path.join(test_dir_path, "subdir", "file3.txt"), "w", encoding="utf-8") as f:
        f.write("File 3")

    try:
        # 1. 上传文件
        wayne_print("\n1. 测试上传文件", "cyan")
        manager.upload_file("test.txt", test_file_path)
        manager.upload_file("1/test.txt", test_file_path)
        manager.upload_file("1/test2.txt", test_file_path)
        manager.upload_file("2/test3.txt", test_file_path)
        manager.upload_file("2/test4.txt", test_file_path)

        # 2. 测试上传文件夹
        wayne_print("\n2. 测试上传文件夹", "cyan")
        manager.upload_directory(test_dir_path, "test_dir")

        # 3. 上传文本
        wayne_print("\n3. 测试上传文本", "cyan")
        manager.upload_text("hello.txt", "Hello, World!")
        manager.upload_text("test.txt", "Hello, World!")

        # 4. 列举所有文件
        wayne_print("\n4. 测试列举文件", "cyan")
        files = manager.list_all_keys()
        wayne_print("文件列表：", "magenta")
        for file in files:
            wayne_print(f"  - {file}", "magenta")

        # 5. 列举指定前缀的文件
        wayne_print("\n5. 测试列举指定前缀的文件", "cyan")
        files_with_prefix = manager.list_keys_with_prefix("1/")
        wayne_print("前缀为 '1/' 的文件列表：", "magenta")
        for file in files_with_prefix:
            wayne_print(f"  - {file}", "magenta")

        # 6. 测试列举目录内容
        wayne_print("\n6. 测试列举目录内容", "cyan")
        # 列举根目录
        wayne_print("根目录内容：", "magenta")
        root_contents = manager.list_directory_contents("")
        for name, is_dir in root_contents:
            wayne_print(f"  {'📁' if is_dir else '📄'} {name}{'/' if is_dir else ''}", "magenta")

        # 列举 test_dir 目录
        wayne_print("\ntest_dir 目录内容：", "magenta")
        test_dir_contents = manager.list_directory_contents("test_dir")
        for name, is_dir in test_dir_contents:
            wayne_print(f"  {'📁' if is_dir else '📄'} {name}{'/' if is_dir else ''}", "magenta")

        # 列举 1 目录
        wayne_print("\n1 目录内容：", "magenta")
        dir1_contents = manager.list_directory_contents("1")
        for name, is_dir in dir1_contents:
            wayne_print(f"  {'📁' if is_dir else '📄'} {name}{'/' if is_dir else ''}", "magenta")

        # 7. 测试读取文件内容
        wayne_print("\n7. 测试读取文件内容", "cyan")
        # 读取文本文件
        content = manager.read_file_content("test.txt")
        if content is not None:
            wayne_print(f"test.txt 的内容：\n{content}", "magenta")
        
        # 尝试读取文件夹（应该会失败）
        content = manager.read_file_content("test_dir/")
        if content is None:
            wayne_print("成功检测到文件夹，拒绝读取", "magenta")
        
        # 读取不存在的文件
        content = manager.read_file_content("nonexistent.txt")
        if content is None:
            wayne_print("成功检测到文件不存在", "magenta")

        # 8. 下载文件
        wayne_print("\n8. 测试下载文件", "cyan")
        manager.download_file("test.txt")
        manager.download_file("1/test.txt", "downloads")

        # 9. 测试下载文件夹
        wayne_print("\n9. 测试下载文件夹", "cyan")
        manager.download_directory("test_dir/", "downloads")

        # 10. 下载指定前缀的文件
        wayne_print("\n10. 测试下载指定前缀的文件", "cyan")
        manager.download_files_with_prefix("2/", "downloads")

        # 11. 删除文件
        wayne_print("\n11. 测试删除文件", "cyan")
        manager.delete_file("test.txt")
        manager.delete_file("hello.txt")

        # 12. 删除指定前缀的文件
        wayne_print("\n12. 测试删除指定前缀的文件", "cyan")
        manager.delete_files_with_prefix("1/")
        manager.delete_files_with_prefix("2/")
        manager.delete_files_with_prefix("test_dir/")

        # --- 新增测试用例 Start ---
        try:
            # 准备测试文件
            wayne_print("\n--- 开始新增方法测试 --- ", "blue")
            test_source_file = "source_test.txt"
            test_target_file = "target_test.txt"
            test_move_source = "move_source.txt"
            test_move_target = "move_target.txt"
            test_content = "This is a test file for copy/move operations."

            with open(test_source_file, "w", encoding="utf-8") as f:
                f.write(test_content)
            with open(test_move_source, "w", encoding="utf-8") as f:
                f.write("This is a test file for move operation.")

            # 先上传一个基础文件用于测试 key_exists 和 get_metadata
            base_test_key = "base_test_file.txt"
            if manager.upload_file(base_test_key, test_source_file):

                # 13. 测试 key_exists
                wayne_print("\n13. 测试 key_exists", "cyan")
                wayne_print(f"检查存在的 key '{base_test_key}':", "magenta")
                manager.key_exists(base_test_key)
                wayne_print(f"检查不存在的 key 'nonexistent_key.txt':", "magenta")
                manager.key_exists("nonexistent_key.txt")

                # 14. 测试 get_file_metadata
                wayne_print("\n14. 测试 get_file_metadata", "cyan")
                wayne_print(f"获取 '{base_test_key}' 的元数据:", "magenta")
                metadata = manager.get_file_metadata(base_test_key)
                if metadata:
                    for k, v in metadata.items():
                        wayne_print(f"  - {k}: {v}", "magenta")
                wayne_print(f"尝试获取不存在文件 'nonexistent_key.txt' 的元数据:", "magenta")
                manager.get_file_metadata("nonexistent_key.txt")

            else:
                wayne_print("基础文件上传失败，跳过部分新测试", "red")

            # 15. 测试 copy_object
            wayne_print("\n15. 测试 copy_object", "cyan")
            # 先上传源文件
            if manager.upload_file(test_source_file, test_source_file):
                wayne_print(f"尝试复制 '{test_source_file}' 到 '{test_target_file}':", "magenta")
                if manager.copy_object(test_source_file, test_target_file):
                    wayne_print(f"验证目标文件 '{test_target_file}' 是否存在:", "magenta")
                    manager.key_exists(test_target_file)
                    # 清理复制的文件
                    manager.delete_file(test_target_file)
                # 清理源文件
                manager.delete_file(test_source_file)
            else:
                 wayne_print(f"上传源文件 {test_source_file} 失败，跳过 copy 测试", "red")
            # 测试复制不存在的文件
            wayne_print(f"尝试复制不存在的文件 'nonexistent_source.txt':", "magenta")
            manager.copy_object("nonexistent_source.txt", "some_target.txt")

            # 16. 测试 move_object
            wayne_print("\n16. 测试 move_object", "cyan")
            # 先上传源文件
            if manager.upload_file(test_move_source, test_move_source):
                wayne_print(f"尝试移动 '{test_move_source}' 到 '{test_move_target}':", "magenta")
                if manager.move_object(test_move_source, test_move_target):
                    wayne_print(f"验证目标文件 '{test_move_target}' 是否存在:", "magenta")
                    manager.key_exists(test_move_target)
                    wayne_print(f"验证源文件 '{test_move_source}' 是否已被删除:", "magenta")
                    manager.key_exists(test_move_source)
                    # 清理移动后的文件
                    manager.delete_file(test_move_target)
                else:
                    # 如果移动失败，可能源文件还在，尝试删除
                    manager.delete_file(test_move_source)
            else:
                wayne_print(f"上传源文件 {test_move_source} 失败，跳过 move 测试", "red")
            # 测试移动不存在的文件
            wayne_print(f"尝试移动不存在的文件 'nonexistent_move_source.txt':", "magenta")
            manager.move_object("nonexistent_move_source.txt", "some_move_target.txt")

            # 最终清理基础测试文件
            manager.delete_file(base_test_key)

        finally:
            # 清理本地临时文件
            if os.path.exists(test_source_file):
                os.remove(test_source_file)
            if os.path.exists(test_move_source):
                os.remove(test_move_source)
            wayne_print("\n--- 新增方法测试结束 --- ", "blue")
        # --- 新增测试用例 End ---

        # --- 新增 Download Basename 测试 Start ---
        # 17. 测试 download_file (use_basename=True)
        wayne_print("\n17. 测试 download_file (use_basename=True)", "cyan")
        download_basename_test_key = "basename/test/download_basename.txt"
        download_basename_local_dir = "downloads_basename_file"
        download_basename_local_file = os.path.join(download_basename_local_dir, "download_basename.txt")
        # 先上传一个带路径的文件
        if manager.upload_text(download_basename_test_key, "Basename test content"):
            wayne_print(f"尝试下载 {download_basename_test_key} 到 {download_basename_local_dir} (use_basename=True):", "magenta")
            if manager.download_file(download_basename_test_key, download_basename_local_dir, use_basename=True):
                wayne_print(f"验证文件是否存在于: {download_basename_local_file}", "magenta")
                if os.path.exists(download_basename_local_file):
                    wayne_print("  - 存在", "magenta")
                    # 可以在这里添加内容验证
                    try:
                        with open(download_basename_local_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            wayne_print(f"  - 内容: {content}", "magenta")
                    except Exception as e:
                         wayne_print(f"  - 读取内容失败: {e}", "yellow")
                else:
                    wayne_print("  - 不存在", "red")
            # 清理本地文件夹 (download_file 会创建父目录)
            shutil.rmtree(download_basename_local_dir, ignore_errors=True)
             # 确保清理干净，即使只有文件存在
            if os.path.exists(download_basename_local_file):
                os.remove(download_basename_local_file)
            # 清理 OSS 文件
            manager.delete_file(download_basename_test_key)
        else:
            wayne_print(f"上传 {download_basename_test_key} 失败，跳过 basename 下载测试", "red")


        # 18. 测试 download_files_with_prefix (use_basename=True)
        wayne_print("\n18. 测试 download_files_with_prefix (use_basename=True)", "cyan")
        prefix_basename_test_prefix = "basename_prefix/"
        prefix_basename_keys = [prefix_basename_test_prefix + "file1.txt", prefix_basename_test_prefix + "sub/file2.txt"]
        prefix_basename_local_dir = "downloads_basename_prefix"
        expected_files = [os.path.join(prefix_basename_local_dir, "file1.txt"), os.path.join(prefix_basename_local_dir, "file2.txt")]
        # 上传测试文件
        upload_success_prefix = all(manager.upload_text(key, f"Content for {key}") for key in prefix_basename_keys)
        if upload_success_prefix:
            wayne_print(f"尝试下载前缀 {prefix_basename_test_prefix} 到 {prefix_basename_local_dir} (use_basename=True):", "magenta")
            if manager.download_files_with_prefix(prefix_basename_test_prefix, prefix_basename_local_dir, use_basename=True):
                 wayne_print(f"验证文件是否存在于 {prefix_basename_local_dir}:", "magenta")
                 all_exist = True
                 for fpath in expected_files:
                     if os.path.exists(fpath):
                         wayne_print(f"  - {os.path.basename(fpath)}: 存在", "magenta")
                     else:
                         wayne_print(f"  - {os.path.basename(fpath)}: 不存在", "red")
                         all_exist = False
                 if not all_exist:
                      wayne_print("部分文件未按预期 (basename) 下载", "red")
            # 清理本地文件
            shutil.rmtree(prefix_basename_local_dir, ignore_errors=True)
            # 清理 OSS 文件
            manager.delete_files_with_prefix(prefix_basename_test_prefix)
        else:
            wayne_print(f"上传前缀 {prefix_basename_test_prefix} 文件失败，跳过 basename prefix 下载测试", "red")


        # 19. 测试 download_directory (use_basename=True)
        wayne_print("\n19. 测试 download_directory (use_basename=True)", "cyan")
        dir_basename_test_prefix = "basename_dir/"
        dir_basename_keys = [dir_basename_test_prefix + "root_file.txt", dir_basename_test_prefix + "sub_dir/nested_file.txt"]
        dir_basename_local_dir = "downloads_basename_dir"
        dir_expected_files = [os.path.join(dir_basename_local_dir, "root_file.txt"), os.path.join(dir_basename_local_dir, "nested_file.txt")]
        # 上传测试文件
        upload_success_dir = all(manager.upload_text(key, f"Content for {key}") for key in dir_basename_keys)
        if upload_success_dir:
            wayne_print(f"尝试下载目录 {dir_basename_test_prefix} 到 {dir_basename_local_dir} (use_basename=True):", "magenta")
            if manager.download_directory(dir_basename_test_prefix, dir_basename_local_dir, use_basename=True):
                wayne_print(f"验证文件是否存在于 {dir_basename_local_dir}:", "magenta")
                dir_all_exist = True
                for fpath in dir_expected_files:
                     if os.path.exists(fpath):
                         wayne_print(f"  - {os.path.basename(fpath)}: 存在", "magenta")
                     else:
                         wayne_print(f"  - {os.path.basename(fpath)}: 不存在", "red")
                         dir_all_exist = False
                if not dir_all_exist:
                    wayne_print("部分文件未按预期 (basename) 下载", "red")

            # 清理本地文件
            shutil.rmtree(dir_basename_local_dir, ignore_errors=True)
            # 清理 OSS 文件
            manager.delete_files_with_prefix(dir_basename_test_prefix)
        else:
            wayne_print(f"上传目录 {dir_basename_test_prefix} 文件失败，跳过 basename dir 下载测试", "red")
        # --- 新增 Download Basename 测试 End ---

    finally:
        # 清理本地临时文件
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        if os.path.exists(test_dir_path):
            shutil.rmtree(test_dir_path)
        if os.path.exists("downloads"):
            shutil.rmtree("downloads")
        # --- 新增清理 Start ---
        if os.path.exists(download_basename_local_dir):
             shutil.rmtree(download_basename_local_dir, ignore_errors=True)
        # 确保 download_file 的 basename 文件也被删除（如果目录删除失败）
        if os.path.exists(download_basename_local_file):
            try:
                os.remove(download_basename_local_file)
            except OSError:
                pass # Ignore error if it's already gone or part of the dir
        if os.path.exists(prefix_basename_local_dir):
             shutil.rmtree(prefix_basename_local_dir, ignore_errors=True)
        if os.path.exists(dir_basename_local_dir):
             shutil.rmtree(dir_basename_local_dir, ignore_errors=True)
        # --- 新增清理 End ---
        wayne_print("\n--- 新增方法测试结束 --- ", "blue")
    # --- 新增测试用例 End ---
