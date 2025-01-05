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

    def download_file(self, key: str, root_dir: Optional[str] = None) -> bool:
        """
        下载指定 key 的文件到本地

        Args:
            key: OSS 中的键值
            root_dir: 下载文件的根目录，默认为当前目录

        Returns:
            是否下载成功
        """
        try:
            # 确定下载路径
            if root_dir:
                save_path = Path(root_dir) / key
            else:
                save_path = Path(key)

            # 创建必要的目录
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 下载文件
            self.bucket.get_object_to_file(key, str(save_path))
            self._print_info(f"成功下载文件：{key} -> {save_path}")
            return True
        except Exception as e:
            self._print_warning(f"下载文件失败：{str(e)}")
            return False

    def download_files_with_prefix(self, prefix: str, root_dir: Optional[str] = None) -> bool:
        """
        下载指定前缀的所有文件到本地

        Args:
            prefix: 键值前缀
            root_dir: 下载文件的根目录，默认为当前目录

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
                if self.download_file(key, root_dir):
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

    def download_directory(self, prefix: str, local_path: str) -> bool:
        """
        从 OSS 下载整个文件夹

        Args:
            prefix: OSS 中的前缀路径
            local_path: 下载到本地的目标路径

        Returns:
            是否全部下载成功
        """
        # 获取所有匹配的文件
        keys = self.list_keys_with_prefix(prefix)
        if not keys:
            self._print_warning(f"未找到前缀为 '{prefix}' 的文件")
            return False

        # 创建本地目录
        os.makedirs(local_path, exist_ok=True)

        success = True
        for key in keys:
            if not self.download_file(key, local_path):
                success = False

        return success


if __name__ == '__main__':
    import shutil

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

        # 6. 下载文件
        wayne_print("\n6. 测试下载文件", "cyan")
        manager.download_file("test.txt")
        manager.download_file("1/test.txt", "downloads")

        # 7. 测试下载文件夹
        wayne_print("\n7. 测试下载文件夹", "cyan")
        manager.download_directory("test_dir/", "downloads")

        # 8. 下载指定前缀的文件
        wayne_print("\n8. 测试下载指定前缀的文件", "cyan")
        manager.download_files_with_prefix("2/", "downloads")

        # 9. 删除文件
        wayne_print("\n9. 测试删除文件", "cyan")
        manager.delete_file("test.txt")
        manager.delete_file("hello.txt")

        # 10. 删除指定前缀的文件
        wayne_print("\n10. 测试删除指定前缀的文件", "cyan")
        manager.delete_files_with_prefix("1/")
        manager.delete_files_with_prefix("2/")
        manager.delete_files_with_prefix("test_dir/")

    finally:
        # 清理测试文件
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        if os.path.exists(test_dir_path):
            shutil.rmtree(test_dir_path)
        if os.path.exists("downloads"):
            shutil.rmtree("downloads")
