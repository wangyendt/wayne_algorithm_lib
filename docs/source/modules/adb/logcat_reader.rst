ADB 日志读取器 (logcat_reader)
======================================

.. automodule:: pywayne.adb.logcat_reader

本模块提供了一个用于读取 Android 设备日志（logcat）的工具，主要通过 `AdbLogcatReader` 类实现日志的启动和读取。该工具支持两种后端实现：

- **cpp 后端**：依赖 C++ 实现的日志读取库，通过动态加载方式使用，速度较快；
- **python 后端**：使用 Python 的 `subprocess` 模块调用 `adb` 命令，作为备用方案。

.. autoclass:: pywayne.adb.logcat_reader.AdbLogcatReader
   :members:
   :undoc-members:
   :show-inheritance:

   **主要功能**：

   - 初始化时根据后端选项准备相应的日志读取库或子进程。
   - 通过调用 `start()` 方法启动日志捕获。
   - 调用 `read()` 方法以生成器方式逐行读取日志数据。

使用示例
----------

以下示例展示了如何使用 AdbLogcatReader 类读取设备日志：

.. code-block:: python

   from pywayne.adb.logcat_reader import AdbLogcatReader
   
   # 创建日志读取器实例，使用默认 cpp 后端
   reader = AdbLogcatReader()
   
   # 启动日志捕获
   reader.start()
   
   # 逐行读取日志并打印
   try:
       for line in reader.read():
           print(line)
   except KeyboardInterrupt:
       print("Logcat reading stopped.")

注意事项
----------

- 请确保 `adb` 已正确安装并配置在系统 PATH 中；
- 如果使用 `cpp` 后端，请确保相关依赖库存在，否则会自动尝试下载；
- 在使用 `python` 后端时，`adb` 命令行工具需要正常工作。

模块扩展建议
--------------

如果未来需要进一步增强 adb 日志捕获功能，可以考虑扩展以下方面：

- 集成日志过滤功能，根据关键字或日志级别筛选日志信息；
- 增加设备管理功能，实现自动检测和管理连接的 Android 设备；
- 提供日志实时批处理和存储功能，便于长期日志监控与分析；
- 支持更多 adb 命令的封装，以便在调试和开发过程中获得更丰富的信息。

高级用法与后端切换指南
-----------------------------

如果您希望在不同的环境下使用不同的后端以获得更好的性能或兼容性，可以通过构造函数参数 `backend` 来切换使用方式。例如：

.. code-block:: python

   # 使用默认的 C++ 后端
   reader_cpp = AdbLogcatReader()

   # 或者使用 Python 后端
   reader_py = AdbLogcatReader(backend='python')

使用 Python 后端时，请确保 `adb` 命令行工具已正确安装并配置在系统 PATH 中，因为该后端依赖于直接调用 `adb logcat` 命令。

故障排查建议：

- 确保已开启设备的 USB 调试模式，并且设备与电脑之间连接稳定。
- 如果使用 C++ 后端且日志无法正常读取，请检查依赖库是否已成功下载并加载。
- 若遇日志读取中断或无输出的情况，可尝试切换至 Python 后端进行验证，或手动调用 `adb logcat -d` 命令查看日志输出。
- 结合上文的注意事项，仔细检查系统环境配置和相关工具的安装情况。

该模块未来还可扩展更多高级功能，例如：

- 实时日志过滤（根据关键字或日志级别筛选日志信息）；
- 日志存储和批处理，便于长期监控和分析；
- 更完善的设备管理和多设备日志处理支持。 