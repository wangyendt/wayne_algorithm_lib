cmdlogger: 命令执行与 I/O 日志记录器
=====================================

``cmdlogger`` 是安装 ``pywayne`` 后提供的命令行工具，用于执行一个子命令并将其标准输入、输出和错误记录到文件中，同时进行实时的 I/O 转发。该工具特别适用于需要同时监控命令执行过程和保存完整执行日志的场景。

功能特点
--------

- **完整 I/O 记录**：捕获并记录子命令的所有输入、输出和错误信息
- **实时转发**：在记录的同时，实时转发 I/O 到控制台，保持交互性
- **多线程处理**：使用独立线程处理 stdin、stdout、stderr，确保流畅的 I/O 操作
- **字符编码处理**：智能处理非 UTF-8 数据，确保日志完整性
- **资源管理**：自动清理进程和文件资源，确保系统稳定性

基本用法
--------

.. code-block:: bash

   cmdlogger [--log-path LOG_PATH] <command> [args...]

   # 示例：记录 git status 的执行过程
   cmdlogger git status

   # 指定日志文件路径
   cmdlogger --log-path /tmp/build.log make -j4

   # 记录交互式命令的执行
   cmdlogger python3 -i

参数说明
--------

- **command**: 要执行的命令及其参数
- **--log-path**: 可选，指定日志文件的路径。如果未提供，将在脚本目录下创建 ``io_log.log``

工作原理
--------

1. **进程创建**：使用 ``subprocess.Popen`` 创建子进程，配置独立的 stdin、stdout、stderr 管道
2. **多线程转发**：启动三个独立的守护线程分别处理：
   
   - stdin 转发：将用户输入转发给子进程
   - stdout 转发：将子进程输出转发到控制台并记录到日志
   - stderr 转发：将子进程错误输出转发到控制台并记录到日志

3. **日志格式**：每行日志都带有前缀标识：
   
   - ``输入: <内容>`` - 标准输入
   - ``输出: <内容>`` - 标准输出  
   - ``错误: <内容>`` - 标准错误

4. **资源清理**：命令执行完成后自动清理所有资源

日志文件示例
------------

执行命令 ``cmdlogger echo "Hello World"`` 后，日志文件内容如下：

.. code-block:: text

   输出: Hello World

执行命令 ``cmdlogger python3 -c "import sys; print('stdout'); print('stderr', file=sys.stderr)"`` 后：

.. code-block:: text

   输出: stdout
   错误: stderr

高级特性
--------

**字符编码处理**
  对于非 UTF-8 编码的输出，工具会自动标记并记录字节长度，避免编码错误导致的日志丢失。

**信号处理**
  当用户中断执行（Ctrl+C）时，工具会优雅地终止子进程并清理资源。

**流结束检测**
  当输入流结束时（如用户按 Ctrl+D），工具会自动关闭子进程的标准输入，确保命令能够正常退出。

应用场景
--------

- **构建过程记录**：记录复杂的编译或构建过程，便于后续分析
- **脚本执行监控**：监控长时间运行的脚本，同时保存完整的执行日志
- **调试支持**：在调试过程中记录程序的完整输入输出
- **自动化测试**：在 CI/CD 流程中记录测试执行过程
- **性能分析**：结合时间戳记录，分析命令执行的性能特征

错误处理
--------

- **命令未找到**：当指定的命令不存在时，返回退出码 127
- **启动失败**：当子进程启动失败时，记录错误信息并返回退出码 1
- **权限问题**：当无法写入日志文件时，提示权限错误
- **I/O 错误**：在 I/O 转发过程中遇到的错误会被记录到日志中

注意事项
--------

- **交互式命令**：对于需要用户交互的命令（如密码输入），用户输入也会被记录到日志中，请注意敏感信息的处理
- **大量输出**：对于产生大量输出的命令，日志文件可能会变得很大，请确保有足够的磁盘空间
- **实时性**：虽然工具支持实时转发，但在某些情况下可能会有轻微的延迟
- **编码问题**：建议在 UTF-8 环境下使用，以获得最佳的日志记录效果

示例用法
--------

**记录编译过程**

.. code-block:: bash

   # 记录 CMake 配置过程
   cmdlogger --log-path cmake_config.log cmake ..
   
   # 记录编译过程
   cmdlogger --log-path build.log make -j$(nproc)

**监控脚本执行**

.. code-block:: bash

   # 记录 Python 脚本执行
   cmdlogger --log-path script_run.log python3 my_script.py --arg1 value1
   
   # 记录 shell 脚本执行
   cmdlogger --log-path deploy.log ./deploy.sh production

**调试程序**

.. code-block:: bash

   # 记录 GDB 调试会话
   cmdlogger --log-path debug_session.log gdb ./my_program
   
   # 记录 Python 交互式会话
   cmdlogger --log-path python_debug.log python3 -i my_module.py

**网络操作记录**

.. code-block:: bash

   # 记录 curl 请求
   cmdlogger --log-path api_test.log curl -v https://api.example.com/data
   
   # 记录 SSH 连接过程
   cmdlogger --log-path ssh_session.log ssh user@remote-host

技术实现
--------

``cmdlogger`` 使用以下关键技术：

- ``subprocess.Popen``：创建和管理子进程
- ``threading.Thread``：实现多线程 I/O 转发
- ``sys.stdin.buffer`` 等：处理二进制 I/O 流
- 文件锁机制：确保日志文件写入的原子性
- 信号处理：优雅地处理进程终止

这种设计确保了工具的稳定性和可靠性，能够处理各种复杂的命令执行场景。
