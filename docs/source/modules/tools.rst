工具函数 (tools)
=================

工具函数模块提供了一系列常用的辅助函数，包括文件操作、日志记录、计时器、单例模式等功能。下面详细介绍各部分功能、应用场景和示例。

装饰器
-----------

1. .. py:decorator:: func_timer

   该装饰器用于测量单个函数的执行时间，便于了解函数性能瓶颈。
   
   **应用场景**：适用于对关键计算或IO密集型函数进行性能调优，监控函数运行耗时。
   
   **示例**::

      @func_timer
      def compute():
          # 执行复杂计算
          pass

2. .. py:decorator:: func_timer_batch

   批量函数计时装饰器，用于统计函数多次调用的平均执行时长。
   
   **应用场景**：适合在大批量数据处理或重复调用的场景下，分析整体性能。
   
   **示例**::

      @func_timer_batch
      def process_data(data):
          # 处理数据列表
          pass

3. .. py:decorator:: maximize_figure

   用于在调用matplotlib绘图函数时自动调整图形窗口至最大化，确保图形展示完整。
   
   **应用场景**：适用于需要全屏展示图形报告或课堂演示时使用。
   
   **示例**::

      @maximize_figure
      def plot_results(results):
          import matplotlib.pyplot as plt
          plt.plot(results)
          plt.show()

4. .. py:decorator:: singleton

   单例模式装饰器，确保一个类只会被实例化一次，常用于共享资源或配置管理。
   
   **应用场景**：适用于数据库连接、配置管理器等只需要一个全局实例的情况。
   
   **示例**::

      @singleton
      class ConfigManager:
          pass

5. .. py:decorator:: binding_press_release

   按键绑定装饰器，用于注册并处理按键按下和释放事件。
   
   **应用场景**：在GUI开发中实现快捷键操作，提高用户交互效率。
   
   **示例**::

      @binding_press_release
      def on_key_event(event):
          # 处理按键事件
          pass

6. .. py:decorator:: trace_calls

   函数调用追踪装饰器，用于记录函数调用过程，帮助调试和性能分析。
   
   **应用场景**：适用于复杂系统中的调试，帮助理解函数间调用关系。
   
   **示例**::

      @trace_calls
      def debug_function():
          pass

文件操作
-----------

1. .. py:function:: list_all_files(root: str, keys_and: Optional[List[str]] = None, keys_or: Optional[List[str]] = None, outliers: Optional[List[str]] = None, full_path: bool = False) -> List[str]

   列出指定目录下的所有文件，并支持关键词过滤。
   
   **应用场景**：适用于日志搜集、项目目录分析和批量文件处理任务。
   
   **示例**::

      files = list_all_files("./data", keys_and=[".txt"], full_path=True)
      print(files)

2. .. py:function:: count_file_lines(file_path: str) -> int

   计算指定文件的行数。
   
   **应用场景**：适合用于代码行数统计、大文件内容验证或快速分析文本文件大小。
   
   **示例**::

      num_lines = count_file_lines("./script.py")
      print(f"行数：{num_lines}")

日志和打印
-----------

1. .. py:function:: wayne_logger(logger_name: str, project_version: str, log_root: str, stream_level=logging.DEBUG, single_file_level=logging.INFO, batch_file_level=logging.DEBUG)

   创建一个带有彩色输出的日志记录器，用于详细记录程序运行时的信息。
   
   **应用场景**：在开发和生产环境中定制日志格式，便于问题排查和性能监控。
   
   **示例**::

      logger = wayne_logger("myLogger", "1.0.0", "./logs")
      logger.info("应用启动")

2. .. py:function:: wayne_print(text: object, color: str = "default", bold: bool = False, verbose: Union[bool, int] = False)

   增强的带颜色打印函数，支持多级调试模式和复杂数据结构的优化显示。
   
   **参数说明**：
   
   - ``text``: 要打印的文本内容（支持任意类型，复杂类型将使用pprint格式化）
   - ``color``: 文本颜色，支持 "default", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
   - ``bold``: 是否加粗显示
   - ``verbose``: 调试级别
     
     - ``0`` 或 ``False``: 无调试信息（默认）
     - ``1`` 或 ``True``: 简单调试模式（时间戳+文件名+行号）
     - ``2``: 完整调试模式（详细调用栈信息）
   
   **新增功能**：
   
   - **多级调试模式**: 支持不同详细程度的调试信息
   - **智能格式化**: 自动检测复杂数据类型（字典、列表、元组等）并使用pprint美化输出
   - **向后兼容**: 完全兼容原有的布尔型verbose参数
   
   **应用场景**：在调试模式下快速确定关键输出，或在命令行工具中增强用户体验。不同级别的verbose模式适合不同的调试需求，pprint功能使复杂数据结构更易读。
   
   **示例**::

      # 基本使用
      wayne_print("操作成功", color="green", bold=True)
      
      # 简单调试模式 - 显示时间戳和位置
      wayne_print("调试信息", color="yellow", verbose=1)
      # 或使用布尔值（向后兼容）
      wayne_print("调试信息", color="yellow", verbose=True)
      
      # 完整调试模式 - 显示详细调用栈
      wayne_print("详细调试", color="red", verbose=2)
      
      # 复杂数据结构会自动使用pprint格式化
      data = {"name": "张三", "skills": ["Python", "Go"]}
      wayne_print(data, color="blue", verbose=1)
      
      # 输出示例：
      # 简单模式: [2025-07-29 11:21:20.897] /path/to/script.py, line 42
      # 完整模式调用栈:
      #   1. 文件: /path/to/script.py, line 42
      #      函数: main() 或 模块顶层执行
      #      代码: wayne_print("调试信息", verbose=2)
      # 复杂数据: {'name': '张三', 'skills': ['Python', 'Go']}

配置文件操作
-----------------

1. .. py:function:: write_yaml_config(config_yaml_file: str, config: dict, update=False, use_lock: bool = False)

   将配置字典写入YAML文件，支持新增或更新配置。
   
   **应用场景**：适用于保存动态配置、用户自定义设置等。
   
   **示例**::

      config = {'version': '1.0.0', 'debug': True}
      write_yaml_config("config.yaml", config)

2. .. py:function:: read_yaml_config(config_yaml_file: str, use_lock: bool = False)

   从YAML文件中读取配置数据，并返回字典。
   
   **应用场景**：在程序启动时加载配置参数，实现动态参数配置。
   
   **示例**::

      config = read_yaml_config("config.yaml")
      print(config)

其他工具
-----------

1. .. py:function:: compose_funcs(*funcs)

   将多个函数组合成一个复合函数，实现链式数据处理。
   
   **应用场景**：适用于流水线数据处理和函数式编程场景。
   
   **示例**::

      def f(x): return x + 1
      def g(x): return x * 2
      h = compose_funcs(f, g)
      print(h(3))  # 输出8

2. .. py:function:: disable_print_wrap_and_suppress(deal_with_numpy=True, deal_with_pandas=True)

   禁用numpy和pandas库的自动换行和部分警告信息，便于查看完整数据输出。
   
   **应用场景**：在终端调试或数据展示时，保证数据输出的完整性。
   
   **示例**::

      disable_print_wrap_and_suppress()
      import numpy as np
      print(np.arange(1000))

3. .. py:function:: say(text, lang='zh')

   文本转语音工具，基于gTTS实现，支持多语言文本转换为语音。
   
   **应用场景**：适用于语音播报、辅助阅读、交互式语音应用等。
   
   **示例**::

      say("你好，欢迎使用pywayne", lang='zh')

4. .. py:function:: wayne_print_table(data, headers=None, align=None, title=None, border='unicode', color='default', bold_header=False)

   在终端打印带边框的格式化表格，支持颜色和表头加粗。

   **参数说明**：

   - ``data``: 二维列表，每个子列表为一行。
   - ``headers``: 列名列表，省略则无表头。
   - ``align``: 每列对齐方向列表，可选 ``'left'``、``'right'``、``'center'``，默认全左对齐。
   - ``title``: 表格标题，显示在顶部居中；省略则不显示。
   - ``border``: 边框风格，``'unicode'``（默认）或 ``'simple'``（纯 ASCII ``+/-/|``）。
   - ``color``: 整体颜色，与 ``wayne_print`` 颜色名一致（``'default'``、``'red'``、``'green'``、``'yellow'``、``'blue'``、``'magenta'``、``'cyan'``、``'white'``）。
   - ``bold_header``: 表头是否加粗，默认 ``False``。

   **应用场景**：模型对比、实验结果汇总、终端调试输出美化。

   **示例**::

      wayne_print_table(
          data=[["ResNet50", "92.3%", "45ms"], ["MobileNet", "88.1%", "12ms"]],
          headers=["模型", "准确率", "延迟"],
          align=["left", "right", "right"],
          title="模型对比",
          color="cyan",
          bold_header=True,
      )


重试与缓存
-----------

1. .. py:decorator:: retry(max_tries=3, delay=1.0, backoff=2.0, exceptions=(Exception,), on_retry=None)

   重试装饰器，函数失败后按指数退避自动重试。

   **参数说明**：

   - ``max_tries``: 最大尝试次数（含首次调用）。
   - ``delay``: 首次重试前等待秒数。
   - ``backoff``: 每次重试后等待时间的倍增系数。
   - ``exceptions``: 触发重试的异常类型元组。
   - ``on_retry``: 每次重试前的可选回调 ``(exception, attempt)``。

   **应用场景**：调用网络接口（飞书、OSS、LLM）时，应对偶发性超时或限速错误。

   **示例**::

      @retry(max_tries=5, delay=0.5, backoff=2.0, exceptions=(IOError, TimeoutError))
      def upload_file(path):
          ...

2. .. py:decorator:: disk_cache(ttl=None, cache_dir=None, ignore_kwargs=None)

   磁盘缓存装饰器，基于 pickle 持久化到 ``~/.wayne_cache``，支持 TTL 过期。
   被装饰函数额外获得 ``func.cache_clear()`` 和 ``func.cache_dir`` 两个属性。

   **参数说明**：

   - ``ttl``: 缓存有效期（秒），``None`` 表示永不过期。
   - ``cache_dir``: 自定义缓存目录，默认 ``~/.wayne_cache``。
   - ``ignore_kwargs``: 计算缓存键时忽略的关键字参数名列表。

   **应用场景**：缓存 LLM 结果、慢查询、大文件解析，进程重启后仍有效。

   **示例**::

      @disk_cache(ttl=3600)
      def query_api(url):
          ...

      query_api.cache_clear()   # 手动清除缓存


并发工具
-----------

1. .. py:function:: parallel_map(func, items, n_workers=8, mode='thread', show_progress=False, desc='', timeout=None)

   对序列中每个元素并发执行函数，保序返回结果列表。

   **参数说明**：

   - ``func``: 单参数函数 ``func(item) -> result``。
   - ``items``: 输入序列。
   - ``n_workers``: 并发线程/进程数。
   - ``mode``: 可选 "thread"（I/O 密集，默认）或 "process"（CPU 密集）。
   - ``show_progress``: 是否显示 tqdm 进度条。
   - ``desc``: 进度条描述文字。
   - ``timeout``: 单任务超时秒数。

   **应用场景**：批量下载文件、批量处理图片、批量调用 API。

   **示例**::

      results = parallel_map(download, url_list, n_workers=16, show_progress=True)
      results = parallel_map(heavy_compute, data, n_workers=4, mode='process')

2. .. py:decorator:: with_progress(desc='', unit='it', total=None)

   装饰器：将被装饰函数的第一个可迭代参数自动包裹上 tqdm 进度条。

   **示例**::

      @with_progress(desc="处理帧")
      def process(frames: list):
          for frame in frames:   # frames 已自动被 tqdm 包裹
              ...

3. .. py:function:: progress_iter(iterable, desc='', total=None, unit='it')

   将任意可迭代对象包裹上 tqdm 进度条并返回。

   **示例**::

      for path in progress_iter(file_list, desc="加载图片"):
          process(path)


文件监视
-----------

1. .. py:class:: FileWatcher(path, on_created=None, on_modified=None, on_deleted=None, extensions=None, recursive=False)

   监视文件或目录的变化，变化时触发回调（基于 watchdog，事件驱动）。
   支持上下文管理器协议，也可手动调用 ``start()`` / ``stop()``。

   **参数说明**：

   - ``path``: 要监视的文件或目录路径。
   - ``on_created``: 新文件创建回调 ``(file_path: str) -> None``。
   - ``on_modified``: 文件修改回调 ``(file_path: str) -> None``。
   - ``on_deleted``: 文件删除回调 ``(file_path: str) -> None``。
   - ``extensions``: 只关注的后缀名列表（如 ``['.log', '.csv']``），空列表监听所有文件。
   - ``recursive``: 是否递归监视子目录。

   **应用场景**：数据落盘自动触发处理、日志异常自动推送飞书告警、配置文件热重载。

   **示例**::

      # 上下文管理器写法
      with FileWatcher(
          "/data/results",
          on_created=lambda p: bot.send_text_to_chat(chat_id, f"新文件: {p}"),
          extensions=[".csv"],
          recursive=True
      ):
          time.sleep(3600)

      # 手动控制
      w = FileWatcher("/tmp/watch_me", on_modified=reload_config).start()
      ...
      w.stop()
