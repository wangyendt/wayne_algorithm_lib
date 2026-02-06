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