import datetime
import functools
import inspect
import logging
import os
import platform
import pprint
import random
import subprocess
import sys
import threading
import time
from typing import *

import matplotlib.pyplot as plt
import yaml
from filelock import FileLock

try:
    from PIL import Image
except ImportError:
    logging.warn('Try pip install Pillow')


# try:
#     from PyQt5.QtCore import *
#     from PyQt5.QtGui import *
#     from PyQt5.QtWidgets import *
# except ImportError:
#     logging.warn('Try pip install pyqt5-tools')
#
# try:
#     matplotlib.use('QT5Agg')  # for mac, TkAgg
# except:
#     logging.warn('backend qt5 not supported')


# Wayne:

def func_timer(func):
    """
    用于计算函数执行时间
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.time()
        r = func(*args, **kw)
        print(f'{func.__name__} excuted in {time.time() - start:.3f} s')
        return r

    return wrapper


def func_timer_batch(func):
    """
    用于计算函数被调用次数和总耗时
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.time()
        r = func(*args, **kw)
        end = time.time()
        wrapper.num_calls += 1
        wrapper.elapsed_time += end - start
        return r

    wrapper.num_calls = 0
    wrapper.elapsed_time = 0
    return wrapper


def maximize_figure(func):
    """
    用于最大化figure
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        backend = plt.get_backend()
        mgr = plt.get_current_fig_manager()

        if backend.lower() == 'tkagg':
            mgr.window.state('zoomed')  # TkAgg backend
        elif backend.lower() == 'wxagg':
            mgr.frame.Maximize(True)  # wxAgg backend
        elif backend.lower() == 'qt4agg' or backend.lower() == 'qt5agg':
            mgr.window.showMaximized()  # Qt4Agg and Qt5Agg backend
        elif backend.lower() in ['gtk3agg', 'gtk3cairo']:
            mgr.window.maximize()  # GTK3 backends
        else:
            print(f"Warning: '{backend}' backend does not support maximize_figure.")
        return ret

    return wrapper


def singleton(cls):
    """
    单例模式装饰器
    :param cls: 需要被单例化的类
    :param args:
    :param kw:
    :return:
    """

    instance = {}
    lock = threading.Lock()

    @functools.wraps(cls)
    def _singleton(*args, **kw):
        if cls not in instance:
            with lock:
                if cls not in instance:
                    instance[cls] = cls(*args, **kw)
        return instance[cls]

    return _singleton


def binding_press_release(func_dict: dict):
    """
    用来绑定figure和键鼠处理函数
    :param func_dict: 映射字典
    :return:

    example:
    func_dict = {
        'button_press_event': on_button_press,
        'button_release_event': on_button_release,
        'key_press_event': on_key_press,
    }
    """

    def binding_press_release_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            for k, v in func_dict.items():
                ret.canvas.mpl_connect(k, v)
            return ret

        return wrapper

    return binding_press_release_decorator


def trace_calls(_func=None, *, print_type='default'):
    """
    A decorator to trace calls to a function, printing detailed call information.

    Parameters:
        _func (callable, optional): The function to be decorated. If None, this decorator
                                    is returned with the print_type set.
        print_type (str, optional): The type of printing method to use for logging call
                                    information. 'default' for wayne_print with 'green' color,
                                    and 'pprint' for pretty-printed logs.

    Returns:
        callable: A decorated function with enhanced logging.

    Usage:
        @trace_calls            # Uses wayne_print with default settings.
        def some_function():
            pass

        @trace_calls(print_type='pprint')  # Uses pprint for logging.
        def another_function():
            pass
    """

    def decorator(func):
        if not hasattr(decorator, "call_counts"):
            decorator.call_counts = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()  # Start time of the function call.
            if func.__name__ not in decorator.call_counts:
                decorator.call_counts[func.__name__] = 0

            # Collect caller information from the stack.
            caller = inspect.stack()[1]
            caller_frame = caller[0]
            caller_function = caller_frame.f_code.co_name
            caller_filename = caller_frame.f_code.co_filename
            caller_lineno = caller_frame.f_lineno

            # Increment the count of calls.
            decorator.call_counts[func.__name__] += 1

            result = func(*args, **kwargs)
            end_time = time.time()  # End time of the function call.

            # Log dictionary creation.
            log_dict = {
                "Caller": caller_function,
                "Callee": func.__name__,
                "Time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "Execution Time": f"{end_time - start_time:.6f}s",
                "Calls": decorator.call_counts[func.__name__],
                "Arguments": args,
                "Keyword Arguments": kwargs,
                "Return Value": result,
                "File": caller_filename,
                "Line": caller_lineno
            }

            # Print the log based on the specified print type.
            if print_type == 'pprint':
                pprint.pp(log_dict)
            else:
                wayne_print(log_dict, 'green')

            return result

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def list_all_files(
        root: str,
        keys_and: Optional[List[str]] = None,
        keys_or: Optional[List[str]] = None,
        outliers: Optional[List[str]] = None,
        full_path: bool = False
) -> List[str]:
    """
    List all file paths under a directory that satisfy given conditions.

    Author:   wangye
    Datetime: 2019/4/16 18:03

    :param root: Root directory to start the search from.
    :param keys_and: List of keywords that must appear in the file paths.
    :param keys_or: List of keywords where at least one must appear in the file paths.
    :param outliers: List of keywords to exclude from the file paths.
    :param full_path: Whether to return the full path or not.
    :return: List of file paths that satisfy the given conditions.
    """
    keys_and = keys_and or []
    keys_or = keys_or or []
    outliers = outliers or []

    files = []
    for item in os.listdir(root):
        path = os.path.join(root, item)

        if os.path.isdir(path):
            files.extend(list_all_files(path, keys_and, keys_or, outliers, full_path))

        if os.path.isfile(path):
            if (all(key in path for key in keys_and) and
                    (not keys_or or any(key in path for key in keys_or)) and
                    not any(outlier in path for outlier in outliers)):
                files.append(os.path.abspath(path) if full_path else path)

    return files


def count_file_lines(file_path: str) -> int:
    def read_file_in_blocks(file: IO[str], block_size: int = 65536) -> Generator[str, None, None]:
        """Read a file in blocks of a given size."""
        while True:
            block_data = file.read(block_size)
            if not block_data:
                break
            yield block_data

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(block.count('\n') for block in read_file_in_blocks(f))


def leader_speech():
    stencil = '{n40}是{v0}{n41}，{v1}行业{n30}。{n42}是{v2}{n20}{n43}，通过{n31}和{n32}达到{n33}。' \
              '{n44}是在{n45}采用{n21}打法达成{n46}。{n47}{n48}作为{n22}为产品赋能，{n49}作为{n23}' \
              '的评判标准。亮点是{n24}，优势是{n25}。{v3}整个{n410}，{v4}{n26}{v5}{n411}。{n34}是{n35}' \
              '达到{n36}标准。'

    num = {'v': 6, 'n2': 7, 'n3': 7, 'n4': 12}

    # 二字动词
    v = '皮实、复盘、赋能、加持、沉淀、倒逼、落地、串联、协同、反哺、兼容、包装、重组、履约、' \
        '响应、量化、发力、布局、联动、细分、梳理、输出、加速、共建、共创、支撑、融合、解耦、聚合、' \
        '集成、对标、对齐、聚焦、抓手、拆解、拉通、抽象、摸索、提炼、打通、吃透、迁移、分发、分层、' \
        '封装、辐射、围绕、复用、渗透、扩展、开拓、给到、死磕、破圈'.split('、')

    # 二字名词
    n2 = '漏斗、中台、闭环、打法、纽带、矩阵、刺激、规模、场景、维度、格局、形态、生态、话术、' \
         '体系、认知、玩法、体感、感知、调性、心智、战役、合力、赛道、基因、因子、模型、载体、横向、' \
         '通道、补位、链路、试点'.split('、')

    # 三字名词
    n3 = '新生态、感知度、颗粒度、方法论、组合拳、引爆点、点线面、精细化、差异化、平台化、结构化、' \
         '影响力、耦合性、易用性、便捷性、一致性、端到端、短平快、护城河'.split('、')

    # 四字名词
    n4 = '底层逻辑、顶层设计、交付价值、生命周期、价值转化、强化认知、资源倾斜、完善逻辑、抽离透传、' \
         '复用打法、商业模式、快速响应、定性定量、关键路径、去中心化、结果导向、垂直领域、归因分析、' \
         '体验度量、信息屏障'.split('、')

    v_list = random.sample(v, num['v'])
    n2_list = random.sample(n2, num['n2'])
    n3_list = random.sample(n3, num['n3'])
    n4_list = random.sample(n4, num['n4'])
    lists = {'v': v_list, 'n2': n2_list, 'n3': n3_list, 'n4': n4_list}

    dic = {}
    for current_type in ['v', 'n2', 'n3', 'n4']:
        current_list = lists[current_type]
        for i in range(0, len(current_list)):
            dic[current_type + str(i)] = current_list[i]

    result = stencil.format(**dic)
    return result


def compose_funcs(*funcs):
    if funcs:
        return functools.reduce(
            lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)), funcs
        )
    else:
        raise ValueError('Composition of empty sequence not supported!')


def disable_print_wrap_and_suppress(deal_with_numpy=True, deal_with_pandas=True):
    """
    Disables the wrapping and suppresses the scientific notation of floating point numbers
    in numpy arrays and pandas DataFrames when printed to the console.

    :param deal_with_numpy: A boolean flag indicating whether to apply settings to numpy arrays.
    :param deal_with_pandas: A boolean flag indicating whether to apply settings to pandas DataFrames.
    """
    if deal_with_numpy:
        import numpy as np
        np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)
    if deal_with_pandas:
        import pandas as pd
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)


def wayne_logger(logger_name: str, project_version: str, log_root: str,
                 stream_level=logging.DEBUG,
                 single_file_level=logging.INFO,
                 batch_file_level=logging.DEBUG):
    """
    A Logger function that sets up a logging system which writes logs to the console and to files.
    Logs can be written at different levels and are colored for console output.

    :param logger_name: Name of the logger.
    :param project_version: Version of the project for logging.
    :param log_root: Root directory for log files.
    :param stream_level: Logging level for console output.
    :param single_file_level: Logging level for the single main log file.
    :param batch_file_level: Logging level for batch log files.
    """

    class ColoredFormatter(logging.Formatter):
        """
        A Formatter that adds color codes to log levels for console output.
        """
        # ANSI escape sequences for colors.
        COLORS = {
            'DEBUG': '\033[36m\033[1m',  # Cyan
            'INFO': '\033[32m\033[1m',  # Bright green
            'WARNING': '\033[33m\033[1m',  # Bright yellow
            'ERROR': '\033[31m\033[1m',  # Bright red
            'CRITICAL': '\033[35m\033[1m',  # Purple
            'ENDC': '\033[0m',  # Reset to default color
        }

        def format(self, record):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['ENDC']}"
            return super().format(record)

    logger = logging.getLogger(logger_name)
    logger.propagate = 0  # Prevents log messages from propagating to the logger's parent.
    logger.setLevel(logging.DEBUG)

    # ColoredFormatter for console readability.
    formatter = ColoredFormatter(
        f'%(asctime)s-%(module)s-line[%(lineno)d]-v{project_version}-%(levelname)s-%(message)s'
    )

    # Setting up StreamHandler for console output.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Setting up FileHandler for a single main log file.
    os.makedirs(log_root, exist_ok=True)
    single_file_handler = logging.FileHandler(
        os.path.join(log_root, 'main.log'), encoding='utf-8'
    )
    single_file_handler.setLevel(single_file_level)
    single_file_handler.setFormatter(formatter)
    logger.addHandler(single_file_handler)

    # Setting up FileHandler for batch log files with unique naming.
    batch_log_directory = os.path.join(log_root, 'batches')
    os.makedirs(batch_log_directory, exist_ok=True)
    batch_file_handler = logging.FileHandler(
        os.path.join(batch_log_directory, f'{project_version}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")}.log'),
        encoding='utf-8'
    )
    batch_file_handler.setLevel(batch_file_level)
    batch_file_handler.setFormatter(formatter)
    logger.addHandler(batch_file_handler)

    return logger


def wayne_print(text: object, color: str = "default", bold: bool = False, verbose: Union[bool, int] = False):
    """
    Function to print text in color and/or bold.

    Parameters:
    - text: Text to be printed
    - color: Text color, options include "default", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
    - bold: Boolean, if True, prints the text in bold
    - verbose: Debug level, 0/False=no debug, 1/True=simple debug (timestamp+file+line), 2=full debug (call stack)
    """
    colors = {
        "default": "\033[0m",  # Default color
        "red": "\033[31m",  # Red
        "green": "\033[32m",  # Green
        "yellow": "\033[33m",  # Yellow
        "blue": "\033[34m",  # Blue
        "magenta": "\033[35m",  # Magenta
        "cyan": "\033[36m",  # Cyan
        "white": "\033[37m"  # White
    }
    
    # Convert verbose to integer for consistent handling
    if isinstance(verbose, bool):
        verbose_level = 1 if verbose else 0
    else:
        verbose_level = int(verbose)
    
    # Print verbose information if requested
    if verbose_level > 0:
        # Get current timestamp with millisecond precision
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Get call stack information
        stack = inspect.stack()
        caller_frame = stack[1]  # Skip frame 0 (current function)
        
        if verbose_level == 1:
            # Simple verbose mode: timestamp + file + line
            filename = os.path.abspath(caller_frame.filename)
            line_number = caller_frame.lineno
            print(f"\033[36m[{timestamp}] {filename}, line {line_number}\033[0m")
            
        elif verbose_level >= 2:
            # Full verbose mode: complete call stack
            print(f"\033[36m{'='*80}\033[0m")
            print(f"\033[36m[VERBOSE] Wayne Print Debug Information\033[0m")
            print(f"\033[36m[TIMESTAMP] {timestamp}\033[0m")
            
            # Print call stack (skip the first frame which is wayne_print itself)
            print(f"\033[36m[CALL STACK] 调用栈信息 (从最近到最远):\033[0m")
            for i, frame_info in enumerate(stack[1:], 1):  # Skip frame 0 (current function)
                filename = os.path.abspath(frame_info.filename)
                function_name = frame_info.function
                line_number = frame_info.lineno
                code_context = frame_info.code_context[0].strip() if frame_info.code_context else "N/A"
                
                print(f"\033[36m  {i}. 文件: {filename}, line {line_number}\033[0m")
                print(f"\033[36m     函数: {function_name}\033[0m")
                print(f"\033[36m     代码: {code_context}\033[0m")
                print()
            
            print(f"\033[36m[MESSAGE] 实际输出内容:\033[0m")
            print(f"\033[36m{'='*80}\033[0m")
    
    # Check if text is a complex data structure that would benefit from pprint
    def is_complex_type(obj):
        return isinstance(obj, (dict, list, tuple, set, frozenset)) and not isinstance(obj, str)
    
    bold_code = "\033[1m" if bold else ""
    color_code = colors.get(color, colors["default"])
    end_code = colors["default"]  # Reset color and style to avoid affecting subsequent prints
    
    if is_complex_type(text):
        # Use pprint for complex data structures
        print(f"{color_code}{bold_code}", end="")
        pprint.pprint(text)
        print(f"{end_code}", end="")
    else:
        # Regular print for simple types
        print(f"{color_code}{bold_code}{text}{end_code}")


def write_yaml_config(config_yaml_file: str, config: dict, update=False, use_lock: bool = False, default_flow_style=False):
    """
    Writes the given configuration dictionary to a YAML file with file lock protection.

    :param config_yaml_file: The path to the YAML file where the config should be written.
    :param config: A dictionary containing the configuration settings to write.
    :param update: If True, the function will update the existing YAML file with the new config. If False, the function will overwrite the existing YAML file with the new config.
    :param use_lock: Whether to use file lock protection. Default is True.
    :param default_flow_style: Whether to use the default flow style for YAML serialization. Default is False.
    """

    def deep_merge_dicts(original, updater):
        """
        Deeply merges two dictionaries. The `updater` dictionary values will
        overwrite those from the `original` in case of conflicts. This function
        is recursive to support nested dictionaries.

        :param original: The original dictionary to be updated.
        :param updater: The dictionary with updates.
        :return: The merged dictionary.
        """
        for key, value in updater.items():
            if isinstance(value, dict) and key in original:
                original_value = original.get(key, {})
                if isinstance(original_value, dict):
                    deep_merge_dicts(original_value, value)
                else:
                    original[key] = value
            else:
                original[key] = value
        return original

    def write_config():
        if update and os.path.exists(config_yaml_file):
            with open(config_yaml_file, 'r', encoding='UTF-8') as f:
                existing_config = yaml.safe_load(f) or {}
            config.update(deep_merge_dicts(existing_config, config))
        with open(config_yaml_file, 'w', encoding='UTF-8') as f:
            yaml.dump(config, f, default_flow_style=default_flow_style, allow_unicode=True)

    if use_lock:
        lock_file = config_yaml_file + ".lock"
        with FileLock(lock_file):
            write_config()
    else:
        write_config()


def read_yaml_config(config_yaml_file: str, use_lock: bool = False):
    """
    Reads and returns the configuration from a YAML file with file lock protection.

    :param config_yaml_file: The path to the YAML file from which to read the config.
    :param use_lock: Whether to use file lock protection. Default is True.
    :return: A dictionary containing the configuration settings.
    """

    def read_config():
        with open(config_yaml_file, 'r', encoding='UTF-8') as f:
            return yaml.safe_load(f) or {}

    if use_lock:
        lock_file = config_yaml_file + ".lock"
        lock = FileLock(lock_file)

        with lock:
            return read_config()
    else:
        return read_config()


def say(text, lang='zh'):
    """
    Converts the given text to speech using the system's text-to-speech engine.
    On macOS, it uses the built-in `say` command. On Linux, it uses `espeak-ng`,
    and will automatically install `espeak-ng` if it is not already installed.

    :param text: The text string that you want to convert to speech.
    :param lang: The language code for the TTS engine (default is 'en' for English).
    :raises NotImplementedError: If the function is called on an unsupported operating system.
    :raises subprocess.CalledProcessError: If the installation of `espeak-ng` fails on Linux.
    """
    system_name = platform.system()

    if system_name == "Darwin":  # macOS
        command = f'say "{text}"'
    elif system_name == "Linux":
        # 检查是否安装了 espeak-ng
        try:
            subprocess.check_output(['which', 'espeak-ng'])
        except subprocess.CalledProcessError:
            print("espeak-ng not found, installing...")
            try:
                # 尝试安装 espeak-ng
                subprocess.check_call(['sudo', 'apt-get', 'install', '-y', 'espeak-ng'])
            except subprocess.CalledProcessError as e:
                print("Failed to install espeak-ng. Please install it manually.")
                raise e

        command = f'espeak-ng -v {lang} "{text}"'
    else:
        raise NotImplementedError("pywayne.tools > say(text) only supports macOS and Linux.")

    # 在新线程中执行命令
    thd = threading.Thread(target=os.system, args=(command,))
    thd.setDaemon(True)
    thd.start()
