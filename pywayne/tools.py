import datetime
import functools
import hashlib
import inspect
import logging
import os
import pickle
import platform
import pprint
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import yaml
from filelock import FileLock
from PIL import Image
from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


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


def retry(
        max_tries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        on_retry: Optional[Callable[[BaseException, int], None]] = None
):
    """
    重试装饰器，支持指数退避。

    :param max_tries: 最大尝试次数（含第一次调用）。
    :param delay: 首次重试前的等待秒数。
    :param backoff: 每次重试后等待时间的倍增系数。
    :param exceptions: 需要触发重试的异常类型元组。
    :param on_retry: 每次重试前调用的可选回调 ``(exception, attempt)``。

    Usage::

        @retry(max_tries=5, delay=0.5, backoff=2.0, exceptions=(IOError, TimeoutError))
        def flaky_request():
            ...

        # 也可以不带参数使用默认值
        @retry()
        def another_func():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_tries:
                        break
                    if on_retry is not None:
                        on_retry(exc, attempt)
                    else:
                        wayne_print(
                            f"[retry] {func.__name__} 第 {attempt}/{max_tries} 次失败: {exc}，"
                            f"{current_delay:.1f}s 后重试...",
                            "yellow"
                        )
                    time.sleep(current_delay)
                    current_delay *= backoff
            raise last_exc

        return wrapper

    return decorator


def disk_cache(
        ttl: Optional[float] = None,
        cache_dir: Optional[str] = None,
        ignore_kwargs: Optional[List[str]] = None
):
    """
    磁盘缓存装饰器，基于 pickle 持久化，支持 TTL 过期。

    :param ttl: 缓存有效期（秒）。``None`` 表示永不过期。
    :param cache_dir: 缓存目录，默认 ``~/.wayne_cache``。
    :param ignore_kwargs: 计算缓存键时忽略的关键字参数名列表。

    被装饰的函数会额外获得两个属性：
    - ``func.cache_clear()``  — 清除该函数所有缓存文件
    - ``func.cache_dir``      — 缓存目录路径字符串

    Usage::

        @disk_cache(ttl=3600)
        def expensive_query(url: str) -> dict:
            ...

        @disk_cache()           # 永不过期
        def compute(x, y):
            ...

        compute.cache_clear()   # 手动清除缓存
    """

    def decorator(func: Callable) -> Callable:
        _cache_root = Path(cache_dir or os.path.expanduser('~/.wayne_cache'))
        _cache_root.mkdir(parents=True, exist_ok=True)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            filtered_kw = {k: v for k, v in kwargs.items() if k not in (ignore_kwargs or [])}
            key_src = f"{func.__module__}.{func.__qualname__}|{repr(args)}|{repr(sorted(filtered_kw.items()))}"
            cache_key = hashlib.md5(key_src.encode('utf-8')).hexdigest()
            cache_file = _cache_root / f"{func.__name__}_{cache_key}.pkl"

            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_ts, cached_result = pickle.load(f)
                    if ttl is None or (time.time() - cached_ts) < ttl:
                        return cached_result
                except Exception:
                    pass

            result = func(*args, **kwargs)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump((time.time(), result), f)
            except Exception as e:
                wayne_print(f"[disk_cache] 写入缓存失败: {e}", "yellow")
            return result

        def cache_clear():
            for f in _cache_root.glob(f"{func.__name__}_*.pkl"):
                f.unlink(missing_ok=True)
            wayne_print(f"[disk_cache] 已清除 {func.__name__} 的所有缓存", "cyan")

        wrapper.cache_clear = cache_clear
        wrapper.cache_dir = str(_cache_root)
        return wrapper

    return decorator


def parallel_map(
        func: Callable,
        items: Iterable,
        n_workers: int = 8,
        mode: str = 'thread',
        show_progress: bool = False,
        desc: str = '',
        timeout: Optional[float] = None
) -> List:
    """
    对列表中每个元素并发执行函数，保序返回结果。

    :param func: 要执行的函数，签名为 ``func(item) -> result``。
    :param items: 输入序列。
    :param n_workers: 并发工作线程/进程数。
    :param mode: ``'thread'``（I/O 密集）或 ``'process'``（CPU 密集）。
    :param show_progress: 是否显示 tqdm 进度条（需安装 tqdm）。
    :param desc: 进度条描述文字。
    :param timeout: 单个任务超时秒数（None 为不限）。
    :return: 与 ``items`` 等长、保序的结果列表。

    Usage::

        results = parallel_map(download, url_list, n_workers=16, show_progress=True)

        results = parallel_map(heavy_compute, data, n_workers=4, mode='process')
    """
    items_list = list(items)
    if not items_list:
        return []

    executor_cls = ThreadPoolExecutor if mode == 'thread' else ProcessPoolExecutor
    results: List[Any] = [None] * len(items_list)

    _progress = tqdm(total=len(items_list), desc=desc or func.__name__) if show_progress else None

    try:
        with executor_cls(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(func, item): idx
                for idx, item in enumerate(items_list)
            }
            for future in as_completed(future_to_idx, timeout=timeout):
                idx = future_to_idx[future]
                results[idx] = future.result()
                if _progress is not None:
                    _progress.update(1)
    finally:
        if _progress is not None:
            _progress.close()

    return results


def with_progress(desc: str = '', unit: str = 'it', total: Optional[int] = None):
    """
    装饰器：将被装饰函数的第一个可迭代参数包裹上 tqdm 进度条。
    若 tqdm 未安装则静默跳过，不影响函数执行。

    :param desc: 进度条描述文字，默认使用函数名。
    :param unit: 进度条单位字符串。
    :param total: 强制指定总量（None 时自动从 ``len()`` 推断）。

    Usage::

        @with_progress(desc="处理帧")
        def process_frames(frames: list):
            for frame in frames:   # frames 已被 tqdm 包裹
                ...

        # 直接包裹迭代器
        for item in progress_iter(my_list, desc="下载"):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_args = list(args)
            if new_args:
                first = new_args[0]
                if hasattr(first, '__iter__') and not isinstance(first, (str, bytes)):
                    _total = total
                    if _total is None and hasattr(first, '__len__'):
                        _total = len(first)
                    new_args[0] = tqdm(first, desc=desc or func.__name__, total=_total, unit=unit)
            return func(*new_args, **kwargs)

        return wrapper

    return decorator


def progress_iter(iterable: Iterable, desc: str = '', total: Optional[int] = None, unit: str = 'it') -> Iterable:
    """
    将任意可迭代对象包裹上 tqdm 进度条并返回。
    若 tqdm 未安装则原样返回，不抛异常。

    :param iterable: 输入可迭代对象。
    :param desc: 进度条描述文字。
    :param total: 强制指定总量（None 时自动从 ``len()`` 推断）。
    :param unit: 进度条单位字符串。

    Usage::

        for img_path in progress_iter(paths, desc="加载图片"):
            process(img_path)
    """
    _total = total
    if _total is None and hasattr(iterable, '__len__'):
        _total = len(iterable)
    return tqdm(iterable, desc=desc, total=_total, unit=unit)


class FileWatcher:
    """
    监视文件或目录的变化，变化时触发回调（基于 watchdog，事件驱动）。

    :param path: 要监视的文件或目录路径。
    :param on_created: 新文件创建时的回调 ``(file_path: str) -> None``。
    :param on_modified: 文件内容修改时的回调 ``(file_path: str) -> None``。
    :param on_deleted: 文件删除时的回调 ``(file_path: str) -> None``。
    :param extensions: 只关注指定后缀名（如 ``['.jpg', '.png']``），空列表表示监听所有文件。
    :param recursive: 是否递归监视子目录（仅在 ``path`` 为目录时有效）。

    支持上下文管理器协议::

        with FileWatcher("/data/logs", on_modified=handle_mod, extensions=[".log"]) as w:
            time.sleep(60)

    也可手动控制::

        w = FileWatcher("/tmp/results", on_created=on_new_file)
        w.start()
        ...
        w.stop()
    """

    def __init__(
            self,
            path: str,
            on_created: Optional[Callable[[str], None]] = None,
            on_modified: Optional[Callable[[str], None]] = None,
            on_deleted: Optional[Callable[[str], None]] = None,
            extensions: Optional[List[str]] = None,
            recursive: bool = False
    ):
        self.path = os.path.abspath(path)
        self.on_created = on_created
        self.on_modified = on_modified
        self.on_deleted = on_deleted
        self.extensions = [ext.lower().lstrip('.') for ext in (extensions or [])]
        self.recursive = recursive
        self._observer: Optional[Observer] = None

    def _should_notify(self, path: str) -> bool:
        if not self.extensions:
            return True
        return any(path.lower().endswith('.' + ext) for ext in self.extensions)

    def start(self) -> 'FileWatcher':
        """在后台线程中启动监视，返回 self 支持链式调用。"""
        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory and watcher._should_notify(event.src_path):
                    if watcher.on_created:
                        watcher.on_created(event.src_path)

            def on_modified(self, event):
                if not event.is_directory and watcher._should_notify(event.src_path):
                    if watcher.on_modified:
                        watcher.on_modified(event.src_path)

            def on_deleted(self, event):
                if not event.is_directory and watcher._should_notify(event.src_path):
                    if watcher.on_deleted:
                        watcher.on_deleted(event.src_path)

        watch_path = self.path if os.path.isdir(self.path) else os.path.dirname(self.path)
        self._observer = Observer()
        self._observer.schedule(_Handler(), watch_path, recursive=self.recursive)
        self._observer.start()
        wayne_print(f"[FileWatcher] 开始监视: {self.path}", "cyan")
        return self

    def stop(self) -> None:
        """停止监视并等待后台线程退出。"""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
        wayne_print(f"[FileWatcher] 已停止监视: {self.path}", "cyan")

    def __enter__(self) -> 'FileWatcher':
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


def wayne_print_table(
        data: List[List],
        headers: Optional[List[str]] = None,
        align: Optional[List[str]] = None,
        title: Optional[str] = None,
        border: str = 'unicode',
        color: str = 'default',
        bold_header: bool = False,
) -> None:
    """
    在终端打印带边框的格式化表格，支持颜色和表头加粗。

    :param data: 二维列表，每个子列表为一行数据。
    :param headers: 列名列表。``None`` 表示无表头。
    :param align: 每列对齐方式列表，可选 ``'left'``、``'right'``、``'center'``，
                  默认全部左对齐。
    :param title: 表格标题，显示在顶部居中位置。
    :param border: 边框风格，``'unicode'``（默认，带框线）或 ``'simple'``（纯 ASCII）。
    :param color: 整体颜色，与 ``wayne_print`` 颜色名一致（``'default'``、``'red'``、
                  ``'green'``、``'yellow'``、``'blue'``、``'magenta'``、``'cyan'``、``'white'``）。
    :param bold_header: 表头是否加粗，默认 ``False``。

    Usage::

        wayne_print_table(
            data=[["ResNet50", "92.3%", "45ms"],
                  ["MobileNet", "88.1%", "12ms"]],
            headers=["模型", "准确率", "延迟"],
            align=["left", "right", "right"],
            title="模型对比",
            color="cyan",
            bold_header=True,
        )
    """
    if not data and not headers:
        return

    _COLORS = {
        "default": "\033[0m",
        "red":     "\033[31m",
        "green":   "\033[32m",
        "yellow":  "\033[33m",
        "blue":    "\033[34m",
        "magenta": "\033[35m",
        "cyan":    "\033[36m",
        "white":   "\033[37m",
    }
    _BOLD  = "\033[1m"
    _RESET = "\033[0m"
    color_code = _COLORS.get(color, _COLORS["default"])

    rows = [[str(cell) for cell in row] for row in data]
    head = [str(h) for h in headers] if headers else []

    ncols = max((len(row) for row in rows), default=0)
    if head:
        ncols = max(ncols, len(head))

    for row in rows:
        while len(row) < ncols:
            row.append('')
    if head:
        while len(head) < ncols:
            head.append('')

    # 宽度按原始字符串计算（不含 ANSI 码）
    col_widths = [0] * ncols
    for col in range(ncols):
        if head:
            col_widths[col] = max(col_widths[col], len(head[col]))
        for row in rows:
            col_widths[col] = max(col_widths[col], len(row[col]))

    aligns = list(align or []) + ['left'] * ncols
    aligns = aligns[:ncols]

    def _fmt(text: str, width: int, a: str) -> str:
        if a == 'right':
            return text.rjust(width)
        if a == 'center':
            return text.center(width)
        return text.ljust(width)

    if border == 'unicode':
        tl, tr, bl, br = '┌', '┐', '└', '┘'
        lj, rj, tj, bj, cj = '├', '┤', '┬', '┴', '┼'
        h_ch, v_ch = '─', '│'
    else:
        tl = tr = bl = br = lj = rj = tj = bj = cj = '+'
        h_ch, v_ch = '-', '|'

    sep = tj if border == 'unicode' else '+'

    def _hline(left, right, inner) -> str:
        return left + inner.join(h_ch * (w + 2) for w in col_widths) + right

    top_line = _hline(tl, tr, sep)
    mid_line = _hline(lj, rj, cj)
    bot_line = _hline(bl, br, bj if border == 'unicode' else '+')

    total_width = sum(col_widths) + 3 * ncols + 1
    lines: List[str] = []

    if title:
        lines.append(_hline(tl, tr, h_ch))
        lines.append(f"{v_ch} {title.center(total_width - 4)} {v_ch}")
        lines.append(_hline(lj, rj, sep) if (head or rows) else bot_line)
    else:
        lines.append(top_line)

    def _row_line(cells: List[str], is_header: bool = False) -> str:
        parts = []
        for c in range(ncols):
            cell = _fmt(cells[c], col_widths[c], aligns[c])
            if is_header and bold_header:
                cell = f"{_BOLD}{cell}{_RESET}{color_code}"
            parts.append(f" {cell} ")
        return v_ch + v_ch.join(parts) + v_ch

    if head:
        lines.append(_row_line(head, is_header=True))
        lines.append(mid_line)

    for row in rows:
        lines.append(_row_line(row))

    lines.append(bot_line)

    output = '\n'.join(lines)
    if color != 'default':
        output = f"{color_code}{output}{_RESET}"
    print(output)
