import functools
import logging
import os
import random
import threading
import time
from typing import *

import matplotlib.pyplot as plt

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
        plt.get_current_fig_manager().window.showMaximized()
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


def list_all_files(root: str, keys_and=[], keys_or=[], outliers=[], full_path=False):
    """
    列出某个文件下所有文件的全路径

    Author:   wangye
    Datetime: 2019/4/16 18:03

    :param root: 根目录
    :param keys_and: 必须出现的关键字
    :param keys_or: 至少出现一次的关键字
    :param outliers: 所有排除关键字
    :param full_path: 是否返回全路径，True为全路径
    :return:
            所有根目录下包含关键字的文件全路径
    """
    _files = []
    _list = os.listdir(root)
    for i, _list_i in enumerate(_list):
        path = os.path.join(root, _list_i)
        if os.path.isdir(path):
            _files.extend(list_all_files(path, keys_and, keys_or, outliers, full_path))
        if (os.path.isfile(path)
                and all(k in path for k in keys_and)
                and (not keys_or or any(k in path for k in keys_or))
                and not any(o in path for o in outliers)):
            _files.append(os.path.abspath(path) if full_path else path)
    return _files


def count_file_lines(file):
    def block(f_: 'File', size_=65536):
        while True:
            b = f_.read(size_)
            if not b: break
            yield b

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(bl.count('\n') for bl in block(f))


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
