import collections
import ctypes
import ctypes.wintypes
import functools
import itertools
import logging
import operator as op
import os
import re
import sys
import time
import traceback
import warnings
import xml.etree.ElementTree as ET

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pygame
except ImportError:
    logging.warn('Try pip install pygame')
import scipy as sp

try:
    import win32api
    import win32clipboard
    import win32con
    import win32gui
    import win32process
except ImportError:
    logging.warn('Try install pywin32 on your computer from below website:\n'
                 'http://www.lfd.uci.edu/~gohlke/pythonlibs')

try:
    from PIL import Image
except ImportError:
    logging.warn('Try pip install Pillow')

try:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
except ImportError:
    logging.warn('Try pip install pyqt5-tools')

try:
    from pykeyboard import PyKeyboard
    from pymouse import PyMouse
except ImportError:
    logging.warn('Try pip install pyuserinput and also have pyHook installed on your computer from below website:\n'
                 'http://www.lfd.uci.edu/~gohlke/pythonlibs')

from scipy import signal

"""
table of content:
1. Useful tools:
    a. (decorator) func_timer
    b. (decorator) maximize_figure
    c. (function) list_all_files
    d. (class) GlobalHotKeys
    e. (class) GuiOperation
2. Data processing:
    a. (function) peak_det
    b. (function) butter_bandpass_filter
    c. (class) FindLocalExtremeValue
    d. (class) DataProcessing
    e. (class) ExtractRollingData
    f. (class) CurveSimilarity
    g. (function) find_extreme_value_in_sliding_window
3. mathematics:
    a. (function) find_all_non_negative_integer_solutions
    b. (function) get_all_factors
    c. (function) digit_counts
4. DataStructure:
    a. (class) ConditionTree
    b. (class) XmlIO
"""


# Wayne:

def func_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.time()
        r = func(*args, **kw)
        print('%s excute in %.3f s' % (func.__name__, (time.time() - start)))
        return r

    return wrapper


def maximize_figure(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        plt.get_current_fig_manager().window.showMaximized()
        return ret

    return wrapper


def list_all_files(root: str, keys=[], outliers=[], full_path=False):
    """
    列出某个文件下所有文件的全路径

    Author:   wangye
    Datetime: 2019/4/16 18:03

    :param root: 根目录
    :param keys: 所有关键字
    :param outliers: 所有排除关键字
    :param full_path: 是否返回全路径，True为全路径
    :return:
            所有根目录下包含关键字的文件全路径
    """
    _files = []
    _list = os.listdir(root)
    for i in range(len(_list)):
        path = os.path.join(root, _list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path, keys, outliers, full_path))
        if os.path.isfile(path) \
                and all([k in path for k in keys]) \
                and not any([o in path for o in outliers]):
            _files.append(os.path.abspath(path) if full_path else path)
    return _files


class GlobalHotKeys:
    """
    Register a key using the register() method, or using the @register decorator
    Use listen() to start the message pump

    Author:   wangye
    Datetime: 2019/5/17 11:00

    Example:
    g = GlobalHotKeys()

    @GlobalHotKeys.register(GlobalHotKeys.VK_F1, GlobalHotKeys.MOD_SHIFT)
    def shift_f1():
        print('hello world')

    # Q and ctrl will stop message loop
    GlobalHotKeys.register(GlobalHotKeys.VK_Q, 0, False)
    GlobalHotKeys.register(GlobalHotKeys.VK_C, GlobalHotKeys.MOD_CTRL, False)

    # start main loop
    GlobalHotKeys.listen()
    """

    key_mapping = []
    user32 = ctypes.windll.user32

    try:
        MOD_ALT = win32con.MOD_ALT
        MOD_CTRL = win32con.MOD_CONTROL
        MOD_CONTROL = win32con.MOD_CONTROL
        MOD_SHIFT = win32con.MOD_SHIFT
        MOD_WIN = win32con.MOD_WIN
    except:
        pass

    def __init__(self):
        self._include_alpha_numeric_vks()
        self._include_defined_vks()

    @classmethod
    def _include_defined_vks(self):
        for item in win32con.__dict__:
            item = str(item)
            if item[:3] == 'VK_':
                setattr(self, item, win32con.__dict__[item])

    @classmethod
    def _include_alpha_numeric_vks(self):
        for key_code in (list(range(ord('A'), ord('Z') + 1)) + list(range(ord('0'), ord('9') + 1))):
            setattr(self, 'VK_' + chr(key_code), key_code)

    @classmethod
    def register(self, vk, modifier=0, func=None):
        """
        vk is a windows virtual key code
         - can use ord('X') for A-Z, and 0-1 (note uppercase letter only)
         - or win32con.VK_* constants
         - for full list of VKs see: http://msdn.microsoft.com/en-us/library/dd375731.aspx

        modifier is a win32con.MOD_* constant

        func is the function to run.  If False then break out of the message loop
        """

        # Called as a decorator?
        if func is None:
            def register_decorator(f):
                self.register(vk, modifier, f)
                return f

            return register_decorator
        else:
            self.key_mapping.append((vk, modifier, func))

    @classmethod
    def listen(self):
        """
        Start the message pump
        """

        for index, (vk, modifiers, func) in enumerate(self.key_mapping):
            # cmd 下没问题, 但是在服务中运行的时候抛出异常
            if not self.user32.RegisterHotKey(None, index, modifiers, vk):
                raise Exception('Unable to register hot key: ' + str(vk) + ' error code is: ' + str(
                    ctypes.windll.kernel32.GetLastError()))

        try:
            msg = ctypes.wintypes.MSG()
            while self.user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
                if msg.message == win32con.WM_HOTKEY:
                    (vk, modifiers, func) = self.key_mapping[msg.wParam]
                    if not func:
                        break
                    func()

                self.user32.TranslateMessage(ctypes.byref(msg))
                self.user32.DispatchMessageA(ctypes.byref(msg))

        finally:
            for index, (vk, modifiers, func) in enumerate(self.key_mapping):
                self.user32.UnregisterHotKey(None, index)


class GuiOperation:
    """
    Using package pywin32 to do some gui operations.

    Author:   wangye
    Datetime: 2019/5/18 22:00

    example:
        gui = GuiOperation()
    notepad = gui.find_window('tt')[0]
    gui.bring_to_top(notepad)
    time.sleep(2)
    st_test_software = gui.find_window('ST')[0]
    gui.bring_to_top(st_test_software)

    for h in gui.get_child_windows(st_test_software):
        ttl, cls = gui.get_windows_attr(h)
        print(ttl, cls)
        if '&Read' in ttl:
            print('-------')
            left, top, right, bottom = gui.get_window_rect(h)
            print(left, top, right, bottom)
            gui.mouse.move((left + right) // 2, (top + bottom) // 2)
            time.sleep(0.2)
            gui.change_window_name(h, 'shit')

    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name'])
        except psutil.NoSuchProcess:
            pass
        else:
            print(pinfo)
    """

    def __init__(self):
        self.mouse = PyMouse()
        self.keyboard = PyKeyboard()
        # 系统常量，标识最高权限打开一个进程
        PROCESS_ALL_ACCESS = (0x000F0000 | 0x00100000 | 0xFFF)

    def find_window(self, *key):
        titles = set()

        def loop_windows(hwnd, _):
            if win32gui.IsWindow(hwnd) \
                    and win32gui.IsWindowEnabled(hwnd) \
                    and win32gui.IsWindowVisible(hwnd):
                titles.add(win32gui.GetWindowText(hwnd))

        win32gui.EnumWindows(loop_windows, 0)
        wanted_window_handles = [win32gui.FindWindow(None, t) for t in titles if all([k in t for k in key])]
        return wanted_window_handles

    def get_windows_attr(self, hwnd):
        if hwnd:
            return win32gui.GetWindowText(hwnd), \
                   win32gui.GetClassName(hwnd)
        return '', ''

    def maximize_window(self, hwnd):
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

    def bring_to_top(self, hwnd):
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.BringWindowToTop(hwnd)
            try:
                win32gui.SetForegroundWindow(hwnd)
            except:
                pass

    def close_window(self, hwnd):
        if hwnd:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

    def get_window_rect(self, hwnd):
        return win32gui.GetWindowRect(hwnd)

    def get_child_windows(self, hwnd):
        hwnd_child_list = set()
        stack = [hwnd]
        while stack:
            s = stack.pop()
            if s in hwnd_child_list:
                continue
            sub_hwnd_child_list = []
            try:
                win32gui.EnumChildWindows(
                    s, lambda h, p: p.append(h), sub_hwnd_child_list
                )
                [stack.append(sh) for sh in sub_hwnd_child_list]
                [hwnd_child_list.add(sh) for sh in sub_hwnd_child_list]
            except:
                continue
        return list(hwnd_child_list)

    def change_window_name(self, hwnd, new_name):
        win32api.SendMessage(hwnd, win32con.WM_SETTEXT, 0, new_name)


def peak_det(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    python version:
    https://gist.github.com/endolith/250860

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        print('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        print('Input argument delta must be a scalar')

    if delta <= 0:
        print('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def butter_bandpass_filter(x, wn=0.2):
    """
    ButterWorth 低通滤波器

    Author:   wangye
    Datetime: 2019/4/24 16:00

    :param x: 待滤波矩阵, 2d-array
    :param wn: wn系数，权衡失真和光滑
    :return: 滤波后矩阵, 2d-array
    """
    b, a = signal.butter(N=2, Wn=wn, btype='low')
    return np.apply_along_axis(
        lambda y: signal.filtfilt(b, a, y),
        0, x
    )


class FindLocalExtremeValue:
    """
    本类用于寻找局部极值

    Author:   wangye
    Datetime: 2019/4/16 15:52
    """

    def __init__(self,
                 local_window=30,
                 min_valid_slp_thd=7,
                 slp_step=3,
                 max_update_num=4
                 ):
        """
        :param local_window: 表示寻找极值的窗长，仅寻找数据x中最后local_window
                            长度的数据相对于x最后一帧的极值。
        :param min_valid_slp_thd: 要成为合理的局部极值，至少要与被比较者相差的值
        :param slp_step: 找当前值和极值之间的最值斜率，斜率计算的步长
        :param max_update_num: local peak连续max_update_num帧不更新则停止计算
        """
        self.local_window = local_window
        self.min_valid_slp_thd = min_valid_slp_thd
        self.min_slp_step = slp_step
        self.max_slp_step = slp_step
        self.max_update_num = max_update_num

    def run(self, x: np.ndarray):
        """ 寻找局部极值

        这个函数用来找局部极值，具体来说，是寻找x的最后local_window帧中
        相对于x最后一帧的局部极值。
        当x的最后一帧相对于附近的数据比较小时，寻找的就是local peak，
        当x的最后一帧相对于附近的数据比较大时，寻找的就是local base

        :param x: x是一维nd-array数据,需要保证x的数据长度大于等于local_window+2
                (需要额外两帧辅助确认是局部极值)

        :return:
                总共返回四个值：
                第一个local_diff是指x的最后一帧与局部极值之间值的差
                第二个local_gap是指x的最后一帧与局部极值之间帧数的差
                第三个min_slope是指x的最后一帧与局部极值之间，最小的斜率
                第四个max_slope是指x的最后一帧与局部极值之间，最大的斜率
                第五个local_extreme_real_value是指局部极值实际的值
        """

        x = x.astype(np.float)
        m = x.shape[0]
        assert m >= max(self.local_window + 2, 4)

        local_diff, local_gap, min_slope, max_slope, local_extreme_real_value = 0, 0, np.inf, -np.inf, 0
        local_x = 0
        update_num = 0
        diff_queue = collections.deque(maxlen=3)
        diff_queue.append(x[-1] - x[-2])
        diff_queue.append(x[-1] - x[-3])
        slope_queue = collections.deque()
        for j in range(1, self.local_window):
            if j + 2 * self.min_slp_step <= m:
                slope_right = sum(x[-j - self.min_slp_step:-j]) if j <= self.min_slp_step else slope_queue.popleft()
                slope_left = sum(x[-j - 2 * self.min_slp_step:-j - self.min_slp_step])
                slope_queue.append(slope_left)
                cur_slope = slope_right - slope_left
                min_slope, max_slope = min(min_slope, cur_slope), max(max_slope, cur_slope)
            diff_queue.append(x[-1] - x[-1 - j - 2])
            a, b, c = diff_queue
            if 0 == local_x:
                if max(abs(a), abs(b), abs(c)) > self.min_valid_slp_thd:
                    local_x = a
                    local_gap = j
                    local_extreme_real_value = x[-j]
                continue
            if (0 < local_x < a) or (a < local_x < 0):
                local_x = a
                local_gap = j
                update_num = 0
                local_extreme_real_value = x[-1 - j]
            else:
                update_num += 1
            if update_num > self.max_update_num:
                break
        local_diff = local_x
        min_slp, max_slp = min_slope / self.min_slp_step ** 2, max_slope / self.min_slp_step ** 2
        return local_diff, local_gap, min_slp, max_slp, local_extreme_real_value


class DataProcessing:
    """
    用于从rawdata中提取force

    Author:   wangye
    Datetime: 2019/5/17 11:17

    example:
    rawdata = np.vstack((
        np.tile(rawdata[0], [200, 1]),
        rawdata,
        np.tile(rawdata[-1], [200, 1])
    ))
    dp = DataProcessing(rawdata, f)
    dp.pre_process()
    dp.limiting_filter()
    dp.calc_moving_avg()
    dp.baseline_removal()
    dp.calc_energy()
    dp.calc_flag()
    dp.calc_force()
    dp.show_fig()
    force = dp.force
    if not os.path.exists('result'):
        os.mkdir('result')
    np.savetxt('result\\' +
               (f[:f.rindex('.')].replace('\\', '_')
                if not dp.simple_file_name
                else f[f.rindex('\\') + 1:f.rindex('.')]
                ) + '_result.txt', force, fmt='%8.2f'
               )
    """

    def __init__(self, data, filename):
        self.data = data
        self.filename = filename
        self.base = data
        self.force_signal = np.zeros_like(data)
        self.energy = None
        self.flag = None
        self.force = None
        self.tds = None
        self.energy_peak = None
        self.energy_valley = None
        self.limiting_thd = 1000
        self.limiting_step_ratio = 0.4
        self.mov_avg_len = 5
        self.sigma_wave = 10
        self.sigma_tsunami = 4
        self.alpha = 3
        self.beta = 5
        self.energy_thd = 30
        self.energy_thd_decay_coef = 0.9
        self.leave_eng_peak_ratio = 0.5
        self.energy_peak_detect_delta = 50
        self.min_td_time = 50
        self.min_tu_time = 50
        self.step_u = 0
        self.step_l = 0
        self.bef = 50
        self.aft = 50
        self.avg = 10
        self.simple_file_name = True

    @func_timer
    def pre_process(self):
        self.data = self.data - self.data[0]
        if np.ndim(self.data) == 1:
            self.data = self.data[:, np.newaxis]
        # self.data = -self.data

    @func_timer
    def limiting_filter(self):
        output = np.zeros_like(self.data)
        output[0] = self.data[0]
        for ii in range(len(self.data) - 1):
            for jj in range(np.shape(self.data)[1]):
                if np.abs(self.data[ii + 1, jj] - output[ii, jj]) >= self.limiting_thd:
                    output[ii + 1, jj] = output[ii, jj] + (
                            self.data[ii + 1, jj] - output[ii, jj]
                    ) * self.limiting_step_ratio
                else:
                    output[ii + 1, jj] = self.data[ii + 1, jj]
        self.data = output

    @func_timer
    def calc_moving_avg(self):
        self.data = np.array([
            self.data[:ii + 1].mean(0) if ii <= self.mov_avg_len else
            self.data[ii - (self.mov_avg_len - 1):ii + 1].mean(0) for ii in range(len(self.data))
        ])

    def _baseline_removal_single(self, data):
        # base = peakutils.baseline(data,100)
        base = np.copy(data)
        for ii in range(1, len(data)):
            base[ii] = base[ii - 1] + \
                       (data[ii] - data[ii - 1]) * \
                       np.exp(-(data[ii] - data[ii - 1]) ** 2 / self.sigma_wave) + \
                       (data[ii - 1] - base[ii - 1]) * \
                       np.exp(-np.abs(data[ii - 1] - base[ii - 1]) / self.sigma_tsunami)
        return base

    @func_timer
    def baseline_removal(self, do_baseline_tracking=False):
        if do_baseline_tracking:
            self.base = np.apply_along_axis(
                lambda x: self._baseline_removal_single(x), 0, self.data
            )
            self.force_signal = self.data - self.base
        else:
            self.force_signal = self.data

    @func_timer
    def calc_energy(self):
        self.force_signal = np.array(self.force_signal)
        if self.force_signal.ndim == 1:
            self.force_signal = self.force_signal[:, np.newaxis]
        m = self.force_signal.shape[0]
        energy_n = np.zeros((m, 1))
        for ii in range(m - self.alpha - self.beta):
            energy_n[ii + self.alpha + self.beta] = \
                1 / self.alpha * \
                np.sum(np.abs(
                    self.force_signal[ii + self.beta:ii + self.beta + self.alpha, :] -
                    self.force_signal[ii:ii + self.alpha, :]
                ))
            # diff_mat = self.force_signal[ii + self.beta:ii + self.beta + self.alpha, :] - self.force_signal[ii:ii + self.alpha, :]
            # diff_mat_max_sub = np.array(np.where(np.abs(diff_mat) == np.max(np.abs(diff_mat))))
            # diff_mat_max_sub = diff_mat_max_sub[:, 0]
            # max_sign = np.sign(diff_mat[diff_mat_max_sub[0], diff_mat_max_sub[1]])
            # energy_n[ii + self.alpha + self.beta] *= max_sign
        self.energy = energy_n.T[0]
        self.energy_peak, self.energy_valley = peak_det(self.energy, self.energy_peak_detect_delta)
        if len(self.energy_peak) > 0 and len(self.energy_peak) * 40 < len(self.data):
            self.energy_thd = min(self.energy_peak[:, 1]) * 0.8

    @staticmethod
    def _update_flag_status(f, r, t, tdf, tuf):
        # t is tvalue < thd
        f_ = f ^ r and f or not (f ^ r) and not (f ^ t)
        f_ = not f and f_ and tuf or f and (f_ or not tdf)
        r_ = not r and f and t or r and not (not f and t)
        return f_, r_

    @func_timer
    def calc_flag(self):
        self.flag = np.zeros(self.energy.shape, dtype=np.bool)
        ready = False
        touch_down_frm = 0
        touch_up_frm = self.min_tu_time + 1
        for ii in range(1, self.flag.shape[0]):
            f = bool(self.flag[ii - 1])
            t = (not f and (self.energy[ii] < self.energy_thd)) or \
                (f and (self.energy[ii] <
                        max(self.energy_thd * self.energy_thd_decay_coef,
                            max(self.energy[ii - touch_down_frm:ii + 1]) * self.leave_eng_peak_ratio
                            )))
            # (f and (self.energy[ii] < self.energy_thd * 1.1)) or \
            touch_down_frm = touch_down_frm + 1 if self.flag[ii - 1] else 0
            touch_up_frm = touch_up_frm + 1 if not self.flag[ii - 1] else 0
            tdf = touch_down_frm >= self.min_td_time
            tuf = touch_up_frm >= self.min_tu_time
            self.flag[ii], ready = self._update_flag_status(f, ready, t, tdf, tuf)
        self.flag = np.array(self.flag, dtype=np.int)

    @func_timer
    def show_fig(self):
        plt.rcParams['font.family'] = 'FangSong'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(2, 1, sharex='all')
        fig.set_size_inches(60, 10)
        ax[0].plot(self.data)
        ax[0].plot(self.flag * (np.max(self.data) - np.min(self.data)) + np.min(self.data), '--')
        if self.tds is not None:
            for i in range(self.tds.shape[0]):
                try:
                    ax[0].plot([self.tds[i] - self.bef for _ in range(np.shape(self.data)[1])],
                               self.data[self.tds[i] - self.bef, :], 'k.', markersize=10)
                    ax[0].plot([self.tds[i] + self.aft for _ in range(np.shape(self.data)[1])],
                               self.data[self.tds[i] + self.aft, :], 'k.', markersize=10)
                except:
                    continue
        ax[0].set_title('rawdata')
        ax[0].legend(tuple([''.join(('rawdata ch', str(ii + 1))) for ii in range(np.shape(self.data)[1])]))
        ax[0].set_ylabel('ADC')
        ax[1].plot(self.energy)
        if len(self.energy_peak) > 0:
            ax[1].plot(self.energy_peak[:, 0], self.energy_peak[:, 1], '.', markersize=20)
        ax[1].plot(self.flag * (np.max(self.energy) - np.min(self.energy)) + np.min(self.energy), '--')
        ax[1].hlines(self.energy_thd, 0, self.data.shape[0], linestyles='--')
        ax[1].hlines(self.energy_thd * self.energy_thd_decay_coef, 0, self.data.shape[0], linestyles='--')
        ax[1].set_title(self.filename)
        ax[1].set_xlabel('Time Series')
        ax[1].set_ylabel('ADC')
        if len(self.energy_peak) > 0:
            ax[1].legend(['energy', 'energy peak', 'touch flag', 'energy threshold'])
        else:
            ax[1].legend(['energy', 'touch flag', 'energy threshold'])
        plt.show()

    @func_timer
    def calc_force(self):
        self.tds = np.array(np.where(np.diff(self.flag) == 1))[0]
        if np.ndim(self.force_signal) == 1:
            self.force_signal = self.force_signal[:, np.newaxis]
        params = np.zeros((self.force_signal.shape[1], self.tds.shape[0]))
        for ii in range(params.shape[1]):
            params[:, ii] = np.mean(
                self.force_signal[self.tds[ii] + self.aft:self.tds[ii] + self.aft + self.avg, :], 0
            ) - np.mean(
                self.force_signal[self.tds[ii] - self.bef - self.avg:self.tds[ii] - self.bef, :], 0
            )
        self.force = params.T


class ExtractRollingData:
    """
    用于从滑动数据中提取数据

    Author:   wangye
    Datetime: 2019/5/22 02:41

    example:
    rd = ExtractRollingData(rawdata, f)
    rd.simple_remove_base()
    rd.find_peak()
    rd.filt_data()
    rd.find_filt_peak()
    rd.show_fig()

    # if not os.path.exists(r'result'):
    #     os.mkdir(r'result')
    # print(os.getcwd())
    # np.savetxt(r'result\\' +
    #            (f[:f.rindex('.')].replace('\\', '_')
    #             if not rd.simple_file_name
    #             else f[f.rindex('\\') + 1:f.rindex('.')]
    #             ) + '_result.txt', rd.rawdata, fmt='%8.2f'
    #            )
    """

    def __init__(self, data, f):
        self.rawdata = data
        self.filename = f
        self.peaks = np.empty((0, 0))
        self.remove_base_window = 50
        self.find_peak_thd = 250
        self.extend_data_length = 300
        self.filtered_data = np.copy(data)
        self.simple_file_name = True

    def simple_remove_base(self):
        def _remove_base(arr, window):
            starts = arr[:window + 1]
            ends = arr[-(window + 1):]
            """
            正态性检验:

            1. Shapiro-Wilk test
            方法：scipy.stats.shapiro(x)
            官方文档：SciPy v1.1.0 Reference Guide
            参数：x - 待检验数据
            返回：W - 统计数；p-value - p值

            2. scipy.stats.kstest
            方法：scipy.stats.kstest (rvs, cdf, args = ( ), N = 20, alternative ='two-sided', mode ='approx')
            官方文档：SciPy v0.14.0 Reference Guide
            参数：rvs - 待检验数据，可以是字符串、数组；
            cdf - 需要设置的检验，这里设置为 norm，也就是正态性检验；
            alternative - 设置单双尾检验，默认为 two-sided
            返回：W - 统计数；p-value - p值

            3. scipy.stats.normaltest
            方法：scipy.stats.normaltest (a, axis=0)
            该方法专门用来检验数据是否为正态性分布，官方文档的描述为：
            Tests whether a sample differs from a normal distribution.
            This function tests the null hypothesis that a sample comes from a normal distribution. It is based on D’Agostino and Pearson’s [R251], [R252] test that combines skew and kurtosis to produce an omnibus test of normality.
            官方文档：SciPy v0.14.0 Reference Guide
            参数：a - 待检验数据；axis - 可设置为整数或置空，如果设置为 none，则待检验数据被当作单独的数据集来进行检验。该值默认为 0，即从 0 轴开始逐行进行检验。
            返回：k2 - s^2 + k^2，s 为 skewtest 返回的 z-score，k 为 kurtosistest 返回的 z-score，即标准化值；p-value - p值

            4. Anderson-Darling test
            方法：scipy.stats.anderson (x, dist ='norm' )
            该方法是由 scipy.stats.kstest 改进而来的，可以做正态分布、指数分布、Logistic 分布、Gumbel 分布等多种分布检验。默认参数为 norm，即正态性检验。
            官方文档：SciPy v1.1.0 Reference Guide
            参数：x - 待检验数据；dist - 设置需要检验的分布类型
            返回：statistic - 统计数；critical_values - 评判值；significance_level - 显著性水平

            Case 0: The mean {\displaystyle \mu } \mu  and the variance {\displaystyle \sigma ^{2}} \sigma ^{2} are both known.
            Case 1: The variance {\displaystyle \sigma ^{2}} \sigma ^{2} is known, but the mean {\displaystyle \mu } \mu  is unknown.
            Case 2: The mean {\displaystyle \mu } \mu  is known, but the variance {\displaystyle \sigma ^{2}} \sigma ^{2} is unknown.
            Case 3: Both the mean {\displaystyle \mu } \mu  and the variance {\displaystyle \sigma ^{2}} \sigma ^{2} are unknown.

            Case	n	    15%	    10%	    5%	    2.5%	1%
            0	    >=5 	1.621	1.933	2.492	3.070	3.878
            1			    0.908	1.105	1.304	1.573
            2	    >=5		1.760	2.323	2.904	3.690
            3	    10	    0.514	0.578	0.683	0.779	0.926
                    20	    0.528	0.591	0.704	0.815	0.969
                    50	    0.546	0.616	0.735	0.861	1.021
                    100	    0.559	0.631	0.754	0.884	1.047
                    Inf 	0.576	0.656	0.787	0.918	1.092
            """
            test_result_starts = sp.stats.anderson(
                np.diff(starts), dist='norm'
            )
            test_result_ends = sp.stats.anderson(
                np.diff(starts), dist='norm'
            )
            # if window == 50
            standard50 = [0.546, 0.616, 0.735, 0.861, 1.021]
            assert all([x > y - 0.02 for x, y in zip(
                test_result_starts.critical_values,
                standard50
            )])
            assert all([x > y - 0.02 for x, y in zip(
                test_result_ends.critical_values,
                standard50
            )])
            x_start, y_start = window // 2, starts.mean()
            x_end, y_end = len(arr) - window // 2, ends.mean()
            assert x_start < x_end
            k = (y_end - y_start) / max(x_end - x_start, 1)
            b = y_start - x_start * k
            arr -= k * np.arange(len(arr)) + b
            return arr

        self.rawdata = np.apply_along_axis(
            lambda x: _remove_base(x, self.remove_base_window), 0, self.rawdata
        )

    def find_peak(self):
        self.peaks = []
        for i in range(self.rawdata.shape[1]):
            peak = peak_det(
                self.rawdata[:, i], self.find_peak_thd
            )[0]
            peak = peak[peak[:, 1].argmax()]
            self.peaks.append(peak)
        self.peaks = np.array(self.peaks)

    def filt_data(self):
        scale = np.array([max(0, self.peaks[:, 0].min() - self.extend_data_length),
                          min(self.rawdata.shape[0] - 1, self.peaks[:, 0].max() + self.extend_data_length)
                          ], dtype=np.int
                         )
        self.rawdata = self.rawdata[scale[0]:scale[1] + 1]
        self.peaks[:, 0] -= scale[0]
        self.filtered_data = butter_bandpass_filter(self.rawdata, 0.1)

    def find_filt_peak(self):
        self.filt_peaks = []
        for i in range(self.filtered_data.shape[1]):
            peak = peak_det(
                self.filtered_data[:, i], self.find_peak_thd
            )[0]
            peak = peak[peak[:, 1].argmax()]
            self.filt_peaks.append(peak)
        self.filt_peaks = np.array(self.filt_peaks)

        scale = np.array([max(0, self.filt_peaks[:, 0].min() - self.extend_data_length),
                          min(self.filt_peaks.shape[0] - 1, self.filt_peaks[:, 0].max() + self.extend_data_length // 2)
                          ], dtype=np.int
                         )
        self.rawdata = self.rawdata[scale[0]:scale[1] + 1]
        self.filtered_data = self.filtered_data[scale[0]:scale[1] + 1]
        self.peaks[:, 0] -= scale[0]
        self.filt_peaks[:, 0] -= scale[0]

    def show_fig(self):
        plt.rcParams['font.family'] = 'YouYuan'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure()
        fig.set_size_inches(60, 10)
        plt.plot(self.rawdata, 'r', linewidth=1)
        plt.plot(self.filtered_data, 'b--', linewidth=1)
        n = self.rawdata.shape[1]
        lgd = ['raw' + str(ch + 1) for ch in range(n)]
        lgd += ['filtered_raw' + str(ch + 1) for ch in range(n)]
        for i in range(len(self.peaks)):
            plt.plot(self.peaks[i, 0], self.peaks[i, 1], 'k.', markersize=10)
        lgd += ['peak' + str(ch + 1) for ch in range(n)]
        plt.legend(lgd)
        plt.xlabel('Time: frame')
        plt.ylabel('ADC')
        plt.title(self.filename)
        plt.show()


class CurveSimilarity:
    """
    用于计算曲线x和曲线y的相似度

    Author:   wangye
    Datetime: 2019/6/24 23:17

    https://zhuanlan.zhihu.com/p/69170491?utm_source=wechat_session&utm_medium=social&utm_oi=664383466599354368
    Example:
    # cs = CurveSimilarity()
    # s1 = np.array([1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1])
    # s2 = np.array([0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2])
    # s3 = np.array([0.8, 1.5, 0, 1.2, 0, 0, 0.6, 1, 1.2, 0, 0, 1, 0.2, 2.4, 0.5, 0.4])
    # print(cs.dtw(s1, s2))
    # print(cs.dtw(s1, s3))
    """

    @staticmethod
    def _check(var):
        if np.ndim(var) == 1:
            return np.reshape(var, (-1, 1))
        else:
            return np.array(var)

    def dtw(self, x, y, mode='global', *params):
        """
        计算曲线x和曲线y的DTW距离，其中global方法将全部数据用于DTW计算，local方法将一部分数据用于DTW计算
        x和y的行数代表数据长度，需要x和y的列数相同
        :param x: 第一条曲线
        :param y: 第二条曲线
        :param mode: 'local'用于计算局部DTW，'global'用于计算全局DTW
        :param params: 若为local DTW，则params为local的窗长
        :return: 曲线x和曲线y的DTW距离
        """
        x, y = self._check(x), self._check(y)
        m, n, p = x.shape[0], y.shape[0], x.shape[1]
        assert x.shape[1] == y.shape[1]
        distance = np.reshape(
            [(x[i, ch] - y[j, ch]) ** 2 for i in range(m) for j in range(n) for ch in range(p)],
            [m, n, p]
        )
        dp = np.zeros((m, n, p))
        dp[0, 0, 0] = distance[0, 0, 0]
        for i in range(1, m):
            dp[i, 0] = dp[i - 1, 0] + distance[i, 0]
        for j in range(1, n):
            dp[0, j] = dp[0, j - 1] + distance[0, j]
        for i in range(1, m):
            for j in range(1, n):
                for ch in range(p):
                    dp[i, j, ch] = min(
                        dp[i - 1, j - 1, ch],
                        dp[i - 1, j, ch],
                        dp[i, j - 1, ch]
                    ) + distance[i, j, ch]
        path = [[[m - 1, n - 1]] for _ in range(p)]
        for ch in range(p):
            pm, pn = m - 1, n - 1
            while pm > 0 and pn > 0:
                if pm == 0:
                    pn -= 1
                elif pn == 0:
                    pm -= 1
                else:
                    c = np.argmin([dp[pm - 1, pn, ch], dp[pm, pn - 1, ch], dp[pm - 1, pn - 1, ch]])
                    if c == 0:
                        pm -= 1
                    elif c == 1:
                        pn -= 1
                    else:
                        pm -= 1
                        pn -= 1
                path[ch].append([pm, pn])
            path[ch].append([0, 0])
        ret = [[(x[path[ch][pi][0], ch] - y[path[ch][pi][1], ch]) ** 2
                for pi in range(len(path[ch]))] for ch in range(p)]
        if mode == 'global':
            return np.squeeze([np.mean(r) for r in ret])
        elif mode == 'local':
            k = params[0]
            return np.squeeze([
                np.array(r)[np.argpartition(r, -k)[-k:]].mean() for r in ret
            ])


def find_extreme_value_in_sliding_window(data: list, k: int) -> list:
    """
    寻找一段数据中每个窗长范围内的最值，时间复杂度O(n)
    :param data: 数据 
    :param k: 窗长
    :return: 包含每个窗长最值的列表
    """
    minQueue = collections.deque()
    maxQueue = collections.deque()
    retMin, retMax = [], []
    for i, n in enumerate(data):
        while minQueue and n < data[minQueue[-1]]: minQueue.pop()
        while maxQueue and n > data[maxQueue[-1]]: maxQueue.pop()
        minQueue.append(i)
        maxQueue.append(i)
        if i - minQueue[0] >= k: minQueue.popleft()
        if i - maxQueue[0] >= k: maxQueue.popleft()
        retMin.append(data[minQueue[0]] if i >= k - 1 else data[i])
        retMax.append(data[maxQueue[0]] if i >= k - 1 else data[i])
    return retMin, retMax


def find_all_non_negative_integer_solutions(const_sum: int, num_vars: int):
    """
    求解所有满足x1+x2+...+x{num_vars}=const_sum的非负整数解

    Author:   wangye
    Datetime: 2019/4/16 17:49

    :param const_sum:
    :param num_vars:
    :return:
            所有非负整数解的list
    """
    if num_vars == 1:
        solution_list = [[const_sum]]
    else:
        solution_list = []
        for i in range(const_sum + 1):
            result = find_all_non_negative_integer_solutions(const_sum - i, num_vars - 1)
            for res in result:
                tmp = [i]
                tmp.extend(res)
                solution_list.append(tmp)
    return solution_list


def get_all_factors(n: int) -> list:
    """
    Return all factors of positive integer n.

    Author:   wangye
    Datetime: 2019/7/16 16:00

    :param n: A positive number
    :return: a list which contains all factors of number n
    """
    return list(set(reduce(
        list.__add__, ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))))


def digitCount(n, k):
    """
    Count the number of occurrences of digit k from 1 to n.
    Author:   wangye
    Datetime: 2019/7/18 14:49

    :param n:
    :param k:
    :return: The count.
    """
    N, ret, dig = n, 0, 1
    while n >= 1:
        m, r = divmod(n, 10)
        if r > k:
            ret += (m + 1) * dig
        elif r < k:
            ret += m * dig
        elif r == k:
            ret += m * dig + (N - n * dig + 1)
        n //= 10
        dig *= 10
    if k == 0:
        if N == 0:
            return 1
        else:
            return ret - dig // 10
    return ret


class ConditionTree:
    """
    逻辑条件树，用来存储逻辑关系的树

    Author:   wangye
    Datetime: 2019/7/31 16:17
    """

    def __init__(self, tag: str):
        self.tag = tag
        self.children = []
        self.attribute = dict()
        self.text = ''

    def append_by_path(self, path: list):
        """
        从路径导入
        :param path:
        :return:
        """
        if not path: return
        p = path[0]
        for ch in self.children:
            if p['tag'] == ch.tag:
                ch.append_by_path(path[1:])
                return
        child = ConditionTree(p['tag'])
        child.attribute = p['attrib']
        child.text = p['text']
        self.children.append(child)
        self.children[-1].append_by_path(path[1:])

    def find(self, nick_name: str) -> 'ConditionTree':
        if not self.children: return None
        for ch in self.children:
            if ch.tag == nick_name:
                return ch
            res = ch.find(nick_name)
            if res: return res
        return None

    def find_by_path(self, path: list) -> 'ConditionTree':
        cur = self
        while path:
            p = path.pop(0)
            for ch in cur.children:
                if p == ch.tag:
                    cur = ch
                    if not path:
                        return cur
                    break
            else:
                return None

    def print_path(self):
        def helper(_tree):
            if not _tree: return []
            if not _tree.children and _tree.text:
                return [[_tree.tag + ': ' + _tree.text]]
            return [[_tree.tag] + res for ch in _tree.children for res in helper(ch)]

        print('-*-*-*-*-*-*- start print tree -*-*-*-*-*-*-')
        [print(' -> '.join(h)) for h in helper(self)]
        print('-*-*-*-*-*-*- end print tree -*-*-*-*-*-*-')


class XmlIO:
    """
    用于xml文件的读取，返回ConditionTree数据结构

    Author:   wangye
    Datetime: 2019/7/31 16:24
    """

    def __init__(self, file_read='', file_write=''):
        self.file_read = file_read
        self.file_write = file_write

    def read(self) -> ConditionTree:
        tree = ET.parse(self.file_read)
        root = tree.getroot()

        def helper(_tree: ET.Element, _is_root=True):
            if not _tree: return [[{'tag': _tree.tag, 'attrib': _tree.attrib, 'text': _tree.text.strip('\t\n')}]]
            ret = [] if _is_root else [{'tag': _tree.tag, 'attrib': _tree.attrib, 'text': _tree.text.strip('\t\n')}]
            return [ret + res for ch in _tree for res in helper(ch, False)]

        c_tree = ConditionTree(root.tag)
        c_tree.attribute = root.attrib
        c_tree.text = root.text.strip('\t\n')
        [c_tree.append_by_path(p) for p in helper(root)]
        # c_tree.print_path()
        return c_tree

    def write(self, root_name, tree: ConditionTree):
        _tree = ET.ElementTree()
        _root = ET.Element(root_name)
        _tree._setroot(_root)

        def helper(etree, c_tree):
            if not c_tree.children and c_tree.kvp:
                for k, v in c_tree.kvp.items():
                    ET.SubElement(etree, k).text = v
            else:
                for ch in c_tree.children:
                    son = ET.SubElement(etree, ch.tag)
                    helper(son, ch)

        helper(_root, tree)
        self._indent(_root)
        _tree.write(self.file_write, 'utf-8', True)

    def _indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


# Leon:

class DataMark:
    """
    对数据进行手工标记，拆分数据成数据集

    Author:   wannuliang
    Datetime: 2019/4/17 09:15

    example:

    def data_mark_example():
        dm = DataMark()
        for f in list_all_files('..\\Debug-ETD1805_Power键', ['长按', '.txt'], []):
            dm.mark(f, 'data')
    """

    mark_flag = False
    fig = None
    ax = None
    mark_line = None
    start = end = 0
    flag = None
    rawData = None
    baseline = None
    forceSig = None
    forceFlag = None
    file = None
    dataSet_path = None

    def onclick(self, event):
        """
        按住 ctrl 时点击鼠标，在图像上标注虚线
        """
        if self.mark_flag:
            if self.mark_line:
                self.mark_line.set_xdata((event.xdata, event.xdata))
            else:
                self.mark_line = self.ax.axvline(event.xdata, plt.ylim()[0], plt.ylim()[1], c='g', ls='--', lw=1)
            self.fig.canvas.draw()

    def key_press_event(self, event):
        """
        键盘事件 ctrl / delete / 右方向键
        """
        # 按住 ctrl 进入标记模式
        if event.key == 'control':
            self.mark_flag = True

        # 按下 delete 删除标记线
        if event.key == 'delete':
            if self.mark_line:
                self.mark_line.remove()
                self.mark_line = None
                self.fig.canvas.draw()

        # 按下 enter 保存此段数据；按下右方向键跳过此段数据
        if event.key == 'enter' or event.key == 'right':

            # 根据标记线判断当前数据的离手情况
            save_data_type = '未标记'
            x_data = self.end

            if self.mark_line and self.mark_line.get_xdata()[0] > self.start:
                x_data = int(self.mark_line.get_xdata()[0])
                if x_data > self.end:
                    save_data_type = '断触'
                else:
                    save_data_type = '延迟释放'
                self.end = max(self.end, x_data)

            # 保存数据时，向前取100帧，向后取300帧
            offset1 = 100
            offset2 = min(self.forceFlag.size - self.end, 300)

            s = self.start - offset1
            e = self.end + offset2

            # 仅保存持续按压超过 5 帧的信号段
            if event.key == 'enter' and self.end - self.start > 5:
                new_flag = self.forceFlag[s:e].copy()
                new_flag[:offset1] = 0
                new_flag[-(offset2 + (self.end - x_data)):] = 0
                new_flag[offset1:-(offset2 + (self.end - x_data))] = 1

                new_f = self.dataSet_path + '/' + self.file
                new_f = new_f.replace('.txt', '_' + str(s) + '-' + str(e) + '.txt')

                contents = np.hstack((self.rawData[s:e],
                                      self.baseline[s:e],
                                      self.forceSig[s:e],
                                      self.forceFlag[s:e].reshape(new_flag.size, 1),
                                      new_flag.reshape(new_flag.size, 1)))

                folder = new_f[0:new_f.rfind('\\')]
                if not os.path.exists(folder):
                    os.makedirs(folder)

                np.savetxt(new_f, contents, fmt='%d', delimiter='\t')
                print(save_data_type, new_f)

            # 绘制下一段图像
            if self.mark_line:
                self.mark_line.remove()
                self.mark_line = None

            if '01' in self.flag[self.end:-1]:
                self.start = self.flag.find('01', self.end, -1) + 1
                self.end = self.flag.find('10', self.start, -1) + 1

                if self.end != 0:
                    force_flag_mark = self.forceFlag.copy() * 100
                    force_flag_mark[:self.start] = None
                    force_flag_mark[self.end:] = None

                    plt.cla()
                    plt.plot(self.rawData - self.rawData[self.start - 5, :], label='rawData')
                    plt.plot(self.forceFlag * 100, label='forceFlag', ls='--', c='gray')
                    plt.plot(force_flag_mark, label='current', lw=2, ls='-', c='black')
                    plt.legend(loc='upper left')

                    margin_x = (self.end - self.start) / 2
                    plt.xlim(self.start - margin_x, max(self.end + margin_x, self.start - margin_x + 200))

                    y_max = np.max((self.rawData - self.rawData[self.start - 5, :])[self.start:self.end])
                    y_min = np.min((self.rawData - self.rawData[self.start - 5, :])[self.start:self.end])
                    margin_y = (y_max - y_min) / 3
                    plt.ylim(min(y_min - margin_y, -20), max(y_max + margin_y, 120))

                    self.fig.canvas.draw()

                else:
                    plt.close()

            else:
                plt.close()

    def key_release_event(self, event):
        """
        松开 ctrl 按键退出标记模式
        """
        if event.key == 'control':
            self.mark_flag = False

    def mark(self, file, data_set_path):
        """
        经手工标记后，切分数据至dataSet_path路径下
        :param file: 待切分文件
        :param data_set_path: 切分后数据保存路径
        """
        self.file = file
        self.dataSet_path = data_set_path

        with open(file, 'r', encoding='utf-8') as fu:
            print('【打开文件】', self.file)
            temp = pd.read_csv(fu, skiprows=1, header=None, delimiter='\t')
            data = np.array(temp.iloc[:, 0:-1])

            # 不同数据格式需要针对性修改此处下标
            self.rawData = data[:, 29:35]
            self.baseline = data[:, 35:41]
            self.forceSig = data[:, 41:47]
            self.forceFlag = data[:, 2]

            self.start = self.end = 0
            self.flag = ''.join(str(x) for x in list(self.forceFlag))
            self.mark_line = None

            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)

            plt.plot(self.rawData, label='rawData')
            plt.plot(self.forceFlag * 100, label='forceFlag', ls='--')
            plt.legend(loc='upper left')

            # 绑定键鼠事件
            self.fig.canvas.mpl_connect('key_press_event', self.key_press_event)
            self.fig.canvas.mpl_connect('key_release_event', self.key_release_event)
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)

            plt.show()


# Shine
def split_root(root_data, path, tail='_result'):
    filename_root = path[len(root_data) + 1:path.rfind('\\')]
    filename = path.split('\\')[-1].split('.')[0]
    filename_tail = '.' + path.split('.')[1]
    result_root = root_data + tail + '\\' + filename_root
    # print('filename_root:' + filename_root)
    # print('filename:' + filename)
    # print('filename_tail', filename_tail)
    # print('result_root:' + result_root)
    return filename, filename_root, result_root

class GeneratePrivateModel:
    """
    根据校准点数据和公共模板，生成私有模板

    Author:   Shine Lin
    Datetime: 2019/8/9 21:42

    校准点规模为1 self.X Chanels，公共模板规模为 points self.X Chanels
    example:
        ways = 'int42'
        MODEL = GeneratePrivateModel(model)
        pri_model, model, data_X, data = MODEL.generate_model(data,ways)
        MODEL.plot_model(root_Cali, path_Cali,title_Cali,title_Model)
        MODEL.save_private_model(root_Cali,path_Cali)
    解释：
        其中pri_model是私有模板,model是处理后的公共模板,data为处理后的校准点数据，data_X为data的横坐标,；
        ways为拟合方法，'app22','app32'为逼近型方法，’int42','int33'为插值型方法，不指定时默认为'int42'；
        MODEL.plot_model(root_Cali, path_Cali,title_Cali,title_Model)为画图函数，
        root_Cali, path_Cali分别为校准点的主目录和各个校准文件的目录，
        title_Cali,title_Model分别为个标题，可省略；
        MODEL.save_private_model(root_Cali,path_Cali)保存私有模板数据，保存在root_Cali之下。
    """

    def __init__(self, model):
        self.model = model
        self.iterations = 9
        self.ways = 'int42'
        # 拟合型app22,app32
        # 逼近型int42,int33

    def model_process(self):
        # 剔除首末小于50%的顶点
        for num_point in range(0, self.model.shape[0]):
            if self.model[num_point, 0] >= np.max(self.model[num_point, 0]) / 2:
                [model0, self.model] = np.split(self.model, [num_point], axis=0)
                break
        for num_point in range(0, self.model.shape[0])[::-1]:
            if self.model[num_point, self.model.shape[1] - 1] >= np.max(
                    self.model[num_point, self.model.shape[1] - 1]) / 2:
                [self.model, model1] = np.split(self.model, [num_point + 1], axis=0)
                break
        return self.model

    def subdivision(self):
        if self.ways == 'app22':
            a, mpoint, pnary, pnary_a, sub_type = 1 / 4, 2, 2, 0, 'dual'
        elif self.ways == 'app32':
            a, mpoint, pnary, pnary_a, sub_type = 1 / 8, 3, 2, 1, 'basic'
        elif self.ways == 'int42':
            a, mpoint, pnary, pnary_a, sub_type = 1 / 16, 4, 2, 1, 'basic'
        elif self.ways == 'int33':
            b = 5 / 18
            a, mpoint, pnary, pnary_a, sub_type = b - 1 / 3, 3, 3, 0, 'dual'
        s = (mpoint * pnary - 1 - pnary_a) // (pnary - 1)
        # 扩充虚拟点
        if sub_type == 'dual':
            n_v = (s - 3) // 2 if self.ways.find('int') >= 0 else (s - 3) // 2 + 1
        else:
            n_v = (s - 2) // 2 if s % 2 == 0 else (s - 1) // 2
            n_v = n_v + 0 if self.ways.find('int') >= 0 else n_v + 1
        py = 0
        while py < n_v:
            self.P = np.vstack([self.P[0, :], self.P, self.P[-1, :]])
            self.X = np.hstack([self.X[0], self.X, self.X[-1]])
            py += 1
        print('延拓虚拟点数：' + str(n_v * 2) + '\n' + '延拓后初始点数：' + str(self.X.size))
        print('延拓后横坐标点集合：')
        print(self.X)
        print('细分方法：' + self.ways + '\n' + '细分次数k=' + str(self.iterations))
        while self.iterations > 0:
            P_new = np.zeros([(self.P.shape[0] - mpoint + 1) * pnary + pnary_a, self.P.shape[1]])
            X_new = np.zeros((self.X.shape[0] - mpoint + 1) * pnary + pnary_a)
            if self.ways == 'app22':
                for i in range(0, self.P.shape[0] - mpoint + 1):
                    P_new[2 * i, :] = (1 - a) * self.P[i, :] + a * self.P[i + 1, :]
                    P_new[2 * i + 1, :] = a * self.P[i, :] + (1 - a) * self.P[i + 1, :]
                    X_new[2 * i] = (1 - a) * self.X[i] + a * self.X[i + 1]
                    X_new[2 * i + 1] = a * self.X[i] + (1 - a) * self.X[i + 1]
            elif self.ways == 'app32':
                for i in range(0, self.P.shape[0] - mpoint + 1):
                    P_new[2 * i, :] = self.P[i, :] / 2 + self.P[i + 1, :] / 2
                    P_new[2 * i + 1, :] = a * self.P[i, :] + (1 - 2 * a) * self.P[i + 1, :] + a * self.P[i + 2, :]
                    X_new[2 * i] = self.X[i] / 2 + self.X[i + 1] / 2
                    X_new[2 * i + 1] = a * self.X[i] + (1 - 2 * a) * self.X[i + 1] + a * self.X[i + 2]
            elif self.ways == 'int42':
                for i in np.arange(0, self.P.shape[0] - mpoint + 1):
                    P_new[2 * i, :] = self.P[i + 1, :]
                    X_new[2 * i] = self.X[i + 1]
                    P_new[2 * i + 1, :] = -a * self.P[i, :] + \
                                          (1 / 2 + a) * self.P[i + 1, :] + \
                                          (1 / 2 + a) * self.P[i + 2, :] - a * self.P[i + 3, :]
                    X_new[2 * i + 1] = -a * self.X[i] + (1 / 2 + a) * self.X[i + 1] + (1 / 2 + a) * self.X[i + 2] - a * \
                                       self.X[i + 3]
            elif self.ways == 'int33':
                for i in np.arange(0, self.P.shape[0] - mpoint + 1):
                    P_new[3 * i, :] = b * self.P[i, :] + (1 - b - a) * self.P[i + 1, :] + a * self.P[i + 2, :]
                    X_new[3 * i] = b * self.X[i] + (1 - b - a) * self.X[i + 1] + a * self.X[i + 2]
                    P_new[3 * i + 1, :] = self.P[i + 1, :]
                    X_new[3 * i + 1] = self.X[i + 1]
                    P_new[3 * i + 2, :] = a * self.P[i, :] + (1 - b - a) * self.P[i + 1, :] + b * self.P[i + 2, :]
                    X_new[3 * i + 2] = a * self.X[i] + (1 - b - a) * self.X[i + 1] + b * self.X[i + 2]
            if self.ways == 'int42':
                i += 1
                P_new[2 * i, :] = self.P[i + 1, :]
                X_new[2 * i] = self.X[i + 1]
            elif self.ways == 'app32':
                i += 1
                P_new[2 * i, :] = self.P[i, :] / 2 + self.P[i + 1, :] / 2
                X_new[2 * i] = self.X[i] / 2 + self.X[i + 1] / 2
            # 拟合型方法需要通过归一输出
            if self.ways.find('app') >= 0:
                for ch in range(0, P_new.shape[1]):
                    P_new[:, ch] = P_new[:, ch] / np.max(np.max(P_new[:, ch])) * 1024
            self.P = P_new
            self.X = X_new
            self.iterations = self.iterations - 1

    def find_axis(self, model_x):
        pri_model_x = np.zeros_like(model_x)
        pri_model = np.zeros([pri_model_x.shape[0], self.P.shape[1]])
        for i in np.arange(model_x.shape[0]):
            pri_model_x[i] = np.argmin(abs(self.X - model_x[i]))
            pri_model[i] = self.P[pri_model_x[i], :]
        return pri_model

    def generate_model(self, data, ways='int42', model_door=True, iterations=9):
        self.iterations = iterations
        self.ways = ways
        self.data = data
        for ch in range(0, self.model.shape[1]):
            self.model[:, ch] = self.model[:, ch] / np.max(np.max(self.model[:, ch])) * 1024
        if model_door:
            # 剔除首末小于50%的顶点
            self.model = self.model_process()
            self.model = butter_bandpass_filter(self.model)
        model_x = np.arange(0, self.model.shape[0])
        self.data_X = np.arange(0, self.data.shape[0])
        for ch in range(0, self.model.shape[1]):
            self.data[:, ch] = self.data[:, ch] / np.max(np.max(self.data[:, ch])) * 1024
            # self.data_X[ch] = self.model[:, ch].argmax()
            self.data_X[ch] = np.argmin(abs(self.data[ch, ch] - self.model[:, ch]))
        # data_begin = self.model[0, :] + self.P[0, :] - self.model[self.X[0], :]
        # data_end = self.model[-1, :] + self.P[-1, :] - self.model[self.X[-1], :]
        # self.P = np.vstack([data_begin, self.P, data_end])
        self.data = np.vstack([self.model[0, :], self.data, self.model[-1, :]])
        self.data_X = np.hstack([model_x[0], self.data_X, model_x[-1]])
        # P和X首尾增加model的首尾点
        self.P = self.data
        self.X = self.data_X
        self.subdivision()
        pri_model = self.find_axis(model_x)
        # np.savetxt('private_model.txt', self.pri_model, '%d')
        self.pri_model = pri_model
        print('shape of private_model:')
        print(self.pri_model.shape)
        return self.pri_model, self.model, self.data_X, self.data

    def plot_model(self,root_Cali, path_Cali, title_Cali='title_Cali',title_Model='title_model'):
        fig,ax=plt.subplots((self.pri_model.shape[1]+1)//2,2,sharex='all',sharey='all',figsize=(20, 9))
        ax=ax.flatten()
        coss = np.zeros(self.pri_model.shape[1])
        plt.suptitle('title_Cali：'+title_Cali + '\n' +'title_Model：'+ title_Model + '\n' +'Ways：'+ self.ways, fontsize=18)
        for ch in range(self.pri_model.shape[1]):
            ax[ch].set_title('CH' + str(ch+1),fontsize=14)
            ax[ch].plot(self.pri_model[:, ch], 'r-*', markersize=4)
            ax[ch].plot(self.model[:, ch], 'b-o', markersize=4)
            ax[ch].plot(self.data_X, self.data[:, ch], 'ko', markersize=6)
            coss[ch] = round(np.dot(self.model[:, ch], self.pri_model[:, ch]) / (
                    np.linalg.norm(self.model[:, ch], ord=2, keepdims=False) *
                    np.linalg.norm(self.pri_model[:, ch], ord=2, keepdims=False)), 5)
            ax[ch].legend(['Pri_model', 'Pub_model', 'Cali_points'],fontsize=12)
        print('各个chanel的拟合度:')
        print(coss)
        # plt.get_current_fig_manager().window.state('zoomed')
        filename, filename_root, result_root = split_root(root_Cali, path_Cali, tail='_pri_model')
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        plt.savefig(result_root + '\\' + filename + '.png')
        plt.show()
        plt.close()
    
    def save_private_model(self,root_Cali,path_Cali):
        filename, filename_root, result_root=split_root(root_Cali, path_Cali, tail='_pri_model')
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        np.savetxt(result_root + '\\' + filename + '.txt', self.pri_model, fmt='%8.2f', delimiter='\t')
        print('Save text succeed!')


# Carry

class ExtractDotData:
    """
    提取密集打点数据

    Carry Chen

    example:
    save_folder = 'result_folder'
    edd = ExtractDotData()
    for fid,f in enumerate(list_all_files('打点数据文件夹'，['.txt'])):
        simple_file_name = get_fig_title(f,(-1,))
        rawdata = 你从file里面读出的rawdata, np.array, shape = N * chs
        edd(rawdata, save_filename = save_folder + '//' + simple_file_name + '.txt',fig_name=simple_file_name)
    files = list_all_files(save_folder, ['.txt'])
    edd.generate_public_template(files, template_len=None)
    """
    # 默认参数
    if True:
        plt.rcParams['font.family'] = 'FangSong'
        plt.rcParams['axes.unicode_minus'] = False
        _markers = ('o', 's', 'p', '*', '+', '<', '>', '^')
        _colors = ('blue', 'green', 'red', 'orange', 'gray', 'yellow', 'pink', 'black')

        # baseline tracking
        _alpha = 0.005
        _limit = 5
        # force flag
        _trigger_th = 20
        _slope_th = 0
        _leave_ratio_th = 0.1
        _leave_cnt_th = 100
        _touch_cnt_th = 100
        # a better setting for PT101
        # base_index = [-150, -50]
        # data_index = [100, 200]
        # chose_index = [10, 90]
        # sig extract -- from Dragon Long's APK
        _base_index = (-90, -50)
        _data_index = (80, 120)
        _chose_index = (15, 25)

    def __init__(self, alpha=_alpha, limit=_limit,
                 trigger_th=_trigger_th, slope_th=_slope_th, leave_ratio_th=_leave_ratio_th,
                 leave_cnt_th=_leave_cnt_th, touch_cnt_th=_touch_cnt_th,
                 base_index=_base_index, data_index=_data_index, chose_index=_chose_index):
        self.__skip_frame = 100
        self.alpha = alpha
        self.limit = limit
        self.trigger_th = trigger_th
        self.slope_th = slope_th
        self.leave_ratio_th = leave_ratio_th
        self.leave_cnt_th = leave_cnt_th
        self.touch_cnt_th = touch_cnt_th
        self.base_index = base_index
        self.data_index = data_index
        self.chose_index = chose_index

    def __from_rawdata_to_matrix(self, rawdata):
        """
        从rawdata中提取打点的矩阵

        :param rawdata: np.array, shape: N * CHS
        :return: out_data: np.array，shape: k * CHS, k为打点次数
        """
        if rawdata.ndim == 1:
            rawdata = np.reshape(rawdata, newshape=(-1, 1))
        CHS = rawdata.shape[1]
        raw = np.vstack(
            (np.tile(rawdata[0], [self.__skip_frame, 1]), rawdata, np.tile(rawdata[-1], [self.__skip_frame, 1])))
        bsl = raw + 0  # equal to deepcopy
        force_flag = np.zeros(shape=(raw.shape[0], 1))
        last_maxch = peak = touch_cnt = 0
        leave_cnt = 9999
        out_data = np.zeros(shape=(0, CHS))
        ff_point = []
        for i in range(self.__skip_frame, raw.shape[0]):
            force_flag[i] = force_flag[i - 1]
            sig = raw[i] - bsl[i - 1]
            maxch = np.argmax(sig, axis=0)
            if force_flag[i - 1] == 0:
                delta = raw[i] - bsl[i - 1]
                leave_cnt += 1
                for ch in range(CHS):  # bsl tracking
                    bsl[i, ch] = bsl[i - 1, ch] + np.sign(delta[ch]) * min(self.alpha * abs(delta[ch]), self.limit)
                if ((raw[i, maxch] - raw[i - 1, maxch] > self.slope_th) or
                    (raw[i - 1, maxch] - raw[i - 2, maxch] > self.slope_th)) and \
                        sig[maxch] > self.trigger_th and leave_cnt > self.leave_cnt_th:  # force_flag trigger
                    force_flag[i] = 1
                    last_maxch = maxch
                    peak = sig[last_maxch]
                    touch_cnt = 0
            else:
                bsl[i] = bsl[i - 1] + 0
                touch_cnt += 1
                if sig[maxch] > peak:
                    peak = sig[maxch]
                    last_maxch = maxch
                if sig[last_maxch] < self.leave_ratio_th * peak and raw[i, last_maxch] > raw[i - 2, last_maxch] \
                        and touch_cnt > self.touch_cnt_th:
                    force_flag[i] = 0
                    leave_cnt = 0
            if force_flag[i - 1] == 0 and force_flag[i] == 1:
                base = np.sort(raw[i + self.base_index[0]:i + self.base_index[1]], axis=0)
                data = np.sort(raw[i + self.data_index[0]:i + self.data_index[1]], axis=0)
                base = np.mean(base[self.chose_index[0]:self.chose_index[1]], axis=0)
                data = np.mean(data[self.chose_index[0]:self.chose_index[1]], axis=0)
                out_data = np.r_[out_data, np.reshape(data - base, (1, CHS))]
                ff_point.append(i)
        return out_data, raw, bsl, force_flag, ff_point

    def __call__(self, rawdata, save_filename=None, fig_name=None, filter_flag=True):
        out_data, raw, bsl, force_flag, ff_point = self.__from_rawdata_to_matrix(rawdata)
        if filter_flag:
            out_data = butter_bandpass_filter(out_data)
        print('out_data shape: ', out_data.shape)
        CHS = out_data.shape[1]
        figM = (CHS + 1) // 2
        plt.figure(1)
        for ch in range(CHS):
            plt.subplot(figM, 2, ch + 1)
            plt.plot(raw[:, ch], label='raw')
            plt.plot(bsl[:, ch], label='baseline')
            plt.plot(raw[0, ch] + force_flag * 100, label='forceflag')
            for i in ff_point:
                plt.plot(np.arange(i + self.base_index[0], i + self.base_index[1]),
                         raw[i + self.base_index[0]:i + self.base_index[1]][:, ch], color='black', marker='o')
                plt.plot(np.arange(i + self.data_index[0], i + self.data_index[1]),
                         raw[i + self.data_index[0]:i + self.data_index[1]][:, ch], color='red', marker='o')
            plt.legend()
            plt.title('CH' + str(ch + 1))
        plt.suptitle(fig_name)
        plt.figure(2)
        plt.plot(out_data, '-o')
        plt.ylabel('ADC')
        plt.grid(True)
        plt.suptitle(fig_name)
        plt.show()
        if save_filename is not None:
            np.savetxt(save_filename, out_data, fmt='%+6.1f', delimiter='\t')
            print(save_filename + ' file saved successfully')
        else:
            warnings.warn('warning message: filename not assigned, so not saved')
        return

    def __get_front_rear_points(self, data_norm, peak_pos, template_len, ratio, front_ch, rear_ch):
        """
        private for generate_template

        :param data_norm:
        :param peak_pos:
        :param template_len:
        :param ratio:
        :param front_ch:
        :param rear_ch:
        :return:
        """
        if isinstance(template_len, int):  # 优先使用 template_len
            for front_points in range(data_norm.shape[0]):
                rear_points = template_len - (peak_pos[rear_ch] - peak_pos[front_ch]) - front_points
                if rear_points + peak_pos[rear_ch] >= data_norm.shape[0] or \
                        peak_pos[front_ch] - front_points >= peak_pos[0]:
                    continue
                if data_norm[peak_pos[front_ch] - front_points, 0] <= \
                        data_norm[peak_pos[rear_ch] + rear_points, -1]:
                    print('template_len:', template_len, ', front_points:', front_points - 1, ', rear_points:',
                          rear_points)
                    return template_len, front_points - 1, rear_points + 1
            print('error: template_len=', template_len, ', 但数据没有这么长')
            exit()
        else:  # 然后使用ratio
            front_points = peak_pos[0]
            for i in range(peak_pos[0]):
                if data_norm[peak_pos[0] - i, 0] < data_norm[peak_pos[0], 0] * ratio:
                    front_points = i - 2
                    break
            rear_points = data_norm.shape[0] - peak_pos[1]
            for i in range(data_norm.shape[0] - peak_pos[-1]):
                if data_norm[peak_pos[-1] + i, -1] < data_norm[peak_pos[-1], -1] * ratio:
                    rear_points = i + 1
                    break
            template_len = peak_pos[-1] - peak_pos[0] + rear_points + front_points
            print('template_len:', template_len, ', front_points:', front_points, ', rear_points:', rear_points - 1)
            return template_len, front_points, rear_points

    def __get_mean_tempalte(self, data_list, peak_poses, template_len, front_points, rear_points, front_ch, rear_ch):
        """
        private for generate_template

        :param data_list:
        :param peak_poses:
        :param template_len:
        :param front_points:
        :param rear_points:
        :param front_ch:
        :param rear_ch:
        :return:
        """
        chs = data_list[0].shape[1]
        template = np.zeros(shape=(template_len, chs))
        for i, data in enumerate(data_list):
            x = np.arange(peak_poses[i, front_ch] - front_points, peak_poses[i, front_ch] - front_points + template_len)
            for ch in range(chs):
                xp = np.arange(peak_poses[i, front_ch] - front_points, peak_poses[i, rear_ch] + rear_points)
                yp = data[peak_poses[i, front_ch] - front_points:peak_poses[i, rear_ch] + rear_points, ch]
                template[:, ch] += np.interp(x * xp.shape[0] / template_len, xp, yp)
        return (template / np.max(template, axis=0) * 1024).astype(np.int)

    def generate_public_template(self, files, template_filename=None, template_len=None,
                                 ratio=0.5, factor=1, front_ch=0, rear_ch=-1):
        """
        使用一系列的打点提取文件，生成公共模板
        :param files: 一系列打点提取之后的文件
        :param template_filename: 为None时不保存
        :param template_len: 模板的长度，如果不为None，ratio不生效
        :param ratio: 边缘通道两侧大于峰值的ratio
        :param factor: 系数 如400/2.67*100，为1时表示ADC,否则为μV/100g
        :param front_ch: 模板的设计时选择的对齐前面个通道，默认为第一通道
        :param rear_ch:模板的设计时选择的对齐后面个通道，默认为最后通道
        :return:
        """

        if len(files) == 0:
            print('错误：传入的files为空')
            exit()
        data_list = []
        ylabel = 'ADC' if factor == 1 else 'μV'
        for fid, file in enumerate(files):
            print(fid, file)
            with open(file, 'r', encoding='utf-8') as f:
                fn = get_fig_title(file, (-1,))
                data = np.array(pd.read_csv(f, header=None, skiprows=0, delimiter='\t'))
                f.close()
            np.set_printoptions(precision=3, suppress=True)
            print('cali_coef:', 400 / data.max(0))
            peak_pos = np.argmax(data, axis=0)
            data_norm = data / np.max(data, axis=0) * 1024
            data_list.append(data_norm)
            try:
                peak_poses = np.r_[(peak_poses, np.reshape(peak_pos, newshape=(1, -1)))]
            except:  # 第一个文件执行
                middle_ch = peak_pos.shape[0] // 2
                peak_pos_default = peak_pos[middle_ch]
                peak_poses = np.reshape(peak_pos, newshape=(1, -1))
            zero_point = peak_pos_default - peak_pos[middle_ch]
            x = np.arange(zero_point, zero_point + data.shape[0])
            # 画图
            colori = fid % len(self._colors)
            makeri = (fid // len(self._markers)) % len(self._markers)
            plt.figure(1)
            plt.plot(x, data[:, :-1] * factor, marker=self._markers[makeri], color=self._colors[colori])
            plt.plot(x, data[:, -1] * factor, marker=self._markers[makeri], color=self._colors[colori], label=fn)
            plt.ylabel(ylabel)
            plt.legend()
            plt.title('密集打点数据--未归一化')
            plt.figure(2)
            plt.plot(x, data_norm[:, :-1] * factor, marker=self._markers[makeri], color=self._colors[colori])
            plt.plot(x, data_norm[:, -1] * factor, marker=self._markers[makeri], color=self._colors[colori], label=fn)
            plt.legend()
            plt.title('密集打点数据--归一化--1024')
        fid = np.argmax(peak_poses[:, -1] - peak_poses[:, 0])
        data_norm = data_list[fid]
        peak_pos = peak_poses[fid]
        template_len, front_points, rear_points = self.__get_front_rear_points(
            data_norm, peak_pos, template_len, ratio, front_ch, rear_ch)
        template = self.__get_mean_tempalte(data_list, peak_poses, template_len, front_points, rear_points, front_ch,
                                            rear_ch)
        print("template shape:", template.shape)
        plt.figure(3)
        plt.plot(template, '-o')
        plt.title('mean template')
        plt.show()
        # 保存template到txt
        if template_filename is not None:
            np.savetxt(template_filename, template, fmt='%+6d', delimiter='\t')
            print('template shape: ', template.shape)
        else:
            warnings.warn('warning message: filename not assigned, so not saved')
        return template

    def read_and_compare_template(self, files):
        """
        画图比较模板
        :param files:
        :return:
        """
        peak_pos_default = 40
        for fid, file in enumerate(files):
            print(fid, file)
            with open(file, 'r', encoding='utf-8') as f:
                fn = get_fig_title(file, (-1,))
                data = np.array(pd.read_csv(f, header=None, skiprows=0, delimiter='\t'))
                f.close()
            zero_point = peak_pos_default - data[:, 3].argmax()
            x_ord = np.arange(zero_point, zero_point + data.shape[0])
            colori = fid % len(self._colors)
            makeri = (fid * (len(self._markers) - 1) // len(self._markers)) % len(self._markers)
            plt.plot(x_ord, data[:, :-1], marker=self._markers[makeri], color=self._colors[colori])
            plt.plot(x_ord, data[:, -1], marker=self._markers[makeri], color=self._colors[colori], label=fn)
        plt.legend()
        plt.title('template compare')
        plt.show()
        return None


def get_fig_title(filename='test.txt', layers=(-1,)):
    """
    目的：有些绝对路径的文件名太长了，画图的时候想要一个短的文件名
    获取文件名,用于画图的title，以及有时候生成存

    Carry Chenli
    :param filename: 完整的filename,一般很长
    :param layers:  用'\'split之后，选择使用那些layer，先父目录layer，再子目录
    :return: out_str: 较短的文件名
    """
    strs = filename.split('\\')
    out_str = ''
    for layer in layers:
        if layer == -1:
            strs_last = strs[layer].split('.')
            for s in strs_last[:-1]:
                out_str += s
        else:
            out_str += strs[layer] + '_'
    return out_str
