import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
import ctypes
import ctypes.wintypes
import win32con
import functools
import time
import win32api
import win32con
import win32gui
import win32clipboard
import win32process
from pymouse import PyMouse
from pykeyboard import PyKeyboard

# Wayne:
"""
table of content:
1. Useful tools:
    a. (function) func_timer
    b. (function) list_all_files
    c. (class) GlobalHotKeys
    d. (class) GuiOperation
2. Data processing:
    a. (function) peak_det
    b. (function) butter_bandpass_filter
    c. (class) FindLocalExtremeValue
    d. (class) LoadData
    e. (class) DataProcessing
    f. (class) ExtractRollingData
3. mathematics:
    a. (function) find_all_non_negative_integer_solutions
"""


def func_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.time()
        r = func(*args, **kw)
        print('%s excute in %.3f s' % (func.__name__, (time.time() - start)))
        return r

    return wrapper


def list_all_files(root: str, keys: list, outliers: list):
    """
    列出某个文件下所有文件的全路径

    Author:   wangye
    Datetime: 2019/4/16 18:03

    :param root: 根目录
    :param keys: 所有关键字
    :param outliers: 所有排除关键字
    :return:
            所有根目录下包含关键字的文件全路径
    """
    _files = []
    _list = os.listdir(root)
    for i in range(len(_list)):
        path = os.path.join(root, _list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path, keys, outliers))
        if os.path.isfile(path) \
                and all([k in path for k in keys]) \
                and not any([o in path for o in outliers]):
            _files.append(path)
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

    MOD_ALT = win32con.MOD_ALT
    MOD_CTRL = win32con.MOD_CONTROL
    MOD_CONTROL = win32con.MOD_CONTROL
    MOD_SHIFT = win32con.MOD_SHIFT
    MOD_WIN = win32con.MOD_WIN

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
                 slp_step=3
                 ):
        """
        :param local_window: 表示寻找极值的窗长，仅寻找数据x中最后local_window
                            长度的数据相对于x最后一帧的极值。
        :param min_valid_slp_thd: 要成为合理的局部极值，至少要与被比较者相差的值
        :param slp_step: 找当前值和极值之间的最值斜率，斜率计算的步长
        """
        self.local_window = local_window
        self.min_valid_slp_thd = min_valid_slp_thd
        self.min_slp_step = slp_step
        self.max_slp_step = slp_step

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
                第四个local_extreme_real_value是指局部极值实际的值
        """

        assert self.local_window + 2 <= x.shape[0] and x.shape[0] >= 4
        x = x.astype(np.float)
        m = len(x)

        local_diff, local_gap, min_slope, max_slope, local_extreme_real_value = 0, 0, np.inf, -np.inf, 0
        local_x = 0
        update_num = 0
        queue = []
        for j in range(1, self.local_window):
            if j + 2 * self.min_slp_step <= m:
                min_slope = np.min((
                    min_slope,
                    (x[-j - self.min_slp_step:-j].mean() -
                     x[-j - 2 * self.min_slp_step:-j - self.min_slp_step].mean()
                     ) / self.min_slp_step
                ))
                max_slope = np.max((
                    max_slope,
                    (x[-j - self.max_slp_step:-j].mean() -
                     x[-j - 2 * self.max_slp_step:-j - self.max_slp_step].mean()
                     ) / self.max_slp_step
                ))
            if not queue:
                queue = [x[-1] - x[-2], x[-1] - x[-3], x[-1] - x[-4]]  # init
            else:
                queue.pop(0)
                queue.append(x[-1] - x[-1 - j - 2])
            a, b, c = queue
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
            if update_num > 4:
                break
        local_diff = local_x
        return local_diff, local_gap, min_slope, max_slope, local_extreme_real_value


class LoadData:
    """
    用于加载数据到内存
    Author:   wangye
    Datetime: 2019/4/23 17:17
    """

    def __init__(self,
                 channel_num: int,
                 begin_ind_dic: dict,
                 data_type: str
                 ):
        """

        :param channel_num: 通道数
        :param begin_ind_dic: 数据起始位置字典, 关键字为:
                                    rawdata, baseline, forcesig,
                                    forceflag, humanflag, stableflag
                                    temperature, fwversion
        :param data_type: 传入数据的类型, 'slice' 或 'normal'
        """
        self.channel_num = channel_num
        self.data_type = data_type
        if self.data_type == 'slice':
            if not begin_ind_dic:
                begin_ind_dic = {
                    'rawdata': 0,
                    'baseline': channel_num,
                    'forcesig': 2 * channel_num,
                    'forceflag': 3 * channel_num,
                    'humanflag': 3 * channel_num + 1
                }
            self.humanflag_column = begin_ind_dic['humanflag']
        elif self.data_type == 'normal':
            if not begin_ind_dic:
                begin_ind_dic = {
                    'rawdata': 28,
                    'baseline': 34,
                    'forcesig': 40,
                    'forceflag': 1,
                    'stableflag': 4,
                    'temperature': 8,
                    'fwversion': 26
                }
            self.stable_flag_range = range(begin_ind_dic['stableflag'],
                                           begin_ind_dic['stableflag'] + channel_num)
            self.temperature_column = begin_ind_dic['temperature']
            self.fw_version = begin_ind_dic['fwversion']
        self.rawdata_range = range(begin_ind_dic['rawdata'],
                                   begin_ind_dic['rawdata'] + channel_num)
        self.baseline_range = range(begin_ind_dic['baseline'],
                                    begin_ind_dic['baseline'] + channel_num)
        self.forcesig_range = range(begin_ind_dic['forcesig'],
                                    begin_ind_dic['forcesig'] + channel_num)
        self.forceflag_column = begin_ind_dic['forceflag']

    def load_data(self, path):
        """
        Load Data 入口
        :param path: 文件路径
        :return:
        """
        if self.data_type == 'slice':
            return self.__load_data_from_slices__(path)
        elif self.data_type == 'normal':
            return self.__load_data_normal__(path)

    def __load_data_from_slices__(self, path):
        """
        From slices
        :param path:
        :return:
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = np.int64(pd.read_table(f))
        rawdata = data[:, self.rawdata_range]
        baseline = data[:, self.baseline_range]
        forcesig = data[:, self.forcesig_range]
        forceflag = data[:, self.forceflag_column]
        forceflagnew = data[:, self.humanflag_column]

        for ch in range(self.channel_num):
            index = np.argmax((rawdata[:, ch] - baseline[:, ch]))
            total_calibration_coeff = forcesig[index, ch] / (rawdata[index, ch] - baseline[index, ch])
            rawdata[:, ch] = rawdata[:, ch] * total_calibration_coeff
            baseline[:, ch] = baseline[:, ch] * total_calibration_coeff

        temperature_err_flag = np.ones_like(data[:, 0])
        sig_err_flag = np.zeros_like(data[:, 0])
        return data, rawdata, baseline, forceflag, forcesig, forceflagnew, temperature_err_flag, sig_err_flag

    def __load_data_only_data__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = np.int64(pd.read_csv(f, header=None, skiprows=1, delimiter='\t').iloc[:, 1:-1])
        return data

    def __load_data_normal__(self, path):
        """
        From firmware
        :param path:
        :return:
        """
        data = self.__load_data_only_data__(path)

        fw_version = data[:, self.fw_version]
        print('固件版本号: {}'.format(fw_version[0]))

        forceflag = data[:, self.forceflag_column] % 2
        rawdata = data[:, self.rawdata_range]
        baseline = data[:, self.baseline_range]
        forcesig = data[:, self.forcesig_range]
        stable_state = data[:, self.stable_flag_range]

        temperature = data[:, self.temperature_column] / 100
        temperature_err_flag = temperature > 30
        rawdata, baseline = self.__rawdata_baseline_calibration__(rawdata, forcesig, baseline, temperature,
                                                                  method=2)
        sig_err_flag = np.zeros_like(forceflag)
        return data, rawdata, baseline, forceflag, forcesig, temperature_err_flag, sig_err_flag, stable_state

    def __rawdata_baseline_calibration__(self, rawdata, forcesig, baseline, temperature, method=2):
        if method == 1:  # 对 rawdata,baseline 逐帧计算系数
            calibration_coeff = self.__temperature_calibration__(rawdata, forcesig, baseline, temperature)
            rawdata = rawdata * calibration_coeff
            baseline = baseline * calibration_coeff
        elif method == 2:  # 都使用forcesig最大帧系数
            for ch in range(forcesig.shape[1]):
                index = np.argmax((rawdata[:, ch] - baseline[:, ch]))
                total_calibration_coeff = forcesig[index, ch] / (rawdata[index, ch] - baseline[index, ch])
                rawdata[:, ch] = rawdata[:, ch] * total_calibration_coeff
                baseline[:, ch] = baseline[:, ch] * total_calibration_coeff
        else:  # 对 rawdata 逐帧计算系数, 对baseline 使用forcesig最大帧系数
            calibration_coeff = self.__temperature_calibration__(rawdata, forcesig, baseline, temperature)
            for ch in range(forcesig.shape[1]):
                index = np.argmax((rawdata[:, ch] - baseline[:, ch]))
                total_calibration_coeff = forcesig[index, ch] / (rawdata[index, ch] - baseline[index, ch])
                baseline[:, ch] = baseline[:, ch] * total_calibration_coeff
            rawdata = rawdata * calibration_coeff
        return rawdata, baseline

    def __temperature_calibration__(self, rawdata, forcesig, baseline, temperature):
        temperatures = [-20.0, -10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0]
        values = [[0.528, 0.574, 0.620, 0.633, 0.486, 0.329],
                  [0.594, 0.676, 0.700, 0.636, 0.517, 0.424],
                  [0.677, 0.771, 0.775, 0.680, 0.588, 0.548],
                  [0.783, 0.859, 0.852, 0.766, 0.708, 0.707],
                  [0.919, 0.947, 0.942, 0.905, 0.889, 0.897],
                  [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
                  [1.092, 1.065, 1.073, 1.114, 1.115, 1.104],
                  [1.309, 1.271, 1.305, 1.400, 1.276, 1.298],
                  [1.573, 1.757, 1.838, 1.712, 1.186, 1.450]]

        def get_relative_coeff(t0, ch):
            if t0 <= temperatures[0]:
                return values[0][ch]
            elif t0 >= temperatures[-1]:
                return values[-1][ch]
            for (ti, t) in enumerate(temperatures):
                if ti == len(temperatures) - 1:
                    break
                if temperatures[ti] <= t0 < temperatures[ti + 1]:
                    c1, c2 = values[ti][ch], values[ti + 1][ch]
                    t1, t2 = temperatures[ti], temperatures[ti + 1]
                    return c1 * c2 * (t2 - t1) / (c1 * (t0 - t1) + c2 * (t2 - t0))

        max_forcesig_relative_coeff = np.array([0.0] * self.channel_num, dtype=np.float32)
        for ch in range(forcesig.shape[1]):
            index = np.argmax((rawdata[:, ch] - baseline[:, ch]))
            calibration_coeff = forcesig[index, ch] / (rawdata[index, ch] - baseline[index, ch])
            max_forcesig_relative_coeff[ch] = calibration_coeff / get_relative_coeff(temperature[index], ch)
            if ch == 0:
                print(max_forcesig_relative_coeff[0], index, get_relative_coeff(temperature[index], ch),
                      calibration_coeff,
                      temperature[index])
        coeffs = np.ones_like(rawdata, dtype=np.float32)
        for i in range(rawdata.shape[0]):
            temperaturei = temperature[i]
            for ch in range(rawdata.shape[1]):
                coeffs[i][ch] = max_forcesig_relative_coeff[ch] * get_relative_coeff(temperaturei, ch)
        return coeffs


class DataProcessing:
    """
    用于从rawdata中提取force

    Author:   wangye
    Datetime: 2019/5/17 11:17

    example:
    dp = DataProcessing(rawdata, f)
    dp.pre_process()
    dp.limiting_filter()
    dp.calc_moving_avg()
    dp.baseline_removal()
    dp.calc_energy()
    dp.calc_flag()
    dp.show_fig()
    dp.calc_force()
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
        plt.rcParams['font.family'] = 'YouYuan'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure()
        fig.set_size_inches(60, 10)
        plt.subplot(211)
        plt.plot(self.data)
        plt.title('rawdata')
        plt.legend(tuple([''.join(('rawdata', str(ii))) for ii in range(np.shape(self.data)[1])]))
        plt.ylabel('ADC')
        plt.subplot(212)
        # plt.plot(self.force_signal, '-', linewidth=3)
        plt.plot(self.energy)
        if len(self.energy_peak) > 0:
            plt.plot(self.energy_peak[:, 0], self.energy_peak[:, 1], '.')
        plt.plot(self.flag * (np.max(self.energy) - np.min(self.energy)) + np.min(self.energy), '--')
        plt.hlines(self.energy_thd, 0, self.data.shape[0], linestyles='--')
        plt.title(self.filename)
        plt.xlabel('Time Series')
        plt.ylabel('ADC')
        if len(self.energy_peak) > 0:
            plt.legend(['energy', 'energy peak', 'touch flag', 'energy threshold'])
        else:
            plt.legend(['energy', 'touch flag', 'energy threshold'])
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
