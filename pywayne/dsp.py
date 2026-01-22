# !/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: Wang Ye (Wayne)
@file: dsp.py
@time: 2022/03/01
@contact: wangye@oppo.com
@site: 
@software: PyCharm
# code is far away from bugs.
"""

import collections
import math
import numpy as np

from pywayne.tools import wayne_print
from sortedcontainers import SortedList
from scipy.signal import butter, lfilter, filtfilt, detrend, medfilt
from scipy.interpolate import interp1d

from typing import Union, Iterable, Tuple, Optional


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
        x = range(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        print('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        print('Input argument delta must be a scalar')

    if delta <= 0:
        print('Input argument delta must be positive')

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan

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
                maxtab.append([mxpos, mx])
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append([mnpos, mn])
                mx = this
                mxpos = x[i]
                lookformax = True
    return maxtab, mintab


def butter_bandpass_filter(x, order=2, lo=0.1, hi=10, fs=0, btype='lowpass', realtime=False):
    """
    ButterWorth 滤波器

    Author:   wangye
    Datetime: 2019/4/24 16:00

    :param x: 待滤波矩阵, 2d-array
    :param order: butterworth滤波器阶数
    :param lo: 下限
    :param hi: 上限
    :param fs: 采样频率，默认为0，如果不为0，则对lo和hi进行变换
    :param btype: butterworth滤波器种类，默认为低通滤波
    :param realtime: 是否采用在线滤波器
    :return: 滤波后矩阵, 2d-array
    """
    if fs:
        lo = lo / (fs / 2)
        hi = hi / (fs / 2)
    if btype == 'lowpass':
        wn = hi
    elif btype == 'highpass':
        wn = lo
    elif btype == 'bandpass':
        wn = [lo, hi]
    elif btype == 'bandstop':
        wn = [lo, hi]
    else:
        raise ValueError(f'btype cannot be {btype}')
    b, a = butter(N=order, Wn=wn, btype=btype)
    filter = filtfilt if not realtime else lfilter
    return np.apply_along_axis(
        lambda y: filter(b, a, y),
        0, x
    )


def find_extremum_in_sliding_window(data: list, k: int) -> list:
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
        if minQueue and i - minQueue[0] >= k: minQueue.popleft()
        if maxQueue and i - maxQueue[0] >= k: maxQueue.popleft()
        while minQueue and n < data[minQueue[-1]]: minQueue.pop()
        while maxQueue and n > data[maxQueue[-1]]: maxQueue.pop()
        minQueue.append(i)
        maxQueue.append(i)
        retMin.append(data[minQueue[0]])
        retMax.append(data[maxQueue[0]])
    return retMin, retMax


class FindSlidingWindowExtremum:
    def __init__(self, win: int, find_max: bool):
        self._win = win
        self._cnt = 0
        self._deque = collections.deque()
        self._find_max = find_max

    def apply(self, val):
        count = 0
        while self._deque and ((self._deque[-1][0] < val) if self._find_max else (self._deque[-1][0] > val)):
            count += self._deque[-1][1] + 1
            self._deque.pop()
        self._deque.append([val, count])
        self._cnt = min(self._cnt + 1, self._win + 1)
        if self._cnt > self._win:
            self._pop()
        if self._cnt >= self._win:
            return self._deque[0][0]
        else:
            return 0.0

    def _pop(self):
        if self._deque[0][1] > 0:
            self._deque[0][1] -= 1
        else:
            self._deque.popleft()


class FindLocalExtremum:
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


class MKAverage:

    def __init__(self, m: int, k: int):
        self._m, self._k = m, k
        self._deque = collections.deque()
        self._sl = SortedList()
        self._total = self._first_k = self._last_k = 0

    def apply(self, val: float) -> float:
        self._total += val
        self._deque.append(val)
        index = self._sl.bisect_left(val)
        if index < self._k:
            self._first_k += val
            if len(self._sl) >= self._k:
                self._first_k -= self._sl[self._k - 1]
        if index >= len(self._sl) + 1 - self._k:
            self._last_k += val
            if len(self._sl) >= self._k:
                self._last_k -= self._sl[-self._k]
        self._sl.add(val)
        if len(self._deque) > self._m:
            val = self._deque.popleft()
            self._total -= val
            index = self._sl.index(val)
            if index < self._k:
                self._first_k -= val
                self._first_k += self._sl[self._k]
            elif index >= len(self._sl) - self._k:
                self._last_k -= val
                self._last_k += self._sl[-self._k - 1]
            self._sl.remove(val)
        if len(self._sl) < self._m:
            return 0.0
        else:
            return self._calculate_mk_average()

    def _calculate_mk_average(self) -> float:
        return (self._total - self._first_k - self._last_k) / (self._m - 2 * self._k)


class WelfordStd:
    def __init__(self, win: int):
        self._avg = 0.0
        self._var = 0.0
        self._std = 0.0
        self._win = win
        self._cnt = 0
        self._deque = collections.deque()

    def apply(self, val):
        self._cnt = min(self._cnt + 1, self._win + 2)
        self._deque.append(val)
        if self._cnt > self._win + 1:
            self._deque.popleft()
        old_data = self._deque[0]
        pre_avg = self._avg
        if self._cnt <= self._win:
            self._avg += (val - pre_avg) / self._cnt
            self._var += (val - self._avg) * (val - pre_avg)
        else:
            self._avg += (val - old_data) / self._win
            self._var += (val - old_data) * (val - self._avg + old_data - pre_avg)
        if self._cnt >= self._win:
            self._std = math.sqrt(max(self._var, 0.0) / (self._win - 1))
        else:
            self._std = math.sqrt(max(self._var, 0.0) / (self._cnt - 1))
        return self._std

    def get_cnt(self):
        return self._cnt


class OneEuroFilter:
    """One Euro Filter implementation for smooth signal filtering.

    A speed-adaptive low-pass filter particularly useful for noisy signals like motion tracking data.
    The filter adapts its cutoff frequency based on speed:
    - At low speeds, it reduces jitter and smooths the signal
    - At high speeds, it reduces lag and follows fast changes

    Parameters
    ----------
    te : float, optional
        Sampling period in seconds. Can be set during initialization or in first apply() call.
    mincutoff : float, default=1.0
        Minimum cutoff frequency in Hz. Lower values result in smoother filtering
        but may introduce more lag.
    beta : float, default=0.007
        Speed coefficient. Higher values make the filter more aggressive at
        following quick changes.
    dcutoff : float, default=1.0
        Cutoff frequency for derivative in Hz.

    References
    ----------
    - Casiez et al. (2012): https://gery.casiez.net/1euro/
    - https://gery.casiez.net/1euro/
    - https://gery.casiez.net/1euro/InteractiveDemo/
    """

    def __init__(self, te=None, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self._val = None
        self._dx = 0
        self._te = te
        self._alpha = None
        self._dalpha = None
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff

    def _calc_alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / self._te)

    def apply(self, val: float, te: float = 0.0) -> float:
        result = val
        if self._te is None:
            self._te = te
        if self._alpha is None:
            self._alpha = self._calc_alpha(self._mincutoff)
            self._dalpha = self._calc_alpha(self._dcutoff)
        if self._val is not None:
            edx = (val - self._val) / self._te
            self._dx = self._dx + (self._dalpha * (edx - self._dx))
            cutoff = self._mincutoff + self._beta * abs(self._dx)
            self._alpha = self._calc_alpha(cutoff)
            result = self._val + self._alpha * (val - self._val)
        self._val = result
        return result


class CalcEnergy:
    def __init__(self, alpha=10, beta=20, gamma=1):
        """
        Calculate energy of given signal
        :param alpha: local mean window
        :param beta: remote diff window
        :param gamma: power of energy, currently set to 1
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def apply(self, data: np.ndarray):
        """
        Apply energy calculator w.r.t. input data
        :param data: N*d input array
        :return: N*d energy
        """
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))
        energy = np.vstack((
            np.zeros((self.alpha + self.beta - 1, data.shape[1])),
            np.apply_along_axis(
                lambda x: np.convolve(np.abs(x), np.ones(self.alpha) / self.alpha, 'valid'),
                0,
                data[self.beta:] - data[:-self.beta]
            )
        ))
        return energy


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


class SignalDetrend:
    """信号去趋势处理类"""

    def __init__(self, method='linear', **kwargs):
        """
        初始化去趋势处理器

        Parameters:
            method: str, 去趋势方法
                - 'none': 不去除趋势
                - 'mean': 去除均值
                - 'linear': 去除线性趋势
                - 'poly': 多项式去趋势
                - 'loess': 局部加权回归平滑
                - 'wavelet': 小波变换去趋势
                - 'emd': EMD去趋势
                - 'ceemdan': CEEMDAN去趋势
                - 'median': 中值滤波去趋势
                - 'custom': 自定义线性去趋势
            **kwargs: 额外参数
                poly_order: int, 多项式阶数 (默认2)
                loess_frac: float, LOESS窗口比例 (默认0.2)
                wavelet: str, 小波类型 (默认'db4')
                wavelet_level: int, 小波分解层数 (默认3)
                median_kernel: int, 中值滤波核大小 (默认11)
        """
        self.method = method
        self.poly_order = kwargs.get('poly_order', 2)
        self.loess_frac = kwargs.get('loess_frac', 0.2)
        self.wavelet = kwargs.get('wavelet', 'db4')
        self.wavelet_level = kwargs.get('wavelet_level', 3)
        self.median_kernel = kwargs.get('median_kernel', 11)

    def __call__(self, x):
        """对信号进行去趋势处理"""
        if self.method == 'none':
            return x

        elif self.method == 'mean':
            return x - np.mean(x)

        elif self.method == 'linear':
            return detrend(x)

        elif self.method == 'poly':
            return self._detrend_polynomial(x)

        elif self.method == 'loess':
            return self._detrend_loess(x)

        elif self.method == 'wavelet':
            return self._detrend_wavelet(x)

        elif self.method == 'emd':
            return self._detrend_emd(x)

        elif self.method == 'ceemdan':
            return self._detrend_ceemdan(x)

        elif self.method == 'median':
            return self._detrend_median(x)

        elif self.method == 'custom':
            return self._detrend(x)

        else:
            raise ValueError(f'不支持的去趋势方法: {self.method}')

    def _detrend_polynomial(self, x):
        """去除多项式趋势"""
        n = len(x)
        if n <= 1:
            return x

        x_idx = np.linspace(0, 1, n)
        coeffs = np.polyfit(x_idx, x, self.poly_order)
        trend = np.polyval(coeffs, x_idx)
        return x - trend

    def _detrend_loess(self, x):
        """使用LOESS去除趋势"""
        n = len(x)
        if n <= 1:
            return x

        x_idx = np.linspace(0, 1, n)
        window = int(self.loess_frac * n)
        window = max(3, min(window, n))

        step = max(1, n // window)
        x_smooth = np.zeros(n)

        for i in range(0, n, step):
            left = max(0, i - window // 2)
            right = min(n, i + window // 2)

            distances = np.abs(x_idx[left:right] - x_idx[i])
            weights = (1 - (distances / np.max(distances)) ** 3) ** 3

            coeffs = np.polyfit(x_idx[left:right], x[left:right], 1, w=weights)
            x_smooth[i] = np.polyval(coeffs, x_idx[i])

        if step > 1:
            idx_calculated = np.arange(0, n, step)
            f = interp1d(idx_calculated, x_smooth[idx_calculated],
                         kind='cubic', fill_value='extrapolate')
            x_smooth = f(np.arange(n))

        return x - x_smooth

    def _detrend_wavelet(self, x):
        """使用小波变换去除趋势"""
        try:
            import pywt
        except ImportError:
            wayne_print("使用小波去趋势需要安装 pywt 库，请运行: pip install PyWavelets", 'yellow')
            return x

        coeffs = pywt.wavedec(x, self.wavelet, level=self.wavelet_level)
        coeffs[0][:] = 0
        return pywt.waverec(coeffs, self.wavelet)

    def _detrend_emd(self, x):
        """使用EMD去除趋势"""
        try:
            from PyEMD import EMD
        except ImportError:
            wayne_print("使用EMD去趋势需要安装 PyEMD 库，请运行: pip install EMD-signal", 'yellow')
            return x

        emd = EMD()
        imfs = emd(x)
        return x - imfs[-1]

    def _detrend_ceemdan(self, x):
        """使用CEEMDAN去除趋势"""
        try:
            from PyEMD import CEEMDAN
        except ImportError:
            wayne_print("使用CEEMDAN去趋势需要安装 PyEMD 库，请运行: pip install EMD-signal", 'yellow')
            return x

        ceemdan = CEEMDAN()
        imfs = ceemdan(x)
        return x - imfs[-1]

    def _detrend_median(self, x):
        """使用中值滤波去除趋势"""
        trend = medfilt(x, self.median_kernel)
        return x - trend

    def _detrend(self, x):
        """使用自定义方法去除线性趋势"""
        n = len(x)
        if n <= 1:
            return x

        x_idx = np.arange(n, dtype=float)
        mean_x = np.mean(x_idx)
        mean_y = np.mean(x)

        dx = x_idx - mean_x
        dy = x - mean_y

        slope = np.sum(dx * dy) / np.sum(dx * dx) if np.sum(dx * dx) != 0 else 0
        return x - (slope * x_idx + (mean_y - slope * mean_x))


class ButterworthFilter:
    """
    纯 numpy 的 1D IIR 滤波器：
      - lfilter: Direct Form II Transposed（对齐 SciPy 的实现形态）
      - lfilter_zi: 解线性方程得到稳态初始条件
      - filtfilt: pad 方法（odd/even/constant/None），默认 padlen=3*ntaps

    支持两种构造：
      - from_ba(b,a)
      - from_params(order, fs, btype, cutoff)

    cache_zi：
      - True: 构造时预计算 zi
      - False: 不预计算；第一次 filtfilt()/zi() 时再算（lazy）
    """

    def __init__(self, b: np.ndarray, a: np.ndarray, cache_zi: bool = True):
        b = np.asarray(b, dtype=np.float64).ravel()
        a = np.asarray(a, dtype=np.float64).ravel()
        if a.size == 0:
            raise ValueError("a must not be empty")
        if a[0] == 0.0:
            raise ValueError("a[0] must be nonzero")

        # normalize so that a[0] == 1
        if a[0] != 1.0:
            b = b / a[0]
            a = a / a[0]

        self._ntaps = int(max(a.size, b.size))
        self._nstate = self._ntaps - 1
        self._a = self._pad_to_len(a, self._ntaps)
        self._b = self._pad_to_len(b, self._ntaps)

        self._zi = None
        if cache_zi and self._nstate > 0:
            self._zi = self._lfilter_zi_impl(self._b, self._a)

    # ---------- constructors ----------

    @classmethod
    def from_ba(cls, b: Union[np.ndarray, Iterable[float]], a: Union[np.ndarray, Iterable[float]], cache_zi: bool = True) -> "ButterworthFilter":
        return cls(np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64), cache_zi=cache_zi)

    @classmethod
    def from_params(
        cls,
        order: int,
        fs: float,
        btype: str,
        cutoff: Union[float, Tuple[float, float]],
        cache_zi: bool = True,
    ) -> "ButterworthFilter":
        """
        纯 numpy Butterworth 设计（数字域），cutoff 单位 Hz：
          - btype: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
          - cutoff:
              low/high: float
              band*: (low, high)
        """
        b, a = cls._butter_ba(order=order, fs=fs, btype=btype, cutoff=cutoff)
        return cls(b, a, cache_zi=cache_zi)

    # ---------- properties ----------

    @property
    def ba(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._b.copy(), self._a.copy()

    @property
    def ntaps(self) -> int:
        return self._ntaps

    @property
    def nstate(self) -> int:
        return self._nstate

    def zi(self) -> np.ndarray:
        """返回稳态 zi（若未缓存则 lazy 计算）。"""
        if self._nstate <= 0:
            return np.zeros(0, dtype=np.float64)
        if self._zi is None:
            self._zi = self._lfilter_zi_impl(self._b, self._a)
        return self._zi.copy()

    # ---------- public filtering APIs ----------

    def lfilter(
        self,
        x: Union[np.ndarray, Iterable[float]],
        zi: Optional[Union[np.ndarray, Iterable[float]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct Form II Transposed（对齐 SciPy 计算顺序）
        返回 (y, zf)
        """
        x = self._as_f64_1d(x)
        n = self._nstate
        if n <= 0:
            y = (self._b[0] * x).astype(np.float64, copy=False)
            return y, np.zeros(0, dtype=np.float64)

        z = np.zeros(n, dtype=np.float64) if zi is None else self._as_f64_1d(zi).copy()
        if z.size != n:
            raise ValueError(f"zi must have length {n}")

        b = self._b
        a = self._a
        b0 = b[0]
        b1 = b[1:n + 1]
        a1 = a[1:n + 1]

        y = np.empty_like(x, dtype=np.float64)

        if n == 1:
            for k in range(x.size):
                xi = x[k]
                yi = b0 * xi + z[0]
                y[k] = yi
                z[0] = b1[0] * xi - a1[0] * yi
            return y, z

        for k in range(x.size):
            xi = x[k]
            yi = b0 * xi + z[0]
            y[k] = yi
            z[:-1] = z[1:] + b1[:-1] * xi - a1[:-1] * yi
            z[-1] = b1[-1] * xi - a1[-1] * yi

        return y, z


    def filtfilt(
        self,
        x: Union[np.ndarray, Iterable[float]],
        padtype: Optional[str] = "odd",
        padlen: Optional[int] = None,
    ) -> np.ndarray:
        """
        padtype: 'odd' | 'even' | 'constant' | None
        padlen: None => 3*ntaps
        """
        x = self._as_f64_1d(x)

        if padtype is None or padlen == 0:
            edge = 0
        else:
            edge = (3 * self._ntaps) if padlen is None else int(padlen)
            if edge < 0:
                raise ValueError("padlen must be >= 0")
            if edge >= x.size:
                raise ValueError(f"Input length must be > padlen (len(x)={x.size}, padlen={edge})")

        ext = x if edge == 0 else self._pad(x, edge, padtype)

        zi = self.zi()
        y, _ = self.lfilter(ext, zi * ext[0] if zi.size else None)

        y0 = y[-1]
        yrev = y[::-1].copy()
        y2, _ = self.lfilter(yrev, zi * y0 if zi.size else None)
        y2 = y2[::-1]

        return y2[edge:-edge] if edge > 0 else y2

    # ---------- detrend ----------

    @staticmethod
    def detrend(x: Union[np.ndarray, Iterable[float]], method: str = "linear", poly_order: int = 2) -> np.ndarray:
        x = ButterworthFilter._as_f64_1d(x)
        if method == "none":
            return x
        if method == "mean":
            return x - x.mean()
        if method == "linear":
            return ButterworthFilter._detrend_linear(x)
        if method == "poly":
            return ButterworthFilter._detrend_poly(x, poly_order)
        raise ValueError(f"unsupported detrend method: {method}")

    # ============================================================
    #                      internals
    # ============================================================

    @staticmethod
    def _as_f64_1d(x: Union[np.ndarray, Iterable[float]]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("x must be 1D")
        return x

    @staticmethod
    def _pad_to_len(v: np.ndarray, n: int) -> np.ndarray:
        if v.size >= n:
            return v[:n].copy()
        out = np.zeros(n, dtype=np.float64)
        out[:v.size] = v
        return out

    @staticmethod
    def _pad(x: np.ndarray, edge: int, padtype: str) -> np.ndarray:
        if padtype == "odd":
            x0, xN = x[0], x[-1]
            left = 2.0 * x0 - x[1:edge + 1][::-1]
            right = 2.0 * xN - x[-edge - 1:-1][::-1]
            return np.concatenate([left, x, right])
        if padtype == "even":
            left = x[1:edge + 1][::-1]
            right = x[-edge - 1:-1][::-1]
            return np.concatenate([left, x, right])
        if padtype == "constant":
            return np.concatenate([np.full(edge, x[0]), x, np.full(edge, x[-1])]).astype(np.float64)
        raise ValueError("padtype must be 'odd','even','constant', or None")

    @staticmethod
    def _solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        # 小维度线性方程：直接用 numpy.linalg.solve（纯 numpy，易移植时也能换成你自己的 Gauss）
        return np.linalg.solve(A, b)

    @staticmethod
    def _lfilter_zi_impl(b: np.ndarray, a: np.ndarray) -> np.ndarray:
        ntaps = int(max(a.size, b.size))
        n = ntaps - 1
        if n <= 0:
            return np.zeros(0, dtype=np.float64)

        a = a[:ntaps]
        b = b[:ntaps]

        # (I - A), A = companion(a).T
        IA = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            IA[i, i] = 1.0
            IA[i, 0] += a[i + 1]
            if i < n - 1:
                IA[i, i + 1] -= 1.0

        b0 = b[0]
        B = np.array([b[i + 1] - a[i + 1] * b0 for i in range(n)], dtype=np.float64)
        zi = ButterworthFilter._solve_linear(IA, B)
        return zi

    # ---------- detrend helpers ----------

    @staticmethod
    def _detrend_linear(x: np.ndarray) -> np.ndarray:
        n = x.size
        if n <= 1:
            return x - x
        i = np.arange(n, dtype=np.float64)
        mi = i.mean()
        mx = x.mean()
        di = i - mi
        dx = x - mx
        den = float(np.dot(di, di))
        slope = float(np.dot(di, dx)) / den if den != 0.0 else 0.0
        intercept = mx - slope * mi
        return x - (slope * i + intercept)

    @staticmethod
    def _detrend_poly(x: np.ndarray, order: int) -> np.ndarray:
        n = x.size
        if n <= 1:
            return x - x
        if order < 0:
            raise ValueError("poly_order must be >= 0")
        idx = np.linspace(0.0, 1.0, n, dtype=np.float64)
        c = np.polyfit(idx, x, order)
        trend = np.polyval(c, idx)
        return x - trend

    # ============================================================
    #                 Butterworth design (pure numpy)
    # ============================================================

    @staticmethod
    def _butter_ba(order: int, fs: float, btype: str, cutoff: Union[float, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        纯 numpy 设计：目标是让 from_params 的 b/a 与 SciPy butter 尽可能一致
        核心：z/p 变换 + bilinear 后，强制做通带单位增益归一化（解决你现在 b 巨大偏差的问题）
        """
        if order <= 0:
            raise ValueError("order must be positive")
        fs = float(fs)
        if fs <= 0:
            raise ValueError("fs must be > 0")

        btype = btype.lower()
        if btype not in ("lowpass", "highpass", "bandpass", "bandstop"):
            raise ValueError("btype must be one of: lowpass/highpass/bandpass/bandstop")

        fs2 = 2.0 * fs

        def prewarp(f_hz: float) -> float:
            f_hz = float(f_hz)
            if not (0.0 < f_hz < 0.5 * fs):
                raise ValueError("cutoff must satisfy 0 < f < fs/2")
            return fs2 * np.tan(np.pi * f_hz / fs)

        # 1) analog prototype (wc=1 rad/s)
        z, p = ButterworthFilter._buttap_zp(order)

        # 2) analog frequency transform
        if btype in ("lowpass", "highpass"):
            wc = prewarp(float(cutoff))
            if btype == "lowpass":
                z, p = ButterworthFilter._lp2lp_zp(z, p, wc)
                w_norm = 0.0
            else:
                z, p = ButterworthFilter._lp2hp_zp(z, p, wc)
                w_norm = np.pi
        else:
            lo, hi = cutoff  # type: ignore
            w1 = prewarp(float(lo))
            w2 = prewarp(float(hi))
            if w2 <= w1:
                raise ValueError("band cutoff must satisfy low < high")

            w0 = np.sqrt(w1 * w2)
            bw = w2 - w1

            if btype == "bandpass":
                z, p = ButterworthFilter._lp2bp_zp(z, p, w0, bw)
                # analog w0 -> digital center via bilinear: w_d = 2*atan(w0/(2fs))
                w_norm = 2.0 * np.arctan(w0 / fs2)
            else:
                z, p = ButterworthFilter._lp2bs_zp(z, p, w0, bw)
                w_norm = 0.0  # bandstop 通常在 DC 处归一化

        # 3) bilinear transform (analog -> digital)
        z_d, p_d = ButterworthFilter._bilinear_zp(z, p, fs)

        # 4) zpk -> ba（先不算 k，最后统一做增益归一化）
        b = np.poly(z_d)
        a = np.poly(p_d)

        b = np.real_if_close(b, tol=1000).astype(np.float64)
        a = np.real_if_close(a, tol=1000).astype(np.float64)

        # normalize a[0]=1
        b = b / a[0]
        a = a / a[0]

        # 5) 强制通带单位增益归一化（关键修复点）
        b = ButterworthFilter._normalize_passband_gain(b, a, w_norm)

        return b, a

    @staticmethod
    def _normalize_passband_gain(b: np.ndarray, a: np.ndarray, w: float) -> np.ndarray:
        """
        让 |H(e^{jw})| = 1
        H = sum_k b[k] e^{-jwk} / sum_k a[k] e^{-jwk}
        """
        k = np.arange(b.size, dtype=np.float64)
        ej = np.exp(-1j * w * k)
        num = np.dot(b, ej)
        den = np.dot(a, ej)
        H = num / den
        g = 1.0 / (np.abs(H) + 1e-30)
        return (b * g).astype(np.float64)

    @staticmethod
    def _buttap_zp(n: int) -> Tuple[np.ndarray, np.ndarray]:
        # Butterworth analog lowpass prototype: no finite zeros
        z = np.array([], dtype=np.complex128)
        p = np.array([np.exp(1j * np.pi * (2*k + n + 1) / (2*n)) for k in range(n)], dtype=np.complex128)
        return z, p

    @staticmethod
    def _lp2lp_zp(z: np.ndarray, p: np.ndarray, wo: float) -> Tuple[np.ndarray, np.ndarray]:
        return z * wo, p * wo

    @staticmethod
    def _lp2hp_zp(z: np.ndarray, p: np.ndarray, wo: float) -> Tuple[np.ndarray, np.ndarray]:
        degree = p.size - z.size
        z2 = (wo / z) if z.size else np.array([], dtype=np.complex128)
        p2 = wo / p
        if degree > 0:
            z2 = np.concatenate([z2, np.zeros(degree, dtype=np.complex128)])
        return z2, p2

    @staticmethod
    def _lp2bp_zp(z: np.ndarray, p: np.ndarray, wo: float, bw: float) -> Tuple[np.ndarray, np.ndarray]:
        degree = p.size - z.size

        def quad_roots(x):
            t = 0.5 * bw * x
            r = np.sqrt(t*t - wo*wo)
            return np.array([t + r, t - r], dtype=np.complex128)

        z2 = np.concatenate([quad_roots(zz) for zz in z]) if z.size else np.array([], dtype=np.complex128)
        p2 = np.concatenate([quad_roots(pp) for pp in p])

        if degree > 0:
            z2 = np.concatenate([z2, np.zeros(degree, dtype=np.complex128)])
        return z2, p2

    @staticmethod
    def _lp2bs_zp(z: np.ndarray, p: np.ndarray, wo: float, bw: float) -> Tuple[np.ndarray, np.ndarray]:
        degree = p.size - z.size

        def quad_roots_inv(x):
            t = 0.5 * bw / x
            r = np.sqrt(t*t - wo*wo)
            return np.array([t + r, t - r], dtype=np.complex128)

        z2 = np.concatenate([quad_roots_inv(zz) for zz in z]) if z.size else np.array([], dtype=np.complex128)
        p2 = np.concatenate([quad_roots_inv(pp) for pp in p])

        if degree > 0:
            z2 = np.concatenate([
                z2,
                1j * wo * np.ones(degree, dtype=np.complex128),
                -1j * wo * np.ones(degree, dtype=np.complex128),
            ])
        return z2, p2

    @staticmethod
    def _bilinear_zp(z: np.ndarray, p: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        fs2 = 2.0 * fs
        degree = p.size - z.size

        z_d = (fs2 + z) / (fs2 - z) if z.size else np.array([], dtype=np.complex128)
        p_d = (fs2 + p) / (fs2 - p)

        if degree > 0:
            z_d = np.concatenate([z_d, -np.ones(degree, dtype=np.complex128)])  # zeros at infinity -> z=-1
        return z_d, p_d
