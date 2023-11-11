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

from sortedcontainers import SortedList

import numpy as np
from scipy.signal import butter, lfilter, filtfilt


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
    """
    常用于pose tracking等场景
    低速时去抖，高速时紧跟
    https://gery.casiez.net/1euro/
    https://gery.casiez.net/1euro/InteractiveDemo/
    """

    def __init__(self, te, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self._val = None
        self._dx = 0
        self._te = te
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff
        self._alpha = self._alpha(self._mincutoff)
        self._dalpha = self._alpha(self._dcutoff)

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / self._te)

    def apply(self, val: float, te: float) -> float:
        result = val
        if self._val is not None:
            edx = (val - self._val) / te
            self._dx = self._dx + (self._dalpha * (edx - self._dx))
            cutoff = self._mincutoff + self._beta * abs(self._dx)
            self._alpha = self._alpha(cutoff)
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
