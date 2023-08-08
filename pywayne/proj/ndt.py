# !/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: Wang Ye (Wayne)
@file: ndt.py
@time: 2022/03/01
@contact: wangye@oppo.com
@site: 
@software: PyCharm
# code is far away from bugs.
"""

import functools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from dsp import peak_det, butter_bandpass_filter


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
        print(f'{func.__name__} excute in {time.time() - start:.3f} s')
        return r

    return wrapper


class ForceSensorDataProcessing:
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
    dp = ForceSensorDataProcessing(rawdata, f)
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
        self.force_val = np.zeros_like(data)
        self.energy = None
        self.flag = None
        self.force = None
        self.tds = None
        self.tus = None
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
        self.tus = np.array(np.where(np.diff(self.flag) == -1))[0]
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

    @func_timer
    def calc_force_value(self):
        self.tds = np.array(np.where(np.diff(self.flag) == 1))[0]
        if np.ndim(self.force_signal) == 1:
            self.force_signal = self.force_signal[:, np.newaxis]
        self.force_val = np.zeros_like(self.force_signal)
        for ii in range(self.tus.shape[0]):
            self.force_val[self.tds[ii]:self.tus[ii]] = self.force_signal[self.tds[ii]:self.tus[ii]]


class ForceSensorExtractRollingData:
    """
    用于从滑动数据中提取数据

    Author:   wangye
    Datetime: 2019/5/22 02:41

    example:
    rd = ForceSensorExtractRollingData(rawdata, f)
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

            1. Shapiro-Wilk test
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


class GeneratePrivateModel:
    """
    根据校准点数据和公共模板，生成私有模板

    Author:   Shine Lin
    Datetime: 2019/8/9 21:42

    校准点规模为1 self.X Chanels，公共模板规模为 points self.X Chanels
    example:
        ways = 'int42'
        MODEL = GeneratePrivateModel(model)
        pri_model, model, data_X, data = MODEL.generate_model(data,ways,400)
        MODEL.plot_model(root_Cali, path_Cali,title_Cali,title_Model)
        MODEL.save_private_model(root_Cali,path_Cali)
    解释：
        其中pri_model是私有模板,model是处理后的公共模板,data（ch*ch）校准点数据，默认校准砝码400g，data_X为data的横坐标；
        ways为拟合方法，'app22','app32'为逼近型方法，’int42','int33'为插值型方法，不指定时默认为'int42'；
        MODEL.plot_model(root_Cali, path_Cali,title_Cali,title_Model)为画图函数，
        root_Cali, path_Cali分别为校准点的主目录和各个校准文件的目录，
        title_Cali,title_Model分别为各个对应的标题，可省略；
        MODEL.save_private_model(root_Cali,path_Cali)保存私有模板数据，保存在root_Cali之下。
    """

    def __init__(self, model):
        self.model = model
        self.iterations = 9
        self.ways = 'int42'
        self.chanels = model.shape[1]
        self.colors = ['r', 'b', 'g', 'orange', 'pink', 'purple', 'gray', 'k', 'skyblue', 'darkgoldenrod', 'c']
        self.fonts = {'family': 'serif',
                      'style': 'normal',
                      'weight': 'bold',
                      'color': 'black',
                      'size': 10
                      }

    def model_process(self):
        # 剔除首末小于50%的顶点
        ch_min = np.argmin(np.argmax(self.model, axis=0))
        ch_max = np.argmax(np.argmax(self.model, axis=0))
        for num_point in range(0, self.model.shape[0]):
            if self.model[num_point, ch_min] >= np.max(self.model[:, ch_min]) / 2:
                [model0, self.model] = np.split(self.model, [num_point], axis=0)
                break
        for num_point in range(0, self.model.shape[0])[::-1]:
            if self.model[num_point, ch_max] >= np.max(self.model[:, ch_max]) / 2:
                [self.model, model1] = np.split(self.model, [num_point + 1], axis=0)
                break

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

    def generate_model(self, data, ways='int42', Cali_weight=400, model_door=True, iterations=9):
        self.iterations = iterations
        self.ways = ways
        self.data = data
        self.Coef = Cali_weight / np.max(self.data, axis=0)
        for ch in range(0, self.chanels):
            self.model[:, ch] = self.model[:, ch] / np.max(np.max(self.model[:, ch])) * 1000
        if model_door:
            # 剔除首末小于50%的顶点
            self.model_process()
            self.model = butter_bandpass_filter(self.model)
        model_x = np.arange(0, self.model.shape[0])
        self.data_X = np.arange(0, self.data.shape[0])
        for ch in range(0, self.chanels):
            self.data[:, ch] = self.data[:, ch] / np.max(np.max(self.data[:, ch])) * 1000
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
        self.pri_model = self.find_axis(model_x)
        print('shape of private_model:')
        print(self.pri_model.shape)
        return self.pri_model, self.model, self.data_X, self.data

    def plot_model(self, root_Cali, path_Cali, title_Cali='title_Cali', title_Model='title_model'):
        fig, ax = plt.subplots((self.chanels + 1) // 2, 2, sharex='all', sharey='all', figsize=(20, 9))
        ax = ax.flatten()
        coss = np.zeros(self.chanels)
        plt.suptitle('title_Cali：' + title_Cali + '\n' + 'title_Model：' + title_Model + '\n' + 'Ways：' + self.ways,
                     fontsize=18)
        CH_label = ['CH' + str(i) for i in range(1, self.chanels + 1)]
        legend_name = ['Pri_model', 'Pub_model', 'Cali_points']
        for ch in range(self.chanels):
            ax[ch].set_title(CH_label[ch], fontsize=14)
            ax[ch].plot(self.pri_model[:, ch], 'r-*', markersize=4)
            ax[ch].plot(self.model[:, ch], 'b-o', markersize=4)
            ax[ch].plot(self.data_X, self.data[:, ch], 'ko', markersize=6)
            coss[ch] = round(np.dot(self.model[:, ch], self.pri_model[:, ch]) / (
                    np.linalg.norm(self.model[:, ch], ord=2, keepdims=False) *
                    np.linalg.norm(self.pri_model[:, ch], ord=2, keepdims=False)), 5)
            ax[ch].legend(['Pri_model', 'Pub_model', 'Cali_points'], fontsize=12)

        print('各个chanel的拟合度:')
        print(coss)

        force_max_index = np.array(np.argmax(self.pri_model, axis=0), dtype=int)
        force_max = np.array(np.max(self.pri_model, axis=0), dtype=int)
        xticks = np.concatenate((np.array([0]), force_max_index, np.array([self.pri_model.shape[0] - 1])), axis=0)
        force_max_half = []
        for ch in range(self.chanels):
            counts = 0
            for x in self.pri_model[:, ch]:
                if x >= force_max[ch] / 2:
                    counts = counts + 1
            force_max_half.append(counts)
        plt.figure(figsize=(16, 9))
        for ch in range(self.chanels):
            plt.plot(self.pri_model[:, ch], '-o', markersize=4)
        plt.plot(force_max_index, force_max, 'k*', markersize=10)
        plt.title('All Chanels', fontsize=14)
        CH_label.append('(Coef,Width)')
        plt.legend(CH_label, fontsize=10, loc='lower right')
        plt.xticks(xticks, size=16)
        plt.xlabel('Position', fontsize=16)
        plt.ylabel('ADC', fontsize=20)
        for ch in range(self.chanels):
            if force_max[ch] > 30:
                text_value = '(' + str(np.round(self.Coef, 2)[ch]) + ',' + str(force_max_half[ch]) + ')'
                plt.text(force_max_index[ch], force_max[ch] * 1.02, text_value, fontdict=self.fonts)
                plt.vlines(force_max_index[ch], np.min(self.pri_model[force_max_index[ch]]),
                           np.max(self.pri_model[force_max_index[ch]]), color='k', linestyle='--')
        plt.hlines(0, 0, self.pri_model.shape[0] - 1, color='k')
        plt.grid()
        print('Save picture succeed!')
        plt.show()
        plt.close()

