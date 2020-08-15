import numpy as np
from copy import deepcopy

from .super_parameters import *
from .baseline_tracking_son_functions import calc_stable_params, \
    calc_diffs_n, calc_stable_state, limit_amplitude, update_stable_cnt
from .force_flag_detect_son_functions import calc_slpcoef_rebound, calc_forceflagzero_params, \
    pop_state_dict, update_sigpeak_lastmaxch, local_ff_leave_detect_I_II, initialize_sig_peak
from .slow_release_functions import initialize_slow_release_dict_median, initialize_slow_release_dict_strong

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'FangSong'
plt.rcParams['axes.unicode_minus'] = False

leave_type_dict = {0: 'nothing',
                   1: '正常离手',
                   2: '强条件离手',
                   3: '中等条件离手',
                   0.1: '常温力阈值',
                   0.2: '常温力阈值',
                   1.1: '高温力阈值',
                   1.2: '高温比值阈值',
                   1.3: '大于n百帧高温力阈值',
                   1.4: '大于n百帧高温比值阈值',
                   4: '连续点击离手',
                   5: 'time out 离手',
                   -1.01: '触发中等条件,大通道',
                   -1.02: '触发中等条件,小通道',
                   -2.01: '触发强条件,大通道',
                   -2.02: '触发强条件,小通道',
                   -2.1: '强条件-V-kill',
                   -3.1: '中等条件-V-kill',
                   -3.2: '中等条件-快速上升-kill',
                   -3.3: '中等条件-timeout-kill',
                   -3.4: '中等条件-touch_cnt<100-kill'}


class BaseLine:
    def __init__(self, rawdata):
        self.rawdata = rawdata.astype(np.int64)
        self.baseline_sw = deepcopy(self.rawdata)
        self.fbase_sw = (self.baseline_sw * 1024).astype(np.int64)
        self.forceflag_sw = np.zeros_like(self.rawdata[:, 0])
        self.stable_state_sw = np.zeros_like(self.rawdata)
        self.stable_cnt = np.zeros((CH_NUMS,))
        self.sig_peak = np.ones((CH_NUMS,)) * (-1000)
        self.slow_release_dict = {'strong_ready_flag': False,
                                  'strong_rule_cnt': -1,
                                  'strong_maxch': 999,
                                  'median_rule_cnt': -1,
                                  'median_ok_count': 0,
                                  'median_valley': 0,
                                  'median_left_peak': 0,
                                  'median_maxch': 999,
                                  'debug': 0}
        self.state_dict = {'leave_cnt': FREEZE_LENTH + 1,
                           'touch_cnt': 0,
                           'last_peak': 0,
                           'half_drop_cnt': -1,
                           'last_maxch': [-1, -1]}
        self.leave_type = np.zeros_like(self.rawdata[:, 0], dtype=np.float)

    def __plot_slow_rule(self, start_frame, i, only_plot_ch56):
        st, x, y, offset = 4, 2, 1, -3
        if not only_plot_ch56:
            assert (CH_NUMS == 6)
            st, x, y, offset = 0, 3, 2, 1
        if self.slow_release_dict['debug'] == 1:
            ch = self.slow_release_dict['median_maxch']
            plt.subplot(x, y, ch + offset)
            plt.plot(i, self.rawdata[i, ch] - self.rawdata[start_frame, ch], 'go')
        if self.slow_release_dict['debug'] == 2:
            ch = self.slow_release_dict['strong_maxch']
            plt.subplot(3, 2, ch + offset)
            plt.plot(i, self.rawdata[i, ch] - self.rawdata[start_frame, ch], 'r*')

    def __plot_6ch(self, forceflag_true, baseline_fw, stable_state_fw, only_plot_ch56):
        st, x, y, offset = 4, 2, 1, -3
        if not only_plot_ch56:
            assert (CH_NUMS == 6)
            st, x, y, offset = 0, 3, 2, 1
        for ch in range(st, 6):
            plt.subplot(x, y, ch + offset)
            plt.title('ch' + str(ch + 1))
            plt.plot(self.rawdata[:, ch], '--')
            plt.plot(self.baseline_sw[:, ch])
            plt.plot(self.forceflag_sw * (self.rawdata[:, ch].max() - self.rawdata[:, ch].min()) * 0.8
                     + self.rawdata[:, ch].min())
            plt.legend(('rawdata', 'baseline', 'forceflag'))
            if forceflag_true is not None:
                assert (forceflag_true.shape == self.forceflag_sw.shape)
                plt.plot(forceflag_true * (self.rawdata[:, ch].max() - self.rawdata[:, ch].min()) * 0.9
                         + self.rawdata[:, ch].min())
                plt.legend(('rawdata', 'baseline', 'forceflag', 'forceflag_true'))
        if (baseline_fw is None) and (stable_state_fw is None):
            plt.show()  # 两个图画完之后才show

    def __plot_compare(self, baseline_fw, stable_state_fw, start_frame):
        plt.figure()
        for ch in range(CH_NUMS):
            plt.subplot(3, 2, ch + 1)
            plt.title('ch' + str(ch + 1))
            plt.plot(self.rawdata[:, ch] - self.rawdata[start_frame, ch], '--')
            legend = ('rawdata',)
            if baseline_fw is not None:
                assert (baseline_fw.shape == self.baseline_sw.shape)
                plt.plot(self.baseline_sw[:, ch] - self.rawdata[start_frame, ch])
                plt.plot(baseline_fw[:, ch] - baseline_fw[start_frame, ch])
                legend += ('baseline', 'baseline_fw')
            if stable_state_fw is not None:
                assert (stable_state_fw.shape == self.stable_state_sw.shape)
                plt.plot(self.stable_state_sw[:, ch] * np.max(self.rawdata[:, ch] - self.baseline_sw[:, ch]) * 0.5)
                plt.plot((stable_state_fw[:, ch] > 0) * np.max(self.rawdata[:, ch] - self.baseline_sw[:, ch]) * 0.6)
                legend += ('stable_state', 'stable_state_fw')
            plt.legend(legend)
        plt.show()

    def __get_statistic_type(self, forceflag_true, i):
        assert (forceflag_true is not None)
        assert (forceflag_true.shape == self.forceflag_sw.shape)
        true_end_id = np.where(np.diff(forceflag_true) == -1)[0]
        assert (true_end_id.shape[0] == 1)
        if i > true_end_id + SLOW_FRAME:  # 延迟释放
            statistic_type = 1
        elif i < true_end_id - BREAK_FRAME:  # 断触
            statistic_type = 3
        else:
            statistic_type = 0  # 正常离手
        return statistic_type

    def force_flag_detect_I_II(self, rawdata, baseline, fbase, force_flag,
                               temperature_err_flag=1, sig_err_flag=0):
        """
        force flag 判断函数
        :param rawdata:  50帧
        :param baseline: 1帧 np.array
        :param fbase:    1帧 np.array
        :param force_flag:   int
        :param temperature_err_flag:  bool
        :param sig_err_flag:    bool
        :return: forceflagnew: 此帧forceflag判断结果
                 leave_type:  见leave_type_dict
        """
        leave_cnt, touch_cnt, last_peak, last_maxch = pop_state_dict(self.state_dict)
        forceflag_new, leave_type = force_flag, 0
        if force_flag == 0:
            # leave_cnt update
            self.state_dict['leave_cnt'] = leave_cnt + 1
            # 温度异常处理和离手30帧保护
            slopecoef, thr_rebound = calc_slpcoef_rebound(temperature_err_flag, leave_cnt, last_peak)
            # 比较时取最大值
            thr_rebound = max(thr_rebound, slopecoef * INTEGRAL_TH)
            # 信号最大通道值和斜率
            integral_max, slp_tmp_max = \
                calc_forceflagzero_params(rawdata, baseline, temperature_err_flag, leave_cnt)
            # forceflag置起
            if self.state_dict['half_drop_cnt'] < -1:
                self.state_dict['half_drop_cnt'] += 1
                if self.state_dict['half_drop_cnt'] == -1:
                    self.state_dict['leave_cnt'] = 0
                    forceflag_new = 1
            elif integral_max > thr_rebound and slp_tmp_max > slopecoef * SLP_TH and not sig_err_flag:
                self.state_dict['leave_cnt'] = 0
                forceflag_new = 1
        else:  # force_flag == 1
            self.state_dict['touch_cnt'] = touch_cnt + 1
            sig = rawdata[-1] - baseline
            # 更新last_maxch和sig_peak
            update_sigpeak_lastmaxch(sig, self.sig_peak, self.state_dict)
            last_maxch = self.state_dict['last_maxch'][0]
            second_ch = self.state_dict['last_maxch'][1]
            # 常温逻辑
            if not temperature_err_flag:  # 常温逻辑
                if (rawdata[-1, last_maxch] - rawdata[-3, last_maxch]) > 0:
                    if sig[last_maxch] < LEAVE_TH:
                        leave_type = 0.1  # 常温力阈值
                    elif sig[last_maxch] < LEAVE_COEFF * self.sig_peak[last_maxch]:
                        leave_type = 0.2  # 常温比值阈值
            # 高温逻辑
            else:
                # 力和比值阈值离手
                if (rawdata[-1, last_maxch] - rawdata[-3, last_maxch]) > 0:
                    if sig[last_maxch] < min(LEAVE_TH + K_TREND * touch_cnt,
                                             LEAVE_TH + K_TREND * 300):
                        leave_type = 1.1  # 高温力阈值
                    elif sig[last_maxch] < min(LEAVE_COEFF * self.sig_peak[last_maxch], 150):
                        leave_type = 1.2  # 高温比值阈值
                # 两阶段法离手
                if leave_type <= 0:
                    if 80 < touch_cnt < 200:
                        leave_type = \
                            local_ff_leave_detect_I_II(rawdata[:, [last_maxch, second_ch]],
                                                       baseline[[last_maxch, second_ch]],
                                                       self.slow_release_dict, median_method='close')
                    elif 200 < touch_cnt < 300:
                        leave_type = \
                            local_ff_leave_detect_I_II(rawdata[:, [last_maxch, second_ch]],
                                                       baseline[[last_maxch, second_ch]],
                                                       self.slow_release_dict, median_method='half')
                    elif touch_cnt > 300:
                        leave_type = \
                            local_ff_leave_detect_I_II(rawdata[:, [last_maxch, second_ch]],
                                                       baseline[[last_maxch, second_ch]],
                                                       self.slow_release_dict, median_method='full')
                # 连续快速点击
                if touch_cnt < 50 and self.state_dict['half_drop_cnt'] < 0:
                    if sig[last_maxch] < self.sig_peak[last_maxch] * 0.5:
                        self.state_dict['half_drop_cnt'] = 1
                if self.state_dict['half_drop_cnt'] > 0:
                    self.state_dict['half_drop_cnt'] += 1
                    if self.state_dict['half_drop_cnt'] > 30:
                        self.state_dict['half_drop_cnt'] = 0  # 只能做一次，因此不能设为-1
                    elif sig[last_maxch] > self.sig_peak[last_maxch] * 0.6:
                        leave_type = 4

                        # time out
            if leave_type <= 0 and touch_cnt > TIME_OUT:
                leave_type = 5
            if leave_type > 0:  # 离手时处理
                if leave_type == 4:
                    half_drop_cnt = self.state_dict['half_drop_cnt']
                    valley_pos = -half_drop_cnt + np.min(rawdata[-half_drop_cnt:, last_maxch])					
                    self.state_dict['half_drop_cnt'] = -5
                    for ch in range(CH_NUMS):
                        print(half_drop_cnt,valley_pos)
                        baseline[ch] = rawdata[valley_pos, ch]
                        fbase[ch] = rawdata[valley_pos, ch] * 1024
                self.state_dict['touch_cnt'] = 0
                self.state_dict['last_maxch'] = [-1, -1]
                self.state_dict['last_peak'] = np.max(self.sig_peak)
                initialize_slow_release_dict_strong(self.slow_release_dict)
                initialize_slow_release_dict_median(self.slow_release_dict)
                initialize_sig_peak(self.sig_peak)
                if temperature_err_flag and leave_type != 4:
                    self.state_dict['half_drop_cnt'] = -1
                    for ch in range(CH_NUMS):
                        baseline[ch] = rawdata[-2, ch]
                        fbase[ch] = rawdata[-2, ch] * 1024
                forceflag_new = 0
        return forceflag_new, leave_type

    def baseline_tracking(self, rawdata, force_flag, fbase, temperature_err_flag=1):
        """
        基线追踪主func
        :param rawdata:  至少32帧
        :param force_flag: int
        :param fbase:  int
        :param temperature_err_flag:
        :return:
        """
        touch_cnt = self.state_dict['touch_cnt']
        # 分开求各通道稳定状态参数
        slp_vect, slp_cur_abs, slp_prd_abs, slp_sum_abs = calc_stable_params(rawdata)
        # 连续三帧差分
        diffs = calc_diffs_n(rawdata, 4)
        # 计算stable_state
        stable_state = calc_stable_state(slp_sum_abs, slp_cur_abs, slp_prd_abs, diffs)
        # 更新 stable_cnt
        update_stable_cnt(self.stable_cnt, stable_state)  # stable_cnt 会被更新

        # 基线跟踪主逻辑
        if force_flag == 0:
            for ch in range(CH_NUMS):
                if stable_state[ch]:
                    add_temp = (rawdata[-1, ch] - (fbase[ch] >> 10)) * ALPHA
                    fbase[ch] += limit_amplitude(add_temp)  # 限定基线单帧变化量为[-2, 2]
            baseline, fbase_r = fbase >> 10, fbase
        else:  # force_flag[-1] == 1
            if temperature_err_flag and touch_cnt > 50:
                for ch in range(CH_NUMS):
                    if self.stable_cnt[ch] > STABLE_CNT_TH:
                        fbase[ch] += (slp_vect[ch] << 10 // SNOISE_TH).astype(int)
            baseline, fbase_r = fbase >> 10, fbase
        return baseline, fbase_r, stable_state  # 非debug版本可以不返回 stable state

    def run_example(self, start_frame=100, plot_out_flag=False, plot_slow_release_flag=False, suptitle='',
                    forceflag_true=None, statistic_flag=False, baseline_fw=None, stable_state_fw=None,
                    only_plot_ch56=False):
        """
        运行例子，
        可以直接运行run_example函数
        也可以参考此函数，重写对force_flag_detect_I_II和baseline_tracking的调用，将更加灵活
        :param start_frame: 开始计算的点,必须大于50
        :param plot_out_flag:   是否 打印基线和forceflag计算结果
        :param plot_slow_release_flag: plot缓慢释放条件触发点，若为真，则一定打印基线和forceflag计算结果
        :param forceflag_true: forcelfag的参考真值，若不为None, 则plot
        :param statistic_flag: 切片数据统计时用，忽略所有的画图。为真时forceflag_true必须存在
        :param suptitle 图的名字,画图时才有意义
        :param baseline_fw       分析baseline追踪趋势和stable_state时用
        :param stable_state_fw   分析baseline追踪趋势和stable_state时用
        :return:
                 只有在statistic_flag为True时有意义
                 statistic_type: 0. 正常释放
                                 1. 延迟释放
                                 2. 无法释放
                                 3. 断触
        """
        statistic_type = 2  # 默认为无法释放
        assert (start_frame > 3 * ORDER16 + 2)
        if (not statistic_flag) and (plot_slow_release_flag or plot_out_flag):
            plt.figure()
            plt.suptitle(suptitle)
        # 主逻辑
        for i in range(start_frame, np.shape(self.rawdata)[0]):
            # 1. 更新 forceflag
            self.forceflag_sw[i], self.leave_type[i] = \
                self.force_flag_detect_I_II(self.rawdata[(i - 3 * ORDER16 - 2):i + 1], self.baseline_sw[i - 1],
                                            self.fbase_sw[i - 1], self.forceflag_sw[i - 1],
                                            temperature_err_flag=1, sig_err_flag=0)
            if statistic_flag:
                if self.leave_type[i] > 0:
                    return self.__get_statistic_type(forceflag_true, i)
            elif plot_slow_release_flag:
                self.__plot_slow_rule(start_frame, i, only_plot_ch56)
            if (not statistic_flag) and self.leave_type[i] != 0:
                print("index: %d, leave type:%4.1f,%s" % (i, self.leave_type[i], leave_type_dict[self.leave_type[i]]))
            # 2. 更新 baseline
            self.baseline_sw[i], self.fbase_sw[i], self.stable_state_sw[i] = \
                self.baseline_tracking(self.rawdata[i - 2 * ORDER16 + 1:i + 1],
                                       self.forceflag_sw[i], self.fbase_sw[i - 1],
                                       temperature_err_flag=1)
        # 画图
        if (not statistic_flag) and (plot_slow_release_flag or plot_out_flag):
            self.__plot_6ch(forceflag_true, baseline_fw, stable_state_fw, only_plot_ch56)
        if (baseline_fw is not None) or (stable_state_fw is not None):
            self.__plot_compare(baseline_fw, stable_state_fw, start_frame)
        return statistic_type


if __name__ == '__main__':
    from .test_baseline import test_baseline

    test_baseline()
