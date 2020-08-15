import numpy as np
from .super_parameters import FREEZE_LENTH, CH_NUMS, ORDER16, \
    STRONG_RIGHT_TH, STRONG_RATIO_TH, \
    MEDIAN_RATIO_TH, MEDIAN_RIGHT_TH, MEDIAN_OK_COUNT_TH, MEDIAN_TIME_OUT
from .slow_release_functions import calculate_I_rules, \
    pop_slow_release_dict_median, pop_slow_release_dict_strong, \
    initialize_slow_release_dict_median, initialize_slow_release_dict_strong


def calc_slpcoef_rebound(temperature_err_flag, leave_cnt, lastpeak):
    """

    :param temperature_err_flag:
    :param leave_cnt:
    :param lastpeak:
    :return:
    """
    slopecoef, thr_rebound = 1, 0
    # 基于情景加严触发forceflag的条件
    if leave_cnt < FREEZE_LENTH:  # 刚离手后的20帧保护
        slopecoef = 2
        thr_rebound = min(int(lastpeak * 0.3), 250)
    if temperature_err_flag:  # 温度异常，加严force条件
        slopecoef = 3
    return slopecoef, thr_rebound


def calc_forceflagzero_params(rawdata, baseline, temperature_err_flag, leave_cnt):
    """
    计算信号最大通道值和斜率
    :param rawdata:
    :param baseline:
    :param temperature_err_flag:
    :param leave_cnt:
    :return:
    """
    slpTmp = rawdata[-1] - rawdata[-3]
    integralTmp = rawdata[-1] - baseline
    if temperature_err_flag == 0 and leave_cnt > FREEZE_LENTH:
        integralTmp = np.int64(integralTmp * [1, 1, 1, 1, 0.4, 0.4])
    maxch = np.argmax(np.abs(integralTmp))
    integralMax = integralTmp[maxch]
    slpTmpMax = slpTmp[maxch]
    return integralMax, slpTmpMax


def update_sigpeak_lastmaxch(sig, sig_peak, state_dict):
    """
    
    :param sig: 
    :param sig_peak: 
    :param state_dict: 
    :return: 
    """""
    for ch in range(CH_NUMS):  # 更新 sig_peak
        sig_peak[ch] = sig[ch] if sig[ch] > sig_peak[ch] else sig_peak[ch]
        # 更新last_maxch
        maxch = np.argmax(sig_peak)
        if maxch == 0 or maxch == 1:
            state_dict['last_maxch'] = [maxch, 1 - maxch]
        elif maxch == 2 or maxch == 3:
            state_dict['last_maxch'] = [maxch, 5 - maxch]
        else:  # maxch == 4 or maxch == 5:
            state_dict['last_maxch'] = [maxch, 9 - maxch]
        state_dict['last_peak'] = sig_peak[maxch]
    return


def copy_valley_rawdata_to_base(rawdata):
    baseline, fbase = rawdata[-3], rawdata[-3] * 1024
    return baseline, fbase


def initialize_sig_peak(sig_peak):
    for ch in range(CH_NUMS):
        sig_peak[ch] = -1000
    return


def initialize_state_dict(state_dict):
    state_dict['leave_cnt'] = 0
    state_dict['touch_cnt'] = 0
    state_dict['last_peak'] = 0
    state_dict['last_maxch'] = -1
    return


def pop_state_dict(state_dict):
    leave_cnt = state_dict['leave_cnt']
    touch_cnt = state_dict['touch_cnt']
    last_peak = state_dict['last_peak']
    last_maxch = state_dict['last_maxch']
    return leave_cnt, touch_cnt, last_peak, last_maxch


def local_ff_leave_detect_I_II(rawdata, baseline, slow_release_dict, median_method='full'):  # 需要50帧rawdata
    """

    :param rawdata:
    :param baseline:
    :param slow_release_dict:
    :param median_method:   'close', 不做median
                            'half',  不增加median_ok_count
                            'full',  功能全开,default
    :return:
    """
    leave_type = 0
    #  计算I阶段参数值
    max_ch = np.argmax(rawdata[-1] - baseline)
    I_rule = calculate_I_rules(rawdata, baseline, max_ch)
    slow_release_dict['debug'] = I_rule
    # I阶段判断
    if I_rule > 2:
        strong_maxch = max_ch if I_rule < 2.015 else 1 - max_ch
        initialize_slow_release_dict_strong(slow_release_dict)
        slow_release_dict['strong_rule_cnt'] = 0
        slow_release_dict['strong_maxch'] = strong_maxch
        slow_release_dict['strong_ready_flag'] = True
        leave_type = - I_rule
        return leave_type  # 不在这帧离手，以便画图
    if I_rule > 1 and (median_method == 'full' or median_method == 'half'):
        median_maxch = max_ch if I_rule < 1.015 else 1 - max_ch
        slow_release_dict['median_maxch'] = median_maxch
        initialize_slow_release_dict_median(slow_release_dict)
        slow_release_dict['median_rule_cnt'] = 0
        leave_type = - I_rule
        return leave_type  # 不在这帧离手，以便画图

    # 导出参数
    strong_ready_flag, strong_rule_cnt, strong_maxch \
        = pop_slow_release_dict_strong(slow_release_dict)
    median_rule_cnt, median_ok_count, median_valley, median_left_peak, median_maxch \
        = pop_slow_release_dict_median(slow_release_dict)

    #  strong leave
    if strong_ready_flag:
        slow_release_dict['strong_rule_cnt'] = strong_rule_cnt + 1
        if strong_rule_cnt <= 10:
            strong_valley = np.min(rawdata[-10:, strong_maxch])
            valley_pos = np.argmin(rawdata[-10 - strong_rule_cnt:, strong_maxch])
            strong_left_peak = np.max(rawdata[-40 - strong_rule_cnt:, strong_maxch]) - strong_valley
            strong_right_peak = np.max(rawdata[-10 - strong_rule_cnt + valley_pos:, strong_maxch]) - strong_valley
            if strong_right_peak > STRONG_RIGHT_TH and \
                    strong_right_peak / strong_left_peak > STRONG_RATIO_TH:  # V字检测
                initialize_slow_release_dict_strong(slow_release_dict)
                initialize_slow_release_dict_median(slow_release_dict)
                median_rule_cnt = -1
                leave_type = -2.1
        elif strong_rule_cnt >= 11:
            leave_type = 2

    # median leave
    if median_rule_cnt >= 0:
        median_rule_cnt += 1
        slow_release_dict['median_rule_cnt'] = median_rule_cnt
        # 找 left_peak和谷
        if median_rule_cnt == 10:
            median_valley = np.min(rawdata[-10:, median_maxch])
            slow_release_dict['median_valley'] = median_valley
            slow_release_dict['median_left_peak'] = np.max(rawdata[-40:, median_maxch]) - median_valley
        # V字检测
        elif 10 < median_rule_cnt <= 50:
            median_right_peak = rawdata[-1, median_maxch] - median_valley
            if median_right_peak > MEDIAN_RIGHT_TH or \
                    median_right_peak / median_left_peak > MEDIAN_RATIO_TH:  # V字检测
                initialize_slow_release_dict_median(slow_release_dict)
                leave_type = -3.1
        # 快速上升检测
        if median_rule_cnt > 30 and \
                rawdata[-1, median_maxch] - rawdata[-5, median_maxch] > 30:
            initialize_slow_release_dict_median(slow_release_dict)
            leave_type = -3.2
        # time out
        if median_rule_cnt == MEDIAN_TIME_OUT:
            initialize_slow_release_dict_median(slow_release_dict)
            leave_type = -3.3
        #   # II条件离手检测
        if 20 < median_rule_cnt < MEDIAN_TIME_OUT and median_maxch == 'full':
            diffs = np.diff(rawdata[-2 * ORDER16:, median_maxch])
            slope = np.mean(rawdata[-ORDER16:, median_maxch] - rawdata[-2 * ORDER16:-ORDER16, median_maxch]) / ORDER16
            rule1_1 = np.max(diffs) <= 6
            rule1_2 = slope < - 0.1
            rule2 = np.max(diffs) <= 3
            rule3 = slope < -0.5
            if (rule1_1 and rule1_2) or rule2 or rule3:  # II条件离手检测
                median_ok_count += 1
                slow_release_dict['median_ok_count'] = median_ok_count
                if median_ok_count >= MEDIAN_OK_COUNT_TH:  # II条件离手检测完全成功
                    initialize_slow_release_dict_median(slow_release_dict)
                    leave_type = 3
    return leave_type
