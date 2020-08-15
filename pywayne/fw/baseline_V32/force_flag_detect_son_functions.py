import numpy as np
from .super_parameters import FREEZE_LENTH, INTEGRAL_TH, CH_NUMS, RIGHT_PEAK_T, RL_RATIO_T, MEDIAN_OK_COUNT_T
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


def calc_forceflagzero_params(rawdata, baseline, temperature_err_flag, thr_rebound):
    slpTmp = rawdata[-1] - rawdata[-3]
    integralTmp = rawdata[-1] - baseline
    if temperature_err_flag == 0 and thr_rebound == 0:
        integralTmp = np.int64(integralTmp * [1, 1, 1, 1, 0.4, 0.4])
    maxch = np.argmax(np.abs(integralTmp))
    # AbsintegralMax = np.abs(integralTmp)[maxch] 固件计算了，但后续没有用到
    integralMax = integralTmp[maxch]
    slpTmpMax = slpTmp[maxch]
    return integralMax, slpTmpMax


def update_sigpeak_lastmaxch(rawdata, sig, sig_peak, state_dict):
    """

    :param rawdata:
    :param sig:
    :param sig_peak:
    :param state_dict:
    :return:
    """
    last_maxch = state_dict['last_maxch']
    maxch = np.argmax(sig)
    for ch in range(CH_NUMS):  # 更新 sig_peak
        sig_peak[ch] = sig[ch] if sig[ch] > sig_peak[ch] else sig_peak[ch]
    # 更新last_maxch
    if ((rawdata[-1, maxch] - rawdata[-3, maxch]) * sig[maxch] > 0 and
        sig[maxch] > sig[last_maxch] + INTEGRAL_TH / 2 and
        sig[maxch] > INTEGRAL_TH and sig[maxch] > sig_peak[last_maxch] * 0.3) or last_maxch == -1:
        state_dict['last_maxch'] = maxch
    return


def local_ff_leave_detect_I_II(rawdata, baseline, touch_cnt, slow_release_dict):  # 需要50帧rawdata
    """

    :param rawdata:
    :param baseline:
    :param touch_cnt:
    :param slow_release_dict:
    :return:
    """
    leave_type = 0
    #  计算I阶段参数值
    max_ch = np.argmax(rawdata[-1] - baseline)
    I_rule = calculate_I_rules(rawdata, baseline, max_ch)
    slow_release_dict['debug'] = I_rule
    # I阶段判断
    if I_rule == 2:
        initialize_slow_release_dict_strong(slow_release_dict)
        slow_release_dict['strong_rule_cnt'] = 0
        slow_release_dict['strong_maxch'] = max_ch
        slow_release_dict['strong_ready_flag'] = True
        return leave_type  # 不在这帧离手，以便画图
    if I_rule == 1 or I_rule == 2:
        slow_release_dict['median_maxch'] = max_ch
        if touch_cnt > 100:
            initialize_slow_release_dict_median(slow_release_dict)
            slow_release_dict['median_maxch'] = max_ch
            slow_release_dict['median_rule_cnt'] = 0
        else:
            leave_type = -3.4
        return leave_type  # 不在这帧离手，以便画图

    # 导出参数
    strong_ready_flag, strong_rule_cnt, strong_maxch \
        = pop_slow_release_dict_strong(slow_release_dict)
    median_rule_cnt, median_ok_count, median_valley, median_left_peak, median_maxch \
        = pop_slow_release_dict_median(slow_release_dict)

    #  strong leave
    if strong_ready_flag:
        slow_release_dict['strong_rule_cnt'] = strong_rule_cnt + 1
        if touch_cnt < 100:  # 不做深V
            if rawdata[-1, strong_maxch] > rawdata[-3, strong_maxch] and strong_rule_cnt < 20:
                leave_type = 2
                initialize_slow_release_dict_strong(slow_release_dict)
        else:  # touch_cnt > 100
            if strong_rule_cnt >= 10:
                strong_valley = np.min(rawdata[-10:, strong_maxch])
                strong_left_peak = np.max(rawdata[-40:, strong_maxch]) - strong_valley
                strong_right_peak = np.max(rawdata[-10:, strong_maxch]) - strong_valley
                if strong_right_peak > 40 and strong_right_peak / strong_left_peak > 0.4:  # 反弹太大
                    initialize_slow_release_dict_strong(slow_release_dict)
                    leave_type = -2.1
                else:
                    leave_type = 2
    # median leave
    if median_rule_cnt >= 0:
        median_rule_cnt += 1
        # print('here',median_rule_cnt,rawdata[-1, median_maxch])
        slow_release_dict['median_rule_cnt'] = median_rule_cnt
        if median_rule_cnt == 20:  # 找 left_peak
            median_valley = np.min(rawdata[-20:, median_maxch])
            # print('herehere',median_valley)
            slow_release_dict['median_valley'] = median_valley
            slow_release_dict['median_left_peak'] = np.max(rawdata[-50:, median_maxch]) - median_valley
        elif 20 < median_rule_cnt <= 60:  # V字检测
            median_right_peak = rawdata[-1, median_maxch] - median_valley
            if median_right_peak > RIGHT_PEAK_T or median_right_peak / median_left_peak > RL_RATIO_T:  # V字检测失败
                # print(median_right_peak,median_left_peak,median_maxch,median_valley)
                initialize_slow_release_dict_median(slow_release_dict)
                leave_type = -3.1
        if median_rule_cnt > 30:  # 快速上升检测
            if rawdata[-1, median_maxch] - rawdata[-5, median_maxch] > 30:
                initialize_slow_release_dict_median(slow_release_dict)
                leave_type = -3.2
        if median_rule_cnt == 200:  # time out
            initialize_slow_release_dict_median(slow_release_dict)
            leave_type = -3.3

        if 20 < median_rule_cnt < 200:  # II条件离手检测
            diffs = np.diff(rawdata[-16:, median_maxch])
            slope = np.mean(rawdata[-16:, median_maxch] - rawdata[-32:-16, median_maxch]) / 16
            rule1_1 = np.max(diffs) <= 6
            rule1_2 = slope < 0
            rule2 = np.max(diffs) <= 3
            rule3 = slope < -0.5
            if (rule1_1 and rule1_2) or rule2 or rule3:  # II条件离手检测
                median_ok_count += 1
                slow_release_dict['median_ok_count'] = median_ok_count
                if median_ok_count >= MEDIAN_OK_COUNT_T:  # II条件离手检测完全成功
                    initialize_slow_release_dict_median(slow_release_dict)
                    leave_type = 3
    # return leave_type
    return leave_type


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
