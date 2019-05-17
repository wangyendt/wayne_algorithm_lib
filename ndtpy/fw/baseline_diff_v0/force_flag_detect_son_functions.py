import numpy as np
from .super_parameters import FREEZE_LENTH, CH_NUMS


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
        slopecoef = 2
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
