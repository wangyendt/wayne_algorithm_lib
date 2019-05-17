import numpy as np
from .super_parameters import *

"""
这些函数用于 baseline_tracking function
"""


def calc_stable_params(rawdata):
    """
    
    :param rawdata:
    :return:
    """
    slp_vect = rawdata[-16:] - rawdata[-32:-16]
    slp_cur_abs = np.abs(rawdata[-1] - rawdata[-1 - 2])
    slp_prd_abs = np.max(np.abs(rawdata[-16:] - rawdata[-18:-2]), axis=0)
    slp_vect = np.sum(slp_vect, axis=0)
    slp_sum_abs = np.abs(slp_vect)
    return slp_vect,slp_cur_abs,slp_prd_abs,slp_sum_abs


def calc_diffs_n(rawdata,n=4):
    diffs = np.diff(rawdata[-n:], axis=0)
    return diffs


def calc_stable_state(slp_sum_abs, slp_cur_abs, slp_prd_abs, diffs):
    stable_flag = np.array([0]*CH_NUMS)
    for ch in range(CH_NUMS):
        non_monotonous_flag = not ( np.all(diffs[:,ch]>1) or np.all(diffs[:,ch]<-1) )   # 三帧差分不同向
        if slp_sum_abs[ch] < FORCE_STABLE_SUM_TH1 and slp_cur_abs[ch] < FORCE_STABLE_CUR_TH1\
                and slp_prd_abs[ch] < FORCE_STABLE_PRD_TH1 and non_monotonous_flag:
            stable_flag[ch] = 1  # 平稳
        else:
            stable_flag[ch] = 0  # 不平稳
    return stable_flag


def update_stable_cnt(stable_cnt,stable_state):
    for ch in range(CH_NUMS):
        # 计算stable_cnt
        if stable_state[ch]:
            stable_cnt[ch] += 1
        else:
            stable_cnt[ch] = 0
    return None


def limit_amplitude(add_temp):
    if add_temp > (1 << 11):
        add_temp = (1 << 11)
    elif add_temp < (-1 << 11):
        add_temp = (-1 << 11)
    return add_temp