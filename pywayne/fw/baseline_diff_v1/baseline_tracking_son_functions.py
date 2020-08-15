import numpy as np
from .super_parameters import *

"""
这些函数用于 baseline_tracking function
"""


def calc_stable_params(rawdata):
    """
    用于计算2*ORDER16帧斜率，以及平稳条件参数
    :param rawdata:  （>=2*ORDER16，CH_NUM）
    :return: 所有返回值shape都为(CH_NUMS,)
            slp_vect      ORDER16个ORDER16帧差分之和,除以ORDER16**2为斜率
            slp_cur_abs   最后1帧的2帧差分绝对值
            slp_prd_abs   ORDER16个2帧差分绝对值的最大值
            slp_sum_abs  slp_vect的绝对值
    """
    slp_cur_abs = np.abs(rawdata[-1] - rawdata[-1 - 2])
    slp_prd_abs = np.max(np.abs(rawdata[-16:] - rawdata[-ORDER16 - 2:-2]), axis=0)

    slp_vect = rawdata[-ORDER16:] - rawdata[-ORDER16 * 2:-ORDER16]
    slp_vect = np.sum(slp_vect, axis=0)
    slp_sum_abs = np.abs(slp_vect)
    return slp_vect, slp_cur_abs, slp_prd_abs, slp_sum_abs


def calc_diffs_n(rawdata, n=4):
    """
    计算连续n帧的差分
    :param rawdata: (n,CH_NUM)
    :param n: int
    :return: diffs (n-1,CH_NUM)
    """
    diffs = np.diff(rawdata[-n:], axis=0)
    return diffs


def calc_stable_state(slp_sum_abs, slp_cur_abs, slp_prd_abs, diffs):
    """
    判断rawdata 是否平稳
    :param slp_sum_abs:  （CH_NUM,）
    :param slp_cur_abs: （CH_NUM,）
    :param slp_prd_abs: （CH_NUM,）
    :param diffs:  （CH_NUM,）
    :return:   （CH_NUM,）
    """
    stable_flag = np.zeros_like(slp_sum_abs, dtype=np.bool)  # 不平稳
    for ch in range(CH_NUMS):
        non_monotonous_flag = not (np.all(diffs[:, ch] > 1) or np.all(diffs[:, ch] < -1))  # 三帧差分不同向
        if slp_sum_abs[ch] < FORCE_STABLE_SUM_TH1 and slp_cur_abs[ch] < FORCE_STABLE_CUR_TH1 \
                and slp_prd_abs[ch] < FORCE_STABLE_PRD_TH1 and non_monotonous_flag:
            stable_flag[ch] = True  # 平稳
    return stable_flag


def update_stable_cnt(stable_cnt, stable_state):
    for ch in range(CH_NUMS):
        if stable_state[ch]:
            stable_cnt[ch] += 1
        else:
            stable_cnt[ch] = 0
    return None


def limit_amplitude(add_temp):
    if add_temp > CHANGE_LIMIT_PER_FRAME:
        add_temp = CHANGE_LIMIT_PER_FRAME
    elif add_temp < -CHANGE_LIMIT_PER_FRAME:
        add_temp = -CHANGE_LIMIT_PER_FRAME
    return add_temp
