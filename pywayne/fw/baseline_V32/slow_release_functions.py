import numpy as np

from .super_parameters import A_TH,B_TH
from pywayne.fw.utils import FindLocalExtremeValue


def calculate_I_3_params(rawdata, baseline, max_ch=0, min_ch=1):
    flev = FindLocalExtremeValue(local_window=30)
    local_diff_l, gap_to_peak_l, near_min_slope_l, _, peak_forcesig_l = flev.run(rawdata[-32:, max_ch])
    local_diff_r, gap_to_peak_r, near_min_slope_r, _, peak_forcesig_r = flev.run(rawdata[-32:, min_ch])

    local_diffs = np.hstack((local_diff_l, local_diff_r))
    near_min_slopes = np.hstack((near_min_slope_l, near_min_slope_r))
    ratios = np.hstack((
        (rawdata[-1, max_ch] - baseline[0, max_ch]) / (peak_forcesig_l - baseline[0, max_ch] + 1e-8),
        (rawdata[-1, min_ch] - baseline[0, min_ch]) / (peak_forcesig_r - baseline[0, min_ch] + 1e-8)
    ))
    return local_diffs, near_min_slopes, ratios


def calculate_I_rules(rawData, baseline,max_ch):
    if max_ch == 0 or max_ch == 1:
        min_ch = 1 - max_ch
    elif max_ch == 2 or max_ch == 3:
        min_ch = 5 - max_ch
    else:  # max_ch == 4 or max_ch == 5:
        min_ch = 9 - max_ch
    rawdata_only_max_min_ch = np.c_[rawData[:, max_ch], rawData[:, min_ch]]
    baseline_only_max_min_ch = np.c_[baseline[max_ch], baseline[min_ch]]
    local_diffs, near_min_slopes, ratios = \
        calculate_I_3_params(rawdata_only_max_min_ch, baseline_only_max_min_ch)

    a1 = local_diffs[0]
    a2 = local_diffs[1]
    b1 = near_min_slopes[0]
    b2 = near_min_slopes[1]
    c1 = ratios[0]
    c2 = ratios[1]
    d1 = rawData[-1, max_ch] - rawData[-1 - 5, max_ch]
    d2 = rawData[-1, min_ch] - rawData[-1 - 5, min_ch]
    # e1 = rawData[-1, max_ch] - baseline[max_ch]
    # e2 = rawData[-1, min_ch] - baseline[min_ch]
    # max_ch_b = a1 < A_TH and b1 < B_TH and c1 < c_th_1(b1) and d1 < -10 and e1 > 0
    # min_ch_b = a2 < A_TH and b2 < B_TH and c2 < c_th_2(b2) and d2 < -10 and e2 > 0
    max_ch_b = a1 < A_TH and b1 < B_TH and c1 < c_th_1(b1) and d1 < -10 and rawData[-1,max_ch]<rawData[-2,max_ch]
    min_ch_b = a2 < A_TH and b2 < B_TH and c2 < c_th_2(b2) and d2 < -10 and rawData[-1,min_ch]<rawData[-2,min_ch]

    n1, n2, n3, n4 = 2, 2, 2, 2
    max_ch_b_1 = a1 < A_TH / n1 and b1 < B_TH / n1 and c1 < 0.7 and d1 < -10 and rawData[-1,max_ch]<rawData[-2,max_ch]
    min_ch_b_1 = a2 < A_TH / n1 and b2 < B_TH / n1 and c2 < 0.7 and d2 < -10 and rawData[-1,min_ch]<rawData[-2,min_ch]

    if max_ch_b or min_ch_b:
        # print('strong', max_ch)
        # print(a1,b1,c1,d1)
        # print(a2,b2,c2,d2)
        # print(max_ch,rawData[-1, max_ch] , rawData[-2, max_ch],rawData[-1,min_ch],rawData[-2,min_ch])
        return 2  # strong rule
    elif max_ch_b_1 or min_ch_b_1:
        # print('median', max_ch)
        # print(a1,b1,c1,d1)
        # print(a2,b2,c2,d2)
        return 1  # median rule
    else:
        return 0  # 啥也没有发生


def c_th_1(_b):
    add = (_b + 8) * -0.015
    return 0.4 + min(0.1, add)


def c_th_2(_b):
    add = (_b + 8) * -0.015
    return 0.35 + min(0.1, add)


def initialize_slow_release_dict_strong(slow_release_dict):
    slow_release_dict['strong_ready_flag'] = False
    slow_release_dict['strong_rule_cnt'] = -1
    # slow_release_dict['strong_maxch'] = 999
    return


def initialize_slow_release_dict_median(slow_release_dict):
    slow_release_dict['median_rule_cnt'] = -1
    slow_release_dict['median_ok_count'] = 0
    slow_release_dict['median_valley'] = 0
    slow_release_dict['median_left_peak'] = 0
    # slow_release_dict['median_maxch'] = 999
    return


def pop_slow_release_dict_strong(slow_release_dict):
    strong_ready_flag = slow_release_dict['strong_ready_flag']
    strong_rule_cnt = slow_release_dict['strong_rule_cnt']
    strong_maxch = slow_release_dict['strong_maxch']
    return strong_ready_flag, strong_rule_cnt, strong_maxch


def pop_slow_release_dict_median(slow_release_dict):
    median_rule_cnt = slow_release_dict['median_rule_cnt']
    median_ok_count = slow_release_dict['median_ok_count']
    median_valley = slow_release_dict['median_valley']
    median_left_peak = slow_release_dict['median_left_peak']
    median_maxch = slow_release_dict['median_maxch']
    return median_rule_cnt,median_ok_count,median_valley,median_left_peak,median_maxch
