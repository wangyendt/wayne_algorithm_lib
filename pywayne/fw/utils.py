# coding=utf-8

import numpy as np
import pandas as pd


class FindLocalExtremeValue:
    """
    本类用于寻找局部极值

    Author:   wangye
    Datetime: 2019/4/16 15:52
    """

    def __init__(self,
                 local_window=30,
                 min_valid_slp_thd=7,
                 slp_step=3
                 ):
        """
        :param local_window: 表示寻找极值的窗长，仅寻找数据x中最后local_window
                            长度的数据相对于x最后一帧的极值。
        :param min_valid_slp_thd: 要成为合理的局部极值，至少要与被比较者相差的值
        :param slp_step: 找当前值和极值之间的最值斜率，斜率计算的步长
        """
        self.local_window = local_window
        self.min_valid_slp_thd = min_valid_slp_thd
        self.min_slp_step = slp_step
        self.max_slp_step = slp_step

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
                第四个local_extreme_real_value是指局部极值实际的值
        """

        assert self.local_window + 2 <= x.shape[0]
        x = x.astype(np.float)
        m = len(x)

        local_diff, local_gap, min_slope, max_slope, local_extreme_real_value = 0, 0, np.inf, -np.inf, 0
        local_x = 0
        update_num = 0
        for j in range(1, self.local_window):
            if j + 2 * self.min_slp_step <= m:
                min_slope = np.min((
                    min_slope,
                    (x[-j - self.min_slp_step:-j].mean() -
                     x[-j - 2 * self.min_slp_step:-j - self.min_slp_step].mean()
                     ) / self.min_slp_step
                ))
                max_slope = np.max((
                    max_slope,
                    (x[-j - self.max_slp_step:-j].mean() -
                     x[-j - 2 * self.max_slp_step:-j - self.max_slp_step].mean()
                     ) / self.max_slp_step
                ))
            a = x[-1] - x[-1 - j]
            b = x[-1] - x[-1 - j - 1]
            c = x[-1] - x[-1 - j - 2]
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
            if update_num > 4:
                break
        local_diff = local_x
        return local_diff, local_gap, min_slope, max_slope, local_extreme_real_value
