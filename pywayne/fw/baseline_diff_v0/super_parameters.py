# encoding: utf-8

"""
定义的全局变量
"""

# 通道数
CH_NUMS = 6

"""baseline 部分"""
# 平稳和斜率帧数
ORDER16 = 16
SNOISE_TH = ORDER16**2
# 平稳条件阈值
# # FORCE_STABLE_SUM_TH3, FORCE_STABLE_CUR_TH3, FORCE_STABLE_PRD_TH3 = (ORDER16 ** 2) >> 1, 5, 20 >> 1
# # FORCE_STABLE_SUM_TH2, FORCE_STABLE_CUR_TH2, FORCE_STABLE_PRD_TH2 = ORDER16 ** 2, 7, 20 >> 1
FORCE_STABLE_SUM_TH1, FORCE_STABLE_CUR_TH1, FORCE_STABLE_PRD_TH1 = (ORDER16 ** 2) << 1, 10, 20
# 每帧最大变化量
CHANGE_LIMIT_PER_FRAME = 1 << 11
# stable_cnt阈值
STABLE_CNT_TH = 3
# 基线追踪迟滞滤波系数 #0.1≈100/1024
ALPHA = 100


"""forceflag 部分"""
# 防止反弹误报键，在此期间增加阈值
FREEZE_LENTH = 50
# force down 信号幅度和两帧变化量阈值
INTEGRAL_TH = 30
SLP_TH = 3

TIME_OUT = 1200
# 离手力和比值阈值
LEAVE_TH = 25
LEAVE_COEFF = 0.3
# 设定的高温离手单帧漂移斜率
K_TREND = 0.2

# 强条件的右峰和右左比阈值
STRONG_RIGHT_TH = 40
STRONG_RATIO_TH = 0.4
# 中条等件的右峰、右左比和计数触发阈值
MEDIAN_RIGHT_TH = 40
MEDIAN_RATIO_TH = 0.3
MEDIAN_OK_COUNT_TH = 30
MEDIAN_TIME_OUT = 200

# 缓慢释放I阶段的两个阈值
A_TH = -40
B_TH = -8

# 统计用，定义缓慢释放和断触
SLOW_FRAME = 40
BREAK_FRAME = 30
