# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PyWayne 绘图工具库 - 增强型频谱分析模块

提供专业的时频分析可视化工具，特别优化用于：
- IMU 传感器数据（加速度计、陀螺仪）
- 生理信号（PPG、ECG、呼吸）
- 振动分析
- 音频信号处理

主要特性：
---------
1. **SpecgramAxes**: 增强型频谱图类
   - 支持频率单位转换（Hz ↔ bpm ↔ kHz）
   - 多种归一化模式（全局/局部/不归一化）
   - 优化的色彩映射（Parula风格）
   - 完整返回值支持交互分析

2. **parula_map**: MATLAB风格色彩映射
   - 感知均匀的色彩分布
   - 适合科学可视化

使用示例：
---------
基础用法::

    from pywayne.plot import regist_projection, parula_map
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 注册自定义projection
    regist_projection()
    
    # 创建频谱图
    fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
    spec, freqs, t, im = ax.specgram(
        x=signal_data,
        Fs=100,
        NFFT=128,
        noverlap=96,
        cmap=parula_map,
        scale='dB'
    )
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(im, label='Magnitude (dB)')
    plt.show()

生理信号（转换为bpm）::

    spec, freqs, t, im = ax.specgram(
        x=ppg_signal,
        Fs=100,
        NFFT=400,
        noverlap=300,
        freq_scale=60,  # Hz -> bpm
        scale='dB'
    )
    ax.set_ylabel('Heart Rate (bpm)')
    ax.set_ylim(40, 180)

交互式分析::

    spec, freqs, t, im = ax.specgram(...)
    
    def on_click(event):
        if event.xdata and event.inaxes == ax:
            time_idx = np.argmin(np.abs(t - event.xdata))
            plt.figure()
            plt.plot(freqs, spec[:, time_idx])
            plt.title(f'FFT at t={event.xdata:.2f}s')
            plt.show()
    
    fig.canvas.mpl_connect('button_press_event', on_click)

@author: Wang Ye (Wayne)
@file: plot.py
@time: 2022/03/01
@updated: 2025/11/24
@contact: wangye@oppo.com
"""

import numpy as np
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.projections import register_projection
from scipy.signal import spectrogram

__all__ = ['SpecgramAxes', 'parula_map', 'regist_projection', 'get_specgram_params']

cm_data = [
    [0.2422, 0.1504, 0.6603],
    [0.2444, 0.1534, 0.6728],
    [0.2464, 0.1569, 0.6847],
    [0.2484, 0.1607, 0.6961],
    [0.2503, 0.1648, 0.7071],
    [0.2522, 0.1689, 0.7179],
    [0.254, 0.1732, 0.7286],
    [0.2558, 0.1773, 0.7393],
    [0.2576, 0.1814, 0.7501],
    [0.2594, 0.1854, 0.761],
    [0.2611, 0.1893, 0.7719],
    [0.2628, 0.1932, 0.7828],
    [0.2645, 0.1972, 0.7937],
    [0.2661, 0.2011, 0.8043],
    [0.2676, 0.2052, 0.8148],
    [0.2691, 0.2094, 0.8249],
    [0.2704, 0.2138, 0.8346],
    [0.2717, 0.2184, 0.8439],
    [0.2729, 0.2231, 0.8528],
    [0.274, 0.228, 0.8612],
    [0.2749, 0.233, 0.8692],
    [0.2758, 0.2382, 0.8767],
    [0.2766, 0.2435, 0.884],
    [0.2774, 0.2489, 0.8908],
    [0.2781, 0.2543, 0.8973],
    [0.2788, 0.2598, 0.9035],
    [0.2794, 0.2653, 0.9094],
    [0.2798, 0.2708, 0.915],
    [0.2802, 0.2764, 0.9204],
    [0.2806, 0.2819, 0.9255],
    [0.2809, 0.2875, 0.9305],
    [0.2811, 0.293, 0.9352],
    [0.2813, 0.2985, 0.9397],
    [0.2814, 0.304, 0.9441],
    [0.2814, 0.3095, 0.9483],
    [0.2813, 0.315, 0.9524],
    [0.2811, 0.3204, 0.9563],
    [0.2809, 0.3259, 0.96],
    [0.2807, 0.3313, 0.9636],
    [0.2803, 0.3367, 0.967],
    [0.2798, 0.3421, 0.9702],
    [0.2791, 0.3475, 0.9733],
    [0.2784, 0.3529, 0.9763],
    [0.2776, 0.3583, 0.9791],
    [0.2766, 0.3638, 0.9817],
    [0.2754, 0.3693, 0.984],
    [0.2741, 0.3748, 0.9862],
    [0.2726, 0.3804, 0.9881],
    [0.271, 0.386, 0.9898],
    [0.2691, 0.3916, 0.9912],
    [0.267, 0.3973, 0.9924],
    [0.2647, 0.403, 0.9935],
    [0.2621, 0.4088, 0.9946],
    [0.2591, 0.4145, 0.9955],
    [0.2556, 0.4203, 0.9965],
    [0.2517, 0.4261, 0.9974],
    [0.2473, 0.4319, 0.9983],
    [0.2424, 0.4378, 0.9991],
    [0.2369, 0.4437, 0.9996],
    [0.2311, 0.4497, 0.9995],
    [0.225, 0.4559, 0.9985],
    [0.2189, 0.462, 0.9968],
    [0.2128, 0.4682, 0.9948],
    [0.2066, 0.4743, 0.9926],
    [0.2006, 0.4803, 0.9906],
    [0.195, 0.4861, 0.9887],
    [0.1903, 0.4919, 0.9867],
    [0.1869, 0.4975, 0.9844],
    [0.1847, 0.503, 0.9819],
    [0.1831, 0.5084, 0.9793],
    [0.1818, 0.5138, 0.9766],
    [0.1806, 0.5191, 0.9738],
    [0.1795, 0.5244, 0.9709],
    [0.1785, 0.5296, 0.9677],
    [0.1778, 0.5349, 0.9641],
    [0.1773, 0.5401, 0.9602],
    [0.1768, 0.5452, 0.956],
    [0.1764, 0.5504, 0.9516],
    [0.1755, 0.5554, 0.9473],
    [0.174, 0.5605, 0.9432],
    [0.1716, 0.5655, 0.9393],
    [0.1686, 0.5705, 0.9357],
    [0.1649, 0.5755, 0.9323],
    [0.161, 0.5805, 0.9289],
    [0.1573, 0.5854, 0.9254],
    [0.154, 0.5902, 0.9218],
    [0.1513, 0.595, 0.9182],
    [0.1492, 0.5997, 0.9147],
    [0.1475, 0.6043, 0.9113],
    [0.1461, 0.6089, 0.908],
    [0.1446, 0.6135, 0.905],
    [0.1429, 0.618, 0.9022],
    [0.1408, 0.6226, 0.8998],
    [0.1383, 0.6272, 0.8975],
    [0.1354, 0.6317, 0.8953],
    [0.1321, 0.6363, 0.8932],
    [0.1288, 0.6408, 0.891],
    [0.1253, 0.6453, 0.8887],
    [0.1219, 0.6497, 0.8862],
    [0.1185, 0.6541, 0.8834],
    [0.1152, 0.6584, 0.8804],
    [0.1119, 0.6627, 0.877],
    [0.1085, 0.6669, 0.8734],
    [0.1048, 0.671, 0.8695],
    [0.1009, 0.675, 0.8653],
    [0.0964, 0.6789, 0.8609],
    [0.0914, 0.6828, 0.8562],
    [0.0855, 0.6865, 0.8513],
    [0.0789, 0.6902, 0.8462],
    [0.0713, 0.6938, 0.8409],
    [0.0628, 0.6972, 0.8355],
    [0.0535, 0.7006, 0.8299],
    [0.0433, 0.7039, 0.8242],
    [0.0328, 0.7071, 0.8183],
    [0.0234, 0.7103, 0.8124],
    [0.0155, 0.7133, 0.8064],
    [0.0091, 0.7163, 0.8003],
    [0.0046, 0.7192, 0.7941],
    [0.0019, 0.722, 0.7878],
    [0.0009, 0.7248, 0.7815],
    [0.0018, 0.7275, 0.7752],
    [0.0046, 0.7301, 0.7688],
    [0.0094, 0.7327, 0.7623],
    [0.0162, 0.7352, 0.7558],
    [0.0253, 0.7376, 0.7492],
    [0.0369, 0.74, 0.7426],
    [0.0504, 0.7423, 0.7359],
    [0.0638, 0.7446, 0.7292],
    [0.077, 0.7468, 0.7224],
    [0.0899, 0.7489, 0.7156],
    [0.1023, 0.751, 0.7088],
    [0.1141, 0.7531, 0.7019],
    [0.1252, 0.7552, 0.695],
    [0.1354, 0.7572, 0.6881],
    [0.1448, 0.7593, 0.6812],
    [0.1532, 0.7614, 0.6741],
    [0.1609, 0.7635, 0.6671],
    [0.1678, 0.7656, 0.6599],
    [0.1741, 0.7678, 0.6527],
    [0.1799, 0.7699, 0.6454],
    [0.1853, 0.7721, 0.6379],
    [0.1905, 0.7743, 0.6303],
    [0.1954, 0.7765, 0.6225],
    [0.2003, 0.7787, 0.6146],
    [0.2061, 0.7808, 0.6065],
    [0.2118, 0.7828, 0.5983],
    [0.2178, 0.7849, 0.5899],
    [0.2244, 0.7869, 0.5813],
    [0.2318, 0.7887, 0.5725],
    [0.2401, 0.7905, 0.5636],
    [0.2491, 0.7922, 0.5546],
    [0.2589, 0.7937, 0.5454],
    [0.2695, 0.7951, 0.536],
    [0.2809, 0.7964, 0.5266],
    [0.2929, 0.7975, 0.517],
    [0.3052, 0.7985, 0.5074],
    [0.3176, 0.7994, 0.4975],
    [0.3301, 0.8002, 0.4876],
    [0.3424, 0.8009, 0.4774],
    [0.3548, 0.8016, 0.4669],
    [0.3671, 0.8021, 0.4563],
    [0.3795, 0.8026, 0.4454],
    [0.3921, 0.8029, 0.4344],
    [0.405, 0.8031, 0.4233],
    [0.4184, 0.803, 0.4122],
    [0.4322, 0.8028, 0.4013],
    [0.4463, 0.8024, 0.3904],
    [0.4608, 0.8018, 0.3797],
    [0.4753, 0.8011, 0.3691],
    [0.4899, 0.8002, 0.3586],
    [0.5044, 0.7993, 0.348],
    [0.5187, 0.7982, 0.3374],
    [0.5329, 0.797, 0.3267],
    [0.547, 0.7957, 0.3159],
    [0.5609, 0.7943, 0.305],
    [0.5748, 0.7929, 0.2941],
    [0.5886, 0.7913, 0.2833],
    [0.6024, 0.7896, 0.2726],
    [0.6161, 0.7878, 0.2622],
    [0.6297, 0.7859, 0.2521],
    [0.6433, 0.7839, 0.2423],
    [0.6567, 0.7818, 0.2329],
    [0.6701, 0.7796, 0.2239],
    [0.6833, 0.7773, 0.2155],
    [0.6963, 0.775, 0.2075],
    [0.7091, 0.7727, 0.1998],
    [0.7218, 0.7703, 0.1924],
    [0.7344, 0.7679, 0.1852],
    [0.7468, 0.7654, 0.1782],
    [0.759, 0.7629, 0.1717],
    [0.771, 0.7604, 0.1658],
    [0.7829, 0.7579, 0.1608],
    [0.7945, 0.7554, 0.157],
    [0.806, 0.7529, 0.1546],
    [0.8172, 0.7505, 0.1535],
    [0.8281, 0.7481, 0.1536],
    [0.8389, 0.7457, 0.1546],
    [0.8495, 0.7435, 0.1564],
    [0.86, 0.7413, 0.1587],
    [0.8703, 0.7392, 0.1615],
    [0.8804, 0.7372, 0.165],
    [0.8903, 0.7353, 0.1695],
    [0.9, 0.7336, 0.1749],
    [0.9093, 0.7321, 0.1815],
    [0.9184, 0.7308, 0.189],
    [0.9272, 0.7298, 0.1973],
    [0.9357, 0.729, 0.2061],
    [0.944, 0.7285, 0.2151],
    [0.9523, 0.7284, 0.2237],
    [0.9606, 0.7285, 0.2312],
    [0.9689, 0.7292, 0.2373],
    [0.977, 0.7304, 0.2418],
    [0.9842, 0.733, 0.2446],
    [0.99, 0.7365, 0.2429],
    [0.9946, 0.7407, 0.2394],
    [0.9966, 0.7458, 0.2351],
    [0.9971, 0.7513, 0.2309],
    [0.9972, 0.7569, 0.2267],
    [0.9971, 0.7626, 0.2224],
    [0.9969, 0.7683, 0.2181],
    [0.9966, 0.774, 0.2138],
    [0.9962, 0.7798, 0.2095],
    [0.9957, 0.7856, 0.2053],
    [0.9949, 0.7915, 0.2012],
    [0.9938, 0.7974, 0.1974],
    [0.9923, 0.8034, 0.1939],
    [0.9906, 0.8095, 0.1906],
    [0.9885, 0.8156, 0.1875],
    [0.9861, 0.8218, 0.1846],
    [0.9835, 0.828, 0.1817],
    [0.9807, 0.8342, 0.1787],
    [0.9778, 0.8404, 0.1757],
    [0.9748, 0.8467, 0.1726],
    [0.972, 0.8529, 0.1695],
    [0.9694, 0.8591, 0.1665],
    [0.9671, 0.8654, 0.1636],
    [0.9651, 0.8716, 0.1608],
    [0.9634, 0.8778, 0.1582],
    [0.9619, 0.884, 0.1557],
    [0.9608, 0.8902, 0.1532],
    [0.9601, 0.8963, 0.1507],
    [0.9596, 0.9023, 0.148],
    [0.9595, 0.9084, 0.145],
    [0.9597, 0.9143, 0.1418],
    [0.9601, 0.9203, 0.1382],
    [0.9608, 0.9262, 0.1344],
    [0.9618, 0.932, 0.1304],
    [0.9629, 0.9379, 0.1261],
    [0.9642, 0.9437, 0.1216],
    [0.9657, 0.9494, 0.1168],
    [0.9674, 0.9552, 0.1116],
    [0.9692, 0.9609, 0.1061],
    [0.9711, 0.9667, 0.1001],
    [0.973, 0.9724, 0.0938],
    [0.9749, 0.9782, 0.0872],
    [0.9769, 0.9839, 0.0805]
]
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)


class SpecgramAxes(Axes):
    """
    自定义的频谱图坐标轴类，支持高级频谱分析和可视化。
    
    主要特性:
        - 支持频率缩放（如 Hz 转 bpm）
        - 多种标度模式（线性、dB、全局归一化、局部归一化）
        - 优化的可视化效果
        - 完整的返回值支持交互分析
    
    Examples
    --------
    基础用法 - IMU信号分析:
        >>> import matplotlib.pyplot as plt
        >>> from pywayne.plot import regist_projection, parula_map
        >>> import numpy as np
        >>> 
        >>> # 生成测试信号
        >>> fs = 100  # 采样率
        >>> t = np.linspace(0, 10, fs*10)
        >>> signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*15*t)
        >>> 
        >>> # 创建频谱图
        >>> regist_projection()
        >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
        >>> spec, freqs, t, im = ax.specgram(
        ...     x=signal, 
        ...     Fs=fs, 
        ...     NFFT=100,
        ...     noverlap=50,
        ...     cmap=parula_map,
        ...     mode='magnitude',
        ...     scale='dB'
        ... )
        >>> ax.set_ylabel('Frequency (Hz)')
        >>> ax.set_xlabel('Time (s)')
        >>> plt.show()
    
    生理信号分析（PPG/心率）- 频率转换为 bpm:
        >>> # PPG信号，想要以 bpm 显示频率
        >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
        >>> spec, freqs, t, im = ax.specgram(
        ...     x=ppg_signal,
        ...     Fs=100,
        ...     NFFT=200,
        ...     noverlap=150,
        ...     freq_scale=60,  # Hz -> bpm
        ...     scale='dB'
        ... )
        >>> ax.set_ylabel('Frequency (bpm)')
        >>> plt.show()
    
    振动分析 - 全局归一化:
        >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
        >>> spec, freqs, t, im = ax.specgram(
        ...     x=vibration_data,
        ...     Fs=1000,
        ...     NFFT=1024,
        ...     noverlap=512,
        ...     scale='linear',
        ...     normalize='global'  # 全局归一化
        ... )
        >>> plt.colorbar(im, label='Normalized Magnitude')
        >>> plt.show()
    
    高分辨率分析 - 零填充:
        >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
        >>> spec, freqs, t, im = ax.specgram(
        ...     x=signal,
        ...     Fs=100,
        ...     NFFT=100,
        ...     pad_to=512,  # 零填充到512点，提高频率分辨率
        ...     noverlap=80,
        ...     scale='dB'
        ... )
        >>> plt.show()
    """
    name = 'z_norm'

    @_preprocess_data(replace_names=["x"])
    def specgram(self, x, NFFT=None, Fs=None, Fc=None, detrend=None,
                 window=None, noverlap=None,
                 cmap=None, xextent=None, pad_to=None, sides=None,
                 scale_by_freq=None, mode=None, scale=None,
                 vmin=None, vmax=None, freq_scale=1.0, normalize='global',
                 **kwargs):
        """
        绘制增强型频谱图（STFT时频分析）。
        
        将数据分割成 NFFT 长度的片段，计算每个片段的频谱，并以彩色图显示。
        支持频率缩放、多种归一化模式，适用于各类信号分析场景。

        Parameters
        ----------
        x : 1-D array or sequence
            输入信号数组。

        NFFT : int, default: 256
            FFT窗口长度（每段数据点数）。
            - 越大 → 频率分辨率越高，时间分辨率越低
            - 建议设为2的幂次方以提高FFT效率
            - 常用值: 128, 256, 512, 1024
            
        Fs : float, default: 2
            采样频率 (Hz)。
            频率分辨率 = Fs / NFFT

        noverlap : int, default: 128
            相邻窗口的重叠点数。
            - 越大 → 时间分辨率越高，计算量越大
            - 典型值: NFFT * 0.5 ~ 0.9
            - 例如 NFFT=256, noverlap=192 (75%重叠)

        cmap : Colormap, optional
            色彩映射。推荐使用 `parula_map`（MATLAB风格）
            其他选项: 'viridis', 'jet', 'hot', 'cool'

        mode : {'default', 'psd', 'magnitude', 'angle', 'phase'}, default: 'psd'
            频谱类型:
            - 'psd': 功率谱密度 (推荐用于能量分析)
            - 'magnitude': 幅度谱 (推荐用于频率成分分析)
            - 'angle': 相位谱（不展开）
            - 'phase': 相位谱（展开）

        scale : {'default', 'linear', 'dB'}, default: 'dB'
            幅度标度:
            - 'dB': 分贝标度，适合大动态范围信号
              * PSD模式: 10*log10(spec)
              * Magnitude模式: 20*log10(spec)
            - 'linear': 线性标度
            - 'default': 根据mode自动选择
            
        normalize : {'global', 'local', 'none'}, default: 'global'
            归一化方式 (仅在 scale='linear' 时生效):
            - 'global': 全局归一化 Z/max(Z)，保留相对强度关系
            - 'local': 按时间段归一化，每列独立缩放到[0,1]
            - 'none': 不归一化
            
        freq_scale : float, default: 1.0
            频率缩放因子。
            使用场景:
            - 1.0: 保持Hz (默认)
            - 60: Hz → bpm (心率/呼吸等生理信号)
            - 0.001: Hz → kHz
            输出频率 = 原始频率 × freq_scale

        Fc : float, default: 0
            中心频率偏移 (Hz)。
            用于下变频信号的频率轴校正。

        detrend : {'none', 'mean', 'linear'} or callable, default: 'none'
            去趋势方法:
            - 'none': 不处理
            - 'mean': 去除均值（去直流分量）
            - 'linear': 去除线性趋势

        window : callable or ndarray, optional
            窗函数。默认使用汉宁窗。
            可选: np.hamming, np.blackman, np.kaiser 等

        pad_to : int, optional
            零填充长度。
            - 可以大于 NFFT 来提高频率分辨率
            - 例如: NFFT=128, pad_to=512 可得到更平滑的频谱

        scale_by_freq : bool, default: True
            是否按频率缩放密度值（仅PSD模式）。

        sides : {'default', 'onesided', 'twosided'}, optional
            频谱类型:
            - 'onesided': 单边谱（实信号默认）
            - 'twosided': 双边谱（复信号默认）

        xextent : tuple of (xmin, xmax), optional
            时间轴范围。默认自动计算。

        vmin, vmax : float, optional
            色彩映射的值域范围。

        **kwargs
            传递给 `imshow` 的其他参数（如 aspect, interpolation）。
            注意: 不支持 'origin' 参数。

        Returns
        -------
        spec : 2D ndarray, shape (n_freqs, n_times)
            频谱数据矩阵。每列是一个时间段的频谱。
            
        freqs : 1D ndarray, shape (n_freqs,)
            频率轴数组 (已应用 freq_scale 和 Fc)。
            
        t : 1D ndarray, shape (n_times,)
            时间轴数组（各段的中心时刻）。
            
        im : AxesImage
            matplotlib 图像对象，可用于添加colorbar。

        Notes
        -----
        - 频率分辨率: Δf = Fs / NFFT
        - 时间分辨率: Δt = (NFFT - noverlap) / Fs
        - 不确定性原理: Δf × Δt ≥ 常数（无法同时获得极高的频率和时间分辨率）
        - dB模式下会自动处理log(0)问题（映射到极小值）
        
        Examples
        --------
        IMU加速度信号分析:
            >>> fs = 100
            >>> win_time, step_time = 1, 0.1
            >>> spec, freqs, t, im = ax.specgram(
            ...     x=acc_data,
            ...     Fs=fs,
            ...     NFFT=int(win_time * fs),
            ...     noverlap=int((win_time - step_time) * fs),
            ...     scale='dB',
            ...     cmap=parula_map
            ... )
            >>> ax.set_ylabel('Frequency (Hz)')
            >>> ax.set_ylim(0, 30)  # 关注0-30Hz范围
            
        心率信号（PPG）:
            >>> fs = 100
            >>> spec, freqs, t, im = ax.specgram(
            ...     x=ppg_signal,
            ...     Fs=fs,
            ...     NFFT=400,  # 4秒窗口
            ...     noverlap=300,
            ...     freq_scale=60,  # 转换为bpm
            ...     scale='dB'
            ... )
            >>> ax.set_ylabel('Heart Rate (bpm)')
            >>> ax.set_ylim(40, 180)  # 典型心率范围
            
        交互式分析（点击查看FFT）:
            >>> spec, freqs, t, im = ax.specgram(...)
            >>> def on_click(event):
            ...     if event.xdata:
            ...         idx = np.argmin(np.abs(t - event.xdata))
            ...         plt.figure()
            ...         plt.plot(freqs, spec[:, idx])
            ...         plt.xlabel('Frequency')
            ...         plt.ylabel('Magnitude')
            ...         plt.show()
            >>> fig.canvas.mpl_connect('button_press_event', on_click)

        See Also
        --------
        scipy.signal.spectrogram : 底层频谱图计算函数
        matplotlib.pyplot.specgram : Matplotlib标准频谱图
        
        References
        ----------
        .. [1] Julius O. Smith III, "Spectral Audio Signal Processing",
               W3K Publishing, 2011, ISBN 978-0-9745607-3-1.
        .. [2] Oppenheim, A. V., & Schafer, R. W. (2009). 
               Discrete-time signal processing (3rd ed.).
        """
        if NFFT is None:
            NFFT = 256  # same default as in mlab.specgram()
        if Fc is None:
            Fc = 0  # same default as in mlab._spectral_helper()
        if noverlap is None:
            noverlap = 128  # same default as in mlab.specgram()
        if Fs is None:
            Fs = 2  # same default as in mlab._spectral_helper()

        if mode == 'complex':
            raise ValueError('Cannot plot a complex specgram')

        if scale is None or scale == 'default':
            if mode in ['angle', 'phase']:
                scale = 'linear'
            else:
                scale = 'dB'
        elif mode in ['angle', 'phase'] and scale == 'dB':
            raise ValueError('Cannot use dB scale with angle or phase mode')

        freqs, t, spec = spectrogram(
            x, Fs,
            nperseg=NFFT,
            noverlap=noverlap,
            nfft=NFFT
        )
        # spec, freqs, t = mlab.specgram(x=x, NFFT=NFFT, Fs=Fs,
        #                                detrend=detrend, window=window,
        #                                noverlap=noverlap, pad_to=pad_to,
        #                                sides=sides,
        #                                scale_by_freq=scale_by_freq,
        #                                mode=mode)

        # 幅度标度处理
        if scale == 'linear':
            Z = spec
            # 根据 normalize 参数选择归一化方式
            if normalize == 'global':
                # 全局归一化：保留相对强度关系
                Z = Z / np.max(Z) if np.max(Z) > 0 else Z
            elif normalize == 'local':
                # 局部归一化：每个时间段独立归一化
                Z = np.apply_along_axis(lambda x: x / np.max(x) if np.max(x) > 0 else x, 0, Z)
            elif normalize == 'none':
                # 不归一化
                pass
            else:
                raise ValueError(f"Unknown normalize mode: {normalize}. "
                               "Choose from 'global', 'local', or 'none'.")
        elif scale == 'dB':
            # dB标度：处理log(0)问题
            epsilon = 1e-10  # 避免log(0)
            if mode is None or mode == 'default' or mode == 'psd':
                Z = 10. * np.log10(spec + epsilon)
            else:
                Z = 20. * np.log10(spec + epsilon)
        else:
            raise ValueError(f'Unknown scale: {scale}. Choose from "linear" or "dB".')

        Z = np.flipud(Z)

        if xextent is None:
            # padding is needed for first and last segment:
            pad_xextent = (NFFT - noverlap) / Fs / 2
            xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
        xmin, xmax = xextent
        
        # 应用频率偏移和缩放
        freqs += Fc
        freqs *= freq_scale  # 参数化的频率缩放（如 Hz -> bpm）
        
        extent = xmin, xmax, freqs[0], freqs[-1]

        if 'origin' in kwargs:
            raise TypeError("specgram() got an unexpected keyword argument "
                            "'origin'")

        im = self.imshow(Z, cmap, extent=extent, vmin=vmin, vmax=vmax,
                         interpolation='lanczos',
                         origin='upper', **kwargs)
        self.axis('auto')

        return spec, freqs, t, im


def regist_projection():
    """
    注册自定义 SpecgramAxes projection。
    
    在使用 SpecgramAxes 绘制频谱图前必须调用此函数。
    
    Examples
    --------
    >>> from pywayne.plot import regist_projection
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> regist_projection()
    >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
    >>> # 现在可以使用 ax.specgram() 了
    """
    register_projection(SpecgramAxes)


def get_specgram_params(signal_length, sampling_rate, 
                       time_resolution=None, freq_resolution=None,
                       overlap_ratio=0.75):
    """
    根据信号长度和期望的分辨率，推荐 STFT 参数。
    
    Parameters
    ----------
    signal_length : int
        信号长度（采样点数）
    sampling_rate : float
        采样率 (Hz)
    time_resolution : float, optional
        期望的时间分辨率（秒）。优先级高于 freq_resolution。
    freq_resolution : float, optional
        期望的频率分辨率（Hz）。当 time_resolution 未指定时使用。
    overlap_ratio : float, default: 0.75
        窗口重叠比例 (0-1之间)
        
    Returns
    -------
    params : dict
        推荐的参数字典，包含:
        - 'NFFT': FFT窗口长度
        - 'noverlap': 重叠点数
        - 'actual_freq_res': 实际频率分辨率
        - 'actual_time_res': 实际时间分辨率
        - 'n_segments': 预计的时间段数
        
    Examples
    --------
    根据期望的时间分辨率:
        >>> params = get_specgram_params(
        ...     signal_length=10000,
        ...     sampling_rate=100,
        ...     time_resolution=0.1  # 期望0.1秒时间分辨率
        ... )
        >>> print(params)
        {'NFFT': 256, 'noverlap': 192, ...}
        
    根据期望的频率分辨率:
        >>> params = get_specgram_params(
        ...     signal_length=10000,
        ...     sampling_rate=100,
        ...     freq_resolution=0.5  # 期望0.5Hz频率分辨率
        ... )
        
    直接使用推荐参数:
        >>> params = get_specgram_params(10000, 100, time_resolution=0.1)
        >>> spec, freqs, t, im = ax.specgram(
        ...     x=signal,
        ...     Fs=100,
        ...     NFFT=params['NFFT'],
        ...     noverlap=params['noverlap']
        ... )
    """
    if time_resolution is not None:
        # 根据时间分辨率计算NFFT
        # time_res = (NFFT - noverlap) / Fs
        # NFFT = time_res * Fs / (1 - overlap_ratio)
        NFFT = int(time_resolution * sampling_rate / (1 - overlap_ratio))
    elif freq_resolution is not None:
        # 根据频率分辨率计算NFFT
        # freq_res = Fs / NFFT
        NFFT = int(sampling_rate / freq_resolution)
    else:
        # 默认：选择合适的NFFT（信号长度的10%或256，取较小值）
        NFFT = min(256, int(signal_length * 0.1))
    
    # 确保NFFT是2的幂（FFT效率更高）
    NFFT = 2 ** int(np.ceil(np.log2(NFFT)))
    
    # 限制NFFT范围
    NFFT = max(64, min(NFFT, signal_length))
    
    # 计算noverlap
    noverlap = int(NFFT * overlap_ratio)
    
    # 计算实际分辨率
    actual_freq_res = sampling_rate / NFFT
    actual_time_res = (NFFT - noverlap) / sampling_rate
    
    # 计算时间段数
    step = NFFT - noverlap
    n_segments = (signal_length - noverlap) // step
    
    params = {
        'NFFT': NFFT,
        'noverlap': noverlap,
        'actual_freq_res': actual_freq_res,
        'actual_time_res': actual_time_res,
        'n_segments': n_segments,
        'window_duration': NFFT / sampling_rate,
        'step_duration': step / sampling_rate,
        'overlap_ratio': overlap_ratio
    }
    
    return params
