绘图工具 (plot)
===============

PyWayne 绘图工具库 - 增强型频谱分析模块

本模块提供专业的时频分析可视化工具，特别优化用于：

- IMU 传感器数据（加速度计、陀螺仪）
- 生理信号（PPG、ECG、呼吸）
- 振动分析
- 音频信号处理

主要特性：

1. **SpecgramAxes**: 增强型频谱图类
   - 支持频率单位转换（Hz ↔ bpm ↔ kHz）
   - 多种归一化模式（全局/局部/不归一化）
   - 优化的色彩映射（Parula风格）
   - 完整返回值支持交互分析

2. **parula_map**: MATLAB风格色彩映射
   - 感知均匀的色彩分布
   - 适合科学可视化

3. **get_specgram_params**: 参数推荐函数
   - 根据信号特性自动推荐STFT参数
   - 支持时间/频率分辨率优化

SpecgramAxes 类
----------------

.. py:class:: SpecgramAxes(Axes)

   自定义的频谱图坐标轴类，支持高级频谱分析和可视化。
   
   **主要特性**:
   
   - 支持频率缩放（如 Hz 转 bpm）
   - 多种标度模式（线性、dB、全局归一化、局部归一化）
   - 优化的可视化效果
   - 完整的返回值支持交互分析
   
   **主要属性和方法**:
   
   - **name**: 类属性，投影名称为 'z_norm'。
   
   .. py:method:: specgram(x, NFFT=None, Fs=None, Fc=None, detrend=None, window=None, noverlap=None, cmap=None, xextent=None, pad_to=None, sides=None, scale_by_freq=None, mode=None, scale=None, vmin=None, vmax=None, freq_scale=1.0, normalize='global', **kwargs)
     
     绘制增强型频谱图（STFT时频分析）。
     
     将数据分割成 NFFT 长度的片段，计算每个片段的频谱，并以彩色图显示。
     支持频率缩放、多种归一化模式，适用于各类信号分析场景。
     
     **参数说明**:
     
     - **x** (1-D array): 输入信号数组。
     - **NFFT** (int, default: 256): FFT窗口长度（每段数据点数）。
       
       - 越大 → 频率分辨率越高，时间分辨率越低
       - 建议设为2的幂次方以提高FFT效率
       - 常用值: 128, 256, 512, 1024
       
     - **Fs** (float, default: 2): 采样频率 (Hz)。频率分辨率 = Fs / NFFT
     - **noverlap** (int, default: 128): 相邻窗口的重叠点数。
       
       - 越大 → 时间分辨率越高，计算量越大
       - 典型值: NFFT * 0.5 ~ 0.9
       - 例如 NFFT=256, noverlap=192 (75%重叠)
       
     - **cmap** (Colormap, optional): 色彩映射。推荐使用 ``parula_map`` （MATLAB风格）
     - **mode** ({'default', 'psd', 'magnitude', 'angle', 'phase'}, default: 'psd'): 频谱类型
       
       - 'psd': 功率谱密度 (推荐用于能量分析)
       - 'magnitude': 幅度谱 (推荐用于频率成分分析)
       - 'angle': 相位谱（不展开）
       - 'phase': 相位谱（展开）
       
     - **scale** ({'default', 'linear', 'dB'}, default: 'dB'): 幅度标度
       
       - 'dB': 分贝标度，适合大动态范围信号
       - 'linear': 线性标度
       - 'default': 根据mode自动选择
       
     - **normalize** ({'global', 'local', 'none'}, default: 'global'): 归一化方式 (仅在 scale='linear' 时生效)
       
       - 'global': 全局归一化 Z/max(Z)，保留相对强度关系
       - 'local': 按时间段归一化，每列独立缩放到[0,1]
       - 'none': 不归一化
       
     - **freq_scale** (float, default: 1.0): 频率缩放因子
       
       - 1.0: 保持Hz (默认)
       - 60: Hz → bpm (心率/呼吸等生理信号)
       - 0.001: Hz → kHz
       - 输出频率 = 原始频率 × freq_scale
       
     - **Fc** (float, default: 0): 中心频率偏移 (Hz)，用于下变频信号的频率轴校正。
     - **detrend** ({'none', 'mean', 'linear'} or callable, default: 'none'): 去趋势方法
     - **window** (callable or ndarray, optional): 窗函数。默认使用汉宁窗。
     - **pad_to** (int, optional): 零填充长度，可以大于 NFFT 来提高频率分辨率
     - **scale_by_freq** (bool, default: True): 是否按频率缩放密度值（仅PSD模式）
     - **sides** ({'default', 'onesided', 'twosided'}, optional): 频谱类型
     - **xextent** (tuple, optional): 时间轴范围 (xmin, xmax)。默认自动计算。
     - **vmin**, **vmax** (float, optional): 色彩映射的值域范围。
     - **kwargs**: 传递给 ``imshow`` 的其他参数（如 aspect, interpolation）。
     
     **返回值**:
     
     - **spec** (2D ndarray): 频谱数据矩阵，shape (n_freqs, n_times)
     - **freqs** (1D ndarray): 频率轴数组 (已应用 freq_scale 和 Fc)
     - **t** (1D ndarray): 时间轴数组（各段的中心时刻）
     - **im** (AxesImage): matplotlib 图像对象，可用于添加colorbar
     
     **应用场景**:
     
     - IMU传感器数据分析（加速度计、陀螺仪）
     - 生理信号处理（PPG心率、ECG、呼吸信号）
     - 振动分析和故障诊断
     - 音频信号频谱分析
     - 支持通过自定义参数调整绘图效果，满足科研和工程中的定制需求
     
     **基础示例**::
     
        >>> import matplotlib.pyplot as plt
        >>> from pywayne.plot import regist_projection, parula_map
        >>> import numpy as np
        >>> 
        >>> # 注册自定义projection
        >>> regist_projection()
        >>> 
        >>> # 创建频谱图
        >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
        >>> fs = 100  # 采样率
        >>> t = np.linspace(0, 10, fs*10)
        >>> signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*15*t)
        >>> 
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
        >>> plt.colorbar(im, label='Magnitude (dB)')
        >>> plt.show()
     
     **生理信号分析示例**::
     
        >>> # PPG信号，转换为bpm显示
        >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
        >>> spec, freqs, t, im = ax.specgram(
        ...     x=ppg_signal,
        ...     Fs=100,
        ...     NFFT=400,  # 4秒窗口
        ...     noverlap=300,
        ...     freq_scale=60,  # Hz -> bpm
        ...     scale='dB'
        ... )
        >>> ax.set_ylabel('Heart Rate (bpm)')
        >>> ax.set_ylim(40, 180)  # 典型心率范围
        >>> plt.show()
     
     **振动分析示例**::
     
        >>> # 振动数据，使用全局归一化
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


parula_map 色彩映射
--------------------

.. py:data:: parula_map

   MATLAB风格的Parula色彩映射，提供感知均匀的色彩分布，特别适合科学可视化。
   
   **特性**:
   
   - 感知均匀的色彩过渡
   - 适合频谱图、热图等科学数据可视化
   - 与MATLAB的parula colormap兼容
   
   **示例**::
   
      >>> from pywayne.plot import parula_map
      >>> import matplotlib.pyplot as plt
      >>> import numpy as np
      >>> 
      >>> # 使用parula色彩映射
      >>> data = np.random.randn(100, 100)
      >>> plt.imshow(data, cmap=parula_map)
      >>> plt.colorbar()
      >>> plt.show()

regist_projection 函数
-------------------------

.. py:function:: regist_projection()

   注册自定义 SpecgramAxes projection。
   
   在使用 SpecgramAxes 绘制频谱图前必须调用此函数。
   
   **应用场景**:
   
   - 在使用 matplotlib 时注册自定义投影，扩展默认的 Axes 功能
   - 允许用户通过指定 projection='z_norm' 来使用特定的频谱图绘制功能
   
   **示例**::
   
      >>> from pywayne.plot import regist_projection
      >>> regist_projection()
      >>> import matplotlib.pyplot as plt
      >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
      >>> # 现在可以使用 ax.specgram() 了

get_specgram_params 函数
---------------------------

.. py:function:: get_specgram_params(signal_length, sampling_rate, time_resolution=None, freq_resolution=None, overlap_ratio=0.75)

   根据信号长度和期望的分辨率，推荐 STFT 参数。
   
   **参数**:
   
   - **signal_length** (int): 信号长度（采样点数）
   - **sampling_rate** (float): 采样率 (Hz)
   - **time_resolution** (float, optional): 期望的时间分辨率（秒）。优先级高于 freq_resolution
   - **freq_resolution** (float, optional): 期望的频率分辨率（Hz）。当 time_resolution 未指定时使用
   - **overlap_ratio** (float, default: 0.75): 窗口重叠比例 (0-1之间)
   
   **返回值**:
   
   - **params** (dict): 推荐的参数字典，包含:
     
     - 'NFFT': FFT窗口长度
     - 'noverlap': 重叠点数
     - 'actual_freq_res': 实际频率分辨率
     - 'actual_time_res': 实际时间分辨率
     - 'n_segments': 预计的时间段数
     - 'window_duration': 窗口持续时间（秒）
     - 'step_duration': 步进持续时间（秒）
     - 'overlap_ratio': 重叠比例
   
   **应用场景**:
   
   - 自动优化STFT参数设置
   - 根据信号特性和分析需求推荐合适的窗口参数
   - 在不确定参数设置时提供专业建议
   
   **示例**::
   
      >>> from pywayne.plot import get_specgram_params
      >>> 
      >>> # 根据期望的时间分辨率推荐参数
      >>> params = get_specgram_params(
      ...     signal_length=10000,
      ...     sampling_rate=100,
      ...     time_resolution=0.1  # 期望0.1秒时间分辨率
      ... )
      >>> print(params)
      {'NFFT': 256, 'noverlap': 192, 'actual_freq_res': 0.390625, ...}
      
      >>> # 根据期望的频率分辨率推荐参数
      >>> params = get_specgram_params(
      ...     signal_length=10000,
      ...     sampling_rate=100,
      ...     freq_resolution=0.5  # 期望0.5Hz频率分辨率
      ... )
      
      >>> # 直接使用推荐参数
      >>> regist_projection()
      >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
      >>> spec, freqs, t, im = ax.specgram(
      ...     x=signal,
      ...     Fs=100,
      ...     NFFT=params['NFFT'],
      ...     noverlap=params['noverlap']
      ... )

使用指南
--------

**完整工作流程示例**::

   >>> import numpy as np
   >>> import matplotlib.pyplot as plt
   >>> from pywayne.plot import regist_projection, parula_map, get_specgram_params
   >>> 
   >>> # 1. 生成测试信号（模拟IMU数据）
   >>> fs = 100  # 采样率100Hz
   >>> duration = 30  # 30秒数据
   >>> t = np.linspace(0, duration, fs * duration)
   >>> signal = (np.sin(2*np.pi*2*t) +  # 2Hz基频
   ...          0.5*np.sin(2*np.pi*8*t) +  # 8Hz谐波
   ...          0.2*np.random.randn(len(t)))  # 噪声
   >>> 
   >>> # 2. 获取推荐参数
   >>> params = get_specgram_params(
   ...     signal_length=len(signal),
   ...     sampling_rate=fs,
   ...     time_resolution=0.5  # 期望0.5秒时间分辨率
   ... )
   >>> print(f"推荐NFFT: {params['NFFT']}, noverlap: {params['noverlap']}")
   >>> 
   >>> # 3. 注册投影并创建频谱图
   >>> regist_projection()
   >>> fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'z_norm'})
   >>> 
   >>> # 4. 绘制频谱图
   >>> spec, freqs, t_spec, im = ax.specgram(
   ...     x=signal,
   ...     Fs=fs,
   ...     NFFT=params['NFFT'],
   ...     noverlap=params['noverlap'],
   ...     cmap=parula_map,
   ...     scale='dB'
   ... )
   >>> 
   >>> # 5. 美化图形
   >>> ax.set_ylabel('Frequency (Hz)')
   >>> ax.set_xlabel('Time (s)')
   >>> ax.set_ylim(0, 20)  # 关注0-20Hz范围
   >>> plt.colorbar(im, label='Magnitude (dB)')
   >>> plt.title('IMU Signal Spectrogram Analysis')
   >>> plt.tight_layout()
   >>> plt.show()

**交互式分析示例**::

   >>> # 添加点击事件，查看特定时刻的频谱
   >>> def on_click(event):
   ...     if event.xdata and event.inaxes == ax:
   ...         time_idx = np.argmin(np.abs(t_spec - event.xdata))
   ...         plt.figure(figsize=(8, 4))
   ...         plt.plot(freqs, spec[:, time_idx])
   ...         plt.xlabel('Frequency (Hz)')
   ...         plt.ylabel('Magnitude')
   ...         plt.title(f'FFT at t={event.xdata:.2f}s')
   ...         plt.grid(True)
   ...         plt.show()
   >>> 
   >>> fig.canvas.mpl_connect('button_press_event', on_click)

--------------------------------------------------

通过上述工具和示例，可以快速掌握 plot 模块中各功能的使用方法，这对于进行专业的时频分析和频谱图绘制提供了强大支持。模块特别适用于IMU传感器数据、生理信号、振动分析等领域的信号处理需求。 