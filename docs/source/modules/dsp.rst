数字信号处理 (dsp)
====================

数字信号处理模块提供了一系列信号处理工具，包括滤波器、峰值检测、信号去趋势、信号相似度计算以及其他高级信号处理算法。该模块常用于数据预处理、信号分析、特征提取和噪声抑制等应用场景。

滤波器
-----------

1. .. py:function:: butter_bandpass_filter(x, order=2, lo=0.1, hi=10, fs=0, btype='lowpass', realtime=False) -> ndarray

   巴特沃斯滤波器函数，用于对输入信号进行滤波处理。

   **参数**:

   - **x**: 输入信号，通常为一维数组。
   - **order**: 滤波器阶数。
   - **lo**: 低频截止频率。
   - **hi**: 高频截止频率。
   - **fs**: 采样频率。
   - **btype**: 滤波器类型，可选值包括 'lowpass', 'highpass', 'bandpass', 'bandstop'。
   - **realtime**: 是否实时处理，默认False。

   **返回**:

   - 滤波后的信号数组。

   **应用场景**:

   用于去除信号中的高频噪音或低频干扰，广泛应用于数据预处理和特征提取。例如，在心电图信号分析中，可以使用该函数滤除工频噪声。

   **示例**::

      filtered_signal = butter_bandpass_filter(signal, order=3, lo=0.5, hi=40, fs=250, btype='bandpass')


2. .. py:class:: ButterworthFilter

   基于 NumPy 的巴特沃斯 1D IIR 滤波器，同时支持传递函数 ``ba`` 和数值更稳定的
   二阶节级联 ``sos``。两种模式都使用 Direct Form II Transposed 执行，数值结果与
   SciPy ``lfilter``/``sosfilt`` 及 ``filtfilt``/``sosfiltfilt`` 对齐。

   **构造方法**:

   - **__init__(self, b: np.ndarray, a: np.ndarray, cache_zi: bool = True)**

     通过滤波器系数构造实例。

     **参数**:

     - **b**: 分子系数（numerator coefficients）。
     - **a**: 分母系数（denominator coefficients）。
     - **cache_zi**: 是否预计算稳态初始条件，默认为 True。

   - **from_ba(cls, b, a, cache_zi=True)** (类方法)

     通过 b、a 系数创建滤波器实例。

     **参数**:

     - **b**: 分子系数。
     - **a**: 分母系数。
     - **cache_zi**: 是否预计算初始条件。

     **返回**:

     - ButterworthFilter 实例。

   - **from_sos(cls, sos, cache_zi=True)** (类方法)

     通过 SciPy 格式的 SOS 矩阵创建滤波器，推荐用于高阶、窄带或截止频率接近
     0/Nyquist 的设计。

     **参数**:

     - **sos**: ``(n_sections, 6)`` 矩阵，每行为 ``[b0, b1, b2, a0, a1, a2]``。
     - **cache_zi**: 是否预计算 SOS 稳态初始条件。

   - **from_params(cls, order, fs, btype, cutoff, cache_zi=True, output='ba')** (类方法)

     通过滤波器参数设计并创建实例。

     **参数**:

     - **order**: 滤波器阶数。
     - **fs**: 采样频率（Hz）。
     - **btype**: 滤波器类型，可选 'lowpass', 'highpass', 'bandpass', 'bandstop'。
     - **cutoff**: 截止频率（Hz），低通/高通为单个浮点数，带通/带阻为 (low, high) 元组。
     - **cache_zi**: 是否预计算初始条件。
     - **output**: ``'ba'`` 或 ``'sos'``。高阶滤波器推荐 ``'sos'``。

     **返回**:

     - ButterworthFilter 实例。

   **属性**:

   - **mode**: 返回当前 ``'ba'`` 或 ``'sos'`` 模式。
   - **ba**: BA 模式下返回 (b, a) 系数元组。
   - **sos**: SOS 模式下返回归一化后的 SOS 矩阵。
   - **ntaps**: 滤波器抽头数（taps）。
   - **nstate**: 滤波器状态数。

   **方法**:

   - **zi(self) -> np.ndarray**

     返回稳态初始条件（若未缓存则懒加载计算）。

     **返回**:

     - BA 模式返回一维数组；SOS 模式返回 ``(n_sections, 2)`` 数组。

   - **lfilter(self, x, zi=None) -> Tuple[np.ndarray, np.ndarray]**

     Direct Form II Transposed 单向滤波。在 SOS 模式下自动逐个二阶节级联执行。

     **参数**:

     - **x**: 输入信号。
     - **zi**: 初始状态，可选。SOS 模式接受 ``(n_sections, 2)`` 或等价扁平数组。

     **返回**:

     - (y, zf)：滤波后的信号和最终状态。

   - **filtfilt(self, x, padtype='odd', padlen=None) -> np.ndarray**

     零相位滤波（前向-后向滤波）。

     **参数**:

     - **x**: 输入信号。
     - **padtype**: 填充方式，可选 'odd', 'even', 'constant', None，默认 'odd'。
     - **padlen**: 填充长度，默认为 3*ntaps。

     **返回**:

     - 零相位滤波后的信号。

   **静态方法**:

   - **lfilter_zi(b, a)**: 计算与 SciPy ``lfilter_zi`` 对齐的 BA 稳态初始条件。
   - **sosfilt_zi(sos)**: 计算与 SciPy ``sosfilt_zi`` 对齐的 SOS 稳态初始条件。

   - **detrend(x, method='linear', poly_order=2)** (静态方法)

     信号去趋势处理。

     **参数**:

     - **x**: 输入信号。
     - **method**: 去趋势方法，可选 'none', 'mean', 'linear', 'poly'。
     - **poly_order**: 多项式阶数（仅当 method='poly' 时有效）。

     **返回**:

     - 去趋势后的信号。

   **应用场景**:

   BA/SOS 执行核心均使用 NumPy 实现，可用于实时流式滤波、离线前后向滤波和信号预处理。
   ``from_params(..., output='sos')`` 借助 SciPy 生成稳定的 SOS 节配对。

   **示例**::

      # 方式1: 通过参数设计
      bf = ButterworthFilter.from_params(order=4, fs=200, btype='bandpass', cutoff=(1, 50))
      filtered = bf.filtfilt(signal)

      # 方式2: 通过系数构造
      bf2 = ButterworthFilter.from_ba(b, a)
      y, zf = bf2.lfilter(signal)

      # 方式3: SOS（高阶/窄带推荐）
      bf_sos = ButterworthFilter.from_params(
          order=8, fs=200, btype='bandpass', cutoff=(0.1, 10), output='sos'
      )

      # 流式执行：首段稳态初始化，后续传递 zf
      state = bf_sos.zi() * chunks[0][0]
      outputs = []
      for chunk in chunks:
          y, state = bf_sos.lfilter(chunk, zi=state)
          outputs.append(y)

      # 去趋势
      detrended = ButterworthFilter.detrend(signal, method='linear')

   .. note::

      ``filtfilt`` 为非实时前后向处理，零相位但仍可能出现边界效应。在线处理应使用
      ``lfilter``，首段通常使用 ``zi() * x[0]``，后续每段把上一段返回的 ``zf``
      作为新的 ``zi``。

峰值检测
-----------

1. .. py:function:: peak_det(v, delta, x=None) -> tuple

   峰值检测函数，用于检测信号中的局部极大值和极小值。

   **参数**:

   - **v**: 输入信号数组。
   - **delta**: 检测阈值，控制检测灵敏度。
   - **x**: 可选的x轴数据，若未提供则默认使用下标。

   **返回**:

   - 一个元组，包含检测到的最大值和最小值的位置及对应值。

   **应用场景**:

   在信号处理或特征提取中，经常需要检测峰值，例如在心电图或声波信号中进行QRS复合波检测。

   **示例**::

      peaks, valleys = peak_det(signal, delta=0.5)

2. .. py:function:: find_extremum_in_sliding_window(data: list, k: int) -> list

   在滑动窗口中查找极值。

   **参数**:

   - **data**: 输入数据列表。
   - **k**: 滑动窗口的大小。

   **返回**:

   - 包含局部极值的列表。

   **应用场景**:

   可用于信号平滑和噪声鲁棒的特征提取，例如在时间序列数据中寻找局部变化的关键点。

   **示例**::

      extrema = find_extremum_in_sliding_window(signal, k=50)

3. .. py:class:: FindSlidingWindowExtremum

   滑动窗口极值查找器类，用于在实时或离线数据流中实时更新窗口内的极值。

   **方法**:

   - **__init__(self, win: int, find_max: bool)**

     初始化参数。

     **参数**:

     - **win**: 窗口大小。
     - **find_max**: 若为True，则查找最大值；若为False，则查找最小值。

   - **apply(self, val)**

     更新窗口数据，并返回当前窗口内的极值。

     **参数**:

     - **val**: 新的输入值

     **返回**:

     - 当前窗口的极值。

   **应用场景**:

   用于实时信号监控中快速检测最新数据窗口内的峰值或谷值。

   **示例**::

      detector = FindSlidingWindowExtremum(win=100, find_max=True)
      for sample in stream:
          current_peak = detector.apply(sample)
          # 进一步处理 current_peak

信号去趋势
-----------

1. .. py:class:: SignalDetrend

   信号去趋势处理器，用于消除信号中的线性或非线性趋势成分。

   **方法**:

   - **__init__(self, method='linear', **kwargs)**

     初始化去趋势方法。

     **参数**:

     - **method**: 去趋势方法，例如 'linear', 'polynomial', 'loess', 'wavelet', 'emd', 'ceemdan', 'median'。
     - **kwargs**: 针对特定方法的其他参数。

   - **__call__(self, x)**

     应用去趋势算法处理输入信号。

     **参数**:

     - **x**: 输入信号。

     **返回**:

     - 去趋势后的信号。

   **应用场景**:

   在数据预处理中非常重要，例如去除温度数据的季节性趋势或金融数据中的长期趋势。

   **示例**::

      detrender = SignalDetrend(method='loess', span=0.3)
      detrended_signal = detrender(raw_signal)

信号相似度
-----------

1. .. py:class:: CurveSimilarity

   曲线相似度计算类，提供动态时间规整（DTW）等算法。

   **方法**:

   - **dtw(self, x, y, mode='global', *params, backend='auto')**

     计算两条曲线之间的DTW距离。

     **参数**:

     - **x**: 第一条曲线数据。
     - **y**: 第二条曲线数据。
     - **mode**: 计算模式，默认为 'global'。
     - **params**: ``local`` 模式所需的正整数窗口长度。
     - **backend**: ``auto``、``python`` 或 ``numba``。``auto`` 优先使用 Numba，未安装时自动回退到 Python。

     **返回**:

     - 两条曲线的相似度距离值。

   **应用场景**:

   用于语音识别、手写识别、股票走势比对等需要比较时间序列相似度的领域。

   **示例**::

      similarity = CurveSimilarity()
      distance = similarity.dtw(curve1, curve2)
      print(f"DTW距离: {distance}")

      # 强制使用纯 Python 参考实现
      distance_python = similarity.dtw(curve1, curve2, backend='python')

      # 安装 pip install "pywayne[performance]" 后可强制使用 Numba
      distance_numba = similarity.dtw(curve1, curve2, backend='numba')

   Numba 为可选依赖，适合重复调用或较长曲线。首次调用会进行 JIT 编译；短生命周期脚本可使用
   ``backend='python'`` 避免编译启动开销。Python 与 Numba 后端保持相同的历史评分语义。

其他工具
-----------

1. .. py:class:: OneEuroFilter

   一欧元滤波器类，用于平滑信号并减少延迟。

   **方法**:

   - **__init__(self, te=None, mincutoff=1.0, beta=0.007, dcutoff=1.0)**

     初始化滤波器参数。

     **参数**:

     - **te**: 采样时间，可为None自动推断。
     - **mincutoff**: 最小截止频率。
     - **beta**: 调整速率的参数。
     - **dcutoff**: 导数截止频率。

   - **apply(self, val: float, te: float = 0.0) -> float**

     应用滤波器，对输入值进行平滑处理。

     **参数**:

     - **val**: 输入信号的当前值。
     - **te**: 时间间隔，默认为0.0。

     **返回**:

     - 平滑后的值。

   **应用场景**:

   常用于机器人姿态估计、传感器数据平滑以及实时控制系统中，能够有效滤除噪音同时保持信号响应的及时性。

   **示例**::

      euro_filter = OneEuroFilter(te=0.02, mincutoff=1.0, beta=0.01, dcutoff=1.0)
      smooth_value = euro_filter.apply(new_measurement, te=0.02)

2. .. py:class:: WelfordStd

   使用Welford算法进行在线标准差计算的类。

   **方法**:

   - **__init__(self, win: int)**

     初始化窗口大小。

     **参数**:

     - **win**: 窗口大小，用于限定计算范围。

   - **apply(self, val)**

     更新标准差计算，并返回当前窗口内的标准差。

     **参数**:

     - **val**: 新的输入数值。

     **返回**:

     - 当前窗口内数据的标准差。

   **应用场景**:

   在线统计和实时监控中，计算数据波动情况，如生产质量控制、传感器数据监控等。

   **示例**::

      std_calculator = WelfordStd(win=50)
      for sample in data_stream:
          current_std = std_calculator.apply(sample)
          # 使用 current_std 做进一步判断

--------------------------------------------------

以上详细介绍了dsp模块中各个函数和类的用途、应用场景以及示例代码，可帮助用户快速理解和使用数字信号处理相关工具。
