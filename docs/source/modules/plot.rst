绘图工具 (plot)
===============

本模块主要提供了一些数据可视化工具，尤其在绘制频谱图和定制 Colormap 绘图方面具有优势。模块中扩展了 matplotlib 的功能，为绘图提供了新的投影选项和方法。

SpecgramAxes 类
----------------

.. py:class:: SpecgramAxes(Axes)

   这是一个扩展自 matplotlib.axes.Axes 的类，通过注册自定义的投影 'z_norm'，支持绘制频谱图。该类内置了专用于显示频谱图的 specgram 方法，允许用户方便地设置 FFT 参数、采样率及其他图形属性。
   
   **主要属性和方法**:
   
   - **name**: 类属性，投影名称为 'z_norm'。
   - **specgram(self, x, NFFT=None, Fs=None, Fc=None, detrend=None, window=None, noverlap=None, cmap=None, xextent=None, pad_to=None, sides=None, scale_by_freq=None, mode=None, scale=None, vmin=None, vmax=None, **kwargs)**
     
     该方法用于绘制输入信号 x 的频谱图。
     
     **参数说明**:
     
     - **x**: 输入信号数组。
     - **NFFT**: FFT 窗口大小。
     - **Fs**: 采样率。
     - **Fc**: （可选）中心频率。
     - **detrend**: 去趋势处理函数或标志。
     - **window**: 指定窗函数。
     - **noverlap**: 每个窗口之间的重叠样本点数。
     - **cmap**: 使用的颜色映射。
     - **xextent**: x 轴的显示范围。
     - **pad_to**: 填充参数，将信号填充到指定长度。
     - **sides**: 指定绘制哪一侧的频谱（例如 'onesided'）。
     - **scale_by_freq**: 是否根据频率缩放幅值。
     - **mode**: 绘图模式，如 'magnitude' 或 'psd' 等。
     - **scale**: 指定缩放类型（例如 'linear' 或 'dB'）。
     - **vmin**, **vmax**: 指定颜色映射的最小和最大值。
     - **kwargs**: 其他附加参数。
     
     **应用场景**:
     
     - 用于生物信号、声音信号等时域数据转换为频域图形展示。
     - 支持通过自定义参数调整绘图效果，满足科研和工程中的定制需求。
     
     **示例**::
     
        >>> import matplotlib.pyplot as plt
        >>> from pywayne.plot import SpecgramAxes, regist_projection
        >>> regist_projection()
        >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
        >>> import numpy as np
        >>> x = np.random.randn(1024)
        >>> ax.specgram(x, NFFT=256, Fs=1000, noverlap=128, cmap='viridis', mode='magnitude')
        >>> plt.show()


regist_projection 函数
-------------------------

.. py:function:: regist_projection()

   用于将自定义投影 'z_norm' 注册到 matplotlib 系统中，使得用户可以通过设置 subplot 的 projection 参数轻松调用 SpecgramAxes 类。
   
   **应用场景**:
   
   - 在使用 matplotlib 时注册自定义投影，扩展默认的 Axes 功能。
   - 允许用户通过指定 projection='z_norm' 来使用特定的频谱图绘制功能。
   
   **示例**::
   
      >>> from pywayne.plot import regist_projection
      >>> regist_projection()
      >>> import matplotlib.pyplot as plt
      >>> fig, ax = plt.subplots(subplot_kw={'projection': 'z_norm'})
      >>> ax.specgram(x, NFFT=256, Fs=1000, noverlap=128)
      >>> plt.show()

--------------------------------------------------

通过上述示例，可以快速掌握 plot 模块中各工具的使用方法，这对于进行专业的数据可视化和频谱图绘制提供了有效支持。 