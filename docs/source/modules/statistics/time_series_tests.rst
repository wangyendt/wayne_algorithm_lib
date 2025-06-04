时间序列检验 (TimeSeriesTests)
===============================

时间序列检验类提供了8种专门用于时间序列数据分析的统计检验方法。这些检验在经济学、金融学、工程学等时间序列分析中具有重要作用。

.. py:class:: TimeSeriesTests

   时间序列检验类，包含平稳性、序列相关性、协整性等多种时间序列特性检验方法。所有方法都返回 ``TestResult`` 对象。

   **主要方法**:

检验方法详解
------------

Augmented Dickey-Fuller检验 (ADF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: adf_test(data: Union[np.ndarray, List], alpha: float = 0.05, regression: str = 'c', autolag: str = 'AIC') -> TestResult

   Augmented Dickey-Fuller单位根检验，检验时间序列的平稳性。

   **参数**:
   
   - **data**: 时间序列数据
   - **regression**: 回归模式（'c': 含常数项, 'ct': 含常数项和趋势, 'ctt': 含常数项、趋势和二次趋势, 'n': 无常数项）
   - **autolag**: 滞后阶数选择方法（'AIC', 'BIC', 'HQIC', 或具体数字）
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 时间序列数据
   - 样本量 ≥ 20
   - 数据按时间顺序排列

   **原假设**: 序列存在单位根（非平稳）

   **应用场景**:
   
   - 检验序列平稳性
   - 回归分析前的预处理
   - 确定差分阶数

   **示例**::

      >>> from pywayne.statistics import TimeSeriesTests
      >>> import numpy as np
      >>> 
      >>> tst = TimeSeriesTests()
      >>> # 非平稳序列（随机游走）
      >>> data = np.cumsum(np.random.normal(0, 1, 100))
      >>> result = tst.adf_test(data)
      >>> print(f"ADF统计量: {result.statistic:.4f}, 平稳性: {result.reject_null}")

KPSS检验
~~~~~~~~

.. py:method:: kpss_test(data: Union[np.ndarray, List], regression: str = 'c', nlags: str = 'auto', alpha: float = 0.05) -> TestResult

   KPSS检验，检验时间序列的平稳性（与ADF检验互补）。

   **参数**:
   
   - **data**: 时间序列数据
   - **regression**: 回归模式（'c': 水平平稳, 'ct': 趋势平稳）
   - **nlags**: 滞后阶数（'auto'或具体数字）
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 时间序列数据
   - 适用于各种样本量
   - 与ADF检验结合使用

   **原假设**: 序列是平稳的

   **应用场景**:
   
   - 与ADF检验结合确认平稳性
   - 区分趋势平稳和差分平稳
   - 平稳性的稳健检验

   **示例**::

      >>> # 平稳序列（白噪声）
      >>> stationary_data = np.random.normal(0, 1, 100)
      >>> result = tst.kpss_test(stationary_data)
      >>> print(f"KPSS统计量: {result.statistic:.4f}, 平稳性: {not result.reject_null}")

Ljung-Box检验
~~~~~~~~~~~~~

.. py:method:: ljung_box_test(data: Union[np.ndarray, List], lags: int = 10, alpha: float = 0.05) -> TestResult

   Ljung-Box检验，检验时间序列的序列相关性（自相关）。

   **参数**:
   
   - **data**: 时间序列数据或残差
   - **lags**: 检验的滞后阶数
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 时间序列数据
   - 常用于模型残差检验
   - 样本量 > 滞后阶数

   **原假设**: 序列无自相关（独立）

   **应用场景**:
   
   - 模型残差的独立性检验
   - ARIMA模型诊断
   - 随机性检验

   **示例**::

      >>> # 有自相关的序列
      >>> ar_data = np.zeros(100)
      >>> ar_data[0] = np.random.normal()
      >>> for i in range(1, 100):
      ...     ar_data[i] = 0.7 * ar_data[i-1] + np.random.normal(0, 0.5)
      >>> result = tst.ljung_box_test(ar_data, lags=10)
      >>> print(f"存在自相关: {result.reject_null}")

游程检验
~~~~~~~~

.. py:method:: runs_test(data: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   游程检验，检验序列的随机性。

   **参数**:
   
   - **data**: 时间序列数据
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 二分类或可二分类的数据
   - 样本量 ≥ 20
   - 用于检验随机性

   **原假设**: 序列是随机的

   **应用场景**:
   
   - 检验序列随机性
   - 质量控制中的模式检验
   - 预测模型残差分析

   **示例**::

      >>> # 周期性模式数据
      >>> pattern_data = np.tile([1, -1], 50)  # 交替模式
      >>> result = tst.runs_test(pattern_data)
      >>> print(f"随机性: {not result.reject_null}")

ARCH效应检验
~~~~~~~~~~~~

.. py:method:: arch_test(data: Union[np.ndarray, List], lags: int = 1, alpha: float = 0.05) -> TestResult

   ARCH效应检验，检验时间序列的条件异方差性。

   **参数**:
   
   - **data**: 时间序列数据或残差
   - **lags**: 滞后阶数
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 时间序列数据
   - 常用于金融数据
   - 样本量 > 滞后阶数

   **原假设**: 无ARCH效应（同方差）

   **应用场景**:
   
   - 金融时间序列分析
   - 波动率建模前的检验
   - GARCH模型的必要性判断

   **示例**::

      >>> # 模拟ARCH效应数据
      >>> arch_data = np.random.normal(0, 1, 100)
      >>> for i in range(1, 100):
      ...     vol = 0.1 + 0.8 * arch_data[i-1]**2
      ...     arch_data[i] = np.random.normal(0, np.sqrt(vol))
      >>> result = tst.arch_test(arch_data)
      >>> print(f"存在ARCH效应: {result.reject_null}")

Granger因果检验
~~~~~~~~~~~~~~~

.. py:method:: granger_causality(data1: Union[np.ndarray, List], data2: Union[np.ndarray, List], max_lag: int = 4, alpha: float = 0.05) -> TestResult

   Granger因果检验，检验一个时间序列是否有助于预测另一个序列。

   **参数**:
   
   - **data1**: 第一个时间序列
   - **data2**: 第二个时间序列
   - **max_lag**: 最大滞后阶数
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 两个时间序列
   - 序列应为平稳的
   - 样本量 > 滞后阶数

   **原假设**: 第一个序列不Granger因果于第二个序列

   **应用场景**:
   
   - 经济变量间的因果关系分析
   - 金融市场传导机制研究
   - 政策效果评估

   **示例**::

      >>> # 创建因果关系数据
      >>> x = np.random.normal(0, 1, 100)
      >>> y = np.zeros(100)
      >>> y[0] = np.random.normal()
      >>> for i in range(1, 100):
      ...     y[i] = 0.5 * x[i-1] + 0.3 * y[i-1] + np.random.normal(0, 0.5)
      >>> data = np.column_stack([x, y])
      >>> result = tst.granger_causality(data[:, 0], data[:, 1])
      >>> print(f"存在Granger因果关系: {result.reject_null}")

Engle-Granger协整检验
~~~~~~~~~~~~~~~~~~~~~

.. py:method:: engle_granger_cointegration(data1: Union[np.ndarray, List], data2: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   Engle-Granger协整检验，检验两个非平稳序列间的长期均衡关系。

   **参数**:
   
   - **data1**: 第一个时间序列
   - **data2**: 第二个时间序列
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 两个序列均为I(1)非平稳序列
   - 序列长度相等
   - 样本量 ≥ 50

   **原假设**: 两序列不存在协整关系

   **应用场景**:
   
   - 经济变量间长期关系分析
   - 金融资产间的均值回复关系
   - 配对交易策略验证

   **示例**::

      >>> # 协整序列示例
      >>> t = np.arange(100)
      >>> trend = 0.05 * t
      >>> y1 = trend + np.cumsum(np.random.normal(0, 1, 100))
      >>> y2 = 2 * trend + y1 + np.random.normal(0, 0.5, 100)
      >>> result = tst.engle_granger_cointegration(y1, y2)
      >>> print(f"存在协整关系: {result.reject_null}")

Breusch-Godfrey检验
~~~~~~~~~~~~~~~~~~~

.. py:method:: breusch_godfrey_test(residuals: Union[np.ndarray, List], lags: int = 1, alpha: float = 0.05) -> TestResult

   Breusch-Godfrey检验，检验回归残差的序列相关性。

   **参数**:
   
   - **residuals**: 回归残差
   - **lags**: 检验的滞后阶数
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 时间序列回归的残差
   - 适用于各种回归模型
   - 样本量 > 滞后阶数

   **原假设**: 残差无序列相关

   **应用场景**:
   
   - 时间序列回归诊断
   - 模型设定检验
   - ARIMA模型残差分析

   **示例**::

      >>> # 有序列相关的残差
      >>> residuals = np.zeros(100)
      >>> for i in range(1, 100):
      ...     residuals[i] = 0.6 * residuals[i-1] + np.random.normal(0, 1)
      >>> result = tst.breusch_godfrey_test(residuals)
      >>> print(f"残差存在序列相关: {result.reject_null}")

使用建议
--------

平稳性检验策略
~~~~~~~~~~~~~~

1. **单位根检验流程**:

   - **Step 1**: 使用ADF检验（原假设：非平稳）
   - **Step 2**: 使用KPSS检验（原假设：平稳）
   - **Step 3**: 结合两个检验结果判断

2. **结果解释**:

   - **ADF拒绝 + KPSS不拒绝**: 平稳序列
   - **ADF不拒绝 + KPSS拒绝**: 非平稳序列
   - **都拒绝**: 需要进一步分析或变换
   - **都不拒绝**: 结果不确定，需要更多证据

3. **差分处理**:

   - 一阶差分后重新检验
   - 注意过度差分问题
   - 考虑季节性差分

序列相关性检验
~~~~~~~~~~~~~~

1. **Ljung-Box检验应用**:

   - 选择适当的滞后阶数（通常为样本量的1/4）
   - 结合ACF/PACF图形分析
   - 用于模型诊断和残差检验

2. **游程检验补充**:

   - 检验序列的随机性
   - 适用于二分类或符号序列
   - 可以检测周期性模式

协整关系分析
~~~~~~~~~~~~

1. **协整检验前提**:

   - 确认变量的积分阶数相同（通常为I(1)）
   - 变量数量不宜过多（Engle-Granger适用于两变量）
   - 样本量要足够大

2. **结果应用**:

   - 建立误差修正模型（ECM）
   - 长期均衡关系分析
   - 短期动态调整机制研究

ARCH效应分析
~~~~~~~~~~~~

1. **检验时机**:

   - 在建立均值方程后
   - 检验残差的条件异方差
   - 确定是否需要GARCH建模

2. **后续处理**:

   - 建立GARCH族模型
   - 考虑结构突变
   - 使用稳健标准误

典型应用示例
------------

时间序列平稳性分析
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.statistics import TimeSeriesTests
   import numpy as np
   import matplotlib.pyplot as plt
   
   # 生成不同类型的时间序列
   np.random.seed(42)
   n = 200
   
   # 1. 平稳序列（AR(1)）
   stationary = np.zeros(n)
   for i in range(1, n):
       stationary[i] = 0.7 * stationary[i-1] + np.random.normal(0, 1)
   
   # 2. 非平稳序列（随机游走）
   non_stationary = np.cumsum(np.random.normal(0, 1, n))
   
   # 3. 趋势序列
   trend_series = 0.05 * np.arange(n) + np.random.normal(0, 1, n)
   
   tst = TimeSeriesTests()
   
   series_data = [
       ("平稳序列", stationary),
       ("随机游走", non_stationary),
       ("趋势序列", trend_series)
   ]
   
   print("时间序列平稳性检验结果:")
   print("=" * 60)
   
   for name, data in series_data:
       print(f"\n{name}:")
       
       # ADF检验
       adf_result = tst.adf_test(data)
       print(f"  ADF检验: 统计量={adf_result.statistic:.4f}, "
             f"p值={adf_result.p_value:.4f}, 拒绝原假设={adf_result.reject_null}")
       
       # KPSS检验
       kpss_result = tst.kpss_test(data)
       print(f"  KPSS检验: 统计量={kpss_result.statistic:.4f}, "
             f"p值={kpss_result.p_value:.4f}, 拒绝原假设={kpss_result.reject_null}")
       
       # 平稳性结论
       if adf_result.reject_null and not kpss_result.reject_null:
           conclusion = "平稳"
       elif not adf_result.reject_null and kpss_result.reject_null:
           conclusion = "非平稳"
       else:
           conclusion = "需要进一步分析"
       
       print(f"  结论: {conclusion}")

ARIMA模型诊断
~~~~~~~~~~~~~

.. code-block:: python

   from statsmodels.tsa.arima.model import ARIMA
   import numpy as np
   
   # 生成ARIMA(1,1,1)数据
   np.random.seed(42)
   n = 200
   
   # 模拟非平稳序列
   data = np.cumsum(np.random.normal(0, 1, n))
   for i in range(1, n):
       data[i] += 0.3 * (data[i-1] - data[i-2]) if i > 1 else 0
   
   # 拟合ARIMA模型
   model = ARIMA(data, order=(1, 1, 1))
   fitted_model = model.fit()
   
   # 获取残差
   residuals = fitted_model.resid
   
   tst = TimeSeriesTests()
   
   print("ARIMA模型残差诊断:")
   print("=" * 40)
   
   # Ljung-Box检验
   lb_result = tst.ljung_box_test(residuals, lags=10)
   print(f"Ljung-Box检验: p值={lb_result.p_value:.4f}, "
         f"残差独立={not lb_result.reject_null}")
   
   # ARCH效应检验
   arch_result = tst.arch_test(residuals)
   print(f"ARCH效应检验: p值={arch_result.p_value:.4f}, "
         f"同方差={not arch_result.reject_null}")
   
   # 游程检验
   runs_result = tst.runs_test(residuals)
   print(f"游程检验: p值={runs_result.p_value:.4f}, "
         f"随机性={not runs_result.reject_null}")
   
   # 模型诊断结论
   if not lb_result.reject_null and not arch_result.reject_null and not runs_result.reject_null:
       print("\n结论: 模型残差表现良好，模型设定合理")
   else:
       print("\n结论: 模型可能存在设定问题，需要调整")

金融时间序列分析
~~~~~~~~~~~~~~~~

.. code-block:: python

   # 模拟股票价格和收益率数据
   np.random.seed(42)
   n = 300
   
   # 股票价格（几何布朗运动）
   returns = np.random.normal(0.001, 0.02, n)  # 日收益率
   prices = 100 * np.exp(np.cumsum(returns))   # 价格序列
   
   # 引入波动率聚集效应
   vol_returns = np.zeros(n)
   vol_returns[0] = returns[0]
   for i in range(1, n):
       vol = 0.02 * (1 + 0.8 * vol_returns[i-1]**2)  # GARCH效应
       vol_returns[i] = np.random.normal(0, vol)
   
   tst = TimeSeriesTests()
   
   print("金融时间序列分析:")
   print("=" * 40)
   
   # 价格序列平稳性
   print("1. 价格序列分析:")
   price_adf = tst.adf_test(prices)
   print(f"   ADF检验: p值={price_adf.p_value:.4f}, 平稳={price_adf.reject_null}")
   
   # 收益率序列平稳性
   print("\n2. 收益率序列分析:")
   return_adf = tst.adf_test(returns)
   print(f"   ADF检验: p值={return_adf.p_value:.4f}, 平稳={return_adf.reject_null}")
   
   # 收益率序列的ARCH效应
   arch_result = tst.arch_test(vol_returns, lags=5)
   print(f"   ARCH效应: p值={arch_result.p_value:.4f}, 存在={arch_result.reject_null}")
   
   # 序列相关性
   lb_result = tst.ljung_box_test(vol_returns, lags=10)
   print(f"   序列相关: p值={lb_result.p_value:.4f}, 存在={lb_result.reject_null}")
   
   print("\n结论:")
   print(f"   价格序列: {'平稳' if price_adf.reject_null else '非平稳'}")
   print(f"   收益率序列: {'平稳' if return_adf.reject_null else '非平稳'}")
   print(f"   波动率聚集: {'存在' if arch_result.reject_null else '不存在'}")

协整关系分析
~~~~~~~~~~~~

.. code-block:: python

   # 模拟协整的经济变量
   np.random.seed(42)
   n = 200
   
   # 生成两个协整序列
   # 假设是消费和收入的长期关系
   common_trend = np.cumsum(np.random.normal(0, 1, n))  # 共同趋势
   
   income = common_trend + np.random.normal(0, 0.5, n)
   consumption = 0.8 * income + np.random.normal(0, 0.3, n)
   
   # 生成非协整序列作为对比
   gdp = np.cumsum(np.random.normal(0, 1, n))
   unemployment = np.cumsum(np.random.normal(0, 1, n))
   
   tst = TimeSeriesTests()
   
   print("协整关系分析:")
   print("=" * 40)
   
   # 检验变量的单整性
   print("1. 单位根检验:")
   variables = [income, consumption, gdp, unemployment]
   var_names = ['收入', '消费', 'GDP', '失业率']
   
   for name, var in zip(var_names, variables):
       adf_result = tst.adf_test(var)
       print(f"   {name}: ADF统计量={adf_result.statistic:.4f}, "
             f"p值={adf_result.p_value:.4f}, I(1)={not adf_result.reject_null}")
   
   # 协整检验
   print("\n2. 协整检验:")
   
   # 理论上协整的变量对
   cointegration_result1 = tst.engle_granger_cointegration(income, consumption)
   print(f"   收入-消费: 统计量={cointegration_result1.statistic:.4f}, "
         f"p值={cointegration_result1.p_value:.4f}, 协整={cointegration_result1.reject_null}")
   
   # 理论上非协整的变量对
   cointegration_result2 = tst.engle_granger_cointegration(gdp, unemployment)
   print(f"   GDP-失业率: 统计量={cointegration_result2.statistic:.4f}, "
         f"p值={cointegration_result2.p_value:.4f}, 协整={cointegration_result2.reject_null}")
   
   print("\n结论:")
   if cointegration_result1.reject_null:
       print("   收入和消费存在长期均衡关系，可以建立误差修正模型")
   if not cointegration_result2.reject_null:
       print("   GDP和失业率不存在协整关系，需要分别建模")

Granger因果关系分析
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 模拟具有因果关系的经济变量
   np.random.seed(42)
   n = 150
   
   # 生成Granger因果关系：货币供应量 → 通胀率
   money_supply = np.random.normal(0, 1, n)
   inflation = np.zeros(n)
   
   for i in range(2, n):
       # 通胀率受过去货币供应量影响
       inflation[i] = (0.4 * money_supply[i-1] + 0.3 * money_supply[i-2] + 
                      0.2 * inflation[i-1] + np.random.normal(0, 0.5))
   
   # 构建多变量时间序列
   data = np.column_stack([money_supply[2:], inflation[2:]])
   
   tst = TimeSeriesTests()
   
   print("Granger因果关系分析:")
   print("=" * 40)
   
   # 检验平稳性（Granger因果检验要求平稳序列）
   print("1. 平稳性检验:")
   for i, name in enumerate(['货币供应量', '通胀率']):
       adf_result = tst.adf_test(data[:, i])
       print(f"   {name}: ADF p值={adf_result.p_value:.4f}, 平稳={adf_result.reject_null}")
   
   # Granger因果检验
   print("\n2. Granger因果检验:")
   
   # 货币供应量 → 通胀率
   granger_result = tst.granger_causality(data[:, 0], data[:, 1])
   print(f"   货币供应量 → 通胀率: F统计量={granger_result.statistic:.4f}, "
         f"p值={granger_result.p_value:.4f}, 因果关系={granger_result.reject_null}")
   
   # 反向检验：通胀率 → 货币供应量
   data_reversed = np.column_stack([data[:, 1], data[:, 0]])
   granger_result_rev = tst.granger_causality(data_reversed[:, 0], data_reversed[:, 1])
   print(f"   通胀率 → 货币供应量: F统计量={granger_result_rev.statistic:.4f}, "
         f"p值={granger_result_rev.p_value:.4f}, 因果关系={granger_result_rev.reject_null}")
   
   print("\n结论:")
   if granger_result.reject_null and not granger_result_rev.reject_null:
       print("   存在单向因果关系：货币供应量Granger因果于通胀率")
   elif granger_result.reject_null and granger_result_rev.reject_null:
       print("   存在双向因果关系")
   else:
       print("   不存在显著的Granger因果关系")

注意事项
--------

1. **数据预处理**:
   - 确保数据按时间顺序排列
   - 处理缺失值和异常值
   - 考虑季节性调整

2. **样本量要求**:
   - 大多数检验需要足够的样本量
   - 滞后阶数不宜过大相对于样本量
   - 考虑样本量对检验功效的影响

3. **模型设定**:
   - 选择适当的趋势项和常数项
   - 滞后阶数的选择很重要
   - 结合经济理论和数据特征

4. **结果解释**:
   - 平稳性检验需要结合多种方法
   - 因果关系不等于真实因果
   - 协整关系表示长期均衡而非短期关系 