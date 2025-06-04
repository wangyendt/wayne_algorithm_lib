正态性检验 (NormalityTests)
=============================

正态性检验类提供了8种不同的方法来检验数据是否符合正态分布或其他特定分布。这些检验在数据预处理、模型假设验证和质量控制中起着重要作用。

.. py:class:: NormalityTests

   正态性检验类，包含多种检验数据分布特性的方法。所有方法都返回 ``TestResult`` 对象。

   **主要方法**:

检验方法详解
------------

Shapiro-Wilk检验
~~~~~~~~~~~~~~~~

.. py:method:: shapiro_wilk(data: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   Shapiro-Wilk正态性检验，是最常用和最强大的正态性检验方法之一。

   **参数**:
   
   - **data**: 输入数据，一维数组或列表
   - **alpha**: 显著性水平，默认0.05

   **适用条件**:
   
   - 样本量：3 ≤ n ≤ 5000
   - 数据类型：连续型数据
   - 假设：数据来自连续分布

   **原假设**: 数据来自正态分布
   
   **应用场景**:
   
   - 小到中等样本量的正态性检验
   - 回归分析前的残差检验
   - 质量控制中的过程稳定性检验

   **示例**::

      >>> from pywayne.statistics import NormalityTests
      >>> import numpy as np
      >>> 
      >>> nt = NormalityTests()
      >>> data = np.random.normal(0, 1, 100)
      >>> result = nt.shapiro_wilk(data)
      >>> print(f"统计量: {result.statistic:.4f}, p值: {result.p_value:.4f}")

Kolmogorov-Smirnov检验（单样本）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: ks_test_normal(data: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   单样本Kolmogorov-Smirnov检验，用于检验样本是否来自指定的理论分布。

   **参数**:
   
   - **data**: 输入数据
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 数据类型：连续型数据
   - 需要完全指定理论分布的参数

   **原假设**: 数据来自指定的理论分布

   **应用场景**:
   
   - 拟合优度检验
   - 模型验证
   - 数据质量评估

   **示例**::

      >>> # 检验是否来自标准正态分布
      >>> result = nt.ks_test_normal(data)
      >>> print(f"KS统计量: {result.statistic:.4f}")

Kolmogorov-Smirnov检验（双样本）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: ks_test_two_sample(data1: Union[np.ndarray, List], data2: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   双样本Kolmogorov-Smirnov检验，检验两个独立样本是否来自同一分布。

   **参数**:
   
   - **data1**: 第一个样本
   - **data2**: 第二个样本  
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 两样本独立
   - 连续型数据
   - 对分布形状敏感

   **原假设**: 两个样本来自同一分布

   **应用场景**:
   
   - A/B测试中比较两组数据分布
   - 检验处理前后数据分布变化
   - 批次间质量一致性检验

Anderson-Darling检验
~~~~~~~~~~~~~~~~~~~~

.. py:method:: anderson_darling(data: Union[np.ndarray, List], dist: str = 'norm', alpha: float = 0.05) -> TestResult

   Anderson-Darling检验，对分布尾部更敏感的正态性检验。

   **参数**:
   
   - **data**: 输入数据
   - **dist**: 检验的分布类型
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 连续型数据
   - 对尾部偏差敏感
   - 样本量不宜过小

   **原假设**: 数据来自指定分布

   **应用场景**:
   
   - 金融数据的尾部风险分析
   - 质量控制中的异常检测
   - 对尾部特性要求严格的场景

D'Agostino-Pearson检验
~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: dagostino_pearson(data: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   D'Agostino-Pearson K²检验，基于偏度和峰度的正态性检验。

   **参数**:
   
   - **data**: 输入数据
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 样本量 ≥ 20
   - 连续型数据
   - 对偏度和峰度敏感

   **原假设**: 数据的偏度和峰度符合正态分布

   **应用场景**:
   
   - 中等样本量的正态性检验
   - 关注数据对称性和尾部厚度
   - 数据变换效果评估

Jarque-Bera检验
~~~~~~~~~~~~~~~

.. py:method:: jarque_bera(data: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   Jarque-Bera检验，常用于回归残差的正态性检验。

   **参数**:
   
   - **data**: 输入数据
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 大样本（n ≥ 30）
   - 基于渐近χ²分布
   - 对偏度和峰度的联合检验

   **原假设**: 数据来自正态分布

   **应用场景**:
   
   - 回归分析中残差检验
   - 时间序列模型诊断
   - 大样本正态性检验

卡方拟合优度检验
~~~~~~~~~~~~~~~~

.. py:method:: chi_square_goodness_of_fit(observed: Union[np.ndarray, List], expected: Union[np.ndarray, List] = None, alpha: float = 0.05) -> TestResult

   卡方拟合优度检验，检验分类数据的观察频数与期望频数的差异。

   **参数**:
   
   - **observed**: 观察频数
   - **expected**: 期望频数（可选）
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 分类数据
   - 期望频数 ≥ 5
   - 样本量 ≥ 50

   **原假设**: 观察频数与期望频数无显著差异

   **应用场景**:
   
   - 离散分布拟合检验
   - 市场研究中的偏好分析
   - 质量控制中的缺陷分类分析

使用建议
--------

方法选择指南
~~~~~~~~~~~~

1. **小样本 (n < 30)**:
   - 首选：Shapiro-Wilk检验
   - 备选：Anderson-Darling检验

2. **中等样本 (30 ≤ n ≤ 300)**:
   - 首选：Shapiro-Wilk检验
   - 备选：D'Agostino-Pearson检验

3. **大样本 (n > 300)**:
   - 首选：Jarque-Bera检验
   - 备选：Kolmogorov-Smirnov检验

4. **对尾部敏感**:
   - 首选：Anderson-Darling检验

5. **分类数据**:
   - 使用：卡方拟合优度检验

数据预处理建议
~~~~~~~~~~~~~~

1. **异常值处理**:
   - 检验前识别和处理极端异常值
   - 考虑使用稳健的检验方法

2. **样本量考虑**:
   - 确保样本量满足检验要求
   - 样本量过大时注意实际显著性

3. **数据变换**:
   - 对数变换：处理右偏数据
   - Box-Cox变换：寻找最佳变换
   - 标准化：消除量纲影响

结果解释
~~~~~~~~

1. **p值解释**:
   - p < 0.05：拒绝正态性假设
   - p ≥ 0.05：不能拒绝正态性假设

2. **实际意义**:
   - 大样本时，微小偏差也可能显著
   - 结合图形化方法（Q-Q图、直方图）
   - 考虑后续分析的稳健性

3. **多重检验**:
   - 使用多种方法交叉验证
   - 注意多重比较校正

典型应用示例
------------

数据预处理检验
~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.statistics import NormalityTests
   import numpy as np
   import matplotlib.pyplot as plt
   
   # 生成测试数据
   normal_data = np.random.normal(0, 1, 100)
   skewed_data = np.random.exponential(2, 100)
   
   nt = NormalityTests()
   
   # 正态数据检验
   result_normal = nt.shapiro_wilk(normal_data)
   print(f"正态数据: p值={result_normal.p_value:.4f}, 正态={not result_normal.reject_null}")
   
   # 偏斜数据检验
   result_skewed = nt.shapiro_wilk(skewed_data)
   print(f"偏斜数据: p值={result_skewed.p_value:.4f}, 正态={not result_skewed.reject_null}")

模型残差检验
~~~~~~~~~~~~

.. code-block:: python

   from sklearn.linear_model import LinearRegression
   from pywayne.statistics import NormalityTests
   
   # 假设已有回归模型和残差
   # residuals = model.predict(X) - y
   
   nt = NormalityTests()
   
   # 使用多种方法检验残差正态性
   sw_result = nt.shapiro_wilk(residuals)
   jb_result = nt.jarque_bera(residuals)
   
   print("残差正态性检验结果:")
   print(f"Shapiro-Wilk: p值={sw_result.p_value:.4f}")
   print(f"Jarque-Bera: p值={jb_result.p_value:.4f}")
   
   if sw_result.p_value > 0.05 and jb_result.p_value > 0.05:
       print("残差近似正态分布，模型假设满足")
   else:
       print("残差非正态，考虑模型变换或稳健方法")

批量检验
~~~~~~~~

.. code-block:: python

   # 对多个数据集进行批量正态性检验
   datasets = [
       np.random.normal(0, 1, 100),      # 正态
       np.random.uniform(-2, 2, 100),    # 均匀
       np.random.exponential(1, 100),    # 指数
   ]
   
   nt = NormalityTests()
   
   for i, data in enumerate(datasets):
       result = nt.shapiro_wilk(data)
       print(f"数据集{i+1}: 统计量={result.statistic:.4f}, "
             f"p值={result.p_value:.4f}, 正态={not result.reject_null}")

注意事项
--------

1. **检验功效**:
   - 小样本时检验功效可能不足
   - 大样本时可能过于敏感

2. **多重检验问题**:
   - 同时使用多种方法时考虑校正
   - 结合实际需求选择合适方法

3. **实际应用**:
   - 正态性检验是手段而非目的
   - 关注后续分析的稳健性
   - 必要时使用非参数方法 