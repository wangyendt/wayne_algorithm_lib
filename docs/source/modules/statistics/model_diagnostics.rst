模型诊断 (ModelDiagnostics)
============================

模型诊断类提供了8种方法来检验回归模型的基本假设和诊断模型质量。这些检验在线性回归、时间序列模型和其他统计模型的验证中至关重要。

.. py:class:: ModelDiagnostics

   模型诊断类，包含方差齐性、正态性、多重共线性等模型假设检验方法。所有方法都返回 ``TestResult`` 对象。

   **主要方法**:

检验方法详解
------------

Breusch-Pagan检验
~~~~~~~~~~~~~~~~~

.. py:method:: breusch_pagan_test(residuals: Union[np.ndarray, List], exog: Union[np.ndarray, List[List]], alpha: float = 0.05) -> TestResult

   Breusch-Pagan检验，检验回归模型的异方差性。

   **参数**:
   
   - **residuals**: 回归残差
   - **exog**: 解释变量矩阵（设计矩阵）
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 线性回归模型
   - 残差应为模型拟合后的残差
   - 样本量 ≥ 变量数 + 10

   **原假设**: 误差项具有同方差性

   **应用场景**:
   
   - 线性回归模型诊断
   - 确定是否需要加权最小二乘法
   - 模型改进和变换指导

   **示例**::

      >>> from pywayne.statistics import ModelDiagnostics
      >>> import numpy as np
      >>> from sklearn.linear_model import LinearRegression
      >>> 
      >>> md = ModelDiagnostics()
      >>> # 生成异方差数据
      >>> X = np.random.normal(0, 1, (100, 2))
      >>> y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 1 + X[:, 0]**2, 100)
      >>> model = LinearRegression().fit(X, y)
      >>> residuals = y - model.predict(X)
      >>> result = md.breusch_pagan_test(residuals, X)
      >>> print(f"存在异方差: {result.reject_null}")

White检验
~~~~~~~~~

.. py:method:: white_test(residuals: Union[np.ndarray, List], exog: Union[np.ndarray, List[List]], alpha: float = 0.05) -> TestResult

   White检验，检验异方差性的稳健检验方法。

   **参数**:
   
   - **residuals**: 回归残差
   - **exog**: 解释变量矩阵
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 线性回归模型
   - 对函数形式误设更稳健
   - 适用于各种异方差形式

   **原假设**: 误差项具有同方差性

   **应用场景**:
   
   - 异方差的稳健检验
   - 函数形式不确定时的诊断
   - Breusch-Pagan检验的补充

   **示例**::

      >>> # 使用相同的异方差数据
      >>> result = md.white_test(residuals, X)
      >>> print(f"White检验检测到异方差: {result.reject_null}")

Goldfeld-Quandt检验
~~~~~~~~~~~~~~~~~~~

.. py:method:: goldfeld_quandt_test(y: Union[np.ndarray, List], x: Union[np.ndarray, List[List]], split: float = 0.5, alpha: float = 0.05) -> TestResult

   Goldfeld-Quandt检验，通过分段回归检验异方差性。

   **参数**:
   
   - **y**: 因变量
   - **x**: 自变量矩阵
   - **split**: 分割点比例，默认0.5
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 线性回归模型
   - 异方差性与某个变量单调相关
   - 样本量足够进行分段

   **原假设**: 两段数据具有相同方差

   **应用场景**:
   
   - 特定变量相关的异方差检验
   - 结构性异方差分析
   - 异方差模式识别

Durbin-Watson检验
~~~~~~~~~~~~~~~~~

.. py:method:: durbin_watson_test(residuals: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   Durbin-Watson检验，检验回归残差的一阶自相关。

   **参数**:
   
   - **residuals**: 回归残差（按时间顺序排列）
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 时间序列回归模型
   - 残差按时间顺序排列
   - 主要检验一阶自相关

   **原假设**: 残差无一阶自相关

   **应用场景**:
   
   - 时间序列回归诊断
   - 检验模型的动态设定
   - 确定是否需要自回归项

   **示例**::

      >>> # 生成有自相关的残差
      >>> residuals = np.zeros(100)
      >>> for i in range(1, 100):
      ...     residuals[i] = 0.7 * residuals[i-1] + np.random.normal(0, 1)
      >>> result = md.durbin_watson_test(residuals)
      >>> print(f"DW统计量: {result.statistic:.4f}, 存在自相关: {result.reject_null}")

方差膨胀因子 (VIF)
~~~~~~~~~~~~~~~~~~

.. py:method:: variance_inflation_factor(X: Union[np.ndarray, List], feature_names: List[str] = None) -> Union[List[float], Dict[str, float]]

   计算方差膨胀因子，检验多重共线性问题。

   **参数**:
   
   - **X**: 解释变量矩阵（不包含常数项）
   - **feature_names**: 特征名称列表

   **适用条件**:
   
   - 多元线性回归
   - 变量数 ≥ 2
   - 样本量 > 变量数

   **原假设**: 不存在严重多重共线性

   **应用场景**:
   
   - 多重共线性诊断
   - 变量选择指导
   - 模型简化建议

   **VIF解释标准**:
   
   - VIF < 5: 无明显共线性问题
   - 5 ≤ VIF < 10: 中等共线性
   - VIF ≥ 10: 严重共线性

   **示例**::

      >>> # 生成有共线性的数据
      >>> X1 = np.random.normal(0, 1, 100)
      >>> X2 = X1 + np.random.normal(0, 0.1, 100)  # 高度相关
      >>> X3 = np.random.normal(0, 1, 100)         # 独立
      >>> X = np.column_stack([X1, X2, X3])
      >>> result = md.variance_inflation_factor(X)
      >>> print(f"VIF检验结果: {result}")

Levene检验
~~~~~~~~~~

.. py:method:: levene_test(*groups, center: str = 'median', alpha: float = 0.05) -> TestResult

   Levene检验，检验多组数据的方差齐性。

   **参数**:
   
   - **groups**: 多个样本组
   - **center**: 中心化方法（'median', 'mean', 'trimmed'）
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 两组或多组独立样本
   - 对正态性假设稳健
   - 适用于各种分布

   **原假设**: 各组方差相等

   **应用场景**:
   
   - 方差分析前的假设检验
   - t检验的方差齐性检验
   - 分组数据的方差比较

   **示例**::

      >>> # 生成不等方差的组
      >>> group1 = np.random.normal(0, 1, 50)     # 标准差=1
      >>> group2 = np.random.normal(0, 2, 50)     # 标准差=2
      >>> group3 = np.random.normal(0, 1.5, 50)   # 标准差=1.5
      >>> result = md.levene_test(group1, group2, group3)
      >>> print(f"方差齐性: {not result.reject_null}")

Bartlett检验
~~~~~~~~~~~~

.. py:method:: bartlett_test(*groups: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   Bartlett检验，检验多组数据的方差齐性（假设正态分布）。

   **参数**:
   
   - **groups**: 多个样本组
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 各组数据近似正态分布
   - 对非正态性敏感
   - 两组或多组独立样本

   **原假设**: 各组方差相等

   **应用场景**:
   
   - 正态数据的方差齐性检验
   - 传统方差分析前的检验
   - 与Levene检验对比使用

残差正态性检验
~~~~~~~~~~~~~~

.. py:method:: residual_normality_test(residuals: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   综合检验回归残差的正态性。

   **参数**:
   
   - **residuals**: 回归残差
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 回归模型残差
   - 样本量 ≥ 20
   - 用于模型假设验证

   **原假设**: 残差服从正态分布

   **应用场景**:
   
   - 回归模型诊断
   - 推断统计的有效性检验
   - 模型改进指导

   **示例**::

      >>> # 生成非正态残差
      >>> residuals = np.random.exponential(1, 100) - 1  # 偏斜分布
      >>> result = md.residual_normality_test(residuals)
      >>> print(f"残差正态性: {not result.reject_null}")

使用建议
--------

模型诊断流程
~~~~~~~~~~~~

1. **基本诊断顺序**:

   - **Step 1**: 残差正态性检验
   - **Step 2**: 异方差性检验（Breusch-Pagan, White）
   - **Step 3**: 自相关检验（Durbin-Watson）
   - **Step 4**: 多重共线性检验（VIF）

2. **针对性诊断**:

   - **时间序列数据**: 重点关注自相关
   - **截面数据**: 重点关注异方差和共线性
   - **面板数据**: 需要综合考虑各种问题

异方差处理策略
~~~~~~~~~~~~~~

1. **检验选择**:

   - **Breusch-Pagan**: 经典检验，适用性广
   - **White**: 对函数形式稳健
   - **Goldfeld-Quandt**: 特定变量相关的异方差

2. **后续处理**:

   - 加权最小二乘法（WLS）
   - 稳健标准误
   - 数据变换（对数、平方根等）

多重共线性处理
~~~~~~~~~~~~~~

1. **VIF判断标准**:

   - VIF < 5: 可接受
   - 5 ≤ VIF < 10: 需要注意
   - VIF ≥ 10: 需要处理

2. **处理方法**:

   - 删除高度相关变量
   - 主成分分析
   - 岭回归等正则化方法

自相关问题处理
~~~~~~~~~~~~~~

1. **Durbin-Watson统计量解释**:

   - DW ≈ 2: 无自相关
   - DW < 1.5 或 DW > 2.5: 可能存在自相关
   - 需要结合临界值表判断

2. **处理方法**:

   - 添加滞后项
   - 自回归误差模型
   - 差分变换

典型应用示例
------------

线性回归模型完整诊断
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.statistics import ModelDiagnostics, NormalityTests
   import numpy as np
   from sklearn.linear_model import LinearRegression
   import matplotlib.pyplot as plt
   
   # 生成回归数据（包含各种问题）
   np.random.seed(42)
   n = 200
   
   # 自变量
   X1 = np.random.normal(0, 1, n)
   X2 = 0.8 * X1 + np.random.normal(0, 0.5, n)  # 与X1相关（共线性）
   X3 = np.random.normal(0, 1, n)
   X = np.column_stack([X1, X2, X3])
   
   # 因变量（引入异方差和非正态误差）
   heterosced_error = np.random.normal(0, 1 + np.abs(X1), n)  # 异方差误差
   y = 2 + 1.5*X1 + 0.8*X2 + 1.2*X3 + heterosced_error
   
   # 拟合模型
   model = LinearRegression()
   model.fit(X, y)
   y_pred = model.predict(X)
   residuals = y - y_pred
   
   md = ModelDiagnostics()
   nt = NormalityTests()
   
   print("回归模型诊断报告")
   print("=" * 50)
   
   # 1. 残差正态性检验
   print("\n1. 残差正态性检验:")
   normality_result = md.residual_normality_test(residuals)
   print(f"   正态性检验: p值={normality_result.p_value:.4f}, "
         f"正态={not normality_result.reject_null}")
   
   # 2. 异方差性检验
   print("\n2. 异方差性检验:")
   X_with_const = np.column_stack([np.ones(n), X])  # 添加常数项
   
   bp_result = md.breusch_pagan_test(residuals, X_with_const)
   print(f"   Breusch-Pagan: p值={bp_result.p_value:.4f}, "
         f"同方差={not bp_result.reject_null}")
   
   white_result = md.white_test(residuals, X_with_const)
   print(f"   White检验: p值={white_result.p_value:.4f}, "
         f"同方差={not white_result.reject_null}")
   
   # 3. 多重共线性检验
   print("\n3. 多重共线性检验:")
   vif_result = md.variance_inflation_factor(X)
   print(f"   VIF检验: {vif_result}")
   
   # 4. 诊断结论
   print("\n4. 诊断结论:")
   issues = []
   if normality_result.reject_null:
       issues.append("残差非正态")
   if bp_result.reject_null or white_result.reject_null:
       issues.append("存在异方差")
   if "高VIF" in vif_result:  # 简化判断
       issues.append("多重共线性")
   
   if issues:
       print(f"   发现问题: {', '.join(issues)}")
       print("   建议: 考虑模型变换、稳健标准误或变量选择")
   else:
       print("   模型诊断良好，基本假设满足")

时间序列回归诊断
~~~~~~~~~~~~~~~~

.. code-block:: python

   # 模拟时间序列回归数据
   np.random.seed(42)
   n = 150
   
   # 时间趋势和周期成分
   t = np.arange(n)
   trend = 0.05 * t
   seasonal = 2 * np.sin(2 * np.pi * t / 12)  # 年度周期
   
   # 自变量
   X = np.random.normal(0, 1, n)
   
   # 因变量（包含自相关误差）
   y = 10 + trend + seasonal + 1.5 * X
   
   # 添加自相关误差
   ar_errors = np.zeros(n)
   for i in range(1, n):
       ar_errors[i] = 0.6 * ar_errors[i-1] + np.random.normal(0, 1)
   
   y += ar_errors
   
   # 简单线性回归（忽略时间结构）
   model = LinearRegression()
   X_reshaped = X.reshape(-1, 1)
   model.fit(X_reshaped, y)
   y_pred = model.predict(X_reshaped)
   residuals = y - y_pred
   
   md = ModelDiagnostics()
   
   print("时间序列回归诊断:")
   print("=" * 40)
   
   # Durbin-Watson检验
   dw_result = md.durbin_watson_test(residuals)
   print(f"Durbin-Watson统计量: {dw_result.statistic:.4f}")
   
   # 解释DW统计量
   dw_stat = dw_result.statistic
   if dw_stat < 1.5:
       autocorr_conclusion = "存在正自相关"
   elif dw_stat > 2.5:
       autocorr_conclusion = "存在负自相关"
   else:
       autocorr_conclusion = "无显著自相关"
   
   print(f"自相关诊断: {autocorr_conclusion}")
   
   # 残差正态性
   norm_result = md.residual_normality_test(residuals)
   print(f"残差正态性: p值={norm_result.p_value:.4f}, "
         f"正态={not norm_result.reject_null}")
   
   # 建议
   if dw_stat < 1.5 or dw_stat > 2.5:
       print("\n建议:")
       print("- 考虑添加滞后因变量")
       print("- 使用自回归误差模型")
       print("- 采用稳健标准误")

方差齐性检验应用
~~~~~~~~~~~~~~~~

.. code-block:: python

   # 生成不同方差的组数据
   np.random.seed(42)
   
   # 三组数据，方差递增
   group1 = np.random.normal(50, 5, 40)    # 均值=50, 标准差=5
   group2 = np.random.normal(52, 8, 40)    # 均值=52, 标准差=8
   group3 = np.random.normal(48, 12, 40)   # 均值=48, 标准差=12
   
   md = ModelDiagnostics()
   
   print("方差齐性检验比较:")
   print("=" * 40)
   
   # Levene检验（稳健）
   levene_result = md.levene_test(group1, group2, group3)
   print(f"Levene检验: 统计量={levene_result.statistic:.4f}, "
         f"p值={levene_result.p_value:.4f}")
   print(f"方差齐性: {not levene_result.reject_null}")
   
   # Bartlett检验（假设正态）
   bartlett_result = md.bartlett_test(group1, group2, group3)
   print(f"Bartlett检验: 统计量={bartlett_result.statistic:.4f}, "
         f"p值={bartlett_result.p_value:.4f}")
   print(f"方差齐性: {not bartlett_result.reject_null}")
   
   # 描述性统计
   print("\n各组描述性统计:")
   groups = [group1, group2, group3]
   for i, group in enumerate(groups, 1):
       print(f"组{i}: 均值={np.mean(group):.2f}, 标准差={np.std(group):.2f}")
   
   # 检验比较
   print("\n检验方法比较:")
   if levene_result.p_value != bartlett_result.p_value:
       print("- Levene检验和Bartlett检验结果可能不同")
       if levene_result.p_value > bartlett_result.p_value:
           print("- Levene检验更稳健，建议采用其结果")
       else:
           print("- 两种检验结果一致性较好")

多重共线性诊断
~~~~~~~~~~~~~~

.. code-block:: python

   # 生成具有不同共线性程度的数据
   np.random.seed(42)
   n = 100
   
   # 独立变量
   X1 = np.random.normal(0, 1, n)
   
   # 不同程度的相关变量
   X2 = X1 + np.random.normal(0, 0.1, n)        # 高度相关
   X3 = 0.5 * X1 + np.random.normal(0, 1, n)    # 中度相关
   X4 = np.random.normal(0, 1, n)               # 独立
   
   # 构建设计矩阵
   X_all = np.column_stack([X1, X2, X3, X4])
   X_subset = np.column_stack([X1, X3, X4])  # 移除高共线性变量
   
   md = ModelDiagnostics()
   
   print("多重共线性诊断:")
   print("=" * 40)
   
   # 全变量VIF
   print("1. 包含所有变量:")
   vif_all = md.variance_inflation_factor(X_all)
   print(f"   VIF结果: {vif_all}")
   
   # 移除高共线性变量后
   print("\n2. 移除高共线性变量后:")
   vif_subset = md.variance_inflation_factor(X_subset)
   print(f"   VIF结果: {vif_subset}")
   
   # 相关性矩阵分析
   print("\n3. 变量相关性分析:")
   correlation_matrix = np.corrcoef(X_all.T)
   variable_names = ['X1', 'X2', 'X3', 'X4']
   
   print("   相关系数矩阵:")
   for i, name_i in enumerate(variable_names):
       for j, name_j in enumerate(variable_names):
           if i < j:
               corr = correlation_matrix[i, j]
               print(f"   {name_i}-{name_j}: {corr:.3f}")
   
   print("\n4. 建议:")
   if "高VIF" in vif_all:
       print("   - 检测到严重多重共线性")
       print("   - 建议移除高度相关的变量")
       print("   - 或考虑主成分分析、岭回归等方法")
   else:
       print("   - 多重共线性问题不严重")
       print("   - 可以保留当前变量设定")

注意事项
--------

1. **检验序列**:
   - 按照逻辑顺序进行诊断
   - 某些问题可能相互影响
   - 综合考虑多个检验结果

2. **样本量要求**:
   - 确保足够的样本量
   - 小样本时结果可能不稳定
   - 考虑检验的功效

3. **实际意义**:
   - 统计显著性不等于实际重要性
   - 结合专业知识解释结果
   - 考虑模型的预测性能

4. **修正方法**:
   - 数据变换（对数、平方根等）
   - 稳健估计方法
   - 模型重新设定 