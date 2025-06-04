相关性检验 (CorrelationTests)
==============================

相关性检验类提供了6种方法来评估变量间的关联性和独立性。这些检验在数据探索、特征选择、依赖性分析等方面有重要应用。

.. py:class:: CorrelationTests

   相关性检验类，包含多种评估变量关联性的统计方法。所有方法都返回 ``TestResult`` 对象。

   **主要方法**:

检验方法详解
------------

Pearson相关检验
~~~~~~~~~~~~~~~

.. py:method:: pearson_correlation(x: Union[np.ndarray, List], y: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   Pearson积矩相关检验，评估两个连续变量间的线性相关性。

   **参数**:
   
   - **x**: 第一个变量的数据
   - **y**: 第二个变量的数据
   - **alpha**: 显著性水平，默认0.05

   **适用条件**:
   
   - 两变量均为连续型数据
   - 数据近似服从双变量正态分布
   - 线性关系假设

   **原假设**: 两变量间无线性相关关系 (ρ = 0)

   **应用场景**:
   
   - 连续变量间线性关系评估
   - 特征选择中的冗余性检测
   - 回归分析前的变量关系探索

   **示例**::

      >>> from pywayne.statistics import CorrelationTests
      >>> import numpy as np
      >>> 
      >>> ct = CorrelationTests()
      >>> x = np.random.normal(0, 1, 100)
      >>> y = 2 * x + np.random.normal(0, 0.5, 100)  # 有相关性
      >>> result = ct.pearson_correlation(x, y)
      >>> print(f"相关系数: {result.statistic:.4f}, p值: {result.p_value:.4f}")

Spearman秩相关检验
~~~~~~~~~~~~~~~~~~

.. py:method:: spearman_correlation(x: Union[np.ndarray, List], y: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   Spearman秩相关检验，评估两个变量间的单调关系。

   **参数**:
   
   - **x**: 第一个变量的数据
   - **y**: 第二个变量的数据
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 数据至少为序数水平
   - 无需正态性假设
   - 评估单调关系（不限于线性）

   **原假设**: 两变量间无单调关系

   **应用场景**:
   
   - 非正态数据的相关分析
   - 序数变量的关联分析
   - 非线性单调关系检测

   **示例**::

      >>> # 非线性但单调的关系
      >>> x = np.random.uniform(0, 10, 100)
      >>> y = x**2 + np.random.normal(0, 5, 100)  # 非线性关系
      >>> result = ct.spearman_correlation(x, y)
      >>> print(f"Spearman系数: {result.statistic:.4f}")

Kendall τ相关检验
~~~~~~~~~~~~~~~~~

.. py:method:: kendall_tau(x: Union[np.ndarray, List], y: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   Kendall τ相关检验，基于一致对和不一致对的秩相关。

   **参数**:
   
   - **x**: 第一个变量的数据
   - **y**: 第二个变量的数据
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 序数或连续数据
   - 对异常值相对稳健
   - 小样本表现良好

   **原假设**: 两变量独立

   **应用场景**:
   
   - 小样本相关分析
   - 存在异常值时的稳健相关
   - 序数数据的关联分析

   **示例**::

      >>> # 包含异常值的数据
      >>> x = np.concatenate([np.random.normal(0, 1, 95), [10, -10]])
      >>> y = np.concatenate([np.random.normal(0, 1, 95), [12, -8]])
      >>> result = ct.kendall_tau(x, y)
      >>> print(f"Kendall τ: {result.statistic:.4f}")

卡方独立性检验
~~~~~~~~~~~~~~

.. py:method:: chi_square_independence(data: Union[np.ndarray, List[List]], alpha: float = 0.05) -> TestResult

   卡方独立性检验，检验分类变量间的独立性。

   **参数**:
   
   - **data**: 列联表数据（二维数组）
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 分类变量
   - 期望频数 ≥ 5（80%以上的单元格）
   - 样本量足够大

   **原假设**: 两个分类变量相互独立

   **应用场景**:
   
   - 分类变量关联性分析
   - 市场研究中的偏好关联
   - 医学研究中的风险因素分析

   **示例**::

      >>> # 2x2列联表：性别 vs 产品偏好
      >>> contingency_table = [[30, 20], [15, 35]]  # 行：性别，列：偏好
      >>> result = ct.chi_square_independence(contingency_table)
      >>> print(f"卡方统计量: {result.statistic:.4f}, 独立性: {not result.reject_null}")

Fisher精确检验
~~~~~~~~~~~~~~

.. py:method:: fisher_exact(data: Union[np.ndarray, List[List]], alternative: str = 'two-sided', alpha: float = 0.05) -> TestResult

   Fisher精确检验，用于2×2列联表的精确独立性检验。

   **参数**:
   
   - **data**: 2×2列联表
   - **alternative**: 备择假设类型（'two-sided', 'less', 'greater'）
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 2×2列联表
   - 小样本或期望频数过小时
   - 精确检验方法

   **原假设**: 两个二分类变量独立

   **应用场景**:
   
   - 小样本的2×2表分析
   - 临床试验的二分类结果
   - 期望频数不满足卡方检验条件时

   **示例**::

      >>> # 小样本的2x2表
      >>> table_2x2 = [[8, 2], [1, 5]]
      >>> result = ct.fisher_exact(table_2x2)
      >>> print(f"Fisher精确检验 p值: {result.p_value:.4f}")

McNemar检验
~~~~~~~~~~~

.. py:method:: mcnemar_test(data: Union[np.ndarray, List[List]], correction: bool = True, alpha: float = 0.05) -> TestResult

   McNemar检验，用于配对二分类数据的边际频率比较。

   **参数**:
   
   - **data**: 2×2列联表（配对数据）
   - **correction**: 是否使用连续性校正，默认True
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 配对二分类数据
   - 同一对象的前后测量
   - 关注边际概率变化

   **原假设**: 边际概率相等

   **应用场景**:
   
   - 治疗前后的二分类结果比较
   - 同一群体的态度变化分析
   - 配对设计的有效性评估

   **示例**::

      >>> # 治疗前后的成功/失败配对数据
      >>> # [治疗前成功&治疗后成功, 治疗前成功&治疗后失败]
      >>> # [治疗前失败&治疗后成功, 治疗前失败&治疗后失败]
      >>> paired_table = [[25, 5], [15, 10]]
      >>> result = ct.mcnemar_test(paired_table)
      >>> print(f"McNemar检验 p值: {result.p_value:.4f}")

使用建议
--------

方法选择指南
~~~~~~~~~~~~

1. **数据类型考虑**:

   - **连续数据**: Pearson相关（正态分布）、Spearman相关（非正态）
   - **序数数据**: Spearman相关、Kendall τ
   - **分类数据**: 卡方检验、Fisher精确检验

2. **分布假设**:

   - **满足正态性**: Pearson相关
   - **不满足正态性**: Spearman相关、Kendall τ
   - **无分布假设**: 非参数方法

3. **样本量考虑**:

   - **大样本**: 所有方法均适用
   - **小样本**: Kendall τ、Fisher精确检验
   - **期望频数小**: Fisher精确检验

4. **研究设计**:

   - **独立样本**: Pearson、Spearman、卡方
   - **配对样本**: McNemar检验

数据准备建议
~~~~~~~~~~~~

1. **数据清洗**:
   - 处理缺失值和异常值
   - 确认数据类型正确
   - 检查数据分布特性

2. **变量转换**:
   - 必要时进行标准化
   - 分类变量适当编码
   - 考虑非线性变换

3. **样本量评估**:
   - 计算所需样本量
   - 评估检验功效
   - 考虑效应量大小

相关系数解释
~~~~~~~~~~~~

1. **强度解释**:
   - r < 0.3: 弱相关
   - 0.3 ≤ r < 0.7: 中等相关
   - r ≥ 0.7: 强相关

2. **方向解释**:
   - r > 0: 正相关
   - r < 0: 负相关

3. **显著性解释**:
   - p < α: 相关显著
   - p ≥ α: 相关不显著

典型应用示例
------------

数据探索性分析
~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.statistics import CorrelationTests
   import numpy as np
   import pandas as pd
   
   # 模拟多变量数据集
   np.random.seed(42)
   n = 100
   
   # 生成相关的变量
   x1 = np.random.normal(0, 1, n)
   x2 = 0.8 * x1 + np.random.normal(0, 0.6, n)  # 与x1强相关
   x3 = np.random.uniform(0, 10, n)
   x4 = x3**0.5 + np.random.normal(0, 0.5, n)   # 与x3非线性相关
   
   ct = CorrelationTests()
   
   # 构建相关矩阵
   variables = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}
   var_names = list(variables.keys())
   
   print("相关性分析结果:")
   print("=" * 50)
   
   for i in range(len(var_names)):
       for j in range(i+1, len(var_names)):
           var1, var2 = var_names[i], var_names[j]
           
           # Pearson相关
           pearson_result = ct.pearson_correlation(variables[var1], variables[var2])
           
           # Spearman相关
           spearman_result = ct.spearman_correlation(variables[var1], variables[var2])
           
           print(f"\n{var1} vs {var2}:")
           print(f"  Pearson:  r={pearson_result.statistic:.3f}, p={pearson_result.p_value:.3f}")
           print(f"  Spearman: ρ={spearman_result.statistic:.3f}, p={spearman_result.p_value:.3f}")

分类变量关联分析
~~~~~~~~~~~~~~~~

.. code-block:: python

   # 模拟分类数据：教育水平 vs 收入水平
   # 教育水平：高中以下、高中、大学、研究生
   # 收入水平：低、中、高
   
   # 列联表数据 (行：教育水平，列：收入水平)
   education_income_table = [
       [40, 35, 15],   # 高中以下
       [25, 45, 30],   # 高中
       [15, 40, 45],   # 大学
       [5,  20, 25]    # 研究生
   ]
   
   ct = CorrelationTests()
   
   # 卡方独立性检验
   chi2_result = ct.chi_square_independence(education_income_table)
   
   print("教育水平与收入水平关联分析:")
   print(f"卡方统计量: {chi2_result.statistic:.4f}")
   print(f"p值: {chi2_result.p_value:.4f}")
   print(f"结论: {'教育水平与收入显著关联' if chi2_result.reject_null else '教育水平与收入无显著关联'}")
   
   # 计算Cramér's V系数（关联强度）
   n = np.sum(education_income_table)
   chi2_stat = chi2_result.statistic
   min_dim = min(len(education_income_table) - 1, len(education_income_table[0]) - 1)
   cramers_v = np.sqrt(chi2_stat / (n * min_dim))
   print(f"Cramér's V: {cramers_v:.3f} (关联强度)")

小样本精确检验
~~~~~~~~~~~~~~

.. code-block:: python

   # 临床试验小样本数据：治疗方法 vs 治疗结果
   # 2x2表：[有效&无效] x [新药&对照]
   
   treatment_result = [
       [12, 3],   # 新药：有效12例，无效3例
       [6, 9]     # 对照：有效6例，无效9例
   ]
   
   ct = CorrelationTests()
   
   # Fisher精确检验
   fisher_result = ct.fisher_exact(treatment_result)
   
   # 卡方检验（对比）
   chi2_result = ct.chi_square_independence(treatment_result)
   
   print("小样本临床试验分析:")
   print(f"Fisher精确检验 p值: {fisher_result.p_value:.4f}")
   print(f"卡方检验 p值: {chi2_result.p_value:.4f}")
   print(f"新药疗效: {'显著优于对照' if fisher_result.reject_null else '与对照无显著差异'}")
   
   # 计算效应量：优势比(Odds Ratio)
   a, b, c, d = treatment_result[0][0], treatment_result[0][1], treatment_result[1][0], treatment_result[1][1]
   odds_ratio = (a * d) / (b * c)
   print(f"优势比: {odds_ratio:.2f}")

配对数据的变化分析
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 治疗前后的改善情况（二分类）
   # McNemar检验示例
   
   # 生成配对数据：治疗前状态 vs 治疗后状态
   np.random.seed(42)
   n_patients = 100
   
   # 模拟治疗前后的改善/未改善状态
   # 假设治疗有一定效果
   pre_improved = np.random.binomial(1, 0.3, n_patients)  # 治疗前30%改善
   
   # 治疗后改善概率增加
   post_improved = np.zeros(n_patients)
   for i in range(n_patients):
       if pre_improved[i] == 1:
           post_improved[i] = np.random.binomial(1, 0.9)  # 已改善者90%维持
       else:
           post_improved[i] = np.random.binomial(1, 0.5)  # 未改善者50%改善
   
   # 构建McNemar表
   mcnemar_table = [
       [np.sum((pre_improved == 1) & (post_improved == 1)),   # 治疗前后都改善
        np.sum((pre_improved == 1) & (post_improved == 0))],  # 治疗前改善后退步
       [np.sum((pre_improved == 0) & (post_improved == 1)),   # 治疗前未改善后改善
        np.sum((pre_improved == 0) & (post_improved == 0))]   # 治疗前后都未改善
   ]
   
   ct = CorrelationTests()
   mcnemar_result = ct.mcnemar_test(mcnemar_table)
   
   print("治疗前后改善情况分析 (McNemar检验):")
   print("配对表:")
   print(f"  治疗前后都改善: {mcnemar_table[0][0]}")
   print(f"  治疗前改善后退步: {mcnemar_table[0][1]}")
   print(f"  治疗前未改善后改善: {mcnemar_table[1][0]}")
   print(f"  治疗前后都未改善: {mcnemar_table[1][1]}")
   print(f"\nMcNemar统计量: {mcnemar_result.statistic:.4f}")
   print(f"p值: {mcnemar_result.p_value:.4f}")
   print(f"结论: {'治疗后改善比例显著变化' if mcnemar_result.reject_null else '治疗前后改善比例无显著变化'}")

特征选择应用
~~~~~~~~~~~~

.. code-block:: python

   # 特征选择中的相关性分析
   from sklearn.datasets import make_regression
   
   # 生成回归数据集
   X, y = make_regression(n_samples=200, n_features=10, n_informative=5, 
                         random_state=42, noise=0.1)
   
   ct = CorrelationTests()
   
   print("特征与目标变量的相关性:")
   print("=" * 40)
   
   correlations = []
   for i in range(X.shape[1]):
       pearson_result = ct.pearson_correlation(X[:, i], y)
       correlations.append({
           'feature': f'Feature_{i}',
           'correlation': pearson_result.statistic,
           'p_value': pearson_result.p_value,
           'significant': pearson_result.reject_null
       })
   
   # 按相关性强度排序
   correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
   
   for corr in correlations:
       print(f"{corr['feature']}: r={corr['correlation']:.3f}, "
             f"p={corr['p_value']:.3f}, 显著={'是' if corr['significant'] else '否'}")
   
   # 选择显著相关的特征
   significant_features = [corr['feature'] for corr in correlations if corr['significant']]
   print(f"\n显著相关特征: {significant_features}")

注意事项
--------

1. **相关性vs因果性**:
   - 相关不意味因果
   - 考虑第三变量的影响
   - 结合领域知识解释

2. **多重检验校正**:
   - 多个变量对比较时
   - 使用Bonferroni或FDR校正
   - 控制整体错误率

3. **异常值影响**:
   - Pearson相关对异常值敏感
   - 考虑使用稳健方法
   - 必要时进行异常值处理

4. **样本量充足性**:
   - 确保有足够的统计功效
   - 小相关系数需要大样本
   - 进行功效分析 