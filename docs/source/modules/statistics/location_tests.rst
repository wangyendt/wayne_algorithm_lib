位置参数检验 (LocationTests)
=============================

位置参数检验类提供了7种方法来比较不同组别或条件下的中心趋势（均值、中位数等）。这些检验在A/B测试、临床试验、质量控制等领域有广泛应用。

.. py:class:: LocationTests

   位置参数检验类，包含参数和非参数方法来比较组间中心趋势。所有方法都返回 ``TestResult`` 对象。

   **主要方法**:

检验方法详解
------------

单样本t检验
~~~~~~~~~~~

.. py:method:: one_sample_ttest(data: Union[np.ndarray, List], popmean: float = 0, alpha: float = 0.05) -> TestResult

   单样本t检验，检验样本均值是否等于给定的总体均值。

   **参数**:
   
   - **data**: 输入数据，一维数组或列表
   - **popmean**: 假设的总体均值，默认为0
   - **alpha**: 显著性水平，默认0.05

   **适用条件**:
   
   - 数据近似正态分布（大样本时可放宽）
   - 样本独立抽取
   - 连续型数据

   **原假设**: 样本均值等于给定的总体均值

   **应用场景**:
   
   - 质量控制：检验产品规格是否达标
   - 实验验证：检验处理效果是否为零
   - 基准比较：与历史标准或目标值比较

   **示例**::

      >>> from pywayne.statistics import LocationTests
      >>> import numpy as np
      >>> 
      >>> lt = LocationTests()
      >>> data = np.random.normal(0.5, 1, 100)  # 均值偏移的数据
      >>> result = lt.one_sample_ttest(data, popmean=0)
      >>> print(f"t统计量: {result.statistic:.4f}, p值: {result.p_value:.4f}")

两样本t检验
~~~~~~~~~~~

.. py:method:: two_sample_ttest(data1: Union[np.ndarray, List], data2: Union[np.ndarray, List], equal_var: bool = True, alpha: float = 0.05) -> TestResult

   两独立样本t检验，比较两个独立组的均值差异。

   **参数**:
   
   - **data1**: 第一个样本
   - **data2**: 第二个样本
   - **equal_var**: 是否假设方差相等，默认True
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 两组数据独立
   - 数据近似正态分布
   - equal_var=True时假设方差齐性

   **原假设**: 两组均值相等

   **应用场景**:
   
   - A/B测试：比较两种方案的效果
   - 临床试验：比较治疗组与对照组
   - 质量比较：比较不同供应商的产品

   **示例**::

      >>> # 比较两组数据
      >>> group_a = np.random.normal(10, 2, 50)
      >>> group_b = np.random.normal(12, 2, 50)
      >>> result = lt.two_sample_ttest(group_a, group_b)
      >>> print(f"两组均值差异显著: {result.reject_null}")

配对t检验
~~~~~~~~~

.. py:method:: paired_ttest(data1: Union[np.ndarray, List], data2: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   配对样本t检验，比较同一对象在两种条件下的差异。

   **参数**:
   
   - **data1**: 第一次测量结果
   - **data2**: 第二次测量结果
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 配对数据（同一对象的前后测量）
   - 差值近似正态分布
   - 测量值连续

   **原假设**: 配对差值的均值为零

   **应用场景**:
   
   - 治疗前后效果比较
   - 训练前后能力测试
   - 产品改进前后性能对比

   **示例**::

      >>> # 治疗前后数据
      >>> before = np.random.normal(100, 10, 30)
      >>> after = before + np.random.normal(5, 5, 30)  # 有改进效果
      >>> result = lt.paired_ttest(before, after)
      >>> print(f"治疗有效: {result.reject_null}")

单因素方差分析
~~~~~~~~~~~~~~

.. py:method:: one_way_anova(*groups: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   单因素方差分析，比较三个或更多独立组的均值。

   **参数**:
   
   - **groups**: 多个独立样本组
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 各组独立
   - 组内数据近似正态分布
   - 方差齐性（可用Levene检验验证）

   **原假设**: 所有组的均值相等

   **应用场景**:
   
   - 多组实验比较
   - 不同处理方式效果对比
   - 多因素实验的主效应分析

   **示例**::

      >>> # 三组数据比较
      >>> group1 = np.random.normal(10, 2, 30)
      >>> group2 = np.random.normal(12, 2, 30)
      >>> group3 = np.random.normal(11, 2, 30)
      >>> result = lt.one_way_anova(group1, group2, group3)
      >>> print(f"F统计量: {result.statistic:.4f}, 组间有差异: {result.reject_null}")

Mann-Whitney U检验
~~~~~~~~~~~~~~~~~~

.. py:method:: mann_whitney_u(x: Union[np.ndarray, List], y: Union[np.ndarray, List], alpha: float = 0.05, alternative: str = 'two-sided') -> TestResult

   Mann-Whitney U检验（Wilcoxon秩和检验），比较两独立样本的分布位置。

   **参数**:
   
   - **x**: 第一个样本
   - **y**: 第二个样本
   - **alpha**: 显著性水平
   - **alternative**: 备择假设类型（'two-sided', 'less', 'greater'）

   **适用条件**:
   
   - 两组独立
   - 数据至少为序数水平
   - 无需正态性假设

   **原假设**: 两组数据来自相同分布

   **应用场景**:
   
   - 非正态数据的组间比较
   - 小样本比较
   - 序数数据分析

   **示例**::

      >>> # 非正态数据比较
      >>> group1 = np.random.exponential(2, 50)
      >>> group2 = np.random.exponential(3, 50)
      >>> result = lt.mann_whitney_u(group1, group2)
      >>> print(f"U统计量: {result.statistic:.4f}, 分布不同: {result.reject_null}")

Wilcoxon符号秩检验
~~~~~~~~~~~~~~~~~~

.. py:method:: wilcoxon_signed_rank(x: Union[np.ndarray, List], y: Union[np.ndarray, List] = None, alpha: float = 0.05) -> TestResult

   Wilcoxon符号秩检验，配对样本的非参数检验。

   **参数**:
   
   - **x**: 第一个样本或差值数据
   - **y**: 第二个样本（可选）
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 配对数据或单样本检验
   - 差值分布对称
   - 无需正态性假设

   **原假设**: 中位数差值为零

   **应用场景**:
   
   - 配对数据的非参数比较
   - 对称分布的单样本检验
   - 小样本配对分析

Kruskal-Wallis检验
~~~~~~~~~~~~~~~~~~

.. py:method:: kruskal_wallis(*groups: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult

   Kruskal-Wallis检验，多组独立样本的非参数检验。

   **参数**:
   
   - **groups**: 多个独立样本组
   - **alpha**: 显著性水平

   **适用条件**:
   
   - 各组独立
   - 数据至少为序数水平
   - 无需正态性和方差齐性假设

   **原假设**: 所有组的分布相同

   **应用场景**:
   
   - 非正态数据的多组比较
   - 序数数据分析
   - ANOVA的非参数替代

   **示例**::

      >>> # 多组非正态数据比较
      >>> group1 = np.random.exponential(1, 30)
      >>> group2 = np.random.exponential(1.5, 30)
      >>> group3 = np.random.exponential(2, 30)
      >>> result = lt.kruskal_wallis(group1, group2, group3)
      >>> print(f"H统计量: {result.statistic:.4f}, 组间有差异: {result.reject_null}")

使用建议
--------

方法选择指南
~~~~~~~~~~~~

1. **数据类型考虑**:

   - **正态数据**: t检验、ANOVA
   - **非正态数据**: Mann-Whitney U、Kruskal-Wallis
   - **序数数据**: 非参数方法

2. **样本数量**:

   - **两组比较**: t检验或Mann-Whitney U
   - **多组比较**: ANOVA或Kruskal-Wallis

3. **数据关系**:

   - **独立样本**: 两样本t检验、ANOVA
   - **配对样本**: 配对t检验、Wilcoxon符号秩

4. **样本量考虑**:

   - **小样本 (n < 30)**: 优先非参数方法
   - **大样本**: 参数方法更有效

数据准备建议
~~~~~~~~~~~~

1. **数据清洗**:
   - 识别和处理异常值
   - 检查数据完整性
   - 确保数据类型正确

2. **假设检验**:
   - 正态性检验（Shapiro-Wilk等）
   - 方差齐性检验（Levene检验）
   - 独立性验证

3. **样本量评估**:
   - 进行功效分析
   - 确保足够的检验功效
   - 考虑效应量大小

结果解释指南
~~~~~~~~~~~~

1. **统计显著性**:
   - p < α: 拒绝原假设
   - p ≥ α: 不能拒绝原假设

2. **效应量**:
   - Cohen's d: 标准化效应量
   - eta-squared: 方差解释比例
   - 实际意义评估

3. **置信区间**:
   - 均值差异的置信区间
   - 效应量的置信区间
   - 不确定性评估

典型应用示例
------------

A/B测试分析
~~~~~~~~~~~

.. code-block:: python

   from pywayne.statistics import LocationTests, NormalityTests
   import numpy as np
   
   # 模拟A/B测试数据
   control_group = np.random.normal(2.5, 0.8, 1000)    # 对照组转化率
   treatment_group = np.random.normal(2.8, 0.8, 1000)  # 实验组转化率
   
   lt = LocationTests()
   nt = NormalityTests()
   
   # 检查数据分布
   norm_control = nt.shapiro_wilk(control_group[:50])  # 抽样检验
   norm_treatment = nt.shapiro_wilk(treatment_group[:50])
   
   # 选择适当的检验方法
   if norm_control.p_value > 0.05 and norm_treatment.p_value > 0.05:
       # 数据正态，使用t检验
       result = lt.two_sample_ttest(control_group, treatment_group)
       test_type = "两样本t检验"
   else:
       # 数据非正态，使用非参数检验
       result = lt.mann_whitney_u(control_group, treatment_group)
       test_type = "Mann-Whitney U检验"
   
   print(f"A/B测试结果 ({test_type}):")
   print(f"统计量: {result.statistic:.4f}")
   print(f"p值: {result.p_value:.4f}")
   print(f"结论: {'实验组效果显著' if result.reject_null else '无显著差异'}")

临床试验分析
~~~~~~~~~~~~

.. code-block:: python

   # 模拟治疗前后数据
   np.random.seed(42)
   n_patients = 50
   
   # 治疗前的症状评分
   before_treatment = np.random.normal(70, 15, n_patients)
   
   # 治疗后的症状评分（假设有改善）
   treatment_effect = np.random.normal(-10, 8, n_patients)
   after_treatment = before_treatment + treatment_effect
   
   lt = LocationTests()
   
   # 配对t检验
   result = lt.paired_ttest(before_treatment, after_treatment)
   
   print("临床试验结果:")
   print(f"治疗前均值: {np.mean(before_treatment):.2f}")
   print(f"治疗后均值: {np.mean(after_treatment):.2f}")
   print(f"均值差异: {np.mean(after_treatment - before_treatment):.2f}")
   print(f"t统计量: {result.statistic:.4f}")
   print(f"p值: {result.p_value:.4f}")
   print(f"治疗效果: {'显著' if result.reject_null else '不显著'}")

多组质量比较
~~~~~~~~~~~~

.. code-block:: python

   # 模拟不同供应商的产品质量数据
   supplier_a = np.random.normal(95, 5, 40)
   supplier_b = np.random.normal(92, 6, 40)
   supplier_c = np.random.normal(98, 4, 40)
   supplier_d = np.random.normal(94, 7, 40)
   
   lt = LocationTests()
   
   # 单因素方差分析
   anova_result = lt.one_way_anova(supplier_a, supplier_b, supplier_c, supplier_d)
   
   print("供应商质量比较 (ANOVA):")
   print(f"F统计量: {anova_result.statistic:.4f}")
   print(f"p值: {anova_result.p_value:.4f}")
   print(f"结论: {'供应商间存在显著差异' if anova_result.reject_null else '供应商间无显著差异'}")
   
   # 如果ANOVA显著，可进行事后比较
   if anova_result.reject_null:
       print("\n事后比较 (两两比较):")
       suppliers = [supplier_a, supplier_b, supplier_c, supplier_d]
       names = ['A', 'B', 'C', 'D']
       
       for i in range(len(suppliers)):
           for j in range(i+1, len(suppliers)):
               result = lt.two_sample_ttest(suppliers[i], suppliers[j])
               print(f"供应商{names[i]} vs {names[j]}: p={result.p_value:.4f}, "
                     f"显著={'是' if result.reject_null else '否'}")

非参数多组比较
~~~~~~~~~~~~~~

.. code-block:: python

   # 模拟非正态分布的数据（如反应时间）
   group1 = np.random.exponential(2, 30)
   group2 = np.random.exponential(2.5, 30)
   group3 = np.random.exponential(3, 30)
   
   lt = LocationTests()
   
   # Kruskal-Wallis检验
   kw_result = lt.kruskal_wallis(group1, group2, group3)
   
   print("非参数多组比较 (Kruskal-Wallis):")
   print(f"H统计量: {kw_result.statistic:.4f}")
   print(f"p值: {kw_result.p_value:.4f}")
   print(f"结论: {'组间分布有显著差异' if kw_result.reject_null else '组间分布无显著差异'}")
   
   # 计算各组中位数
   medians = [np.median(group1), np.median(group2), np.median(group3)]
   print(f"各组中位数: {medians}")

注意事项
--------

1. **假设验证**:
   - 参数检验需要验证正态性假设
   - 方差分析需要检验方差齐性
   - 独立性假设通常由实验设计保证

2. **多重比较**:
   - 多次比较时考虑α水平校正
   - 使用Bonferroni校正或FDR控制
   - 计划比较 vs 事后比较

3. **效应量报告**:
   - 不仅报告统计显著性
   - 计算和报告效应量
   - 提供置信区间

4. **实际应用考虑**:
   - 大样本时微小差异也可能显著
   - 结合专业知识解释结果
   - 考虑统计功效和检验效能 