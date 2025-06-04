统计检验 (Statistics)
========================

统计检验模块为数据分析提供了全面的统计检验方法集合，涵盖正态性检验、位置参数检验、相关性检验、时间序列检验和模型诊断等五大类别。本模块集成了37种常用的统计检验方法，为科学研究、商业分析和数据科学项目提供强大的统计推断工具。

模块概述
--------

统计检验模块包含以下五个主要类别：

1. **正态性检验** (`NormalityTests`): 检验数据是否符合正态分布
2. **位置参数检验** (`LocationTests`): 比较样本均值或中位数的差异
3. **相关性检验** (`CorrelationTests`): 分析变量间的相关关系和独立性
4. **时间序列检验** (`TimeSeriesTests`): 时间序列的平稳性、相关性等特性检验
5. **模型诊断** (`ModelDiagnostics`): 回归模型的假设检验和诊断

快速开始
--------

以下是一个简单的示例，展示如何使用统计检验模块：

.. code-block:: python

   from pywayne.statistics import NormalityTests, LocationTests
   import numpy as np
   
   # 创建检验对象
   nt = NormalityTests()
   lt = LocationTests()
   
   # 生成测试数据
   data1 = np.random.normal(100, 15, 50)
   data2 = np.random.normal(105, 15, 50)
   
   # 正态性检验
   result1 = nt.shapiro_wilk(data1)
   print(f"正态性检验: p值={result1.p_value:.4f}, 正态分布={not result1.reject_null}")
   
   # 两样本t检验
   result2 = lt.two_sample_ttest(data1, data2)
   print(f"t检验: p值={result2.p_value:.4f}, 差异显著={result2.reject_null}")

统计检验方法分类
----------------

正态性检验 (NormalityTests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

正态性检验用于检验数据是否服从正态分布，这是许多参数统计方法的重要前提。

**可用方法:**

.. list-table:: 正态性检验方法
   :header-rows: 1
   :widths: 20 40 40

   * - 方法名
     - 描述
     - 适用场景
   * - shapiro_wilk
     - Shapiro-Wilk检验
     - 小中样本(≤5000)，功效较高
   * - ks_test_normal
     - K-S正态性检验  
     - 中大样本，检验连续分布
   * - ks_test_two_sample
     - K-S双样本检验
     - 比较两样本分布差异
   * - anderson_darling
     - Anderson-Darling检验
     - 对尾部更敏感的正态性检验
   * - dagostino_pearson
     - D'Agostino-Pearson检验
     - 基于偏度和峰度的正态性检验
   * - jarque_bera
     - Jarque-Bera检验
     - 回归残差的正态性检验
   * - chi_square_goodness_of_fit
     - 卡方拟合优度检验
     - 分类数据的分布检验
   * - lilliefors_test
     - Lilliefors检验
     - 参数未知时的K-S检验

位置参数检验 (LocationTests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

位置参数检验用于比较不同组间的均值、中位数等位置参数的差异。

**可用方法:**

.. list-table:: 位置参数检验方法
   :header-rows: 1
   :widths: 20 40 40

   * - 方法名
     - 描述
     - 适用场景
   * - one_sample_ttest
     - 单样本t检验
     - 检验样本均值与给定值的差异
   * - two_sample_ttest
     - 两样本t检验
     - 比较两独立样本均值差异
   * - paired_ttest
     - 配对t检验
     - 比较配对样本的均值差异
   * - one_way_anova
     - 单因素方差分析
     - 比较多组均值差异
   * - mann_whitney_u
     - Mann-Whitney U检验
     - 非参数两样本位置检验
   * - wilcoxon_signed_rank
     - Wilcoxon符号秩检验
     - 非参数配对样本检验
   * - kruskal_wallis
     - Kruskal-Wallis检验
     - 非参数多组比较

相关性检验 (CorrelationTests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

相关性检验用于分析变量间的线性或非线性相关关系，以及分类变量的独立性。

**可用方法:**

.. list-table:: 相关性检验方法
   :header-rows: 1
   :widths: 20 40 40

   * - 方法名
     - 描述
     - 适用场景
   * - pearson_correlation
     - Pearson相关检验
     - 线性相关关系检验
   * - spearman_correlation
     - Spearman秩相关检验
     - 单调相关关系检验
   * - kendall_tau
     - Kendall τ相关检验
     - 秩相关检验(小样本)
   * - chi_square_independence
     - 卡方独立性检验
     - 分类变量独立性检验
   * - fisher_exact_test
     - Fisher精确检验
     - 2×2表的精确独立性检验
   * - mcnemar_test
     - McNemar检验
     - 配对分类数据的边际同质性检验

时间序列检验 (TimeSeriesTests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

时间序列检验用于分析时间序列数据的统计特性，包括平稳性、自相关、协整等。

**可用方法:**

.. list-table:: 时间序列检验方法
   :header-rows: 1
   :widths: 20 40 40

   * - 方法名
     - 描述
     - 适用场景
   * - adf_test
     - ADF单位根检验
     - 检验时间序列平稳性
   * - kpss_test
     - KPSS平稳性检验
     - 平稳性检验(与ADF互补)
   * - ljung_box_test
     - Ljung-Box检验
     - 检验序列自相关性
   * - runs_test
     - 游程检验
     - 检验序列随机性
   * - arch_test
     - ARCH效应检验
     - 检验条件异方差
   * - granger_causality
     - Granger因果检验
     - 检验序列间因果关系
   * - engle_granger_cointegration
     - Engle-Granger协整检验
     - 检验序列间长期均衡关系
   * - breusch_godfrey_test
     - Breusch-Godfrey检验
     - 检验回归残差自相关

模型诊断 (ModelDiagnostics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

模型诊断提供回归模型假设检验和诊断工具，帮助验证模型的有效性。

**可用方法:**

.. list-table:: 模型诊断方法
   :header-rows: 1
   :widths: 20 40 40

   * - 方法名
     - 描述
     - 适用场景
   * - breusch_pagan_test
     - Breusch-Pagan检验
     - 检验回归残差异方差性
   * - white_test
     - White检验
     - 通用异方差检验
   * - goldfeld_quandt_test
     - Goldfeld-Quandt检验
     - 结构断点异方差检验
   * - durbin_watson_test
     - Durbin-Watson检验
     - 检验一阶自相关
   * - variance_inflation_factor
     - 方差膨胀因子(VIF)
     - 诊断多重共线性
   * - levene_test
     - Levene检验
     - 检验方差齐性
   * - bartlett_test
     - Bartlett检验
     - 检验方差齐性(正态假设)
   * - residual_normality_test
     - 残差正态性检验
     - 检验回归残差正态性

高级使用指南
------------

批量检验处理
~~~~~~~~~~~~

对于大量数据的批量检验，建议使用以下模式：

.. code-block:: python

   from pywayne.statistics import NormalityTests
   import numpy as np
   
   nt = NormalityTests()
   
   # 批量处理多个数据集
   datasets = [np.random.normal(0, 1, 100) for _ in range(10)]
   results = []
   
   for i, data in enumerate(datasets):
       result = nt.shapiro_wilk(data)
       results.append({
           'dataset_id': i,
           'test_name': result.test_name,
           'p_value': result.p_value,
           'is_normal': not result.reject_null
       })
   
   # 多重比较校正
   from statsmodels.stats.multitest import multipletests
   p_values = [r['p_value'] for r in results]
   rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
   
   for result, corrected_p, is_rejected in zip(results, p_corrected, rejected):
       result['p_corrected'] = corrected_p
       result['significant_after_correction'] = is_rejected

结果对象使用
~~~~~~~~~~~~

所有检验方法返回统一的TestResult对象，提供一致的接口：

.. code-block:: python

   from pywayne.statistics import LocationTests
   import numpy as np
   
   lt = LocationTests()
   data1 = np.random.normal(100, 15, 50)
   data2 = np.random.normal(105, 15, 50)
   
   result = lt.two_sample_ttest(data1, data2)
   
   # 访问基本结果
   print(f"检验名称: {result.test_name}")
   print(f"统计量: {result.statistic:.4f}")
   print(f"p值: {result.p_value:.4f}")
   print(f"拒绝原假设: {result.reject_null}")
   
   # 访问额外信息(如果可用)
   if result.effect_size:
       print(f"效应量: {result.effect_size:.4f}")
   
   if result.confidence_interval:
       ci_lower, ci_upper = result.confidence_interval
       print(f"95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
   
   if result.additional_info:
       print("额外信息:", result.additional_info)

常见应用场景
------------

数据质量检查
~~~~~~~~~~~~

在数据分析前，使用统计检验评估数据质量：

.. code-block:: python

   def data_quality_check(data):
       """数据质量检查流程"""
       nt = NormalityTests()
       
       # 正态性检验
       normality = nt.shapiro_wilk(data)
       
       # 异常值检测(使用IQR方法)
       Q1 = np.percentile(data, 25)
       Q3 = np.percentile(data, 75)
       IQR = Q3 - Q1
       outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
       
       return {
           'size': len(data),
           'normal_distribution': not normality.reject_null,
           'normality_p_value': normality.p_value,
           'outlier_count': len(outliers),
           'outlier_rate': len(outliers) / len(data)
       }

时间序列建模准备
~~~~~~~~~~~~~~~~

在时间序列建模前的统计检验：

.. code-block:: python

   def prepare_time_series(ts_data):
       """时间序列建模前的统计检验"""
       tst = TimeSeriesTests()
       
       # 平稳性检验
       adf_result = tst.adf_test(ts_data)
       kpss_result = tst.kpss_test(ts_data)
       
       # 自相关检验
       ljung_box_result = tst.ljung_box_test(ts_data, lags=10)
       
       return {
           'stationary_adf': adf_result.reject_null,
           'stationary_kpss': not kpss_result.reject_null,
           'has_autocorrelation': ljung_box_result.reject_null,
           'need_differencing': not adf_result.reject_null
       }

回归模型诊断
~~~~~~~~~~~~

回归分析后的模型诊断：

.. code-block:: python

   def diagnose_regression_model(y, X, fitted_model):
       """回归模型完整诊断"""
       md = ModelDiagnostics()
       residuals = y - fitted_model.predict(X)
       
       # 异方差检验
       bp_test = md.breusch_pagan_test(residuals, X)
       white_test = md.white_test(residuals, X)
       
       # 自相关检验
       dw_test = md.durbin_watson_test(residuals)
       
       # 正态性检验
       norm_test = md.residual_normality_test(residuals)
       
       # 多重共线性检验
       vif_scores = md.variance_inflation_factor(X)
       
       return {
           'heteroscedasticity_bp': bp_test.reject_null,
           'heteroscedasticity_white': white_test.reject_null,
           'autocorrelation_dw': dw_test.statistic,
           'residuals_normal': not norm_test.reject_null,
           'multicollinearity_max_vif': max(vif_scores),
           'model_assumptions_satisfied': all([
               not bp_test.reject_null,
               not white_test.reject_null,
               1.5 <= dw_test.statistic <= 2.5,
               not norm_test.reject_null,
               max(vif_scores) < 10
           ])
       }

性能优化建议
------------

1. **数据预处理**: 检验前移除缺失值和异常值
2. **样本量考虑**: 小样本优选精确检验，大样本可用渐近检验
3. **检验选择**: 根据数据分布和假设选择合适的检验方法
4. **并行处理**: 对于大批量检验，考虑使用多进程处理

注意事项与最佳实践
------------------

1. **假设验证**: 使用前确认检验方法的假设条件
2. **多重比较**: 进行多次检验时务必进行p值校正
3. **效应量**: 不仅关注统计显著性，还要考虑实际意义
4. **样本量**: 进行功效分析，确保有足够的统计功效
5. **结果解释**: 正确理解p值含义，避免过度解释

.. toctree::
   :maxdepth: 2
   :caption: 详细文档:

   statistics/normality_tests
   statistics/location_tests
   statistics/correlation_tests
   statistics/time_series_tests
   statistics/model_diagnostics
   statistics/base_classes
   statistics/examples

模块扩展建议
------------

本模块为进一步扩展预留了充分的空间，可以考虑添加：

1. **贝叶斯统计检验**: 贝叶斯因子、后验预测检验等
2. **生存分析**: Kaplan-Meier估计、Cox回归诊断等  
3. **非参数检验**: 更多的非参数检验方法
4. **bootstrap方法**: 自助法置信区间和检验
5. **稳健统计**: 对异常值稳健的检验方法

这样的设计使得模块既满足当前需求，又具备良好的可扩展性。 