基础类 (Base Classes)
=======================

统计检验模块的基础类定义了通用的数据结构和接口，为所有统计检验方法提供一致的返回格式和操作方式。

.. py:class:: TestResult

   所有统计检验方法的统一返回对象，包含检验的完整结果信息。

   **属性**:

   .. py:attribute:: test_name
      :type: str

      检验方法的名称

   .. py:attribute:: statistic
      :type: float

      检验统计量的值

   .. py:attribute:: p_value
      :type: float

      检验的p值

   .. py:attribute:: reject_null
      :type: bool

      是否拒绝原假设（True表示拒绝原假设）

   .. py:attribute:: critical_value
      :type: Optional[float]

      临界值（如果适用）

   .. py:attribute:: confidence_interval
      :type: Optional[Tuple[float, float]]

      置信区间（如果适用）

   .. py:attribute:: effect_size
      :type: Optional[float]

      效应量（如果适用）

   .. py:attribute:: additional_info
      :type: Dict

      额外的检验相关信息

   **示例**::

      >>> from pywayne.statistics import NormalityTests
      >>> import numpy as np
      >>> 
      >>> nt = NormalityTests()
      >>> data = np.random.normal(0, 1, 100)
      >>> result = nt.shapiro_wilk(data)
      >>> 
      >>> print(f"检验名称: {result.test_name}")
      >>> print(f"统计量: {result.statistic:.4f}")
      >>> print(f"p值: {result.p_value:.4f}")
      >>> print(f"拒绝原假设: {result.reject_null}")

.. py:class:: BaseStatisticalTest

   所有统计检验类的基类，提供通用的方法和属性。

   **方法**:

   .. py:method:: get_class_description() -> str

      返回检验类的详细描述信息。

      **返回**:
      
      - **str**: 包含类的功能描述、适用场景等信息

   .. py:method:: list_tests() -> List[str]

      返回该类中所有可用的检验方法名称。

      **返回**:
      
      - **List[str]**: 检验方法名称列表

   **示例**::

      >>> from pywayne.statistics import NormalityTests
      >>> 
      >>> nt = NormalityTests()
      >>> print(nt.get_class_description())
      >>> print("可用方法:", nt.list_tests())

测试结果解释指南
----------------

p值解释
~~~~~~~

p值是假设检验中最重要的统计量，表示在原假设为真的情况下，观察到当前结果或更极端结果的概率。

**解释原则**:

1. **显著性水平**: 通常设定α = 0.05
2. **判断规则**: 
   - p < α: 拒绝原假设（reject_null = True）
   - p ≥ α: 不能拒绝原假设（reject_null = False）

**常见显著性水平**:

- **0.05**: 标准显著性水平，5%的第一类错误率
- **0.01**: 严格显著性水平，1%的第一类错误率  
- **0.10**: 宽松显著性水平，10%的第一类错误率

**注意事项**:

- p值不是原假设为真的概率
- p值不表示效应量的大小
- 多重比较时需要校正

效应量
~~~~~~

效应量衡量差异或关联的实际大小，是对统计显著性的重要补充。

**常见效应量**:

1. **Cohen's d**: 标准化的均值差异
   - 小效应: d ≈ 0.2
   - 中等效应: d ≈ 0.5
   - 大效应: d ≈ 0.8

2. **相关系数**: 线性关联强度
   - 弱相关: r < 0.3
   - 中等相关: 0.3 ≤ r < 0.7
   - 强相关: r ≥ 0.7

3. **eta-squared (η²)**: 方差解释比例
   - 小效应: η² ≈ 0.01
   - 中等效应: η² ≈ 0.06
   - 大效应: η² ≈ 0.14

置信区间
~~~~~~~~

置信区间提供参数估计的不确定性范围，比点估计更具信息量。

**解释要点**:

1. **含义**: 在重复抽样中，该区间包含真实参数值的概率
2. **宽度**: 反映估计的精确性
3. **应用**: 参数估计和假设检验的结合

**示例**::

   >>> # 均值差异的95%置信区间
   >>> if result.confidence_interval:
   ...     lower, upper = result.confidence_interval
   ...     print(f"95%置信区间: [{lower:.3f}, {upper:.3f}]")
   ...     if 0 not in result.confidence_interval:
   ...         print("差异显著（置信区间不包含0）")

错误类型与功效
--------------

第一类错误（α错误）
~~~~~~~~~~~~~~~~~~~

错误地拒绝了真实的原假设。

**特点**:

- 概率由显著性水平α控制
- 也称为"假阳性"错误
- 通过设定较小的α值来控制

第二类错误（β错误）
~~~~~~~~~~~~~~~~~~~

错误地接受了错误的原假设。

**特点**:

- 概率记为β
- 也称为"假阴性"错误
- 与样本量、效应量、α水平相关

统计功效
~~~~~~~~

统计功效 = 1 - β，表示正确拒绝错误原假设的概率。

**影响因素**:

1. **样本量**: 样本越大，功效越高
2. **效应量**: 效应越大，功效越高
3. **显著性水平**: α越大，功效越高
4. **测量精度**: 误差越小，功效越高

**功效分析**::

   >>> import numpy as np
   >>> from scipy import stats
   >>> 
   >>> # 计算所需样本量
   >>> effect_size = 0.5  # Cohen's d
   >>> alpha = 0.05
   >>> power = 0.8
   >>> 
   >>> # 使用功效分析确定样本量
   >>> # （需要额外的统计包如statsmodels.stats.power）

多重比较校正
------------

当进行多次假设检验时，需要调整显著性水平以控制整体错误率。

Bonferroni校正
~~~~~~~~~~~~~~

最保守的校正方法，将显著性水平除以检验次数。

**公式**: α_corrected = α / m

**特点**:
- 严格控制家族错误率（FWER）
- 可能过于保守，降低统计功效

**示例**::

   >>> # 进行5次检验的Bonferroni校正
   >>> alpha = 0.05
   >>> num_tests = 5
   >>> corrected_alpha = alpha / num_tests
   >>> print(f"校正后显著性水平: {corrected_alpha:.3f}")

Benjamini-Hochberg校正
~~~~~~~~~~~~~~~~~~~~~~

控制错误发现率（FDR）的方法，相对不那么保守。

**步骤**:
1. 将p值从小到大排序
2. 计算每个p值的BH临界值
3. 从最大的p值开始，找到第一个小于临界值的p值

**示例**::

   >>> import numpy as np
   >>> from statsmodels.stats.multitest import multipletests
   >>> 
   >>> # 多个p值的FDR校正
   >>> p_values = [0.01, 0.05, 0.15, 0.03, 0.08]
   >>> reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
   ...     p_values, alpha=0.05, method='fdr_bh')
   >>> print("原始p值:", p_values)
   >>> print("校正后p值:", p_corrected)
   >>> print("拒绝原假设:", reject)

实际应用建议
------------

结果报告
~~~~~~~~

完整的统计检验结果应包含：

1. **描述性统计**: 样本量、均值、标准差等
2. **检验统计量**: 具体数值
3. **p值**: 准确到合适的小数位
4. **效应量**: 及其置信区间
5. **结论**: 基于统计和实际意义

**报告示例**::

   >>> def report_test_result(result, sample_stats=None):
   ...     print(f"统计检验结果报告")
   ...     print(f"=" * 30)
   ...     print(f"检验方法: {result.test_name}")
   ...     if sample_stats:
   ...         print(f"样本统计: {sample_stats}")
   ...     print(f"检验统计量: {result.statistic:.4f}")
   ...     print(f"p值: {result.p_value:.4f}")
   ...     if result.effect_size:
   ...         print(f"效应量: {result.effect_size:.4f}")
   ...     if result.confidence_interval:
   ...         print(f"95%置信区间: {result.confidence_interval}")
   ...     print(f"统计结论: {'拒绝原假设' if result.reject_null else '不能拒绝原假设'}")

假设检验的限制
~~~~~~~~~~~~~~

1. **假设依赖**: 结果的有效性依赖于假设的满足
2. **样本代表性**: 样本应代表感兴趣的总体
3. **测量误差**: 测量精度影响检验结果
4. **因果推断**: 相关不等于因果

最佳实践
~~~~~~~~

1. **事前计划**: 
   - 明确研究假设和检验方法
   - 进行功效分析确定样本量
   - 预设显著性水平

2. **数据探索**:
   - 检验前进行描述性分析
   - 验证假设检验的前提条件
   - 识别异常值和缺失值

3. **结果解释**:
   - 结合统计和实际意义
   - 考虑效应量和置信区间
   - 避免过度解释非显著结果

4. **重现性**:
   - 报告完整的统计信息
   - 提供数据和代码
   - 描述分析过程和决策

代码示例：完整分析流程
----------------------

.. code-block:: python

   from pywayne.statistics import NormalityTests, LocationTests
   import numpy as np
   
   def complete_analysis_example():
       """完整的统计分析示例"""
       
       # 1. 数据准备
       np.random.seed(42)
       group_a = np.random.normal(100, 15, 50)  # 对照组
       group_b = np.random.normal(110, 15, 50)  # 实验组
       
       print("统计分析完整流程示例")
       print("=" * 50)
       
       # 2. 描述性统计
       print("\n1. 描述性统计:")
       print(f"对照组: n={len(group_a)}, 均值={np.mean(group_a):.2f}, 标准差={np.std(group_a, ddof=1):.2f}")
       print(f"实验组: n={len(group_b)}, 均值={np.mean(group_b):.2f}, 标准差={np.std(group_b, ddof=1):.2f}")
       
       # 3. 假设检验前提条件检查
       print("\n2. 正态性检验:")
       nt = NormalityTests()
       
       norm_a = nt.shapiro_wilk(group_a)
       norm_b = nt.shapiro_wilk(group_b)
       
       print(f"对照组正态性: p={norm_a.p_value:.3f}, 正态={not norm_a.reject_null}")
       print(f"实验组正态性: p={norm_b.p_value:.3f}, 正态={not norm_b.reject_null}")
       
       # 4. 选择适当的检验方法
       lt = LocationTests()
       
       if not norm_a.reject_null and not norm_b.reject_null:
           # 数据正态，使用t检验
           result = lt.two_sample_ttest(group_a, group_b)
           test_type = "两样本t检验"
       else:
           # 数据非正态，使用非参数检验
           result = lt.mann_whitney_u(group_a, group_b)
           test_type = "Mann-Whitney U检验"
       
       # 5. 结果报告
       print(f"\n3. {test_type}结果:")
       print(f"统计量: {result.statistic:.4f}")
       print(f"p值: {result.p_value:.4f}")
       print(f"显著性: {'是' if result.reject_null else '否'}")
       
       # 6. 效应量计算
       if test_type == "两样本t检验":
           # Cohen's d
           pooled_std = np.sqrt(((len(group_a)-1)*np.var(group_a, ddof=1) + 
                               (len(group_b)-1)*np.var(group_b, ddof=1)) / 
                               (len(group_a)+len(group_b)-2))
           cohens_d = (np.mean(group_b) - np.mean(group_a)) / pooled_std
           print(f"Cohen's d: {cohens_d:.3f}")
       
       # 7. 实际意义解释
       print(f"\n4. 结论:")
       if result.reject_null:
           print("统计学意义: 两组存在显著差异")
           print("实际意义: 需要结合专业知识判断差异的实际重要性")
       else:
           print("统计学意义: 两组无显著差异")
           print("注意: 这不等于两组完全相同，可能是样本量不足或效应量较小")
   
   # 运行示例
   if __name__ == "__main__":
       complete_analysis_example()

该示例展示了从数据准备到结果解释的完整统计分析流程，体现了统计检验模块中基础类的设计理念和实际应用方式。 