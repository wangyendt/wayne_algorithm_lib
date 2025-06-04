应用示例 (Examples)
===================

本文档提供了统计检验模块在实际场景中的综合应用示例，涵盖数据科学、商业分析、科学研究等多个领域。

.. contents:: 目录
   :local:
   :depth: 2

数据科学场景
------------

机器学习模型评估
~~~~~~~~~~~~~~~~

在机器学习项目中，统计检验可以帮助评估模型性能的显著性差异。

.. code-block:: python

   from pywayne.statistics import LocationTests, NormalityTests
   import numpy as np
   from sklearn.model_selection import cross_val_score
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.svm import SVC
   from sklearn.datasets import make_classification
   
   def model_comparison_analysis():
       """比较不同机器学习模型的性能"""
       
       # 生成数据集
       X, y = make_classification(n_samples=1000, n_features=20, 
                                 n_informative=10, random_state=42)
       
       # 定义模型
       models = {
           'Random Forest': RandomForestClassifier(random_state=42),
           'SVM': SVC(random_state=42)
       }
       
       # 交叉验证评估
       cv_scores = {}
       for name, model in models.items():
           scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
           cv_scores[name] = scores
           print(f"{name}: 均值={np.mean(scores):.4f}, 标准差={np.std(scores):.4f}")
       
       # 统计检验比较
       lt = LocationTests()
       nt = NormalityTests()
       
       rf_scores = cv_scores['Random Forest']
       svm_scores = cv_scores['SVM']
       
       # 检验正态性
       rf_norm = nt.shapiro_wilk(rf_scores)
       svm_norm = nt.shapiro_wilk(svm_scores)
       
       print(f"\n正态性检验:")
       print(f"Random Forest: p={rf_norm.p_value:.4f}, 正态={not rf_norm.reject_null}")
       print(f"SVM: p={svm_norm.p_value:.4f}, 正态={not svm_norm.reject_null}")
       
       # 配对t检验（交叉验证是配对的）
       paired_result = lt.paired_ttest(rf_scores, svm_scores)
       
       print(f"\n配对t检验结果:")
       print(f"t统计量: {paired_result.statistic:.4f}")
       print(f"p值: {paired_result.p_value:.4f}")
       print(f"模型性能差异显著: {paired_result.reject_null}")
       
       # 效应量计算
       diff_mean = np.mean(rf_scores - svm_scores)
       diff_std = np.std(rf_scores - svm_scores, ddof=1)
       cohens_d = diff_mean / diff_std
       
       print(f"效应量 (Cohen's d): {cohens_d:.4f}")
       
       return paired_result

A/B测试分析
~~~~~~~~~~~

在产品优化和营销活动中，A/B测试是常用的实验方法。

.. code-block:: python

   def ab_test_analysis():
       """A/B测试完整分析流程"""
       
       # 模拟A/B测试数据
       np.random.seed(42)
       
       # 控制组和实验组的转化率数据
       control_conversions = np.random.binomial(1, 0.12, 5000)  # 12%转化率
       treatment_conversions = np.random.binomial(1, 0.14, 5000)  # 14%转化率
       
       # 转化率统计
       control_rate = np.mean(control_conversions)
       treatment_rate = np.mean(treatment_conversions)
       
       print("A/B测试分析报告")
       print("=" * 40)
       print(f"控制组: n={len(control_conversions)}, 转化率={control_rate:.4f}")
       print(f"实验组: n={len(treatment_conversions)}, 转化率={treatment_rate:.4f}")
       print(f"提升幅度: {(treatment_rate - control_rate)/control_rate*100:.2f}%")
       
       # 统计检验
       from scipy import stats
       
       # 双比例Z检验
       count = np.array([np.sum(treatment_conversions), np.sum(control_conversions)])
       nobs = np.array([len(treatment_conversions), len(control_conversions)])
       
       from statsmodels.stats.proportion import proportions_ztest
       z_stat, p_value = proportions_ztest(count, nobs)
       
       print(f"\n双比例Z检验:")
       print(f"Z统计量: {z_stat:.4f}")
       print(f"p值: {p_value:.4f}")
       print(f"差异显著: {p_value < 0.05}")
       
       # 置信区间
       from statsmodels.stats.proportion import proportion_confint
       ci_control = proportion_confint(np.sum(control_conversions), 
                                     len(control_conversions), alpha=0.05)
       ci_treatment = proportion_confint(np.sum(treatment_conversions), 
                                       len(treatment_conversions), alpha=0.05)
       
       print(f"\n95%置信区间:")
       print(f"控制组: [{ci_control[0]:.4f}, {ci_control[1]:.4f}]")
       print(f"实验组: [{ci_treatment[0]:.4f}, {ci_treatment[1]:.4f}]")

商业分析场景
------------

销售数据分析
~~~~~~~~~~~~

分析不同销售策略、区域或时间段的销售表现差异。

.. code-block:: python

   def sales_analysis():
       """销售数据统计分析"""
       
       # 模拟销售数据
       np.random.seed(42)
       
       # 三个销售区域的月销售额（万元）
       region_a = np.random.normal(150, 30, 24)  # 24个月
       region_b = np.random.normal(180, 35, 24)
       region_c = np.random.normal(165, 25, 24)
       
       lt = LocationTests()
       nt = NormalityTests()
       
       print("销售区域分析报告")
       print("=" * 40)
       
       # 描述性统计
       regions = {'A区': region_a, 'B区': region_b, 'C区': region_c}
       for name, data in regions.items():
           print(f"{name}: 均值={np.mean(data):.2f}万, 标准差={np.std(data, ddof=1):.2f}万")
       
       # 正态性检验
       print(f"\n正态性检验:")
       for name, data in regions.items():
           norm_result = nt.shapiro_wilk(data)
           print(f"{name}: p={norm_result.p_value:.4f}, 正态={not norm_result.reject_null}")
       
       # 方差齐性检验
       from scipy.stats import levene
       levene_stat, levene_p = levene(region_a, region_b, region_c)
       print(f"\nLevene方差齐性检验: p={levene_p:.4f}, 方差齐={levene_p >= 0.05}")
       
       # 单因素方差分析
       anova_result = lt.one_way_anova(region_a, region_b, region_c)
       print(f"\n单因素ANOVA:")
       print(f"F统计量: {anova_result.statistic:.4f}")
       print(f"p值: {anova_result.p_value:.4f}")
       print(f"区域间差异显著: {anova_result.reject_null}")
       
       # 两两比较（如果ANOVA显著）
       if anova_result.reject_null:
           print(f"\n两两比较(Bonferroni校正):")
           region_list = [region_a, region_b, region_c]
           region_names = ['A区', 'B区', 'C区']
           
           alpha_corrected = 0.05 / 3  # Bonferroni校正
           
           for i in range(len(region_list)):
               for j in range(i+1, len(region_list)):
                   result = lt.two_sample_ttest(region_list[i], region_list[j])
                   significant = result.p_value < alpha_corrected
                   print(f"{region_names[i]} vs {region_names[j]}: "
                         f"p={result.p_value:.4f}, 显著={significant}")

质量控制分析
~~~~~~~~~~~~

在制造业中使用统计检验进行质量控制和过程改进。

.. code-block:: python

   def quality_control_analysis():
       """质量控制统计分析"""
       
       # 模拟生产线质量数据
       np.random.seed(42)
       
       # 改进前后的产品重量数据（目标：100g）
       before_improvement = np.random.normal(99.5, 2.5, 100)  # 偏离目标
       after_improvement = np.random.normal(100.1, 1.8, 100)  # 接近目标
       
       lt = LocationTests()
       nt = NormalityTests()
       
       print("质量控制分析报告")
       print("=" * 40)
       
       # 描述性统计
       print(f"改进前: 均值={np.mean(before_improvement):.3f}g, "
             f"标准差={np.std(before_improvement, ddof=1):.3f}g")
       print(f"改进后: 均值={np.mean(after_improvement):.3f}g, "
             f"标准差={np.std(after_improvement, ddof=1):.3f}g")
       
       # 单样本t检验：检验是否达到目标值100g
       target_weight = 100.0
       
       before_target_test = lt.one_sample_ttest(before_improvement, target_weight)
       after_target_test = lt.one_sample_ttest(after_improvement, target_weight)
       
       print(f"\n目标值检验 (100g):")
       print(f"改进前: t={before_target_test.statistic:.4f}, "
             f"p={before_target_test.p_value:.4f}, 达标={not before_target_test.reject_null}")
       print(f"改进后: t={after_target_test.statistic:.4f}, "
             f"p={after_target_test.p_value:.4f}, 达标={not after_target_test.reject_null}")
       
       # 配对比较：改进效果
       # 注意：这里假设是相同条件下的对比
       improvement_test = lt.two_sample_ttest(before_improvement, after_improvement)
       
       print(f"\n改进效果检验:")
       print(f"t统计量: {improvement_test.statistic:.4f}")
       print(f"p值: {improvement_test.p_value:.4f}")
       print(f"改进有效: {improvement_test.reject_null}")
       
       # 方差比较：一致性改善
       from scipy.stats import bartlett
       variance_test_stat, variance_test_p = bartlett(before_improvement, after_improvement)
       print(f"\n方差齐性检验 (一致性):")
       print(f"统计量: {variance_test_stat:.4f}, p值: {variance_test_p:.4f}")
       print(f"方差显著不同: {variance_test_p < 0.05}")

科学研究场景
------------

临床试验分析
~~~~~~~~~~~~

医学研究中的临床试验数据分析。

.. code-block:: python

   def clinical_trial_analysis():
       """临床试验统计分析"""
       
       # 模拟临床试验数据
       np.random.seed(42)
       
       # 治疗前后的血压数据（mmHg）
       n_patients = 80
       baseline_bp = np.random.normal(160, 20, n_patients)  # 基线血压
       
       # 治疗后血压（假设有治疗效果）
       treatment_effect = np.random.normal(-15, 8, n_patients)  # 平均降低15mmHg
       follow_up_bp = baseline_bp + treatment_effect
       
       # 对照组（安慰剂）
       placebo_baseline = np.random.normal(158, 18, n_patients)
       placebo_effect = np.random.normal(-3, 6, n_patients)  # 轻微安慰剂效应
       placebo_follow_up = placebo_baseline + placebo_effect
       
       lt = LocationTests()
       ct = CorrelationTests()
       
       print("临床试验分析报告")
       print("=" * 40)
       
       # 基线比较
       baseline_comparison = lt.two_sample_ttest(baseline_bp, placebo_baseline)
       print(f"基线血压比较:")
       print(f"治疗组: {np.mean(baseline_bp):.1f}±{np.std(baseline_bp, ddof=1):.1f} mmHg")
       print(f"对照组: {np.mean(placebo_baseline):.1f}±{np.std(placebo_baseline, ddof=1):.1f} mmHg")
       print(f"基线差异: p={baseline_comparison.p_value:.4f}, 平衡={not baseline_comparison.reject_null}")
       
       # 治疗效果分析
       treatment_change = follow_up_bp - baseline_bp
       placebo_change = placebo_follow_up - placebo_baseline
       
       print(f"\n治疗效果:")
       print(f"治疗组变化: {np.mean(treatment_change):.1f}±{np.std(treatment_change, ddof=1):.1f} mmHg")
       print(f"对照组变化: {np.mean(placebo_change):.1f}±{np.std(placebo_change, ddof=1):.1f} mmHg")
       
       # 组间比较
       effect_comparison = lt.two_sample_ttest(treatment_change, placebo_change)
       print(f"\n效果比较:")
       print(f"t统计量: {effect_comparison.statistic:.4f}")
       print(f"p值: {effect_comparison.p_value:.4f}")
       print(f"治疗有效: {effect_comparison.reject_null}")
       
       # 效应量
       pooled_std = np.sqrt(((len(treatment_change)-1)*np.var(treatment_change, ddof=1) + 
                           (len(placebo_change)-1)*np.var(placebo_change, ddof=1)) / 
                           (len(treatment_change)+len(placebo_change)-2))
       cohens_d = (np.mean(treatment_change) - np.mean(placebo_change)) / pooled_std
       print(f"效应量 (Cohen's d): {cohens_d:.3f}")

教育研究分析
~~~~~~~~~~~~

教育效果评估和学习方法比较。

.. code-block:: python

   def education_research_analysis():
       """教育研究统计分析"""
       
       # 模拟教育实验数据
       np.random.seed(42)
       
       # 三种教学方法的学生成绩
       traditional = np.random.normal(75, 12, 50)  # 传统教学
       online = np.random.normal(78, 15, 48)       # 在线教学
       blended = np.random.normal(82, 10, 52)      # 混合教学
       
       # 学习时间与成绩的关系
       study_hours = np.random.uniform(1, 8, 150)
       performance = 60 + 3 * study_hours + np.random.normal(0, 5, 150)
       
       lt = LocationTests()
       ct = CorrelationTests()
       
       print("教育研究分析报告")
       print("=" * 40)
       
       # 教学方法比较
       print("1. 教学方法效果比较:")
       methods = {'传统': traditional, '在线': online, '混合': blended}
       
       for name, scores in methods.items():
           print(f"{name}教学: n={len(scores)}, 均值={np.mean(scores):.2f}, "
                 f"标准差={np.std(scores, ddof=1):.2f}")
       
       # 方差分析
       anova_result = lt.one_way_anova(traditional, online, blended)
       print(f"\nANOVA结果:")
       print(f"F统计量: {anova_result.statistic:.4f}")
       print(f"p值: {anova_result.p_value:.4f}")
       print(f"教学方法差异显著: {anova_result.reject_null}")
       
       # 学习时间与成绩相关性
       correlation_result = ct.pearson_correlation(study_hours, performance)
       print(f"\n2. 学习时间与成绩相关性:")
       print(f"Pearson相关系数: {correlation_result.statistic:.4f}")
       print(f"p值: {correlation_result.p_value:.4f}")
       print(f"相关显著: {correlation_result.reject_null}")
       
       # 回归分析预测
       from sklearn.linear_model import LinearRegression
       from sklearn.metrics import r2_score
       
       lr = LinearRegression()
       X = study_hours.reshape(-1, 1)
       lr.fit(X, performance)
       
       print(f"\n简单线性回归:")
       print(f"回归方程: 成绩 = {lr.intercept_:.2f} + {lr.coef_[0]:.2f} × 学习时间")
       print(f"R²: {r2_score(performance, lr.predict(X)):.4f}")

金融数据分析
------------

时间序列和风险分析
~~~~~~~~~~~~~~~~~~

金融市场数据的统计分析。

.. code-block:: python

   def financial_analysis():
       """金融时间序列分析"""
       
       # 模拟股票收益率数据
       np.random.seed(42)
       n_days = 252  # 一年交易日
       
       # 股票A：稳定型
       returns_a = np.random.normal(0.0005, 0.015, n_days)  # 日收益率
       
       # 股票B：波动型（GARCH效应）
       returns_b = np.zeros(n_days)
       h = 0.015**2  # 初始方差
       
       for i in range(n_days):
           h = 0.00001 + 0.85*h + 0.1*returns_b[i-1]**2 if i > 0 else h
           returns_b[i] = np.random.normal(0.0008, np.sqrt(h))
       
       tst = TimeSeriesTests()
       lt = LocationTests()
       
       print("金融数据分析报告")
       print("=" * 40)
       
       # 描述性统计
       print("1. 收益率统计:")
       print(f"股票A: 均值={np.mean(returns_a)*252:.3f} (年化), "
             f"波动率={np.std(returns_a, ddof=1)*np.sqrt(252):.3f} (年化)")
       print(f"股票B: 均值={np.mean(returns_b)*252:.3f} (年化), "
             f"波动率={np.std(returns_b, ddof=1)*np.sqrt(252):.3f} (年化)")
       
       # 收益率差异检验
       returns_comparison = lt.two_sample_ttest(returns_a, returns_b)
       print(f"\n2. 收益率比较:")
       print(f"t统计量: {returns_comparison.statistic:.4f}")
       print(f"p值: {returns_comparison.p_value:.4f}")
       print(f"收益率差异显著: {returns_comparison.reject_null}")
       
       # ARCH效应检验
       arch_a = tst.arch_test(returns_a, lags=5)
       arch_b = tst.arch_test(returns_b, lags=5)
       
       print(f"\n3. 波动率聚集检验 (ARCH):")
       print(f"股票A: LM统计量={arch_a.statistic:.4f}, p={arch_a.p_value:.4f}, "
             f"ARCH效应={arch_a.reject_null}")
       print(f"股票B: LM统计量={arch_b.statistic:.4f}, p={arch_b.p_value:.4f}, "
             f"ARCH效应={arch_b.reject_null}")
       
       # 风险度量
       var_95_a = np.percentile(returns_a, 5)  # 5% VaR
       var_95_b = np.percentile(returns_b, 5)
       
       print(f"\n4. 风险度量 (95% VaR):")
       print(f"股票A: {var_95_a:.4f} ({var_95_a*np.sqrt(252):.3f} 年化)")
       print(f"股票B: {var_95_b:.4f} ({var_95_b*np.sqrt(252):.3f} 年化)")

综合案例：电商平台分析
----------------------

综合运用多种统计检验方法分析电商平台数据。

.. code-block:: python

   def ecommerce_comprehensive_analysis():
       """电商平台综合统计分析"""
       
       np.random.seed(42)
       
       # 模拟电商数据
       n_users = 1000
       
       # 用户特征
       age = np.random.normal(35, 12, n_users)
       income = np.random.normal(50000, 15000, n_users)  # 年收入
       
       # 购买行为（受年龄和收入影响）
       purchase_prob = 0.1 + 0.003 * (age - 20) + 0.000008 * income
       purchase_prob = np.clip(purchase_prob, 0, 1)
       purchases = np.random.binomial(1, purchase_prob, n_users)
       
       # 购买金额（仅对购买用户）
       purchase_amounts = []
       for i, purchased in enumerate(purchases):
           if purchased:
               base_amount = 100 + 2 * age[i] + 0.002 * income[i]
               amount = np.random.gamma(2, base_amount/2)
               purchase_amounts.append(amount)
       
       purchase_amounts = np.array(purchase_amounts)
       
       # 不同推荐算法的点击率
       algo_a_clicks = np.random.binomial(1, 0.08, 500)  # 传统算法
       algo_b_clicks = np.random.binomial(1, 0.12, 500)  # 新算法
       
       # 分析开始
       lt = LocationTests()
       ct = CorrelationTests()
       nt = NormalityTests()
       
       print("电商平台综合分析报告")
       print("=" * 50)
       
       # 1. 用户画像分析
       print("1. 用户画像:")
       print(f"   总用户数: {n_users}")
       print(f"   平均年龄: {np.mean(age):.1f}岁")
       print(f"   平均收入: {np.mean(income)/1000:.1f}千元")
       print(f"   购买率: {np.mean(purchases):.3f}")
       print(f"   平均购买金额: {np.mean(purchase_amounts):.2f}元")
       
       # 2. 年龄与购买行为关系
       purchase_ages = age[purchases == 1]
       non_purchase_ages = age[purchases == 0]
       
       age_purchase_test = lt.two_sample_ttest(purchase_ages, non_purchase_ages)
       print(f"\n2. 年龄与购买行为:")
       print(f"   购买用户平均年龄: {np.mean(purchase_ages):.1f}岁")
       print(f"   未购买用户平均年龄: {np.mean(non_purchase_ages):.1f}岁")
       print(f"   差异显著: {age_purchase_test.reject_null} (p={age_purchase_test.p_value:.4f})")
       
       # 3. 收入与购买金额相关性
       purchase_incomes = income[purchases == 1]
       correlation_income_amount = ct.pearson_correlation(purchase_incomes, purchase_amounts)
       print(f"\n3. 收入与购买金额相关性:")
       print(f"   相关系数: {correlation_income_amount.statistic:.4f}")
       print(f"   p值: {correlation_income_amount.p_value:.4f}")
       print(f"   显著相关: {correlation_income_amount.reject_null}")
       
       # 4. 推荐算法效果比较
       algo_a_rate = np.mean(algo_a_clicks)
       algo_b_rate = np.mean(algo_b_clicks)
       
       # 双比例检验
       from scipy.stats import fisher_exact
       contingency = [[np.sum(algo_b_clicks), len(algo_b_clicks) - np.sum(algo_b_clicks)],
                     [np.sum(algo_a_clicks), len(algo_a_clicks) - np.sum(algo_a_clicks)]]
       
       odds_ratio, p_value = fisher_exact(contingency)
       
       print(f"\n4. 推荐算法效果比较:")
       print(f"   算法A点击率: {algo_a_rate:.4f}")
       print(f"   算法B点击率: {algo_b_rate:.4f}")
       print(f"   提升: {(algo_b_rate - algo_a_rate)/algo_a_rate*100:.2f}%")
       print(f"   Fisher精确检验 p值: {p_value:.4f}")
       print(f"   算法B显著更好: {p_value < 0.05}")
       
       # 5. 购买金额分布分析
       normality_test = nt.shapiro_wilk(purchase_amounts[:50])  # 抽样检验
       print(f"\n5. 购买金额分布:")
       print(f"   中位数: {np.median(purchase_amounts):.2f}元")
       print(f"   75%分位数: {np.percentile(purchase_amounts, 75):.2f}元")
       print(f"   正态性检验 p值: {normality_test.p_value:.4f}")
       print(f"   近似正态分布: {not normality_test.reject_null}")
       
       # 6. 业务建议
       print(f"\n6. 分析结论与建议:")
       
       if age_purchase_test.reject_null:
           avg_purchase_age = np.mean(purchase_ages)
           if avg_purchase_age > np.mean(age):
               print("   - 年龄较大的用户更倾向于购买，建议针对成熟用户群体设计产品")
           else:
               print("   - 年轻用户更活跃，建议加强年轻用户的转化策略")
       
       if correlation_income_amount.reject_null:
           print("   - 收入与购买金额正相关，可考虑收入分层的定价策略")
       
       if p_value < 0.05:
           improvement = (algo_b_rate - algo_a_rate) / algo_a_rate * 100
           print(f"   - 新推荐算法效果显著提升{improvement:.1f}%，建议全面推广")
       
       return {
           'user_profile': {'age': np.mean(age), 'income': np.mean(income)},
           'conversion_rate': np.mean(purchases),
           'avg_purchase_amount': np.mean(purchase_amounts),
           'algo_improvement': (algo_b_rate - algo_a_rate) / algo_a_rate * 100
       }

运行所有示例
------------

.. code-block:: python

   def run_all_examples():
       """运行所有应用示例"""
       
       print("统计检验模块应用示例集合")
       print("=" * 60)
       
       try:
           print("\n" + "="*20 + " 机器学习模型比较 " + "="*20)
           model_comparison_analysis()
       except Exception as e:
           print(f"模型比较分析出错: {e}")
       
       try:
           print("\n" + "="*25 + " A/B测试分析 " + "="*25)
           ab_test_analysis()
       except Exception as e:
           print(f"A/B测试分析出错: {e}")
       
       try:
           print("\n" + "="*25 + " 销售数据分析 " + "="*25)
           sales_analysis()
       except Exception as e:
           print(f"销售数据分析出错: {e}")
       
       try:
           print("\n" + "="*25 + " 质量控制分析 " + "="*25)
           quality_control_analysis()
       except Exception as e:
           print(f"质量控制分析出错: {e}")
       
       try:
           print("\n" + "="*25 + " 临床试验分析 " + "="*25)
           clinical_trial_analysis()
       except Exception as e:
           print(f"临床试验分析出错: {e}")
       
       try:
           print("\n" + "="*25 + " 教育研究分析 " + "="*25)
           education_research_analysis()
       except Exception as e:
           print(f"教育研究分析出错: {e}")
       
       try:
           print("\n" + "="*25 + " 金融数据分析 " + "="*25)
           financial_analysis()
       except Exception as e:
           print(f"金融数据分析出错: {e}")
       
       try:
           print("\n" + "="*20 + " 电商平台综合分析 " + "="*20)
           ecommerce_comprehensive_analysis()
       except Exception as e:
           print(f"电商综合分析出错: {e}")
       
       print("\n" + "="*60)
       print("所有示例运行完成！")

   if __name__ == "__main__":
       run_all_examples()

这些示例展示了统计检验模块在实际业务场景中的应用，涵盖了从数据科学到商业分析的各个领域。每个示例都提供了完整的分析流程，包括数据准备、假设检验、结果解释和业务建议。 