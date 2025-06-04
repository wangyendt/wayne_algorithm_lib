# 统计检验方法清单

以下按照不同类型的统计检验进行分类汇总，并以表格形式给出每种方法的名称、所属分类（参数/非参数/贝叶斯）、典型用途、适用条件、支持的数据类型以及常用的 Python 实现库。

## 分布拟合与正态性检验

常用于检验数据分布形态是否符合特定分布假设（如正态分布）或检验观测分布与理论分布的拟合优度。

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|:-----|------|----------|----------|----------|-------------|
| Shapiro–Wilk 正态性检验(Shapiro–Wilk test) | 非参数 | 检验单一样本数据是否来自正态分布。 | 样本量需在 3～5000 范围内，样本来自连续型分布（>5000时W统计量准确但p值不可靠）。 | 连续型变量 | scipy.stats.shapiro |
| Kolmogorov–Smirnov 检验(K–S test, 单样本) | 非参数 | 拟合优度检验：比较单一样本的经验分布函数与给定理论分布是否一致。 | 样本独立。同样可用于检验数据是否服从特定分布（需完全指定理论分布参数）。 | 连续型变量 | scipy.stats.kstest |
| Kolmogorov–Smirnov 双样本检验(Two-sample K–S test) | 非参数 | 检验两独立样本是否来自同一分布（比较两样本分布差异）。 | 两样本独立，测量尺度至少连续型。用于分布整体形状差异的检验。 | 连续型变量 | scipy.stats.ks_2samp |
| Anderson–Darling 检验(Anderson–Darling test) | 非参数 | 正态性检验或其他分布拟合优度检验，比K–S对尾部更敏感。 | 样本独立。同SciPy 实现支持正态、指数等特定分布的检验。 | 连续型变量 | scipy.stats.anderson |
| D'Agostino–Pearson K² 检验(D'Agostino's K²) | 非参数 | 正态性检验：利用样本的偏度和峰度检验数据偏离正态的程度。 | 样本独立。需中等样本量（偏度/峰度估计需要一定样本）。 | 连续型变量 | scipy.stats.normaltest |
| Jarque–Bera 检验(Jarque–Bera test) | 非参数 | 正态性检验：基于偏度和峰度的联合检验，常用于回归残差正态性检验。 | 样本独立。通常用于大样本（基于渐近$$\chi^2$$分布统计量）。 | 连续型变量 | statsmodels.stats.stattools.jarque_bera 等 |
| 卡方拟合优度检验(Chi-square goodness-of-fit) | 非参数 | 检验分类数据的观察频数分布与期望分布是否有显著差异。 | 样本来自多分类总体，期望频数需>=5，样本量宜>50。 | 分类数据(名义尺度) | scipy.stats.chisquare |
| G 检验(G-test, 似然比拟合优度检验) | 非参数 | 拟合优度检验的替代方案：基于对数似然比计算，与卡方检验结果相近。 | 样本分类频数。适用于较小样本或希望使用对数似然方法的情形。 | 分类数据 | statsmodels （需自行计算或使用函数组合） |

## 参数估计与假设检验（均值/位置参数比较）

用于估计总体参数或比较组间位置参数（如均值、中位数）的检验方法，包括参数检验和非参数检验。

### 单样本检验

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| 单样本 t 检验(One-sample t-test) | 参数 | 检验单个样本均值是否等于给定值（总体均值假设检验）。 | 样本独立，数据近似正态；若样本较大，t 检验对正态性不太敏感。 | 连续型变量 | scipy.stats.ttest_1samp |
| 单样本 Z 检验(One-sample z-test) | 参数 | 大样本下检验单个样本均值与指定值差异（总体方差已知或样本量足够大）。 | 样本独立，需要已知总体方差或 n 大（根据中心极限定理近似正态）。 | 连续型变量 | statsmodels （需自行计算） |
| 符号检验(Sign test) | 非参数 | 检验样本中位数是否等于给定值（基于正负符号的简单非参数方法）。 | 无需正态假设，样本取自连续或有序分布；以中位数为假设测试量。 | 连续或有序变量 | 自行实现（ binom_test 可用作双尾符号检验） |
| 二项检验(Binomial test) | 非参数 | 检验二项分布的成功概率是否等于某值（如样本中某事件概率）。 | 二分类数据，独立试验。精确检验基于二项分布，适用于任意样本量。 | 二分类数据 | scipy.stats.binomtest （精确） |

### 两独立样本检验

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| 独立样本 t 检验(Independent two-sample t-test) | 参数 | 比较两独立样本均值是否有显著差异。 | 两组独立，组内数据近似正态，且方差齐性（不齐时建议Welch t 检验）。 | 连续型变量 | scipy.stats.ttest_ind （ equal_var 参数） |
| Welch t 检验(Welch's t-test) | 参数 | 两独立样本均值比较的方差不等情形，用于替代标准 t 检验。 | 两组独立，正态假设同上；不要求方差相等。 | 连续型变量 | scipy.stats.ttest_ind （ equal_var=False ） |
| Mann–Whitney U 检验(Mann–Whitney U test) | 非参数 | 比较两独立样本的中位数或分布位置差异（秩和检验)。 | 两组独立，测量至少为有序数据。无需正态假设，检验两个分布总体是否相同位置。 | 连续或有序变量 | scipy.stats.mannwhitneyu |
| 两样本比例检验(Two-proportion z-test) | 参数 | 比较两独立样本中某事件发生的比例差异是否显著。 | 两组独立，样本量较大使得比例差近似正态（例如每组成功次数≥5）。 | 二分类数据 | statsmodels.stats.proportions_ztest |
| 卡方独立性检验(Chi-square test, 2×2 情况) | 非参数 | 检验两个分类变量（2×2表）的频率分布是否独立，即比例是否相等。 | 需要所有期望频数≥5；样本来自简单随机抽样。 | 分类数据 | scipy.stats.chi2_contingency (适用于任意列联表) |
| Fisher 精确检验(Fisher's exact test) | 非参数 | 当样本量较小或期望频数过低时，精确检验2×2列联表的独立性。 | 用于2×2分类表，小样本条件下替代卡方检验。 | 分类数据 | scipy.stats.fisher_exact |

### 成对（配对）样本检验

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| 配对 t 检验(Paired t-test) | 参数 | 比较配对实验前后或成对观测的均值差异（对同一对象两测量）。 | 成对差值近似正态。假设差值的总体均值=0为原假设。 | 连续型变量 | scipy.stats.ttest_rel |
| Wilcoxon 符号秩检验(Wilcoxon signed-rank test) | 非参数 | 配对样本的中位数差异检验：对配对差值的秩符号进行检验。 | 配对差值为连续或有序，分布对称为佳；无需正态假设。 | 连续或有序变量 | scipy.stats.wilcoxon |
| McNemar 检验(McNemar's test) | 非参数 | 配对前后（如处理前后）二分类结果变化的检验，常用于配对案例对照。 | 2×2配对列联表，关注类别改变与否。要求总和固定，适用于中等样本。 | 二分类配对数据 | statsmodels.stats.contingency_tables.mcnemar |

### 多组比较检验

用于比较三组及以上样本的总体位置参数差异，包含方差分析及非参数方法等。

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| 单因素方差分析(One-way ANOVA) | 参数 | 比较三组或以上独立样本均值是否存在总体显著差异。 | 各组独立，组内数据近似正态，方差齐性假定成立（Bartlett/Levene 检验可验证）。 | 连续型变量 | scipy.stats.f_oneway |
| 多因素方差分析(Factorial ANOVA) | 参数 | 两因素或更多因素的实验设计比较均值差异（可考察交互作用）。 | 各细分组正态且方差齐。同单因素ANOVA条件，且需要各组合有足够样本。 | 连续型变量 | statsmodels.formula.api.ols + statsmodels.stats.anova_lm |
| 重复测量方差分析(Repeated Measures ANOVA) | 参数 | 比较相同受试对象在不同处理下的均值差异（配对的多水平比较）。 | 正态、方差齐性假设应用于差值；需要考虑测量间相关性（可用混合模型替代）。 | 连续型变量(配对设计) | statsmodels (MixedLM 或 AnovaRM ) |
| Kruskal–Wallis 检验(Kruskal–Wallis H test) | 非参数 | 非参数单因素多组比较：检测 k 组独立样本分布位置是否相同。 | 各组独立，测量至少为有序。无需正态假设；样本量较小时仍可使用。 | 连续或有序变量 | scipy.stats.kruskal |
| Friedman 检验(Friedman test) | 非参数 | 非参数重复测量比较：配对的 k 组（区组设计）样本是否来自同一分布。 | 需方差分析的区组设计，各区组内排序。无需正态，需至少ordinal 数据。 | 连续或有序变量(配对) | scipy.stats.friedmanchisquare |
| Mood 中位数检验(Mood's median test) | 非参数 | 比较多个独立样本的中位数是否相等。 | 各组独立。对偏态分布有效但效率较低。无需正态，但要求有序或连续数据。 | 连续或有序变量 | scipy.stats.median_test （在 scipy 1.11+） |
| 方差齐性检验(Homogeneity of variance tests) | 参数/非参数 | 检验多个组方差是否相等，为ANOVA前提假设做验证。如Bartlett检验（参数，需正态）、Levene检验（较稳健）等。 | Bartlett 需各组正态；Levene 对偏态更稳健，可用于非正态情况（可基于均值或中位数）。 | 连续型变量 | scipy.stats.bartlettscipy.stats.levene |

**注：** 方差分析显著时常需事后两两比较，例如 Tukey HSD 检验（参数）或 Dunn 检验（非参数，Kruskal–Wallis事后），这些方法通常有专门实现（如 statsmodels.stats.multicomp.pairwise_tukeyhsd 或scikit-posthocs 库）。

## 相关性分析与独立性检验

用于分析变量间的相关关系及分类变量间的独立性。

### 连续型变量相关性

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| Pearson 积差相关系数检验(Pearson correlation test) | 参数 | 检验两个连续变量线性相关的强度及显著性（H0: ρ = 0）。 | 假设两变量近似服从双变量正态分布，线性关系。样本量 n 足够大时即使非正态也可近似。 | 连续型变量 | scipy.stats.pearsonr |
| Spearman 秩相关检验(Spearman's rank correlation) | 非参数 | 检验两个变量的秩次（排序）相关性，衡量单调关系强度。 | 无需假定正态或线性，仅要求变量可排序。对离群值不敏感。 | 连续或有序变量 | scipy.stats.spearmanr |
| Kendall 协方差检验(Kendall's tau) | 非参数 | 检验两个变量等级相关性的另一种方法，适合样本较小或有许多并列值情况。 | 无需假定正态。对样本量要求较低；处理有序分类数据或小样本更稳健。 | 连续或有序变量 | scipy.stats.kendalltau |

### 分类变量独立性

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| Pearson 卡方独立性检验(Chi-square test of independence) | 非参数 | 检验两个或多个分类变量的列联表中是否存在关联（H0:独立）。 | 需所有期望频数不过低（常用规则：期望值≥5）；样本来自独立观察。 | 分类数据(列联表) | scipy.stats.chi2_contingency |
| Fisher 精确检验(Fisher's exact test) | 非参数 | 检验2×2列联表变量独立性的小样本精确方法。 | 适用于2×2表的较小样本或稀疏数据，计算精确$$p$$值，无需大样本近似。 | 分类数据(2×2表) | scipy.stats.fisher_exact |
| Cochran's Q 检验(Cochran's Q test) | 非参数 | 检验 k 组配对的二分类变量总体概率是否相等（McNemar的扩展）。 | 每个受试对象在 k 种处理下产生二分类结果的比较；要求样本来自区组设计（配对）。 | 二分类配对数据 | statsmodels.stats.contingency_tables.cochrans_q |

## 时间序列分析检验

用于时间序列模型和特性检验，包括平稳性、单位根、协整、自相关等。

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| Augmented Dickey–Fuller 检验(ADF 单位根检验) | 参数 | 单根检验：检验时间序列是否存在单位根（H0: 序列非平稳，存在单位根）。 | 需要选择滞后阶数以消除自相关。原假设为存在单位根（非平稳）；p<0.05 拒绝则认为平稳。 | 时间序列（连续） | statsmodels.tsa.stattools.adfuller |
| KPSS 平稳性检验(KPSS test) | 参数 | 平稳性检验：与ADF相反，H0为序列平稳（趋势平稳），检验是否拒绝平稳假设。 | 需假设趋势或水平平稳形式。原假设为序列平稳；p<0.05 若拒绝则存在单位根（非平稳）。 | 时间序列（连续） | statsmodels.tsa.stattools.kpss |
| Phillips–Perron 检验(PP test) | 参数 | 单位根检验：与 ADF 类似但允许更一般的自相关结构，利用新西兰Phillips-Perron方法修正。 | 与 ADF 类似，但对异方差和自相关更稳健。原假设同为存在单位根。 | 时间序列（连续） | statsmodels.tsa.stattools.adfuller (可选regression='ct' ) |
| Ljung–Box Q 检验(Ljung–Box test) | 非参数 | 检验时间序列整体的自相关是否显著（对多个滞后一起的白噪声检验）。 | 原假设：序列在选定的滞后数内无自相关。需要足够长的序列，大样本下统计量~$$\chi^2$$. | 时间序列（连续） | statsmodels.stats.diagnostic.acorr_ljungbox |
| Durbin–Watson 检验(Durbin–Watson test) | 参数 | 检验回归残差（或序列）一阶自相关是否存在（正相关显著性）。 | 用于回归模型残差或时间序列。统计量介于0~4，约2为无自相关；需参考临界值表判断显著性。 | 时间序列（连续） | statsmodels.stats.stattools.durbin_watson |
| Breusch–Godfrey 检验(Breusch–Godfrey test) | 参数 | 回归残差高阶自相关的LM检验，可检验任意阶的序列相关性。 | 用于回归模型，多阶滞后下原假设为无自相关。样本较大会更可靠，基于$$\chi^2$$近似。 | 时间序列（连续） | statsmodels.stats.diagnostic.acorr_breusch_godfrey |
| Granger 因果检验(Granger causality test) | 参数 | 检验两个时间序列间的因果关系：一个序列的滞后值是否有助预测另一序列。 | 需要选择滞后阶数。原假设：不存在因果影响（滞后项系数=0）。 | 时间序列（连续） | statsmodels.tsa.stattools.grangercausalitytests |
| Engle–Granger 协整检验(Engle–Granger cointegration test) | 参数 | 检验多元时间序列是否存在协整关系（长期稳定均衡关系）。 | 分两步：先回归求残差，再对残差做单位根检验（ADF）。原假设：无协整（残差有单位根）。 | 时间序列（连续，多元） | statsmodels.tsa.stattools.coint |
| Johansen 协整检验(Johansen test) | 参数 | 多变量协整检验，可同时检验协整关系的个数（基于特征值LR检验）。 | 需较大样本；假设模型误差正态。可检验存在r个协整向量的假设。 | 时间序列（连续，多元） | statsmodels.tsa.vector_ar.coint_johansen |
| ARCH 效应检验(Engle's ARCH test) | 参数 | 检验时间序列残差中是否存在异方差（ARCH波动聚集效应）。 | 原假设：无 ARCH 效应（残差方差无相关）。需指定滞后阶数。 | 时间序列（连续） | statsmodels.stats.diagnostic.het_arch |
| 游程检验(Runs test, 随机性检验) | 非参数 | 检验序列值的正负或高低符号是否随机排列（整体随机性/独立性）。 | 原假设：序列符号随机独立分布。可用于检验序列随机走势或控制图中的随机模式。 | 时间序列（二值序列） | statsmodels.sandbox.stats.runs.runstest_1samp (实验性) |

## 模型诊断与假设检验

用于回归等统计模型中，对模型假定的检验和诊断，包括残差分布、异方差、多重共线性等。

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| 残差正态性检验(Residual normality test) | 非参数 | 检验模型残差是否符合正态分布假设（如线性回归假定）。 | 收集模型残差并应用正态性检验，如Shapiro-Wilk、Jarque-Bera 等。 | 连续型（模型残差） | scipy.stats.shapiro 等 |
| Breusch–Pagan 异方差检验(Breusch–Pagan test) | 参数 | 检验回归模型中残差的方差是否随预测变量变化（异方差）。 | 原假设：残差方差恒定。需模型残差近似正态、大样本下卡方近似有效。 | 连续型（模型残差） | statsmodels.stats.diagnostic.het_breuschpagan |
| White 异方差检验(White's test) | 参数 | 无需指定形式的通用异方差检验，通过残差平方回归的R²检验异方差。 | 原假设：残差方差恒定。样本较大会提高检验有效性；统计量服从$$\chi^2$$近似。 | 连续型（模型残差） | statsmodels.stats.diagnostic.het_white |
| Goldfeld–Quandt 检验(Goldfeld–Quandt test) | 参数 | 异方差检验的一种：比较回归残差在高低子样本中的方差差异。 | 原假设：两子样本方差相等。需按某序排序数据并排除中间部分。 | 连续型（模型残差） | statsmodels.stats.diagnostic.het_goldfeldquandt |
| 多重共线性诊断 (VIF)(Variance Inflation Factor) | 诊断 | 诊断回归预测变量间是否存在强共线性（VIF > 10 常指严重共线性）。 | 计算每个自变量针对其他自变量回归的 $$R^2$$ 决定 VIF = 1/(1-$$R^2$$)。 | 连续型（设计矩阵） | statsmodels.stats.outliers_influence.variance_inflation_factor |
| Ramsey RESET 检验(Ramsey RESET test) | 参数 | 检验回归模型的函数形式是否正确（是否有高次项遗漏等）。 | 原假设：模型无遗漏变量/形式正确。需正态误差假设，大样本F检验。 | 连续型（模型残差） | statsmodels.stats.diagnostic.linear_reset (在dev版本中) |
| Chow 分割检验(Chow test) | 参数 | 检验两个回归模型系数是否相同（结构性断裂检验，如前后时期）。 | 原假设：两个样本的回归系数相同。要求模型误差正态同方差，使用F检验。 | 连续型变量 | linearmodels.breaks.chow_test （ linearmodels 库） |
| 似然比检验(LRT)(Likelihood ratio test) | 参数 | 比较嵌套模型的拟合优度差异：检验较复杂模型相对于简单模型的改进是否显著。 | 需要两个嵌套模型的对数似然。原假设：简单模型足够好。统计量$$-2\Delta\ln L$$ 近似$$\chi^2$$分布。 | 视模型而定 | statsmodels （模型结果 .llr 和 .llr_pvalue ） |
| Wald 检验(Wald test) | 参数 | 检验模型中单个或多个回归系数是否为0（或满足线性约束）的假设。 | 原假设：指定的系数组合满足约束（如=0）。需大样本，估计值近似正态。 | 连续型变量 | statsmodels.regression.linear_model.OLSResults.wald_test |

**注：** 回归模型整体显著性通常通过 F 检验 或 似然比检验 实现，例如线性回归的整体 F 检验，洛吉斯蒂回归的整体$$\chi^2$$ 检验等，这些在模型拟合结果中自带。上表中的 Wald 检验用于单独系数或一般线性约束的检验。

## 统计过程控制（SPC）检验

常用于工业过程控制，通过控制图监测过程稳定性，包含Shewhart控制图及各种规则检验异常。

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| Shewhart 控制图(Shewhart control charts) | 参数 | 经典控制图方法，包括平均值-极差（X̄–R）图、平均值-标准差（X̄–S）图，用于监测过程平均和波动。 | 假设过程在受控时服从稳定分布（如正态用于计量值，二项/泊松用于计数值）；中心线及控制限依据历史数据计算（±3σ原则）。 | 连续型或计数型时间序列 | pyshewhart 库（第三方）或自行计算 |
| 属性控制图(Attribute control charts) | 参数 | 监测分类数据不合格率或缺陷数的控制图，如 p 图（不合格率）、np 图（不合格个数）、c 图（单位缺陷数）、u 图（单位缺陷率）。 | 假设过程在受控时符合相应分布：p/np 图假定二项分布，c/u 图假定泊松分布。需足够样本计算控制限。 | 二分类或计数时间序列 | 自行实现（可用scipy.stats.binom / poisson 计算控制限） |
| Western Electric (WECO) 规则(Western Electric rules) | 非参数 | 控制图辅助判异规则，一组经验决策规则，用于检测控制图上的异常模式。 | 将控制图区分成3个σ区域，通过一系列规则（如连续超过中心线的一定点数、超出控制限等）判定失控信号。 | 连续型或计数型时间序列 | pyshewhart 等库内含规则实现；或自行编码逻辑 |
| CUSUM 检验图(CUSUM chart) | 参数 | 累积和控制图，检测过程均值发生细小持久偏移的快速检验方法。 | 假设已知目标均值和方差，通过累积偏差和控制限判断。对小幅持续偏移灵敏，但对单点大偏差反应慢。 | 连续型时间序列 | 自行实现（或使用wqchart 等质量控制库） |
| EWMA 控制图(EWMA chart) | 参数 | 指数加权移动平均控制图，对近期数据赋予更大权重，敏感检测小的过程偏移。 | 需选择平滑系数 λ（常用0.2~0.3）。假设初始受控过程均值和方差已知，用加权平均控制限判异。 | 连续型时间序列 | 自行实现（或使用wqchart 等库） |
| 运行规则检验(Run rules tests) | 非参数 | 一系列基于顺序模式的规则（如Nelson 规则），检测趋势、循环等非随机模式。 | 不假定特定分布，仅基于数据顺序特征：如连续若干点上升/下降、交替、高于/低于均值等模式。 | 连续型或二值时间序列 | pyshewhart 或自行实现 |

**注：** 控制图本身是假设过程稳定时的数据分布已知，然后根据统计偏差来判断失控。Western Electric/Nelson 等规则进一步提高了对非随机模式的检测灵敏度，常作为控制图的补充工具。

## 贝叶斯统计检验方法

以贝叶斯方法为基础的检验和模型检查手段，利用后验分布和贝叶斯因子等进行推断。

| 方法 | 分类 | 典型用途 | 适用条件 | 数据类型 | Python 实现 |
|------|------|----------|----------|----------|-------------|
| 贝叶斯因子检验(Bayes factor test) | 贝叶斯 | 通过 贝叶斯因子 (BF) 比较两竞争假设的支持力度：BF<1倾向支持H0，>1 倾向支持 H1。 | 需要为 H0 和 H1 定义先验和似然。结果依赖先验选择。常用于模型比较或替代经典假设检验。 | 任意类型（需建模) | pymc / PyMC3 （计算边际似然），pingouin.bayesfactor_ttest 等 |
| 后验置信区间检验(Credible interval approach) | 贝叶斯 | 利用后验分布的 可信区间 (例如95% HPD区间) 判断参数是否包含感兴趣值（如0）。类似于检验参数显著性：若后验可信区间不包含0，则认为效果显著。 | 需获得参数后验分布（通过解析或MCMC抽样）。结论依赖可信区间和实用等价区间(ROPE)选择。 | 任意类型（需参数估计） | pymc / Pyro (MCMC 抽样计算后验)， arviz （提取HPD区间） |
| 后验预测检验(Posterior predictive check) | 贝叶斯 | 检验模型拟合优度：根据模型的后验预测分布生成模拟数据，与真实数据分布对比，检查模型能否再现数据特征。 | 需要可生成后验预测样本的模型。通常通过绘图或检验统计量（如分位数、分布距离）比较。 | 任意类型（需模型） | pymc / Pyro (产生后验预测)，arviz （ plot_ppc 可视化） |

**注：** 贝叶斯方法侧重于提供效应量的不确定度范围和相对支持证据，而非单一的显著性$$p$$值。例如，贝叶斯因子可以定量比较模型，后验预测检查则是评估模型与数据吻合度的重要工具。用户可借助 PyMC , Pyro , ArviZ 等库构建自定义的贝叶斯检验函数。

---

## 参考资料

1. [Kolmogorov–Smirnov test - Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
2. [shapiro — SciPy v1.15.3 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
3. [List of statistical tests - Wikipedia](https://en.wikipedia.org/wiki/List_of_statistical_tests)
4. [Western Electric rules - Wikipedia](https://en.wikipedia.org/wiki/Western_Electric_rules)
5. [Augmented Dickey–Fuller test - Wikipedia](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
6. [KPSS test - Wikipedia](https://en.wikipedia.org/wiki/KPSS_test)
7. [How to Interpret Cointegration Test Results | Aptech](https://www.aptech.com/blog/how-to-interpret-cointegration-test-results/)
8. [pingouin.bayesfactor_ttest — pingouin 0.5.5 documentation](https://pingouin-stats.org/build/html/generated/pingouin.bayesfactor_ttest.html)