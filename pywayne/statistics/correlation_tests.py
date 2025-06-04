"""
相关性分析与独立性检验

实现各种分析变量间相关关系和独立性的统计检验方法。
"""

import numpy as np
from typing import Union, List, Optional, Tuple
from scipy import stats
import warnings
from .base import BaseStatisticalTest, TestResult


class CorrelationTests(BaseStatisticalTest):
    """
    相关性分析与独立性检验类
    
    提供多种分析变量间相关关系的检验方法。
    """
    
    def __init__(self):
        super().__init__()
    
    def pearson_correlation(self, x: Union[np.ndarray, List], 
                           y: Union[np.ndarray, List],
                           alpha: float = 0.05,
                           alternative: str = 'two-sided') -> TestResult:
        """
        Pearson积差相关系数检验 - 检验两个连续变量的线性相关性
        """
        x = self._validate_array(x, "x")
        y = self._validate_array(y, "y")
        
        if len(x) != len(y):
            raise ValueError("两个变量的长度必须相同")
        
        # 移除配对的NaN值
        x, y = self._remove_nan((x, y), paired=True)
        
        if len(x) < 3:
            raise ValueError("Pearson相关检验需要至少3对观测值")
        
        correlation, p_value = stats.pearsonr(x, y)
        
        # 调整p值以适应不同的备择假设
        if alternative == 'less':
            p_value = p_value / 2 if correlation < 0 else 1 - p_value / 2
        elif alternative == 'greater':
            p_value = p_value / 2 if correlation > 0 else 1 - p_value / 2
        
        reject_null = p_value < alpha
        
        # 计算置信区间（Fisher z变换）
        n = len(x)
        z_r = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se_z = 1 / np.sqrt(n - 3)
        z_critical = stats.norm.ppf(1 - alpha/2)
        z_lower = z_r - z_critical * se_z
        z_upper = z_r + z_critical * se_z
        
        # 转换回相关系数
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return TestResult(
            test_name="Pearson积差相关检验",
            statistic=correlation,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=n - 2,
            confidence_interval=(r_lower, r_upper),
            additional_info={
                "样本量": n,
                "相关系数": correlation,
                "相关强度": self._interpret_correlation(abs(correlation)),
                "备择假设": alternative
            }
        )
    
    def spearman_correlation(self, x: Union[np.ndarray, List], 
                            y: Union[np.ndarray, List],
                            alpha: float = 0.05,
                            alternative: str = 'two-sided') -> TestResult:
        """
        Spearman秩相关检验 - 检验两个变量的单调关系（非参数）
        """
        x = self._validate_array(x, "x")
        y = self._validate_array(y, "y")
        
        if len(x) != len(y):
            raise ValueError("两个变量的长度必须相同")
        
        # 移除配对的NaN值
        x, y = self._remove_nan((x, y), paired=True)
        
        if len(x) < 3:
            raise ValueError("Spearman相关检验需要至少3对观测值")
        
        correlation, p_value = stats.spearmanr(x, y, alternative=alternative)
        reject_null = p_value < alpha
        
        n = len(x)
        
        return TestResult(
            test_name="Spearman秩相关检验",
            statistic=correlation,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=n - 2,
            additional_info={
                "样本量": n,
                "秩相关系数": correlation,
                "相关强度": self._interpret_correlation(abs(correlation)),
                "备择假设": alternative
            }
        )
    
    def kendall_tau(self, x: Union[np.ndarray, List], 
                   y: Union[np.ndarray, List],
                   alpha: float = 0.05,
                   alternative: str = 'two-sided',
                   variant: str = 'b') -> TestResult:
        """
        Kendall τ相关检验 - 检验两个变量的等级相关性
        """
        x = self._validate_array(x, "x")
        y = self._validate_array(y, "y")
        
        if len(x) != len(y):
            raise ValueError("两个变量的长度必须相同")
        
        # 移除配对的NaN值
        x, y = self._remove_nan((x, y), paired=True)
        
        if len(x) < 3:
            raise ValueError("Kendall检验需要至少3对观测值")
        
        correlation, p_value = stats.kendalltau(x, y, alternative=alternative, variant=variant)
        reject_null = p_value < alpha
        
        n = len(x)
        
        return TestResult(
            test_name=f"Kendall τ{variant}相关检验",
            statistic=correlation,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            additional_info={
                "样本量": n,
                "Kendall τ": correlation,
                "相关强度": self._interpret_correlation(abs(correlation)),
                "备择假设": alternative,
                "变体": variant
            }
        )
    
    def chi_square_independence(self, contingency_table: Union[np.ndarray, List],
                               alpha: float = 0.05,
                               correction: bool = True) -> TestResult:
        """
        卡方独立性检验 - 检验两个分类变量是否独立
        """
        table = self._validate_array(contingency_table, "contingency_table")
        
        if table.ndim != 2:
            raise ValueError("列联表必须是二维数组")
        
        if np.any(table < 0):
            raise ValueError("列联表中不能有负数")
        
        if np.sum(table) == 0:
            raise ValueError("列联表不能全为零")
        
        # 进行卡方检验
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=correction)
        reject_null = p_value < alpha
        
        # 检查期望频数
        min_expected = np.min(expected)
        low_expected_count = np.sum(expected < 5)
        
        if min_expected < 1:
            warnings.warn("有期望频数小于1，检验结果可能不可靠", UserWarning)
        elif low_expected_count > 0:
            warnings.warn(f"有{low_expected_count}个单元格期望频数小于5，检验结果可能不可靠", UserWarning)
        
        # 计算效应量 (Cramér's V)
        n = np.sum(table)
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
        
        return TestResult(
            test_name="卡方独立性检验",
            statistic=chi2,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=dof,
            effect_size=cramers_v,
            additional_info={
                "列联表形状": table.shape,
                "总频数": int(n),
                "最小期望频数": min_expected,
                "期望频数<5的单元格数": int(low_expected_count),
                "Cramér's V": cramers_v,
                "连续性校正": correction,
                "观察频数": table.tolist(),
                "期望频数": expected.tolist()
            }
        )
    
    def fisher_exact(self, table: Union[np.ndarray, List],
                    alpha: float = 0.05,
                    alternative: str = 'two-sided') -> TestResult:
        """
        Fisher精确检验 - 2×2列联表的精确独立性检验
        """
        table = self._validate_array(table, "table")
        
        if table.shape != (2, 2):
            raise ValueError("Fisher精确检验要求2×2列联表")
        
        if np.any(table < 0):
            raise ValueError("列联表中不能有负数")
        
        if not np.all(table == table.astype(int)):
            raise ValueError("列联表必须包含整数频数")
        
        table = table.astype(int)
        
        odds_ratio, p_value = stats.fisher_exact(table, alternative=alternative)
        reject_null = p_value < alpha
        
        # 计算置信区间（对数几率比的置信区间）
        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        
        if a == 0 or b == 0 or c == 0 or d == 0:
            # 有零单元格时的处理
            log_or_se = float('inf')
            ci_lower, ci_upper = 0, float('inf')
        else:
            log_or = np.log(odds_ratio)
            log_or_se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            z_critical = stats.norm.ppf(1 - alpha/2)
            log_ci_lower = log_or - z_critical * log_or_se
            log_ci_upper = log_or + z_critical * log_or_se
            ci_lower = np.exp(log_ci_lower)
            ci_upper = np.exp(log_ci_upper)
        
        return TestResult(
            test_name="Fisher精确检验",
            statistic=odds_ratio,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            confidence_interval=(ci_lower, ci_upper),
            additional_info={
                "几率比": odds_ratio,
                "列联表": table.tolist(),
                "总频数": int(np.sum(table)),
                "备择假设": alternative
            }
        )
    
    def mcnemar_test(self, table: Union[np.ndarray, List],
                    alpha: float = 0.05,
                    exact: bool = True,
                    correction: bool = True) -> TestResult:
        """
        McNemar检验 - 配对二分类数据的变化检验
        """
        table = self._validate_array(table, "table")
        
        if table.shape != (2, 2):
            raise ValueError("McNemar检验要求2×2配对列联表")
        
        if np.any(table < 0):
            raise ValueError("列联表中不能有负数")
        
        if not np.all(table == table.astype(int)):
            raise ValueError("列联表必须包含整数频数")
        
        table = table.astype(int)
        
        # 提取不一致的配对
        b = table[0, 1]  # (0,1)配对：前否后是
        c = table[1, 0]  # (1,0)配对：前是后否
        
        n_discordant = b + c
        
        if n_discordant == 0:
            # 没有不一致的配对
            return TestResult(
                test_name="McNemar检验",
                statistic=0,
                p_value=1.0,
                reject_null=False,
                alpha=alpha,
                additional_info={
                    "配对列联表": table.tolist(),
                    "不一致配对数": 0,
                    "注释": "没有不一致的配对，无法检验"
                }
            )
        
        if exact and n_discordant <= 25:
            # 精确检验（二项检验）
            p_value = 2 * min(stats.binom.cdf(min(b, c), n_discordant, 0.5),
                            1 - stats.binom.cdf(min(b, c), n_discordant, 0.5))
            statistic = min(b, c)
            test_type = "精确"
        else:
            # 大样本近似（卡方检验）
            if correction and n_discordant < 30:
                # 连续性校正
                statistic = (abs(b - c) - 1)**2 / (b + c)
                test_type = "连续性校正"
            else:
                statistic = (b - c)**2 / (b + c)
                test_type = "卡方近似"
            
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        reject_null = p_value < alpha
        
        return TestResult(
            test_name=f"McNemar检验 ({test_type})",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=1 if not exact else None,
            additional_info={
                "配对列联表": table.tolist(),
                "不一致配对": {"前否后是": int(b), "前是后否": int(c)},
                "不一致配对总数": int(n_discordant),
                "检验类型": test_type
            }
        )
    
    @staticmethod
    def _interpret_correlation(r: float) -> str:
        """解释相关系数的强度"""
        if r < 0.1:
            return "微弱"
        elif r < 0.3:
            return "弱"
        elif r < 0.5:
            return "中等"
        elif r < 0.7:
            return "强"
        else:
            return "很强" 