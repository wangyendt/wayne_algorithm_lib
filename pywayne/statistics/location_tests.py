"""
位置参数检验

实现各种用于比较样本位置参数（均值、中位数等）的统计检验方法。
"""

import numpy as np
from typing import Union, List, Optional, Tuple
from scipy import stats
import warnings
from .base import BaseStatisticalTest, TestResult


class LocationTests(BaseStatisticalTest):
    """
    位置参数检验类
    
    提供多种比较样本位置参数的检验方法。
    """
    
    def __init__(self):
        super().__init__()
    
    def one_sample_ttest(self, data: Union[np.ndarray, List], 
                        popmean: float = 0, alpha: float = 0.05,
                        alternative: str = 'two-sided') -> TestResult:
        """
        单样本t检验 - 检验单个样本均值是否等于给定值
        """
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 2:
            raise ValueError("t检验需要至少2个观测值")
        
        statistic, p_value = stats.ttest_1samp(data, popmean, alternative=alternative)
        reject_null = p_value < alpha
        
        # 计算效应量 (Cohen's d)
        effect_size = (np.mean(data) - popmean) / np.std(data, ddof=1)
        
        # 计算置信区间
        n = len(data)
        df = n - 1
        se = stats.sem(data)
        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin_error = t_critical * se
        mean_data = np.mean(data)
        ci = (mean_data - margin_error, mean_data + margin_error)
        
        return TestResult(
            test_name="单样本t检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=df,
            effect_size=effect_size,
            confidence_interval=ci,
            additional_info={
                "样本量": n,
                "样本均值": mean_data,
                "假设均值": popmean,
                "标准误": se,
                "备择假设": alternative
            }
        )
    
    def two_sample_ttest(self, data1: Union[np.ndarray, List],
                        data2: Union[np.ndarray, List],
                        equal_var: bool = True,
                        alpha: float = 0.05,
                        alternative: str = 'two-sided') -> TestResult:
        """
        独立样本t检验 - 比较两独立样本均值是否有显著差异
        """
        data1 = self._validate_array(data1, "data1")
        data2 = self._validate_array(data2, "data2")
        data1 = self._remove_nan(data1)
        data2 = self._remove_nan(data2)
        
        if len(data1) < 2 or len(data2) < 2:
            raise ValueError("t检验需要每个样本至少2个观测值")
        
        statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var, 
                                           alternative=alternative)
        reject_null = p_value < alpha
        
        # 计算效应量 (Cohen's d)
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        
        if equal_var:
            s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            effect_size = (mean1 - mean2) / pooled_std
            df = n1 + n2 - 2
        else:
            s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
            effect_size = (mean1 - mean2) / np.sqrt((s1**2 + s2**2) / 2)
            s1_sq_n1 = s1**2 / n1
            s2_sq_n2 = s2**2 / n2
            df = (s1_sq_n1 + s2_sq_n2)**2 / (s1_sq_n1**2/(n1-1) + s2_sq_n2**2/(n2-1))
        
        test_name = "独立样本t检验" if equal_var else "Welch t检验"
        
        return TestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=df,
            effect_size=effect_size,
            additional_info={
                "样本1大小": n1,
                "样本2大小": n2,
                "样本1均值": mean1,
                "样本2均值": mean2,
                "方差齐性假设": equal_var,
                "备择假设": alternative
            }
        )
    
    def paired_ttest(self, data1: Union[np.ndarray, List],
                    data2: Union[np.ndarray, List],
                    alpha: float = 0.05,
                    alternative: str = 'two-sided') -> TestResult:
        """
        配对t检验 - 比较配对样本的均值差异
        """
        data1 = self._validate_array(data1, "data1")
        data2 = self._validate_array(data2, "data2")
        
        if len(data1) != len(data2):
            raise ValueError("配对t检验要求两组数据长度相同")
        
        # 移除配对的NaN值
        data1, data2 = self._remove_nan((data1, data2), paired=True)
        
        if len(data1) < 2:
            raise ValueError("配对t检验需要至少2对观测值")
        
        statistic, p_value = stats.ttest_rel(data1, data2, alternative=alternative)
        reject_null = p_value < alpha
        
        # 计算差值和效应量
        diff = data1 - data2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        effect_size = mean_diff / std_diff
        
        n = len(diff)
        df = n - 1
        
        return TestResult(
            test_name="配对t检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=df,
            effect_size=effect_size,
            additional_info={
                "配对数": n,
                "均值差": mean_diff,
                "差值标准差": std_diff,
                "备择假设": alternative
            }
        )
    
    def mann_whitney_u(self, data1: Union[np.ndarray, List],
                      data2: Union[np.ndarray, List],
                      alpha: float = 0.05,
                      alternative: str = 'two-sided') -> TestResult:
        """
        Mann-Whitney U检验 - 比较两独立样本的分布位置差异（非参数）
        """
        data1 = self._validate_array(data1, "data1")
        data2 = self._validate_array(data2, "data2")
        data1 = self._remove_nan(data1)
        data2 = self._remove_nan(data2)
        
        if len(data1) < 1 or len(data2) < 1:
            raise ValueError("Mann-Whitney U检验需要每个样本至少1个观测值")
        
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
        reject_null = p_value < alpha
        
        # 计算效应量
        n1, n2 = len(data1), len(data2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return TestResult(
            test_name="Mann-Whitney U检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            effect_size=effect_size,
            additional_info={
                "样本1大小": n1,
                "样本2大小": n2,
                "样本1中位数": np.median(data1),
                "样本2中位数": np.median(data2),
                "备择假设": alternative
            }
        )
    
    def wilcoxon_signed_rank(self, data1: Union[np.ndarray, List],
                            data2: Optional[Union[np.ndarray, List]] = None,
                            alpha: float = 0.05,
                            alternative: str = 'two-sided') -> TestResult:
        """
        Wilcoxon符号秩检验 - 配对样本的中位数差异检验（非参数）
        """
        data1 = self._validate_array(data1, "data1")
        
        if data2 is not None:
            data2 = self._validate_array(data2, "data2")
            if len(data1) != len(data2):
                raise ValueError("Wilcoxon检验要求两组数据长度相同")
            data1, data2 = self._remove_nan((data1, data2), paired=True)
            diff = data1 - data2
        else:
            diff = self._remove_nan(data1)
        
        if len(diff) < 1:
            raise ValueError("Wilcoxon符号秩检验需要至少1个观测值")
        
        statistic, p_value = stats.wilcoxon(diff, alternative=alternative)
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="Wilcoxon符号秩检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            additional_info={
                "有效对数": len(diff),
                "中位数差": np.median(diff),
                "备择假设": alternative,
                "零值对数": np.sum(diff == 0)
            }
        )
    
    def one_way_anova(self, *groups: Union[np.ndarray, List],
                     alpha: float = 0.05) -> TestResult:
        """
        单因素方差分析 - 比较三组或以上独立样本均值差异
        """
        if len(groups) < 2:
            raise ValueError("方差分析需要至少2组数据")
        
        # 验证和清理数据
        clean_groups = []
        for i, group in enumerate(groups):
            group_array = self._validate_array(group, f"group_{i+1}")
            group_clean = self._remove_nan(group_array)
            if len(group_clean) < 2:
                raise ValueError(f"第{i+1}组需要至少2个观测值")
            clean_groups.append(group_clean)
        
        statistic, p_value = stats.f_oneway(*clean_groups)
        reject_null = p_value < alpha
        
        # 计算自由度和效应量
        k = len(clean_groups)
        n_total = sum(len(group) for group in clean_groups)
        df_between = k - 1
        df_within = n_total - k
        
        # 计算eta squared
        ss_total = sum(np.sum((group - np.mean(np.concatenate(clean_groups)))**2) 
                      for group in clean_groups)
        ss_within = sum(np.sum((group - np.mean(group))**2) for group in clean_groups)
        ss_between = ss_total - ss_within
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return TestResult(
            test_name="单因素方差分析",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=(df_between, df_within),
            effect_size=eta_squared,
            additional_info={
                "组数": k,
                "总样本量": n_total,
                "组大小": [len(group) for group in clean_groups],
                "组均值": [np.mean(group) for group in clean_groups],
                "eta²": eta_squared
            }
        )
    
    def kruskal_wallis(self, *groups: Union[np.ndarray, List],
                      alpha: float = 0.05) -> TestResult:
        """
        Kruskal-Wallis H检验 - 非参数多组比较
        """
        if len(groups) < 2:
            raise ValueError("Kruskal-Wallis检验需要至少2组数据")
        
        clean_groups = []
        for i, group in enumerate(groups):
            group_array = self._validate_array(group, f"group_{i+1}")
            group_clean = self._remove_nan(group_array)
            if len(group_clean) < 1:
                raise ValueError(f"第{i+1}组需要至少1个观测值")
            clean_groups.append(group_clean)
        
        statistic, p_value = stats.kruskal(*clean_groups)
        reject_null = p_value < alpha
        
        k = len(clean_groups)
        df = k - 1
        n_total = sum(len(group) for group in clean_groups)
        eta_squared = (statistic - k + 1) / (n_total - k) if n_total > k else 0
        
        return TestResult(
            test_name="Kruskal-Wallis H检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=df,
            effect_size=eta_squared,
            additional_info={
                "组数": k,
                "总样本量": n_total,
                "组大小": [len(group) for group in clean_groups],
                "组中位数": [np.median(group) for group in clean_groups]
            }
        ) 