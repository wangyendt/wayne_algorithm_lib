"""
分布拟合与正态性检验

实现各种用于检验数据分布形态的统计检验方法。
"""

import numpy as np
from typing import Union, List, Optional, Tuple
from scipy import stats
import warnings
from .base import BaseStatisticalTest, TestResult


class NormalityTests(BaseStatisticalTest):
    """
    分布拟合与正态性检验类
    
    提供多种检验数据分布形态的方法，包括正态性检验和拟合优度检验。
    
    主要方法：
    - shapiro_wilk: Shapiro-Wilk正态性检验
    - kolmogorov_smirnov_one_sample: 单样本K-S检验
    - kolmogorov_smirnov_two_sample: 双样本K-S检验
    - anderson_darling: Anderson-Darling检验
    - dagostino_pearson: D'Agostino-Pearson K²检验
    - jarque_bera: Jarque-Bera检验
    - chi_square_goodness_of_fit: 卡方拟合优度检验
    """
    
    def __init__(self):
        super().__init__()
    
    def shapiro_wilk(self, data: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult:
        """
        Shapiro-Wilk正态性检验
        
        检验单一样本数据是否来自正态分布。适用于样本量在3-5000范围内。
        
        Parameters:
            data: 待检验的数据，一维数组
            alpha: 显著性水平，默认0.05
            
        Returns:
            TestResult: 检验结果
            
        原假设 H0: 数据来自正态分布
        备择假设 H1: 数据不来自正态分布
        """
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 3:
            raise ValueError("Shapiro-Wilk检验需要至少3个观测值")
        if len(data) > 5000:
            warnings.warn("样本量>5000时，W统计量准确但p值可能不可靠", UserWarning)
        
        statistic, p_value = stats.shapiro(data)
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="Shapiro-Wilk正态性检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            additional_info={
                "样本量": len(data),
                "解释": "W统计量越接近1，数据越接近正态分布"
            }
        )
    
    def kolmogorov_smirnov_one_sample(self, data: Union[np.ndarray, List], 
                                     cdf: str = 'norm', args: Tuple = (), 
                                     alpha: float = 0.05) -> TestResult:
        """
        单样本Kolmogorov-Smirnov检验
        
        比较单一样本的经验分布函数与给定理论分布是否一致。
        
        Parameters:
            data: 待检验的数据，一维数组
            cdf: 理论分布名称，默认'norm'（正态分布）
            args: 分布参数，默认为空（使用样本估计）
            alpha: 显著性水平，默认0.05
            
        Returns:
            TestResult: 检验结果
            
        原假设 H0: 数据来自指定的理论分布
        备择假设 H1: 数据不来自指定的理论分布
        """
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 2:
            raise ValueError("K-S检验需要至少2个观测值")
        
        # 如果没有提供分布参数，则使用样本估计
        if not args:
            if cdf == 'norm':
                args = (np.mean(data), np.std(data, ddof=1))
            else:
                # 对于其他分布，可以使用MLE估计
                dist = getattr(stats, cdf)
                args = dist.fit(data)
        
        statistic, p_value = stats.kstest(data, cdf, args=args)
        reject_null = p_value < alpha
        
        return TestResult(
            test_name=f"K-S单样本检验 ({cdf}分布)",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            additional_info={
                "样本量": len(data),
                "理论分布": cdf,
                "分布参数": args,
                "解释": "D统计量为经验分布函数与理论分布函数的最大差异"
            }
        )
    
    def kolmogorov_smirnov_two_sample(self, data1: Union[np.ndarray, List],
                                     data2: Union[np.ndarray, List],
                                     alternative: str = 'two-sided',
                                     alpha: float = 0.05) -> TestResult:
        """
        双样本Kolmogorov-Smirnov检验
        
        检验两独立样本是否来自同一分布。
        
        Parameters:
            data1: 第一个样本数据
            data2: 第二个样本数据  
            alternative: 备择假设类型，可选'two-sided', 'less', 'greater'
            alpha: 显著性水平，默认0.05
            
        Returns:
            TestResult: 检验结果
            
        原假设 H0: 两样本来自同一分布
        备择假设 H1: 两样本来自不同分布
        """
        data1 = self._validate_array(data1, "data1")
        data2 = self._validate_array(data2, "data2")
        data1 = self._remove_nan(data1)
        data2 = self._remove_nan(data2)
        
        if len(data1) < 2 or len(data2) < 2:
            raise ValueError("K-S双样本检验需要每个样本至少2个观测值")
        
        statistic, p_value = stats.ks_2samp(data1, data2, alternative=alternative)
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="K-S双样本检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            additional_info={
                "样本1大小": len(data1),
                "样本2大小": len(data2),
                "备择假设": alternative,
                "解释": "D统计量为两个经验分布函数的最大差异"
            }
        )
    
    def anderson_darling(self, data: Union[np.ndarray, List], 
                        dist: str = 'norm', alpha: float = 0.05) -> TestResult:
        """
        Anderson-Darling检验
        
        正态性检验或其他分布拟合优度检验，比K-S检验对尾部更敏感。
        
        Parameters:
            data: 待检验的数据，一维数组
            dist: 分布类型，可选'norm', 'expon', 'gumbel', 'extreme1', 'logistic'
            alpha: 显著性水平，默认0.05
            
        Returns:
            TestResult: 检验结果
            
        原假设 H0: 数据来自指定分布
        备择假设 H1: 数据不来自指定分布
        """
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 8:
            raise ValueError("Anderson-Darling检验需要至少8个观测值")
        
        result = stats.anderson(data, dist=dist)
        statistic = result.statistic
        critical_values = result.critical_values
        significance_levels = result.significance_level
        
        # 找到对应的临界值
        alpha_percent = alpha * 100
        if alpha_percent in significance_levels:
            idx = list(significance_levels).index(alpha_percent)
            critical_value = critical_values[idx]
            reject_null = statistic > critical_value
        else:
            # 插值估计临界值
            if alpha_percent < significance_levels[0]:
                critical_value = critical_values[0]
                reject_null = statistic > critical_value
            elif alpha_percent > significance_levels[-1]:
                critical_value = critical_values[-1]
                reject_null = statistic > critical_value
            else:
                # 线性插值
                for i in range(len(significance_levels) - 1):
                    if significance_levels[i] <= alpha_percent <= significance_levels[i+1]:
                        ratio = (alpha_percent - significance_levels[i]) / (significance_levels[i+1] - significance_levels[i])
                        critical_value = critical_values[i] + ratio * (critical_values[i+1] - critical_values[i])
                        reject_null = statistic > critical_value
                        break
        
        return TestResult(
            test_name=f"Anderson-Darling检验 ({dist}分布)",
            statistic=statistic,
            p_value=None,  # Anderson-Darling检验不直接提供p值
            reject_null=reject_null,
            alpha=alpha,
            critical_value=critical_value,
            additional_info={
                "样本量": len(data),
                "分布类型": dist,
                "临界值": dict(zip(significance_levels, critical_values)),
                "解释": "统计量越大，偏离指定分布越明显"
            }
        )
    
    def dagostino_pearson(self, data: Union[np.ndarray, List], 
                         alpha: float = 0.05) -> TestResult:
        """
        D'Agostino-Pearson K²正态性检验
        
        利用样本的偏度和峰度检验数据偏离正态的程度。
        
        Parameters:
            data: 待检验的数据，一维数组
            alpha: 显著性水平，默认0.05
            
        Returns:
            TestResult: 检验结果
            
        原假设 H0: 数据来自正态分布
        备择假设 H1: 数据不来自正态分布
        """
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 8:
            raise ValueError("D'Agostino-Pearson检验需要至少8个观测值")
        
        statistic, p_value = stats.normaltest(data)
        reject_null = p_value < alpha
        
        # 计算偏度和峰度
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        return TestResult(
            test_name="D'Agostino-Pearson K²正态性检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=2,
            additional_info={
                "样本量": len(data),
                "偏度": skewness,
                "峰度": kurtosis,
                "解释": "基于偏度和峰度的联合检验，K²统计量服从卡方分布(df=2)"
            }
        )
    
    def jarque_bera(self, data: Union[np.ndarray, List], alpha: float = 0.05) -> TestResult:
        """
        Jarque-Bera正态性检验
        
        基于偏度和峰度的联合检验，常用于回归残差正态性检验。
        
        Parameters:
            data: 待检验的数据，一维数组
            alpha: 显著性水平，默认0.05
            
        Returns:
            TestResult: 检验结果
            
        原假设 H0: 数据来自正态分布
        备择假设 H1: 数据不来自正态分布
        """
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 2:
            raise ValueError("Jarque-Bera检验需要至少2个观测值")
        
        # 计算偏度和峰度
        n = len(data)
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # JB统计量
        statistic = (n / 6) * (skewness**2 + (kurtosis**2) / 4)
        p_value = 1 - stats.chi2.cdf(statistic, df=2)
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="Jarque-Bera正态性检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=2,
            additional_info={
                "样本量": n,
                "偏度": skewness,
                "峰度": kurtosis,
                "解释": "JB = n/6 * (S² + K²/4)，其中S为偏度，K为峰度"
            }
        )
    
    def chi_square_goodness_of_fit(self, observed: Union[np.ndarray, List],
                                  expected: Optional[Union[np.ndarray, List]] = None,
                                  alpha: float = 0.05) -> TestResult:
        """
        卡方拟合优度检验
        
        检验分类数据的观察频数分布与期望分布是否有显著差异。
        
        Parameters:
            observed: 观察频数，一维数组
            expected: 期望频数，一维数组。如果为None，则假设均匀分布
            alpha: 显著性水平，默认0.05
            
        Returns:
            TestResult: 检验结果
            
        原假设 H0: 观察频数与期望频数无显著差异
        备择假设 H1: 观察频数与期望频数有显著差异
        """
        observed = self._validate_array(observed, "observed")
        
        if np.any(observed < 0):
            raise ValueError("观察频数不能为负数")
        
        if expected is None:
            # 均匀分布假设
            total = np.sum(observed)
            k = len(observed)
            expected = np.full(k, total / k)
        else:
            expected = self._validate_array(expected, "expected")
            if len(observed) != len(expected):
                raise ValueError("观察频数和期望频数长度必须相同")
            if np.any(expected <= 0):
                raise ValueError("期望频数必须大于0")
        
        # 检查期望频数是否满足要求（通常要求≥5）
        if np.any(expected < 5):
            warnings.warn("有期望频数小于5，检验结果可能不可靠", UserWarning)
        
        statistic, p_value = stats.chisquare(observed, expected)
        reject_null = p_value < alpha
        
        df = len(observed) - 1
        
        return TestResult(
            test_name="卡方拟合优度检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=df,
            additional_info={
                "类别数": len(observed),
                "观察频数": observed.tolist(),
                "期望频数": expected.tolist(),
                "总频数": np.sum(observed),
                "解释": "χ² = Σ[(Oi - Ei)² / Ei]，其中Oi为观察频数，Ei为期望频数"
            }
        ) 