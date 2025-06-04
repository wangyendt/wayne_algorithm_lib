"""
模型诊断与假设检验

实现各种用于回归等统计模型诊断的检验方法。
"""

import numpy as np
from typing import Union, List, Optional, Tuple
from scipy import stats
import warnings
from .base import BaseStatisticalTest, TestResult

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white, het_goldfeldquandt
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ModelDiagnostics(BaseStatisticalTest):
    """
    模型诊断检验类
    
    提供多种回归模型诊断的检验方法。
    注意：部分方法需要statsmodels库，如未安装会抛出异常。
    """
    
    def __init__(self):
        super().__init__()
        if not STATSMODELS_AVAILABLE:
            warnings.warn("statsmodels未安装，部分模型诊断检验方法不可用", UserWarning)
    
    def breusch_pagan_test(self, residuals: Union[np.ndarray, List],
                          exog: Union[np.ndarray, List],
                          alpha: float = 0.05) -> TestResult:
        """
        Breusch-Pagan异方差检验 - 检验回归残差的方差是否随预测变量变化
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用Breusch-Pagan检验")
        
        residuals = self._validate_array(residuals, "residuals")
        exog = self._validate_array(exog, "exog")
        
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        
        if len(residuals) != len(exog):
            raise ValueError("残差和解释变量的长度必须相同")
        
        # 移除包含NaN的行
        valid_mask = ~(np.isnan(residuals) | np.any(np.isnan(exog), axis=1))
        residuals_clean = residuals[valid_mask]
        exog_clean = exog[valid_mask]
        
        if len(residuals_clean) < exog_clean.shape[1] + 2:
            raise ValueError("有效样本量不足")
        
        # 进行Breusch-Pagan检验
        lm_stat, p_value, f_stat, f_p_value = het_breuschpagan(residuals_clean, exog_clean)
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="Breusch-Pagan异方差检验",
            statistic=lm_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=exog_clean.shape[1],
            additional_info={
                "样本量": len(residuals_clean),
                "解释变量数": exog_clean.shape[1],
                "LM统计量": lm_stat,
                "F统计量": f_stat,
                "F检验p值": f_p_value,
                "结论": "存在异方差" if reject_null else "方差齐性"
            }
        )
    
    def white_test(self, residuals: Union[np.ndarray, List],
                   exog: Union[np.ndarray, List],
                   alpha: float = 0.05) -> TestResult:
        """
        White异方差检验 - 无需指定形式的通用异方差检验
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用White检验")
        
        residuals = self._validate_array(residuals, "residuals")
        exog = self._validate_array(exog, "exog")
        
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        
        if len(residuals) != len(exog):
            raise ValueError("残差和解释变量的长度必须相同")
        
        # 移除包含NaN的行
        valid_mask = ~(np.isnan(residuals) | np.any(np.isnan(exog), axis=1))
        residuals_clean = residuals[valid_mask]
        exog_clean = exog[valid_mask]
        
        if len(residuals_clean) < exog_clean.shape[1] + 2:
            raise ValueError("有效样本量不足")
        
        # 进行White检验
        lm_stat, p_value, f_stat, f_p_value = het_white(residuals_clean, exog_clean)
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="White异方差检验",
            statistic=lm_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            additional_info={
                "样本量": len(residuals_clean),
                "解释变量数": exog_clean.shape[1],
                "LM统计量": lm_stat,
                "F统计量": f_stat,
                "F检验p值": f_p_value,
                "结论": "存在异方差" if reject_null else "方差齐性"
            }
        )
    
    def goldfeld_quandt_test(self, residuals: Union[np.ndarray, List],
                            exog: Union[np.ndarray, List],
                            split: Union[int, float] = 0.5,
                            alpha: float = 0.05) -> TestResult:
        """
        Goldfeld-Quandt异方差检验 - 比较两个子样本的方差差异
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用Goldfeld-Quandt检验")
        
        residuals = self._validate_array(residuals, "residuals")
        exog = self._validate_array(exog, "exog")
        
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        
        if len(residuals) != len(exog):
            raise ValueError("残差和解释变量的长度必须相同")
        
        # 移除包含NaN的行
        valid_mask = ~(np.isnan(residuals) | np.any(np.isnan(exog), axis=1))
        residuals_clean = residuals[valid_mask]
        exog_clean = exog[valid_mask]
        
        if len(residuals_clean) < 10:
            raise ValueError("样本量太小，无法进行Goldfeld-Quandt检验")
        
        # 进行Goldfeld-Quandt检验
        f_stat, p_value, ordering = het_goldfeldquandt(residuals_clean, exog_clean, split=split)
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="Goldfeld-Quandt异方差检验",
            statistic=f_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            additional_info={
                "样本量": len(residuals_clean),
                "分割比例": split,
                "F统计量": f_stat,
                "排序方式": ordering,
                "结论": "存在异方差" if reject_null else "方差齐性"
            }
        )
    
    def durbin_watson_test(self, residuals: Union[np.ndarray, List],
                          alpha: float = 0.05) -> TestResult:
        """
        Durbin-Watson检验 - 检验回归残差的一阶自相关
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用Durbin-Watson检验")
        
        residuals = self._validate_array(residuals, "residuals")
        residuals = self._remove_nan(residuals)
        
        if len(residuals) < 3:
            raise ValueError("Durbin-Watson检验需要至少3个观测值")
        
        # 计算Durbin-Watson统计量
        dw_stat = durbin_watson(residuals)
        
        # Durbin-Watson统计量的解释
        if dw_stat < 1.5:
            conclusion = "可能存在正自相关"
        elif dw_stat > 2.5:
            conclusion = "可能存在负自相关"
        else:
            conclusion = "无明显自相关"
        
        # 注意：Durbin-Watson检验没有标准的p值计算
        # 需要查表或使用特殊的临界值
        return TestResult(
            test_name="Durbin-Watson检验",
            statistic=dw_stat,
            p_value=None,  # DW检验通常不提供p值
            reject_null=None,  # 需要查表确定
            alpha=alpha,
            additional_info={
                "样本量": len(residuals),
                "DW统计量": dw_stat,
                "解释": conclusion,
                "参考": "DW≈2表示无自相关，DW<2表示正自相关，DW>2表示负自相关"
            }
        )
    
    def variance_inflation_factors(self, exog: Union[np.ndarray, List],
                                  add_constant: bool = True) -> TestResult:
        """
        方差膨胀因子(VIF)计算 - 诊断多重共线性
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用VIF诊断")
        
        exog = self._validate_array(exog, "exog")
        
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        
        # 移除包含NaN的行
        valid_mask = ~np.any(np.isnan(exog), axis=1)
        exog_clean = exog[valid_mask]
        
        if len(exog_clean) < exog_clean.shape[1] + 2:
            raise ValueError("样本量不足")
        
        # 如果需要，添加常数项
        if add_constant:
            exog_clean = sm.add_constant(exog_clean)
        
        # 计算VIF
        vif_values = []
        for i in range(exog_clean.shape[1]):
            vif = variance_inflation_factor(exog_clean, i)
            vif_values.append(vif)
        
        max_vif = max(vif_values)
        
        # VIF判断标准
        if max_vif > 10:
            multicollinearity = "严重多重共线性"
        elif max_vif > 5:
            multicollinearity = "中等多重共线性"
        else:
            multicollinearity = "无严重多重共线性"
        
        return TestResult(
            test_name="方差膨胀因子(VIF)诊断",
            statistic=max_vif,
            p_value=None,
            reject_null=max_vif > 10,  # 通常以VIF>10为判断标准
            alpha=None,
            additional_info={
                "样本量": len(exog_clean),
                "变量数": exog_clean.shape[1],
                "VIF值": vif_values,
                "最大VIF": max_vif,
                "诊断结果": multicollinearity,
                "解释": "VIF>10表示严重多重共线性，VIF>5表示中等多重共线性"
            }
        )
    
    def residual_normality_tests(self, residuals: Union[np.ndarray, List],
                                alpha: float = 0.05) -> TestResult:
        """
        残差正态性检验 - 综合多种正态性检验方法
        """
        residuals = self._validate_array(residuals, "residuals")
        residuals = self._remove_nan(residuals)
        
        if len(residuals) < 3:
            raise ValueError("正态性检验需要至少3个观测值")
        
        # 进行多种正态性检验
        tests_results = {}
        
        # Shapiro-Wilk检验
        if len(residuals) <= 5000:
            sw_stat, sw_p = stats.shapiro(residuals)
            tests_results['Shapiro-Wilk'] = {'statistic': sw_stat, 'p_value': sw_p}
        
        # Jarque-Bera检验
        n = len(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        jb_stat = (n / 6) * (skewness**2 + (kurtosis**2) / 4)
        jb_p = 1 - stats.chi2.cdf(jb_stat, df=2)
        tests_results['Jarque-Bera'] = {'statistic': jb_stat, 'p_value': jb_p}
        
        # D'Agostino-Pearson检验
        if len(residuals) >= 8:
            dp_stat, dp_p = stats.normaltest(residuals)
            tests_results["D'Agostino-Pearson"] = {'statistic': dp_stat, 'p_value': dp_p}
        
        # 计算综合判断
        p_values = [result['p_value'] for result in tests_results.values()]
        min_p_value = min(p_values)
        reject_null = min_p_value < alpha
        
        # 使用最保守的检验结果
        main_test = min(tests_results.items(), key=lambda x: x[1]['p_value'])
        
        return TestResult(
            test_name="残差正态性检验",
            statistic=main_test[1]['statistic'],
            p_value=main_test[1]['p_value'],
            reject_null=reject_null,
            alpha=alpha,
            additional_info={
                "样本量": n,
                "偏度": skewness,
                "峰度": kurtosis,
                "检验结果": tests_results,
                "主要检验": main_test[0],
                "结论": "残差不服从正态分布" if reject_null else "残差可能服从正态分布"
            }
        )
    
    def bartlett_test(self, *groups: Union[np.ndarray, List],
                     alpha: float = 0.05) -> TestResult:
        """
        Bartlett方差齐性检验 - 检验多个组的方差是否相等（需要正态假设）
        """
        if len(groups) < 2:
            raise ValueError("Bartlett检验需要至少2组数据")
        
        # 验证和清理数据
        clean_groups = []
        for i, group in enumerate(groups):
            group_array = self._validate_array(group, f"group_{i+1}")
            group_clean = self._remove_nan(group_array)
            if len(group_clean) < 2:
                raise ValueError(f"第{i+1}组需要至少2个观测值")
            clean_groups.append(group_clean)
        
        statistic, p_value = stats.bartlett(*clean_groups)
        reject_null = p_value < alpha
        
        # 计算各组方差
        group_vars = [np.var(group, ddof=1) for group in clean_groups]
        
        return TestResult(
            test_name="Bartlett方差齐性检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=len(groups) - 1,
            additional_info={
                "组数": len(groups),
                "组大小": [len(group) for group in clean_groups],
                "组方差": group_vars,
                "结论": "方差不齐" if reject_null else "方差齐性"
            }
        )
    
    def levene_test(self, *groups: Union[np.ndarray, List],
                   center: str = 'median',
                   alpha: float = 0.05) -> TestResult:
        """
        Levene方差齐性检验 - 对偏态分布更稳健的方差齐性检验
        """
        if len(groups) < 2:
            raise ValueError("Levene检验需要至少2组数据")
        
        # 验证和清理数据
        clean_groups = []
        for i, group in enumerate(groups):
            group_array = self._validate_array(group, f"group_{i+1}")
            group_clean = self._remove_nan(group_array)
            if len(group_clean) < 2:
                raise ValueError(f"第{i+1}组需要至少2个观测值")
            clean_groups.append(group_clean)
        
        statistic, p_value = stats.levene(*clean_groups, center=center)
        reject_null = p_value < alpha
        
        # 计算各组方差
        group_vars = [np.var(group, ddof=1) for group in clean_groups]
        
        return TestResult(
            test_name="Levene方差齐性检验",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=(len(groups) - 1, sum(len(g) for g in clean_groups) - len(groups)),
            additional_info={
                "组数": len(groups),
                "组大小": [len(group) for group in clean_groups],
                "组方差": group_vars,
                "中心化方法": center,
                "结论": "方差不齐" if reject_null else "方差齐性"
            }
        ) 