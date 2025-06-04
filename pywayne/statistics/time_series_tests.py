"""
时间序列分析检验

实现各种用于时间序列分析的统计检验方法。
"""

import numpy as np
from typing import Union, List, Optional, Tuple
from scipy import stats
import warnings
from .base import BaseStatisticalTest, TestResult

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss, coint, grangercausalitytests
    from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey, het_arch
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class TimeSeriesTests(BaseStatisticalTest):
    """
    时间序列分析检验类
    
    提供多种时间序列特性检验的方法。
    注意：部分方法需要statsmodels库，如未安装会抛出异常。
    """
    
    def __init__(self):
        super().__init__()
        if not STATSMODELS_AVAILABLE:
            warnings.warn("statsmodels未安装，部分时间序列检验方法不可用", UserWarning)
    
    def augmented_dickey_fuller(self, data: Union[np.ndarray, List],
                               maxlag: Optional[int] = None,
                               regression: str = 'c',
                               alpha: float = 0.05) -> TestResult:
        """
        ADF单位根检验 - 检验时间序列是否存在单位根（非平稳）
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用ADF检验")
        
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 3:
            raise ValueError("ADF检验需要至少3个观测值")
        
        # 进行ADF检验
        result = adfuller(data, maxlag=maxlag, regression=regression, autolag='AIC')
        adf_stat, p_value, lags_used, nobs, critical_values, icbest = result
        
        reject_null = p_value < alpha
        
        # 根据显著性水平确定临界值
        if alpha == 0.01:
            critical_key = '1%'
        elif alpha == 0.05:
            critical_key = '5%'
        elif alpha == 0.10:
            critical_key = '10%'
        else:
            # 使用最接近的临界值
            critical_key = '5%'
        
        critical_value = critical_values[critical_key]
        
        return TestResult(
            test_name="ADF单位根检验",
            statistic=adf_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            critical_value=critical_value,
            additional_info={
                "样本量": len(data),
                "使用滞后阶数": lags_used,
                "有效观测数": nobs,
                "临界值": dict(critical_values),
                "回归类型": regression,
                "AIC值": icbest,
                "结论": "时间序列平稳" if reject_null else "时间序列非平稳（存在单位根）"
            }
        )
    
    def kpss_test(self, data: Union[np.ndarray, List],
                  regression: str = 'c',
                  nlags: str = 'auto',
                  alpha: float = 0.05) -> TestResult:
        """
        KPSS平稳性检验 - 检验时间序列是否平稳
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用KPSS检验")
        
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 3:
            raise ValueError("KPSS检验需要至少3个观测值")
        
        # 进行KPSS检验
        kpss_stat, p_value, lags_used, critical_values = kpss(data, regression=regression, nlags=nlags)
        
        reject_null = p_value < alpha
        
        # 根据显著性水平确定临界值
        alpha_str = f"{alpha:.0%}"
        if alpha_str in critical_values:
            critical_value = critical_values[alpha_str]
        else:
            critical_value = critical_values.get('5%', None)
        
        return TestResult(
            test_name="KPSS平稳性检验",
            statistic=kpss_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            critical_value=critical_value,
            additional_info={
                "样本量": len(data),
                "使用滞后阶数": lags_used,
                "临界值": dict(critical_values),
                "回归类型": regression,
                "结论": "时间序列非平稳" if reject_null else "时间序列平稳"
            }
        )
    
    def ljung_box_test(self, data: Union[np.ndarray, List],
                      lags: Optional[int] = None,
                      alpha: float = 0.05,
                      return_df: bool = False) -> TestResult:
        """
        Ljung-Box Q检验 - 检验时间序列的自相关性
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用Ljung-Box检验")
        
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 3:
            raise ValueError("Ljung-Box检验需要至少3个观测值")
        
        if lags is None:
            lags = min(10, len(data) // 4)
        
        # 进行Ljung-Box检验
        result = acorr_ljungbox(data, lags=lags, return_df=return_df)
        
        if return_df:
            # 返回DataFrame格式
            lb_stat = result['lb_stat'].iloc[-1]
            p_value = result['lb_pvalue'].iloc[-1]
        else:
            # 返回Series格式
            lb_stat = result['lb_stat']
            p_value = result['lb_pvalue']
            if isinstance(lb_stat, (list, np.ndarray)):
                lb_stat = lb_stat[-1]
                p_value = p_value[-1]
        
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="Ljung-Box Q检验",
            statistic=lb_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=lags,
            additional_info={
                "样本量": len(data),
                "滞后阶数": lags,
                "结论": "存在自相关" if reject_null else "无显著自相关"
            }
        )
    
    def granger_causality_test(self, data: Union[np.ndarray, List],
                              maxlag: int = 4,
                              alpha: float = 0.05,
                              addconst: bool = True) -> TestResult:
        """
        Granger因果检验 - 检验两个时间序列间的因果关系
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用Granger因果检验")
        
        data = self._validate_array(data, "data")
        
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Granger因果检验需要两列时间序列数据")
        
        if len(data) < maxlag + 2:
            raise ValueError(f"样本量不足，需要至少{maxlag + 2}个观测值")
        
        # 移除包含NaN的行
        data = data[~np.any(np.isnan(data), axis=1)]
        
        if len(data) < maxlag + 2:
            raise ValueError("移除NaN后样本量不足")
        
        # 进行Granger因果检验
        result = grangercausalitytests(data, maxlag=maxlag, addconst=addconst, verbose=False)
        
        # 修复：正确解析grangercausalitytests的返回结果
        # 使用第一个滞后阶数的结果（简化处理）
        first_lag = min(result.keys())
        test_results = result[first_lag]
        
        # 提取F统计量和p值 - test_results是一个元组：(test_results_dict, regression_results)
        f_test_result = test_results[0]['ssr_ftest']  # F统计量检验结果
        f_stat = f_test_result[0]  # F统计量
        p_value = f_test_result[1]  # p值
        df_num = f_test_result[2]  # 分子自由度
        df_den = f_test_result[3]  # 分母自由度
        
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="Granger因果检验",
            statistic=f_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=(df_num, df_den),
            additional_info={
                "样本量": len(data),
                "滞后阶数": first_lag,
                "最大滞后阶数": maxlag,
                "F统计量": f_stat,
                "结论": "存在Granger因果关系" if reject_null else "不存在Granger因果关系"
            }
        )
    
    def engle_granger_cointegration(self, y: Union[np.ndarray, List],
                                   x: Union[np.ndarray, List],
                                   trend: str = 'c',
                                   alpha: float = 0.05) -> TestResult:
        """
        Engle-Granger协整检验 - 检验两个时间序列是否存在协整关系
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用协整检验")
        
        y = self._validate_array(y, "y")
        x = self._validate_array(x, "x")
        
        if len(y) != len(x):
            raise ValueError("两个时间序列长度必须相同")
        
        # 移除配对的NaN值
        y, x = self._remove_nan((y, x), paired=True)
        
        if len(y) < 3:
            raise ValueError("协整检验需要至少3个观测值")
        
        # 进行协整检验
        coint_t, p_value, critical_values = coint(y, x, trend=trend)
        
        reject_null = p_value < alpha
        
        # 根据显著性水平确定临界值
        if alpha == 0.01:
            critical_key = 0
        elif alpha == 0.05:
            critical_key = 1
        elif alpha == 0.10:
            critical_key = 2
        else:
            critical_key = 1  # 默认使用5%水平
        
        critical_value = critical_values[critical_key]
        
        return TestResult(
            test_name="Engle-Granger协整检验",
            statistic=coint_t,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            critical_value=critical_value,
            additional_info={
                "样本量": len(y),
                "趋势项": trend,
                "临界值": {"1%": critical_values[0], "5%": critical_values[1], "10%": critical_values[2]},
                "结论": "存在协整关系" if reject_null else "不存在协整关系"
            }
        )
    
    def arch_test(self, residuals: Union[np.ndarray, List],
                 nlags: int = 4,
                 alpha: float = 0.05) -> TestResult:
        """
        ARCH效应检验 - 检验时间序列残差中是否存在异方差
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用ARCH检验")
        
        residuals = self._validate_array(residuals, "residuals")
        residuals = self._remove_nan(residuals)
        
        if len(residuals) < nlags + 2:
            raise ValueError(f"ARCH检验需要至少{nlags + 2}个观测值")
        
        # 进行ARCH检验
        lm_stat, p_value, f_stat, f_p_value = het_arch(residuals, nlags=nlags)
        
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="ARCH效应检验",
            statistic=lm_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=nlags,
            additional_info={
                "样本量": len(residuals),
                "滞后阶数": nlags,
                "LM统计量": lm_stat,
                "F统计量": f_stat,
                "F检验p值": f_p_value,
                "结论": "存在ARCH效应" if reject_null else "无ARCH效应"
            }
        )
    
    def breusch_godfrey_test(self, residuals: Union[np.ndarray, List],
                            nlags: int = 4,
                            alpha: float = 0.05) -> TestResult:
        """
        Breusch-Godfrey检验 - 检验回归残差的高阶自相关
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("需要安装statsmodels库才能使用Breusch-Godfrey检验")
        
        from statsmodels.regression.linear_model import OLS
        
        residuals = self._validate_array(residuals, "residuals")
        residuals = self._remove_nan(residuals)
        
        if len(residuals) < nlags + 2:
            raise ValueError(f"Breusch-Godfrey检验需要至少{nlags + 2}个观测值")
        
        # 修复：创建一个简单的回归模型来生成符合要求的残差对象
        # 使用残差本身作为因变量，常数项作为自变量进行虚拟回归
        X = np.ones((len(residuals), 1))  # 常数项
        y = residuals  # 残差作为因变量
        
        # 进行OLS回归
        ols_model = OLS(y, X).fit()
        
        # 使用OLS结果调用Breusch-Godfrey检验
        lm_stat, p_value, f_stat, f_p_value = acorr_breusch_godfrey(ols_model, nlags)
        
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="Breusch-Godfrey自相关检验",
            statistic=lm_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            degrees_of_freedom=nlags,
            additional_info={
                "样本量": len(residuals),
                "滞后阶数": nlags,
                "LM统计量": lm_stat,
                "F统计量": f_stat,
                "F检验p值": f_p_value,
                "结论": "存在自相关" if reject_null else "无显著自相关"
            }
        )
    
    def runs_test(self, data: Union[np.ndarray, List],
                 cutoff: str = 'median',
                 alpha: float = 0.05) -> TestResult:
        """
        游程检验 - 检验序列的随机性
        """
        data = self._validate_array(data, "data")
        data = self._remove_nan(data)
        
        if len(data) < 3:
            raise ValueError("游程检验需要至少3个观测值")
        
        # 确定分割点
        if cutoff == 'median':
            cut_value = np.median(data)
        elif cutoff == 'mean':
            cut_value = np.mean(data)
        elif isinstance(cutoff, (int, float)):
            cut_value = cutoff
        else:
            raise ValueError("cutoff必须是'median', 'mean'或数值")
        
        # 转换为二进制序列
        binary_seq = (data > cut_value).astype(int)
        
        # 计算游程数
        runs = 1
        for i in range(1, len(binary_seq)):
            if binary_seq[i] != binary_seq[i-1]:
                runs += 1
        
        # 计算正负符号个数
        n1 = np.sum(binary_seq == 1)
        n2 = np.sum(binary_seq == 0)
        n = n1 + n2
        
        if n1 == 0 or n2 == 0:
            return TestResult(
                test_name="游程检验",
                statistic=runs,
                p_value=1.0,
                reject_null=False,
                alpha=alpha,
                additional_info={
                    "样本量": n,
                    "游程数": runs,
                    "分割值": cut_value,
                    "注释": "所有值都在分割点的同一侧，无法进行检验"
                }
            )
        
        # 计算期望游程数和方差
        expected_runs = (2 * n1 * n2) / n + 1
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
        
        # 正态近似（连续性校正）
        if runs > expected_runs:
            z_stat = (runs - 0.5 - expected_runs) / np.sqrt(var_runs)
        else:
            z_stat = (runs + 0.5 - expected_runs) / np.sqrt(var_runs)
        
        # 双尾检验
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        reject_null = p_value < alpha
        
        return TestResult(
            test_name="游程检验",
            statistic=z_stat,
            p_value=p_value,
            reject_null=reject_null,
            alpha=alpha,
            additional_info={
                "样本量": n,
                "游程数": runs,
                "期望游程数": expected_runs,
                "游程方差": var_runs,
                "分割值": cut_value,
                "高于分割值数量": n1,
                "低于分割值数量": n2,
                "结论": "序列非随机" if reject_null else "序列随机"
            }
        ) 