"""
Wayne Algorithm Library - Statistics Module

提供各种统计检验方法的封装，具有统一的输入输出接口。

所有检验方法都继承自BaseStatisticalTest基类，具有以下特性：
1. 统一的输入格式（np.ndarray）
2. 统一的输出格式（TestResult对象）
3. 完整的文档和使用说明
4. 易于使用的API

主要模块：
- normality_tests: 分布拟合与正态性检验
- location_tests: 位置参数检验（均值/中位数比较）
- correlation_tests: 相关性与独立性检验
- time_series_tests: 时间序列分析检验
- model_diagnostics: 模型诊断检验
"""

from .base import BaseStatisticalTest, TestResult
from .normality_tests import NormalityTests
from .location_tests import LocationTests
from .correlation_tests import CorrelationTests
from .time_series_tests import TimeSeriesTests
from .model_diagnostics import ModelDiagnostics
from .utils import list_all_tests, show_test_usage

__all__ = [
    'BaseStatisticalTest',
    'TestResult', 
    'NormalityTests',
    'LocationTests',
    'CorrelationTests',
    'TimeSeriesTests',
    'ModelDiagnostics',
    'list_all_tests',
    'show_test_usage'
]
