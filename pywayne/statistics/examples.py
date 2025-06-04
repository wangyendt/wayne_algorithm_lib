"""
统计检验库使用示例与测试

展示如何使用Wayne Algorithm Library的统计检验模块，包含完整的功能测试。
"""

import sys
import os
# 添加当前项目路径，确保可以导入本地的pywayne模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def run_basic_examples():
    """运行基本使用示例"""
    print("Wayne Algorithm Library - 统计检验模块示例")
    print("=" * 60)
    
    # 生成示例数据
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100)
    uniform_data = np.random.uniform(-2, 2, 100)
    data1 = np.random.normal(0, 1, 50)
    data2 = np.random.normal(0.5, 1, 50)
    
    # 1. 正态性检验示例
    print("\n1. 正态性检验示例")
    print("-" * 30)
    
    try:
        from pywayne.statistics.normality_tests import NormalityTests
        
        nt = NormalityTests()
        print("可用的正态性检验方法:")
        for method in nt.list_tests():
            print(f"  - {method}")
        
        # Shapiro-Wilk检验
        result = nt.shapiro_wilk(normal_data)
        print(f"\n正态分布数据的Shapiro-Wilk检验:")
        print(result)
        
        result = nt.shapiro_wilk(uniform_data)
        print(f"\n均匀分布数据的Shapiro-Wilk检验:")
        print(result)
        
    except Exception as e:
        print(f"正态性检验示例出错: {e}")
    
    # 2. 位置参数检验示例
    print("\n2. 位置参数检验示例")
    print("-" * 30)
    
    try:
        from pywayne.statistics.location_tests import LocationTests
        
        lt = LocationTests()
        
        # 独立样本t检验
        result = lt.two_sample_ttest(data1, data2)
        print("独立样本t检验:")
        print(result)
        
        # Mann-Whitney U检验
        result = lt.mann_whitney_u(data1, data2)
        print("\nMann-Whitney U检验:")
        print(result)
        
    except Exception as e:
        print(f"位置参数检验示例出错: {e}")
    
    # 3. 相关性检验示例
    print("\n3. 相关性检验示例")
    print("-" * 30)
    
    try:
        from pywayne.statistics.correlation_tests import CorrelationTests
        
        ct = CorrelationTests()
        
        # 生成相关数据
        x = np.random.normal(0, 1, 100)
        y = 0.7 * x + 0.3 * np.random.normal(0, 1, 100)
        
        # Pearson相关检验
        result = ct.pearson_correlation(x, y)
        print("Pearson相关检验:")
        print(result)
        
        # Spearman相关检验
        result = ct.spearman_correlation(x, y)
        print("\nSpearman相关检验:")
        print(result)
        
    except Exception as e:
        print(f"相关性检验示例出错: {e}")
    
    # 4. 时间序列检验示例
    print("\n4. 时间序列检验示例")
    print("-" * 30)
    
    try:
        from pywayne.statistics.time_series_tests import TimeSeriesTests
        
        tst = TimeSeriesTests()
        
        # 生成时间序列数据
        ts_data = np.cumsum(np.random.normal(0, 1, 100))  # 随机游走
        
        # 游程检验
        result = tst.runs_test(ts_data)
        print("游程检验:")
        print(result)
        
    except Exception as e:
        print(f"时间序列检验示例出错: {e}")
    
    # 5. 模型诊断示例
    print("\n5. 模型诊断示例")  
    print("-" * 30)
    
    try:
        from pywayne.statistics.model_diagnostics import ModelDiagnostics
        
        md = ModelDiagnostics()
        
        # 生成残差数据
        residuals = np.random.normal(0, 1, 100)
        
        # 残差正态性检验
        result = md.residual_normality_tests(residuals)
        print("残差正态性检验:")
        print(result)
        
        # Bartlett方差齐性检验
        group1 = np.random.normal(0, 1, 30)
        group2 = np.random.normal(0, 1.5, 30)  # 不同方差
        group3 = np.random.normal(0, 1, 30)
        
        result = md.bartlett_test(group1, group2, group3)
        print("\nBartlett方差齐性检验:")
        print(result)
        
    except Exception as e:
        print(f"模型诊断示例出错: {e}")


def show_usage_examples():
    """展示使用说明功能"""
    print("\n" + "=" * 60)
    print("使用说明功能示例")
    print("=" * 60)
    
    try:
        from pywayne.statistics import list_all_tests, show_test_usage
        
        # 列出所有检验方法
        print("\n所有可用的检验方法:")
        list_all_tests()
        
        # 显示特定方法的使用说明
        print("\n显示Shapiro-Wilk检验的详细说明:")
        show_test_usage("NormalityTests", "shapiro_wilk")
        
    except Exception as e:
        print(f"使用说明示例出错: {e}")


def test_all_functionality():
    """测试所有功能"""
    print("\n" + "=" * 60)
    print("全功能测试")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100)
    uniform_data = np.random.uniform(-2, 2, 100)
    data1 = np.random.normal(0, 1, 50)
    data2 = np.random.normal(0.5, 1, 50)
    
    # 1. 正态性检验 - 测试所有方法
    print("\n1. 正态性检验")
    print("-" * 30)
    
    try:
        from pywayne.statistics.normality_tests import NormalityTests
        
        nt = NormalityTests()
        methods = nt.list_tests()
        
        for method_name in methods:
            if method_name == 'get_class_description':
                continue
            
            try:
                method = getattr(nt, method_name)
                if method_name in ['kolmogorov_smirnov_two_sample']:
                    result = method(normal_data, uniform_data)
                elif method_name in ['chi_square_goodness_of_fit']:
                    # 生成分类数据
                    observed = np.array([20, 30, 25, 25])
                    expected = np.array([25, 25, 25, 25])
                    result = method(observed, expected)
                elif method_name == 'anderson_darling':
                    # Anderson-Darling 检验，确保数据格式正确
                    result = method(normal_data, dist='norm')
                else:
                    result = method(normal_data)
                
                # 处理p值可能为None的情况
                p_val = result.p_value
                if p_val is not None:
                    print(f"✓ {method_name}: p值={p_val:.6f}")
                else:
                    print(f"✓ {method_name}: 统计量={result.statistic:.6f} (无p值)")
            except Exception as e:
                print(f"✗ {method_name}: {e}")
        
    except Exception as e:
        print(f"✗ 正态性检验模块出错: {e}")
    
    # 2. 位置参数检验 - 测试所有方法
    print("\n2. 位置参数检验")
    print("-" * 30)
    
    try:
        from pywayne.statistics.location_tests import LocationTests
        
        lt = LocationTests()
        methods = lt.list_tests()
        
        for method_name in methods:
            if method_name == 'get_class_description':
                continue
                
            try:
                method = getattr(lt, method_name)
                if method_name in ['two_sample_ttest', 'mann_whitney_u']:
                    result = method(data1, data2)
                elif method_name in ['paired_ttest', 'wilcoxon_signed_rank']:
                    result = method(data1[:30], data2[:30])  # 配对数据
                elif method_name in ['one_way_anova', 'kruskal_wallis']:
                    group3 = np.random.normal(1, 1, 30)
                    result = method(data1[:30], data2[:30], group3)
                else:  # one_sample_ttest
                    result = method(data1)
                print(f"✓ {method_name}: p值={result.p_value:.6f}")
            except Exception as e:
                print(f"✗ {method_name}: {e}")
        
    except Exception as e:
        print(f"✗ 位置参数检验模块出错: {e}")
    
    # 3. 相关性检验 - 测试所有方法
    print("\n3. 相关性检验")
    print("-" * 30)
    
    try:
        from pywayne.statistics.correlation_tests import CorrelationTests
        
        ct = CorrelationTests()
        methods = ct.list_tests()
        
        x = np.random.normal(0, 1, 100)
        y = 0.7 * x + 0.3 * np.random.normal(0, 1, 100)
        
        for method_name in methods:
            if method_name == 'get_class_description':
                continue
                
            try:
                method = getattr(ct, method_name)
                if method_name in ['pearson_correlation', 'spearman_correlation', 'kendall_tau']:
                    result = method(x, y)
                elif method_name in ['chi_square_independence', 'fisher_exact']:
                    # 生成2x2列联表
                    contingency_table = np.array([[20, 10], [15, 25]])
                    result = method(contingency_table)
                elif method_name == 'mcnemar_test':
                    # 修复：生成正确的配对数据 - 2x2列联表格式
                    # 表示处理前后的变化：[[改进->改进, 改进->未改进], [未改进->改进, 未改进->未改进]]
                    mcnemar_table = np.array([[25, 10], [5, 20]])
                    result = method(mcnemar_table)
                print(f"✓ {method_name}: p值={result.p_value:.6f}")
            except Exception as e:
                print(f"✗ {method_name}: {e}")
        
    except Exception as e:
        print(f"✗ 相关性检验模块出错: {e}")
    
    # 4. 时间序列检验 - 测试所有方法
    print("\n4. 时间序列检验")
    print("-" * 30)
    
    try:
        from pywayne.statistics.time_series_tests import TimeSeriesTests
        
        tst = TimeSeriesTests()
        methods = tst.list_tests()
        
        ts_data = np.cumsum(np.random.normal(0, 1, 100))
        ts_data2 = np.cumsum(np.random.normal(0, 1, 100))
        
        for method_name in methods:
            if method_name == 'get_class_description':
                continue
                
            try:
                method = getattr(tst, method_name)
                if method_name in ['runs_test', 'augmented_dickey_fuller', 'kpss_test']:
                    result = method(ts_data)
                elif method_name == 'granger_causality_test':
                    # 修复：提供正确的两列数据格式，并使用更少的滞后数减少复杂度
                    ts_matrix = np.column_stack([ts_data[:50], ts_data2[:50]])  # 使用较短的序列
                    result = method(ts_matrix, maxlag=1)
                elif method_name == 'engle_granger_cointegration':
                    result = method(ts_data, ts_data2)
                elif method_name == 'ljung_box_test':
                    # 修复：使用return_df=True避免Series的布尔值问题
                    diff_data = np.array(np.diff(ts_data))
                    result = method(diff_data, lags=5, return_df=True)
                elif method_name == 'breusch_godfrey_test':
                    # 修复：不传递nlags参数，使用方法的默认值
                    diff_data = np.diff(ts_data)
                    result = method(diff_data)
                elif method_name == 'arch_test':
                    result = method(np.diff(ts_data))
                print(f"✓ {method_name}: p值={result.p_value:.6f}")
            except Exception as e:
                print(f"✗ {method_name}: {e}")
        
    except Exception as e:
        print(f"✗ 时间序列检验模块出错: {e}")
    
    # 5. 模型诊断 - 测试所有方法
    print("\n5. 模型诊断")
    print("-" * 30)
    
    try:
        from pywayne.statistics.model_diagnostics import ModelDiagnostics
        
        md = ModelDiagnostics()
        methods = md.list_tests()
        
        residuals = np.random.normal(0, 1, 100)
        group1 = np.random.normal(0, 1, 30)
        group2 = np.random.normal(0, 1.5, 30)
        group3 = np.random.normal(0, 1, 30)
        
        for method_name in methods:
            if method_name == 'get_class_description':
                continue
                
            try:
                method = getattr(md, method_name)
                if method_name == 'residual_normality_tests':
                    result = method(residuals)
                elif method_name == 'durbin_watson_test':
                    result = method(residuals)
                elif method_name in ['bartlett_test', 'levene_test']:
                    result = method(group1, group2, group3)
                elif method_name in ['breusch_pagan_test', 'white_test']:
                    # 修复：添加常数列到设计矩阵
                    X_base = np.random.normal(0, 1, (100, 2))
                    X_with_const = np.column_stack([np.ones(100), X_base])  # 添加常数列
                    y = X_base[:, 0] + 0.5 * X_base[:, 1] + residuals
                    result = method(y, X_with_const)
                elif method_name == 'goldfeld_quandt_test':
                    # Goldfeld-Quandt测试不一定需要常数列
                    X = np.random.normal(0, 1, (100, 2))
                    y = X[:, 0] + 0.5 * X[:, 1] + residuals
                    result = method(y, X)
                elif method_name == 'variance_inflation_factors':
                    X = np.random.normal(0, 1, (100, 3))
                    result = method(X)
                
                # 获取p值，如果没有p值则显示其他信息
                p_val = getattr(result, 'p_value', None)
                if p_val is not None:
                    print(f"✓ {method_name}: p值={p_val:.6f}")
                else:
                    # 对于没有p值的测试（如VIF），显示统计量或其他信息
                    stat_val = getattr(result, 'statistic', None)
                    if stat_val is not None:
                        if isinstance(stat_val, (list, np.ndarray)):
                            print(f"✓ {method_name}: VIF值={[f'{v:.3f}' for v in stat_val[:3]]}")
                        else:
                            print(f"✓ {method_name}: 统计量={stat_val:.6f}")
                    else:
                        print(f"✓ {method_name}: 测试完成")
            except Exception as e:
                print(f"✗ {method_name}: {e}")
        
    except Exception as e:
        print(f"✗ 模型诊断模块出错: {e}")


def comprehensive_demo():
    """综合演示"""
    print("\n" + "=" * 60)
    print("综合使用演示")
    print("=" * 60)
    
    try:
        from pywayne.statistics import NormalityTests, LocationTests, CorrelationTests
        
        # 生成测试数据
        np.random.seed(123)
        normal_data = np.random.normal(100, 15, 30)
        group1 = np.random.normal(100, 10, 25)
        group2 = np.random.normal(105, 10, 25)
        x = np.random.normal(0, 1, 50)
        y = 0.8 * x + 0.2 * np.random.normal(0, 1, 50)
        
        print(f"\n数据描述:")
        print(f"  正态样本 (n=30): 均值={np.mean(normal_data):.2f}, 标准差={np.std(normal_data, ddof=1):.2f}")
        print(f"  样本1 (n=25): 均值={np.mean(group1):.2f}, 标准差={np.std(group1, ddof=1):.2f}")
        print(f"  样本2 (n=25): 均值={np.mean(group2):.2f}, 标准差={np.std(group2, ddof=1):.2f}")
        
        # 1. 正态性检验
        nt = NormalityTests()
        norm_result = nt.shapiro_wilk(normal_data)
        print(f"\n正态性检验 (Shapiro-Wilk):")
        print(f"  H0: 数据来自正态分布")
        print(f"  统计量: {norm_result.statistic:.4f}")
        print(f"  p值: {norm_result.p_value:.6f}")
        print(f"  结论: {('拒绝' if norm_result.reject_null else '不拒绝')}H0 (α=0.05)")
        
        # 2. 两组比较
        lt = LocationTests()
        t_result = lt.two_sample_ttest(group1, group2)
        print(f"\n两样本t检验:")
        print(f"  H0: 两组均值相等")
        print(f"  统计量: {t_result.statistic:.4f}")
        print(f"  p值: {t_result.p_value:.6f}")
        print(f"  结论: {('拒绝' if t_result.reject_null else '不拒绝')}H0 (α=0.05)")
        
        # 3. 相关性分析
        ct = CorrelationTests()
        corr_result = ct.pearson_correlation(x, y)
        print(f"\n相关性分析 (Pearson):")
        print(f"  H0: 变量间无相关性")
        print(f"  相关系数: {corr_result.statistic:.4f}")
        print(f"  p值: {corr_result.p_value:.6f}")
        print(f"  结论: {('拒绝' if corr_result.reject_null else '不拒绝')}H0 (α=0.05)")
        
        print("✓ 综合演示完成")
        
    except Exception as e:
        print(f"✗ 综合演示失败: {e}")


if __name__ == "__main__":
    print("Wayne Algorithm Library - 统计检验模块完整示例与测试")
    print("=" * 80)
    
    # 运行所有功能
    run_basic_examples()
    show_usage_examples()
    test_all_functionality()
    comprehensive_demo()
    
    print("\n" + "=" * 80)
    print("所有示例和测试完成！")
    print("=" * 80)
