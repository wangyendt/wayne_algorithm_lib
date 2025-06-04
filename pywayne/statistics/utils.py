"""
统计检验工具函数

提供全局的使用说明和测试列表功能。
"""

from typing import Dict, List, Optional, Type
import importlib
from .base import BaseStatisticalTest


def get_all_test_classes() -> Dict[str, Type[BaseStatisticalTest]]:
    """
    获取所有检验类
    
    Returns:
        包含所有检验类的字典，键为类名，值为类对象
    """
    classes = {}
    
    # 导入所有检验类
    modules = [
        'normality_tests',
        'location_tests', 
        'correlation_tests',
        'time_series_tests',
        'model_diagnostics'
    ]
    
    for module_name in modules:
        try:
            module = importlib.import_module(f'.{module_name}', package='pywayne.statistics')
            # 获取模块中的所有类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseStatisticalTest) and 
                    attr != BaseStatisticalTest):
                    classes[attr_name] = attr
        except ImportError as e:
            print(f"警告: 无法导入模块 {module_name}: {e}")
    
    return classes


def list_all_tests() -> None:
    """
    列出所有可用的统计检验方法
    """
    print("Wayne Algorithm Library - 统计检验方法清单")
    print("=" * 60)
    
    classes = get_all_test_classes()
    
    for class_name, test_class in classes.items():
        print(f"\n{class_name}:")
        print(f"  描述: {test_class.get_class_description()}")
        
        # 创建实例以获取方法列表
        try:
            instance = test_class()
            methods = instance.list_tests()
            print(f"  包含检验方法 ({len(methods)} 个):")
            for method in methods:
                method_obj = getattr(instance, method)
                doc = method_obj.__doc__ or "无文档说明"
                first_line = doc.split('\n')[0].strip()
                print(f"    - {method}: {first_line}")
        except Exception as e:
            print(f"    错误: 无法创建实例 - {e}")


def show_test_usage(class_name: str, test_name: Optional[str] = None) -> None:
    """
    显示特定检验方法的使用说明
    
    Parameters:
        class_name: 检验类名称
        test_name: 检验方法名称，如果为None则显示该类的所有方法
    """
    classes = get_all_test_classes()
    
    if class_name not in classes:
        print(f"错误: 检验类 '{class_name}' 不存在")
        print(f"可用的检验类: {list(classes.keys())}")
        return
    
    test_class = classes[class_name]
    
    try:
        instance = test_class()
        instance.show_usage(test_name)
    except Exception as e:
        print(f"错误: 无法显示使用说明 - {e}")


def get_test_info(class_name: str, test_name: str) -> Optional[Dict]:
    """
    获取特定检验方法的详细信息
    
    Parameters:
        class_name: 检验类名称
        test_name: 检验方法名称
        
    Returns:
        包含检验方法信息的字典，如果不存在则返回None
    """
    classes = get_all_test_classes()
    
    if class_name not in classes:
        return None
    
    test_class = classes[class_name]
    
    try:
        instance = test_class()
        if test_name not in instance.list_tests():
            return None
        
        method = getattr(instance, test_name)
        return {
            'class_name': class_name,
            'test_name': test_name,
            'description': test_class.get_class_description(),
            'method_doc': method.__doc__,
            'signature': str(method.__annotations__) if hasattr(method, '__annotations__') else None
        }
    except Exception:
        return None


if __name__ == "__main__":
    # 测试代码
    list_all_tests() 