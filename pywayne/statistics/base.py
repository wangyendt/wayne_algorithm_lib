"""
统计检验基类

定义了所有统计检验方法的基础接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import inspect


@dataclass
class TestResult:
    """
    统计检验结果的标准数据结构
    
    Attributes:
        test_name: 检验方法名称
        statistic: 检验统计量
        p_value: p值
        reject_null: 是否拒绝原假设
        alpha: 显著性水平
        degrees_of_freedom: 自由度（如适用）
        effect_size: 效应量（如适用）
        confidence_interval: 置信区间（如适用）
        critical_value: 临界值（如适用）
        additional_info: 其他额外信息
    """
    test_name: str
    statistic: float
    p_value: float
    reject_null: bool
    alpha: float = 0.05
    degrees_of_freedom: Optional[Union[int, float]] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    critical_value: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """格式化输出检验结果"""
        result = f"{self.test_name} 检验结果:\n"
        result += f"  统计量: {self.statistic:.6f}\n"
        result += f"  p值: {self.p_value:.6f}\n"
        result += f"  显著性水平: {self.alpha}\n"
        result += f"  结论: {'拒绝原假设' if self.reject_null else '不能拒绝原假设'}\n"
        
        if self.degrees_of_freedom is not None:
            result += f"  自由度: {self.degrees_of_freedom}\n"
        if self.effect_size is not None:
            result += f"  效应量: {self.effect_size:.6f}\n"
        if self.confidence_interval is not None:
            result += f"  置信区间: [{self.confidence_interval[0]:.6f}, {self.confidence_interval[1]:.6f}]\n"
        if self.critical_value is not None:
            result += f"  临界值: {self.critical_value:.6f}\n"
        if self.additional_info:
            result += "  其他信息:\n"
            for key, value in self.additional_info.items():
                result += f"    {key}: {value}\n"
        
        return result


class BaseStatisticalTest(ABC):
    """
    统计检验基类
    
    所有统计检验方法都应该继承此类，并实现相应的检验方法。
    每个检验方法都应该返回TestResult对象。
    """
    
    def __init__(self):
        """初始化基类"""
        self._test_methods = self._get_test_methods()
    
    def _get_test_methods(self) -> List[str]:
        """获取所有检验方法名称"""
        methods = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith('_') and name not in ['list_tests', 'show_usage']:
                methods.append(name)
        return methods
    
    def list_tests(self) -> List[str]:
        """列出所有可用的检验方法"""
        return self._test_methods.copy()
    
    def show_usage(self, test_name: Optional[str] = None) -> None:
        """
        显示检验方法的使用说明
        
        Parameters:
            test_name: 检验方法名称，如果为None则显示所有方法
        """
        if test_name is None:
            print(f"{self.__class__.__name__} 包含的检验方法:")
            for method_name in self._test_methods:
                method = getattr(self, method_name)
                doc = method.__doc__ or "无文档说明"
                # 提取第一行作为简短描述
                first_line = doc.split('\n')[0].strip()
                print(f"  - {method_name}: {first_line}")
        else:
            if test_name not in self._test_methods:
                print(f"错误: 方法 '{test_name}' 不存在")
                return
            
            method = getattr(self, test_name)
            print(f"方法: {test_name}")
            print("=" * 50)
            if method.__doc__:
                print(method.__doc__)
            else:
                print("暂无文档说明")
            
            # 显示方法签名
            sig = inspect.signature(method)
            print(f"\n方法签名: {test_name}{sig}")
    
    @staticmethod
    def _validate_array(data: Union[np.ndarray, list], name: str = "data") -> np.ndarray:
        """
        验证并转换输入数据为numpy数组
        
        Parameters:
            data: 输入数据
            name: 数据名称（用于错误信息）
            
        Returns:
            转换后的numpy数组
            
        Raises:
            ValueError: 当数据无效时
        """
        if data is None:
            raise ValueError(f"{name} 不能为None")
        
        try:
            arr = np.asarray(data)
        except Exception as e:
            raise ValueError(f"无法将{name}转换为numpy数组: {e}")
        
        if arr.size == 0:
            raise ValueError(f"{name} 不能为空")
        
        return arr
    
    @staticmethod
    def _remove_nan(data: np.ndarray, paired: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        移除数据中的NaN值
        
        Parameters:
            data: 输入数据（可以是单个数组或数组元组）
            paired: 是否为配对数据（如果是，则同时移除所有数组中相同位置的NaN）
            
        Returns:
            清理后的数据
        """
        if isinstance(data, tuple):
            if paired:
                # 配对数据：找到任意数组中的NaN位置，然后同时移除
                nan_mask = np.zeros(len(data[0]), dtype=bool)
                for arr in data:
                    nan_mask |= np.isnan(arr)
                
                return tuple(arr[~nan_mask] for arr in data)
            else:
                # 非配对数据：分别移除每个数组的NaN
                return tuple(arr[~np.isnan(arr)] for arr in data)
        else:
            return data[~np.isnan(data)]
    
    @classmethod
    def get_class_description(cls) -> str:
        """获取类的描述信息"""
        return cls.__doc__ or f"{cls.__name__} 统计检验类" 