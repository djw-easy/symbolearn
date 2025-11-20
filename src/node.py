import numpy as np
import numpy.ma as ma
from typing import Union, Optional, List
from joblib import wrap_non_picklable_objects
import random


class NodeContent(object):
    """Base class for the content of a SymbolicNode."""
    pass


class Variable(NodeContent):
    __slots__ = ['variable', '__name']
    def __init__(self, variable: int, name: str = None):
        self.variable = variable
        self.__name = name
        
    def __call__(self, X):
        return X[:, self.variable]
    
    @property
    def name(self):
        return self.__name if self.__name is not None else f'x{self.variable}'
    
    @property
    def degree(self):
        return 0

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.variable == other.variable
        return False


class DynamicAggregation(NodeContent):
    __slots__ = ['v_start', 'v_end', 'op_name', 'n_variables', 'valid_op']

    _op_map = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'median': np.median,
        'std': np.std, 
        'var': np.var,
        'sum': np.sum
    }

    def __init__(self, v_start: int, v_end: int, op_name: str, n_variables: int, valid_op: list = None):
        if n_variables < 2:
            raise ValueError('n_variables must be greater than 1')
        if op_name not in self._op_map:
            raise ValueError(f"Unsupported aggregation operator: {op_name}, valid options are: {list(self._op_map.keys())}")
        if v_start < 0 or v_end >= n_variables or v_end <= v_start:
            raise ValueError(f"Invalid band range: [{v_start}, {v_end}] for n_variables={n_variables}")
        if isinstance(valid_op, list) and valid_op:
            for op in valid_op:
                if op not in self._op_map:
                    raise ValueError(f"Unsupported aggregation operator: {op}, valid options are: {list(self._op_map.keys())}")
            if op_name not in self._op_map:
                raise ValueError(f"Unsupported aggregation operator: {op_name}, valid options are: {valid_op}")

        self.v_start = v_start
        self.v_end = v_end
        self.op_name = op_name
        self.n_variables = n_variables
        self.valid_op = valid_op

    def __call__(self, X: np.ndarray):
        op_func = self._op_map[self.op_name]
        band_slice = X[:, self.v_start: self.v_end + 1]
        return op_func(band_slice, axis=1)

    @property
    def degree(self):
        return 0

    @property
    def name(self):
        return f"{self.op_name}(v{self.v_start+1}-v{self.v_end+1})"

    def __eq__(self, other):
        if not isinstance(other, DynamicAggregation):
            return False
        return (
            self.v_start == other.v_start and
            self.v_end == other.v_end and
            self.op_name == other.op_name
        )


class Constant(NodeContent):
    __slots__ = ['value']
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, X):
        return self.value
    
    @property
    def name(self):
        return str(self.value)
    
    @property
    def degree(self):
        return 0

    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value
        return False


class Operator(NodeContent):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    degree : int
        The number of arguments that the ``function`` takes.

    """
    __slots__ = ['function', 'name', 'degree', 'elementwise']
    def __init__(self, function, name, degree, elementwise):
        self.function = function
        self.name = name
        self.degree = degree
        self.elementwise = elementwise

    def __call__(self, *args):
        try:
            return self.function(*args) if self.elementwise else float(self.function(args))
        except Exception:  # pragma: no cover
            return args[0]

    def __eq__(self, other):
        if isinstance(other, Operator):
            return self.name == other.name and self.degree == other.degree
        return False




"""
NumPy 保护函数 - 普通版本
提供数值稳定的数学运算函数，不处理 MaskedArray
"""
import numpy as np
from typing import Union


def _protected_addition(x1, x2):
    """加法闭包,处理溢出情况"""
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.add(x1, x2)
        # 检测溢出:结果是否在合理范围内
        safe_mask = np.isfinite(result) & (np.abs(result) < 1e10)
        return np.where(safe_mask, result, 0.)


def _protected_subtraction(x1, x2):
    """减法闭包,处理溢出情况"""
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.subtract(x1, x2)
        # 检测溢出:结果是否在合理范围内
        safe_mask = np.isfinite(result) & (np.abs(result) < 1e10)
        return np.where(safe_mask, result, 0.)


def _protected_multiplication(x1, x2):
    """乘法闭包,处理溢出情况"""
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        # 预先检查是否会溢出
        safe_mask = (np.abs(x1) < 1e10) & (np.abs(x2) < 1e10)
        # 对于可能溢出的情况,进一步检查乘积
        result = np.where(safe_mask, np.multiply(x1, x2), 0.)
        # 再次验证结果是否有限
        result = np.where(np.isfinite(result) & (np.abs(result) < 1e10), result, 0.)
        return result

def _protected_division(x1, x2):
    """除法闭包，处理零除数情况"""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        safe_mask = (np.abs(x2) > 0.0001) & (np.abs(x1 / x2) < 1e10)
        return np.where(safe_mask, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """平方根闭包，处理负数参数"""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """对数闭包，处理零和负数参数"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """倒数闭包，处理零参数"""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _protected_exp(x1):
    """指数闭包，限制大参数避免溢出"""
    with np.errstate(over='ignore', under='ignore'):
        clipped_x1 = np.clip(x1, a_min=None, a_max=700.)
        return np.exp(clipped_x1)


def _protected_expsq(x1):
    """高斯函数闭包，处理大参数避免溢出"""
    with np.errstate(over='ignore', under='ignore'):
        # 限制输入范围，防止指数部分过大
        clipped_x1 = np.clip(x1, a_min=-30., a_max=30.)
        return np.exp(-np.square(clipped_x1))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    计算数值稳定的 Sigmoid 函数
    
    对输入数组中的每个元素计算 sigmoid 值：sigmoid(x) = 1 / (1 + exp(-x))
    采用分段计算策略保证数值稳定性：
    - 当 x >= 0 时，使用 1 / (1 + exp(-x))
    - 当 x < 0 时，使用 exp(x) / (1 + exp(x))
    
    参数:
        x: 输入数组，支持任意维度的数值数组
    
    返回:
        与输入同形状的 sigmoid 计算结果
    
    示例:
        >>> x = np.array([-1, 0, 1])
        >>> _sigmoid(x)
        array([0.26894142, 0.5, 0.73105858])
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("输入必须是 np.ndarray")
    
    EXP_LOWER_BOUND = -88.0
    EXP_UPPER_BOUND = 88.0
    
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        x_clipped = np.clip(x, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
        pos_mask = (x_clipped >= 0)
        neg_mask = ~pos_mask
        
        result = np.empty_like(x, dtype=np.float64)
        
        # 正值部分
        result[pos_mask] = 1.0 / (1.0 + np.exp(-x_clipped[pos_mask]))
        
        # 负值部分
        z = np.exp(x_clipped[neg_mask])
        result[neg_mask] = z / (1.0 + z)
        
        # 处理极端值
        result[x <= EXP_LOWER_BOUND] = 0.0
        result[x >= EXP_UPPER_BOUND] = 1.0
        
        return result.astype(x.dtype)



def _softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus 激活函数
    
    f(x) = ln(1 + exp(x))
    ReLU 的平滑近似
    
    参数:
        x: 输入数组
    
    返回:
        Softplus 激活后的结果
    
    示例:
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> _softplus(x)
        array([0.12692801, 0.31326169, 0.69314718, 1.31326169, 2.12692801])
    """
    with np.errstate(over='ignore', under='ignore'):
        x_clipped = np.clip(x, -88, 88)
        return np.log(1.0 + np.exp(x_clipped))


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    计算数值稳定的 Softmax 函数
    
    沿指定轴对输入数组计算 softmax 值：
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    通过减去最大值保证数值稳定性，避免指数运算溢出
    
    参数:
        x: 输入数组，通常为二维矩阵（样本数×类别数）
        axis: 计算轴，默认为1（按行计算，每行和为1）
    
    返回:
        与输入同形状的 softmax 计算结果，每行（或列）的和为1
    
    示例:
        >>> x = np.array([[1, 2, 3], [1, 2, 1]])
        >>> _softmax(x, axis=1)
        array([[0.09003057, 0.24472847, 0.66524096],
               [0.21194156, 0.57611688, 0.21194156]])
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("输入必须是 np.ndarray")
    if axis not in (0, 1):
        raise ValueError("axis 必须是 0 或 1")
    
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        x_max = np.max(x, axis=axis, keepdims=True)
        x_shifted = x - x_max
        
        # 对极端值进行保护
        x_shifted = np.clip(x_shifted, -700, 700)
        
        exps = np.exp(x_shifted)
        sum_exps = np.sum(exps, axis=axis, keepdims=True)
        
        # 处理 sum_exps 为 0 的情况
        safe_sum = np.where(sum_exps == 0, 1.0, sum_exps)
        result = exps / safe_sum
        
        return result



add2 = Operator(function=_protected_addition, name='add', degree=2, elementwise=True)
sub2 = Operator(function=_protected_subtraction, name='sub', degree=2, elementwise=True)
mul2 = Operator(function=_protected_multiplication, name='mul', degree=2, elementwise=True)
div2 = Operator(function=_protected_division, name='div', degree=2, elementwise=True)
sqrt1 = Operator(function=_protected_sqrt, name='sqrt', degree=1, elementwise=True)
log1 = Operator(function=_protected_log, name='log', degree=1, elementwise=True)
neg1 = Operator(function=np.negative, name='neg', degree=1, elementwise=True)
inv1 = Operator(function=_protected_inverse, name='inv', degree=1, elementwise=True)
max2 = Operator(function=np.max, name='max', degree=1, elementwise=False)
abs1 = Operator(function=np.abs, name='abs', degree=1, elementwise=True)
maximum2 = Operator(function=np.maximum, name='maximum', degree=2, elementwise=True)
min2 = Operator(function=np.min, name='min', degree=1, elementwise=False)
minimum2 = Operator(function=np.minimum, name='minimum', degree=2, elementwise=True)
sin1 = Operator(function=np.sin, name='sin', degree=1, elementwise=True)
cos1 = Operator(function=np.cos, name='cos', degree=1, elementwise=True)
tan1 = Operator(function=np.tan, name='tan', degree=1, elementwise=True)
sinh1 = Operator(function=np.sinh, name='sinh', degree=1, elementwise=True)
cosh1 = Operator(function=np.cosh, name='cosh', degree=1, elementwise=True)
tanh1 = Operator(function=np.tanh, name='tanh', degree=1, elementwise=True)
exp1 = Operator(function=_protected_exp, name='exp', degree=1, elementwise=True)
expsq1 = Operator(function=_protected_expsq, name='expsq', degree=1, elementwise=True)

sigmoid = Operator(function=_sigmoid, name='sigmoid', degree=1, elementwise=True)
softplus = Operator(function=_softplus, name='softplus', degree=1, elementwise=True)
softmax = Operator(function=_softmax, name='softmax', degree=2, elementwise=True)

_operator_map = {
    '+': add2, 
    'add': add2, 
    '-': sub2,
    'sub': sub2,
    '*': mul2,
    'mul': mul2,
    '/': div2,
    'div': div2,
    'sqrt': sqrt1,
    'log': log1,
    'exp': exp1,
    'abs': abs1,
    'neg': neg1,
    'inv': inv1,
    'maximum': maximum2,
    'minimum': minimum2,
    'sin': sin1,
    'cos': cos1,
    'tan': tan1,
    'sinh': sinh1,
    'cosh': cosh1,
    'tanh': tanh1,
    'sigmoid': sigmoid, 
    'softmax': softmax,
    'expsq': expsq1,
    'softplus': softplus
}




