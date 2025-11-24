import numpy as np
import numpy.ma as ma
from typing import Union, Optional, List
from joblib import wrap_non_picklable_objects
from numba import vectorize, njit, float32, float64


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
    __slots__ = ['function', 'name', 'degree']
    def __init__(self, function, name, degree):
        self.function = function
        self.name = name
        self.degree = degree

    def __call__(self, *args):
        return self.function(*args)

    def __eq__(self, other):
        if isinstance(other, Operator):
            return self.name == other.name and self.degree == other.degree
        return False






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
    """高效的保护除法函数，适用于频繁调用场景"""
    # 直接使用numpy操作，避免不必要的类型检查和条件判断
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = np.divide(x1, x2)
        # 使用位运算&代替逻辑运算and，提高效率
        safe_mask = (np.abs(x2) > 1e-10) & np.isfinite(result) & (np.abs(result) < 1e10)
        return np.where(safe_mask, result, 1.0)


def _protected_sqrt(x1):
    """平方根闭包，处理负数参数"""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """对数闭包，处理零和负数参数"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """使用np.where处理标量和数组"""
    x1 = np.asarray(x1)  # 转换为numpy数组
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = np.where(np.abs(x1) > 1e-10, 1.0 / x1, 0.0)
        return result.item() if result.ndim == 0 else result


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
    EXP_LOWER_BOUND = -88.0
    EXP_UPPER_BOUND = 88.0
    
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        x_clipped = np.clip(x, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
        pos_mask = (x_clipped >= 0)
        neg_mask = ~pos_mask
        
        result = np.empty_like(x, dtype=np.float32)
        
        # 正值部分
        result[pos_mask] = 1.0 / (1.0 + np.exp(-x_clipped[pos_mask]))
        
        # 负值部分
        z = np.exp(x_clipped[neg_mask])
        result[neg_mask] = z / (1.0 + z)
        
        # 处理极端值
        result[x <= EXP_LOWER_BOUND] = 0.0
        result[x >= EXP_UPPER_BOUND] = 1.0
        
        return result.astype(np.float32)


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


# Softmax 的核心实现 (仅处理 2D 数组)
# 专门针对 float32 优化，也兼容 float64
@njit(fastmath=True)
def _softmax_2d_impl(x):
    rows, cols = x.shape
    result = np.empty_like(x)
    
    for i in range(rows):
        row_max = -np.inf
        for j in range(cols):
            if x[i, j] > row_max:
                row_max = x[i, j]
        
        row_sum = 0.0
        for j in range(cols):
            val = x[i, j] - row_max
            # 裁剪保护
            if val < -700.0: val = -700.0
            elif val > 700.0: val = 700.0
            
            e_val = np.exp(val)
            result[i, j] = e_val
            row_sum += e_val
        
        if row_sum == 0.0:
            row_sum = 1.0
            
        factor = 1.0 / row_sum
        for j in range(cols):
            result[i, j] *= factor
            
    return result

# Softmax 的 Python 包装器，处理标量和维度
def _softmax(x, axis=1):
    # 情况 1: 输入是标量 (Scalar)
    if np.ndim(x) == 0:
        # Softmax 对标量的结果数学上是 1.0
        # 如果 x 是 float32，返回 float32(1.0)
        return np.array(1.0, dtype=np.asarray(x).dtype)

    x_arr = np.asarray(x)
    
    # 情况 2: 输入是 1D 数组 (视为单样本) -> 转 2D 处理
    if x_arr.ndim == 1:
        x_2d = x_arr.reshape(1, -1)
        res = _softmax_2d_impl(x_2d)
        return res.reshape(-1) # 还原回 1D
    
    # 情况 3: 输入是 2D 数组 (标准情况)
    if x_arr.ndim == 2:
        return _softmax_2d_impl(x_arr)
    
    # 情况 4: 更高维，回退到 Numpy (或抛出异常，视需求定)
    # 这里为了安全回退到 numpy 实现
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        e_x = np.exp(x_arr - np.max(x_arr, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)



add2 = Operator(function=_protected_addition, name='add', degree=2)
sub2 = Operator(function=_protected_subtraction, name='sub', degree=2)
mul2 = Operator(function=_protected_multiplication, name='mul', degree=2)
div2 = Operator(function=_protected_division, name='div', degree=2)
sqrt1 = Operator(function=_protected_sqrt, name='sqrt', degree=1)
log1 = Operator(function=_protected_log, name='log', degree=1)
neg1 = Operator(function=np.negative, name='neg', degree=1)
inv1 = Operator(function=_protected_inverse, name='inv', degree=1)
abs1 = Operator(function=np.abs, name='abs', degree=1)
maximum2 = Operator(function=np.maximum, name='maximum', degree=2)
minimum2 = Operator(function=np.minimum, name='minimum', degree=2)
sin1 = Operator(function=np.sin, name='sin', degree=1)
cos1 = Operator(function=np.cos, name='cos', degree=1)
tan1 = Operator(function=np.tan, name='tan', degree=1)
sinh1 = Operator(function=np.sinh, name='sinh', degree=1)
cosh1 = Operator(function=np.cosh, name='cosh', degree=1)
tanh1 = Operator(function=np.tanh, name='tanh', degree=1)
exp1 = Operator(function=_protected_exp, name='exp', degree=1)
expsq1 = Operator(function=_protected_expsq, name='expsq', degree=1)

sigmoid = Operator(function=_sigmoid, name='sigmoid', degree=1)
softplus = Operator(function=_softplus, name='softplus', degree=1)
softmax = Operator(function=_softmax, name='softmax', degree=2)

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




