import jax.numpy as jnp
from jax import jit


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
        'mean': jnp.mean,
        'max': jnp.max,
        'min': jnp.min,
        'median': jnp.median,
        'std': jnp.std, 
        'var': jnp.var,
        'sum': jnp.sum
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

    def __call__(self, X: jnp.array):
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
        self.value = jnp.array(value, float)

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
        try:
            return self.function(*args)
        except Exception:
            return args[0]

    def __eq__(self, other):
        if isinstance(other, Operator):
            return self.name == other.name and self.degree == other.degree
        return False




def _protected_addition(x1, x2):
    """加法闭包,处理溢出情况"""
    result = jnp.add(x1, x2)
    # 检测溢出:结果是否在合理范围内
    safe_mask = jnp.isfinite(result) & (jnp.abs(result) < 1e10)
    return jnp.where(safe_mask, result, 0.)


def _protected_subtraction(x1, x2):
    """减法闭包,处理溢出情况"""
    result = jnp.subtract(x1, x2)
    # 检测溢出:结果是否在合理范围内
    safe_mask = jnp.isfinite(result) & (jnp.abs(result) < 1e10)
    return jnp.where(safe_mask, result, 0.)


def _protected_multiplication(x1, x2):
    """乘法闭包,处理溢出情况"""
    # 预先检查是否会溢出
    safe_mask = (jnp.abs(x1) < 1e10) & (jnp.abs(x2) < 1e10)
    # 对于可能溢出的情况,进一步检查乘积
    result = jnp.where(safe_mask, jnp.multiply(x1, x2), 0.)
    # 再次验证结果是否有限
    result = jnp.where(jnp.isfinite(result) & (jnp.abs(result) < 1e10), result, 0.)
    return result

def _protected_division(x1, x2):
    """除法闭包，处理零除数情况"""
    safe_mask = (jnp.abs(x2) > 0.0001) & (jnp.abs(x1 / x2) < 1e10)
    return jnp.where(safe_mask, jnp.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """平方根闭包，处理负数参数"""
    return jnp.sqrt(jnp.abs(x1))


def _protected_log(x1):
    """对数闭包，处理零和负数参数"""
    return jnp.where(jnp.abs(x1) > 0.001, jnp.log(jnp.abs(x1)), 0.)


def _protected_inverse(x1):
    """倒数闭包，处理零参数"""
    return jnp.where(jnp.abs(x1) > 0.001, 1. / x1, 0.)


def _protected_exp(x1):
    """指数闭包，限制大参数避免溢出"""
    clipped_x1 = jnp.clip(x1, a_min=None, a_max=700.)
    return jnp.exp(clipped_x1)


def _protected_expsq(x1):
    """高斯函数闭包，处理大参数避免溢出"""
    # 限制输入范围，防止指数部分过大
    clipped_x1 = jnp.clip(x1, a_min=-30., a_max=30.)
    return jnp.exp(-jnp.square(clipped_x1))



@jit
def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
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
    """
    EXP_LOWER_BOUND = -88.0
    EXP_UPPER_BOUND = 88.0
    
    x_clipped = jnp.clip(x, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
    pos_mask = (x_clipped >= 0)
    neg_mask = ~pos_mask
    
    # 使用 lax.cond 或 jnp.where 来处理分支
    def pos_case(x_val):
        return 1.0 / (1.0 + jnp.exp(-x_val))
    
    def neg_case(x_val):
        z = jnp.exp(x_val)
        return z / (1.0 + z)
    
    # 使用向量化的 where
    result = jnp.where(pos_mask, pos_case(x_clipped), neg_case(x_clipped))
    
    # 处理极端值
    result = jnp.where(x <= EXP_LOWER_BOUND, 0.0, result)
    result = jnp.where(x >= EXP_UPPER_BOUND, 1.0, result)
    
    return result.astype(x.dtype)



@jit
def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    """
    Softplus 激活函数
    
    f(x) = ln(1 + exp(x))
    ReLU 的平滑近似
    """
    x_clipped = jnp.clip(x, -88, 88)
    return jnp.log(1.0 + jnp.exp(x_clipped))


@jit
def _softmax(x: jnp.ndarray, axis: int = 1) -> jnp.ndarray:
    """
    计算数值稳定的 Softmax 函数
    
    沿指定轴对输入数组计算 softmax 值：
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    通过减去最大值保证数值稳定性，避免指数运算溢出
    """
    if axis not in (0, 1):
        raise ValueError("axis 必须是 0 或 1")
    
    x_max = jnp.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    
    # 对极端值进行保护
    x_shifted = jnp.clip(x_shifted, -700, 700)
    
    exps = jnp.exp(x_shifted)
    sum_exps = jnp.sum(exps, axis=axis, keepdims=True)
    
    # 处理 sum_exps 为 0 的情况
    safe_sum = jnp.where(sum_exps == 0, 1.0, sum_exps)
    result = exps / safe_sum
    
    return result



# 可选：创建所有函数的 JIT 编译版本
protected_functions = {
    'addition': jit(_protected_addition),
    'subtraction': jit(_protected_subtraction),
    'multiplication': jit(_protected_multiplication),
    'division': jit(_protected_division),
    'sqrt': jit(_protected_sqrt),
    'log': jit(_protected_log),
    'inverse': jit(_protected_inverse),
    'exp': jit(_protected_exp),
    'expsq': jit(_protected_expsq),
    'sigmoid': _sigmoid,  # 已经 JIT
    'softplus': _softplus,
    'softmax': _softmax
}



add2 = Operator(function=protected_functions['addition'], name='add', degree=2)
sub2 = Operator(function=protected_functions['subtraction'], name='sub', degree=2)
mul2 = Operator(function=protected_functions['multiplication'], name='mul', degree=2)
div2 = Operator(function=protected_functions['division'], name='div', degree=2)
sqrt1 = Operator(function=protected_functions['sqrt'], name='sqrt', degree=1)
log1 = Operator(function=protected_functions['log'], name='log', degree=1)
neg1 = Operator(function=jnp.negative, name='neg', degree=1)
inv1 = Operator(function=protected_functions['inverse'], name='inv', degree=1)
abs1 = Operator(function=jnp.abs, name='abs', degree=1)
maximum2 = Operator(function=jnp.maximum, name='maximum', degree=2)
minimum2 = Operator(function=jnp.minimum, name='minimum', degree=2)
sin1 = Operator(function=jnp.sin, name='sin', degree=1)
cos1 = Operator(function=jnp.cos, name='cos', degree=1)
tan1 = Operator(function=jnp.tan, name='tan', degree=1)
sinh1 = Operator(function=jnp.sinh, name='sinh', degree=1)
cosh1 = Operator(function=jnp.cosh, name='cosh', degree=1)
tanh1 = Operator(function=jnp.tanh, name='tanh', degree=1)
exp1 = Operator(function=protected_functions['exp'], name='exp', degree=1)
expsq1 = Operator(function=protected_functions['expsq'], name='expsq', degree=1)

sigmoid = Operator(function=protected_functions['sigmoid'], name='sigmoid', degree=1)
softplus = Operator(function=protected_functions['softplus'], name='softplus', degree=1)
softmax = Operator(function=protected_functions['softmax'], name='softmax', degree=2)

_operator_jax_map = {
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