import math
import bisect
import warnings
import numpy as np
import jax.numpy as jnp
from typing import Union, Optional, List, Tuple, Callable


from src.node import Operator, Constant, Variable, _operator_map, NodeContent, DynamicAggregation
from src.tree import count_trees, generate_random_tree, get_mth_tree, SymbolicNode, PreOrderIter
from src.expression import Expression, ExpressionSet
from src.fitness import Fitness, _fitness_map
from src.utils import check_random_state




class ExprGenerator:
    # 类级别的缓存，所有实例共享
    _global_count_memo = {}
    _global_size_tree_counts = {}
    _global_size_prob = {}
    _global_valid_degrees_cache = {}

    def __init__(self,
                 *,
                 maxsize: int,
                 operators: List[str | Operator],
                 n_variables: int, 
                 use_constants: bool = True,
                 use_variables: bool = True,
                 use_aggregations: bool = False,
                 variable_names: List[str] | None = None,
                 aggregation_operators: List[str] | None = None,
                 metric: Optional[Callable | str | Operator] = None,
                 out_func: Optional[Callable | str | Operator] = None,
                 random_state: Union[int, np.random.RandomState] = None):
        self.maxsize = maxsize
        self.n_variables = n_variables
        self.use_constants = use_constants
        self.use_variables = use_variables
        self.variable_names = variable_names
        self.use_aggregations = use_aggregations
        self.aggregation_operators = aggregation_operators
        self.random_state = check_random_state(random_state)
        
        self.metric = self._init_metric(metric)
        self.out_func = self._init_out_func(out_func)
        self.operators = self._init_operators(operators)
        self.constants = self._init_constants(n_variables)
        self.variables = self._init_variables(n_variables, variable_names)
        
        # 按度数分组操作符（快速查找）
        self.unary_operators = []
        self.binary_operators = []
        self._degree_operators = {}
        for op in self.operators:
            if op.degree not in self._degree_operators:
                self._degree_operators[op.degree] = []
            self._degree_operators[op.degree].append(op)
            if op.degree > 0:
                if op.degree == 1: self.unary_operators.append(op)
                elif op.degree == 2: self.binary_operators.append(op)
                
        if (not self.variables) and (not self.aggregations):
            raise ValueError("The expression must have at least one variable or aggregation variable.")
        if not any(value > 0 for value in self._degree_operators.keys()):
            raise ValueError("The expression must have at least one operator with arity > 0.")
        
        # 使用缓存键来共享计算结果
        self._cache_key = self._get_cache_key()
        self._ensure_cache_computed()

    def _init_metric(self, metric: Union[str, Fitness]):
        if isinstance(metric, Fitness):
            _metric = metric
        elif isinstance(metric, str):
            if metric not in _fitness_map:
                raise ValueError('Unsupported metric: %s' % metric)
            _metric = _fitness_map[metric]
        else:
            raise ValueError('Invalid type %s found in `metric`.' % type(metric))
        return _metric

    def _init_operators(self, operators: List[str | Operator]):
        _operators = []
        for operator in operators:
            if isinstance(operator, str):
                if operator not in _operator_map:
                    raise ValueError('invalid operator name %s found in `operators`.' % operator)
                _operators.append(_operator_map[operator])
            elif isinstance(operator, Operator):
                _operators.append(operator)
            else:
                raise ValueError('invalid type %s found in `operators`.' % type(operator))
        if not _operators:
            raise ValueError('No valid operators found in `operators`.')
        return _operators

    def _init_out_func(self, out_func: Optional[Union[str, Operator, callable]] = None):
        if isinstance(out_func, Operator):
            assert out_func.degree == 1, \
                    "Out operator only support elementwise operator with 1 degree. "
            _out_func = out_func
        elif isinstance(out_func, str):
            if out_func not in _operator_map:
                raise ValueError('Unsupported metric: %s' % out_func)
            _out_func = _operator_map[out_func]
        elif out_func is None:
            _out_func = out_func
        else:
            raise ValueError('Unsupported out_func: %s, ' % out_func,
                             "out_func must be a Operator class, None or operator with degree 1. ")
        return _out_func

    def _init_variables(self, n_variables, variable_names=None):
        variables = []
        if not self.use_variables:
            return variables
        
        if variable_names is not None:
            if n_variables != len(variable_names):
                raise ValueError('The supplied `variable_names` has different '
                                    'length to n_features. Expected %d, got %d.'
                                    % (n_variables, len(variable_names)))
            for variable_name in variable_names:
                if not isinstance(variable_name, str):
                    raise ValueError('invalid type %s found in '
                                        '`variable_names`.' % type(variable_name))
        else:
            variable_names = [f'x{i}' for i in range(n_variables)]
        
        for i, variable_name in enumerate(variable_names):
            variables.append(Variable(i, name=variable_name))
        
        return variables

    def _init_constants(self, n_variables):
        if not self.use_constants: return []
        constants = [Constant(self.random_state.normal(0, 3)) for _ in range(
                max(1, math.ceil(n_variables/2))
            )]

        return constants

    def _init_aggregation(self, n_variables: int, aggregation_operators: list):
        v_start = self.random_state.randint(0, n_variables-2)
        v_end   = self.random_state.randint(v_start+1, n_variables)
        op_name = self.random_state.choice(aggregation_operators)
        return DynamicAggregation(v_start, v_end, op_name, n_variables, aggregation_operators)

    def _update_constants(self, constants: List[Constant]) -> None:
        """Replace constants with the given ones."""
        self.constants = list(constants)

    # ------------------------------------------------------------------------
    # 关键属性缓存
    # ------------------------------------------------------------------------

    def _get_cache_key(self):
        """生成用于缓存的唯一键"""
        operator_names = tuple(sorted(op.name for op in self.operators))
        degrees = tuple(sorted(self._degree_operators.keys()))
        return (operator_names, degrees, self.maxsize)
    
    def _ensure_cache_computed(self):
        """确保缓存已计算（懒加载）"""
        if self._cache_key not in ExprGenerator._global_valid_degrees_cache:
            # 第一次计算，存入全局缓存
            self.valid_degrees = [0] + list(set(self._degree_operators.keys()))
            ExprGenerator._global_valid_degrees_cache[self._cache_key] = self.valid_degrees
            
            size_tree_counts = {}
            count_memo = ExprGenerator._global_count_memo.setdefault(self._cache_key, {})
            
            for size in range(1, self.maxsize + 1):
                tree_count = count_trees(size, self.valid_degrees, count_memo)
                if tree_count > 0:
                    size_tree_counts[size] = tree_count
            
            ExprGenerator._global_size_tree_counts[self._cache_key] = size_tree_counts
            
            total_combinations = sum(size_tree_counts.values())
            if total_combinations > 0:
                probabilities = np.array(list(size_tree_counts.values()), dtype=np.int64) / float(total_combinations)
                size_prob = {size: prob for size, prob in zip(size_tree_counts.keys(), probabilities)}
            else:
                size_prob = {}
            
            ExprGenerator._global_size_prob[self._cache_key] = size_prob
        
        # 使用全局缓存
        self.valid_degrees = ExprGenerator._global_valid_degrees_cache[self._cache_key]
        self.size_tree_counts = ExprGenerator._global_size_tree_counts[self._cache_key]
        self.size_prob = ExprGenerator._global_size_prob[self._cache_key]
    
    def __getstate__(self):
        """控制序列化，排除计算出的缓存。"""
        state = self.__dict__.copy()
        # 移除实例级缓存属性（它们会从类级缓存重建）
        state.pop('size_tree_counts', None)
        state.pop('valid_degrees', None)
        state.pop('size_prob', None)
        state.pop('_cache_key', None)
        return state

    def __setstate__(self, state):
        """控制反序列化，重新计算缓存。"""
        self.__dict__.update(state)
        # 重新生成缓存键并确保缓存存在
        self._cache_key = self._get_cache_key()
        self._ensure_cache_computed()

    # ------------------------------------------------------------------------
    # 基于复杂度的表达式初始化
    # ------------------------------------------------------------------------

    def generate_random_expr(self, size: Optional[int] = None):
        tree = self.build_tree(size)
        expression = Expression(
            tree=tree, metric=self.metric, out_func=self.out_func
        )
        return expression

    def build_tree(self, size: Optional[int] = None) -> SymbolicNode:
        """根据大小限制，公平地生成一个随机符号树。"""
        # 使用全局缓存的 count_memo
        count_memo = ExprGenerator._global_count_memo.setdefault(self._cache_key, {})
        
        if size is None:
            size = self.random_state.choice(list(self.size_prob.keys()), p=list(self.size_prob.values()))
        
        if self.maxsize <= 31:
            tree_index = self.random_state.randint(self.size_tree_counts[size], dtype=np.int64)
            tree = get_mth_tree(size, self.valid_degrees, tree_index, count_memo)
        else:
            tree = generate_random_tree(size, self.valid_degrees, count_memo, self.random_state)
        
        # 递归生成树
        for node in PreOrderIter(tree):
            if node.degree > 0:
                node.node_content = self._get_random_operator(node.degree)
            elif node.degree == 0:
                node.node_content = self._get_random_leaf()
            else:
                raise ValueError("Invalid degree")
        
        return tree

    def _get_random_operator(self, degree: Optional[int] = None, exclude: Operator = None) -> Operator:
        """Helper to get a random operator (non-leaf)."""
        if degree is None:
            options = [operator for operator in self.operators if operator != exclude]
        elif isinstance(degree, int) and degree in self._degree_operators:
            options = [operator for operator in self._degree_operators[degree] if operator != exclude]
        else:
            raise ValueError("Invalid degree")
        if not options: return None
        return self.random_state.choice(options)

    def _get_random_leaf(self, exclude: Optional[Variable] = None) -> NodeContent:
        """Gets a random leaf node (variable or constant)."""
        if exclude is not None:
            if isinstance(exclude, Variable):
                options = [var for var in self.variables if var != exclude]
                return self.random_state.choice(options)
        
        probs = np.array([
            2 if self.use_variables else 0,
            1 if self.use_constants else 0,
            1 if self.use_aggregations else 0
        ])
        leaf_type = self.random_state.choice([0, 1, 2], p=probs/sum(probs))
        if leaf_type == 0:
            return self.random_state.choice(self.variables)
        elif leaf_type == 1:
            return self.random_state.choice(self.constants)
        else:
            return self._init_aggregation(self.n_variables, self.aggregation_operators)




def max_tree_set_structures(size, N, tree_count):
    """
    计算符号树集合的最大合法结构数量（优化版）
    
    参数:
        size: 符号树集合的总大小
        N: 符号树的数量
        tree_count: 字典，映射单个树的大小到其结构数量（单调递增）
    
    返回:
        最大的结构数量
    """
    if N == 0:
        return 0 if size > 0 else 0
    
    # 获取排序后的有效大小
    valid_sizes = sorted(tree_count.keys())
    if not valid_sizes:
        return 0
    
    min_size = valid_sizes[0]
    max_size = valid_sizes[-1]
    
    # 快速检查是否可行（树的大小可以为0或更大）
    if size > N * max_size or size < N * min_size:
        return 0
    
    # 使用动态规划：dp[i][j] = 用i个树组成大小为j时的最大结构数
    # 但这个空间复杂度太高，改用记忆化搜索
    memo = {}
    
    def dp(remaining_size, remaining_trees):
        """
        remaining_size: 剩余需要分配的大小
        remaining_trees: 剩余树的数量
        返回最大结构数，如果无法分配返回None
        """
        if remaining_trees == 0:
            return 0 if remaining_size == 0 else None
        
        if remaining_size < remaining_trees * min_size or remaining_size > remaining_trees * max_size:
            return None
        
        # 记忆化
        state = (remaining_size, remaining_trees)
        if state in memo:
            return memo[state]
        
        max_count = None
        
        # 尝试为第一个树分配不同的大小
        # 关键优化：只尝试那些使得剩余部分可行的大小
        for tree_size in valid_sizes:
            rest = remaining_size - tree_size
            rest_trees = remaining_trees - 1
            
            # 检查剩余部分是否可行
            if rest < rest_trees * min_size or rest > rest_trees * max_size:
                continue
            
            # 递归处理剩余部分
            if rest_trees == 0:
                if rest == 0:
                    max_count = tree_count[tree_size] if max_count is None else max(max_count, tree_count[tree_size])
            else:
                sub_count = dp(rest, rest_trees)
                if sub_count is not None:
                    total_count = tree_count[tree_size] + sub_count
                    max_count = total_count if max_count is None else max(max_count, total_count)
        
        memo[state] = max_count
        return max_count
    
    result = dp(size, N)
    return result if result is not None else 0



class ExprSetGenerator(ExprGenerator):
    def __init__(self,
                 *,
                 maxsize: int,
                 order: int | Tuple[int, int],
                 operators: List[str | Operator],
                 n_variables: int, 
                 use_constants: bool = True,
                 use_variables: bool = True,
                 use_aggregations: bool = False,
                 variable_names: List[str] | None = None,
                 aggregation_operators: List[str] | None = None,
                 metric: Optional[Callable | str | Operator] = None,
                 out_func: Optional[Callable | str | Operator] = None,
                 random_state: Union[int, np.random.RandomState] = None):
        super().__init__(
            maxsize=maxsize,
            operators=operators,
            n_variables=n_variables,
            use_constants=use_constants,
            use_variables=use_variables,
            variable_names=variable_names,
            use_aggregations=use_aggregations,
            aggregation_operators=aggregation_operators,
            metric=metric, out_func=out_func, random_state=random_state
        )
        if isinstance(order, int):
            self.minorder = order
            self.maxorder = order
            self.fixed = True
        elif isinstance(order, tuple):
            if len(order) != 2:
                raise ValueError("order tuple must contain two integers (minorder, maxorder).")
            minorder, maxorder = order
            if not (isinstance(minorder, int) and isinstance(maxorder, int)):
                raise ValueError("minorder and maxorder must be integers.")
            if not (minorder < maxorder):
                raise ValueError(f"minorder ({minorder}) must be < maxorder ({maxorder}).")
            if minorder < 1:
                raise ValueError("minorder must be at least 1.")
            
            self.minorder = minorder
            self.maxorder = maxorder
            self.fixed = False
        else:
            raise TypeError("order must be an integer or a tuple of two integers.")
        
        size_tree_counts = self.size_tree_counts.copy()
        size_tree_counts[0] = 0
        self.size_maxcounts = {}
        for size in range(1, self.maxorder * maxsize + 1):
            maxcounts = max_tree_set_structures(size, self.maxorder, size_tree_counts)
            if maxcounts > 0:
                self.size_maxcounts[size] = maxcounts

    def generate_random_exprset(self, size: Optional[int] = None):
        if self.fixed:
            expressions = [
                self.generate_random_expr(size) for _ in range(self.maxorder)
            ]
        else:
            n_expressions = self.random_state.randint(self.minorder, self.maxorder + 1)
            expressions = [None] * self.maxorder
            indices = self.random_state.permutation(self.maxorder)[:n_expressions]
            for i in indices:
                expressions[i] = self.generate_random_expr(size)
        
        return ExpressionSet(
            expressions=expressions, out_func=self.out_func, metric=self.metric
        )



