import jax
import bisect
import warnings
import numpy as np
import jax.numpy as jnp
from functools import lru_cache
from scipy.optimize import minimize
from typing import Union, Optional, List, Tuple, Iterator


from src.node import Operator, Constant, Variable, NodeContent, DynamicAggregation
from src.node_jax import DynamicAggregation as DynamicAggregation_jax
from src.node_jax import _operator_jax_map
from src.fitness import Fitness



class _RenderNode:
    """内部辅助节点，用于从 RPN 动态构建树形结构。"""
    def __init__(self, content: NodeContent):
        self.content = content
        self.children: List['_RenderNode'] = []
    
    @property
    def name(self) -> str:
        """委托给 NodeContent 的 name 属性"""
        return self.content.name
    
    @property
    def degree(self) -> int:
        """委托给 NodeContent 的 degree 属性"""
        return self.content.degree

def _build_tree_from_rpn(genes: List[NodeContent]) -> _RenderNode:
    """
    核心转换：将 RPN 基因列表转换为树。
    使用标准栈算法。
    """
    stack: List[_RenderNode] = []
    
    for gene in genes:
        node = _RenderNode(content=gene)
        
        if node.degree > 0:
            # 这是一个操作符，弹出它的操作数
            if len(stack) < node.degree:
                # RPN 无效
                err_node = _RenderNode(Constant(f"INVALID RPN (Stack underflow for {gene.name})"))
                err_node.children = stack
                return err_node
                
            # 从栈中弹出子节点
            children = [stack.pop() for _ in range(node.degree)]
            # RPN 是 L, R, OP，所以弹出的是 R, L
            # 我们需要反转它们以匹配 (L, R) 顺序
            node.children = list(reversed(children)) 
            
        stack.append(node)
    
    if len(stack) != 1:
        # 最终栈中应只有一个根节点
        err_node = _RenderNode(Constant(f"INVALID RPN (Final stack size {len(stack)})"))
        err_node.children = stack
        return err_node
        
    return stack[0] # 唯一的元素就是根节点

def RenderTree(expression: 'Expression') -> Iterator[Tuple[str, str, _RenderNode]]:
    """
    一个类似于 anytree.RenderTree 的迭代器（生成器）。
    
    它遍历一个 Expression 对象，并为每个节点生成 (prefix, fill, node)
    
    Yields:
        (prefix, fill, node):
        - prefix: 打印此节点所用的前缀 (e.g., "├── ")
        - fill:   打印此节点子节点所用的前缀 (e.g., "│   ")
        - node:   _RenderNode 辅助节点对象
    """
    
    # 1. 从 RPN 重建树
    root = _build_tree_from_rpn(expression.genes)
    
    # 2. 产生根节点 (没有前缀)
    yield "", "", root
    
    # 3. 递归产生子节点
    yield from _render_level_helper(root, "")

def _render_level_helper(node: _RenderNode, prefix: str) -> Iterator[Tuple[str, str, _RenderNode]]:
    """RenderTree 的递归辅助函数"""
    
    if not node.children:
        return # 叶子节点，停止递归
        
    num_children = len(node.children)
    for i, child in enumerate(node.children):
        is_last = (i == num_children - 1)
        
        # 确定分支和填充字符
        branch = "└── " if is_last else "├── "
        fill = "    " if is_last else "│   "
        
        # 产生当前子节点
        yield prefix + branch, prefix + fill, child
        
        # 为子节点的子节点（孙子节点）递归
        yield from _render_level_helper(child, prefix + fill)




class Expression(object):
    """
    基于后缀表达式（RPN）的符号表达式
    
    表达式编码为线性数组：[gene1, gene2, ..., geneN]
    求值顺序：从左到右，使用栈
    
    示例：
        表达式: (x1 + x2) * 3
        树形式: mul(add(x1, x2), 3)
        后缀: [x1, x2, add, 3, mul]
    """
    def __init__(self,
                 genes: List[NodeContent],
                 out_func: Operator | None = None, 
                 metric: Fitness | None = None,
                 complexity_of_operators: dict[str, int | float] | None = None,
                 complexity_of_constants: int | float | None = None,
                 complexity_of_variables: int | float | None = None, 
                 complexity_of_aggregations: int | float | None = None):
        self.genes = genes
        self.metric = metric
        self.out_func = out_func
        self.complexity_of_operators = complexity_of_operators or {}
        self.complexity_of_constants = complexity_of_constants or 1
        self.complexity_of_variables = complexity_of_variables or 1
        self.complexity_of_aggregations = complexity_of_aggregations or 1
        
        # 预分析表达式结构
        self._constant_indices = None
        
        # 懒编译梯度函数（只在需要时编译一次）
        self._grad_fn_compiled = None

    def _is_valid(self, genes=None) -> bool:
        """
        验证后缀表达式是否合法
        
        规则：
        1. 模拟栈执行，栈深度不能为负
        2. 最终栈深度必须为 1
        3. 禁止模式检查（x-x, x/x, *0, /0等）
        """
        genes = self.genes if genes is None else genes
        if not genes:
            return False
        
        # 栈深度检查
        stack_depth = 0
        for gene in genes:
            if gene.degree == 0:
                stack_depth += 1
            else:
                stack_depth -= gene.degree
                if stack_depth < 0:
                    return False
                stack_depth += 1
        
        if stack_depth != 1:
            return False
        
        return True

    @property
    def constant_indices(self):
        if self._constant_indices is not None:
            return self._constant_indices
        self._constant_indices = [i for i, g in enumerate(self.genes) if isinstance(g, Constant)]
        return self._constant_indices

    def _count_scalar_constants(self) -> int:
        """Recursively counts the number of scalar constants in the tree."""
        return sum(1 if isinstance(g, Constant) else 0 for g in self.genes)
    
    def _count_scalar_variables(self) -> int:
        """Recursively counts the number of scalar variables in the tree."""
        return sum(1 if isinstance(g, Variable) else 0 for g in self.genes)
    
    def _count_scalar_aggregations(self) -> int:
        """Recursively counts the number of scalar aggregations in the tree."""
        return sum(1 if isinstance(g, DynamicAggregation) else 0 for g in self.genes)

    def _has_binary_operator(self) -> bool:
        """Checks if the tree contains any binary operators."""
        return any(True if g.degree==2 else False for g in self.genes)

    @property
    def size(self) -> int:
        """The size of the expression tree."""
        return len(self.genes)

    def _calculate_complexity(self) -> float:
        """
        计算表达式的复杂度。
        """
        total_complexity = 0.0
        for gene in self.genes:
            if isinstance(gene, Operator):
                total_complexity += self.complexity_of_operators.get(gene.name, 1)
            elif isinstance(gene, Constant):
                total_complexity += self.complexity_of_constants
            elif isinstance(gene, Variable):
                total_complexity += self.complexity_of_variables
            elif isinstance(gene, DynamicAggregation):
                total_complexity += self.complexity_of_aggregations
            else:
                # 对于未知类型，默认复杂度为1
                total_complexity += 1
        return total_complexity

    @property
    def complexity(self) -> float:
        """表达式的复杂度。"""
        return self._calculate_complexity()

    @staticmethod
    def _genes_equal(a, b) -> bool:
        """比较两个基因是否相等"""
        if isinstance(a, NodeContent) and isinstance(b, NodeContent):
            return a == b
        if isinstance(a, tuple) and isinstance(b, tuple):
            # 递归比较表达式树
            return (a[0] == b[0] and 
                    Expression._genes_equal(a[1], b[1]) and
                    all(Expression._genes_equal(x, y) for x, y in zip(a[2:], b[2:])))
        return False

    def __eq__(self, other):
        """Recursively checks if two expression trees are identical."""
        if not isinstance(other, Expression):
            return NotImplemented
        return Expression._genes_equal(self.genes, other.genes)

    def __str__(self):
        """转换为中缀表达式字符串"""
        return self._to_infix_string()
    
    def _to_infix_string(self) -> str:
        """后缀转中缀（用于可视化）"""
        stack = []
        
        for gene in self.genes:
            if gene.degree == 0:
                if isinstance(gene, Constant):
                    stack.append(f"{gene.value:.3f}")
                else:
                    stack.append(gene.name)
            elif gene.degree == 1:
                arg = stack.pop()
                if gene.name == 'neg':
                    stack.append(f"(-{arg})")
                elif gene.name == 'inv':
                    stack.append(f"(1/{arg})")
                else:
                    stack.append(f"{gene.name}({arg})")
            elif gene.degree == 2:
                right = stack.pop()
                left = stack.pop()
                
                op_map = {
                    'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
                    'gt': '>', 'lt': '<', 'eq': '=', 'geq': '>=', 'leq': '<='
                }
                
                if gene.name in op_map:
                    symbol = op_map[gene.name]
                    stack.append(f"({left} {symbol} {right})")
                else:
                    stack.append(f"{gene.name}({left}, {right})")
        
        return stack[0] if stack else "EMPTY"
    
    def __repr__(self):
        return f"Expression(size={self.size}, formula='{self}')"
    
    def __getitem__(self, index) -> NodeContent:
        return self.genes[index]
    
    def copy(self) -> 'Expression':
        """Returns a deep copy of the expression."""
        return Expression(
            self.genes.copy(), out_func=self.out_func, metric=self.metric
        )

    def execute(self, X: np.ndarray) -> np.ndarray:
        stack = []
        
        for gene in self.genes:
            if gene.degree == 0:
                # Variable, Constant, DynamicAggregation
                result = gene(X)
                # 转换为NumPy（如果是JAX数组）
                if hasattr(result, 'device'):
                    result = np.array(result)
                stack.append(result)
            else:
                # Operator
                operands = [stack.pop() for _ in range(gene.degree)]
                operands.reverse()
                result = gene(*operands)
                stack.append(result)
        
        result = stack[0]
        
        # 处理标量
        if np.isscalar(result) or result.ndim == 0:
            result = np.full(X.shape[0], result)
        
        # 应用输出函数
        if self.out_func is not None:
            result = self._apply_operator_numpy(self.out_func, [result])
        
        return result
    
    def fitness(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the raw fitness of the expression according to X, y."""
        y_pred = self.execute(X)
        
        raw_fitness = float(self.metric(y, y_pred))
        return raw_fitness

    # ============ JAX梯度计算（仅用于常量优化） ============
    def _genes_to_jax_executable(self):
        """
        将基因列表转换为JAX可执行版本（仅在需要梯度时调用）
        关键优化：使用静态结构避免动态循环
        """
        genes = []
        for gene in self.genes:
            if gene.degree > 0:
                genes.append(_operator_jax_map[gene.name])
            elif isinstance(gene, DynamicAggregation):
                genes.append(
                    DynamicAggregation(
                        v_start=gene.v_start, 
                        v_end=gene.v_end, 
                        op_name=gene.op_name, 
                        n_variables=gene.n_variables,
                        valid_op=gene.valid_op
                    )
                )
                genes.append(gene)
            else:
                genes.append(gene)
        return genes

    def _build_jax_executable(self):
        """
        构建JAX可执行版本（仅在需要梯度时调用）
        关键优化：使用静态结构避免动态循环
        """
        genes = self._genes_to_jax_executable()
        
        out_func = self.out_func
        out_func = _operator_jax_map[out_func.name] if out_func is not None else None
        
        # 提取常量索引映射
        constant_indices = self.constant_indices
        const_idx_map = {idx: i for i, idx in enumerate(constant_indices)}
        
        def _execute_with_constants(X_jax, constants):
            """JAX执行函数（可微分）"""
            stack = []
            
            for i, gene in enumerate(genes):
                if gene.degree == 0:
                    if i in const_idx_map:
                        # 使用可优化的常量
                        stack.append(constants[const_idx_map[i]])
                    else:
                        # Variable或DynamicAggregation
                        stack.append(gene(X_jax))
                else:
                    # Operator
                    operands = [stack.pop() for _ in range(gene.degree)]
                    operands.reverse()
                    result = gene(*operands)
                    stack.append(result)
            
            result = stack[0]
            
            # 处理标量
            if result.ndim == 0:
                result = jnp.full(X_jax.shape[0], result)
            
            # 应用输出函数
            if out_func is not None:
                result = out_func(result)
            
            return result
        
        return _execute_with_constants
    
    def _get_gradient_function(self):
        """懒编译梯度函数"""
        if self._grad_fn_compiled is None:
            executable = self._build_jax_executable()
            metric = self.metric
            
            def loss_fn(constants, X_jax, y_jax):
                """损失函数（用于计算梯度）"""
                y_pred = executable(X_jax, constants)
                loss = metric(y_jax, y_pred)
                # 如果是最大化指标，取负数
                if metric.greater_is_better:
                    loss = -loss
                return loss
            
            # JIT编译梯度函数
            self._grad_fn_compiled = jax.jit(jax.grad(loss_fn, argnums=0))
        
        return self._grad_fn_compiled
    
    def compute_constant_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算常量的梯度（使用JAX）
        返回：(梯度数组, 当前损失值)
        """
        if not (len(self.constant_indices) > 0):
            return None, None
        
        # 转换为JAX数组
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        # 提取当前常量值
        current_constants = jnp.array([
            self.genes[idx].value for idx in self.constant_indices
        ])
        
        # 计算梯度
        grad_fn = self._get_gradient_function()
        gradients = grad_fn(current_constants, X_jax, y_jax)
        
        # 计算当前损失
        executable = self._build_jax_executable()
        y_pred = executable(X_jax, current_constants)
        loss = float(self.metric(y_jax, y_pred))
        if self.metric.greater_is_better:
            loss = -loss
        
        return np.array(gradients), loss
    
    def update_constants(self, new_values: np.ndarray):
        """更新常量值"""
        if len(new_values) != len(self.constant_indices):
            raise ValueError(f"Expected {len(self.constant_indices)} values, got {len(new_values)}")
        
        new_expr = self.copy()
        for i, idx in enumerate(self.constant_indices):
            new_expr.genes[idx] = Constant(float(new_values[i]))
        
        return new_expr




class ExpressionSet(object):
    """
    A collection of `Expression` objects, representing a system of equations.
    """
    def __init__(self, 
                 expressions: List[Expression | None], 
                 out_func: Operator | None = None, 
                 metric: Fitness | None = None):
        self.out_func = out_func
        self.metric = metric
        if not all(isinstance(expr, (type(None), Expression)) for expr in expressions):
            raise ValueError("All items in expressions must be GeneticExpression objects.")
        
        self.expressions = expressions

    def __len__(self) -> int:
        """Returns the number of expressions in the set."""
        return len(self.expressions)
    
    def __getitem__(self, key: int) -> 'Expression':
        return self.expressions[key]

    def __str__(self) -> str:
        return '[' + '; '.join([str(expr) for expr in self.expressions]) + ']'

    def __repr__(self) -> str:
        expr_reprs = ", ".join(repr(expr) for expr in self.expressions if expr is not None)
        return f"ExpressionSet(n_expressions={len(self)}, expressions=[{expr_reprs}])"

    @property
    def size(self) -> int:
        """The total size of all expression trees in the set."""
        return sum(expr.size for expr in self if expr is not None)

    @property
    def order(self) -> int:
        non_none_count = sum([1 if expr is not None else 0 for expr in self.expressions])
        return non_none_count

    def copy(self) -> 'ExpressionSet':
        """Returns a copy of the expression set."""
        return ExpressionSet(
            [expr.copy() for expr in self.expressions], 
            out_func=self.out_func, metric=self.metric
        )

    def execute(self, X: jnp.ndarray) -> jnp.ndarray:
        outputs = [expr.execute(X).reshape(-1, 1) for expr in self.expressions if expr is not None]
        if not outputs:
            return jnp.array([]).reshape(X.shape[0], 0)

        result = jnp.hstack(outputs)
        return result if self.out_func is None else self.out_func(result)

    def fitness(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Evaluate the raw fitness of the expression set according to X, y."""
        y_pred = self.execute(X)
        if y is None:
            raw_fitness = self.metric(X, y_pred)
        else:
            raw_fitness = self.metric(y, y_pred)
        return raw_fitness

    # ============ JAX梯度计算（仅用于常量优化） ============
    def _build_jax_executable(self):
        """
        构建JAX可执行版本（仅在需要梯度时调用）
        关键优化：使用静态结构避免动态循环
        """
        expr_set = [expr for expr in self.expressions if expr is not None]
        const_counter = 0
        expr_set_genes = []
        expr_set_const_idx_map = {}
        for expr_idx, expr in enumerate(expr_set):
            genes = expr._genes_to_jax_executable()
            expr_set_genes.append(genes)
            constant_indices = expr.constant_indices
            if constant_indices:
                expr_const_idx_map = {(expr_idx, idx): i+const_counter for i, idx in enumerate(constant_indices)}
                expr_set_const_idx_map[expr_idx] = expr_const_idx_map
                const_counter += len(constant_indices)
            else:
                expr_set_const_idx_map[expr_idx] = {}
        
        out_func = self.out_func
        out_func = _operator_jax_map[out_func.name] if out_func is not None else None
        
        def _execute_with_constants(X_jax, constants):
            def _execute_expr_with_constants(genes, X_jax, constants, expr_idx, const_idx_map):
                """JAX执行函数（可微分）"""
                stack = []
                
                for i, gene in enumerate(genes):
                    if gene.degree == 0:
                        if (expr_idx, i) in const_idx_map:
                            # 使用可优化的常量
                            stack.append(constants[const_idx_map[(expr_idx, i)]])
                        else:
                            # Variable或DynamicAggregation
                            stack.append(gene(X_jax))
                    else:
                        # Operator
                        operands = [stack.pop() for _ in range(gene.degree)]
                        operands.reverse()
                        result = gene(*operands)
                        stack.append(result)
                
                result = stack[0]
                
                # 处理标量
                if result.ndim == 0:
                    result = jnp.full(X_jax.shape[0], result)
                
                return result
            
            results = []
            for expr_idx, expr_genes in enumerate(expr_set_genes):
                result = _execute_expr_with_constants(
                    expr_genes, X_jax, constants, expr_idx, expr_set_const_idx_map[expr_idx]
                )
                results.append(result)
            results = jnp.hstack(results)
            # 应用输出函数
            if out_func is not None:
                results = out_func(results)
        
            return results
        
        return _execute_with_constants

    def _get_gradient_function(self):
        """懒编译梯度函数"""
        if self._grad_fn_compiled is None:
            executable = self._build_jax_executable()
            metric = self.metric
            
            def loss_fn(constants, X_jax, y_jax):
                """损失函数（用于计算梯度）"""
                y_pred = executable(X_jax, constants)
                loss = metric(y_jax, y_pred)
                # 如果是最大化指标，取负数
                if metric.greater_is_better:
                    loss = -loss
                return loss
            
            # JIT编译梯度函数
            self._grad_fn_compiled = jax.jit(jax.grad(loss_fn, argnums=0))
        
        return self._grad_fn_compiled

    def compute_constant_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算常量的梯度（使用JAX）
        返回：(梯度数组, 当前损失值)
        """
        constant_indices = []
        for expr_idx, expr in enumerate(self.expressions):
            if expr is not None:
                constant_indices.extend([
                    (expr_idx, gene_idx) for gene_idx in expr.constant_indices
                ])
        
        # 转换为JAX数组
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        # 提取当前常量值
        current_constants = jnp.array([
            self[expr_idx][gene_idx].value for (expr_idx, gene_idx) in constant_indices
        ])
        
        # 计算梯度
        grad_fn = self._get_gradient_function()
        gradients = grad_fn(current_constants, X_jax, y_jax)
        
        # 计算当前损失
        executable = self._build_jax_executable()
        y_pred = executable(X_jax, current_constants)
        loss = float(self.metric(y_jax, y_pred))
        if self.metric.greater_is_better:
            loss = -loss
        
        return np.array(gradients), loss

    def update_constants(self, new_values: np.ndarray):
        """更新常量值"""
        constant_indices = []
        for expr_idx, expr in enumerate(self.expressions):
            if expr is not None:
                constant_indices.extend([
                    (expr_idx, gene_idx) for gene_idx in expr.constant_indices
                ])
        
        if len(new_values) != len(constant_indices):
            raise ValueError(f"Expected {len(constant_indices)} values, got {len(new_values)}")
        
        new_expr_set = self.copy()
        for i, (expr_idx, gene_idx) in enumerate(constant_indices):
            new_expr_set.expressions[expr_idx][gene_idx] = Constant(float(new_values[i]))
        
        return new_expr_set

