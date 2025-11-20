import jax
import bisect
import warnings
import numpy as np
import jax.numpy as jnp
from functools import lru_cache
from scipy.optimize import minimize
from typing import Union, Optional, List, Tuple, Iterator


from src.node import Operator, Constant, Variable, _operator_map, NodeContent, DynamicAggregation
from src.tree import count_trees, generate_random_tree, get_mth_tree
from src.utils import check_random_state
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
    
    def __getitem__(self, index):
        return self.genes[index]
    
    def copy(self) -> 'Expression':
        """Returns a deep copy of the expression."""
        return Expression(
            self.genes.copy(), out_func=self.out_func, metric=self.metric
        )

    def execute(self, X) -> jnp.ndarray:
        stack = []
        
        for gene in self.genes:
            if gene.degree == 0:
                # Variable, Constant, or DynamicAggregation
                stack.append(gene(X))
            else:
                # Operator
                if len(stack) < gene.degree:
                    raise ValueError("Invalid RPN expression: stack underflow.")
                
                # Pop operands in reverse order
                operands = [stack.pop() for _ in range(gene.degree)]
                operands.reverse()
                
                result = gene(*operands)
                stack.append(result)
        
        if len(stack) != 1:
            raise ValueError("Invalid RPN expression: final stack size is not 1.")
        
        final_result = stack[0]

        # If the result is a scalar constant, broadcast it to the shape of the input.
        if hasattr(final_result, 'ndim') and final_result.ndim == 0:
            final_result = jnp.full(X.shape[0], final_result)

        if self.out_func is not None:
            return self.out_func(final_result)
        
        return final_result

    def fitness(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Evaluate the raw fitness of the expression according to X, y."""
        y_pred = self.execute(X)
        raw_fitness = self.metric(y, y_pred)
        return raw_fitness

    def _execute_for_grad(self, X: jnp.ndarray, constants: jnp.ndarray = None) -> jnp.ndarray:
        stack = []
        
        constant_indice = 0
        for gene in self.genes:
            if gene.degree == 0:
                # Variable, Constant, or DynamicAggregation
                if isinstance(gene, Constant):
                    stack.append(constants[constant_indice])
                    constant_indice += 1
                else:
                    stack.append(gene(X))
            else:
                # Operator
                if len(stack) < gene.degree:
                    raise ValueError("Invalid RPN expression: stack underflow.")
                
                # Pop operands in reverse order
                operands = [stack.pop() for _ in range(gene.degree)]
                operands.reverse()
                
                result = gene(*operands)
                stack.append(result)
        
        if len(stack) != 1:
            raise ValueError("Invalid RPN expression: final stack size is not 1.")
        
        final_result = stack[0]

        # If the result is a scalar constant, broadcast it to the shape of the input.
        if hasattr(final_result, 'ndim') and final_result.ndim == 0:
            final_result = jnp.full(X.shape[0], final_result)

        if self.out_func is not None:
            return self.out_func(final_result)
        
        return final_result

    def _fitness_for_grad(self, X: jnp.ndarray, y: jnp.ndarray, constants: jnp.ndarray = None) -> float:
        """Evaluate the raw fitness of the expression according to X, y."""
        y_pred = self._execute_for_grad(X, constants)
        raw_fitness = self.metric(y, y_pred)
        return raw_fitness




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
    
    def __getitem__(self, key: int) -> 'ExpressionSet':
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

    def _execute_for_grad(self, X: jnp.ndarray, constants: jnp.ndarray, 
                          expr_constants_num: List[int] = None) -> jnp.ndarray:
        """Execute the expression set for gradient calculation."""
        if expr_constants_num is None:
            expr_constants_num = [expr._count_scalar_constants() for expr in self.expressions 
                                  if expr is not None]
        outputs = []
        for i, expr in enumerate(self.expressions):
            if expr is not None:
                start_idx = sum(expr_constants_num[:i])
                end_idx = sum(expr_constants_num[:i+1])
                outputs.append(expr._execute_for_grad(X, constants[start_idx: end_idx]).reshape(-1, 1))
        
        if not outputs:
            return jnp.array([]).reshape(X.shape[0], 0)

        result = jnp.hstack(outputs)
        return result if self.out_func is None else self.out_func(result)

    def _fitness_for_grad(self, X: jnp.ndarray, y: jnp.ndarray, constants: jnp.ndarray, 
                          expr_constants_num: List[int] = None) -> float:
        """Evaluate the raw fitness of the expression set according to X, y."""
        y_pred = self._execute_for_grad(X, constants, expr_constants_num)
        if y is None:
            raw_fitness = self.metric(X, y_pred)
        else:
            raw_fitness = self.metric(y, y_pred)
        return raw_fitness


