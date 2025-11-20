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
            result = self.out_func(result)
        
        return result
    
    def fitness(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the raw fitness of the expression according to X, y."""
        y_pred = self.execute(X)
        
        raw_fitness = np.float32(self.metric(y, y_pred))
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
    优化后的表达式集合
    关键改进：
    1. 预计算常量索引映射（扁平化）
    2. 静态执行计划（避免动态查找）
    3. 向量化处理多个表达式
    """
    def __init__(self, 
                 expressions: List[Expression | None], 
                 out_func: Operator | None = None, 
                 metric: Fitness | None = None):
        self.out_func = out_func
        self.metric = metric
        
        if not all(isinstance(expr, (type(None), Expression)) for expr in expressions):
            raise ValueError("All items in expressions must be Expression objects.")
        
        self.expressions = expressions
        
        # 懒编译梯度函数
        self._grad_fn_compiled = None
        
        # 预计算的执行计划（缓存）
        self._execution_plan = None
        self._constant_info = None

    def __len__(self) -> int:
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
        return sum(expr.size for expr in self if expr is not None)

    @property
    def order(self) -> int:
        return sum(1 if expr is not None else 0 for expr in self.expressions)

    def copy(self) -> 'ExpressionSet':
        return ExpressionSet(
            [expr.copy() if expr is not None else None for expr in self.expressions], 
            out_func=self.out_func, 
            metric=self.metric
        )

    def execute(self, X: np.ndarray) -> np.ndarray:
        """NumPy快速执行路径"""
        outputs = [expr.execute(X).reshape(-1, 1) for expr in self.expressions if expr is not None]
        if not outputs:
            return np.array([]).reshape(X.shape[0], 0)

        result = np.hstack(outputs)
        return result if self.out_func is None else self.out_func(result)

    def fitness(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.execute(X)
        if y is None:
            raw_fitness = self.metric(X, y_pred)
        else:
            raw_fitness = self.metric(y, y_pred)
        return np.float32(raw_fitness)

    # ============ 优化的常量信息预处理 ============
    def _build_constant_info(self):
        """
        预计算常量信息，避免运行时查找
        返回：扁平化的常量索引映射
        """
        if self._constant_info is not None:
            return self._constant_info
        
        # 扁平化所有常量
        flat_constant_indices = []  # [(expr_idx, gene_idx), ...]
        expr_constant_ranges = []   # [(start, end), ...] 每个表达式的常量在扁平数组中的范围
        
        global_const_idx = 0
        for expr_idx, expr in enumerate(self.expressions):
            if expr is None:
                expr_constant_ranges.append(None)
                continue
            
            const_indices = expr.constant_indices
            if const_indices:
                start_idx = global_const_idx
                end_idx = global_const_idx + len(const_indices)
                expr_constant_ranges.append((start_idx, end_idx))
                
                # 记录每个常量的位置
                for gene_idx in const_indices:
                    flat_constant_indices.append((expr_idx, gene_idx))
                
                global_const_idx = end_idx
            else:
                expr_constant_ranges.append(None)
        
        self._constant_info = {
            'flat_indices': flat_constant_indices,
            'ranges': expr_constant_ranges,
            'total_constants': global_const_idx
        }
        
        return self._constant_info

    # ============ 优化的执行计划 ============
    def _build_execution_plan(self):
        """
        构建静态执行计划，避免运行时动态处理
        """
        if self._execution_plan is not None:
            return self._execution_plan
        
        const_info = self._build_constant_info()
        
        # 为每个表达式构建指令序列
        expr_plans = []
        for expr_idx, expr in enumerate(self.expressions):
            if expr is None:
                expr_plans.append(None)
                continue
            
            genes_jax = expr._genes_to_jax_executable()
            const_range = const_info['ranges'][expr_idx]
            
            # 预处理：为每个基因标记类型和常量索引
            instructions = []
            local_const_idx = 0
            
            for gene_idx, gene in enumerate(genes_jax):
                if gene.degree == 0:
                    # 检查是否是常量
                    if const_range is not None and gene_idx in expr.constant_indices:
                        # 常量：记录在全局常量数组中的位置
                        global_const_idx = const_range[0] + local_const_idx
                        instructions.append(('const', global_const_idx))
                        local_const_idx += 1
                    else:
                        # 变量或聚合
                        instructions.append(('term', gene))
                else:
                    # 操作符
                    instructions.append(('op', gene))
            
            expr_plans.append({
                'instructions': tuple(instructions),
                'genes': genes_jax
            })
        
        # 转换输出函数
        out_func_jax = None
        if self.out_func is not None:
            out_func_jax = _operator_jax_map[self.out_func.name]
        
        self._execution_plan = {
            'expr_plans': expr_plans,
            'out_func': out_func_jax,
            'const_info': const_info
        }
        
        return self._execution_plan

    # ============ 高效的JAX执行 ============
    def _build_jax_executable(self):
        """
        构建优化的JAX可执行版本
        关键改进：
        1. 使用预计算的执行计划
        2. 扁平化常量数组（避免嵌套索引）
        3. 静态指令序列（避免动态查找）
        """
        plan = self._build_execution_plan()
        expr_plans = plan['expr_plans']
        out_func = plan['out_func']
        
        def _execute_single_expr(instructions, X_jax, constants):
            """执行单个表达式（优化版）"""
            stack = []
            
            for instr_type, instr_data in instructions:
                if instr_type == 'const':
                    # 直接从扁平常量数组中取值
                    stack.append(constants[instr_data])
                elif instr_type == 'term':
                    # 变量或聚合
                    stack.append(instr_data(X_jax))
                elif instr_type == 'op':
                    # 操作符
                    gene = instr_data
                    operands = [stack.pop() for _ in range(gene.degree)]
                    operands.reverse()
                    result = gene(*operands)
                    stack.append(result)
            
            result = stack[0]
            
            # 处理标量
            if result.ndim == 0:
                result = jnp.full(X_jax.shape[0], result)
            
            return result
        
        def _execute_with_constants(X_jax, constants):
            """执行整个表达式集合"""
            results = []
            
            for expr_plan in expr_plans:
                if expr_plan is None:
                    continue
                
                result = _execute_single_expr(
                    expr_plan['instructions'], 
                    X_jax, 
                    constants
                )
                results.append(result.reshape(-1, 1))
            
            if not results:
                return jnp.array([]).reshape(X_jax.shape[0], 0)
            
            results = jnp.hstack(results)
            
            # 应用输出函数
            if out_func is not None:
                results = out_func(results)
            
            return results
        
        return _execute_with_constants

    # ============ 梯度计算 ============
    def _get_gradient_function(self):
        """懒编译梯度函数"""
        if self._grad_fn_compiled is None:
            executable = self._build_jax_executable()
            metric = self.metric
            
            def loss_fn(constants, X_jax, y_jax):
                y_pred = executable(X_jax, constants)
                loss = metric(y_jax, y_pred)
                if metric.greater_is_better:
                    loss = -loss
                return loss
            
            # JIT编译梯度函数
            self._grad_fn_compiled = jax.jit(jax.grad(loss_fn, argnums=0))
        
        return self._grad_fn_compiled

    def compute_constant_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算常量的梯度（优化版）
        """
        const_info = self._build_constant_info()
        
        if const_info['total_constants'] == 0:
            return None, None
        
        # 转换为JAX数组
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        # 提取当前常量值（扁平化）
        current_constants = jnp.array([
            self.expressions[expr_idx].genes[gene_idx].value 
            for expr_idx, gene_idx in const_info['flat_indices']
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
        """更新常量值（优化版）"""
        const_info = self._build_constant_info()
        
        if len(new_values) != const_info['total_constants']:
            raise ValueError(
                f"Expected {const_info['total_constants']} values, got {len(new_values)}"
            )
        
        new_expr_set = self.copy()
        
        # 使用扁平化索引快速更新
        for i, (expr_idx, gene_idx) in enumerate(const_info['flat_indices']):
            new_expr_set.expressions[expr_idx].genes[gene_idx] = Constant(float(new_values[i]))
        
        return new_expr_set

