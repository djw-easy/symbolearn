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
from src.tree import clone_tree, PostOrderIter, SymbolicNode
from src.fitness import Fitness, _fitness_jax_map
from src.node_jax import _operator_jax_map





class Expression(object):
    """
    符号树形式的表达式
    """
    def __init__(self,
                 tree: SymbolicNode,
                 out_func: Operator | None = None, 
                 metric: Fitness | None = None,
                 complexity_of_operators: dict[str, int | float] | None = None,
                 complexity_of_constants: int | float | None = None,
                 complexity_of_variables: int | float | None = None, 
                 complexity_of_aggregations: int | float | None = None):
        self.tree = tree
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

    def _validate_tree(self, tree: SymbolicNode):
        """Recursively validates the genes in the provided tree."""
        if not isinstance(tree, SymbolicNode):
            raise ValueError("The tree must be a SymbolicNode.")
        if tree.size > self.maxsize:
            raise ValueError(f"The tree size {tree.size} exceeds the maximum size {self.maxsize}.")
        valid_operator_names = {op.name for op in self.operators}
        
        for node in PostOrderIter(tree):
            if node.node_content is None:
                raise ValueError("A symbolic node in the symbolic tree has no node_content.")
            
            if isinstance(node.node_content, Operator):
                if node.node_content.name not in valid_operator_names:
                    raise ValueError(f"Invalid operator '{node.node_content.name}' found in the tree.")
            elif not isinstance(node.node_content, (Variable, Constant, DynamicAggregation)):
                 raise ValueError(f"Invalid node type '{type(node.node_content)}' found in the tree.")

            if len(node.children) != node.degree:
                raise ValueError(
                    f"Node '{node.name}' has degree {node.degree} but found {len(node.children)} children."
                )

    def _tree_to_str(self, node: SymbolicNode) -> str:
        """Recursively converts a node to a string formula."""
        if node.is_leaf:
            if isinstance(node.node_content, Constant):
                return f"{node.node_content.value:.3f}"
            elif isinstance(node.node_content, Variable):
                return node.node_content.name
            elif isinstance(node.node_content, DynamicAggregation):
                return node.node_content.name
            return str(node.node_content.name)

        children_strs = [self._tree_to_str(child) for child in node.children]
        
        op_name = node.node_content.name
        if node.degree == 1:
            if op_name == 'neg':
                if isinstance(node.children[0].node_content, Constant):
                    if node.children[0].node_content.value > 0:
                        return f"(-{children_strs[0]})"
                    elif node.children[0].node_content.value == 0:
                        return f"0"
                    else:
                        return f"{-node.children[0].node_content.value:.3f}"
                else:
                    return f"(-{children_strs[0]})"
            elif op_name == 'inv':
                return f"(1/{children_strs[0]})"
            return f"{op_name}({children_strs[0]})"
        elif node.degree == 2:
            op_map = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 
                      'gt': '>', 'lt': '<', 'eq': '=', 'geq': '>=', 'leq': '<='}
            if op_name in op_map.keys():
                symbol = op_map.get(op_name, op_name)
                return f"({children_strs[0]} {symbol} {children_strs[1]})"
            elif op_name in op_map.values():
                return f"({children_strs[0]} {op_name} {children_strs[1]})"
            else:
                return f"{op_name}({', '.join(children_strs)})"
        else:
            return f"{op_name}({', '.join(children_strs)})"

    def __str__(self):
        """Converts the symbolic tree to a string formula."""
        return self._tree_to_str(self.tree)

    def __repr__(self):
        """Provides a developer-friendly representation of the object."""
        return f"Expression(formula='{self._tree_to_str(self.tree)}')"

    def _count_scalar_constants(self) -> int:
        """Recursively counts the number of scalar constants in the tree."""
        count = 0
        for node in PostOrderIter(self.tree):
            if isinstance(node.node_content, Constant):
                count += 1
        return count
    
    def _count_scalar_variables(self) -> int:
        """Recursively counts the number of scalar variables in the tree."""
        count = 0
        for node in PostOrderIter(self.tree):
            if isinstance(node.node_content, Variable):
                count += 1
        return count
    
    def _count_scalar_aggregations(self) -> int:
        """Recursively counts the number of scalar aggregations in the tree."""
        count = 0
        for node in PostOrderIter(self.tree):
            if isinstance(node.node_content, DynamicAggregation):
                count += 1
        return count

    def _has_binary_operator(self) -> bool:
        """Checks if the tree contains any binary operators."""
        for node in PostOrderIter(self.tree):
            if node.degree == 2:
                return True
        return False

    @property
    def constant_indices(self):
        if self._constant_indices is not None:
            return self._constant_indices
        self._constant_indices = [i for i, node in enumerate(PostOrderIter(self.tree)) 
                                  if isinstance(node.node_content, Constant)]
        return self._constant_indices

    @property
    def size(self) -> int:
        """The size of the expression tree."""
        return self.tree.size

    def _calculate_complexity(self) -> float:
        """
        计算表达式的复杂度。
        """
        total_complexity = 0.0
        for node in PostOrderIter(self.tree):
            if isinstance(node.node_content, Operator):
                total_complexity += self.complexity_of_operators.get(node.name, 1)
            elif isinstance(node.node_content, Constant):
                total_complexity += self.complexity_of_constants
            elif isinstance(node.node_content, Variable):
                total_complexity += self.complexity_of_variables
            elif isinstance(node.node_content, DynamicAggregation):
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
    def _trees_are_equal(node1: SymbolicNode, node2: SymbolicNode) -> bool:
        """Recursively checks if two symbolic node trees are identical."""
        if node1 is None and node2 is None:
            return True
        if node1 is None or node2 is None:
            return False
        # Compare operators first
        if node1.node_content != node2.node_content:
            return False
        # Compare number of children
        if len(node1.children) != len(node2.children):
            return False
        # Recursively compare children
        return all(Expression._trees_are_equal(c1, c2) for c1, c2 in zip(node1.children, node2.children))

    def __eq__(self, other):
        """Recursively checks if two expression trees are identical."""
        if not isinstance(other, Expression):
            return NotImplementedError
        return Expression._trees_are_equal(self.tree, other.tree)

    def copy(self) -> 'Expression':
        """Returns a deep copy of the expression."""
        return Expression(
            clone_tree(self.tree), out_func=self.out_func, metric=self.metric
        )

    def execute(self, X: np.ndarray) -> np.ndarray:
        """Execute the expression according to X."""
        result = self.tree(X)

        if not isinstance(result, np.ndarray) or result.ndim == 0:
            result = np.full(X.shape[0], result)

        result = result.ravel()
        return result if self.out_func is None else self.out_func(result)

    def _execute_postorder(self, X: np.ndarray, constants: np.ndarray = None) -> np.ndarray:
        stack = []
        
        constant_counter = 0
        for node in PostOrderIter(self.tree):
            if node.degree == 0:
                # Variable, Constant, DynamicAggregation
                if (constants is not None) and isinstance(node.node_content, Constant):
                    result = constants[constant_counter]
                    constant_counter += 1
                else:
                    result = node.node_content(X)
                stack.append(result)
            else:
                # Operator
                operands = [stack.pop() for _ in range(node.degree)]
                operands.reverse()
                result = node.node_content(*operands)
                stack.append(result)
        
        result = stack[0]
        
        # 处理标量
        if np.isscalar(result) or result.ndim == 0:
            result = np.full(X.shape[0], result)
        
        # 应用输出函数
        if self.out_func is not None:
            result = self.out_func(result)
        
        return result

    def fitness(self, X: np.ndarray, y: np.ndarray, 
                constants: np.ndarray = None) -> np.float32:
        """Evaluate the raw fitness of the expression according to X, y."""
        if constants is None:
            y_pred = self.execute(X)
        else:
            y_pred = self._execute_postorder(X, constants)
        raw_fitness = self.metric(y, y_pred)
        return np.float32(raw_fitness)

    # ============ JAX梯度计算（仅用于常量优化） ============
    def _tree_to_jax_executable(self):
        """
        将基因列表转换为JAX可执行版本（仅在需要梯度时调用）
        关键优化：使用静态结构避免动态循环
        """
        genes = []
        for node in PostOrderIter(self.tree):
            if node.degree > 0:
                genes.append(_operator_jax_map[node.name])
            elif isinstance(node.node_content, DynamicAggregation):
                genes.append(
                    DynamicAggregation_jax(
                        v_start=node.node_content.v_start, 
                        v_end=node.node_content.v_end, 
                        op_name=node.node_content.op_name, 
                        n_variables=node.node_content.n_variables,
                        valid_op=node.node_content.valid_op
                    )
                )
            else:
                genes.append(node.node_content)
        return genes

    def _build_jax_executable(self):
        """
        构建JAX可执行版本（仅在需要梯度时调用）
        关键优化：使用静态结构避免动态循环
        """
        genes = self._tree_to_jax_executable()
        
        out_func = self.out_func
        out_func = _operator_jax_map[out_func.name] if out_func is not None else None
        
        # 提取常量索引映射
        constant_indices = self.constant_indices
        const_idx_map = {idx: i for i, idx in enumerate(constant_indices)}
        
        def _execute_with_constants(X_jax: jnp.ndarray, constants: jnp.ndarray):
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
            metric = _fitness_jax_map[self.metric.name]
            
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
    
    def compute_constant_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.float32]:
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
            node.node_content.value for node in PostOrderIter(self.tree) 
                if isinstance(node.node_content, Constant)
        ])
        
        # 计算梯度
        grad_fn = self._get_gradient_function()
        gradients = grad_fn(current_constants, X_jax, y_jax)
        
        # 计算当前损失
        executable = self._build_jax_executable()
        y_pred = executable(X_jax, current_constants)
        loss = self.metric(y_jax, y_pred)
        if self.metric.greater_is_better:
            loss = -loss
        
        return np.array(gradients), loss
    
    def update_constants(self, new_values: np.ndarray):
        """更新常量值"""
        if len(new_values) != len(self.constant_indices):
            raise ValueError(f"Expected {len(self.constant_indices)} values, got {len(new_values)}")
        
        new_expr = self.copy()
        constant_counter = 0
        for i, node in enumerate(PostOrderIter(new_expr.tree)):
            if isinstance(node.node_content, Constant):
                node.node_content = Constant(new_values[constant_counter])
                constant_counter += 1
        
        return new_expr




class ExpressionSet(object):
    def __init__(self, 
                 expressions: List[Optional['Expression']], 
                 out_func: Operator | None = None, 
                 metric: Fitness | None = None):
        self.out_func = out_func
        self.metric = metric
        
        if not all(isinstance(expr, (type(None), type(expressions[0]))) for expr in expressions):
            raise ValueError("All items in expressions must be Expression objects.")
        
        self.expressions = expressions
        
        # 预分析常量信息
        self._constant_info = None
        
        # 懒编译梯度函数
        self._grad_fn_compiled = None
        
        # 预计算的执行计划（缓存）
        self._execution_plan = None

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

    # ============ NumPy快速执行 ============
    def execute(self, X: np.ndarray) -> np.ndarray:
        """NumPy快速执行路径（不带常量参数）"""
        outputs = [expr.execute(X).reshape(-1, 1) for expr in self.expressions if expr is not None]
        if not outputs:
            return np.array([]).reshape(X.shape[0], 0)

        result = np.hstack(outputs)
        return result if self.out_func is None else self.out_func(result)

    def _execute_postorder(self, X: np.ndarray, constants: np.ndarray = None) -> np.ndarray:
        """
        使用后序遍历执行表达式集合（支持传入常量数组）
        参考Expression._execute_postorder的实现
        """
        if constants is None:
            return self.execute(X)
        
        # 构建常量信息（如果未缓存）
        const_info = self._build_constant_info()
        
        # 执行每个表达式
        outputs = []
        for expr_idx, expr in enumerate(self.expressions):
            if expr is None:
                continue
            
            # 获取该表达式的常量范围
            const_range = const_info['ranges'][expr_idx]
            
            if const_range is None:
                # 该表达式没有常量，直接执行
                result = expr.execute(X)
            else:
                # 提取该表达式的常量
                expr_constants = constants[const_range[0]:const_range[1]]
                # 使用表达式的后序执行方法
                result = expr._execute_postorder(X, expr_constants)
            
            outputs.append(result.reshape(-1, 1))
        
        if not outputs:
            return np.array([]).reshape(X.shape[0], 0)
        
        result = np.hstack(outputs)
        return result if self.out_func is None else self.out_func(result)

    def fitness(self, X: np.ndarray, y: np.ndarray, 
                constants: np.ndarray = None) -> np.float32:
        """
        评估表达式集合的适应度
        
        参数:
            X: 输入数据
            y: 目标数据
            constants: 可选的常量数组，如果提供则替换表达式中的常量
        """
        if constants is None:
            y_pred = self.execute(X)
        else:
            y_pred = self._execute_postorder(X, constants)
        
        if y is None:
            raw_fitness = self.metric(X, y_pred)
        else:
            raw_fitness = self.metric(y, y_pred)
        
        return np.float32(raw_fitness)

    # ============ 常量信息预处理 ============
    def _build_constant_info(self):
        """
        预计算常量信息，扁平化所有表达式的常量索引
        返回：
            {
                'flat_indices': [(expr_idx, node), ...],  # 所有常量的扁平列表
                'ranges': [(start, end), ...],            # 每个表达式的常量范围
                'total_constants': int                    # 总常量数
            }
        """
        if self._constant_info is not None:
            return self._constant_info
        
        flat_constant_nodes = []  # 存储 (expr_idx, node)
        expr_constant_ranges = []  # 存储每个表达式的常量范围
        
        global_const_idx = 0
        for expr_idx, expr in enumerate(self.expressions):
            if expr is None:
                expr_constant_ranges.append(None)
                continue
            
            # 收集该表达式的所有常量节点
            expr_constants = []
            for node in PostOrderIter(expr.tree):
                if isinstance(node.node_content, Constant):
                    expr_constants.append(node)
            
            if expr_constants:
                start_idx = global_const_idx
                end_idx = global_const_idx + len(expr_constants)
                expr_constant_ranges.append((start_idx, end_idx))
                
                # 记录每个常量节点
                for node in expr_constants:
                    flat_constant_nodes.append((expr_idx, node))
                
                global_const_idx = end_idx
            else:
                expr_constant_ranges.append(None)
        
        self._constant_info = {
            'flat_nodes': flat_constant_nodes,
            'ranges': expr_constant_ranges,
            'total_constants': global_const_idx
        }
        
        return self._constant_info

    # ============ JAX梯度计算 ============
    def _build_execution_plan(self):
        """
        构建静态执行计划
        为每个表达式预处理指令序列
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
            
            # 转换为JAX可执行的基因列表
            genes_jax = self._expr_to_jax_genes(expr)
            const_range = const_info['ranges'][expr_idx]
            
            # 构建指令序列
            instructions = []
            local_const_idx = 0
            
            for gene_idx, gene in enumerate(genes_jax):
                if gene.degree == 0:
                    # 检查是否是常量
                    if const_range is not None:
                        # 需要检查该位置是否对应常量节点
                        nodes = list(PostOrderIter(expr.tree))
                        if isinstance(nodes[gene_idx].node_content, Constant):
                            # 记录在全局常量数组中的位置
                            global_const_idx = const_range[0] + local_const_idx
                            instructions.append(('const', global_const_idx))
                            local_const_idx += 1
                        else:
                            # 变量或聚合
                            instructions.append(('term', gene))
                    else:
                        # 该表达式没有常量
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

    def _expr_to_jax_genes(self, expr):
        """
        将表达式转换为JAX可执行的基因列表
        参考Expression._tree_to_jax_executable
        """
        genes = []
        for node in PostOrderIter(expr.tree):
            if node.degree > 0:
                # 操作符
                genes.append(_operator_jax_map[node.name])
            elif hasattr(node.node_content, 'op_name'):
                # DynamicAggregation
                genes.append(
                    DynamicAggregation_jax(
                        v_start=node.node_content.v_start,
                        v_end=node.node_content.v_end,
                        op_name=node.node_content.op_name,
                        n_variables=node.node_content.n_variables,
                        valid_op=getattr(node.node_content, 'valid_op', None)
                    )
                )
            else:
                # Variable 或 Constant
                genes.append(node.node_content)
        return genes

    def _build_jax_executable(self):
        """
        构建优化的JAX可执行版本
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

    def _get_gradient_function(self):
        """懒编译梯度函数"""
        if self._grad_fn_compiled is None:
            executable = self._build_jax_executable()
            metric = _fitness_jax_map[self.metric.name]
            
            def loss_fn(constants, X_jax, y_jax):
                y_pred = executable(X_jax, constants)
                loss = metric(y_jax, y_pred)
                if metric.greater_is_better:
                    loss = -loss
                return loss
            
            # JIT编译梯度函数
            self._grad_fn_compiled = jax.jit(jax.grad(loss_fn, argnums=0))
        
        return self._grad_fn_compiled

    def compute_constant_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.float32]:
        """
        计算常量的梯度（优化版）
        返回：(梯度数组, 当前损失值)
        """
        const_info = self._build_constant_info()
        
        if const_info['total_constants'] == 0:
            return None, None
        
        # 转换为JAX数组
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        # 提取当前常量值（扁平化）
        current_constants = jnp.array([
            node.node_content.value 
            for expr_idx, node in const_info['flat_nodes']
        ])
        
        # 计算梯度
        grad_fn = self._get_gradient_function()
        gradients = grad_fn(current_constants, X_jax, y_jax)
        
        # 计算当前损失
        executable = self._build_jax_executable()
        y_pred = executable(X_jax, current_constants)
        loss = self.metric(y_jax, y_pred)
        if self.metric.greater_is_better:
            loss = -loss
        
        return np.array(gradients), np.float32(loss)

    def update_constants(self, new_values: np.ndarray) -> 'ExpressionSet':
        const_info = self._build_constant_info()
        
        if len(new_values) != const_info['total_constants']:
            raise ValueError(
                f"Expected {const_info['total_constants']} values, got {len(new_values)}"
            )
        
        # 为每个表达式准备新常量
        new_expressions = []
        for expr_idx, expr in enumerate(self.expressions):
            if expr is None:
                new_expressions.append(None)
                continue
            
            const_range = const_info['ranges'][expr_idx]
            if const_range is not None:
                expr_constants = new_values[const_range[0]:const_range[1]]
                new_expressions.append(expr.update_constants(expr_constants))
            else:
                new_expressions.append(expr.copy())
        
        # 直接创建新对象（避免先copy再修改）
        return ExpressionSet(new_expressions, self.out_func, self.metric)




