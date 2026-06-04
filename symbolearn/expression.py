import bisect
import warnings
import numpy as np
from scipy import sparse
from typing import Union, Optional, List


from symbolearn.fitness import Fitness
from symbolearn.node import Operator, Constant, Variable, DynamicAggregation
from symbolearn.tree import clone_tree, PreOrderIter, PostOrderIter, SymbolicNode




class Expression(object):
    """
    A symbolic mathematical expression represented as a tree structure.

    This class encapsulates a symbolic expression tree (SymbolicNode) along with
    associated metadata such as the fitness metric, output transformation function,
    and complexity configuration. Expressions are the fundamental individuals
    evolved by the genetic programming algorithm.

    The expression tree consists of:
    - Internal nodes: Operators (mathematical functions like add, multiply, sin, etc.)
    - Leaf nodes: Variables (input features), Constants (literal numeric values),
                 or DynamicAggregations (spatial/spectral aggregations)

    Expression trees can be:
    - Evaluated on input data to produce predictions
    - Simplified using algebraic rules
    - Mutated or crossed over with other expressions
    - Serialized to string representations for human interpretation

    Parameters
    ----------
    tree : SymbolicNode
        The root node of the symbolic expression tree.
    out_func : Operator, optional
        A unary operator applied to the output of the expression during evaluation.
        For classification, this is typically softmax or sigmoid.
    metric : Fitness, optional
        The fitness function used to evaluate the expression quality.
    ndigits : int, default=7
        Number of digits for formatting floating-point constant values in string output.
    complexity_of_operators : dict, optional
        Custom complexity weights for each operator type. Keys are operator names,
        values are the complexity cost. Default uses unit complexity.
    complexity_of_constants : float, default=1.0
        Complexity cost of constant leaf nodes.
    complexity_of_variables : float or list, default=1.0
        Complexity cost of variable leaf nodes. Can be a list to assign
        different costs to different variables.
    complexity_of_aggregations : float, default=1.0
        Complexity cost of dynamic aggregation leaf nodes.
    constraints : dict, optional
        Operator argument complexity constraints. Format:
        - Unary operators: {'op_name': max_complexity}
        - Binary operators: {'op_name': (max_left, max_right)} or {'op_name': max_both}
        Use -1 or None for unlimited complexity.
    nested_constraints : dict, optional
        Limits on consecutive nesting of operators. Format:
        {'outer_op': {'inner_op': max_depth}}
        depth=0 means forbidden, -1 means unlimited.

    Attributes
    ----------
    tree : SymbolicNode
        The root node of the expression tree.
    metric : Fitness
        The fitness function for evaluation.
    out_func : Operator or None
        Output transformation function.
    complexity : float
        The complexity score of the expression (sum of node complexities).

    Examples
    --------
    >>> from symbolearn.tree import SymbolicNode
    >>> from symbolearn.node import add2, Variable, Constant
    >>> # Create: x0 + 2.5
    >>> const_node = SymbolicNode(node_content=Constant(2.5), degree=0)
    >>> var_node = SymbolicNode(node_content=Variable(0), degree=0)
    >>> root = SymbolicNode(node_content=add2)
    >>> root.children = [var_node, const_node]
    >>> expr = Expression(tree=root, metric=Fitness(mse_func, False))
    >>> result = expr.execute(X)  # Evaluate on data

    Notes
    -----
    The complexity of an expression is computed as the weighted sum of its nodes,
    where operators, variables, constants, and aggregations can each have
    custom complexity weights. This enables the evolutionary process to
    prefer simpler expressions through Pareto-based selection.

    Expression simplification applies algebraic rules including:
    - Constant folding: (2 + 3) -> 5
    - Identity laws: x + 0 -> x, x * 1 -> x
    - Zero laws: x * 0 -> 0, 0 / x -> 0
    - Absorption laws: x - x -> 0, x / x -> 1
    - Double negation: -(-x) -> x
    """
    def __init__(self,
                 tree: SymbolicNode,
                 out_func: Optional[Operator] = None, 
                 metric: Optional[Fitness] = None, ndigits: int = 7,
                 complexity_of_operators: dict[str, int | float] | None = None,
                 complexity_of_constants: int | float | None = None,
                 complexity_of_variables: int | float | list[int | float] | None = None, 
                 complexity_of_aggregations: int | float | None = None,
                 constraints: dict[str, int | tuple[int, int]] | None = None,
                 nested_constraints: dict[str, dict] | None = None):
        self.tree = tree
        self.metric = metric
        self.ndigits = ndigits
        self.out_func = out_func
        self.complexity_of_operators = complexity_of_operators or {}
        self.complexity_of_constants = complexity_of_constants or 1
        self.complexity_of_variables = complexity_of_variables or 1
        self.complexity_of_aggregations = complexity_of_aggregations or 1
        self.constraints = constraints or {}
        self.nested_constraints = nested_constraints or {}
        
        # 预分析表达式结构
        self._constant_indices = None
        
        self._should_calc_complexity = False
        # 检查操作符复杂度
        if self.complexity_of_operators:
            for op_complexity in self.complexity_of_operators.values():
                if abs(op_complexity - 1.0) > 1e-6:  # 不等于1
                    self._should_calc_complexity = True
        # 检查常量复杂度
        if abs(self.complexity_of_constants - 1.0) > 1e-6:
            self._should_calc_complexity = True
        # 检查变量复杂度
        if isinstance(self.complexity_of_variables, (list, tuple)):
            for var_complexity in self.complexity_of_variables:
                if abs(var_complexity - 1.0) > 1e-6:
                    self._should_calc_complexity = True
        else:
            if abs(self.complexity_of_variables - 1.0) > 1e-6:
                self._should_calc_complexity = True
        # 检查聚合复杂度
        if abs(self.complexity_of_aggregations - 1.0) > 1e-6:
            self._should_calc_complexity = True

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

    def to_str(self, node: SymbolicNode) -> str:
        """Recursively converts a node to a string formula."""
        if node.is_leaf:
            if isinstance(node.node_content, Constant):
                return str(round(float(node.node_content.value), self.ndigits))
            elif isinstance(node.node_content, Variable):
                return node.node_content.name
            elif isinstance(node.node_content, DynamicAggregation):
                return node.node_content.name
            return str(node.node_content.name)

        children_strs = [self.to_str(child) for child in node.children]
        
        op_name = node.node_content.name
        if node.degree == 1:
            if op_name == 'neg':
                if isinstance(node.children[0].node_content, Constant):
                    if node.children[0].node_content.value > 0:
                        return f"(-{children_strs[0]})"
                    elif node.children[0].node_content.value == 0:
                        return f"0"
                    else:
                        return f"{-node.children[0].node_content.value:.5f}"
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
        return self.to_str(self.tree)

    def __repr__(self):
        """Provides a developer-friendly representation of the object."""
        return f"Expression(formula='{self.to_str(self.tree)}')"

    def _count_scalar_constants(self) -> int:
        """Recursively counts the number of scalar constants in the tree."""
        count = 0
        for node in PreOrderIter(self.tree):
            if isinstance(node.node_content, Constant):
                count += 1
        return count
    
    def _count_scalar_variables(self) -> int:
        """Recursively counts the number of scalar variables in the tree."""
        count = 0
        for node in PreOrderIter(self.tree):
            if isinstance(node.node_content, Variable):
                count += 1
        return count
    
    def _count_scalar_aggregations(self) -> int:
        """Recursively counts the number of scalar aggregations in the tree."""
        count = 0
        for node in PreOrderIter(self.tree):
            if isinstance(node.node_content, DynamicAggregation):
                count += 1
        return count

    def _has_constants(self) -> bool:
        """Check if the tree contains any constants."""
        for node in PostOrderIter(self.tree):
            if isinstance(node.node_content, Constant):
                return True
        return False

    def _has_variables(self) -> bool:
        """Check if the tree contains any variables."""
        for node in PostOrderIter(self.tree):
            if isinstance(node.node_content, Variable):
                return True
        return False

    def _has_aggregations(self) -> bool:
        """Check if the tree contains any aggregations."""
        for node in PostOrderIter(self.tree):
            if isinstance(node.node_content, DynamicAggregation):
                return True
        return False

    def _has_binary_operator(self) -> bool:
        """Checks if the tree contains any binary operators."""
        for node in PreOrderIter(self.tree):
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
    def constants(self) -> np.ndarray:
        current_constants = [node.node_content.value 
                             for i, node in enumerate(PostOrderIter(self.tree)) 
                             if isinstance(node.node_content, Constant)]
        return np.array(current_constants)

    @property
    def size(self) -> int:
        """The size of the expression tree."""
        return self.tree.size

    def _get_node_complexity(self, node: SymbolicNode) -> float:
        """
        Get the complexity score for a single node.

        Parameters
        ----------
        node : SymbolicNode
            The node to evaluate.

        Returns
        -------
        float
            Complexity score based on node type and configured weights.
        """
        if isinstance(node.node_content, Operator):
            return self.complexity_of_operators.get(node.name, 1)
        elif isinstance(node.node_content, Constant):
            return self.complexity_of_constants
        elif isinstance(node.node_content, Variable):
            # Support per-variable complexity via list
            if isinstance(self.complexity_of_variables, list):
                var_idx = node.node_content.variable
                return self.complexity_of_variables[var_idx] if var_idx < len(self.complexity_of_variables) else 1
            else:
                return self.complexity_of_variables
        elif isinstance(node.node_content, DynamicAggregation):
            return self.complexity_of_aggregations
        else:
            return 1

    def _calculate_complexity(self, node: Optional[SymbolicNode] = None) -> float:
        """
        Calculate the complexity of an expression or subtree.

        Complexity is computed as the weighted sum of all node complexities
        in post-order traversal.

        Parameters
        ----------
        node : SymbolicNode, optional
            Root node of the subtree to evaluate. If None, evaluates entire tree.

        Returns
        -------
        float
            Total complexity score.
        """
        if node is None:
            node = self.tree
        
        total_complexity = 0.0
        for n in PostOrderIter(node):
            total_complexity += self._get_node_complexity(n)
        
        return total_complexity

    @property
    def complexity(self) -> float:
        """
        The complexity score of the expression.

        Returns the weighted complexity if custom weights are configured,
        otherwise returns the tree size.
        """
        if self._should_calc_complexity:
            return self._calculate_complexity()
        return self.size

    def _check_constraints(self, tree: Optional[SymbolicNode] = None) -> bool:
        """
        Check whether the expression satisfies all constraint conditions.

        Parameters
        ----------
        tree : SymbolicNode, optional
            Root node to check. If None, checks entire tree.

        Returns
        -------
        bool
            True if all constraints are satisfied.
        """
        if tree is None:
            tree = self.tree
        
        # Check operator argument complexity constraints
        if not self._check_operator_constraints(tree):
            return False
        
        # Check nested operator depth constraints
        if not self._check_nested_constraints(tree):
            return False
        
        return True

    def _check_operator_constraints(self, tree: SymbolicNode) -> bool:
        """
        Check operator argument complexity constraints.

        Constraint format:
        - Unary operators: {'sin': 3} means sin's argument complexity <= 3
        - Binary operators: {'pow': (-1, 1)} means left arg unlimited, right arg <= 1

        Parameters
        ----------
        tree : SymbolicNode
            Root node to check.

        Returns
        -------
        bool
            True if all operator constraints are satisfied.
        """
        for node in PreOrderIter(tree):
            if not isinstance(node.node_content, Operator):
                continue
            
            op_name = node.node_content.name
            if op_name not in self.constraints:
                continue
            
            constraint = self.constraints[op_name]
            
            if node.degree == 1:
                # Unary operator
                if isinstance(constraint, int):
                    max_complexity = constraint
                    if max_complexity >= 0:  # -1 means unlimited
                        child_complexity = self._calculate_complexity(node.children[0])
                        if child_complexity > max_complexity:
                            return False
                elif isinstance(constraint, tuple):
                    # Allow tuple format (max_complexity,) or (max_complexity, -1)
                    max_complexity = constraint[0]
                    if max_complexity >= 0:
                        child_complexity = self._calculate_complexity(node.children[0])
                        if child_complexity > max_complexity:
                            return False
            
            elif node.degree == 2:
                # Binary operator
                if isinstance(constraint, tuple) and len(constraint) == 2:
                    max_left, max_right = constraint
                    
                    # Check left argument
                    if max_left >= 0:
                        left_complexity = self._calculate_complexity(node.children[0])
                        if left_complexity > max_left:
                            return False
                    
                    # Check right argument
                    if max_right >= 0:
                        right_complexity = self._calculate_complexity(node.children[1])
                        if right_complexity > max_right:
                            return False
                elif isinstance(constraint, int):
                    # Allow single integer for both arguments
                    max_complexity = constraint
                    if max_complexity >= 0:
                        for child in node.children:
                            child_complexity = self._calculate_complexity(child)
                            if child_complexity > max_complexity:
                                return False
        
        return True

    def _check_nested_constraints(self, tree: SymbolicNode) -> bool:
        """
        Validate nested operator depth constraints.

        Constraint semantics:
        nested_constraints[outer][inner] = k
            - k == 0: inner cannot be a direct child of outer
            - k >= 1: if inner is a direct child, continuous nesting depth <= k
            - k == -1: allowed (skip check)

        Parameters
        ----------
        tree : SymbolicNode
            Root node to check.

        Returns
        -------
        bool
            True if all nested constraints are satisfied.
        """
        if not self.nested_constraints:
            return True

        # Compute continuous depth of same-name operator nesting on-demand
        def get_continuous_depth(node: SymbolicNode) -> int:
            if node.degree == 0:
                return 0
            op = node.name
            max_child_depth = 0
            for child in node.children:
                if child.degree > 0 and child.name == op:
                    child_depth = get_continuous_depth(child)
                    if child_depth > max_child_depth:
                        max_child_depth = child_depth
            return 1 + max_child_depth

        # Traverse all nodes and check direct child constraints
        for node in PreOrderIter(tree):
            if node.degree == 0:
                continue

            outer_op = node.name
            if outer_op not in self.nested_constraints:
                continue

            child_constraints = self.nested_constraints[outer_op]

            for child in node.children:
                if child.degree == 0:
                    continue  # Skip leaves (variables/constants)

                child_op = child.name
                if child_op not in child_constraints:
                    continue

                k = child_constraints[child_op]

                if k == 0:
                    return False
                elif k > 0:
                    # Only compute depth if k > 0
                    if get_continuous_depth(child) > k:
                        return False
                # k == -1: allowed, skip check

        return True

    def is_valid(self, max_complexity: Optional[float] = None) -> bool:
        """
        Check whether the expression is valid.

        An expression is valid if:
        1. Its complexity does not exceed max_complexity (if specified)
        2. All operator constraints are satisfied
        3. All nested constraints are satisfied

        Parameters
        ----------
        max_complexity : float, optional
            Maximum allowed complexity. If None, complexity is not checked.

        Returns
        -------
        bool
            True if the expression is valid.
        """
        # Check complexity
        if max_complexity is not None:
            if self.complexity > max_complexity:
                return False
        
        # Check constraints
        if not self._check_constraints():
            return False
        
        return True

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
            clone_tree(self.tree), 
            out_func=self.out_func, 
            metric=self.metric, ndigits=self.ndigits,
            complexity_of_operators=self.complexity_of_operators,
            complexity_of_constants=self.complexity_of_constants,
            complexity_of_variables=self.complexity_of_variables,
            complexity_of_aggregations=self.complexity_of_aggregations,
            constraints=self.constraints, nested_constraints=self.nested_constraints
        )

    def _execute_tree(self, X: np.ndarray) -> np.ndarray:
        """
        Execute the expression on input data using direct tree evaluation.

        Parameters
        ----------
        X : ndarray
            Input features with shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Expression output with shape dependent on expression structure.
        """
        result = self.tree(X)

        # Handle scalar results
        if np.isscalar(result) or result.ndim == 0:
            result = np.full(X.shape[:-1], result)

        # Apply output function (e.g., softmax for classification)
        if self.out_func is not None:
            result = self.out_func(result)
        
        return result

    def _execute_postorder(self, X: np.ndarray, valid_mask: Optional[np.ndarray] = None, 
                           constants: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute the expression using post-order traversal.

        This method supports masked evaluation and optimized constant handling
        for scenarios like batch processing with missing values.

        Parameters
        ----------
        X : ndarray
            Input features.
        valid_mask : ndarray, optional
            Boolean mask indicating valid samples.
        constants : ndarray, optional
            Pre-computed constant values for optimization.

        Returns
        -------
        ndarray
            Expression output for valid samples only.
        """
        stack = []
        
        constant_counter = 0
        for node in PostOrderIter(self.tree):
            if node.degree == 0:
                # Variable, Constant, DynamicAggregation
                if isinstance(node.node_content, Constant):
                    if constants is not None:
                        result = constants[constant_counter]
                    else:
                        result = node.node_content.value
                    constant_counter += 1
                elif isinstance(node.node_content, (Variable, DynamicAggregation)):
                    result = node.node_content(X, valid_mask=valid_mask)
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
        
        # Handle scalar results
        if np.isscalar(result) or result.ndim == 0:
            result = np.full(X.shape[:-1], result)
            result = result[valid_mask] if valid_mask is not None else result
        
        # Apply output function
        if self.out_func is not None:
            result = self.out_func(result)
        
        return result

    def execute(self, X: np.ndarray, valid_mask: Optional[np.ndarray] = None, 
                constants: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute the expression on input data.

        Automatically selects the appropriate execution strategy based on
        whether masking or optimized constants are needed.

        Parameters
        ----------
        X : ndarray
            Input features.
        valid_mask : ndarray, optional
            Boolean mask for valid samples.
        constants : ndarray, optional
            Pre-computed constant values.

        Returns
        -------
        ndarray
            Expression output.
        """
        if constants is None and valid_mask is None:
            result = self._execute_tree(X)
        else:
            result = self._execute_postorder(X, valid_mask, constants)
        
        return result

    def fitness(self, X: np.ndarray, y: np.ndarray, 
               sample_weight: Optional[np.ndarray] = None,
               constants: Optional[np.ndarray] = None) -> np.float32:
        """
        Evaluate the fitness of the expression on given data.

        Parameters
        ----------
        X : ndarray
            Input features.
        y : ndarray
            Target values.
        sample_weight : ndarray, optional
            Sample weights for weighted evaluation.
        constants : ndarray, optional
            Pre-computed constant values.

        Returns
        -------
        np.float32
            Fitness value (higher is better if metric.greater_is_better).
        """
        raw_fitness = self.metric(self, X, y, constants=constants,
                                  sample_weight=sample_weight)

        return np.float32(raw_fitness)
    
    def update_constants(self, new_values: np.ndarray):
        """
        Update constant values in the expression tree.

        Parameters
        ----------
        new_values : ndarray
            New values for constants in post-order traversal order.

        Returns
        -------
        Expression
            Self (modified in place).
        """
        if len(new_values) != len(self.constant_indices):
            raise ValueError(f"Expected {len(self.constant_indices)} values, got {len(new_values)}")
        
        constant_counter = 0
        for i, node in enumerate(PostOrderIter(self.tree)):
            if isinstance(node.node_content, Constant):
                node.node_content = Constant(new_values[constant_counter])
                constant_counter += 1
        
        return self

    def simplify(self, constants_tolerance: float = 1e-5) -> 'Expression':
        """
        Simplify the expression by applying algebraic rewrite rules.

        Simplification Rules
        --------------------
        1. Constant folding: Evaluate constant expressions (e.g., 2 + 3 -> 5)
        2. Identity laws: x + 0 -> x, x * 1 -> x, x / 1 -> x
        3. Zero laws: x * 0 -> 0, 0 / x -> 0
        4. Algebraic absorption: x - x -> 0, x / x -> 1
        5. Double negation: -(-x) -> x
        6. Distribution and association where applicable

        Parameters
        ----------
        constants_tolerance : float, default=1e-5
            Tolerance for considering constants as zero or one.

        Returns
        -------
        Expression
            Simplified expression (new object).
        """
        new_expr = self.copy()
        
        # Iterate simplification until no further changes
        max_iterations = 10
        for iteration in range(max_iterations):
            original_tree = clone_tree(new_expr.tree)
            new_expr.tree = self._recursive_simplify(new_expr.tree, constants_tolerance)
            
            # If tree unchanged, simplification is complete
            if Expression._trees_are_equal(original_tree, new_expr.tree):
                break
        
        return new_expr

    def _recursive_simplify(self, node: SymbolicNode, tolerance: float) -> SymbolicNode:
        """
        Recursively simplify a symbolic tree node.

        Applies algebraic simplification rules in post-order:
        first simplifies children, then applies node-level rules.

        Parameters
        ----------
        node : SymbolicNode
            Node to simplify.
        tolerance : float
            Tolerance for considering constants as zero or one.

        Returns
        -------
        SymbolicNode
            Simplified node (may be a new node or the original).
        """
        # Base Case: Leaf node
        if node.is_leaf:
            if isinstance(node.node_content, Constant):
                value = node.node_content.value
                # Values near zero become zero
                if abs(value) < tolerance:
                    return SymbolicNode(node_content=Constant(0.0))
                # Values near one become one
                if abs(value - 1.0) < tolerance:
                    return SymbolicNode(node_content=Constant(1.0))
            return node

        # Recursive Step: Simplify all children first
        node.children = [self._recursive_simplify(child, tolerance) for child in node.children]

        op_name = node.node_content.name
        children = node.children

        # Constant folding: all children are constants
        if all(isinstance(child.node_content, Constant) for child in children):
            try:
                child_values = [child.node_content.value for child in children]
                new_value = node.node_content(*child_values)
                return SymbolicNode(node_content=Constant(new_value))
            except Exception:
                # If computation fails (e.g., division by zero), keep as-is
                pass

        # Binary operator simplification rules
        if node.degree == 2:
            left, right = children[0], children[1]
            left_op, right_op = left.node_content, right.node_content

            # Addition rules
            if op_name == 'add' or op_name == '+':
                # x + 0 = x, 0 + x = x
                if isinstance(right_op, Constant) and abs(right_op.value) < tolerance:
                    return left
                if isinstance(left_op, Constant) and abs(left_op.value) < tolerance:
                    return right
                # x + (-y) = x - y
                if isinstance(right_op, Operator) and right_op.name in ('neg', '-'):
                    if right_op.degree == 1:
                        sub_op = self._get_operator_by_name('sub')
                        if sub_op:
                            new_node = SymbolicNode(node_content=sub_op)
                            new_node.children = [left, right.children[0]]
                            return new_node

            # Subtraction rules
            elif op_name == 'sub' or op_name == '-':
                # x - 0 = x
                if isinstance(right_op, Constant) and abs(right_op.value) < tolerance:
                    return left
                # 0 - x = -x
                if isinstance(left_op, Constant) and abs(left_op.value) < tolerance:
                    neg_op = self._get_operator_by_name('neg')
                    if neg_op:
                        new_node = SymbolicNode(node_content=neg_op)
                        new_node.children = [right]
                        return new_node
                # x - x = 0
                if Expression._trees_are_equal(left, right):
                    return SymbolicNode(node_content=Constant(0.0))
                # x - (-y) = x + y
                if isinstance(right_op, Operator) and right_op.name in ('neg', '-'):
                    if right_op.degree == 1:
                        add_op = self._get_operator_by_name('add')
                        if add_op:
                            new_node = SymbolicNode(node_content=add_op)
                            new_node.children = [left, right.children[0]]
                            return new_node

            # Multiplication rules
            elif op_name == 'mul' or op_name == '*':
                # x * 1 = x, 1 * x = x
                if isinstance(right_op, Constant) and abs(right_op.value - 1.0) < tolerance:
                    return left
                if isinstance(left_op, Constant) and abs(left_op.value - 1.0) < tolerance:
                    return right
                # x * 0 = 0, 0 * x = 0
                if isinstance(right_op, Constant) and abs(right_op.value) < tolerance:
                    return SymbolicNode(node_content=Constant(0.0))
                if isinstance(left_op, Constant) and abs(left_op.value) < tolerance:
                    return SymbolicNode(node_content=Constant(0.0))
                # x * (-1) = -x, (-1) * x = -x
                if isinstance(right_op, Constant) and abs(right_op.value + 1.0) < tolerance:
                    neg_op = self._get_operator_by_name('neg')
                    if neg_op:
                        new_node = SymbolicNode(node_content=neg_op)
                        new_node.children = [left]
                        return new_node
                if isinstance(left_op, Constant) and abs(left_op.value + 1.0) < tolerance:
                    neg_op = self._get_operator_by_name('neg')
                    if neg_op:
                        new_node = SymbolicNode(node_content=neg_op)
                        new_node.children = [right]
                        return new_node
                # x * (y / x) = y
                if isinstance(right_op, Operator) and right_op.name in ('div', '/'):
                    if Expression._trees_are_equal(left, right.children[1]):
                        return right.children[0]
                # (y / x) * x = y
                if isinstance(left_op, Operator) and left_op.name in ('div', '/'):
                    if Expression._trees_are_equal(right, left.children[1]):
                        return left.children[0]

            # Division rules
            elif op_name == 'div' or op_name == '/':
                # x / 1 = x
                if isinstance(right_op, Constant) and abs(right_op.value - 1.0) < tolerance:
                    return left
                # 0 / x = 0 (x != 0)
                if isinstance(left_op, Constant) and abs(left_op.value) < tolerance:
                    return SymbolicNode(node_content=Constant(0.0))
                # x / 0 = 1 (protective handling)
                if isinstance(right_op, Constant) and abs(right_op.value) < tolerance:
                    return SymbolicNode(node_content=Constant(1.0))
                # x / x = 1
                if Expression._trees_are_equal(left, right):
                    return SymbolicNode(node_content=Constant(1.0))
                # (x * y) / x = y, (x * y) / y = x
                if isinstance(left_op, Operator) and left_op.name in ('mul', '*'):
                    if Expression._trees_are_equal(right, left.children[0]):
                        return left.children[1]
                    if Expression._trees_are_equal(right, left.children[1]):
                        return left.children[0]
                # x / (x * y) = 1 / y, x / (y * x) = 1 / y
                if isinstance(right_op, Operator) and right_op.name in ('mul', '*'):
                    one_node = SymbolicNode(node_content=Constant(1.0))
                    div_op = self._get_operator_by_name('div')
                    if div_op:
                        if Expression._trees_are_equal(left, right.children[0]):
                            new_node = SymbolicNode(node_content=div_op)
                            new_node.children = [one_node, right.children[1]]
                            return new_node
                        if Expression._trees_are_equal(left, right.children[1]):
                            new_node = SymbolicNode(node_content=div_op)
                            new_node.children = [one_node, right.children[0]]
                            return new_node

            # Algebraic addition simplification
            if op_name in ('add', '+'):
                # x + (y - x) = y
                if isinstance(right_op, Operator) and right_op.name in ('sub', '-'):
                    if Expression._trees_are_equal(left, right.children[1]):
                        return right.children[0]
                # (y - x) + x = y
                if isinstance(left_op, Operator) and left_op.name in ('sub', '-'):
                    if Expression._trees_are_equal(right, left.children[1]):
                        return left.children[0]

        # Unary operator simplification rules
        if node.degree == 1:
            child = children[0]
            child_op = child.node_content
            
            # -(-x) = x
            if op_name in ('neg', '-'):
                if isinstance(child_op, Operator) and child_op.name in ('neg', '-'):
                    if child_op.degree == 1:
                        return child.children[0]
                # -(c) = -c (constant folding)
                if isinstance(child_op, Constant):
                    return SymbolicNode(node_content=Constant(-child_op.value))

        return node

    def _get_operator_by_name(self, op_name: str) -> Optional[Operator]:
        """
        Retrieve an operator by its name.

        Parameters
        ----------
        op_name : str
            Name of the operator to find.

        Returns
        -------
        Operator or None
            The operator if found, None otherwise.
        """
        # Requires access to the operators list
        if not hasattr(self, 'operators'):
            return None
        
        for op in self.operators:
            if op.name == op_name:
                return op
        return None



class ExpressionSet(object):
    """
    A collection of Expression objects representing a multi-output symbolic model.

    ExpressionSet is used for multi-output tasks such as multi-class classification
    where each expression in the set corresponds to one output. The set evolves
    as a single individual in the genetic programming framework, meaning that
    crossover and mutation operations affect the entire set together.

    The set maintains a fixed capacity (length) but individual expressions can be
    None (placeholder) when the actual number of outputs is less than the capacity.

    Parameters
    ----------
    expressions : list of Expression or None
        List of Expression objects. The length determines the set's capacity.
        Individual entries can be None to represent unused output slots.
    out_func : Operator, optional
        Output transformation applied to the combined output of all expressions.
        For classification, this is typically softmax to produce probability estimates.
    metric : Fitness, optional
        The fitness function used to evaluate the entire expression set.

    Attributes
    ----------
    expressions : list
        The list of Expression objects (including None placeholders).
    out_func : Operator or None
        Output transformation function.
    metric : Fitness
        Fitness evaluation function.

    Properties
    ----------
    size : int
        Total number of nodes across all expressions in the set.
    complexity : float
        Sum of complexities of all non-None expressions.
    order : int
        Number of non-None expressions in the set.

    Examples
    --------
    >>> # Create a 3-class classifier expression set
    >>> expr1 = Expression(tree=tree1, ...)
    >>> expr2 = Expression(tree=tree2, ...)
    >>> expr3 = Expression(tree=tree3, ...)
    >>> expr_set = ExpressionSet([expr1, expr2, expr3], out_func=softmax)
    >>> # Evaluate all expressions on input data
    >>> scores = expr_set.execute(X)  # Returns (n_samples, 3) array
    >>> # Apply softmax to get probabilities
    >>> probabilities = softmax(scores)

    Notes
    -----
    ExpressionSet is particularly designed for:
    - Multi-class classification (each expression computes a discriminant function)
    - Multi-output regression
    - Any problem requiring multiple symbolic expressions that evolve together

    The genetic operators (mutation, crossover) defined in ExpressionSetGP operate
    on the set level, allowing expressions to be added, deleted, swapped, or
    mutated as a coordinated group.
    """

    def __init__(self, 
                 expressions: List[Optional['Expression']], 
                 out_func: Optional[Operator] = None, 
                 metric: Optional[Fitness] = None):
        self.metric = metric
        self.out_func = out_func
        
        if not all(isinstance(expr, (type(None), Expression)) for expr in expressions):
            raise ValueError("All items in expressions must be Expression objects.")
        
        self.expressions = expressions
        
        # 预分析常量信息
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
    def complexity(self) -> float:
        """
        Total complexity of the expression set.

        Returns the sum of complexities for all non-None expressions.
        """
        total = sum(expr.complexity for expr in self.expressions if expr is not None)
        return round(total)

    @property
    def order(self) -> int:
        """
        Number of non-None expressions in the set.
        """
        return sum(1 if expr is not None else 0 for expr in self.expressions)

    def is_valid(self, max_complexity: Optional[float] = None) -> bool:
        """
        Check whether the expression set is valid.

        Parameters
        ----------
        max_complexity : float, optional
            Maximum allowed complexity for individual expressions.

        Returns
        -------
        bool
            True if all expressions are valid.
        """
        
        # Check each expression
        for expr in self.expressions:
            if expr is not None:
                if not expr.is_valid(max_complexity):
                    return False
        
        return True

    def _check_constraints(self) -> bool:
        """
        Check whether all expressions in the set satisfy their constraints.

        Returns
        -------
        bool
            True if all constraints are satisfied by all expressions.
        """
        # Check operator argument complexity constraints
        if not all(
            expr._check_operator_constraints() for expr in self.expressions if expr is not None
        ):
            return False
        
        # Check nested operator depth constraints
        if not all(
            expr._check_nested_constraints() for expr in self.expressions if expr is not None
        ):
            return False
        
        return True

    def copy(self) -> 'ExpressionSet':
        return ExpressionSet(
            [expr.copy() if expr is not None else None for expr in self.expressions], 
            out_func=self.out_func, metric=self.metric
        )
    
    @property
    def constants(self) -> np.ndarray:
        const_info = self._build_constant_info()
        if const_info['total_constants'] == 0:
            return np.array([])
        current_constants = np.array([
            node.node_content.value 
            for expr_idx, node in const_info['flat_nodes']
        ])
        return current_constants

    def _execute_tree(self, X: np.ndarray) -> np.ndarray:
        """
        Fast NumPy execution path (without constant parameters).
        Optimized for memory efficiency and 3D input support.
        """
        # 1. Determine input shape and number of samples
        if X.ndim == 3:
            n_samples = X.shape[0] * X.shape[1]
            spatial_shape = (X.shape[0], X.shape[1])
        else:
            n_samples = X.shape[0]
            spatial_shape = None
        
        # 2. Identify valid expressions (filter out None)
        valid_indices = [i for i, expr in enumerate(self.expressions) if expr is not None]
        n_valid = len(valid_indices)
        
        # 3. Handle empty case
        if n_valid == 0:
            if spatial_shape:
                return np.empty((*spatial_shape, 0), dtype=float)
            return np.empty((n_samples, 0), dtype=float)
        
        # 4. Pre-allocate output buffer for efficiency
        # Using float dtype to accommodate potential continuous values
        output_buffer = np.empty((n_samples, n_valid), dtype=float)
        
        # 5. Execute each expression and fill the buffer column-wise
        for col_idx, expr_idx in enumerate(valid_indices):
            expr = self.expressions[expr_idx]
            # Execute and flatten result directly into the buffer column
            output_buffer[:, col_idx] = expr._execute_tree(X).ravel()
        
        # 6. Apply output transformation function if defined
        if self.out_func is not None:
            output_buffer = self.out_func(output_buffer)
        
        # 7. Restore spatial dimensions if input was 3D
        if spatial_shape:
            output_buffer = output_buffer.reshape(*spatial_shape, n_valid)
        
        return output_buffer

    def _execute_postorder(self, X: np.ndarray, valid_mask: Optional[np.ndarray] = None, 
                           constants: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute expression set using post-order traversal (supports constant arrays).
        Optimized for memory efficiency and 3D input support.
        
        Note: If valid_mask is provided, the output will be 2D (n_valid_samples, n_expressions),
        regardless of whether X is 3D.
        """
        # 1. Determine input shape and number of samples
        # Priority: valid_mask > X.ndim
        if valid_mask is not None:
            # If mask is provided, output is a list of valid points (2D)
            n_samples = int(np.sum(valid_mask))
            spatial_shape = None
        elif X.ndim == 3:
            # No mask, 3D input -> treat as spatial grid
            n_samples = X.shape[0] * X.shape[1]
            spatial_shape = (X.shape[0], X.shape[1])
        else:
            # No mask, 2D input
            n_samples = X.shape[0]
            spatial_shape = None
        
        # 2. Build constant information (cached internally if available)
        const_info = self._build_constant_info()
        
        # 3. Identify valid expressions (filter out None)
        valid_indices = [i for i, expr in enumerate(self.expressions) if expr is not None]
        n_valid = len(valid_indices)
        
        # 4. Handle empty case
        if n_valid == 0:
            if spatial_shape:
                return np.empty((*spatial_shape, 0), dtype=float)
            return np.empty((n_samples, 0), dtype=float)
        
        # 5. Pre-allocate output buffer for efficiency
        # Shape is always (n_samples, n_valid). 
        # If mask is used, n_samples is count of True.
        output_buffer = np.empty((n_samples, n_valid), dtype=float)
        
        # 6. Execute each expression and fill the buffer column-wise
        for col_idx, expr_idx in enumerate(valid_indices):
            expr = self.expressions[expr_idx]
            
            # Get constant range for this specific expression index
            const_range = const_info['ranges'][expr_idx]
            
            # Determine execution strategy
            if const_range is not None and constants is not None:
                expr_constants = constants[const_range[0]:const_range[1]]
                result = expr._execute_postorder(X, valid_mask, expr_constants)
            else:
                # Even if no constants, use postorder to support mask
                result = expr._execute_postorder(X, valid_mask, None)
            
            # Flatten and assign to buffer column
            # If masked, result is already 1D (n_valid_samples).
            # If not masked and 3D, result might be 2D/3D, ravel makes it 1D (n_total_samples).
            output_buffer[:, col_idx] = result.ravel()
        
        # 7. Apply output transformation function if defined
        if self.out_func is not None:
            output_buffer = self.out_func(output_buffer)
        
        # 8. Restore spatial dimensions ONLY if input was 3D AND no mask was used
        # If valid_mask was used, spatial structure is broken, output remains 2D.
        if spatial_shape:
            output_buffer = output_buffer.reshape(*spatial_shape, n_valid)
        
        return output_buffer

    def execute(self, X: np.ndarray, valid_mask: Optional[np.ndarray] = None, 
                constants: Optional[np.ndarray] = None) -> np.ndarray:
        """Execute the expression according to X."""
        if constants is None and valid_mask is None:
            result = self._execute_tree(X)
        else:
            result = self._execute_postorder(X, valid_mask, constants)
        
        return result

    def fitness(self, X: np.ndarray, y: np.ndarray,
                sample_weight: Optional[np.ndarray] = None, 
                constants: Optional[np.ndarray] = None) -> np.float32:
        """Evaluate the raw fitness of the expressionset according to X, y."""
        raw_fitness = self.metric(self, X, y, constants=constants,
                                  sample_weight=sample_weight)
        
        return np.float32(raw_fitness)

    def _build_constant_info(self):
        """
        Pre-compute and cache constant information across all expressions.

        Flattens constant nodes from all expressions into a single list with
        their ranges for efficient batch constant optimization.

        Returns
        -------
        dict
            Dictionary containing:
            - 'flat_nodes': List of (expr_idx, node) tuples for all constants
            - 'ranges': List of (start, end) tuples for each expression's constants
            - 'total_constants': Total number of constants across all expressions
        """
        if self._constant_info is not None:
            return self._constant_info
        
        flat_constant_nodes = []  # Store (expr_idx, node)
        expr_constant_ranges = []  # Store range for each expression
        
        global_const_idx = 0
        for expr_idx, expr in enumerate(self.expressions):
            if expr is None:
                expr_constant_ranges.append(None)
                continue
            
            # Collect all constant nodes from this expression
            expr_constants = []
            for node in PostOrderIter(expr.tree):
                if isinstance(node.node_content, Constant):
                    expr_constants.append(node)
            
            if expr_constants:
                start_idx = global_const_idx
                end_idx = global_const_idx + len(expr_constants)
                expr_constant_ranges.append((start_idx, end_idx))
                
                # Record each constant node
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
                new_expressions.append(expr)
        
        self.expressions = new_expressions
        return self

    def simplify(self, constants_tolerance: float = 1e-5) -> 'Expression':
        expressions = [None] * len(self.expressions)
        for i, expr in enumerate(self.expressions):
            if expr is not None:
                simplified_expr = expr.simplify(constants_tolerance)
                expressions[i] = simplified_expr
        
        new_expr_set = ExpressionSet(
            expressions, out_func=self.out_func, metric=self.metric
        )
        
        return new_expr_set


