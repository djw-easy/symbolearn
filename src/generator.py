import math
import bisect
import warnings
import numpy as np
from collections import defaultdict
from typing import Union, Optional, List, Tuple, Callable, Dict


from src.node import Operator, Constant, Variable, _operator_map, NodeContent, DynamicAggregation
from src.tree import count_trees, generate_random_tree, SymbolicNode, PreOrderIter
from src.expression import Expression, ExpressionSet
from src.fitness import Fitness, _loss_function_map
from src.utils import check_random_state




class ExprGenerator:
    """
    Constraint-aware random expression tree generator with controlled complexity.

    ExprGenerator creates random symbolic expression trees that conform to
    complex constraints while producing trees with a controlled distribution
    of complexities. This is critical for effective genetic programming
    evolution.

    Core Design Principles
    ----------------------
    1. Precomputation: On initialization, counts all valid trees at each
       complexity level using dynamic programming.

    2. Top-down generation: Builds trees from root to leaves, making
       locally optimal choices that lead to globally valid trees.

    3. Weighted random selection: Uses complexity distributions to bias
       toward simpler trees (parsimony pressure) while maintaining diversity.

    4. Semantic constraints: Prevents invalid trees such as x - x or x / x.

    Parameters
    ----------
    maxsize : int
        Maximum complexity of generated trees.
    operators : list of str or Operator
        Available mathematical operators (e.g., ['add', 'sub', 'mul', 'div']).
    input_shape : tuple
        Shape of input data (n_samples, n_features) or (H, W, n_features).
    use_constants : bool, default=True
        Whether to include random constants in expressions.
    use_variables : bool, default=True
        Whether to include input variables in expressions.
    variable_names : list of str, optional
        Names for each input variable (for display).
    spectral_stats : list of str, optional
        Spectral aggregation functions (e.g., ['mean', 'max', 'slope']).
    spatial_stats : list of str, optional
        Spatial aggregation functions for 3D input.
    valid_window_sizes : list of int, optional
        Valid window sizes for spatial aggregation.
    valid_spectral_length : int or tuple, optional
        Valid spectral range(s) for spectral aggregation.
    initial_constants : int or list, optional
        Number of constants or specific values to initialize.
    out_func : callable, str, or Operator, optional
        Output transformation function.
    complexity_of_operators : dict, optional
        Custom complexity weights for operators.
    complexity_of_constants : float, default=1.0
        Complexity cost for constant nodes.
    complexity_of_variables : float or list, default=1.0
        Complexity cost for variable nodes.
    complexity_of_aggregations : float, default=1.0
        Complexity cost for aggregation nodes.
    constraints : dict, optional
        Operator argument complexity limits.
        Format: {'op': max_complexity} or {'op': (max_left, max_right)}.
    nested_constraints : dict, optional
        Limits on operator nesting depth.
        Format: {'outer': {'inner': max_depth}}.
    parsimony_coefficient : float, default=0.9
        Controls complexity distribution. Higher values bias toward simpler trees.
    ndigits : int, optional
        Number of digits for constant formatting.
    random_state : int or RandomState, optional
        Random seed for reproducibility.

    Attributes
    ----------
    operators : list of Operator
        Resolved operator objects.
    variables : list of Variable
        Variable nodes available to the generator.
    constants : list of Constant
        Constant values available for insertion.
    complexity_probs : dict
        Probability distribution over valid complexity values.
    valid_complexities : list
        Sorted list of achievable complexity values.

    Methods
    -------
    generate_random_expr(complexity=None)
        Generate a random Expression with specified or sampled complexity.
    build_tree(complexity=None)
        Generate just the tree structure (no Expression wrapper).
    get_available_complexities()
        Return all achievable complexity values.

    Examples
    --------
    >>> from src.generator import ExprGenerator
    >>> gen = ExprGenerator(
    ...     maxsize=21,
    ...     operators=['add', 'sub', 'mul', 'div'],
    ...     input_shape=(100, 10),
    ...     parsimony_coefficient=0.5
    ... )
    >>> expr = gen.generate_random_expr()
    >>> print(expr)  # e.g., "(x0 + 2.5) * x1"
    >>> print(expr.complexity)  # e.g., 5.0

    Notes
    -----
    The parsimony_coefficient controls the probability distribution over
    tree complexities using: P(c) ∝ N(c) * exp(-α * c), where N(c) is
    the number of valid trees at complexity c and α is the coefficient.

    The generator maintains semantic validity by enforcing:
    - No division by zero (x/x is simplified to 1 at creation)
    - No subtraction of identical subtrees (x-x simplified to 0)
    - At least one non-constant child for binary operators

    See Also
    --------
    ExprSetGenerator : Generator for multi-output ExpressionSet individuals.
    Expression : The generated expression wrapper class.
    """
    
    def __init__(
        self,
        *,
        maxsize: int,
        operators: List[Union[str, Operator]],
        input_shape: tuple,
        use_constants: bool = True,
        use_variables: bool = True,
        metric: Optional[str | Fitness] = None,
        variable_names: Optional[List[str]] = None,
        spectral_stats: Optional[List[str]] = None,
        spatial_stats: Optional[List[str]] = None,
        valid_window_sizes: Optional[List[int]] = None,
        valid_spectral_length: Optional[int | Tuple[int, int]] = None,
        initial_constants: Optional[Union[int, List[float]]] = None,
        out_func: Optional[Callable | str | Operator] = None,
        complexity_of_operators: Optional[Dict[str, Union[int, float]]] = None,
        complexity_of_constants: Union[int, float] = 1.0,
        complexity_of_variables: Union[int, float, List[Union[int, float]]] = 1.0,
        complexity_of_aggregations: Union[int, float] = 1.0,
        constraints: Optional[Dict[str, Union[int, Tuple[int, ...]]]] = None,
        nested_constraints: Optional[Dict[str, Dict[str, int]]] = None,
        parsimony_coefficient: float = 0.9, ndigits: Optional[int] = 5,
        random_state: Union[int, np.random.RandomState] = None
    ):
        """Initialize the expression generator configuration.

        Parameters
        ----------
        maxsize : int
            Maximum target complexity permitted during generation.
        operators : list of str or Operator
            Operator pool used to build internal nodes.
        input_shape : tuple
            Shape of the input data used to infer the number of variables.
        use_constants : bool
            Whether constant terminals can be generated.
        use_variables : bool
            Whether variable terminals can be generated.
        metric : str or Fitness, optional
            Fitness object or metric name attached to generated expressions.
        variable_names : list of str, optional
            Display names for input variables.
        spectral_stats : list of str, optional
            Spectral aggregation operators available to aggregation nodes.
        spatial_stats : list of str, optional
            Spatial aggregation operators available to aggregation nodes.
        valid_window_sizes : list of int, optional
            Candidate spatial window sizes for spatial aggregation.
        valid_spectral_length : int or tuple of int, optional
            Allowed spectral aggregation lengths.
        initial_constants : int or list of float, optional
            Initial constant pool or the number of constants to sample.
        out_func : callable, str, or Operator, optional
            Output transformation applied to generated expressions.
        complexity_of_operators : dict, optional
            Custom complexity weights for operators.
        complexity_of_constants : float
            Complexity assigned to constant terminals.
        complexity_of_variables : float or list of float
            Complexity assigned to variable terminals.
        complexity_of_aggregations : float
            Complexity assigned to aggregation terminals.
        constraints : dict, optional
            Per-operator argument-complexity constraints.
        nested_constraints : dict, optional
            Direct parent-child nesting restrictions between operators.
        parsimony_coefficient : float
            Target ratio controlling the complexity sampling preference.
        ndigits : int, optional
            Number of digits used when formatting floating constants.
        random_state : int or RandomState, optional
            Random-number generator specification.
        """

        self.maxsize = maxsize
        self.ndigits = ndigits
        self.input_shape = input_shape
        self.n_variables = input_shape[-1]
        self.use_constants = use_constants
        self.use_variables = use_variables
        self.variable_names = variable_names
        self.spatial_stats = spatial_stats or []
        self.spectral_stats = spectral_stats or []
        self.valid_window_sizes = valid_window_sizes or []
        self.parsimony_coefficient = parsimony_coefficient
        self.valid_spectral_length = valid_spectral_length
        self.random_state = check_random_state(random_state)
        
        # Complexity configuration
        self.complexity_of_operators = complexity_of_operators or {}
        self.complexity_of_constants = float(complexity_of_constants)
        self.complexity_of_aggregations = float(complexity_of_aggregations)
        
        # Normalize per-variable complexity settings.
        if isinstance(complexity_of_variables, (list, tuple)):
            self.complexity_of_variables = [float(c) for c in complexity_of_variables]
        else:
            self.complexity_of_variables = [float(complexity_of_variables)] * self.n_variables
        
        # Constraint configuration
        self.constraints = constraints or {}
        self.nested_constraints = nested_constraints or {}
        
        # Decide automatically whether full complexity-aware generation is required.
        self.use_complexity_constraints = self._should_use_complexity_constraints(

            complexity_of_operators=self.complexity_of_operators,
            complexity_of_constants=self.complexity_of_constants,
            complexity_of_variables=self.complexity_of_variables,
            complexity_of_aggregations=self.complexity_of_aggregations,
            constraints=self.constraints,
            nested_constraints=self.nested_constraints
        )
        
        # Initialize operators, variables, and constants.
        self.out_func = self._init_out_func(out_func)
        self.operators = self._init_operators(operators)
        self.metric = self._init_metric(metric) if metric else None
        self.variables = self._init_variables(self.n_variables, variable_names)
        self.constants = self._init_constants(self.n_variables, initial_constants)

        
        self._use_spectral_aggregation = (
            isinstance(spectral_stats, (tuple, list)) and 
            len(spectral_stats) >= 1 and
            self.n_variables > 1
        )
        self._use_spatial_aggregation = (
            isinstance(spatial_stats, (tuple, list)) and 
            len(spatial_stats) >= 1 and
            len(self.input_shape) == 3 and
            isinstance(self.valid_window_sizes, (tuple, list)) and
            len(self.valid_window_sizes) >= 1
        )
        
        self.use_aggregation = self._use_spectral_aggregation or self._use_spatial_aggregation
        
        # Group operators by arity.
        self.degree_to_operators = defaultdict(list)
        self.degrees = set()
        for op in self.operators:
            self.degree_to_operators[op.degree].append(op)
            self.degrees.add(op.degree)
        
        # Add degree 0 to represent terminal nodes.
        self.degrees.add(0)
        self.degrees = sorted(self.degrees)
        
        # Validate the search space.
        if not self.variables and not self.constants and not self.use_aggregation:
            raise ValueError("Must have at least one type of terminal (variable, constant, or aggregation)")
        if not any(d > 0 for d in self.degrees):
            raise ValueError("Must have at least one operator with degree > 0")
        
        # Precompute the number of valid trees at each complexity value.
        self.complexity_counts = {}  # complexity -> count
        self._precompute_tree_counts()
        
        # Compute the effective parsimony coefficient.
        self.parsimony_coefficient = self._compute_coefficient_for_target(
            parsimony_coefficient
        )
        
        # Build the complexity sampling distribution.
        self._compute_complexity_distribution()

        self.global_count_memo = {}
    
    def _should_use_complexity_constraints(
        self,
        complexity_of_operators: Optional[Dict[str, float]],
        complexity_of_constants: float,
        complexity_of_variables: Union[float, List[float]],
        complexity_of_aggregations: float,
        constraints: Optional[Dict[str, Union[int, Tuple[int, int]]]],
        nested_constraints: Optional[Dict[str, Dict[str, int]]]
    ) -> bool:
        """Determine whether complexity-aware generation must be enabled.

        Complexity constraints are required when any part of the configuration
        deviates from the original size-only counting scheme, including nested
        operator restrictions, argument-complexity constraints, or non-unit
        complexity costs for operators, constants, variables, or aggregations.

        Returns
        -------
        bool
            ``True`` when complexity-aware counting and generation should be
            used, otherwise ``False``.
        """
        # Rule 1: explicit nested constraints require the complexity-aware path.
        if nested_constraints and len(nested_constraints) > 0:
            return True
        
        # Rule 2: explicit argument-complexity constraints also require it.
        if constraints and len(constraints) > 0:
            return True
        
        # Rule 3: any non-unit operator complexity activates the advanced path.
        if complexity_of_operators:
            for op_complexity in complexity_of_operators.values():
                if abs(op_complexity - 1.0) > 1e-6:  # Different from the unit baseline.
                    return True
        
        # Rule 4: constant complexity differs from the baseline.
        if abs(complexity_of_constants - 1.0) > 1e-6:
            return True
        
        # Rule 5: variable complexity differs from the baseline.
        if isinstance(complexity_of_variables, list):
            for var_complexity in complexity_of_variables:
                if abs(var_complexity - 1.0) > 1e-6:
                    return True
        else:
            if abs(complexity_of_variables - 1.0) > 1e-6:
                return True
        
        # Rule 6: aggregation complexity differs from the baseline.
        if abs(complexity_of_aggregations - 1.0) > 1e-6:
            return True
        
        # All checks passed, so the simpler size-based path is sufficient.
        return False

    
    def _init_metric(self, metric: Union[str, Fitness]):
        if isinstance(metric, Fitness):
            _metric = metric
        elif isinstance(metric, str):
            if metric not in _loss_function_map:
                raise ValueError('Unsupported metric: %s' % metric)
            if metric not in self.typical_metrics:
                warnings.warn(f"Fitness function '{metric}' is not a typical function. "
                              f"Please use {', '.join(self.typical_metrics[:-1])}, or {self.typical_metrics[-1]}.")
            loss_func, greater_is_better = _loss_function_map[metric]
            _metric = Fitness(loss_func, greater_is_better, 
                              penalty=self.penalty, C=self.C,
                              function_kwargs=self.metric_params)
        else:
            raise ValueError('Invalid type %s found in `metric`.' % type(metric))
        
        return _metric
    
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
    
    def _init_operators(self, operators: List[Union[str, Operator]]) -> List[Operator]:
        """Initialize the operator list."""
        _operators = []
        for op in operators:
            if isinstance(op, str):
                if op not in _operator_map:
                    raise ValueError(f'Invalid operator name: {op}')
                _operators.append(_operator_map[op])
            elif isinstance(op, Operator):
                _operators.append(op)
            else:
                raise ValueError(f'Invalid operator type: {type(op)}')
        
        if not _operators:
            raise ValueError('No valid operators provided')
        return _operators
    
    def _init_variables(self, n_variables: int, variable_names: Optional[List[str]] = None) -> List[Variable]:
        """Initialize the variable list."""
        if not self.use_variables:
            return []
        
        if variable_names is not None:
            if len(variable_names) != n_variables:
                raise ValueError(f'variable_names length mismatch: expected {n_variables}, got {len(variable_names)}')
            names = variable_names
        else:
            names = [f'x{i}' for i in range(n_variables)]
        
        return [Variable(i, name=name) for i, name in enumerate(names)]
    
    def _init_constants(self, n_variables: int, initial_constants: Optional[Union[int, List[float]]]) -> List[Constant]:
        """Initialize the constant list."""
        if not self.use_constants:
            return []
        
        if isinstance(initial_constants, (list, tuple)):
            return [Constant(c) for c in initial_constants]
        elif isinstance(initial_constants, int):
            return [Constant(self.random_state.normal(0, 3)) for _ in range(initial_constants)]
        else:
            # Default: roughly one third of the number of variables.
            n_constants = max(1, math.ceil(n_variables / 3))
            return [Constant(self.random_state.normal(0, 3)) for _ in range(n_constants)]
    
    def _update_constants(self, constants: List[Constant]) -> None:
        """Replace constants with the given ones."""
        self.constants = list(constants)
    
    def _get_operator_complexity(self, op: Operator) -> float:
        """Get the complexity assigned to an operator."""
        return float(self.complexity_of_operators.get(op.name, 1.0))
    
    def _get_variable_complexity(self, var_idx: int) -> float:
        """Get the complexity assigned to a variable."""
        if var_idx < len(self.complexity_of_variables):
            return self.complexity_of_variables[var_idx]
        return 1.0
    
    def _get_terminal_complexities(self) -> Dict[str, List[float]]:
        """
        Return complexity values for all terminal types.
        Returns: {'variable': [1.0, 1.0, 2.0], 'constant': [1.5], 'aggregation': [2.0]}
        """
        result = {}
        
        if self.use_variables and self.variables:
            result['variable'] = self.complexity_of_variables
        
        if self.use_constants and self.constants:
            result['constant'] = [self.complexity_of_constants]
        
        if self.use_aggregation:
            result['aggregation'] = [self.complexity_of_aggregations]
        
        return result
    
    def _calculate_complexity(self, node: SymbolicNode) -> float:
        """
        Compute node complexity recursively.
        """
        if node.degree == 0:
            # Leaf node
            if isinstance(node.node_content, Variable):
                return self._get_variable_complexity(node.node_content.variable)
            elif isinstance(node.node_content, Constant):
                return self.complexity_of_constants
            elif isinstance(node.node_content, DynamicAggregation):
                return self.complexity_of_aggregations
            else:
                return 1.0
        else:
            # Internal node: operator complexity plus child complexities.
            op_complexity = self._get_operator_complexity(node.node_content)
            children_complexity = sum(
                self._calculate_complexity(child) for child in node.children
            )
            return op_complexity + children_complexity
    
    def _precompute_tree_counts(self):
        """
        Precompute the number of valid trees at each complexity value.
        Use dynamic programming from low to high complexity.
        """
        # Determine the search range for complexity values.
        terminal_complexities = self._get_terminal_complexities()
        if not terminal_complexities:
            raise ValueError("No terminals available")
        
        # Minimum complexity: the smallest terminal complexity.
        all_terminal_complexities = []
        for complexities in terminal_complexities.values():
            all_terminal_complexities.extend(complexities)
        self.min_complexity = min(all_terminal_complexities)
        
        # Maximum complexity: maxsize.
        self.max_complexity = self.maxsize
        
        # Discretize using a step size of 0.1 (adjustable).
        self.complexity_step = 0.1
        complexity_values = []
        c = self.min_complexity
        while c <= self.max_complexity + self.complexity_step:
            complexity_values.append(round(c, 2))
            c += self.complexity_step
        
        # Initialization: terminal nodes.
        for c in complexity_values:
            self.complexity_counts[c] = self._count_terminals_at_complexity(c)
        
        # Dynamic programming: increase complexity step by step.
        for c in complexity_values:
            if c <= self.min_complexity + self.complexity_step:
                continue  # Already initialized
            
            # Try all possible operators.
            count = self._count_terminals_at_complexity(c)
            
            for degree in self.degrees:
                if degree == 0:
                    continue
                
                for op in self.degree_to_operators[degree]:
                    op_complexity = self._get_operator_complexity(op)
                    remaining = c - op_complexity
                    
                    if remaining < self.min_complexity - self.complexity_step:
                        continue
                    
                    # Count valid subtree allocations.
                    if degree == 1:
                        # Unary operator
                        child_count = self._get_count_at_complexity(remaining)
                        if child_count > 0:
                            # Check argument constraints
                            if self._check_operator_param_constraint(op.name, [remaining]):
                                count += child_count
                    
                    elif degree == 2:
                        # Binary operator: enumerate left/right subtree complexity allocations.
                        count += self._count_binary_partitions(op.name, remaining)
                    
                    else:
                        # Higher-arity operator: simplified handling.
                        # TODO: Can be optimized
                        avg_child = remaining / degree
                        child_count = self._get_count_at_complexity(avg_child)
                        if child_count > 0:
                            count += child_count ** degree
            
            self.complexity_counts[c] = count
        self.complexity_counts = {size: count for size, count in self.complexity_counts.items() if count>0}
    
    def _count_terminals_at_complexity(self, target_complexity: float, tolerance: float = 0.05) -> int:
        """Count terminal nodes at the specified complexity."""
        count = 0
        
        # Variables
        if self.use_variables:
            for var_c in self.complexity_of_variables:
                if abs(var_c - target_complexity) <= tolerance:
                    count += 1
        
        # Constants
        if self.use_constants:
            if abs(self.complexity_of_constants - target_complexity) <= tolerance:
                count += len(self.constants)
        
        # Aggregations
        if self.use_aggregation:
            if abs(self.complexity_of_aggregations - target_complexity) <= tolerance:
                count += 1  # Simplification: assume there is only one aggregation type.
        
        return count
    
    def _get_count_at_complexity(self, complexity: float) -> int:
        """Get the tree count at the specified complexity with interpolation."""
        # Round to the nearest step size.
        c = round(complexity / self.complexity_step) * self.complexity_step
        c = round(c, 2)
        return self.complexity_counts.get(c, 0)
    
    def _count_binary_partitions(self, op_name: str, remaining_complexity: float) -> int:
        """
        Count subtree allocations for a binary operator.
        Account for argument and semantic constraints.
        """
        total = 0
        
        # Get constraints
        constraint = self.constraints.get(op_name)
        max_left = remaining_complexity
        max_right = remaining_complexity
        
        if constraint is not None:
            if isinstance(constraint, tuple) and len(constraint) == 2:
                if constraint[0] >= 0:
                    max_left = min(max_left, constraint[0])
                if constraint[1] >= 0:
                    max_right = min(max_right, constraint[1])
            elif isinstance(constraint, int) and constraint >= 0:
                max_left = min(max_left, constraint)
                max_right = min(max_right, constraint)
        
        # Enumerate allocations
        left_c = self.min_complexity
        while left_c <= max_left and left_c <= remaining_complexity:
            right_c = remaining_complexity - left_c
            
            if right_c < self.min_complexity - self.complexity_step:
                left_c += self.complexity_step
                continue
            
            if right_c > max_right:
                left_c += self.complexity_step
                continue
            
            left_count = self._get_count_at_complexity(left_c)
            right_count = self._get_count_at_complexity(right_c)
            
            if left_count > 0 and right_count > 0:
                # Check the semantic constraint that both children cannot be constants.
                # If both subtree complexities match the constant complexity, exclude the case where both are constants.
                if self._both_could_be_constants(left_c, right_c):
                    # Subtract the both-constant case.
                    # Number of left-constant subtrees multiplied by number of right-constant subtrees.
                    left_const_count = self._count_constants_at_complexity(left_c)
                    right_const_count = self._count_constants_at_complexity(right_c)
                    invalid_count = left_const_count * right_const_count
                    total += left_count * right_count - invalid_count
                else:
                    total += left_count * right_count
            
            left_c += self.complexity_step
            left_c = round(left_c, 2)
        
        return total
    
    def _both_could_be_constants(self, left_c: float, right_c: float, tolerance: float = 0.05) -> bool:
        """Check whether both subtrees could be constants."""
        if not self.use_constants:
            return False
        left_match = abs(left_c - self.complexity_of_constants) <= tolerance
        right_match = abs(right_c - self.complexity_of_constants) <= tolerance
        return left_match and right_match
    
    def _count_constants_at_complexity(self, complexity: float, tolerance: float = 0.05) -> int:
        """Count constants at the specified complexity."""
        if not self.use_constants:
            return 0
        if abs(complexity - self.complexity_of_constants) <= tolerance:
            return len(self.constants)
        return 0
    
    def _check_operator_param_constraint(self, op_name: str, child_complexities: List[float]) -> bool:
        """Check operator argument-complexity constraints."""
        if op_name not in self.constraints:
            return True
        
        constraint = self.constraints[op_name]
        
        if isinstance(constraint, int):
            # Maximum complexity for all arguments.
            if constraint >= 0:
                return all(c <= constraint for c in child_complexities)
            return True
        
        elif isinstance(constraint, tuple):
            # Maximum complexity for each argument.
            for i, max_c in enumerate(constraint):
                if i >= len(child_complexities):
                    break
                if max_c >= 0 and child_complexities[i] > max_c:
                    return False
            return True
        
        return True
    
    def _compute_complexity_distribution(self):
        """
        Compute the complexity distribution used for random sampling.
        
        Use the formula: P(c) ∝ N(c) * exp(-α * c).
        Where:
        - N(c) is the number of valid trees at complexity c.
        - α is the parsimony coefficient (parsimony_coefficient).
        - exp(-α * c) is the penalty term; higher complexity yields lower probability.
        
        Effect of the parsimony coefficient:
        - α = 0: based purely on tree counts (high complexity dominates).
        - α = 0.001-0.01: mild preference for simpler trees (recommended range).
        - α = 0.1-1.0: strong preference for simpler trees.
        - α > 1.0: extremely strong preference for simpler trees.
        """
        if not self.complexity_counts:
            raise ValueError("No valid trees can be generated with the given constraints")
        
        # Compute weights with the parsimony penalty.
        weighted_counts = {}
        for c, count in self.complexity_counts.items():
            if count > 0:
                # Apply parsimony penalty: weight = count * exp(-α * complexity).
                penalty = np.exp(-self.parsimony_coefficient * c)
                weighted_counts[c] = count * penalty
        
        if not weighted_counts:
            raise ValueError("No valid trees can be generated with the given constraints")
        
        # Normalize to a probability distribution.
        total = sum(weighted_counts.values())
        self.complexity_probs = {
            c: weight / total
            for c, weight in weighted_counts.items()
        }
        
        self.valid_complexities = sorted(self.complexity_probs.keys())
    
    def _compute_coefficient_for_target(
        self,
        target_ratio: float,
        tolerance: float = 0.001
    ) -> float:
        """
        Compute the parsimony coefficient required to reach the target expected complexity.
        
        Use binary search to find an α such that:
        E[complexity] / maxsize ≈ target_ratio
        
        Parameters:
        ----
        target_ratio: Target ratio in the range [0, 1].
        tolerance: Convergence tolerance.
        
        Returns:
        ----
        alpha: Parsimony coefficient.
        
        Notes:
        ----
        Must be called after _precompute_tree_counts().
        The routine depends on the realized complexity_counts values.
        """
        if not self.complexity_counts:
            raise ValueError("complexity_counts is empty. Must call _precompute_tree_counts() first.")
        
        # Use the realized complexity values and tree counts.
        complexities = np.array(sorted(self.complexity_counts.keys()))
        tree_counts = np.array([self.complexity_counts[c] for c in complexities])
        
        target_mean = target_ratio * self.maxsize
        
        def compute_mean(alpha):
            """Compute the expected complexity for a given α."""
            weights = tree_counts * np.exp(-alpha * complexities)
            total_weight = np.sum(weights)
            if total_weight == 0:
                return complexities[0]  # Return the minimum complexity.
            probs = weights / total_weight
            return np.sum(complexities * probs)
        
        # Test the α = 0 case.
        mean_at_zero = compute_mean(0.0)
        
        # If the expectation at α = 0 is already below the target, the target cannot be reached.
        if mean_at_zero < target_mean:
            return 1.0
        
        # Binary search
        alpha_min, alpha_max = 0.0, 20.0
        max_iterations = 100
        
        for iteration in range(max_iterations):
            alpha_mid = (alpha_min + alpha_max) / 2
            current_mean = compute_mean(alpha_mid)
            
            if abs(current_mean - target_mean) < tolerance:
                return alpha_mid
            
            if current_mean > target_mean:
                # Expectation is too high, increase α for a stronger penalty.
                alpha_min = alpha_mid
            else:
                # Expectation is too low, decrease α.
                alpha_max = alpha_mid
        
        # If the search does not converge, return the closest value found.
        final_alpha = (alpha_min + alpha_max) / 2
        final_mean = compute_mean(final_alpha)
        return final_alpha
    
    def generate_tree_with_complexity(
        self,
        target_complexity: Optional[float] = None,
        tolerance: float = 0.05
    ) -> Optional[SymbolicNode]:
        """Generate an expression tree at the requested complexity.

        Parameters
        ----------
        target_complexity : float, optional
            Target complexity. If ``None``, a value is sampled from the
            learned complexity distribution.
        tolerance : float, default=0.05
            Allowed tolerance when matching discrete complexity buckets.

        Returns
        -------
        SymbolicNode or None
            Generated symbolic tree, or ``None`` if no valid tree can be
            produced at the requested complexity.
        """
        # If no complexity is provided, sample one from the distribution.
        if target_complexity is None:
            target_complexity = self._sample_complexity()
        
        # Round to the nearest discretization step.
        target_complexity = round(target_complexity / self.complexity_step) * self.complexity_step
        target_complexity = round(target_complexity, 2)
        
        # Abort early when the requested complexity is infeasible.
        if self._get_count_at_complexity(target_complexity) == 0:
            return None
        
        # Generate the tree in a top-down manner.
        tree = self._generate_tree_top_down(target_complexity, parent_op=None, tolerance=tolerance)
        
        return tree
    
    def _sample_complexity(self) -> float:
        """Sample a complexity value from the learned distribution."""

        complexities = list(self.complexity_probs.keys())
        probs = list(self.complexity_probs.values())
        return self.random_state.choice(complexities, p=probs)
    
    def _generate_tree_top_down(
        self,
        target_complexity: float,
        parent_op: Optional[str] = None,
        is_constant_sibling: bool = False,
        tolerance: float = 0.05
    ) -> SymbolicNode:
        """Generate a tree recursively in a top-down manner.

        Parameters
        ----------
        target_complexity : float
            Target complexity for the subtree being generated.
        parent_op : str, optional
            Name of the parent operator, used when checking nesting
            constraints.
        is_constant_sibling : bool, default=False
            Whether the sibling node is a constant, which affects semantic
            validity checks for binary operators.
        tolerance : float, default=0.05
            Tolerance used when matching discrete complexity buckets.

        Returns
        -------
        SymbolicNode
            Generated subtree root.
        """
        # Collect all feasible choices.
        choices = []
        
        # Option 1: terminal node.
        terminal_choices = self._get_terminal_choices(target_complexity, is_constant_sibling, tolerance)
        if terminal_choices:
            total_terminal_weight = sum(c['weight'] for c in terminal_choices)
            choices.append({
                'type': 'terminal_group',
                'weight': total_terminal_weight,
                'options': terminal_choices
            })
        
        # Option 2: operator node.
        for degree in self.degrees:
            if degree == 0:
                continue
            
            for op in self.degree_to_operators[degree]:
                # Check nesting constraints.
                if not self._check_nested_constraint(parent_op, op.name):
                    continue
                
                op_complexity = self._get_operator_complexity(op)
                remaining = target_complexity - op_complexity
                
                if remaining < self.min_complexity - self.complexity_step:
                    continue
                
                # Evaluate whether this operator is feasible.
                if degree == 1:
                    # Unary operator.
                    child_count = self._get_count_at_complexity(remaining)
                    if child_count > 0:
                        if self._check_operator_param_constraint(op.name, [remaining]):
                            choices.append({
                                'type': 'operator',
                                'op': op,
                                'degree': 1,
                                'remaining': remaining,
                                'weight': child_count
                            })
                elif degree == 2:
                    # Binary operator.
                    partition_count = self._count_binary_partitions(op.name, remaining)
                    if partition_count > 0:
                        choices.append({
                            'type': 'operator',
                            'op': op,
                            'degree': 2,
                            'remaining': remaining,
                            'weight': partition_count
                        })
                else:
                    # Higher-arity operator.
                    avg_child = remaining / degree
                    child_count = self._get_count_at_complexity(avg_child)
                    if child_count > 0:
                        choices.append({
                            'type': 'operator',
                            'op': op,
                            'degree': degree,
                            'remaining': remaining,
                            'weight': child_count ** degree
                        })
        
        if not choices:
            # Fallback: return a terminal node.
            return self._create_fallback_terminal()
        
        # Weighted random selection.
        weights = np.array([c['weight'] for c in choices], dtype=float)
        probs = weights / np.sum(weights)
        chosen_idx = self.random_state.choice(len(choices), p=probs)
        chosen = choices[chosen_idx]
        
        # Create the chosen node.
        if chosen['type'] == 'terminal_group':
            # Choose the concrete terminal type.
            terminal_options = chosen['options']
            term_weights = np.array([t['weight'] for t in terminal_options], dtype=float)
            term_probs = term_weights / np.sum(term_weights)
            term_idx = self.random_state.choice(len(terminal_options), p=term_probs)
            terminal_choice = terminal_options[term_idx]
            
            return self._create_terminal_node(terminal_choice)
        else:  # operator
            op = chosen['op']
            degree = chosen['degree']
            remaining = chosen['remaining']
            
            # Create the operator node.
            node = SymbolicNode(node_content=op)
            
            # Generate child nodes.
            if degree == 1:
                child = self._generate_tree_top_down(remaining, parent_op=op.name, tolerance=tolerance)
                node.children = [child]
            elif degree == 2:
                # Binary operator: choose a subtree complexity split.
                left_c, right_c = self._sample_binary_partition(op.name, remaining)
                
                # Generate the left and right subtrees.
                # Check whether the right subtree could be constant for the
                # left-subtree semantic constraint.
                right_could_be_const = self._complexity_matches_constant(right_c, tolerance)
                left_child = self._generate_tree_top_down(
                    left_c, parent_op=op.name, is_constant_sibling=right_could_be_const,
                    tolerance=tolerance
                )
                
                # Check whether the left subtree is constant for the
                # right-subtree semantic constraint.
                left_is_const = self._is_constant_node(left_child)
                right_child = self._generate_tree_top_down(
                    right_c, parent_op=op.name, is_constant_sibling=left_is_const, tolerance=tolerance
                )
                
                node.children = [left_child, right_child]
            else:
                # Higher-arity operator.
                avg_child = remaining / degree
                children = []
                for _ in range(degree):
                    child = self._generate_tree_top_down(avg_child, parent_op=op.name, tolerance=tolerance)
                    children.append(child)
                node.children = children
            
            return node
    
    def _get_terminal_choices(
        self,
        target_complexity: float,
        is_constant_sibling: bool = False,
        tolerance: float = 0.05
    ) -> List[Dict]:
        """Get candidate terminal choices for a target complexity.

        Parameters
        ----------
        target_complexity : float
            Target complexity for the terminal node.
        is_constant_sibling : bool, default=False
            Whether the sibling is constant; if so, the current node may not
            be sampled as a constant.
        tolerance : float, default=0.05
            Allowed complexity-matching tolerance.

        Returns
        -------
        list of dict
            Candidate terminal specifications such as
            ``{'type': 'variable', 'index': 0, 'weight': 1}``.
        """
        choices = []
        
        # Variables
        if self.use_variables:
            for i, var_c in enumerate(self.complexity_of_variables):
                if abs(var_c - target_complexity) <= tolerance:
                    choices.append({
                        'type': 'variable',
                        'index': i,
                        'weight': 1
                    })
        
        # Constants subject to the semantic sibling constraint.
        if self.use_constants and not is_constant_sibling:
            if abs(self.complexity_of_constants - target_complexity) <= tolerance:
                for i in range(len(self.constants)):
                    choices.append({
                        'type': 'constant',
                        'index': i,
                        'weight': 1
                    })
        
        # Aggregations
        if self.use_aggregation:
            if abs(self.complexity_of_aggregations - target_complexity) <= tolerance:
                choices.append({
                    'type': 'aggregation',
                    'weight': self.n_variables / (2 * self.complexity_of_aggregations)
                })
        
        return choices
    
    def _create_terminal_node(self, choice: Dict) -> SymbolicNode:
        """Create a terminal node from a sampled choice."""
        node = SymbolicNode(degree=0)

        
        if choice['type'] == 'variable':
            node.node_content = self.variables[choice['index']]
        elif choice['type'] == 'constant':
            node.node_content = self.constants[choice['index']]
        elif choice['type'] == 'aggregation':
            node.node_content = self._create_aggregation()
        
        return node
    
    def _create_fallback_terminal(self) -> SymbolicNode:
        """Create a default terminal node for fallback use."""
        node = SymbolicNode(degree=0)
        if self.variables:
            node.node_content = self.random_state.choice(self.variables)
        elif self.constants:
            node.node_content = self.random_state.choice(self.constants)
        else:
            node.node_content = self._create_aggregation()
        return node
    
    def _create_aggregation(self) -> DynamicAggregation:
        """Create an aggregation node."""

        if self._use_spectral_aggregation:
            v_start = self.random_state.randint(0, self.n_variables - 1)
            if isinstance(self.valid_spectral_length, int):
                length = self.random_state.randint(
                    1, min(self.valid_spectral_length, self.n_variables - v_start)
                )
            elif isinstance(self.valid_spectral_length, tuple):
                min_length, max_length = self.valid_spectral_length
                v_start = self.random_state.randint(0, self.n_variables - min_length - 1)
                length = self.random_state.randint(
                    min_length - 1, min(max_length, self.n_variables - v_start)
                )
            else:
                length = self.random_state.randint(1, self.n_variables - v_start)
            v_end = v_start + length
            stat_name_spectral = self.random_state.choice(self.spectral_stats)
        else:
            v_start, v_end, stat_name_spatial = None, None, None
        
        if self._use_spatial_aggregation:
            stat_name_spatial = self.random_state.choice(self.spatial_stats)
            window_size = self.random_state.choice(self.valid_window_sizes)
        else:
            window_size, stat_name_spatial = None, None
        
        return DynamicAggregation(
            v_start=v_start, v_end=v_end, 
            stat_name_spectral=stat_name_spectral, 
            n_variables=self.n_variables,
            window_size=window_size,
            stat_name_spatial=stat_name_spatial
        )
    
    def _check_nested_constraint(self, parent_op: Optional[str], child_op: str) -> bool:
        """Check whether ``child_op`` may appear directly under ``parent_op``."""
        if parent_op is None:
            return True
        
        if parent_op not in self.nested_constraints:
            return True
        
        child_constraints = self.nested_constraints[parent_op]
        
        if child_op not in child_constraints:
            return True
        
        k = child_constraints[child_op]
        
        # k=0 forbids the pattern, k=-1 allows it, and k>0 allows it with
        # a depth limit. Only direct parent-child admissibility is checked here.
        return k != 0

    
    def _sample_binary_partition(self, op_name: str, remaining_complexity: float) -> Tuple[float, float]:
        """Sample a subtree complexity split for a binary operator.

        Returns
        -------
        tuple of float
            A ``(left_complexity, right_complexity)`` pair.
        """
        # Get constraints
        constraint = self.constraints.get(op_name)
        max_left = remaining_complexity
        max_right = remaining_complexity
        
        if constraint is not None:
            if isinstance(constraint, tuple) and len(constraint) == 2:
                if constraint[0] >= 0:
                    max_left = min(max_left, constraint[0])
                if constraint[1] >= 0:
                    max_right = min(max_right, constraint[1])
            elif isinstance(constraint, int) and constraint >= 0:
                max_left = min(max_left, constraint)
                max_right = min(max_right, constraint)
        
        # Collect all valid splits.
        valid_partitions = []

        
        left_c = self.min_complexity
        while left_c <= max_left and left_c <= remaining_complexity:
            right_c = remaining_complexity - left_c
            right_c = round(right_c, 2)
            
            if right_c < self.min_complexity - self.complexity_step:
                left_c += self.complexity_step
                left_c = round(left_c, 2)
                continue
            
            if right_c > max_right:
                left_c += self.complexity_step
                left_c = round(left_c, 2)
                continue
            
            left_count = self._get_count_at_complexity(left_c)
            right_count = self._get_count_at_complexity(right_c)
            
            if left_count > 0 and right_count > 0:
                # Compute weights while accounting for semantic constraints.
                weight = left_count * right_count

                
                if self._both_could_be_constants(left_c, right_c):
                    left_const_count = self._count_constants_at_complexity(left_c)
                    right_const_count = self._count_constants_at_complexity(right_c)
                    weight -= left_const_count * right_const_count
                
                if weight > 0:
                    valid_partitions.append({
                        'left': left_c,
                        'right': right_c,
                        'weight': weight
                    })
            
            left_c += self.complexity_step
            left_c = round(left_c, 2)
        
        if not valid_partitions:
            # Fallback: split the remaining complexity evenly.
            half = remaining_complexity / 2
            return half, half
        
        # Weighted random selection.
        weights = np.array([p['weight'] for p in valid_partitions], dtype=float)

        probs = weights / np.sum(weights)
        idx = self.random_state.choice(len(valid_partitions), p=probs)
        chosen = valid_partitions[idx]
        
        return chosen['left'], chosen['right']
    
    def _complexity_matches_constant(self, complexity: float, tolerance: float = 0.05) -> bool:
        """Check whether a complexity matches the constant complexity."""
        if not self.use_constants:
            return False
        return abs(complexity - self.complexity_of_constants) <= tolerance
    
    def _is_constant_node(self, node: SymbolicNode) -> bool:
        """Check whether a node is a constant."""
        return isinstance(node.node_content, Constant)

    
    def generate_random_expr(self, complexity: Optional[int] = None):
        tree = self.build_tree(complexity)
        expression = Expression(
            tree=tree, metric=self.metric, 
            out_func=self.out_func, ndigits=self.ndigits,
            complexity_of_operators=self.complexity_of_operators,
            complexity_of_constants=self.complexity_of_constants,
            complexity_of_variables=self.complexity_of_variables,
            complexity_of_aggregations=self.complexity_of_aggregations,
            constraints=self.constraints, nested_constraints=self.nested_constraints
        )
        return expression
    
    def _simple_generate_tree(self, target_complexity: Optional[int] = None) -> SymbolicNode:
        """Generate a random symbolic tree under size and constraint limits."""  
        # If no complexity is specified, sample one randomly.
        if target_complexity is None:
            target_complexity = self._sample_complexity()
        
        # Round to the nearest discretization step.
        target_complexity = round(target_complexity / self.complexity_step) * self.complexity_step
        target_complexity = round(target_complexity, 2)
        
        # Check whether the requested complexity is feasible.
        if self._get_count_at_complexity(target_complexity) == 0:
            return None
        
        # Generate the tree structure under the combinatorial constraints.
        tree = generate_random_tree(
            int(target_complexity), self.degrees, 
            self.global_count_memo, self.random_state
        )
        
        if tree is None:
            raise ValueError(f"Failed to generate tree of size {target_complexity} with constraints")
        
        # Fill node contents recursively.
        for node in PreOrderIter(tree):
            if node.degree > 0:
                node.node_content = self._get_random_operator(node.degree)
            elif node.degree == 0:
                node.node_content = self._get_leaf_with_rules(node)
            else:
                raise ValueError("Invalid degree")
        
        return tree

    
    def build_tree(self, complexity: Optional[float] = None) -> SymbolicNode:
        """Generate a tree while keeping compatibility with the original API.

        Parameters
        ----------
        complexity : float, optional
            Target complexity. If ``None``, one is sampled randomly.

        Returns
        -------
        SymbolicNode
            Generated tree.
        """
        if self.use_complexity_constraints:
            return self.generate_tree_with_complexity(complexity)
        
        return self._simple_generate_tree(complexity)
    
    def get_available_complexities(self) -> List[float]:
        """Get all complexity values that can currently be generated."""
        return self.valid_complexities
    
    def get_complexity_range(self) -> Tuple[float, float]:
        """Get the minimum and maximum achievable complexity values."""
        return self.min_complexity, self.max_complexity
    
    def get_tree_count_at_complexity(self, complexity: float) -> int:
        """Get the number of trees available at a given complexity."""
        return self._get_count_at_complexity(complexity)


    def _get_leaf_with_rules(self, node: SymbolicNode) -> NodeContent:
        """Choose leaf content according to the semantic sampling rules.

        The current implementation enforces two rules:
        1. If the parent has degree 1, use a non-constant leaf.
        2. Leaf siblings under the same parent cannot all be constants.
        """
        parent = node.parent
        
        # Rule 1: if the parent has degree 1, use a non-constant leaf.
        if parent is not None and parent.degree == 1:
            return self._get_random_nonconstant_leaf()
        
        # Rule 2: leaf siblings under the same parent cannot all be constants.
        if parent is not None and len(parent.children) > 1:
            # Collect sibling leaves whose contents have already been assigned.
            sibling_leaves = []
            all_siblings_filled = True
            
            for child in parent.children:
                if child is node:
                    continue  # Skip the current node.
                if hasattr(child, 'node_content') and child.node_content is not None:
                    sibling_leaves.append(child)
                else:
                    # If any sibling has not been filled yet, Rule 2 cannot be applied.
                    all_siblings_filled = False
                    break
            
            # Check Rule 2 only after all sibling leaves have been assigned.
            if all_siblings_filled and sibling_leaves:
                # Check whether all sibling leaves are constants.
                all_siblings_are_constants = all(
                    isinstance(leaf.node_content, Constant) for leaf in sibling_leaves
                )
                
                # If all sibling leaves are constants, force a non-constant leaf.
                if all_siblings_are_constants:
                    return self._get_random_nonconstant_leaf()
        
        # Default case: choose a random leaf.
        return self._get_random_leaf()


    def _get_random_operator(self, degree: Optional[int] = None, exclude: Operator = None) -> Operator:
        """Helper to get a random operator (non-leaf)."""
        if degree is None:
            options = [operator for operator in self.operators if operator != exclude]
        elif isinstance(degree, int) and degree in self.degree_to_operators:
            options = [operator for operator in self.degree_to_operators[degree] if operator != exclude]
        else:
            raise ValueError("Invalid degree")
        if not options: return None
        return self.random_state.choice(options)

    def _get_random_leaf(self) -> NodeContent:
        """Gets a random leaf node (variable, constant or aggregation)."""
        probs = np.array([
            self.n_variables if self.use_variables else 0,
            len(self.constants) if self.use_constants else 0,
            self.n_variables / (2 * self.complexity_of_aggregations) if self.use_aggregation else 0
        ])
        leaf_type = self.random_state.choice([0, 1, 2], p=probs/sum(probs))
        if leaf_type == 0:
            return self.random_state.choice(self.variables)
        elif leaf_type == 1:
            return self.random_state.choice(self.constants)
        else:
            return self._create_aggregation()

    def _get_random_nonconstant_leaf(self) -> NodeContent:
        probs = np.array([
            self.n_variables if self.use_variables else 0,
            self.n_variables/(2 * self.complexity_of_aggregations) if self.use_aggregation else 0
        ])
        leaf_type = self.random_state.choice([0, 1], p=probs/sum(probs))
        if leaf_type == 0:
            return self.random_state.choice(self.variables)
        else:
            return self._create_aggregation()




def max_tree_set_structures(size, N, tree_count):
    """Compute the maximum legal structure count for a set of symbolic trees.

    Parameters
    ----------
    size : int
        Total size budget of the symbolic-tree set.
    N : int
        Number of trees in the set.
    tree_count : dict
        Mapping from a single-tree size to its number of structures,
        assumed to be monotonic in the available sizes.

    Returns
    -------
    int
        Maximum achievable structure count.
    """
    if N == 0:
        return 0 if size > 0 else 0
    
    # Get the sorted list of valid sizes.
    valid_sizes = sorted(tree_count.keys())
    if not valid_sizes:
        return 0
    
    min_size = valid_sizes[0]
    max_size = valid_sizes[-1]
    
    # Quickly reject infeasible budgets.
    if size > N * max_size or size < N * min_size:
        return 0
    
    # Use memoized search instead of a full DP table to reduce memory usage.
    memo = {}
    
    def dp(remaining_size, remaining_trees):
        """Return the best count for the remaining size and tree budget."""
        if remaining_trees == 0:
            return 0 if remaining_size == 0 else None
        
        if remaining_size < remaining_trees * min_size or remaining_size > remaining_trees * max_size:
            return None
        
        # Memoization
        state = (remaining_size, remaining_trees)
        if state in memo:
            return memo[state]
        
        max_count = None
        
        # Try assigning a size to the first tree, but only when the remainder
        # stays feasible for the remaining trees.
        for tree_size in valid_sizes:
            rest = remaining_size - tree_size
            rest_trees = remaining_trees - 1
            
            # Check whether the remaining budget is still feasible.
            if rest < rest_trees * min_size or rest > rest_trees * max_size:
                continue
            
            # Solve the remaining subproblem recursively.
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
    def __init__(
        self,
        *,
        maxsize: int,
        order: int | Tuple[int, int],
        operators: List[str | Operator],
        input_shape: tuple, 
        use_constants: bool = True,
        use_variables: bool = True,
        variable_names: List[str] | None = None,
        spectral_stats: List[str] | None = None,
        spatial_stats: Optional[List[str]] = None,
        valid_window_sizes: Optional[List[int]] = None,
        valid_spectral_length: Optional[int | Tuple[int, int]] = None,
        metric: Optional[Callable | str | Operator] = None,
        out_func: Optional[Callable | str | Operator] = None,
        initial_constants: Optional[int | List[float]] = None,
        complexity_of_operators: Optional[Dict[str, Union[int, float]]] = None,
        complexity_of_constants: Union[int, float] = 1.0,
        complexity_of_variables: Union[int, float, List[Union[int, float]]] = 1.0,
        complexity_of_aggregations: Union[int, float] = 1.0,
        constraints: Dict[str, int | Tuple[int, int]] | None = None,
        nested_constraints: Dict[str, Dict] | None = None,
        parsimony_coefficient: float = 2.0, ndigits: Optional[int] = 5,
        random_state: Union[int, np.random.RandomState] = None
    ):
        super().__init__(
            maxsize=maxsize,
            ndigits=ndigits,
            operators=operators,
            input_shape=input_shape,
            use_constants=use_constants,
            use_variables=use_variables,
            spatial_stats=spatial_stats,
            spectral_stats=spectral_stats,
            metric=metric, out_func=None,
            variable_names=variable_names,
            initial_constants=initial_constants,
            valid_window_sizes=valid_window_sizes,
            valid_spectral_length=valid_spectral_length,
            complexity_of_operators=complexity_of_operators,
            complexity_of_constants=complexity_of_constants,
            complexity_of_variables=complexity_of_variables,
            complexity_of_aggregations=complexity_of_aggregations,
            constraints=constraints, nested_constraints=nested_constraints,
            parsimony_coefficient=parsimony_coefficient, random_state=random_state
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
        
        self.out_func = self._init_out_func(out_func)
        size_tree_counts = {size: count for size, count in self.complexity_counts.items() if count>0}
        
        size_tree_counts[0] = 0
        self.size_maxcounts = {}
        for size in range(1, self.maxorder * maxsize + 1):
            maxcounts = max_tree_set_structures(size, self.maxorder, size_tree_counts)
            if maxcounts > 0:
                self.size_maxcounts[size] = maxcounts

    def generate_random_exprset(self, target_complexity: Optional[int] = None) -> Optional[ExpressionSet]:
        """Generate an expression set while respecting complexity constraints."""
        if target_complexity is None:
            target_complexity = self._sample_complexity()
        
        if self.fixed:
            expressions = [
                self.generate_random_expr(target_complexity) for _ in range(self.maxorder)
            ]
        else:
            n_expressions = self.random_state.randint(self.minorder, self.maxorder + 1)
            expressions = [None] * self.maxorder
            indices = self.random_state.permutation(self.maxorder)[:n_expressions]
            for i in indices:
                expressions[i] = self.generate_random_expr(target_complexity)
        
        # Build the expression set wrapper.
        expr_set = ExpressionSet(
            expressions=expressions, out_func=self.out_func, metric=self.metric
        )

        
        return expr_set



