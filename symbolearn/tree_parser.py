import re
import warnings
import pandas as pd
from typing import List, Optional, Callable, Union


from symbolearn.node import _operator_map, Variable, Constant, DynamicAggregation, Operator
from symbolearn.expression import Expression, ExpressionSet
from symbolearn.fitness import Fitness, _loss_function_map
from symbolearn.tree import SymbolicNode


class TreeParser:
    """Parse string expressions into SymbolicNode objects.

    This class provides functionality to parse mathematical expressions
    represented as strings into a tree structure (SymbolicNode) that
    can be executed within the genetic programming framework.

    Parameters
    ----------
    operators : List[str]
        List of available operators (e.g., '+', '*', 'tanh', 'softplus').
    n_variables : int
        Total number of features/variables in the dataset.
    spectral_stats : List[str]
        List of available dynamic aggregation functions (e.g., 'mean', 'max').
    maxsize : Optional[int], default=None
        Maximum number of nodes allowed in an expression. None means no limit.
    variable_names : Optional[List[str]]
        Custom names for variables. If None, defaults to 'x0', 'x1', ...
    metric : Optional[Callable | str | Operator]
        Fitness metric for expression evaluation.
    out_func : Optional[Callable | str | Operator]
        Output transformation function.
    """

    def __init__(self,
                 operators: List[str],
                 n_variables: int,
                 spectral_stats: List[str],
                 maxsize: Optional[int] = None,
                 variable_names: Optional[List[str]] = None,
                 metric: Optional[Callable | str | Operator] = None,
                 out_func: Optional[Callable | str | Operator] = None):
        """Initialize the TreeParser."""
        self.maxsize = maxsize
        self.operators = operators
        self.n_variables = n_variables
        self.spectral_stats = spectral_stats
        self.out_func = self._init_out_func(out_func)
        self.metric = self._init_metric(metric) if metric else None

        # Build operator lookup dictionary
        self._op_map = {op_name: _operator_map[op_name] for op_name in operators}

        # --- Build Variables ---
        self.variables = []
        if variable_names:
            if len(variable_names) != n_variables:
                raise ValueError(
                    f"Length of variable_names ({len(variable_names)}) must equal "
                    f"n_variables ({n_variables})"
                )
            self.variables = [Variable(i, name=name) for i, name in enumerate(variable_names)]
        else:
            self.variables = [Variable(i) for i in range(n_variables)]

        self._var_map = {var.name: var for var in self.variables}

        # --- Dynamically Build Aggregation Function Regex ---
        if not spectral_stats:
            self._agg_pattern = re.compile(r'^\b\B')
        else:
            agg_ops_regex = '|'.join(re.escape(op) for op in spectral_stats)
            self._agg_pattern = re.compile(rf'({agg_ops_regex})\(v(\d+)-(\d+)\)')

    def _init_metric(self, metric: Union[str, Fitness]):
        if isinstance(metric, Fitness):
            _metric = metric
        elif isinstance(metric, str):
            if metric not in _loss_function_map:
                raise ValueError('Unsupported metric: %s' % metric)
            loss_func, greater_is_better = _loss_function_map[metric]
            _metric = Fitness(loss_func, greater_is_better)
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

    def parse_expression_str(self, expr_str: str, part_of_set: bool = False) -> Expression:
        """Parse a single expression string into an Expression object.

        Parameters
        ----------
        expr_str : str
            Expression string, e.g., "(x1 + x2)" or "mul(add(x0, x1), x2)".

        Returns
        -------
        Expression
            The parsed Expression object.

        Raises
        ------
        ValueError
            If the expression cannot be parsed, or if the parsed tree exceeds maxsize.
        """
        expr_str = expr_str.strip()
        tree = self._parse_to_tree(expr_str)

        # --- Check maxsize ---
        # tree.size triggers the cached computation in SymbolicNode
        if self.maxsize is not None and tree.size > self.maxsize:
            raise ValueError(
                f"Expression '{expr_str[:100]}...' parsed to {tree.size} nodes, "
                f"which exceeds maxsize ({self.maxsize})"
            )
        if part_of_set:
            return Expression(tree)

        return Expression(tree, metric=self.metric, out_func=self.out_func)
    
    def parse_expression_set_str(self, set_str: str) -> ExpressionSet:
        """Parse an ExpressionSet string into multiple Expression objects.

        Splits the string by semicolons (respecting parentheses depth) and
        parses each component expression. Empty or 'None' strings become None.

        Parameters
        ----------
        set_str : str
            ExpressionSet string, e.g., "[x0 + x1; x2 * x3; None]".

        Returns
        -------
        ExpressionSet
            The parsed ExpressionSet object containing individual Expressions.
        """
        set_str = set_str.strip()

        if set_str.startswith('[') and set_str.endswith(']'):
            set_str = set_str[1:-1]

        expr_strs = self._split_expressions(set_str)

        expressions = []
        for expr_str in expr_strs:
            expr_str = expr_str.strip()
            if expr_str.lower() == 'none' or expr_str == '':
                expressions.append(None)
            else:
                expressions.append(self.parse_expression_str(expr_str, True))

        return ExpressionSet(expressions, metric=self.metric, out_func=self.out_func)
    
    def _split_expressions(self, set_str: str) -> List[str]:
        """Split an ExpressionSet string into individual expression strings.

        Splits by semicolons while respecting parentheses depth to avoid
        breaking expressions that contain function calls.

        Parameters
        ----------
        set_str : str
            The concatenated expression string.

        Returns
        -------
        List[str]
            List of individual expression strings.
        """
        expressions = []
        current = []
        depth = 0
        
        for char in set_str:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ';' and depth == 0:
                expressions.append(''.join(current))
                current = []
            else:
                current.append(char)
        
        if current:
            expressions.append(''.join(current))
        
        return expressions
    
    def _is_balanced_parentheses(self, s: str) -> bool:
        """Check if parentheses in a string are balanced and properly nested.

        Parameters
        ----------
        s : str
            Input string to check.

        Returns
        -------
        bool
            True if all parentheses are properly balanced, False otherwise.
        """
        depth = 0
        for char in s:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            if depth < 0:
                return False
        return depth == 0
    
    def _parse_to_tree(self, expr_str: str) -> SymbolicNode:
        """Recursively parse an expression string into a tree structure.

        Handles three cases:
        1. Leaf nodes: constants, variables, or aggregation operations
        2. Prefix (function) notation: op(arg1, arg2, ...)
        3. Infix (binary operator) notation: arg1 op arg2

        Parameters
        ----------
        expr_str : str
            The expression string to parse.

        Returns
        -------
        SymbolicNode
            The root node of the parsed expression tree.

        Raises
        ------
        ValueError
            If the expression cannot be parsed.
        """
        expr_str = expr_str.strip()

        open_count = expr_str.count('(')
        close_count = expr_str.count(')')
        if open_count == close_count + 1:
            expr_str += ')'
        elif close_count == open_count + 1:
            expr_str = '(' + expr_str

        while (expr_str.startswith('(') and
               expr_str.endswith(')') and
               self._is_balanced_parentheses(expr_str[1:-1])):
            expr_str = expr_str[1:-1].strip()

        # --- Step 1: Base Case (Leaf Nodes) ---

        # Try to parse as a numeric constant
        try:
            val = float(expr_str)
            return SymbolicNode(Constant(val))
        except ValueError:
            pass

        # Try to parse as a variable
        if expr_str in self._var_map:
            return SymbolicNode(self._var_map[expr_str])

        # Try to parse as a spectral aggregation (e.g., mean(v1-v90))
        agg_match = self._agg_pattern.match(expr_str)
        if agg_match and agg_match.group(0) == expr_str:
            op_name = agg_match.group(1)
            v_start = int(agg_match.group(2)) - 1
            v_end = int(agg_match.group(3)) - 1

            content = DynamicAggregation(v_start=v_start, v_end=v_end,
                                         stat_name_spectral=op_name,
                                         n_variables=self.n_variables)
            return SymbolicNode(content)

        # --- Step 2: Recursive Case (Internal Nodes) ---

        # 2a: Prefix function notation (e.g., sin(x), add(x, y))
        idx_paren = expr_str.find('(')
        if idx_paren != -1 and expr_str.endswith(')'):
            op_name = expr_str[:idx_paren]

            if op_name in self._op_map:
                op = self._op_map[op_name]

                args_str_outer = expr_str[idx_paren:]
                if (args_str_outer.startswith('(') and
                    args_str_outer.endswith(')') and
                    self._is_balanced_parentheses(args_str_outer[1:-1])):

                    args_str_inner = args_str_outer[1:-1]
                    arg_list = self._split_args(args_str_inner)

                    if len(arg_list) == op.degree:
                        children = [self._parse_to_tree(arg) for arg in arg_list]
                        return SymbolicNode(node_content=op, children=children)

        # 2b: Infix binary operator notation (e.g., x + y, x * y)
        inner_str = expr_str
        depth = 0
        
        for i in range(len(inner_str) - 1, -1, -1):
            char = inner_str[i]
            if char == ')':
                depth += 1
            elif char == '(':
                depth -= 1
            elif depth == 0:
                for op_name, op in self._op_map.items():
                    if op.degree == 2:
                        op_with_spaces = f" {op_name} "
                        op_len = len(op_with_spaces)
                        
                        if inner_str.startswith(op_with_spaces, i):
                            left_arg = inner_str[:i].strip()
                            right_arg = inner_str[i + op_len:].strip()
                            
                            left_child = self._parse_to_tree(left_arg)
                            right_child = self._parse_to_tree(right_arg)
                            
                            return SymbolicNode(node_content=op, children=[left_child, right_child])

        # --- Step 3: Failure ---
        raise ValueError(f"Cannot parse expression: '{expr_str[:200]}...'")


    def _split_args(self, args_str: str) -> List[str]:
        """Split a function argument string into individual arguments.

        Respects parentheses depth and comma separators at depth 0.

        Parameters
        ----------
        args_str : str
            The argument string to split.

        Returns
        -------
        List[str]
            List of individual argument strings.
        """
        args = []
        current = []
        depth = 0
        
        for char in args_str:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            args.append(''.join(current).strip())
        
        return args



def load_expressions_from_csv(
    csv_path: str,
    operators: List[str],
    n_variables: int,
    maxsize: Optional[int] = None,
    spectral_stats: List[str] = None,
    variable_names: Optional[List[str]] = None,
    metric: Optional[Callable | str | Operator] = None,
    out_func: Optional[Callable | str | Operator] = None,
    expression_col: str = 'expression'
) -> pd.DataFrame:
    """Load expressions from a CSV file and parse them into Expression/ExpressionSet objects.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing expression strings.
    operators : List[Operator]
        List of available operators (e.g., '+', '*', 'tanh', 'softplus').
    n_variables : int
        Total number of features/variables.
        Essential for creating DynamicAggregation nodes (e.g., min(v1-v90))
        and variable references (e.g., x99).
    variable_names : Optional[List[str]], default=None
        List of available variable names.
    maxsize : Optional[int], default=None
        Maximum number of nodes allowed per expression. None means unlimited.
    spectral_stats : List[str]
        List of available dynamic aggregation functions (e.g., 'mean', 'max').
    metric : Fitness, default=None
        Fitness metric for expression evaluation.
    out_func : Operator, default=None
        Output transformation function.
    expression_col : str
        Name of the column containing expression strings. Default is 'expression'.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame where the expression column contains Expression
        or ExpressionSet objects.

    Examples
    --------
    >>> operators = ['*', '+', '-', 'tanh', 'softplus']
    >>> df = load_expressions_from_csv('results.csv', operators, n_variables=10)
    >>> print(type(df.loc[0, 'expression']))
    <class 'ExpressionSet'>
    """
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Check if expression column exists
    if expression_col not in df.columns:
        raise ValueError(f"Column '{expression_col}' not found in CSV file")

    # Create parser
    parser = TreeParser(
        maxsize=maxsize,
        operators=operators,
        n_variables=n_variables,
        variable_names=variable_names,
        spectral_stats=spectral_stats,
        metric=metric, out_func=out_func
    )

    # Parse each row's expression string
    def parse_row(expr_str):
        if pd.isna(expr_str):
            return None

        expr_str = str(expr_str).strip()

        # Determine whether it's an ExpressionSet or single Expression
        if expr_str.startswith('[') and expr_str.endswith(']'):
            return parser.parse_expression_set_str(expr_str)
        else:
            return parser.parse_expression_str(expr_str)

    df[expression_col] = df[expression_col].apply(parse_row)

    return df






