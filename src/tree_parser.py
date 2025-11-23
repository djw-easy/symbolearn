import re
import pandas as pd
from typing import List, Optional, Callable, Union


from src.node import _operator_map, Variable, Constant, DynamicAggregation, Operator
from src.expression import Expression, ExpressionSet
from src.fitness import Fitness, _fitness_map
from src.tree import SymbolicNode


class TreeParser:
    """解析字符串表达式为SymbolicNode对象"""
    
    def __init__(self, 
                 operators: List[str], 
                 n_variables: int,
                 aggregation_operators: List[str],
                 maxsize: Optional[int] = None,
                 variable_names: Optional[List[str]] = None,
                 metric: Optional[Callable | str | Operator] = None,
                 out_func: Optional[Callable | str | Operator] = None):
        """
        初始化解析器
        
        Parameters
        ----------
        operators : List[str]
            可用的运算符列表 (例如 +, *, tanh, softplus)
        n_variables : int
            特征/变量的总数。
        aggregation_operators : List[str]
            可用的动态聚合函数列表 (e.g., 'mean', 'max')
        maxsize : Optional[int], default=None
            表达式允许的最大节点数。如果为 None，则不限制。
        variable_names : Optional[List[str]]
            变量的自定义名称列表。如果为 None，则默认为 'x0', 'x1', ...
        """
        self.operators = operators
        self.n_variables = n_variables
        self.aggregation_operators = aggregation_operators
        self.maxsize = maxsize
        self.metric = self._init_metric(metric)
        self.out_func = self._init_out_func(out_func)

        # 构建运算符查找字典
        self._op_map = {op_name: _operator_map[op_name] for op_name in operators}
        
        # --- 构建变量 ---
        self.variables = []
        if variable_names:
            if len(variable_names) != n_variables:
                raise ValueError(f"variable_names 的长度 ({len(variable_names)}) 必须等于 n_variables ({n_variables})")
            self.variables = [Variable(i, name=name) for i, name in enumerate(variable_names)]
        else:
            self.variables = [Variable(i) for i in range(n_variables)]
            
        self._var_map = {var.name: var for var in self.variables}
        
        # --- 动态构建聚合函数正则表达式 ---
        if not aggregation_operators:
            self._agg_pattern = re.compile(r'^\b\B') 
        else:
            agg_ops_regex = '|'.join(re.escape(op) for op in aggregation_operators)
            self._agg_pattern = re.compile(rf'({agg_ops_regex})\(v(\d+)-v(\d+)\)')

    def _init_metric(self, metric: Union[str, Fitness]):
        if isinstance(metric, Fitness):
            _metric = metric
        elif isinstance(metric, str):
            if metric not in _fitness_map:
                raise ValueError('Unsupported metric: %s' % metric)
            _metric = _fitness_map[metric]
        else:
            return None
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
            return None
        return _out_func

    def parse_expression_str(self, expr_str: str, part_of_set: bool = False) -> Expression:
        """
        解析单个表达式字符串为Expression对象
        
        Parameters
        ----------
        expr_str : str
            表达式字符串，例如 "(x1 + x2)"
            
        Returns
        -------
        Expression
        
        Raises
        ------
        ValueError
            如果表达式无法解析，或者解析后的节点数超过 self.maxsize
        """
        expr_str = expr_str.strip()
        tree = self._parse_to_tree(expr_str)
        
        # --- 检查 maxsize ---
        # tree.size 会触发 SymbolicNode 中缓存的计算
        if self.maxsize is not None and tree.size > self.maxsize:
            raise ValueError(
                f"表达式 '{expr_str[:100]}...' 解析后的节点数 ({tree.size}) "
                f"超过了 maxsize ({self.maxsize})"
            )
        if part_of_set:
            return Expression(tree)
        
        return Expression(tree, metric=self.metric, out_func=self.out_func)
    
    def parse_expression_set_str(self, set_str: str) -> ExpressionSet:
        """
        解析ExpressionSet字符串为多个Expression对象
        
        如果
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
        """分割ExpressionSet字符串中的各个表达式"""
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
        """检查字符串中的括号是否平衡且匹配"""
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
        """
        将表达式字符串递归解析为树结构
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

        # --- 步骤 1: 基本情况 (叶子节点) ---
        
        try:
            val = float(expr_str)
            return SymbolicNode(Constant(val))
        except ValueError:
            pass 

        if expr_str in self._var_map:
            return SymbolicNode(self._var_map[expr_str])

        agg_match = self._agg_pattern.match(expr_str)
        if agg_match and agg_match.group(0) == expr_str:
            op_name = agg_match.group(1)
            v_start = int(agg_match.group(2)) - 1
            v_end = int(agg_match.group(3)) - 1
            
            content = DynamicAggregation(v_start, v_end, 
                                         op_name, self.n_variables, 
                                         valid_op=self.aggregation_operators)
            return SymbolicNode(content)

        # --- 步骤 2: 递归情况 (内部节点) ---

        # 2a: 前缀函数
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

        # 2b: 中缀函数
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

        # --- 步骤 3: 失败 ---
        raise ValueError(f"无法解析表达式: '{expr_str[:200]}...'")


    def _split_args(self, args_str: str) -> List[str]:
        """分割函数参数"""
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
    aggregation_operators: List[str] = None,
    variable_names: Optional[List[str]] = None,
    metric: Optional[Callable | str | Operator] = None,
    out_func: Optional[Callable | str | Operator] = None,
    expression_col: str = 'expression'
) -> pd.DataFrame:
    """
    从CSV文件加载表达式，并将字符串转换为Expression或ExpressionSet对象
    
    Parameters
    ----------
    csv_path : str
        CSV文件路径
    operators : List[Operator]
        可用的运算符列表 (例如 +, *, tanh, softplus)
    n_variables : int
        特征/变量的总数。
        这对于创建 DynamicAggregation (例如 min(v1-v90)) 
        和变量 (例如 x99) 都至关重要。
    variable_names : Optional[List[str]], default=None
        可用的变量名称列表。
    maxsize : Optional[int], default=None
        表达式允许的最大节点数。如果为 None，则不限制。
    aggregation_operators : List[str]
        可用的动态聚合函数列表 (e.g., 'mean', 'max')
    metric : Optional[Callable | str | Operator], default=None
        可用的度量函数，可以是函数或字符串。
    out_func : Optional[Callable | str | Operator], default=None
        可用的输出函数，可以是函数或字符串。
    expression_col : str
        包含表达式字符串的列名，默认为'expression'
    
    Returns
    -------
    pd.DataFrame
        处理后的DataFrame，expression列包含Expression或ExpressionSet对象

    Examples
    --------
    >>> operators = ['*', '+', '-', 'tanh', 'softplus']
    >>> df = load_expressions_from_csv('results.csv', operators, n_variables=10)
    >>> print(type(df.loc[0, 'expression']))
    <class 'ExpressionSet'>
    """
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 检查expression列是否存在
    if expression_col not in df.columns:
        raise ValueError(f"列'{expression_col}'不存在于CSV文件中")
    
    # 创建解析器
    parser = TreeParser(
        maxsize=maxsize,
        operators=operators,
        n_variables=n_variables,
        variable_names=variable_names, 
        aggregation_operators=aggregation_operators,
        **({'metric': metric} if metric is not None else {}),
        **({'out_func': out_func} if out_func is not None else {})
    )
    
    # 解析每一行的表达式
    def parse_row(expr_str):
        if pd.isna(expr_str):
            return None
        
        expr_str = str(expr_str).strip()
        
        # 判断是ExpressionSet还是Expression
        if expr_str.startswith('[') and expr_str.endswith(']'):
            return parser.parse_expression_set_str(expr_str)
        else:
            return parser.parse_expression_str(expr_str)
    
    df[expression_col] = df[expression_col].apply(parse_row)
    
    return df






