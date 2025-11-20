import jax
import optax
import warnings
import numpy as np
import jax.numpy as jnp
from functools import lru_cache
from scipy.optimize import minimize
from typing import Union, Optional, List, Tuple, Iterator


from src.node import Operator, Constant, Variable, _operator_map, NodeContent, DynamicAggregation
from src.generator import ExprGenerator, ExprSetGenerator
from src.expression import Expression, ExpressionSet
from src.utils import check_random_state
from src.fitness import Fitness


METHODS_WITH_EPS = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'SLSQP']


def weighted_random_choice(weights_dict: dict, random_state: np.random.RandomState) -> str:
    """
    根据权重字典随机选择一个键。
    
    Args:
        weights_dict: 键值对字典，值为权重
        random_state: 随机状态
        
    Returns:
        str: 被选中的键
    """
    random_state = check_random_state(random_state)
    
    # 将字典转换为可哈希的元组用于缓存
    weights_tuple = tuple(sorted(weights_dict.items()))
    names, cumulative_probs = _get_cumulative_probs(weights_tuple)
    
    # 轮盘赌选择
    random_value = random_state.uniform()
    idx = jnp.searchsorted(cumulative_probs, random_value, side='right')
    return names[idx]


@lru_cache(maxsize=128)
def _get_cumulative_probs(weights_tuple):
    """
    计算累积概率（带缓存）。
    
    Args:
        weights_tuple: 排序后的 (name, weight) 元组
        
    Returns:
        tuple: (names列表, 累积概率数组)
    """
    names, weights = zip(*weights_tuple)
    weights_array = np.array(weights, dtype=np.float64)
    
    # 归一化并计算累积概率
    total_weight = weights_array.sum()
    if total_weight <= 0:
        raise ValueError("总权重必须为正数")
    
    probabilities = weights_array / total_weight
    cumulative_probs = np.cumsum(probabilities)
    cumulative_probs[-1] = 1.0  # 修正浮点误差
    
    return names, cumulative_probs



class ExpressionGP:
    def __init__(self,
                 generator: ExprGenerator,
                 mutation_weights: dict = None,
                 perturbation_factor: float = 0.129,
                 probability_negate_constant: float = 0.00743,
                 constants_tolerance: float = 1e-5,
                 random_state: Union[int, np.random.Generator] = None):
        self.generator = generator
        self.maxsize = generator.maxsize
        self.mutation_weights = mutation_weights
        self.perturbation_factor = perturbation_factor
        self.constants_tolerance = constants_tolerance
        self.random_state = check_random_state(random_state)
        self.probability_negate_constant = probability_negate_constant

    def _get_valid_cut_points(self, genes) -> List[int]:
        """
        获取所有合法切点
        
        合法切点定义：切割后左右两边都能形成完整表达式
        
        方法：从左到右模拟栈，记录栈深度为1的位置
        """
        valid_points = []
        stack_depth = 0
        
        for i, gene in enumerate(genes):
            if gene.degree == 0:
                stack_depth += 1
            else:
                stack_depth = stack_depth - gene.degree + 1
            
            # 栈深度为1表示一个完整子表达式结束
            if stack_depth == 1:
                valid_points.append(i + 1)
        
        return valid_points

    def _find_operands_start(self, genes, op_pos: int, degree: int) -> int:
        """
        找到操作符的第一个操作数的起始位置
        
        方法：从 op_pos 向前扫描，计算需要消耗的表达式数量
        """
        needed = degree
        pos = op_pos - 1
        
        while needed > 0 and pos >= 0:
            gene = genes[pos]
            if gene.degree == 0:
                needed -= 1
            else:
                needed = needed + gene.degree - 1
            pos -= 1
        
        return pos + 1
    
    def _find_first_subexpr_end(self, genes, start: int) -> int:
        """找到从 start 开始的第一个完整子表达式的结束位置"""
        stack_depth = 0
        for i in range(start, len(genes)):
            gene = genes[i]
            if gene.degree == 0:
                stack_depth += 1
            else:
                stack_depth = stack_depth - gene.degree + 1
            
            if stack_depth == 1:
                return i
        
        return len(genes) - 1

    def _find_subexpr_start(self, genes, end_idx: int) -> int:
        """
        找到在 end_idx 处结束的子表达式的起始索引。
        (这是 _find_operands_start 的 RPN 泛化)
        
        示例: [A, B, +]
        end_idx = 2 (at '+') -> returns 0
        
        示例: [A, B, C, +, *]
        end_idx = 3 (at '+') -> returns 1
        end_idx = 4 (at '*') -> returns 0
        """
        gene = genes[end_idx]
        if gene.degree == 0:
            # 叶子节点, 子表达式就是它自己
            return end_idx
        
        # 我们需要找到 'gene.degree' 个完整的表达式
        needed = gene.degree
        pos = end_idx - 1
        
        while needed > 0 and pos >= 0:
            g = genes[pos]
            # 每遇到一个节点，栈深度变化为 (g.degree - 1)
            # 我们从 'needed' 开始倒推
            needed = needed - 1 + g.degree
            pos -= 1
        
        # pos 是最后一个被消耗的基因的索引
        # 所以起始位置是 pos + 1
        return pos + 1

    def reproduce(self, genes) -> Expression:
        """创建副本（浅拷贝基因数组）"""
        new_expr = Expression(
            genes=genes, 
            metric=self.generator.metric, 
            out_func=self.generator.out_func
        )
        
        return new_expr

    def crossover(self, parent: Expression, donor: Expression):
        """
        单点交叉（保证合法性）
        
        策略：
        1. 在两个表达式中选择合法切点
        2. 交换后验证是否合法
        3. 检查大小限制
        
        时间复杂度：O(n) 其中 n 是表达式长度
        """
        if parent == donor:
            return None, None, False
        
        # 获取合法切点
        valid_cuts_self = self._get_valid_cut_points(parent.genes)
        valid_cuts_donor = self._get_valid_cut_points(donor.genes)
        
        if not valid_cuts_self or not valid_cuts_donor:
            return None, None, False
        
        # 随机选择切点
        cut1 = self.random_state.choice(valid_cuts_self)
        cut2 = self.random_state.choice(valid_cuts_donor)
        
        # 执行交叉
        genes1 = parent.genes[:cut1] + donor.genes[cut2:]
        genes2 = donor.genes[:cut2] + parent.genes[cut1:]
        
        # 检查大小
        if len(genes1) > self.maxsize or len(genes2) > self.maxsize:
            return None, None, False
        
        # 创建后代
        offspring1 = self.reproduce(genes=genes1)
        offspring2 = self.reproduce(genes=genes2)
        return offspring1, offspring2, True

    def _condition_mutation_weights(self, expr: Expression) -> dict:
        """
        Dynamically adjusts mutation weights based on the current state of the expression tree.
        """
        # Start with a copy of the default weights
        weights = self.mutation_weights.copy()

        # Handle single-node trees (tree.degree == 0)
        if expr.size == 0:
            weights['mutate_operator'] = 0.0
            weights['swap_operands'] = 0.0
            weights['delete_node'] = 0.0
            weights['simplify_tree'] = 0.0
            weights['hoist_tree'] = 0.0
            if not isinstance(expr[0], Constant):
                weights['mutate_constant'] = 0.0

        # Handle no binary operators
        if not expr._has_binary_operator():
            weights['swap_operands'] = 0.0

        # Adjust mutate_constant weight based on number of constants
        if self.generator.use_constants:
            n_constants = expr._count_scalar_constants()
            if n_constants == 0:
                weights['mutate_constant'] = 0.0
            else:
                weights['mutate_constant'] *= (min(8, n_constants) / 8.0)
        else:
            weights['mutate_constant'] = 0.0
        
        # Adjust mutate_variable weight based on number of variables
        if len(self.generator.variables) > 5:
            n_variables = expr._count_scalar_variables()
            if n_variables == 0:
                weights['mutate_variable'] = 0.0
            else:
                weights['mutate_variable'] *= min(8, n_variables) / 8.0
                weights['mutate_variable'] *= (np.log(n_variables + 1) + 1)
        else:
            weights['mutate_variable'] = 0.0
        
        # Adjust mutate_aggregation weight based on number of aggregations
        if self.generator.use_aggregations:
            n_aggregations = expr._count_scalar_aggregations()
            if n_aggregations == 0:
                weights['mutate_aggregation'] = 0.0
            else:
                weights['mutate_aggregation'] *= min(5, n_aggregations) / 8.0
                weights['mutate_aggregation'] *= (np.log(n_aggregations + 1) + 1)
        else:
            weights['mutate_aggregation'] = 0.0

        # Handle overly complex trees
        if expr.size >= self.maxsize:
            weights['add_node'] = 0.0
            weights['insert_node'] = 0.0

        return weights

    def mutation(self, parent: Expression):
        """
        Perform the mutation genetic operation on the symbolic tree.

        This method acts as a dispatcher, selecting one of several mutation
        strategies based on dynamically adjusted weights and executing it.
        Returns (Expression, bool) or (None, False)
        """
        # Get dynamically adjusted weights
        conditioned_weights = self._condition_mutation_weights(parent)
        # Select a mutation
        mutation_name = weighted_random_choice(conditioned_weights, self.random_state)
        if mutation_name == 'rotate_tree':
            new_expr, mutation_succeeded = self.rotate_tree(parent)
        elif mutation_name == 'add_node':
            new_expr, mutation_succeeded = self.add_node(parent)
        elif mutation_name == 'delete_node':
            new_expr, mutation_succeeded = self.delete_node(parent)
        elif mutation_name == 'mutate_aggregation':
            new_expr, mutation_succeeded = self.mutate_aggregation(parent)
        elif mutation_name == 'mutate_operator':
            new_expr, mutation_succeeded = self.mutate_operator(parent)
        elif mutation_name == 'do_nothing_tree':
            new_expr, mutation_succeeded = self.do_nothing_tree(parent)
        elif mutation_name == 'swap_operands':
            new_expr, mutation_succeeded = self.swap_operands(parent)
        elif mutation_name == 'mutate_constant':
            new_expr, mutation_succeeded = self.mutate_constant(parent)
        elif mutation_name == 'mutate_variable':
            new_expr, mutation_succeeded = self.mutate_variable(parent)
        elif mutation_name == 'insert_node':
            new_expr, mutation_succeeded = self.insert_node(parent)
        elif mutation_name == 'simplify_tree':
            new_expr, mutation_succeeded = self.simplify(parent)
        elif mutation_name == 'hoist_tree':
            new_expr, mutation_succeeded = self.hoist_tree(parent)
        elif mutation_name == 'randomize_tree':
            new_expr, mutation_succeeded = self.randomize_tree(parent)
        else:
            raise ValueError(f'Invalid mutation name: {mutation_name}')
        
        return new_expr, mutation_succeeded, mutation_name

    def insert_node(self, parent) -> Tuple[Optional['Expression'], bool]:
        """
        插入突变：在随机位置插入一个操作符和必要的操作数
        
        策略：
        1. 选择插入位置（合法切点）
        2. 插入一个操作符 + 补充操作数
        3. 验证合法性
        """
        if parent.size >= self.maxsize - 1:
            return None, False
        
        valid_points = self._get_valid_cut_points(parent.genes)
        if not valid_points:
            return None, False
        
        insert_pos = self.random_state.choice(valid_points)
        
        # 选择一个操作符
        new_op = self.generator._get_random_operator()
        
        # 计算需要补充的操作数（new_op.degree - 1）
        operands_needed = new_op.degree - 1
        
        if parent.size + 1 + operands_needed > self.maxsize:
            return None, False
        
        # 生成新操作数
        new_operands = [self.generator._get_random_leaf() for _ in range(operands_needed)]
        # 构建新基因序列
        new_genes = (parent.genes[:insert_pos] + new_operands + [new_op] + parent.genes[insert_pos:])
        
        new_expr = self.reproduce(new_genes)
        return new_expr, True

    def delete_node(self, parent, random_state = None) -> Tuple[Optional['Expression'], bool]:
        """
        删除突变：删除一个操作符及其子表达式
        
        策略：
        1. 选择一个非叶子基因
        2. 删除该基因及其操作数
        3. 保留其中一个操作数（提升）
        """
        if parent.size <= 1:
            return None, False
        random_state = check_random_state(random_state) if random_state is not None else self.random_state
        
        # 找到所有操作符位置
        op_positions = [i for i, gene in enumerate(parent.genes) if gene.degree > 0]
        if not op_positions:
            return None, False
        
        # 随机选择一个操作符
        op_pos = random_state.choice(op_positions)
        op = parent.genes[op_pos]
        
        # 找到该操作符的操作数范围
        # 在后缀表达式中，操作符前面的 op.degree 个完整子表达式是它的操作数
        operand_start = self._find_operands_start(parent.genes, op_pos, op.degree)
        
        # 随机保留一个操作数
        # 简化：保留第一个操作数
        preserved_end = self._find_first_subexpr_end(parent.genes, operand_start)
        
        # 构建新基因序列
        new_genes = (parent.genes[:operand_start] + 
                     parent.genes[operand_start:preserved_end + 1] + 
                     parent.genes[op_pos + 1:])
        
        new_expr = self.reproduce(new_genes)
        return new_expr, True

    def rotate_tree(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """
        增强的树旋转突变（完整版本）
        
        支持的旋转模式：
        
        1. **标准左旋** (Standard Left Rotation)
        树: A(B(...), C) -> B(..., A(..., C))
        效果: 父节点与左子节点交换位置
        
        2. **标准右旋** (Standard Right Rotation)  
        树: A(C, B(...)) -> B(A(C, ...), ...)
        效果: 父节点与右子节点交换位置
        
        3. **深度左旋** (Deep Left Rotation)
        树: A(B(D, E), C) -> B(D, A(E, C))
        效果: 提升左子树的左子节点
        
        4. **深度右旋** (Deep Right Rotation)
        树: A(C, B(D, E)) -> B(A(C, D), E)
        效果: 提升右子树的右子节点
        
        5. **交换子树** (Swap Subtrees)
        树: A(left_tree, right_tree) -> A(right_tree, left_tree)
        效果: 交换二元操作符的左右子树
        
        关键实现：
        - 通过栈模拟识别子表达式边界
        - 精确提取和重组子表达式
        - 保证旋转后的语义正确性
        """
        if parent.size < 3:  # 至少需要: leaf, leaf, op
            return None, False
        
        # 1. 识别所有可旋转的操作符位置
        rotation_candidates = self._find_rotation_candidates(parent.genes)
        
        if not rotation_candidates:
            return None, False
        
        # 2. 随机选择一个候选
        candidate = self.random_state.choice(rotation_candidates)
        rotation_type = candidate['type']
        op_pos = candidate['position']
        
        # 3. 根据旋转类型执行旋转
        if rotation_type == 'standard_left':
            new_genes, _ = self._standard_left_rotation(parent.genes, op_pos)
        elif rotation_type == 'standard_right':
            new_genes, _ = self._standard_right_rotation(parent.genes, op_pos)
        elif rotation_type == 'deep_left':
            new_genes, _ = self._deep_left_rotation(parent.genes, op_pos)
        elif rotation_type == 'deep_right':
            new_genes, _ = self._deep_right_rotation(parent.genes, op_pos)
        elif rotation_type == 'swap_subtrees':
            new_genes, _ = self._swap_subtrees(parent.genes, op_pos)
        else:
            return None, False
        
        # 4. 验证并创建新表达式
        if new_genes is None:
            return None, False
        
        new_expr = self.reproduce(genes=new_genes)
        return new_expr, True

    def _find_rotation_candidates(self, genes) -> List[dict]:
        """
        识别所有可旋转的操作符位置
        
        返回格式：
        [
            {'type': 'standard_left', 'position': idx, 'details': {...}},
            {'type': 'standard_right', 'position': idx, 'details': {...}},
            ...
        ]
        """
        candidates = []
        
        # 遍历所有操作符
        for i, gene in enumerate(genes):
            if gene.degree < 2:
                continue
            
            # 找到该操作符的所有操作数边界
            operands = self._find_operands_ranges(genes, i)
            
            if len(operands) < gene.degree:
                continue  # 数据不完整，跳过
            
            # 检查标准左旋：左操作数必须是一个子表达式（以操作符结尾）
            left_operand = operands[0]
            if genes[left_operand['end']].degree > 0:
                candidates.append({
                    'type': 'standard_left',
                    'position': i,
                    'details': {
                        'left_op_end': left_operand['end'],
                        'operands': operands
                    }
                })
            
            # 检查标准右旋：右操作数必须是一个子表达式（以操作符结尾）
            if gene.degree == 2:
                right_operand = operands[1]
                if genes[right_operand['end']].degree > 0:
                    candidates.append({
                        'type': 'standard_right',
                        'position': i,
                        'details': {
                            'right_op_end': right_operand['end'],
                            'operands': operands
                        }
                    })
            
            # 检查深度旋转：需要嵌套的操作符
            if genes[left_operand['end']].degree >= 2:
                candidates.append({
                    'type': 'deep_left',
                    'position': i,
                    'details': {
                        'left_op_end': left_operand['end'],
                        'operands': operands
                    }
                })
            
            if gene.degree == 2 and genes[operands[1]['end']].degree >= 2:
                candidates.append({
                    'type': 'deep_right',
                    'position': i,
                    'details': {
                        'right_op_end': operands[1]['end'],
                        'operands': operands
                    }
                })
            
            # 检查交换子树：二元操作符
            if gene.degree == 2:
                candidates.append({
                    'type': 'swap_subtrees',
                    'position': i,
                    'details': {'operands': operands}
                })
        
        return candidates

    def _find_operands_ranges(self, genes, op_pos: int) -> List[dict]:
        """
        找到操作符的所有操作数的范围 [start, end]
        
        算法：从操作符位置向前模拟栈，识别每个操作数的边界
        
        返回：
        [
            {'start': start1, 'end': end1},  # 第一个操作数
            {'start': start2, 'end': end2},  # 第二个操作数
            ...
        ]
        
        注意：返回顺序与计算顺序相同（左到右）
        """
        op = genes[op_pos]
        needed = op.degree
        
        if needed == 0:
            return []
        
        operands = []
        pos = op_pos - 1
        stack_depth = 0
        current_end = op_pos - 1
        
        # 向前扫描，识别操作数边界
        while pos >= 0 and len(operands) < needed:
            gene = genes[pos]
            
            if gene.degree == 0:
                stack_depth += 1
            else:
                stack_depth = stack_depth - gene.degree + 1
            
            # 当栈深度为 1 时，找到一个完整的操作数
            if stack_depth == 1:
                operands.append({
                    'start': pos,
                    'end': current_end
                })
                stack_depth = 0
                current_end = pos - 1
            
            pos -= 1
        
        # 反转以保持正确顺序（从左到右）
        return operands[::-1]

    def _standard_left_rotation(self, genes, op_pos: int) -> Tuple[Optional[List], str]:
        """
        标准左旋：A(B(...), C) -> B(..., A(..., C))
        
        RPN 示例：
        原: [D, E, +, C, *]  表示 mul(add(D,E), C)
        后: [D, E, C, *, +]  表示 add(D, mul(E,C))
        
        步骤：
        1. 提取左操作数 B(...)
        2. 提取 B 的子操作数和 A 的其他操作数
        3. 重组: [B的左子] + [A的其他操作数] + [A] + [B]
        """
        op_A = genes[op_pos]
        operands = self._find_operands_ranges(genes, op_pos)
        
        if not operands or genes[operands[0]['end']].degree == 0:
            return None, ''
        
        # 提取左操作数 B(...)
        left_operand_range = operands[0]
        op_B_pos = left_operand_range['end']
        op_B = genes[op_B_pos]
        
        # 找到 B 的操作数
        B_operands = self._find_operands_ranges(genes, op_B_pos)
        
        if not B_operands:
            return None, ''
        
        # 提取各部分
        # B 的左子树：从 B 的第一个操作数开始到倒数第二个操作数结束
        B_left_part_start = B_operands[0]['start']
        B_left_part_end = B_operands[-2]['end'] if len(B_operands) > 1 else B_operands[0]['start'] - 1
        
        # B 的最后一个操作数
        B_last_operand_start = B_operands[-1]['start']
        B_last_operand_end = B_operands[-1]['end']
        
        # A 的其他操作数（除了左操作数）
        A_other_operands_start = operands[1]['start'] if len(operands) > 1 else op_pos
        A_other_operands_end = operands[-1]['end'] if len(operands) > 1 else op_pos - 1
        
        # 重组基因序列
        new_genes = []
        
        # 1. 前缀部分（op_A 之前的所有基因）
        if B_left_part_start > 0:
            new_genes.extend(genes[:B_left_part_start])
        
        # 2. B 的左子树部分（除了最后一个操作数）
        if B_left_part_end >= B_left_part_start:
            new_genes.extend(genes[B_left_part_start:B_left_part_end + 1])
        
        # 3. B 的最后一个操作数
        new_genes.extend(genes[B_last_operand_start:B_last_operand_end + 1])
        
        # 4. A 的其他操作数
        if len(operands) > 1:
            new_genes.extend(genes[A_other_operands_start:A_other_operands_end + 1])
        
        # 5. 操作符 A
        new_genes.append(op_A)
        
        # 6. 操作符 B
        new_genes.append(op_B)
        
        # 7. 后缀部分（op_A 之后的所有基因）
        if op_pos + 1 < len(genes):
            new_genes.extend(genes[op_pos + 1:])
        
        return new_genes, 'rotate_tree_left'


    def _standard_right_rotation(self, genes, op_pos: int) -> Tuple[Optional[List], str]:
        """
        标准右旋：A(C, B(...)) -> B(A(C, ...), ...)
        
        RPN 示例：
        原: [C, D, E, +, *]  表示 mul(C, add(D,E))
        后: [C, D, *, E, +]  表示 add(mul(C,D), E)
        """
        op_A = genes[op_pos]
        
        if op_A.degree != 2:
            return None, ''
        
        operands = self._find_operands_ranges(genes, op_pos)
        
        if len(operands) < 2 or genes[operands[1]['end']].degree == 0:
            return None, ''
        
        # 提取右操作数 B(...)
        right_operand_range = operands[1]
        op_B_pos = right_operand_range['end']
        op_B = genes[op_B_pos]
        
        # 找到 B 的操作数
        B_operands = self._find_operands_ranges(genes, op_B_pos)
        
        if not B_operands:
            return None, ''
        
        # 提取各部分
        # A 的左操作数
        A_left_start = operands[0]['start']
        A_left_end = operands[0]['end']
        
        # B 的第一个操作数
        B_first_start = B_operands[0]['start']
        B_first_end = B_operands[0]['end']
        
        # B 的其他操作数
        B_other_start = B_operands[1]['start'] if len(B_operands) > 1 else op_B_pos
        B_other_end = B_operands[-1]['end'] if len(B_operands) > 1 else op_B_pos - 1
        
        # 重组基因序列
        new_genes = []
        
        # 1. 前缀部分
        if A_left_start > 0:
            new_genes.extend(genes[:A_left_start])
        
        # 2. A 的左操作数
        new_genes.extend(genes[A_left_start:A_left_end + 1])
        
        # 3. B 的第一个操作数
        new_genes.extend(genes[B_first_start:B_first_end + 1])
        
        # 4. 操作符 A
        new_genes.append(op_A)
        
        # 5. B 的其他操作数
        if len(B_operands) > 1:
            new_genes.extend(genes[B_other_start:B_other_end + 1])
        
        # 6. 操作符 B
        new_genes.append(op_B)
        
        # 7. 后缀部分
        if op_pos + 1 < len(genes):
            new_genes.extend(genes[op_pos + 1:])
        
        return new_genes, 'rotate_tree_right'


    def _deep_left_rotation(self, genes, op_pos: int) -> Tuple[Optional[List], str]:
        """
        深度左旋：A(B(D, E), C) -> B(D, A(E, C))
        
        RPN 示例：
        原: [D, E, +, C, *]  表示 mul(add(D,E), C)
        后: [D, E, C, *, +]  表示 add(D, mul(E,C))
        
        这是标准左旋的变体，同时提升 B 的左子节点
        """
        op_A = genes[op_pos]
        operands = self._find_operands_ranges(genes, op_pos)
        
        if not operands:
            return None, ''
        
        left_operand_range = operands[0]
        op_B_pos = left_operand_range['end']
        op_B = genes[op_B_pos]
        
        if op_B.degree < 2:
            return None, ''
        
        B_operands = self._find_operands_ranges(genes, op_B_pos)
        
        if len(B_operands) < 2:
            return None, ''
        
        # 提取各部分
        # B 的左子节点 D
        D_start = B_operands[0]['start']
        D_end = B_operands[0]['end']
        
        # B 的其他操作数 E
        E_start = B_operands[1]['start']
        E_end = B_operands[-1]['end']
        
        # A 的其他操作数 C
        C_start = operands[1]['start'] if len(operands) > 1 else op_pos
        C_end = operands[-1]['end'] if len(operands) > 1 else op_pos - 1
        
        # 重组: [prefix] + [D] + [E] + [C] + [A] + [B] + [suffix]
        new_genes = []
        
        if D_start > 0:
            new_genes.extend(genes[:D_start])
        
        new_genes.extend(genes[D_start:D_end + 1])
        new_genes.extend(genes[E_start:E_end + 1])
        
        if len(operands) > 1:
            new_genes.extend(genes[C_start:C_end + 1])
        
        new_genes.append(op_A)
        new_genes.append(op_B)
        
        if op_pos + 1 < len(genes):
            new_genes.extend(genes[op_pos + 1:])
        
        return new_genes, 'rotate_tree_deep_left'


    def _deep_right_rotation(self, genes, op_pos: int) -> Tuple[Optional[List], str]:
        """
        深度右旋：A(C, B(D, E)) -> B(A(C, D), E)
        
        RPN 示例：
        原: [C, D, E, +, *]  表示 mul(C, add(D,E))
        后: [C, D, *, E, +]  表示 add(mul(C,D), E)
        """
        op_A = genes[op_pos]
        
        if op_A.degree != 2:
            return None, ''
        
        operands = self._find_operands_ranges(genes, op_pos)
        
        if len(operands) < 2:
            return None, ''
        
        right_operand_range = operands[1]
        op_B_pos = right_operand_range['end']
        op_B = genes[op_B_pos]
        
        if op_B.degree < 2:
            return None, ''
        
        B_operands = self._find_operands_ranges(genes, op_B_pos)
        
        if len(B_operands) < 2:
            return None, ''
        
        # 提取各部分
        # A 的左操作数 C
        C_start = operands[0]['start']
        C_end = operands[0]['end']
        
        # B 的第一个操作数 D
        D_start = B_operands[0]['start']
        D_end = B_operands[0]['end']
        
        # B 的最后一个操作数 E
        E_start = B_operands[-1]['start']
        E_end = B_operands[-1]['end']
        
        # 重组: [prefix] + [C] + [D] + [A] + [E] + [B] + [suffix]
        new_genes = []
        
        if C_start > 0:
            new_genes.extend(genes[:C_start])
        
        new_genes.extend(genes[C_start:C_end + 1])
        new_genes.extend(genes[D_start:D_end + 1])
        new_genes.append(op_A)
        new_genes.extend(genes[E_start:E_end + 1])
        new_genes.append(op_B)
        
        if op_pos + 1 < len(genes):
            new_genes.extend(genes[op_pos + 1:])
        
        return new_genes, 'rotate_tree_deep_right'


    def _swap_subtrees(self, genes, op_pos: int) -> Tuple[Optional[List], str]:
        """
        交换子树：A(left, right) -> A(right, left)
        
        RPN 示例：
        原: [B, C, +, D, *]  表示 mul(add(B,C), D)
        后: [D, B, C, +, *]  表示 mul(D, add(B,C))
        
        这对于非交换操作符（如减法、除法）特别有用
        """
        op = genes[op_pos]
        
        if op.degree != 2:
            return None, ''
        
        operands = self._find_operands_ranges(genes, op_pos)
        
        if len(operands) < 2:
            return None, ''
        
        # 提取左右操作数
        left_start = operands[0]['start']
        left_end = operands[0]['end']
        
        right_start = operands[1]['start']
        right_end = operands[1]['end']
        
        # 重组: [prefix] + [right] + [left] + [op] + [suffix]
        new_genes = []
        
        if left_start > 0:
            new_genes.extend(genes[:left_start])
        
        new_genes.extend(genes[right_start:right_end + 1])
        new_genes.extend(genes[left_start:left_end + 1])
        new_genes.append(op)
        
        if op_pos + 1 < len(genes):
            new_genes.extend(genes[op_pos + 1:])
        
        return new_genes, 'swap_subtrees'

    def randomize_tree(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """随机替换子树 (RPN version)"""
        # 1. 随机选择一个 '节点' (即 genes 列表中的一个索引)
        target_idx = self.random_state.randint(parent.size)
        target_gene = parent.genes[target_idx]
        
        # 2. 找到这个节点所代表的整个子表达式的边界
        if target_gene.degree == 0:
            # 它是一个叶子节点
            start_idx = target_idx
            end_idx = target_idx
        else:
            # 它是一个操作符, 找到它所代表的子表达式的开头
            end_idx = target_idx
            start_idx = self._find_subexpr_start(parent.genes, end_idx)
        
        target_size = end_idx - start_idx + 1
        
        # 3. 计算替换后的最大尺寸 (复制自您的逻辑)
        size_of_rest = parent.size - target_size
        max_target_size = self.maxsize - size_of_rest
        
        valid_sizes = np.array(list(self.generator.size_prob.keys()))
        size_probs = np.array(list(self.generator.size_prob.values()))
        mask = valid_sizes <= max_target_size
        
        if not np.any(mask):
            return None, False
        
        new_size = self.random_state.choice(
            valid_sizes[mask], 
            p=size_probs[mask] / size_probs[mask].sum()
        )
        
        # 4. 生成新的子表达式 (基因列表)
        new_subtree_genes = self.generator.build_tree(size=new_size)
        
        # 5. 构建最终的基因列表
        new_genes = (
            parent.genes[:start_idx] + 
            new_subtree_genes + 
            parent.genes[end_idx + 1:]
        )
        
        # 6. 复制和验证
        new_expr = self.reproduce(genes=new_genes)
        return new_expr, True

    def hoist_tree(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """提升子树 (RPN version)"""
        # 1. 找到所有非叶子 '节点' (操作符)
        operator_indices = [i for i, gene in enumerate(parent.genes) if gene.degree > 0]
        if not operator_indices:
            return None, False
        
        # 2. 随机选择一个操作符 (作为 'subtree' 的根)
        op_idx = self.random_state.choice(operator_indices)
        op_gene = parent.genes[op_idx]
        
        # 3. 找到这个 'subtree' 的起始位置
        start_idx = self._find_subexpr_start(parent.genes, op_idx)
        
        # 4. 找到其所有 '子节点' (操作数) 的边界
        child_boundaries = []
        current_start = start_idx
        # 迭代直到我们到达操作符之前
        while current_start < op_idx:
            # 找到当前子表达式的结束位置
            current_end = self._find_first_subexpr_end(parent.genes, current_start)
            child_boundaries.append((current_start, current_end))
            current_start = current_end + 1
        
        if len(child_boundaries) != op_gene.degree:
            # RPN 表达式无效 (理论上不应发生)
            return None, False
        
        # 5. 随机选择一个 '子节点' (subsubtree) 来提升
        selected_idx = self.random_state.choice(len(child_boundaries))
        hoist_start, hoist_end = child_boundaries[selected_idx]
        hoisted_genes = parent.genes[hoist_start : hoist_end + 1]
        
        # 6. 构建新基因 (用 'subsubtree' 替换 'subtree')
        new_genes = (
            parent.genes[:start_idx] + 
            hoisted_genes + 
            parent.genes[op_idx + 1:]
        )
        
        # 7. 复制和验证
        new_expr = self.reproduce(genes=new_genes)
        return new_expr, True

    def do_nothing_tree(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """返回一个新的、相同的表达式，表示没有变化。"""
        new_expr = self.reproduce(genes=parent.genes)
        return new_expr, True

    def mutate_constant(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """突变常量 (RPN version)"""
        # 1. 收集候选索引
        constant_indices = [i for i, gene in enumerate(parent.genes) 
                            if isinstance(gene, Constant)]
        if not constant_indices:
            return None, False
        
        # 2. 随机选择一个
        target_idx = self.random_state.choice(constant_indices)
        target_gene = parent.genes[target_idx]
        
        # 3. 计算新值
        perturbation = 1 + self.perturbation_factor * self.random_state.random() + 0.1
        perturbation = perturbation if self.random_state.uniform() > 0.5 else 1/perturbation
        if self.random_state.uniform() < self.probability_negate_constant:
            perturbation = -perturbation
        new_value = target_gene.value * perturbation
        new_constant = Constant(value=new_value)
        
        # 4. 创建新基因列表
        new_genes = parent.genes[:target_idx] + [new_constant] + parent.genes[target_idx + 1:]
        
        # 5. 复制和验证
        new_expr = self.reproduce(genes=new_genes)
        return new_expr, True

    def mutate_variable(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """突变变量 (RPN version)"""
        # 1. 收集候选索引
        variable_indices = [i for i, gene in enumerate(parent.genes) 
                            if isinstance(gene, Variable)]
        if not variable_indices:
            return None, False

        # 2. 随机选择一个
        target_idx = self.random_state.choice(variable_indices)
        target_gene = parent.genes[target_idx]
        
        # 3. 选择新变量
        new_variable = self.random_state.choice(self.generator.variables)
        if new_variable == target_gene:
            return None, False
        
        # 4. 创建新基因列表
        new_genes = parent.genes[:target_idx] + [new_variable] + parent.genes[target_idx + 1:]
        
        # 5. 复制和验证
        new_expr = self.reproduce(genes=new_genes)
        return new_expr, True

    def mutate_operator(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """突变操作符 (RPN version)"""
        # 1. 收集候选索引 (非叶子)
        operator_indices = [i for i, gene in enumerate(parent.genes) 
                            if gene.degree > 0]
        if not operator_indices:
            return None, False
        
        # 2. 随机选择一个
        target_idx = self.random_state.choice(operator_indices)
        target_gene = parent.genes[target_idx]
        degree = target_gene.degree
        
        # 3. 寻找相同度数的替代品
        alternatives = [op for op in self.generator._degree_operators.get(degree, []) 
                        if op != target_gene]
        
        if not alternatives:
            return None, False
        
        new_operator = self.random_state.choice(alternatives)
        
        # 4. 创建新基因列表
        new_genes = parent.genes[:target_idx] + [new_operator] + parent.genes[target_idx + 1:]
        
        # 5. 复制和验证
        new_expr = self.reproduce(genes=new_genes)
        return new_expr, True

    def mutate_aggregation(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """突变聚合节点 (RPN version)"""
        # 1. 收集候选索引
        agg_indices = [i for i, gene in enumerate(parent.genes) 
                       if isinstance(gene, DynamicAggregation)]
        if not agg_indices:
            return None, False
        
        # 2. 随机选择一个
        target_idx = self.random_state.choice(agg_indices)
        aggregation = parent.genes[target_idx]
        
        # 3. 计算新的聚合参数 (复制您的逻辑)
        valid_op_num = len(aggregation.valid_op)
        prob_mutate_operator = 0.0001*valid_op_num if valid_op_num>1 else 0.0
        if self.random_state.random() < prob_mutate_operator:
            new_op_name = self.random_state.choice(aggregation.valid_op)
            new_aggregation = DynamicAggregation(
                v_start=aggregation.v_start,
                v_end=aggregation.v_end,
                op_name=new_op_name,
                n_variables=aggregation.n_variables,
                valid_op=aggregation.valid_op
            )
        else:
            v_start, v_end = aggregation.v_start, aggregation.v_end
            n_variables = aggregation.n_variables
            current_window_size = v_end - v_start + 1
            
            max_change_ratio = 0.5
            max_shift = max(1, int(current_window_size * max_change_ratio))
            mutation_type = self.random_state.choice(
                ['shift_both', 'shift_start', 'shift_end', 'expand', 'shrink']
            )
            
            # 执行突变逻辑
            if mutation_type == 'shift_both':
                shift = self.random_state.randint(-max_shift, max_shift + 1)
                new_start = v_start + shift
                new_end = v_end + shift
                if new_start < 0:
                    offset = -new_start
                    new_start = 0
                    new_end = min(new_end + offset, n_variables - 1)
                elif new_end >= n_variables:
                    offset = new_end - (n_variables - 1)
                    new_end = n_variables - 1
                    new_start = max(new_start - offset, 0)
                if new_end <= new_start:
                    new_start = max(0, new_end - 1)
            elif mutation_type == 'shift_start':
                shift = self.random_state.randint(-max_shift, max_shift + 1)
                new_start = max(0, min(v_start + shift, v_end - 1))
                new_end = v_end
            elif mutation_type == 'shift_end':
                shift = self.random_state.randint(-max_shift, max_shift + 1)
                new_end = max(v_start + 1, min(v_end + shift, n_variables - 1))
                new_start = v_start
            elif mutation_type == 'expand':
                expand_amount = self.random_state.randint(1, max_shift + 1)
                expand_direction = self.random_state.choice(['left', 'right', 'both'])
                if expand_direction == 'left' and v_start > 0:
                    new_start = max(0, v_start - expand_amount)
                    new_end = v_end
                elif expand_direction == 'right' and v_end < n_variables - 1:
                    new_start = v_start
                    new_end = min(n_variables - 1, v_end + expand_amount)
                else:
                    left_expand = expand_amount // 2
                    right_expand = expand_amount - left_expand
                    new_start = max(0, v_start - left_expand)
                    new_end = min(n_variables - 1, v_end + right_expand)
            else:  # shrink
                max_shrink = min(max_shift, current_window_size - 2)
                if max_shrink < 1:
                    shift = self.random_state.randint(-max_shift, max_shift + 1)
                    new_start = max(0, min(v_start + shift, n_variables - 2))
                    new_end = min(new_start + 1, n_variables - 1)
                else:
                    shrink_amount = self.random_state.randint(1, max_shrink + 1)
                    shrink_direction = self.random_state.choice(['left', 'right', 'both'])
                    if shrink_direction == 'left':
                        new_start = min(v_start + shrink_amount, v_end - 1)
                        new_end = v_end
                    elif shrink_direction == 'right':
                        new_start = v_start
                        new_end = max(v_start + 1, v_end - shrink_amount)
                    else:
                        left_shrink = shrink_amount // 2
                        right_shrink = shrink_amount - left_shrink
                        new_start = min(v_start + left_shrink, v_end - 1)
                        new_end = max(new_start + 1, v_end - right_shrink)
            
            new_start = max(0, min(new_start, n_variables - 2))
            new_end = max(new_start + 1, min(new_end, n_variables - 1))
            new_aggregation = DynamicAggregation(
                v_start=new_start, v_end=new_end,
                op_name=aggregation.op_name,
                n_variables=aggregation.n_variables,
                valid_op=aggregation.valid_op
            )
        
        # 4. 创建新基因列表
        new_genes = parent.genes[:target_idx] + [new_aggregation] + parent.genes[target_idx + 1:]
        
        # 5. 复制和验证
        new_expr = self.reproduce(genes=new_genes)
        return new_expr, True

    def swap_operands(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """交换二元操作符的两个操作数 (RPN version)"""
        # 1. 找到所有二元操作符的索引
        binary_op_indices = [i for i, gene in enumerate(parent.genes) 
                             if gene.degree == 2]
        if not binary_op_indices:
            return None, False
        
        # 2. 随机选择一个二元操作符的位置
        op_pos = self.random_state.choice(binary_op_indices)
        
        # 3. 确定两个操作数子表达式的边界
        # RPN 结构: [... (左操作数子树 L_SUBTREE) ... (右操作数子树 R_SUBTREE) ... BINARY_OP]
        # BINARY_OP 在 op_pos
        
        # 首先找到右操作数子树 (R_SUBTREE)
        # R_SUBTREE 结束于 op_pos - 1
        end_R_subtree = op_pos - 1
        # 找到 R_SUBTREE 的起始位置
        start_R_subtree = self._find_subexpr_start(parent.genes, end_R_subtree)
        
        # 然后找到左操作数子树 (L_SUBTREE)
        # L_SUBTREE 结束于 start_R_subtree - 1
        end_L_subtree = start_R_subtree - 1
        # 找到 L_SUBTREE 的起始位置
        start_L_subtree = self._find_subexpr_start(parent.genes, end_L_subtree)
        
        # 边界有效性检查
        if start_L_subtree < 0 or start_R_subtree <= end_L_subtree or op_pos <= end_R_subtree:
            # 这表示 RPN 结构不符合预期，或者操作数不存在
            return None, False
            
        # 4. 提取两个子表达式 (基因列表)
        L_SUBTREE_genes = parent.genes[start_L_subtree : end_L_subtree + 1]
        R_SUBTREE_genes = parent.genes[start_R_subtree : end_R_subtree + 1]
        
        # 5. 构建交换后的新基因列表
        # 结构变为: [... (右操作数子树 R_SUBTREE) ... (左操作数子树 L_SUBTREE) ... BINARY_OP]
        new_genes_list = (
            parent.genes[:start_L_subtree] +  # 前缀部分
            R_SUBTREE_genes +                 # 交换后的右操作数子树
            L_SUBTREE_genes +                 # 交换后的左操作数子树
            parent.genes[op_pos:]             # 操作符及之后的部分
        )
        
        # 6. 复制和验证
        # 假设 Expression 构造函数会调用 _is_valid 进行验证
        new_expr = Expression(genes=new_genes_list, metric=parent.metric) 
        
        return new_expr, True

    def optimize_constants(
        self,
        parent: Expression, 
        X: np.ndarray, 
        y: np.ndarray,
        optimizer_algorithm='L-BFGS-B',
        optimizer_nrestarts=3,
        optimizer_iterations=10
    ):
        """
        混合策略的常量优化
        - 梯度计算：JAX（快速自动微分）
        - 适应度评估：NumPy（避免重复编译）
        """
        # 检查是否有常量
        if not (len(parent.constant_indices) > 0):
            return parent, False, np.nan
        
        # 获取初始常量
        initial_constants = np.array([
            parent.genes[idx].value for idx in parent._constant_indices
        ])
        
        # 预编译JAX梯度函数（只编译一次）
        grad_fn = parent._get_gradient_function()
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        # 定义优化目标（使用预编译的梯度）
        def objective_and_grad(constants_np):
            constants_jax = jnp.array(constants_np)
            
            # 计算梯度（使用预编译的函数）
            grad = grad_fn(constants_jax, X_jax, y_jax)
            
            # 计算损失（使用快速的NumPy执行）
            temp_expr = parent.update_constants(constants_np)
            fitness = temp_expr.fitness(X, y)
            loss = -fitness if parent.metric.greater_is_better else fitness
            
            return float(loss), np.array(grad)
        
        # 多次重启优化
        best_loss = float('inf')
        best_constants = initial_constants.copy()
        
        for restart in range(optimizer_nrestarts):
            # 第一次使用原始值，后续添加噪声
            if restart == 0:
                x0 = initial_constants.copy()
            else:
                # 噪声强度递减（避免后期扰动过大）
                noise_scale = 0.05 / np.sqrt(restart)
                # restart=1: 5%, restart=2: 3.5%, restart=3: 2.9%
                noise = self.random_state.normal(0, noise_scale, size=len(initial_constants))
                constants_scale = np.abs(initial_constants) + 1e-6  # 处理零值
                x0 = initial_constants + noise * constants_scale
            
            # 执行优化
            if optimizer_algorithm in METHODS_WITH_EPS:
                result = minimize(
                    objective_and_grad, x0,
                    method=optimizer_algorithm, jac=True,
                    options={'maxiter': optimizer_iterations, 
                             'eps': self.constants_tolerance
                    }
                )
            else:
                result = minimize(
                    objective_and_grad, x0,
                    method=optimizer_algorithm, jac=True,
                    options={'maxiter': optimizer_iterations}
                )
            
            # 更新最佳结果
            if result.fun < best_loss:
                best_loss = result.fun
                best_constants = result.x
        
        # 创建优化后的表达式
        optimized_expr = parent.update_constants(best_constants)
        
        # 最终适应度
        final_fitness = -best_loss if parent.metric.greater_is_better else best_loss
        
        return optimized_expr, True, final_fitness

    def optimize_aggregations(self, parent: Expression, X, y, 
                              optimizer_iterations=10, max_shift_ratio=0.1, 
                              early_exaggeration_iter=3, early_stopping_patience=4, 
                              exaggeration_factor=2.5) -> Tuple[Optional['Expression'], bool]:
        """
        优化表达式中的 DynamicAggregation 节点参数（局部搜索）
        
        使用贪心爬山算法优化聚合窗口的位置和大小
        
        参数
        ----------
        X : array-like, shape (n_samples, n_features)
            输入特征
        y : array-like, shape (n_samples,)
            目标变量
        sample_weight : array-like, shape (n_samples,)
            样本权重
        optimizer_iterations : int
            最大迭代次数
        max_shift_ratio : float
            窗口移动的最大比例（相对于当前窗口大小）
        early_exaggeration_iter : int
            早期放大阶段的迭代次数（使用更大的搜索步长）
        early_stopping_patience : int
            早停耐心值（连续无改进的迭代次数）
        exaggeration_factor : float
            早期阶段的步长放大因子
        
        返回
        -------
        new_expr : AdvancedLinearExpression
            优化后的表达式
        success : bool
            优化是否成功（相比初始表达式有改进）
        
        优化策略：
        1. 移动窗口（左/右平移）
        2. 扩展窗口（增加覆盖范围）
        3. 收缩窗口（减少覆盖范围）
        4. 调整边界（单独移动 v_start 或 v_end）
        
        时间复杂度：O(iterations * neighbors * n_samples)
        其中 neighbors ≈ O(n_aggregations * shift_amount * 操作类型)
        """
        # 1. 收集所有聚合节点的索引（一次遍历）
        agg_indices = [(i, gene) for i, gene in enumerate(self.genes)
                      if isinstance(gene, DynamicAggregation)]
        
        if not agg_indices:
            return None, False, np.nan
        
        # 2. 创建副本并记录初始状态
        new_expr = parent.copy()
        n_variables = X.shape[1]
        early_exaggeration_iter = min(early_exaggeration_iter, optimizer_iterations)
        
        # 初始状态：[(index, v_start, v_end, op_name, valid_op), ...]
        initial_states = [
            {
                'idx': agg_idx,
                'v_start': agg.v_start,
                'v_end': agg.v_end,
                'op_name': agg.op_name,
                'valid_op': agg.valid_op
            }
            for agg_idx, agg in agg_indices
        ]
        
        # 3. 计算初始适应度
        best_fitness = new_expr.fitness(X, y)
        best_states = [s.copy() for s in initial_states]
        
        # 4. 定义快速应用状态的函数
        def apply_states_fast(states):
            """快速应用聚合参数（原地修改）"""
            for state in states:
                idx = state['idx']
                new_expr.genes[idx] = DynamicAggregation(
                    v_start=state['v_start'],
                    v_end=state['v_end'],
                    op_name=state['op_name'],
                    n_variables=n_variables,
                    valid_op=state['valid_op']
                )
        
        # 6. 贪心爬山算法
        current_states = initial_states
        current_fitness = best_fitness
        
        iterations = 0
        no_improvement_count = 0
        
        while iterations < optimizer_iterations and no_improvement_count < early_stopping_patience:
            iterations += 1
            improved = False
            
            # 动态调整步长（早期放大）
            current_exaggeration_factor = (exaggeration_factor 
                                          if iterations <= early_exaggeration_iter 
                                          else 1.0)
            
            # 生成邻居
            neighbors = self._get_neighbors(
                n_variables, current_states, max_shift_ratio, 
                current_exaggeration_factor, self.random_state
            )
            
            # 评估邻居（首次改进即停止）
            for neighbor_states in neighbors:
                apply_states_fast(neighbor_states)
                
                neighbor_fitness = new_expr.fitness(X, y)
                
                # 检查是否改进
                is_better = (neighbor_fitness > current_fitness 
                            if parent.metric.greater_is_better 
                            else neighbor_fitness < current_fitness)
                
                if is_better:
                    # 接受改进
                    current_states = neighbor_states
                    current_fitness = neighbor_fitness
                    improved = True
                    
                    # 更新全局最佳
                    best_states = [s.copy() for s in current_states]
                    best_fitness = current_fitness
                    no_improvement_count = 0
                    break  # 首次改进策略
            
            # 早停计数（跳过早期放大阶段）
            if not improved and iterations > early_exaggeration_iter:
                no_improvement_count += 1
        
        # 7. 应用最佳状态
        apply_states_fast(best_states)
        raw_fitness = best_fitness
        
        return new_expr, True, raw_fitness

    # 5. 邻居生成函数（批量生成候选）
    @staticmethod
    def _get_neighbors(n_variables, states, max_shift_ratio, 
                       current_exaggeration_factor, random_state):
        """
        生成邻居状态
        
        返回：[(new_states, operation_description), ...]
        """
        neighbors = []
        
        for i, state in enumerate(states):
            v_start, v_end = state['v_start'], state['v_end']
            window_size = v_end - v_start + 1
            
            # 计算步长
            base_max_shift = max(1, int(window_size * max_shift_ratio) + 1)
            max_shift = int(base_max_shift * current_exaggeration_factor)
            max_shift = max(1, max_shift)
            
            # 随机步长
            shift_amount = random_state.randint(1, max_shift + 1)
            
            # --- 操作1: 平移窗口 ---
            for direction in [-1, 1]:
                shift = direction * shift_amount
                new_start = v_start + shift
                new_end = v_end + shift
                
                if (0 <= new_start < new_end < n_variables) and \
                        (new_start != v_start or new_end != v_end):
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_start'] = new_start
                    new_states[i]['v_end'] = new_end
                    neighbors.append(new_states)
            
            # --- 操作2: 仅移动 v_start ---
            for direction in [-1, 1]:
                shift = direction * shift_amount
                new_start = v_start + shift
                
                if (0 <= new_start < v_end - 1 < n_variables) and \
                        (new_start != v_start):
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_start'] = new_start
                    neighbors.append(new_states)
            
            # --- 操作3: 仅移动 v_end ---
            for direction in [-1, 1]:
                shift = direction * shift_amount
                new_end = v_end + shift
                
                if (0 <= v_start < v_start + 1 < new_end < n_variables) and \
                        (new_end != v_end):
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_end'] = new_end
                    neighbors.append(new_states)
            
            # --- 操作4: 扩展窗口 ---
            expand_amount = random_state.randint(1, max_shift + 1)
            
            # 双向扩展
            left_expand = expand_amount // 2
            right_expand = expand_amount - left_expand
            new_start = max(0, v_start - left_expand)
            new_end = min(n_variables - 1, v_end + right_expand)
            
            if (new_start < new_end) and \
                    (new_start != v_start or new_end != v_end):
                new_states = [s.copy() for s in states]
                new_states[i]['v_start'] = new_start
                new_states[i]['v_end'] = new_end
                neighbors.append(new_states)
            
            # 仅左扩展
            if (v_start > 0) and (new_start != v_start):
                new_start = max(0, v_start - expand_amount)
                new_states = [s.copy() for s in states]
                new_states[i]['v_start'] = new_start
                neighbors.append(new_states)
            
            # 仅右扩展
            if (v_end < n_variables - 1) and (new_end != v_end):
                new_end = min(n_variables - 1, v_end + expand_amount)
                new_states = [s.copy() for s in states]
                new_states[i]['v_end'] = new_end
                neighbors.append(new_states)
            
            # --- 操作5: 收缩窗口（保持至少2个元素）---
            if window_size > 2:
                max_shrink = min(max_shift, window_size - 2)
                shrink_amount = random_state.randint(1, max_shrink + 1)
                
                # 双向收缩
                left_shrink = shrink_amount // 2
                right_shrink = shrink_amount - left_shrink
                new_start = v_start + left_shrink
                new_end = v_end - right_shrink
                
                if (new_start < new_end) and \
                        (new_start != v_start or new_end != v_end):
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_start'] = new_start
                    new_states[i]['v_end'] = new_end
                    neighbors.append(new_states)
                
                # 仅左收缩
                new_start = min(v_start + shrink_amount, v_end - 1)
                if new_start < v_end and new_start != v_start:
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_start'] = new_start
                    neighbors.append(new_states)
                
                # 仅右收缩
                new_end = max(v_start + 1, v_end - shrink_amount)
                if v_start < new_end and new_end != v_end:
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_end'] = new_end
                    neighbors.append(new_states)
        
        return neighbors

    def simplify(self, parent: Expression) -> Tuple[Optional['Expression'], bool]:
        """
        简化表达式（基于模式匹配）
        
        使用 RPN 栈进行迭代简化，以正确处理嵌套表达式。
        
        支持的规则：
        - 常量折叠 (e.g., 1 + 1 -> 2)
        - 恒等操作 (e.g., x + 0 -> x, x * 1 -> x)
        - 零/湮灭操作 (e.g., x * 0 -> 0, 0 / x -> 0)
        - 保护性除零 (e.g., x / 0 -> 1)
        - 幂等操作 (e.g., x - x -> 0, x / x -> 1)
        - 逆运算 (e.g., -(-x) -> x, x - (-y) -> x + y)
        - 分数简化 (e.g., (x * y) / x -> y, x / (x * y) -> inv(y))
        """
        # 'stack' 是一个“元堆栈”，它包含 *子表达式（基因列表）*
        # e.g., for [x, 2, *], stack will be [[x], [Constant(2)]]
        # then they are popped and pushed back as [[x, Constant(2), *]]
        stack: List[List[NodeContent]] = []
        
        # 跟踪是否发生了任何变化
        simplified_once = False

        for gene in parent.genes:
            if gene.degree == 0:
                # 叶子节点，将其作为 [gene] 列表压栈
                stack.append([gene])
                continue

            # 操作符，弹出所需的操作数（它们是列表）
            if len(stack) < gene.degree:
                # 这种情况不应该在合法的 RPN 中发生
                # 停止简化，将原基因压栈
                unsimplified_expr = []
                for arg_genes in stack:
                    unsimplified_expr.extend(arg_genes)
                unsimplified_expr.append(gene)
                stack = [unsimplified_expr]
                break

            args = [stack.pop() for _ in range(gene.degree)]
            args.reverse() # 恢复 L-R 顺序
            
            # 尝试简化
            simplified_genes = None
            if gene.degree == 1:
                simplified_genes = self._try_simplify_unary(args[0], gene)
            elif gene.degree == 2:
                simplified_genes = self._try_simplify_binary(args[0], args[1], gene)
            
            if simplified_genes is not None:
                # 简化成功
                stack.append(simplified_genes)
                simplified_once = True
            else:
                # 无法简化，将原始 RPN 重新组合并压栈
                # [op1_genes] + [op2_genes] + [op]
                unsimplified_expr = []
                for arg_genes in args:
                    unsimplified_expr.extend(arg_genes)
                unsimplified_expr.append(gene)
                stack.append(unsimplified_expr)

        # 最终，栈中应该只剩下一个元素：完整的（可能简化的）表达式
        if len(stack) != 1:
            # 简化过程出错或 RPN 本身无效
            return None, False
            
        new_genes = stack[0]

        # 检查是否有简化
        if not simplified_once:
             return None, False
        
        new_expr = self.reproduce(new_genes)
        # 比较新旧表达式的字符串形式，因为 [x, 1, *] -> [x] 长度可能不变
        if new_expr == parent:
            return None, False
        
        return new_expr, True

    def _is_constant_value(self, gene_list: List[NodeContent], value: float) -> bool:
        """辅助函数：检查一个子表达式是否为特定值的常量"""
        if len(gene_list) == 1 and isinstance(gene_list[0], Constant):
            return abs(gene_list[0].value - value) < self.constants_tolerance
        return False
        
    def _get_constant_value(self, gene_list: List[NodeContent]) -> Optional[float]:
        """辅助函数：如果子表达式是常量，返回其值"""
        if len(gene_list) == 1 and isinstance(gene_list[0], Constant):
            return gene_list[0].value
        return None
        
    def _find_subexpr_start_in_list(self, gene_list: List[NodeContent], end_idx: int) -> int:
        """
        在 RPN 基因列表切片中，找到在 end_idx 处结束的子表达式的起始索引。
        """
        gene = gene_list[end_idx]
        if gene.degree == 0:
            return end_idx
        
        needed = gene.degree
        pos = end_idx - 1
        
        while needed > 0 and pos >= 0:
            g = gene_list[pos]
            needed = needed - 1 + g.degree
            pos -= 1
        
        # pos 是最后一个被消耗的基因的索引
        # 所以起始位置是 pos + 1
        return pos + 1

    def _try_simplify_binary(self, 
                             left_genes: List[NodeContent], 
                             right_genes: List[NodeContent], 
                             op: Operator) -> Optional[List[NodeContent]]:
        """
        尝试简化二元操作（RPN 列表版）
        
        返回：简化后的基因列表，或 None（无法简化）
        """
        op_name = op.name
        
        # 1. 常量折叠
        left_val = self._get_constant_value(left_genes)
        right_val = self._get_constant_value(right_genes)
        
        if left_val is not None and right_val is not None:
            try:
                # 检查除零 (x / 0)
                if op_name in ['div', '/'] and abs(right_val) < self.constants_tolerance:
                    # 保护性除零：x / 0 = 1
                    return [Constant(1.0)]
                
                result = op(left_val, right_val)
                return [Constant(result)]
            except:
                return None # 计算失败（例如 log(-1)）

        # 2. 恒等操作 (Identity)
        
        # x + 0 = x
        if op_name in ['add', '+'] and self._is_constant_value(right_genes, 0):
            return left_genes
        # 0 + x = x
        if op_name in ['add', '+'] and self._is_constant_value(left_genes, 0):
            return right_genes
            
        # x - 0 = x
        if op_name in ['sub', '-'] and self._is_constant_value(right_genes, 0):
            return left_genes
            
        # x * 1 = x
        if op_name in ['mul', '*'] and self._is_constant_value(right_genes, 1):
            return left_genes
        # 1 * x = x
        if op_name in ['mul', '*'] and self._is_constant_value(left_genes, 1):
            return right_genes
            
        # x / 1 = x
        if op_name in ['div', '/'] and self._is_constant_value(right_genes, 1):
            return left_genes
            
        # 3. 零/湮灭操作 (Annihilator)
        
        # x * 0 = 0
        if op_name in ['mul', '*'] and self._is_constant_value(right_genes, 0):
            return [Constant(0.0)]
        # 0 * x = 0
        if op_name in ['mul', '*'] and self._is_constant_value(left_genes, 0):
            return [Constant(0.0)]
            
        # 0 / x = 0 (x != 0)
        if op_name in ['div', '/'] and self._is_constant_value(left_genes, 0):
            if right_val is not None and abs(right_val) < self.constants_tolerance:
                # 0 / 0 -> 保护性处理
                return [Constant(1.0)]
            # 0 / x = 0
            return [Constant(0.0)]
            
        # x / 0 = 1 (保护性处理, 在常量折叠中已处理, 这里再加一层)
        if op_name in ['div', '/'] and self._is_constant_value(right_genes, 0):
            return [Constant(1.0)]
            
        # 4. 幂等操作 (Idempotent)
        
        # x / x = 1
        if op_name in ['div', '/'] and left_genes == right_genes:
            return [Constant(1.0)]
            
        # x - x = 0
        if op_name in ['sub', '-'] and left_genes == right_genes:
            return [Constant(0.0)]
            
        # 5. 逆运算 (Inverse)
        
        # x + (-y) = x - y
        # 检查 right_genes 是否为 neg(y) -> [y, neg]
        if op_name in ['add', '+'] and \
           len(right_genes) > 1 and \
           right_genes[-1].name == 'neg':
            y_genes = right_genes[:-1] # 提取 y
            sub_op = _operator_map.get('sub') 
            if sub_op:
                return left_genes + y_genes + [sub_op]
        
        # x - (-y) = x + y
        if op_name in ['sub', '-'] and \
           len(right_genes) > 1 and \
           right_genes[-1].name == 'neg':
            y_genes = right_genes[:-1] # 提取 y
            add_op = _operator_map.get('add') 
            if add_op:
                return left_genes + y_genes + [add_op]
                
        # (y - x) + x = y
        # 检查 left_genes 是否为 (y - x) -> [y, x, sub]
        if op_name in ['add', '+'] and \
           len(left_genes) > 2 and \
           left_genes[-1].name == 'sub':
            
            op_pos = len(left_genes) - 1
            # 找到 x 和 y 的 RPN 切片
            start_x = self._find_subexpr_start_in_list(left_genes, op_pos)
            end_x = op_pos - 1
            start_y = self._find_subexpr_start_in_list(left_genes, start_x - 1)
            end_y = start_x - 1
            
            x_genes = left_genes[start_x : end_x + 1]
            y_genes = left_genes[start_y : end_y + 1]
            
            if x_genes == right_genes:
                return y_genes
                
        # x + (y - x) = y
        # 检查 right_genes 是否为 (y - x) -> [y, x, sub]
        if op_name in ['add', '+'] and \
           len(right_genes) > 2 and \
           right_genes[-1].name == 'sub':
            
            op_pos = len(right_genes) - 1
            start_x = self._find_subexpr_start_in_list(right_genes, op_pos)
            end_x = op_pos - 1
            start_y = self._find_subexpr_start_in_list(right_genes, start_x - 1)
            end_y = start_x - 1
            
            x_genes = right_genes[start_x : end_x + 1]
            y_genes = right_genes[start_y : end_y + 1]
            
            if x_genes == left_genes:
                return y_genes
                
        # 6. 分数简化
        # 检查 left_genes 是否为 (A * B) -> [A, B, mul]
        if op_name in ['div', '/'] and \
           len(left_genes) > 2 and \
           left_genes[-1].name == 'mul':
            op_pos = len(left_genes) - 1
            start_B = self._find_subexpr_start_in_list(left_genes, op_pos)
            end_B = op_pos - 1
            start_A = self._find_subexpr_start_in_list(left_genes, start_B - 1)
            end_A = start_B - 1
            
            A_genes = left_genes[start_A : end_A + 1]
            B_genes = left_genes[start_B : end_B + 1]
            
            # (A * B) / A = B
            if A_genes == right_genes:
                return B_genes
            # (A * B) / B = A
            if B_genes == right_genes:
                return A_genes
                
        # 检查 right_genes 是否为 (A * B) -> [A, B, mul]
        if op_name in ['div', '/'] and \
           len(right_genes) > 2 and \
           right_genes[-1].name == 'mul':
            op_pos = len(right_genes) - 1
            start_B = self._find_subexpr_start_in_list(right_genes, op_pos)
            end_B = op_pos - 1
            start_A = self._find_subexpr_start_in_list(right_genes, start_B - 1)
            end_A = start_B - 1
            
            A_genes = right_genes[start_A : end_A + 1]
            B_genes = right_genes[start_B : end_B + 1]
            inv_op = _operator_map.get('inv') # 需要 'inv' (1/x) 在 self.operators 中
            
            # x / (x * B) = 1 / B
            if A_genes == left_genes and inv_op:
                return B_genes + [inv_op]
            # x / (A * x) = 1 / A
            if B_genes == left_genes and inv_op:
                return A_genes + [inv_op]
                
        # 检查 right_genes 是否为 (A / B) -> [A, B, div]
        if op_name in ['mul', '*'] and \
           len(right_genes) > 2 and \
           right_genes[-1].name == 'div':
            op_pos = len(right_genes) - 1
            start_B = self._find_subexpr_start_in_list(right_genes, op_pos)
            end_B = op_pos - 1
            start_A = self._find_subexpr_start_in_list(right_genes, start_B - 1)
            end_A = start_B - 1
            
            A_genes = right_genes[start_A : end_A + 1]
            B_genes = right_genes[start_B : end_B + 1]
            
            # B * (A / B) = A
            if B_genes == left_genes:
                return A_genes
                
        # 检查 left_genes 是否为 (A / B) -> [A, B, div]
        if op_name in ['mul', '*'] and \
           len(left_genes) > 2 and \
           left_genes[-1].name == 'div':
            op_pos = len(left_genes) - 1
            start_B = self._find_subexpr_start_in_list(left_genes, op_pos)
            end_B = op_pos - 1
            start_A = self._find_subexpr_start_in_list(left_genes, start_B - 1)
            end_A = start_B - 1
            
            A_genes = left_genes[start_A : end_A + 1]
            B_genes = left_genes[start_B : end_B + 1]
            
            # (A / B) * B = A
            if B_genes == right_genes:
                return A_genes
        
        return None

    def _try_simplify_unary(self, 
                            arg_genes: List[NodeContent], 
                            op: Operator) -> Optional[List[NodeContent]]:
        """尝试简化一元操作（RPN 列表版）"""
        op_name = op.name
        
        # 1. 常量折叠
        arg_val = self._get_constant_value(arg_genes)
        
        if arg_val is not None:
            try:
                result = op(arg_val)
                return [Constant(result)]
            except:
                return None
        
        # 2. 逆运算
        # -(-x) = x
        # 检查 arg_genes 是否为 neg(x) -> [x, neg]
        if op_name == 'neg' and \
           len(arg_genes) > 1 and \
           arg_genes[-1].name == 'neg':
            x_genes = arg_genes[:-1] # 提取 x
            return x_genes
            
        # inv(inv(x)) = x
        if op_name == 'inv' and \
           len(arg_genes) > 1 and \
           arg_genes[-1].name == 'inv':
            x_genes = arg_genes[:-1] # 提取 x
            return x_genes

        return None




class ExpressionSetGP:
    def __init__(self,
                 generator: ExprSetGenerator,
                 gpoperator: ExpressionGP,
                 set_mutation_weights: dict = None,
                 set_crossover_method: str = 'single_point',
                 random_state: Union[int, np.random.Generator] = None):
        self.generator = generator
        self.gpoperator = gpoperator
        self.maxsize = generator.maxsize
        self.set_mutation_weights = set_mutation_weights
        self.set_crossover_method = set_crossover_method
        self.random_state = check_random_state(random_state)

    def reproduce(self, expressions) -> 'ExpressionSet':
        """Returns a list of nodes in the tree in pre-order."""
        return ExpressionSet(
            expressions=expressions,
            out_func=self.generator.out_func, 
            metric=self.generator.metric
        )

    def crossover(self, 
                  parent: ExpressionSet, donor: ExpressionSet, 
        ) -> Tuple[Optional[ExpressionSet], Optional[ExpressionSet], bool]:
        """
        Performs crossover between two ExpressionSets, producing two new offspring.
        """
        if len(parent) != len(donor):
            raise ValueError("Crossover requires the same length of expressions for both parents.")

        if self.random_state.uniform() < self.set_crossover_probability:
            if self.set_crossover_method == 'single_point':
                cross_point = self.random_state.randint(1, len(parent.expressions))
                exprs1 = parent.expressions[:cross_point] + donor.expressions[cross_point:]
                exprs2 = donor.expressions[:cross_point] + parent.expressions[cross_point:]
            elif self.set_crossover_method == 'two_point':
                points = self.random_state.choice(range(0, len(parent.expressions)), 2, replace=False)
                start, end = min(points), max(points)
                exprs1 = parent.expressions[:start] + donor.expressions[start:end] + parent.expressions[end:]
                exprs2 = donor.expressions[:start] + parent.expressions[start:end] + donor.expressions[end:]
            elif self.set_crossover_method == 'multi_point':
                num_points = self.random_state.randint(1, len(parent.expressions))
                points = self.random_state.choice(range(0, len(parent.expressions)), num_points, replace=False)
                exprs1 = [None] * len(parent.expressions)
                exprs2 = [None] * len(parent.expressions)
                for i in range(len(parent.expressions)):
                    if i not in points:
                        exprs1[i] = parent.expressions[i]
                        exprs2[i] = donor.expressions[i]
                    else:
                        exprs1[i] = donor.expressions[i]
                        exprs2[i] = parent.expressions[i]
            else:
                raise ValueError(f"Invalid set_crossover_method: {self.set_crossover_method}")
        else:
            exprs1 = [None] * len(parent.expressions)
            exprs2 = [None] * len(parent.expressions)
            for i in range(len(parent.expressions)):
                expr1, expr2 = parent.expressions[i], donor.expressions[i]
                if expr1 is not None and expr2 is not None:
                    new_expr1, new_expr2, success = self.gpoperator.crossover(expr1, expr2)
                    if success:
                        exprs1[i], exprs2[i] = new_expr1, new_expr2
                    else:
                        exprs1[i], exprs2[i] = expr1, expr2
                else:
                    exprs1[i], exprs2[i] = expr1, expr2

        offspring1 = self.reproduce(exprs1)
        offspring2 = self.reproduce(exprs2)

        return offspring1, offspring2, True

    def _condition_mutation_weights(self, exprset) -> dict:
        """
        Dynamically adjusts mutation weights based on the current state of the expression set.
        """
        weights = self.set_mutation_weights.copy()

        # If order is fixed, no adding or deleting expressions
        if self.generator.fixed:
            weights['add_expr'] = 0.0
            weights['delete_expr'] = 0.0
        else:
            # If at max order, no adding
            if exprset.order >= self.generator.maxorder:
                weights['add_expr'] = 0.0
            # If at min order, no deleting
            if exprset.order <= self.minorder:
                weights['delete_expr'] = 0.0
        
        if exprset.order < 2:
            weights['swap_exprs'] = 0.0
        
        return weights

    def mutation(self, parent: ExpressionSet) -> Tuple[Optional['ExpressionSet'], bool]:
        """Perform a mutation operation on the ExpressionSet. 
        
        This method acts as a dispatcher, selecting one of several mutation
        strategies based on pre-defined weights and executing it.
        """
        conditioned_weights = self._condition_mutation_weights()
        mutation_name = weighted_random_choice(conditioned_weights, self.random_state)
        
        # Dispatch to the correct mutation method
        if mutation_name == 'mutate_expr':
            new_expr_set, mutation_succeeded = self.mutate_expr(parent)
        elif mutation_name == 'randomize_expr':
            new_expr_set, mutation_succeeded = self.randomize_expr(parent)
        elif mutation_name == 'do_nothing_set':
            new_expr_set, mutation_succeeded = self.do_nothing_set(parent)
        elif mutation_name == 'swap_exprs':
            new_expr_set, mutation_succeeded = self.swap_exprs(parent)
        elif mutation_name == 'add_expr':
            new_expr_set, mutation_succeeded = self.add_expr(parent)
        elif mutation_name == 'delete_expr':
            new_expr_set, mutation_succeeded = self.delete_expr(parent)
        elif mutation_name == 'mutate_constant':
            new_expr_set, mutation_succeeded = self.mutate_constant(parent)
        elif mutation_name == 'simplify_set':
            new_expr_set, mutation_succeeded = self.simplify(parent)
        elif mutation_name == 'randomize_set':
            new_expr_set, mutation_succeeded = self.randomize_set(parent)
        else:
            raise ValueError(f'Invalid mutation name: {mutation_name}')

        return new_expr_set, mutation_succeeded, mutation_name

    def mutate_expr(self, parent: ExpressionSet):
        """Perform the mutation operation on a single GeneticExpression."""
        valid_points = [i for i, v in enumerate(parent.expressions) if v is not None]
        if not valid_points:
            return None, False
        
        mutation_point = self.random_state.choice(valid_points)
        parent_expr = parent.expressions[mutation_point]
        
        mutated_expr, mutation_succeeded = self.gpoperator.mutation(parent_expr)
        
        # 1. 提早失败
        if not mutation_succeeded:
            return None, False
        
        # 2. 构建一个全新的列表
        new_exprs = (
            parent.expressions[:mutation_point] + 
            [mutated_expr] + 
            parent.expressions[mutation_point+1:]
        )
        
        # 3. 使用新列表创建副本
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True

    def mutate_constant(self, parent: ExpressionSet):
        """Mutate a random constant in the expression set."""
        # 1. 找到可突变的表达式
        valid_points = [i for i, v in enumerate(parent.expressions) if v is not None]
        if not valid_points:
            return None, False
        
        # 尝试随机选择一个进行突变
        #（注意：如果选中的没有常量，parent_expr.mutate_constant() 会自动返回 False）
        mutation_point = self.random_state.choice(valid_points)
        parent_expr = parent.expressions[mutation_point]

        # 2. 尝试突变
        mutated_expr, mutation_succeeded = self.gpoperator.mutation(parent_expr)

        if not mutation_succeeded:
            return None, False
            
        # 3. 构建新列表
        new_exprs = (
            parent.expressions[:mutation_point] + 
            [mutated_expr] + 
            parent.expressions[mutation_point+1:]
        )
        
        # 4. 创建副本
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True

    def delete_expr(self, parent: ExpressionSet):
        """Deletes an expression from the set."""
        valid_points = [i for i, v in enumerate(parent.expressions) if v is not None]
        
        # 1. 检查是否满足删除条件
        if len(valid_points) <= self.generator.minorder:
            return None, False
        if not valid_points:
            return None, False

        # 2. 选择删除点
        point_to_delete = self.random_state.choice(valid_points)
        
        # 3. 构建新列表 (用 None 替换)
        new_exprs = (
            parent.expressions[:point_to_delete] + 
            [None] + 
            parent.expressions[point_to_delete+1:]
        )
        
        # 4. 创建副本
        new_expr_set = self.reproduce(new_exprs, self.random_state)
        
        return new_expr_set, True

    def swap_exprs(self, parent: ExpressionSet):
        """Swaps two expressions in the set."""
        valid_points = [i for i, v in enumerate(parent.expressions) if v is not None]
        if len(valid_points) < 2:
            return None, False
        
        idx1, idx2 = self.random_state.choice(valid_points, size=2, replace=False)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        # Swap in the new set
        new_exprs = (
            parent.expressions[:idx1] + 
            [parent.expressions[idx2]] + 
            parent.expressions[idx1+1:idx2] + 
            [parent.expressions[idx1]] + 
            parent.expressions[idx2+1:]
        )
        new_expr_set = self.reproduce(new_exprs)
        new_expr_set.mutation_name = 'swap_exprs'
        
        return new_expr_set, True

    def randomize_set(self, parent: ExpressionSet):
        """Replace all expressions with new, randomly generated."""
        new_expr_set = self.generator.generate_random_exprset()
        return new_expr_set, True

    def randomize_expr(self, parent: ExpressionSet):
        """Replace a random expression with a new, randomly generated one."""
        # 1. 随机选择一个替换点（可以替换 None）
        mutation_point = self.random_state.choice(len(parent.expressions))
        
        # 2. 生成新表达式
        new_expr = self.generator.generate_random_expr()
        
        # 3. 构建新列表
        new_exprs = (
            parent.expressions[:mutation_point] + 
            [new_expr] + 
            parent.expressions[mutation_point+1:]
        )

        # 4. 创建副本
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True

    def simplify(self, parent: ExpressionSet):
        """Simplifies all expressions in the set."""
        expressions = [None] * len(parent.expressions)
        any_simplified = False
        for i, expr in enumerate(parent.expressions):
            if expr is not None:
                simplified_expr, success = self.gpoperator.simplify(expr)
                if success:
                    expressions[i] = simplified_expr
                    any_simplified = True
                else:
                    expressions[i] = expr
        
        if not any_simplified:
            return None, False
        
        new_expr_set = self.reproduce(expressions)
        
        return new_expr_set, False

    def do_nothing_set(self, parent: ExpressionSet):
        """Return a new, identical expression set."""
        new_expr_set = self.reproduce(parent.expressions)
        return new_expr_set, True

    def add_expr(self, parent: ExpressionSet):
        """Add a new expression to the expression set (in a None slot)."""
        # 1. 找到一个空位
        empty_indices = [i for i, expr in enumerate(parent.expressions) if expr is None]
        if not empty_indices:
            return None, False

        point_to_add = self.random_state.choice(empty_indices)
        
        # 2. 生成新表达式
        new_expr = self.gpoperator.generate_random_expr()
        
        # 3. 构建新列表
        new_exprs = (
            parent.expressions[:point_to_add] + 
            [new_expr] + 
            parent.expressions[point_to_add+1:]
        )
        
        # 4. 创建副本
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True

    def optimize_constants(self, parent: ExpressionSet, X, y, 
                           optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, 
                           optimizer_iterations=10) -> Tuple[Optional['ExpressionSet'], bool]:
        # 收集所有常量节点，提取初始常量值
        constant_indices = []
        for expr_idx, expr in enumerate(parent.expressions):
            if expr is not None:
                constant_indices.extend([
                    (expr_idx, gene_idx) for gene_idx, gene in enumerate(expr.genes) 
                        if isinstance(gene, Constant)
                ])
        
        if not constant_indices:
            return None, False
        
        # 转换为JAX数组
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        # 提取初始常量值（NumPy 数组以便向量化操作）
        initial_constants = jnp.array([
            self[expr_idx][gene_idx].value for (expr_idx, gene_idx) in constant_indices
        ])
        
        # 预编译JAX梯度函数（只编译一次）
        grad_fn = parent._get_gradient_function()
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)

        # 定义优化目标（使用预编译的梯度）
        def objective_and_grad(constants_np):
            constants_jax = jnp.array(constants_np)
            
            # 计算梯度（使用预编译的函数）
            grad = grad_fn(constants_jax, X_jax, y_jax)
            
            # 计算损失（使用快速的NumPy执行）
            temp_expr = parent.update_constants(constants_np)
            fitness = temp_expr.fitness(X, y)
            loss = -fitness if parent.metric.greater_is_better else fitness
        
        # 多次重启优化
        best_loss = float('inf')
        best_constants = initial_constants.copy()
        
        for restart in range(optimizer_nrestarts):
            # 第一次使用原始值，后续添加噪声
            if restart == 0:
                x0 = initial_constants.copy()
            else:
                # 噪声强度递减（避免后期扰动过大）
                noise_scale = 0.05 / np.sqrt(restart)
                # restart=1: 5%, restart=2: 3.5%, restart=3: 2.9%
                noise = self.random_state.normal(0, noise_scale, size=len(initial_constants))
                constants_scale = np.abs(initial_constants) + 1e-6  # 处理零值
                x0 = initial_constants + noise * constants_scale
            
            # 执行优化
            if optimizer_algorithm in METHODS_WITH_EPS:
                result = minimize(
                    objective_and_grad, x0,
                    method=optimizer_algorithm, jac=True,
                    options={
                        'maxiter': optimizer_iterations,
                        'eps': self.gpoperator.constants_tolerance
                    }
                )
            else:
                result = minimize(
                    objective_and_grad, x0,
                    method=optimizer_algorithm, jac=True,
                    options={'maxiter': optimizer_iterations}
                )
            
            # 更新最佳结果
            if result.fun < best_loss:
                best_loss = result.fun
                best_constants = result.x
        
        # 创建优化后的表达式
        optimized_expr_set = parent.update_constants(best_constants)
        
        # 最终适应度
        final_fitness = -best_loss if parent.metric.greater_is_better else best_loss
        
        return optimized_expr_set, True, final_fitness

    def optimize_aggregations(
        self, parent: ExpressionSet, X, y, 
        optimizer_iterations=10, max_shift_ratio=0.1, 
        early_exaggeration_iter=3, early_stopping_patience=4, exaggeration_factor=2.5
    ) -> Tuple[Optional['ExpressionSet'], bool]:
        """
        优化表达式集合中的聚合节点（高效版本）
        
        关键优化：
        1. 一次性收集所有聚合节点
        2. 状态字典批量操作（减少对象创建）
        3. 快速状态应用（原地修改节点）
        4. 向量化邻居生成（减少循环开销）
        
        时间复杂度：
        - 节点收集: O(sum(tree_size)) 一次
        - 每次迭代: O(neighbors × k) 其中 k = 聚合节点数
        - 总计: O(iterations × neighbors × k)
        
        性能提升：相比原实现约 2-3x
        """
        # 1. 一次性收集所有聚合节点（关键优化）
        agg_indices = []
        for expr_idx, expr in enumerate(parent.expressions):
            if expr is not None:
                agg_indices.extend([
                    (expr_idx, gene_idx) for gene_idx, gene in enumerate(expr.genes)
                        if isinstance(gene, DynamicAggregation)
                ])
        
        if not agg_indices:
            return None, False
        
        # 2. 创建副本
        n_variables = X.shape[1]
        new_expr_set = parent.copy()
        early_exaggeration_iter = min(early_exaggeration_iter, optimizer_iterations)
        
        # 3. 记录初始状态（使用列表推导式）
        initial_states = [
            {
                'expr_idx': expr_idx, 'gene_idx': gene_idx,
                'v_start': new_expr_set[expr_idx][gene_idx].v_start,
                'v_end': new_expr_set[expr_idx][gene_idx].v_end,
                'op_name': new_expr_set[expr_idx][gene_idx].op_name,
                'valid_op': new_expr_set[expr_idx][gene_idx].valid_op
            }
            for expr_idx, gene_idx in agg_indices
        ]
        
        # 4. 计算初始适应度
        best_fitness = new_expr_set.fitness(X, y)
        best_states = [s.copy() for s in initial_states]
        
        # 5. 定义快速状态应用函数（闭包优化）
        def apply_states_fast(states):
            """快速应用聚合参数（原地修改）"""
            for state in states:
                expr_idx, gene_idx = state['expr_idx'], state['gene_idx']
                new_expr_set.expressions[expr_idx].genes[gene_idx] = DynamicAggregation(
                    v_start=state['v_start'],
                    v_end=state['v_end'],
                    op_name=state['op_name'],
                    n_variables=n_variables,
                    valid_op=state['valid_op']
                )
        
        # 7. 贪心爬山算法
        current_states = initial_states
        current_fitness = best_fitness
        
        iterations = 0
        no_improvement_count = 0
        
        while iterations < optimizer_iterations and no_improvement_count < early_stopping_patience:
            iterations += 1
            improved = False
            
            # 动态调整步长
            current_exaggeration_factor = (exaggeration_factor 
                                        if iterations <= early_exaggeration_iter 
                                        else 1.0)
            
            # 生成邻居
            neighbors = self.gpoperator._get_neighbors(
                n_variables, current_states, max_shift_ratio, 
                current_exaggeration_factor, self.random_state
            )
            
            # 评估邻居（首次改进即停止）
            for neighbor_states in neighbors:
                apply_states_fast(neighbor_states)
                
                neighbor_fitness = new_expr_set.fitness(X, y)
                
                # 检查是否改进
                is_better = (neighbor_fitness > current_fitness 
                        if self.metric.greater_is_better 
                        else neighbor_fitness < current_fitness)
                
                if is_better:
                    current_states = neighbor_states
                    current_fitness = neighbor_fitness
                    improved = True
                    
                    # 更新全局最佳
                    best_states = [s.copy() for s in current_states]
                    best_fitness = current_fitness
                    no_improvement_count = 0
                    break  # 首次改进策略
            
            # 早停计数（跳过早期放大阶段）
            if not improved and iterations > early_exaggeration_iter:
                no_improvement_count += 1
        
        # 8. 应用最佳状态
        apply_states_fast(best_states)
        raw_fitness = best_fitness
        
        return new_expr_set, True, raw_fitness



