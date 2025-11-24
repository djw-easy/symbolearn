import jax
import optax
import warnings
import numpy as np
import jax.numpy as jnp
from functools import lru_cache
from scipy.optimize import minimize
from typing import Union, Optional, List, Tuple, Iterator


from src.tree import PreOrderIter, PostOrderIter, SymbolicNode, clone_tree, RenderTree
from src.node import Operator, Constant, Variable, NodeContent, DynamicAggregation
from src.generator import ExprGenerator, ExprSetGenerator
from src.expression import Expression, ExpressionSet
from src.utils import check_random_state
from src.fitness import _fitness_jax_map



METHODS_WITH_EPS = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'SLSQP']


def weighted_random_choice_expr(weights_dict: dict, random_state: np.random.RandomState) -> str:
    random_state = check_random_state(random_state)
    
    # 将字典转换为可哈希的元组用于缓存
    weights_tuple = tuple(sorted(weights_dict.items()))
    names, cumulative_probs = _get_cumulative_probs_expr(weights_tuple)
    
    # 轮盘赌选择
    random_value = random_state.uniform()
    idx = np.searchsorted(cumulative_probs, random_value, side='right')
    return names[idx]


@lru_cache(maxsize=128)
def _get_cumulative_probs_expr(weights_tuple):
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
    """
    优化后的遗传编程类
    
    关键改进:
    1. 移除不必要的get_node_by_index调用
    2. 直接对复制后的树进行操作
    3. 简化逻辑，提高效率
    """
    def __init__(self,
                 generator: ExprGenerator,
                 mutation_weights: dict = None,
                 constants_tolerance: float = 1e-5,
                 perturbation_factor: float = 0.129,
                 probability_negate_constant: float = 0.00743,
                 random_state: Union[int, np.random.Generator] = None):
        self.generator = generator
        self.maxsize = generator.maxsize
        self.mutation_weights = mutation_weights
        self.perturbation_factor = perturbation_factor
        self.constants_tolerance = constants_tolerance
        self.random_state = check_random_state(random_state)
        self.probability_negate_constant = probability_negate_constant

    def _contains_forbidden_patterns(self, tree: SymbolicNode) -> bool:
        """检查树是否包含禁止模式如 x-x, x/x, *0, /0"""
        for node in PreOrderIter(tree):
            if node.degree == 2:
                left, right = node.children[0], node.children[1]
                op_name = node.node_content.name
                # 检查相同子树的减法或除法
                if op_name in ['sub', 'div'] and Expression._trees_are_equal(left, right):
                    return True
                # 检查乘以0或除以0
                if op_name in ['mul', 'div']:
                    for child in [left, right]:
                        if isinstance(child.node_content, Constant) and abs(child.node_content.value) < self.constants_tolerance:
                            return True
            if node.degree == 0:
                if isinstance(node.node_content, Constant) and not self.generator.use_constants:
                    return True
                if isinstance(node.node_content, Variable) and not self.generator.use_variables:
                    return True
                if isinstance(node.node_content, DynamicAggregation) and not self.generator.use_aggregations:
                    return True
        return False

    def _get_random_operator(self, degree: Optional[int] = None, exclude: Operator = None):
        return self.generator._get_random_operator(degree, exclude)

    def _get_random_leaf(self):
        return self.generator._get_random_leaf()

    def _get_leaf_with_rules(self, node: SymbolicNode):
        return self.generator._get_leaf_with_rules(node)

    def get_subtree(self, tree: Optional[SymbolicNode], 
                    not_root: bool = False, not_leaf: bool = False):
        """
        获取一个随机子树
        
        Returns:
            node: 返回选中的节点（直接返回节点对象）
        """
        nodes = []
        for node in PreOrderIter(tree):
            if not_leaf and node.is_leaf:
                continue
            if not_root and node.is_root:
                continue
            nodes.append(node)
        
        if not nodes:
            return None
        
        selected_idx = self.random_state.randint(len(nodes))
        return nodes[selected_idx]

    @staticmethod
    def _crossover(parent_subtree: SymbolicNode, donor_subtree: SymbolicNode):
        """替换父节点的子节点"""
        if parent_subtree.is_root:
            raise ValueError('Cannot crossover the root node.')
        
        parent_node = parent_subtree.parent
        children_list = list(parent_node.children)
        replace_index = children_list.index(parent_subtree)
        children_list[replace_index] = donor_subtree
        parent_node.children = children_list

    def reproduce(self, parent: Expression) -> Expression:
        """创建副本（深拷贝树）"""
        new_expr = Expression(
            tree=clone_tree(parent.tree), 
            metric=self.generator.metric, 
            out_func=self.generator.out_func
        )
        return new_expr

    # def crossover(self, parent: Expression, donor: Expression) -> Tuple[Expression, Expression, bool]:
    #     """
    #     交叉操作：交换两个表达式的子树
        
    #     策略：
    #     1. 检查两个父代是否相同
    #     2. 检查大小约束（预估）
    #     3. 复制两棵树
    #     4. 直接在新树上选择交叉点
    #     5. 交换子树
        
    #     Args:
    #         parent: 父代表达式1
    #         donor: 父代表达式2（供体）
            
    #     Returns:
    #         (offspring1, offspring2, success): 两个后代表达式和成功标志
    #     """
    #     # 1. 提前检查：如果父代相同，直接返回失败
    #     if parent == donor:
    #         return None, None, False

    #     # 2. 粗略检查大小约束（假设交叉最坏情况）
    #     # 如果两棵树都已经很大，交叉可能失败
    #     if parent.size >= self.maxsize or donor.size >= self.maxsize:
    #         return None, None, False

    #     # 3. 创建副本
    #     offspring1 = self.reproduce(parent)
    #     offspring2 = self.reproduce(donor)

    #     # 4. 在新树上选择交叉点
    #     point1 = self.get_subtree(offspring1.tree)
    #     point2 = self.get_subtree(offspring2.tree)
        
    #     if point1 is None or point2 is None:
    #         return None, None, False

    #     # 5. 检查交叉后的大小约束
    #     new_size1 = offspring1.size - point1.size + point2.size
    #     new_size2 = offspring2.size - point2.size + point1.size
        
    #     if new_size1 > self.maxsize or new_size2 > self.maxsize:
    #         return None, None, False

    #     # 6. 执行交叉
    #     # 克隆要交换的子树（避免引用问题）
    #     point1_clone = clone_tree(point1)
    #     point2_clone = clone_tree(point2)
        
    #     # 交换子树
    #     if point1.is_root:
    #         # 如果point1是根节点，直接替换整棵树
    #         offspring1.tree = point2_clone
    #     else:
    #         # 否则在父节点中替换
    #         parent1 = point1.parent
    #         children_list1 = list(parent1.children)
    #         replace_index1 = children_list1.index(point1)
    #         children_list1[replace_index1] = point2_clone
    #         parent1.children = children_list1
        
    #     if point2.is_root:
    #         # 如果point2是根节点，直接替换整棵树
    #         offspring2.tree = point1_clone
    #     else:
    #         # 否则在父节点中替换
    #         parent2 = point2.parent
    #         children_list2 = list(parent2.children)
    #         replace_index2 = children_list2.index(point2)
    #         children_list2[replace_index2] = point1_clone
    #         parent2.children = children_list2

    #     return offspring1, offspring2, True

    def crossover(self, parent: Expression, donor: Expression) -> Tuple[Expression, Expression, bool]:
        """
        交叉操作：交换两个表达式的子树（使用路径匹配优化）
        
        策略：
        1. 在原树上选择交叉点
        2. 提前检查大小约束
        3. 通过检查后再复制树（避免不必要的复制）
        4. 通过路径匹配找到新树上的对应节点
        5. 交换子树
        
        优势：
        - 提前检查，减少不必要的复制
        - 路径匹配比重新选择更可靠
        
        Args:
            parent: 父代表达式1
            donor: 父代表达式2（供体）
            
        Returns:
            (offspring1, offspring2, success): 两个后代表达式和成功标志
        """
        # 1. 提前检查：如果父代相同，直接返回失败
        if parent == donor:
            return None, None, False

        # 2. 在原树上选择交叉点（不复制，节省开销）
        point1 = self.get_subtree(parent.tree)
        point2 = self.get_subtree(donor.tree)
        
        if point1 is None or point2 is None:
            return None, None, False

        # 3. 提前检查大小约束（避免不必要的复制）
        new_size1 = parent.size - point1.size + point2.size
        new_size2 = donor.size - point2.size + point1.size
        
        if new_size1 > self.maxsize or new_size2 > self.maxsize:
            return None, None, False

        # 4. 通过检查后再创建副本（节省不必要的复制）
        offspring1 = self.reproduce(parent)
        offspring2 = self.reproduce(donor)

        # 5. 通过路径匹配找到新树上的对应节点
        new_point1 = self._find_corresponding_node(offspring1.tree, point1)
        new_point2 = self._find_corresponding_node(offspring2.tree, point2)

        if new_point1 is None or new_point2 is None:
            return None, None, False

        # 6. 克隆要交换的子树（避免引用问题）
        # 重要：必须先克隆，否则在断开关系时会影响原树
        point1_clone = clone_tree(new_point1)
        point2_clone = clone_tree(new_point2)
        
        # 7. 执行交叉
        if new_point1.is_root:
            # point1是根节点，直接替换整棵树
            offspring1.tree = point2_clone
        else:
            # point1不是根节点，在父节点中替换
            parent1 = new_point1.parent
            children_list1 = list(parent1.children)
            replace_index1 = children_list1.index(new_point1)
            children_list1[replace_index1] = point2_clone
            parent1.children = children_list1
            # ✅ 重要：设置新子树的父节点为None，断开与新树的连接
            # 注意：new_point1已经被替换掉了，不需要手动断开
        
        if new_point2.is_root:
            # point2是根节点，直接替换整棵树
            offspring2.tree = point1_clone
        else:
            # point2不是根节点，在父节点中替换
            parent2 = new_point2.parent
            children_list2 = list(parent2.children)
            replace_index2 = children_list2.index(new_point2)
            children_list2[replace_index2] = point1_clone
            parent2.children = children_list2
            # ✅ 同样，new_point2已经被替换掉了

        return offspring1, offspring2, True

    def _find_corresponding_node(self, new_tree: SymbolicNode, 
                                old_node: SymbolicNode) -> SymbolicNode:
        """
        在新树中找到与旧节点对应的节点（通过路径匹配）
        
        注意：这个方法在当前版本中不需要使用，因为所有操作都直接在新树上进行
        保留此方法以保持向后兼容
        
        Args:
            new_tree: 新树的根节点
            old_node: 旧树中的目标节点
            
        Returns:
            new_tree中对应的节点，如果找不到则返回None
        """
        # 特殊情况：如果旧节点就是根节点
        if old_node.parent is None:
            return new_tree
        
        # 计算旧节点的路径（从根到节点的子节点索引序列）
        path = []
        current = old_node
        while current.parent is not None:
            parent = current.parent
            child_index = list(parent.children).index(current)
            path.append(child_index)
            current = parent
        
        path.reverse()  # 从根到节点的路径
        
        # 在新树中按路径查找
        current = new_tree
        for child_index in path:
            if child_index >= len(current.children):
                return None
            current = current.children[child_index]
        
        return current

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
            if not isinstance(expr.tree.node_content, Constant):
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
                weights['mutate_constant'] *= (np.log(len(self.generator.constants) + 1) + 1)
        else:
            weights['mutate_constant'] = 0.0
        
        # Adjust mutate_variable weight based on number of variables
        if len(self.generator.variables) > 5:
            n_variables = expr._count_scalar_variables()
            if n_variables == 0:
                weights['mutate_variable'] = 0.0
            else:
                weights['mutate_variable'] *= min(8, n_variables) / 8.0
                weights['mutate_variable'] *= (np.log(len(self.generator.variables) + 1) + 1)
        else:
            weights['mutate_variable'] = 0.0
        
        # Adjust mutate_aggregation weight based on number of aggregations
        if self.generator.use_aggregations:
            n_aggregations = expr._count_scalar_aggregations()
            if n_aggregations == 0:
                weights['mutate_aggregation'] = 0.0
            else:
                weights['mutate_aggregation'] *= min(5, n_aggregations) / 8.0
                weights['mutate_aggregation'] *= (np.log(len(self.generator.variables)/2.0 + 1) + 1)
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
        mutation_name = weighted_random_choice_expr(conditioned_weights, self.random_state)
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

    def add_node(self, parent: Expression):
        """
        添加节点突变：在树中添加一个新的操作符节点
        
        策略：
        1. 随机选择一个degree > 0的操作符
        2. 50%概率替换根节点，或当树大小=1时必须替换根节点
        3. 否则选择一个叶子节点，将其作为新算子的一个子节点
        4. 新算子的其他子节点用随机生成的叶子填充
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 检查大小限制
        min_new_size = self.maxsize - parent.tree.size
        if min_new_size <= 0:
            return None, False
        
        valid_degrees = np.array(self.generator.valid_degrees[1:])  # 排除degree=0
        if not any(valid_degrees <= min_new_size):
            return None, False
        
        # 2. 选择操作符
        target_degree = self.random_state.choice(valid_degrees[valid_degrees <= min_new_size])
        new_operator = self._get_random_operator(int(target_degree))
        
        # 3. 确定操作策略
        should_replace_root = (self.random_state.random() < 0.5) or (parent.tree.size == 1)
        
        # 4. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 5. 执行操作
        if should_replace_root:
            # 替换根节点：原根节点成为新算子的一个子节点
            original_root = new_expr.tree
            new_root = SymbolicNode(node_content=new_operator)
            children = [original_root]
            for _ in range(new_operator.degree - 1):
                other_child = SymbolicNode(node_content=self._get_random_leaf())
                children.append(other_child)
            self.random_state.shuffle(children)
            new_root.children = children
            new_expr.tree = new_root
        else:
            # 替换叶子节点：叶子节点成为新算子的一个子节点
            leaves = new_expr.tree.leaves
            if not leaves:
                return None, False
        
            leaf_to_replace = self.random_state.choice(leaves)
            leaf_parent = leaf_to_replace.parent
            cloned_leaf = SymbolicNode(node_content=leaf_to_replace.node_content)
        
            new_node = SymbolicNode(node_content=new_operator)
            children = [cloned_leaf]
            for _ in range(new_operator.degree - 1):
                other_child = SymbolicNode(node_content=self._get_random_leaf())
                children.append(other_child)
            self.random_state.shuffle(children)
            new_node.children = children
            
            children_list = list(leaf_parent.children)
            replacement_idx = children_list.index(leaf_to_replace)
            children_list[replacement_idx] = new_node
            leaf_parent.children = children_list

        return new_expr, True

    def insert_node(self, parent: Expression):
        """
        插入节点突变：在树的中间插入一个新的操作符节点
        
        策略：
        1. 随机选择一个degree > 0的操作符
        2. 选择一个非叶子且非根的节点作为目标
        3. 将目标节点（整个子树）作为新算子的一个子节点
        4. 新算子的其他子节点用随机生成的叶子填充
        5. 新算子替换原目标节点的位置
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 检查大小限制
        min_new_size = self.maxsize - parent.tree.size
        if min_new_size <= 0:
            return None, False
        
        valid_degrees = np.array(self.generator.valid_degrees[1:])
        if not any(valid_degrees <= min_new_size):
            return None, False
        
        # 2. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 3. 在新树上选择目标节点
        target_node = self.get_subtree(new_expr.tree, not_leaf=True, not_root=True)
        if target_node is None:
            return None, False
        
        # 4. 选择操作符
        target_degree = self.random_state.choice(valid_degrees[valid_degrees <= min_new_size])
        new_operator = self._get_random_operator(int(target_degree))
        
        target_parent = target_node.parent
        
        # 5. 执行插入
        new_node = SymbolicNode(node_content=new_operator)
        children = [target_node]  # 直接使用target_node
        for _ in range(new_operator.degree - 1):
            other_child = SymbolicNode(node_content=self._get_random_leaf())
            children.append(other_child)
        self.random_state.shuffle(children)
        new_node.children = children
        
        children_list = list(target_parent.children)
        replacement_idx = children_list.index(target_node)
        children_list[replacement_idx] = new_node
        target_parent.children = children_list

        return new_expr, True

    def delete_node(self, parent: Expression):
        """
        删除节点突变：从树中删除一个随机节点
        
        策略：
        1. 如果删除的是叶子节点：替换为另一个随机叶子
        2. 如果删除的是非叶子节点：随机提升其一个子节点
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 检查大小限制
        if parent.tree.size <= 1:
            return None, False
        
        # 2. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 3. 选择目标节点（直接在新树上选择）
        target_node = self.get_subtree(new_expr.tree)
        if target_node is None:
            return None, False
        
        # 4. 执行删除
        if target_node.is_leaf:
            # 叶子节点：替换为新的随机叶子
            new_leaf_op = self._get_leaf_with_rules(target_node)
            target_node.node_content = new_leaf_op
        else:
            # 非叶子节点：提升一个子节点
            promoted_child = self.random_state.choice(list(target_node.children))
            target_parent = target_node.parent
            if target_parent is None:
                # 如果是根节点，直接替换树
                new_expr.tree = promoted_child
            else:
                # 否则替换父节点的子节点
                children_list = list(target_parent.children)
                idx = children_list.index(target_node)
                children_list[idx] = promoted_child
                target_parent.children = children_list
        
        return new_expr, True

    def do_nothing_tree(self, parent: Expression):
        """
        空操作：返回一个新的、相同的表达式
        
        用途：在突变策略中保持部分个体不变
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        new_expr = self.reproduce(parent)
        return new_expr, True

    def mutate_constant(self, parent: Expression):
        """
        常量突变：随机选择一个常量并改变其值
        
        策略：
        1. 随机选择一个常量节点
        2. 对其值施加随机扰动（乘以一个扰动因子）
        3. 小概率取负值
        
        扰动公式：
            perturbation = 1 + perturbation_factor * random() + 0.1
            new_value = old_value * perturbation (或 * (1/perturbation))
            如果random() < probability_negate_constant: perturbation = -perturbation
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 检查是否包含常量
        if not parent._has_constants():
            return None, False
        
        # 2. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 3. 在新树上收集候选节点
        candidates = [node for node in PreOrderIter(new_expr.tree) 
                    if isinstance(node.node_content, Constant)]
        
        if not candidates:
            return None, False
        
        # 4. 随机选择一个常量节点
        target_node = self.random_state.choice(candidates)
        
        # 5. 计算新值
        perturbation = 1 + self.perturbation_factor * self.random_state.random() + 0.1
        perturbation = perturbation if self.random_state.uniform() > 0.5 else 1/perturbation
        if self.random_state.uniform() < self.probability_negate_constant:
            perturbation = -perturbation
        new_value = target_node.node_content.value * perturbation
        
        # 6. 在新树上修改
        target_node.node_content = Constant(value=new_value)
        
        return new_expr, True

    def mutate_variable(self, parent: Expression):
        """
        变量突变：随机选择一个变量并替换为另一个变量
        
        策略：
        1. 随机选择一个变量节点
        2. 使用高斯加权选择新变量（偏向选择相邻的变量索引）
        
        高斯权重公式：
            weight[i] = exp(-0.5 * distance^2)
            其中distance = |i - current_variable_index|
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 检查是否包含变量
        if not parent._has_variables():
            return None, False
        
        # 2. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 3. 在新树上收集候选节点
        candidates = [node for node in PreOrderIter(new_expr.tree) 
                    if isinstance(node.node_content, Variable)]

        if not candidates:
            return None, False

        # 4. 随机选择一个变量节点
        target_node = self.random_state.choice(candidates)
        
        # 5. 选择新变量（高斯加权）
        old_variable = target_node.node_content
        variable_idx = self.generator.variables.index(old_variable)
        variable_indices = np.delete(np.arange(len(self.generator.variables)), variable_idx)
        
        if len(variable_indices) == 0:
            return None, False
        
        distances = np.abs(variable_indices - variable_idx)
        weights = np.exp(-0.5 * distances ** 2)
        new_idx = np.random.choice(variable_indices, p=weights/weights.sum())
        new_variable = self.generator.variables[new_idx]
        
        # 6. 在新树上修改
        target_node.node_content = new_variable
        
        return new_expr, True

    def mutate_operator(self, parent: Expression):
        """
        操作符突变：随机选择一个操作符并替换为同degree的另一个操作符
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 检查是否包含算子
        if parent.size == 1:
            return None, False
        
        # 2. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 3. 在新树上选择目标节点
        target_node = self.get_subtree(new_expr.tree, not_leaf=True)
        if target_node is None:
            return None, False
        
        # 4. 选择新算子（相同degree，排除当前算子）
        target_degree = target_node.degree
        target_operator = target_node.node_content
        new_operator = self._get_random_operator(target_degree, exclude=target_operator)
        
        if not new_operator:
            return None, False
        
        # 5. 在新树上修改
        target_node.node_content = new_operator
        
        return new_expr, True

    def mutate_aggregation(self, parent: Expression):
        """
        聚合节点突变：改变聚合操作的窗口范围或操作类型
        
        突变类型：
        1. 改变操作类型（mean, max, min等）- 低概率（0.0001 * valid_op数量）
        2. 改变窗口范围 - 高概率：
        - shift_both: 整体平移窗口
        - shift_start: 移动起始位置
        - shift_end: 移动结束位置
        - expand: 扩展窗口（向左、向右或两边）
        - shrink: 收缩窗口（从左、从右或两边）
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 检查是否包含聚合节点
        if not parent._has_aggregations():
            return None, False
        
        # 2. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 3. 在新树上收集候选节点
        candidates = [node for node in PreOrderIter(new_expr.tree) 
                    if isinstance(node.node_content, DynamicAggregation)]
        
        if not candidates:
            return None, False
        
        # 4. 选择目标节点
        target_node = self.random_state.choice(candidates)
        aggregation = target_node.node_content
        
        # 5. 计算新的聚合参数
        valid_op_num = len(aggregation.valid_op) if aggregation.valid_op else 1
        prob_mutate_operator = 0.0001 * valid_op_num if valid_op_num > 1 else 0.0
        
        if self.random_state.random() < prob_mutate_operator and aggregation.valid_op:
            # 改变操作类型
            new_op_name = self.random_state.choice(aggregation.valid_op)
            new_aggregation = DynamicAggregation(
                v_start=aggregation.v_start,
                v_end=aggregation.v_end,
                op_name=new_op_name,
                n_variables=aggregation.n_variables,
                valid_op=aggregation.valid_op
            )
        else:
            # 改变窗口范围
            v_start, v_end = aggregation.v_start, aggregation.v_end
            n_variables = aggregation.n_variables
            current_window_size = v_end - v_start + 1
            
            max_change_ratio = 0.5
            max_shift = max(1, int(current_window_size * max_change_ratio))
            mutation_type = self.random_state.choice(
                ['shift_both', 'shift_start', 'shift_end', 'expand', 'shrink']
            )
            
            if mutation_type == 'shift_both':
                # 整体平移窗口
                shift = self.random_state.randint(-max_shift, max_shift + 1)
                new_start = v_start + shift
                new_end = v_end + shift
                # 边界处理
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
                # 移动起始位置
                shift = self.random_state.randint(-max_shift, max_shift + 1)
                new_start = max(0, min(v_start + shift, v_end - 1))
                new_end = v_end
                
            elif mutation_type == 'shift_end':
                # 移动结束位置
                shift = self.random_state.randint(-max_shift, max_shift + 1)
                new_end = max(v_start + 1, min(v_end + shift, n_variables - 1))
                new_start = v_start
                
            elif mutation_type == 'expand':
                # 扩展窗口
                expand_amount = self.random_state.randint(1, max_shift + 1)
                expand_direction = self.random_state.choice(['left', 'right', 'both'])
                if expand_direction == 'left' and v_start > 0:
                    new_start = max(0, v_start - expand_amount)
                    new_end = v_end
                elif expand_direction == 'right' and v_end < n_variables - 1:
                    new_start = v_start
                    new_end = min(n_variables - 1, v_end + expand_amount)
                else:  # both
                    left_expand = expand_amount // 2
                    right_expand = expand_amount - left_expand
                    new_start = max(0, v_start - left_expand)
                    new_end = min(n_variables - 1, v_end + right_expand)
            else:  # shrink
                # 收缩窗口
                max_shrink = min(max_shift, current_window_size - 2)
                if max_shrink < 1:
                    # 窗口太小，无法收缩，改为平移
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
                    else:  # both
                        left_shrink = shrink_amount // 2
                        right_shrink = shrink_amount - left_shrink
                        new_start = min(v_start + left_shrink, v_end - 1)
                        new_end = max(new_start + 1, v_end - right_shrink)
            
            # 最终边界检查
            new_start = max(0, min(new_start, n_variables - 2))
            new_end = max(new_start + 1, min(new_end, n_variables - 1))
            new_aggregation = DynamicAggregation(
                v_start=new_start, v_end=new_end,
                op_name=aggregation.op_name,
                n_variables=aggregation.n_variables,
                valid_op=aggregation.valid_op
            )
        
        # 6. 在新树上修改
        target_node.node_content = new_aggregation
        
        return new_expr, True

    def swap_operands(self, parent: Expression):
        """
        交换操作数：随机选择一个二元操作符并交换其两个子节点
        
        适用于：所有degree=2的操作符
        效果：改变表达式结构但可能不改变语义（对于交换律成立的操作）
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 检查是否有二元操作符
        if not parent._has_binary_operator():
            return None, False
        
        # 2. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 3. 在新树上收集候选节点
        candidates = [node for node in PreOrderIter(new_expr.tree) if node.degree == 2]
        
        if not candidates:
            return None, False
        
        # 4. 选择目标节点
        target_node = self.random_state.choice(candidates)
        
        # 5. 交换子节点
        swapped_children = list(target_node.children)[::-1]
        target_node.children = swapped_children
        
        return new_expr, True

    def rotate_tree(self, parent: Expression):
        r"""
        树旋转：对树进行左旋或右旋操作
        
        右旋示例（A是父节点，B是左子节点）：
        A              B
        / \            / \
        B   C    =>    D   A
        / \                / \
        D   E              E   C
        
        左旋示例（A是父节点，B是右子节点）：
        A                B
        / \              / \
        C   B      =>    A   E
            / \          / \
            D  E        C   D
        
        适用条件：
        - 右旋：左子节点不是叶子
        - 左旋：右子节点不是叶子且父节点是二元的
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        def is_valid_rotation_node(node: SymbolicNode) -> bool:
            """检查节点是否可以旋转"""
            if node.is_leaf:
                return False
            return any(not child.is_leaf for child in node.children)

        # 1. 检查是否可以执行旋转
        for node in PreOrderIter(parent.tree):
            if is_valid_rotation_node(node):
                break
        else:
            return None, False

        # 2. 创建新表达式
        new_expr = self.reproduce(parent)

        # 3. 在新树上收集可旋转的节点
        candidates = [node for node in PreOrderIter(new_expr.tree) if is_valid_rotation_node(node)]
        if not candidates:
            return None, False

        # 4. 选择目标节点
        subtree_root = self.random_state.choice(candidates)
        
        # 5. 确定可以进行的旋转方向
        can_rotate_right = not subtree_root.children[0].is_leaf
        can_rotate_left = subtree_root.degree == 2 and not subtree_root.children[1].is_leaf
        
        if not can_rotate_left and not can_rotate_right:
            return None, False

        # 6. 选择旋转方向
        if can_rotate_left and can_rotate_right:
            direction = self.random_state.choice(['left', 'right'])
        elif can_rotate_left:
            direction = 'left'
        else:
            direction = 'right'

        original_parent = subtree_root.parent
        
        # 7. 执行旋转
        if direction == 'right':
            # 右旋
            A = subtree_root
            B = A.children[0]
            C = A.children[1] if A.degree == 2 else None
            D = B.children[0]
            E = B.children[1] if B.degree == 2 else None

            new_A = SymbolicNode(node_content=A.node_content)
            new_B = SymbolicNode(node_content=B.node_content)
            new_D = clone_tree(D)
            new_E = clone_tree(E) if E is not None else None
            new_C = clone_tree(C) if C is not None else None
            
            if B.degree == 1:
                # B是一元操作符
                if A.degree == 1:
                    return None, False
                elif A.degree == 2:
                    children_A = []
                    if new_D: children_A.append(new_D)
                    if new_C: children_A.append(new_C)
                    if len(children_A) != 2:
                        return None, False
                    new_A.children = children_A
                    new_B.children = [new_A]
                else:
                    return None, False
                
            elif B.degree == 2:
                # B是二元操作符
                children_A = []
                if new_E: children_A.append(new_E)
                if new_C: children_A.append(new_C)
                
                if A.degree == 1:
                    if len(children_A) == 0:
                        return None, False
                    new_A.children = children_A[:1]
                elif A.degree == 2:
                    if len(children_A) != 2:
                        return None, False
                    new_A.children = children_A
                else:
                    return None, False
                
                new_B.children = [new_D, new_A]
            else:
                return None, False
            
            # 替换原节点
            if original_parent is None:
                new_expr.tree = new_B
            else:
                children_list = list(original_parent.children)
                idx = children_list.index(subtree_root)
                children_list[idx] = new_B
                original_parent.children = children_list
                
        else:  # left rotation
            # 左旋
            A = subtree_root
            C = A.children[0]
            B = A.children[1]
            D = B.children[0]
            E = B.children[1] if B.degree == 2 else None

            new_A = SymbolicNode(node_content=A.node_content)
            new_B = SymbolicNode(node_content=B.node_content)
            new_C = clone_tree(C)
            new_D = clone_tree(D)
            new_E = clone_tree(E) if E is not None else None
            
            if B.degree == 1:
                # B是一元操作符
                if A.degree == 1:
                    return None, False
                elif A.degree == 2:
                    new_A.children = [new_C, new_D]
                    new_B.children = [new_A]
                else:
                    return None, False
            elif B.degree == 2:
                # B是二元操作符
                if A.degree == 1:
                    new_A.children = [new_C]
                elif A.degree == 2:
                    new_A.children = [new_C, new_D]
                else:
                    return None, False
                    
                children_B = [new_A]
                if new_E: 
                    children_B.append(new_E)
                if len(children_B) != 2:
                    return None, False
                new_B.children = children_B
            else:
                return None, False
            
            # 替换原节点
            if original_parent is None:
                new_expr.tree = new_B
            else:
                children_list = list(original_parent.children)
                idx = children_list.index(subtree_root)
                children_list[idx] = new_B
                original_parent.children = children_list
        
        return new_expr, True

    def randomize_tree(self, parent: Expression):
        """
        随机化树：随机选择一个子树并替换为随机生成的新子树
        
        策略：
        1. 随机选择树中的一个节点
        2. 根据可用大小确定新子树的大小
        3. 生成一棵随机新子树
        4. 用新子树替换选中的节点
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 2. 在新树上选择目标节点
        target_node = self.get_subtree(new_expr.tree)
        if target_node is None:
            return None, False
        
        # 3. 计算可用大小
        max_target_size = self.maxsize - (new_expr.tree.size - target_node.size)
        valid_sizes = np.array(list(self.generator.size_prob.keys()))
        size_probs = np.array(list(self.generator.size_prob.values()))
        mask = valid_sizes <= max_target_size
        
        if not np.any(mask):
            return None, False
        
        # 4. 选择新子树大小
        target_size = self.random_state.choice(
            valid_sizes[mask], 
            p=size_probs[mask]/size_probs[mask].sum()
        )
        
        # 5. 生成新子树
        new_subtree = self.generator.build_tree(target_size)
        
        # 6. 替换节点
        if target_node.is_root:
            new_expr.tree = new_subtree
        else:
            target_parent = target_node.parent
            children_list = list(target_parent.children)
            replacement_idx = children_list.index(target_node)
            children_list[replacement_idx] = new_subtree
            target_parent.children = children_list
        
        return new_expr, True

    def hoist_tree(self, parent: Expression):
        """
        提升子树：选择一个子树，用其子树之一替换它
        
        策略：
        1. 随机选择一个非叶子节点作为子树
        2. 从该子树中随机选择一个非根节点作为子子树
        3. 用子子树替换子树的位置
        
        效果：减小树的大小，保留部分结构
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        # 1. 检查大小限制
        if parent.tree.size <= 1:
            return None, False
        
        # 2. 创建新表达式
        new_expr = self.reproduce(parent)
        
        # 3. 在新树上选择子树（非叶子节点）
        subtree = self.get_subtree(new_expr.tree, not_leaf=True)
        if subtree is None:
            return None, False
        
        # 4. 在子树中选择子子树（非根节点）
        subsubtree = self.get_subtree(subtree, not_root=True)
        if subsubtree is None:
            return None, False
        
        # 5. 执行提升：用子子树替换子树
        if subtree.is_root:
            # 如果子树是根节点，直接替换整棵树
            new_expr.tree = subsubtree
        else:
            # 否则在父节点中替换
            target_parent = subtree.parent
            children_list = list(target_parent.children)
            replacement_idx = children_list.index(subtree)
            children_list[replacement_idx] = subsubtree
            target_parent.children = children_list
        
        return new_expr, True

    def simplify(self, parent: Expression):
        """
        简化树：应用代数规则简化表达式
        
        简化规则包括：
        1. 常量折叠：计算常量表达式
        2. 恒等式：x + 0 = x, x * 1 = x, x / 1 = x
        3. 零元素：x * 0 = 0, 0 / x = 0
        4. 代数化简：x - x = 0, x / x = 1, -(-x) = x
        5. 分配律和结合律的应用
        
        Args:
            parent: 父代表达式
            
        Returns:
            (new_expr, success): 新表达式和成功标志
        """
        new_expr = parent.simplify(self.constants_tolerance)
        
        # 判断是否发生了简化
        simplified = not Expression._trees_are_equal(parent.tree, new_expr.tree)
        return new_expr, simplified

    def optimize_constants(
        self, parent: Expression, X: np.ndarray, y: np.ndarray,
        optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, optimizer_iterations=10
    ):
        # 检查是否有常量
        total_constants = len(parent.constant_indices)
        if not (total_constants > 0):
            return None, False, np.nan

        use_jax_optimization = (
            parent.metric.name in _fitness_jax_map and 
            parent.size >= 9 and
            (parent.size >= 15 or X.shape[0] >= 5000)
        )

        if use_jax_optimization:
            return self._optimize_constants_jax(
                parent, X, y, optimizer_algorithm, 
                optimizer_nrestarts, optimizer_iterations
            )
        else:
            return self._optimize_constants_numpy(
                parent, X, y, optimizer_algorithm, 
                optimizer_nrestarts, optimizer_iterations
            )

    def _optimize_constants_numpy(
        self, parent: Expression, X: np.ndarray, y: np.ndarray,
        optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, optimizer_iterations=10
    ):
        # 获取初始常量
        initial_constants = np.array([
            node.node_content.value for node in PostOrderIter(parent.tree) 
                if isinstance(node.node_content, Constant)
        ])

        # 定义优化目标（使用预编译的梯度）
        def objective(constants_np: np.ndarray):
            # 计算损失（使用快速的NumPy执行）
            fitness = parent.fitness(X, y, constants_np)
            loss = -fitness if parent.metric.greater_is_better else fitness
            
            return loss

        # 多次重启优化
        best_loss = float('inf')
        best_constants = initial_constants.copy()
        
        for restart in range(optimizer_nrestarts):
            # 初始点
            if restart == 0:
                x0 = initial_constants.copy()
            else:
                noise_scale = 0.1 / np.sqrt(restart)
                noise = self.random_state.uniform(-noise_scale, noise_scale, size=len(initial_constants))
                constants_scale = np.abs(initial_constants) + 1e-6
                x0 = initial_constants + noise * constants_scale
            
            # 执行优化
            if optimizer_algorithm in METHODS_WITH_EPS:
                result = minimize(
                    objective, x0, method=optimizer_algorithm, 
                    options={'maxiter': optimizer_iterations, 'eps': self.constants_tolerance}
                )
            else:
                result = minimize(
                    objective, x0,
                    method=optimizer_algorithm, 
                    options={'maxiter': optimizer_iterations}
                )
            # 更新最佳结果
            if result.fun < best_loss:
                best_loss = result.fun
                best_constants = result.x

        # 创建优化后的表达式
        optimized_expr = parent.update_constants(best_constants)
        final_fitness = -best_loss if parent.metric.greater_is_better else best_loss
        
        return optimized_expr, True, final_fitness

    def _optimize_constants_jax(
        self, parent: Expression, X: np.ndarray, y: np.ndarray,
        optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, optimizer_iterations=10
    ):
        # 获取初始常量
        initial_constants = jnp.array([
            node.node_content.value for node in PostOrderIter(parent.tree) 
                if isinstance(node.node_content, Constant)
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
            fitness = parent.fitness(X, y, constants_np)
            loss = -fitness if parent.metric.greater_is_better else fitness
            
            return float(loss), np.array(grad)
        
        # 多次重启优化
        best_loss = float('inf')
        best_constants = initial_constants.copy()
        
        for restart in range(optimizer_nrestarts):
            # 初始点
            if restart == 0:
                x0 = initial_constants.copy()
            else:
                noise_scale = 0.1 / np.sqrt(restart)
                noise = self.random_state.uniform(-noise_scale, noise_scale, size=len(initial_constants))
                constants_scale = np.abs(initial_constants) + 1e-6
                x0 = initial_constants + noise * constants_scale
            
            # 执行优化
            if optimizer_algorithm in METHODS_WITH_EPS:
                result = minimize(
                    objective_and_grad, x0,
                    method=optimizer_algorithm, jac=True,
                    options={'maxiter': optimizer_iterations, 'eps': 0.00001}
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
        if parent._count_scalar_aggregations() == 0:
            return None, False, np.nan
        
        # 1. 收集所有聚合节点的索引（一次遍历）
        agg_nodes = [node for node in PreOrderIter(parent.tree) 
                    if isinstance(node.node_content, DynamicAggregation)]
        
        # 2. 创建副本并记录初始状态
        new_expr = parent.copy()
        n_variables = X.shape[1]
        early_exaggeration_iter = min(early_exaggeration_iter, optimizer_iterations)
        
        # 初始状态：[(index, v_start, v_end, op_name, valid_op), ...]
        initial_states = []
        for node in agg_nodes:
            agg = node.node_content
            initial_states.append({
                'node': node,
                'v_start': agg.v_start,
                'v_end': agg.v_end,
                'op_name': agg.op_name,
                'valid_op': agg.valid_op
            })
        
        # 3. 计算初始适应度
        best_fitness = new_expr.fitness(X, y)
        best_states = [s.copy() for s in initial_states]
        
        # 4. 定义快速应用状态的函数
        def apply_states(states):
            """将参数状态应用到节点"""
            for state in states:
                node = state['node']
                node.node_content = DynamicAggregation(
                    v_start=state['v_start'],
                    v_end=state['v_end'],
                    op_name=state['op_name'],
                    n_variables=n_variables,
                    valid_op=state['valid_op']
                )
            return True
        
        # 6. 贪心爬山算法
        current_states = initial_states
        current_fitness = best_fitness
        
        iterations = 0
        no_improvement_count = 0  # 记录连续无改进的迭代次数
        
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
                apply_states(neighbor_states)
                
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
        apply_states(best_states)
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



def weighted_random_choice_expr_set(weights_dict: dict, random_state: np.random.RandomState) -> str:
    random_state = check_random_state(random_state)
    
    # 将字典转换为可哈希的元组用于缓存
    weights_tuple = tuple(sorted(weights_dict.items()))
    names, cumulative_probs = _get_cumulative_probs_expr_set(weights_tuple)
    
    # 轮盘赌选择
    random_value = random_state.uniform()
    idx = np.searchsorted(cumulative_probs, random_value, side='right')
    return names[idx]


@lru_cache(maxsize=128)
def _get_cumulative_probs_expr_set(weights_tuple):
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



class ExpressionSetGP:
    def __init__(self,
                 generator: ExprSetGenerator,
                 gpoperator: ExpressionGP,
                 set_mutation_weights: dict = None,
                 set_crossover_method: str = 'single_point',
                 set_crossover_probability: float = 0.0369,
                 random_state: Union[int, np.random.Generator] = None):
        self.generator = generator
        self.gpoperator = gpoperator
        self.maxsize = generator.maxsize
        self.set_mutation_weights = set_mutation_weights
        self.set_crossover_method = set_crossover_method
        self.random_state = check_random_state(random_state)
        self.set_crossover_probability = set_crossover_probability

    def reproduce(self, expressions: List[Optional['Expression']]) -> 'ExpressionSet':
        """Returns a list of nodes in the tree in pre-order."""
        return ExpressionSet(
            expressions=expressions,
            metric=self.generator.metric,
            out_func=self.generator.out_func
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
            if exprset.order <= self.generator.minorder:
                weights['delete_expr'] = 0.0
        
        if exprset.order < 2:
            weights['swap_exprs'] = 0.0
        
        return weights

    def mutation(self, parent: ExpressionSet) -> Tuple[Optional['ExpressionSet'], bool]:
        """Perform a mutation operation on the ExpressionSet. 
        
        This method acts as a dispatcher, selecting one of several mutation
        strategies based on pre-defined weights and executing it.
        """
        conditioned_weights = self._condition_mutation_weights(parent)
        mutation_name = weighted_random_choice_expr_set(conditioned_weights, self.random_state)
        
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
        
        mutated_expr, mutation_succeeded, _ = self.gpoperator.mutation(parent_expr)
        
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
        mutated_expr, mutation_succeeded = self.gpoperator.mutate_constant(parent_expr)

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
        new_expr_set = self.reproduce(new_exprs)
        
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
        new_expr_set = self.reproduce(
            [expr.copy() if expr is not None else None for expr in parent.expressions]
        )
        return new_expr_set, True

    def add_expr(self, parent: ExpressionSet):
        """Add a new expression to the expression set (in a None slot)."""
        # 1. 找到一个空位
        empty_indices = [i for i, expr in enumerate(parent.expressions) if expr is None]
        if not empty_indices:
            return None, False

        point_to_add = self.random_state.choice(empty_indices)
        
        # 2. 生成新表达式
        new_expr = self.generator.generate_random_expr()
        
        # 3. 构建新列表
        new_exprs = (
            parent.expressions[:point_to_add] + 
            [new_expr] + 
            parent.expressions[point_to_add+1:]
        )
        
        # 4. 创建副本
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True

    def optimize_constants(
        self, parent: Expression, X: np.ndarray, y: np.ndarray,
        optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, optimizer_iterations=10
    ):
        # 检查是否有常量
        const_info = parent._build_constant_info()
        if const_info['total_constants'] == 0:
            return None, False, np.nan

        use_jax_optimization = (
            parent.metric.name in _fitness_jax_map
            and parent.size >= parent.order * 5
            and (parent.size >= parent.order * 11 or X.shape[0] >= 3000)
        )

        if use_jax_optimization:
            return self._optimize_constants_jax(
                parent, X, y, optimizer_algorithm, 
                optimizer_nrestarts, optimizer_iterations
            )
        else:
            return self._optimize_constants_numpy(
                parent, X, y, optimizer_algorithm, 
                optimizer_nrestarts, optimizer_iterations
            )

    def _optimize_constants_numpy(self, parent: ExpressionSet, X: np.ndarray, y: np.ndarray, 
                                  optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, 
                                  optimizer_iterations=10) -> Tuple[Optional['ExpressionSet'], bool]:
        # 1. 检查是否有常量
        const_info = parent._build_constant_info()
        if const_info['total_constants'] == 0:
            return None, False, np.nan
        
        # 2. 提取初始常量
        initial_constants = np.array([
            node.node_content.value 
            for expr_idx, node in const_info['flat_nodes']
        ])
        
        # 3. 定义优化目标
        def objective(constants):
            # 计算损失（使用NumPy快速执行）
            fitness = parent.fitness(X, y, constants=constants)
            loss = -fitness if parent.metric.greater_is_better else fitness
            
            return loss
        
        # 4. 多次重启优化
        best_loss = float('inf')
        best_constants = initial_constants.copy()
        
        for restart in range(optimizer_nrestarts):
            # 初始点
            if restart == 0:
                x0 = initial_constants.copy()
            else:
                noise_scale = 0.1 / np.sqrt(restart)
                noise = self.random_state.uniform(-noise_scale, noise_scale, size=len(initial_constants))
                constants_scale = np.abs(initial_constants) + 1e-6
                x0 = initial_constants + noise * constants_scale
            
            # 执行优化
            if optimizer_algorithm in METHODS_WITH_EPS:
                result = minimize(
                    objective, x0, method=optimizer_algorithm, 
                    options={'maxiter': optimizer_iterations, 
                             'eps': self.gpoperator.constants_tolerance
                    }
                )
            else:
                result = minimize(
                    objective, x0,
                    method=optimizer_algorithm, 
                    options={'maxiter': optimizer_iterations}
                )
            # 更新最佳结果
            if result.fun < best_loss:
                best_loss = result.fun
                best_constants = result.x
        
        # 5. 创建优化后的表达式
        optimized_expr = parent.update_constants(best_constants)
        final_fitness = -best_loss if parent.metric.greater_is_better else best_loss
        
        return optimized_expr, True, final_fitness

    def _optimize_constants_jax(self, parent: ExpressionSet, X: np.ndarray, y: np.ndarray, 
                                optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, 
                                optimizer_iterations=10) -> Tuple[Optional['ExpressionSet'], bool]:
        # 1. 检查是否有常量
        const_info = parent._build_constant_info()
        if const_info['total_constants'] == 0:
            return None, False, np.nan
        
        # 2. 准备JAX数组（只转换一次）
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        # 3. 预编译梯度函数
        grad_fn = parent._get_gradient_function()
        
        # 4. 定义优化目标
        def objective_and_grad(constants):
            constants_jax = jnp.array(constants)
            
            # 计算梯度（使用预编译的函数）
            grad = grad_fn(constants_jax, X_jax, y_jax)
            
            # 计算损失（使用NumPy快速执行）
            fitness = parent.fitness(X, y, constants=constants)
            loss = -fitness if parent.metric.greater_is_better else fitness
            
            return float(loss), np.array(grad)
        
        # 5. 提取初始常量
        initial_constants = np.array([
            node.node_content.value 
            for expr_idx, node in const_info['flat_nodes']
        ])
        
        # 6. 执行优化
        best_loss = float('inf')
        best_constants = initial_constants.copy()
        
        for restart in range(optimizer_nrestarts):
            # 第一次使用原始值，后续添加噪声
            if restart == 0:
                x0 = initial_constants.copy()
            else:
                # 噪声强度递减（避免后期扰动过大）
                noise_scale = 0.1 / np.sqrt(restart)
                # restart=1: 10%, restart=2: 7.1%, restart=3: 5.8%
                noise = self.random_state.uniform(-noise_scale, noise_scale, size=len(initial_constants))
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
        
        # 7. 返回优化后的表达式集合
        optimized_expr_set = parent.update_constants(best_constants)
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
        # 1. 检查是否满足条件
        if sum(expr._count_scalar_aggregations() for expr in parent.expressions if expr is not None) <= 0:
            return None, False, np.nan
        
        # 2. 复制原始表达式集合
        new_expr_set = self.reproduce([expr.copy() if expr is not None else None for expr in parent.expressions])
        
        # 3. 一次性收集所有聚合节点（关键优化）
        agg_nodes = []
        for expr in new_expr_set.expressions:
            if expr is not None:
                agg_nodes.extend([node for node in PreOrderIter(expr.tree) 
                                if isinstance(node.node_content, DynamicAggregation)])
        
        n_variables = X.shape[1]
        early_exaggeration_iter = min(early_exaggeration_iter, optimizer_iterations)
        
        # 3. 记录初始状态（使用列表推导式）
        initial_states = []
        for node in agg_nodes:
            agg = node.node_content
            initial_states.append({
                'node': node,            # 节点对象
                'v_start': agg.v_start,  # 窗口起始索引
                'v_end': agg.v_end,      # 窗口结束索引
                'op_name': agg.op_name,  # 聚合操作名称
                'valid_op': agg.valid_op # 有效操作标志
            })
        
        # 4. 计算初始适应度
        best_fitness = new_expr_set.fitness(X, y)
        best_states = [s.copy() for s in initial_states]
        
        # 5. 定义快速状态应用函数（闭包优化）
        def apply_states(states):
            """快速应用聚合参数（原地修改）"""
            for state in states:
                node = state['node']
                node.node_content = DynamicAggregation(
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
                apply_states(neighbor_states)
                
                neighbor_fitness = new_expr_set.fitness(X, y)
                
                # 检查是否改进
                is_better = (neighbor_fitness > current_fitness 
                        if parent.metric.greater_is_better 
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
        apply_states(best_states)
        raw_fitness = best_fitness
        
        return new_expr_set, True, raw_fitness



