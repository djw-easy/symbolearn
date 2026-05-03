from typing import Union, Optional, List, Dict, Tuple
import numpy as np


from src.node import NodeContent
from src.utils import check_random_state


class SymbolicNode:
    """
    A tree node representing a component of a symbolic expression.

    SymbolicNode is a lightweight, high-performance tree node implementation
    that does not depend on external tree libraries. It provides explicit
    parent-child relationships with optional size caching for efficiency.

    Each SymbolicNode contains:
    - node_content: The actual data (Operator, Variable, Constant, or DynamicAggregation)
    - degree: Number of children (0 for leaves)
    - name: Human-readable name derived from node_content
    - parent: Reference to the parent node (None for root)
    - children: Tuple of child SymbolicNodes

    The node uses __slots__ for memory efficiency and caches the subtree size
    to avoid repeated computation.

    Parameters
    ----------
    node_content : NodeContent, optional
        The content of this node (Operator, Variable, Constant, or DynamicAggregation).
        If None, the node is created as a placeholder with the specified degree.
    parent : SymbolicNode, optional
        The parent node. If provided, this node is appended to parent's children.
    children : SymbolicNode, list, or tuple, optional
        The child nodes. Must match the degree of node_content.
    degree : int, optional
        Required when node_content is None. Specifies the number of children.

    Attributes
    ----------
    node_content : NodeContent
        The content stored in this node.
    name : str
        Human-readable name (e.g., 'add', 'x0', '2.5').
    degree : int
        Number of children (arity of the operator). 0 for leaf nodes.
    parent : SymbolicNode or None
        Reference to parent node.
    children : tuple
        Tuple of child SymbolicNodes (empty tuple for leaves).

    Properties
    ----------
    is_leaf : bool
        True if this node has no children.
    is_root : bool
        True if this node has no parent.
    size : int
        Number of nodes in the subtree rooted at this node (cached).
    leaves : list
        List of all leaf descendant nodes.
    root : SymbolicNode
        The root node of this tree.
    depth : int
        Depth of this node from the root (root has depth 0).

    Methods
    -------
    __call__(X)
        Execute the node (and its subtree) on input data X.
        For operators: applies the operator to children's results.
        For leaves: delegates to node_content(X).

    Examples
    --------
    >>> from src.node import add2, Variable, Constant
    >>> # Build: add(x0, 2.5)
    >>> leaf1 = SymbolicNode(node_content=Variable(0))
    >>> leaf2 = SymbolicNode(node_content=Constant(2.5))
    >>> root = SymbolicNode(node_content=add2)
    >>> root.children = [leaf1, leaf2]
    >>> # Evaluate
    >>> import numpy as np
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> result = root(X)  # Returns array([3.5, 4.5, 5.5])

    Notes
    -----
    The node enforces that children match the degree of the operator:
    - degree=0 (leaf): no children allowed
    - degree=1 (unary): exactly 1 child (e.g., sin, neg)
    - degree=2 (binary): exactly 2 children (e.g., add, mul)

    Size caching is invalidated automatically when children are modified.
    The cache is also invalidated for all ancestors when any node's
    children are changed.

    See Also
    --------
    Expression : Wraps a SymbolicNode tree with additional functionality.
    NodeContent : Base class for node content types.
    """
    __slots__ = ('_node_content', 'name', 'degree', '_parent', '_children', '_cached_size', '_temp_op_name')
    
    def __init__(self,
                 node_content: Optional[NodeContent] = None,
                 parent: Optional['SymbolicNode'] = None,
                 children: Union['SymbolicNode', List['SymbolicNode'], None] = None,
                 degree: Optional[int] = None):
        
        self._node_content = node_content
        self._parent = None
        self._children = []
        self._cached_size = None
        self._temp_op_name = None  # For constraint-aware generation
        
        if node_content is not None:
            self.name = node_content.name
            self.degree = node_content.degree
        else:
            self.degree = degree
            self.name = f"degree={degree}"
        
        # 设置父节点
        if parent is not None:
            self.parent = parent
        
        # 设置子节点
        if children is not None:
            self.children = children
    
    @property
    def node_content(self):
        return self._node_content
    
    @node_content.setter
    def node_content(self, node_content):
        self._node_content = node_content
        self.name = node_content.name
        self.degree = node_content.degree
    
    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self, parent):
        """设置父节点"""
        if self._parent is not None and parent is not None:
            # 从旧父节点中移除
            if self in self._parent._children:
                self._parent._children.remove(self)
                self._parent._invalidate_size_cache()
        
        self._parent = parent
        
        if parent is not None:
            # 添加到新父节点
            if self not in parent._children:
                parent._children.append(self)
                parent._invalidate_size_cache()
    
    @property
    def children(self):
        """返回子节点的元组（只读）"""
        return tuple(self._children)
    
    @children.setter
    def children(self, children):
        """设置子节点"""
        if self.degree == 0 and children:
            raise ValueError(f"Node with degree 0 cannot have children. Operator: {self.name}")
        
        if not isinstance(children, (list, tuple)):
            children = list(children)
        
        if self.degree > 0 and len(children) != self.degree:
            raise ValueError(
                f"Node `{self.name}` with degree {self.degree} must have exactly "
                f"{self.degree} children, but {len(children)} is given."
            )
        
        # 清除旧子节点的父引用
        for child in self._children:
            child._parent = None
        
        # 设置新子节点
        self._children = list(children)
        
        # 设置新子节点的父引用
        for child in self._children:
            child._parent = self
        
        # 使大小缓存失效
        self._invalidate_size_cache()
    
    @property
    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return len(self._children) == 0
    
    @property
    def is_root(self) -> bool:
        """判断是否为根节点"""
        return self._parent is None
    
    @property
    def size(self) -> int:
        """获取以该节点为根的子树大小（带缓存）"""
        if self._cached_size is None:
            self._cached_size = self._compute_size()
        return self._cached_size
    
    def _compute_size(self) -> int:
        """计算树的大小（迭代版本，避免递归栈溢出）"""
        count = 0
        stack = [self]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node._children)
        return count
    
    def _invalidate_size_cache(self):
        """使当前节点及所有祖先的大小缓存失效"""
        node = self
        while node is not None:
            node._cached_size = None
            node = node._parent
    
    @property
    def leaves(self):
        """获取所有叶子节点"""
        result = []
        stack = [self]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                result.append(node)
            else:
                stack.extend(reversed(node._children))
        return result
    
    @property
    def root(self):
        """获取根节点"""
        node = self
        while node._parent is not None:
            node = node._parent
        return node
    
    @property
    def depth(self) -> int:
        """获取节点深度"""
        depth = 0
        node = self._parent
        while node is not None:
            depth += 1
            node = node._parent
        return depth
    
    @property
    def height(self) -> int:
        """获取以该节点为根的子树高度"""
        if self.is_leaf:
            return 0
        return 1 + max(child.height for child in self._children)
    
    def __call__(self, X):
        """执行节点计算"""
        if self.degree == 0:
            return self._node_content(X)
        return self._node_content(*[child(X) for child in self._children])
    
    def __repr__(self):
        return f"SymbolicNode(name='{self.name}', degree={self.degree}, size={self.size})"


def clone_tree(node: SymbolicNode) -> SymbolicNode:
    """递归地克隆一个节点及其所有后代（带大小缓存优化）"""
    if not node:
        return None
    
    new_node = SymbolicNode(node_content=node.node_content)
    
    if node._children:
        new_children = [clone_tree(child) for child in node._children]
        new_node.children = new_children
    
    # 如果原节点有缓存，直接复制
    if node._cached_size is not None:
        new_node._cached_size = node._cached_size
    
    return new_node


def RenderTree(node: SymbolicNode):
    """
    兼容 anytree.RenderTree 的迭代器
    返回 (prefix, fill, node) 元组用于树的遍历
    """
    def _render(node, prefix="", is_last=True):
        """递归生成树的渲染"""
        connector = "└── " if is_last else "├── "
        yield (prefix + connector, "", node)
        
        if node._children:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node._children):
                is_last_child = (i == len(node._children) - 1)
                yield from _render(child, prefix + extension, is_last_child)
    
    # 根节点特殊处理
    yield ("", "", node)
    if node._children:
        for i, child in enumerate(node._children):
            is_last_child = (i == len(node._children) - 1)
            yield from _render(child, "", is_last_child)


def PreOrderIter(node: SymbolicNode):
    """
    兼容 anytree.PreOrderIter 的前序遍历迭代器
    """
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        # 反向添加子节点以保持正确的前序顺序
        stack.extend(reversed(current._children))


def PostOrderIter(node: SymbolicNode):
    """后序遍历迭代器（得到后缀表达式）"""
    def _postorder(node):
        for child in node._children:
            yield from _postorder(child)
        yield node
    
    return _postorder(node)


def get_integer_partitions(n: int, k: int):
    """
    一个生成器，用于将整数 n 划分为 k 个正整数部分。
    """
    if k == 1:
        if n > 0:
            yield (n,)
        return

    for i in range(1, n - (k - 1) + 1):
        for partition in get_integer_partitions(n - i, k - 1):
            yield (i,) + partition


def count_trees(
    size: int,
    degrees: List[int],
    memo: Dict[int, int]=None
) -> int:
    """
    计算大小恰好为 'size' 的树有多少种可能的组合。
    使用记忆化来存储和重用结果。
    """
    if memo is None:
        memo = {}
    
    if size in memo:
        return memo[size]
    
    if size <= 0:
        return 0

    # 基本情况: 大小为 1 的树
    if size == 1:
        # 只有当 degree=0 存在时，才能构成大小为1的树
        count = 1 if 0 in degrees else 0
        memo[1] = count
        return count

    # 递归步骤:
    total_count = 0
    nodes_for_children = size - 1

    for degree in degrees:
        if degree == 0:
            continue
        
        if nodes_for_children < degree:
            continue

        for partition in get_integer_partitions(nodes_for_children, degree):
            # 对于这个划分，计算有多少种组合
            # 这是各子树组合数的乘积
            combinations_for_partition = 1
            possible = True
            for child_size in partition:
                child_count = count_trees(child_size, degrees, memo)
                if child_count == 0:
                    possible = False
                    break
                combinations_for_partition *= child_count
            
            if possible:
                total_count += combinations_for_partition
    
    memo[size] = total_count
    return total_count


def generate_random_tree(
    size: int,
    degrees: List[int],
    count_memo: Optional[Dict[int, int]] = None, 
    random_state: Optional[np.random.RandomState] = None
) -> Optional[SymbolicNode]:
    """
    (优化版) 给定确切的大小和 RandomState，随机返回一个满足条件的树。
    通过一系列加权随机选择来避免遍历所有可能性，并保证可复现性。
    """
    if count_memo is None:
        count_memo = {}
    random_state = check_random_state(random_state)
    
    # 前置检查
    if count_trees(size, degrees, count_memo) == 0:
        return None

    # 基本情况
    if size == 1:
        return SymbolicNode(degree=0)

    # 递归步骤
    nodes_for_children = size - 1

    # === 1. 加权随机选择 degree ===
    degree_choices = []
    for degree in sorted(degrees):
        if degree == 0 or nodes_for_children < degree:
            continue
        
        count_for_this_degree = 0
        for partition in get_integer_partitions(nodes_for_children, degree):
            combinations_for_partition = 1
            possible = True
            for child_size in partition:
                child_count = count_trees(child_size, degrees, count_memo)
                if child_count == 0:
                    possible = False
                    break
                combinations_for_partition *= child_count
            if possible:
                count_for_this_degree += combinations_for_partition
        
        if count_for_this_degree > 0:
            degree_choices.append({'degree': degree, 'weight': count_for_this_degree})

    # 从可能的 degree 中选择一个
    population_degrees = [c['degree'] for c in degree_choices]
    weights_degrees = np.array([c['weight'] for c in degree_choices], dtype=np.int64)
    # 将权重转换为概率
    probabilities_degrees = weights_degrees / np.sum(weights_degrees)
    
    # 使用 random_state.choice
    chosen_degree = random_state.choice(population_degrees, p=probabilities_degrees)

    # === 2. 针对已选定的 degree，加权随机选择 Partition ===
    partition_choices = []
    for partition in get_integer_partitions(nodes_for_children, chosen_degree):
        combinations_for_partition = 1
        possible = True
        for child_size in partition:
            child_count = count_trees(child_size, degrees, count_memo)
            if child_count == 0:
                possible = False
                break
            combinations_for_partition *= child_count
        
        if possible:
            partition_choices.append({'partition': partition, 'weight': combinations_for_partition})

    # 为了让 np.random.choice 处理复杂对象（元组），我们选择索引
    indices = np.arange(len(partition_choices))
    weights_partitions = np.array([p['weight'] for p in partition_choices], dtype=np.int64)
    # 将权重转换为概率
    probabilities_partitions = weights_partitions / np.sum(weights_partitions)

    # 使用 random_state.choice 选择一个索引
    chosen_index = random_state.choice(indices, p=probabilities_partitions)
    chosen_partition = partition_choices[chosen_index]['partition']
    
    # === 3. 为选定的 Partition 递归构建子树 ===
    root = SymbolicNode(degree=int(chosen_degree))
    children = []
    for child_size in chosen_partition:
        # 确保将 random_state 传入递归调用
        child_tree = generate_random_tree(
            child_size, degrees, count_memo, random_state
        )
        children.append(child_tree)
    
    root.children = children
    return root








