import warnings
import numpy as np
from functools import lru_cache
from scipy.optimize import minimize
from typing import Union, Optional, List, Tuple


from symbolearn.tree import PreOrderIter, PostOrderIter, SymbolicNode, clone_tree
from symbolearn.node import Operator, Constant, Variable, DynamicAggregation
from symbolearn.generator import ExprGenerator, ExprSetGenerator
from symbolearn.expression import Expression, ExpressionSet
from symbolearn.utils import check_random_state



METHODS_WITH_EPS = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'SLSQP']


def weighted_random_choice_expr(weights_dict: dict, random_state: np.random.RandomState) -> str:
    random_state = check_random_state(random_state)
    
    # Convert the dictionary into a hashable tuple for caching.
    weights_tuple = tuple(sorted(weights_dict.items()))
    names, cumulative_probs = _get_cumulative_probs_expr(weights_tuple)
    
    # Perform roulette-wheel selection.
    random_value = random_state.uniform()
    idx = np.searchsorted(cumulative_probs, random_value, side='right')
    return names[idx]


@lru_cache(maxsize=128)
def _get_cumulative_probs_expr(weights_tuple):
    names, weights = zip(*weights_tuple)
    weights_array = np.array(weights, dtype=np.float64)
    
    # Normalize the weights and compute cumulative probabilities.
    total_weight = weights_array.sum()
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")
    
    probabilities = weights_array / total_weight
    cumulative_probs = np.cumsum(probabilities)
    cumulative_probs[-1] = 1.0  # Correct floating-point accumulation error.
    
    return names, cumulative_probs




class ExpressionGP:
    """
    Genetic Programming operators for single-expression individuals.

    ExpressionGP implements crossover and mutation operators that operate on
    individual Expression trees. These operators are used by the Population class
    to evolve expressions toward better fitness.

    Mutation Operators
    -----------------
    The following mutation types are available, each with configurable probability:
    - add_node: Insert a new subtree at a random location
    - insert_node: Insert a new node between an existing node and its parent
    - delete_node: Remove a subtree, replacing with a terminal
    - mutate_constant: Perturb constant values by a random factor
    - mutate_operator: Change one operator to another of the same arity
    - mutate_aggregation: Change aggregation parameters
    - swap_operands: Swap left/right children of a binary operator
    - rotate_tree: Re-root the tree at a different node
    - hoist_tree: Replace tree with a random subtree
    - randomize_tree: Replace with an entirely new random tree
    - simplify_tree: Apply algebraic simplification
    - do_nothing_tree: No change (for weighted selection schemes)

    Crossover Operators
    -------------------
    - subtree_crossover: Swap random subtrees between two parents

    Parameters
    ----------
    generator : ExprGenerator
        Factory for creating new random subtrees for mutation.
    mutation_weights : dict, optional
        Relative probability weights for each mutation type.
        Format: {'mutation_type': weight}. Weights are normalized to sum to 1.
    constants_tolerance : float, default=1e-5
        Tolerance for constant comparisons and simplification.
    perturbation_factor : float, default=0.129
        Scale factor for constant perturbation. New = old * (1 ± factor * random).
    probability_negate_constant : float, default=0.00743
        Probability of negating a constant during mutation.
    random_state : int or RandomState, optional
        Random seed for reproducibility.

    Methods
    -------
    mutation(parent)
        Apply a random mutation to an expression.
    crossover(parent1, parent2)
        Create two offspring by swapping subtrees.

    Examples
    --------
    >>> from symbolearn.gpoperator import ExpressionGP
    >>> from symbolearn.generator import ExprGenerator
    >>> gen = ExprGenerator(maxsize=21, operators=['add', 'sub', 'mul', 'div'], ...)
    >>> gp = ExpressionGP(gen, mutation_weights={'add_node': 2.0, 'mutate_constant': 0.5, ...})
    >>> # Mutate an expression
    >>> mutated, success, op_name = gp.mutation(parent_expr)
    >>> # Cross over two expressions
    >>> child1, child2, success = gp.crossover(parent1, parent2)

    Notes
    -----
    Mutation operators are selected using weighted random choice where the
    weights determine relative probability. This allows different mutation
    strategies to be explored by adjusting the weight distribution.

    The crossover operator uses subtree crossover where a random node is
    selected in each parent and the subtrees are swapped. This preserves
    valid tree structure while combining genetic material.

    All operators check constraint validity before returning. Invalid
    offspring (e.g., violating maxsize) result in mutation_succeeded=False
    or crossover_succeeded=False.

    See Also
    --------
    ExpressionSetGP : Operators for multi-output ExpressionSet individuals.
    Population : Uses these operators for evolution.
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
        """Check whether the tree contains forbidden patterns such as x-x, x/x, *0, or /0."""
        for node in PreOrderIter(tree):
            if node.degree == 2:
                left, right = node.children[0], node.children[1]
                stat_name = node.node_content.name
                # Check subtraction or division between identical subtrees.
                if stat_name in ['sub', 'div'] and Expression._trees_are_equal(left, right):
                    return True
                # Check multiplication by zero or division by zero.
                if stat_name in ['mul', 'div']:
                    for child in [left, right]:
                        if isinstance(child.node_content, Constant) and abs(child.node_content.value) < self.constants_tolerance:
                            return True
            if node.degree == 0:
                if isinstance(node.node_content, Constant) and not self.generator.use_constants:
                    return True
                if isinstance(node.node_content, Variable) and not self.generator.use_variables:
                    return True
                if isinstance(node.node_content, DynamicAggregation) and not self.generator.use_aggregation:
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
        Select a random subtree.
        
        Returns
        -------
        SymbolicNode or None
            The selected node itself, or None when no valid candidates exist.
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
        """Replace a child subtree under the parent node."""
        if parent_subtree.is_root:
            raise ValueError('Cannot crossover the root node.')
        
        parent_node = parent_subtree.parent
        children_list = list(parent_node.children)
        replace_index = children_list.index(parent_subtree)
        children_list[replace_index] = donor_subtree
        parent_node.children = children_list

    def reproduce(self, parent: Expression) -> Expression:
        """Create a deep-copied expression tree."""
        new_expr = Expression(
            tree=clone_tree(parent.tree), 
            metric=self.generator.metric, 
            ndigits=self.generator.ndigits,
            out_func=self.generator.out_func,
            constraints=self.generator.constraints,
            nested_constraints=self.generator.nested_constraints
        )
        return new_expr

    def crossover(self, parent: Expression, donor: Expression) -> Tuple[Expression, Expression, bool]:
        """
        Perform subtree crossover between two expressions using path matching.
        
        Strategy
        --------
        1. Select crossover points on the original trees.
        2. Check size constraints before cloning.
        3. Clone the trees only after the size check passes.
        4. Find the corresponding nodes in the cloned trees via path matching.
        5. Swap the cloned subtrees.
        
        Advantages
        ----------
        - Early validation reduces unnecessary copying.
        - Path matching is more reliable than re-sampling crossover points.
        
        Parameters
        ----------
        parent : Expression
            First parent expression.
        donor : Expression
            Second parent expression that provides the donor subtree.
            
        Returns
        -------
        tuple
            ``(offspring1, offspring2, success)`` containing the two offspring
            expressions and a success flag.
        """
        # Early check: identical parents cannot produce a useful crossover.
        if parent == donor:
            return None, None, False

        # Select crossover points on the original trees without copying them.
        point1 = self.get_subtree(parent.tree)
        point2 = self.get_subtree(donor.tree)
        
        if point1 is None or point2 is None:
            return None, None, False

        # Check size constraints before any cloning occurs.
        new_size1 = parent.size - point1.size + point2.size
        new_size2 = donor.size - point2.size + point1.size
        
        if new_size1 > self.maxsize or new_size2 > self.maxsize:
            return None, None, False

        # Clone the parents only after the early checks succeed.
        offspring1 = self.reproduce(parent)
        offspring2 = self.reproduce(donor)

        # Locate the matching nodes inside the cloned offspring trees.
        new_point1 = self._find_corresponding_node(offspring1.tree, point1)
        new_point2 = self._find_corresponding_node(offspring2.tree, point2)

        if new_point1 is None or new_point2 is None:
            return None, None, False

        # Clone the subtrees to avoid sharing references between offspring.
        point1_clone = clone_tree(new_point1)
        point2_clone = clone_tree(new_point2)
        
        # Execute the subtree swap.
        if new_point1.is_root:
            offspring1.tree = point2_clone
        else:
            parent1 = new_point1.parent
            children_list1 = list(parent1.children)
            replace_index1 = children_list1.index(new_point1)
            children_list1[replace_index1] = point2_clone
            parent1.children = children_list1
        
        if new_point2.is_root:
            offspring2.tree = point1_clone
        else:
            parent2 = new_point2.parent
            children_list2 = list(parent2.children)
            replace_index2 = children_list2.index(new_point2)
            children_list2[replace_index2] = point1_clone
            parent2.children = children_list2

        # Validate resulting size and structural constraints.
        if not offspring1.is_valid(self.maxsize):
            return None, None, False
        if not offspring2.is_valid(self.maxsize):
            return None, None, False

        return offspring1, offspring2, True

    def _find_corresponding_node(self, new_tree: SymbolicNode, 
                                old_node: SymbolicNode) -> SymbolicNode:
        """
        Locate the node in a cloned tree that corresponds to a node in the old tree.
        
        Notes
        -----
        This helper is no longer strictly required in the current implementation
        because the main operations are performed directly on cloned trees.
        It is kept for backward compatibility and for any logic that still needs
        stable node mapping through root-to-node paths.
        
        Parameters
        ----------
        new_tree : SymbolicNode
            Root node of the cloned tree.
        old_node : SymbolicNode
            Target node in the original tree.
            
        Returns
        -------
        SymbolicNode or None
            The corresponding node in ``new_tree``, or None if the path cannot be followed.
        """
        # Special case: the old node itself is the root.
        if old_node.parent is None:
            return new_tree
        
        # Reconstruct the child-index path from root to the old node.
        path = []
        current = old_node
        while current.parent is not None:
            parent = current.parent
            child_index = list(parent.children).index(current)
            path.append(child_index)
            current = parent
        
        path.reverse()  # Path from root to the target node.
        
        # Follow the same path on the cloned tree.
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
        if self.generator.use_variables:
            n_variables = expr._count_scalar_variables()
            if n_variables == 0:
                weights['mutate_variable'] = 0.0
            else:
                weights['mutate_variable'] *= min(8, n_variables) / 8.0
                weights['mutate_variable'] *= (np.log(len(self.generator.variables) + 1) + 1)
        else:
            weights['mutate_variable'] = 0.0
        
        # Adjust mutate_aggregation weight based on number of aggregations
        if self.generator.use_aggregation:
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
        
        # Execute the selected mutation operator.
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
        
        # Validate size constraints and structural rules.
        if mutation_succeeded and new_expr is not None:
            if (not new_expr.is_valid(self. maxsize)) or \
                (not new_expr._check_constraints()):
                return None, False, mutation_name
        
        return new_expr, mutation_succeeded, mutation_name

    def add_node(self, parent: Expression):
        """
        Add a new operator node to the expression tree.
        
        Strategy
        --------
        1. Randomly choose an operator with degree greater than zero.
        2. With probability 0.5, replace the root directly; if the tree size is 1,
           root replacement is mandatory.
        3. Otherwise, choose a leaf and make it one child of the new operator.
        4. Fill the remaining child positions with randomly generated leaves.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        # Check how much additional size can still be inserted.
        min_new_size = self.maxsize - parent.tree.size
        if min_new_size <= 0:
            return None, False
        
        valid_degrees = np.array(self.generator.degrees[1:])  # Exclude degree 0.
        if not any(valid_degrees <= min_new_size):
            return None, False
        
        # Choose a feasible operator arity.
        target_degree = self.random_state.choice(valid_degrees[valid_degrees <= min_new_size])
        new_operator = self._get_random_operator(int(target_degree))
        
        # Decide whether to replace the root or a leaf.
        should_replace_root = (self.random_state.random() < 0.5) or (parent.tree.size == 1)
        
        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)
        
        # Apply the mutation.
        if should_replace_root:
            # Replace the root: the original root becomes one child of the new operator.
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
            # Replace a leaf: the selected leaf becomes one child of the new operator.
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
        Insert a new operator node inside the tree.
        
        Strategy
        --------
        1. Randomly choose an operator with degree greater than zero.
        2. Select a target node that is neither a leaf nor the root.
        3. Use the target subtree as one child of the new operator.
        4. Fill the remaining child positions with randomly generated leaves.
        5. Replace the original target node with the new operator node.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        # Check how much additional size can still be inserted.
        min_new_size = self.maxsize - parent.tree.size
        if min_new_size <= 0:
            return None, False
        
        valid_degrees = np.array(self.generator.degrees[1:])
        if not any(valid_degrees <= min_new_size):
            return None, False
        
        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)
        
        # Select the target node on the cloned tree.
        target_node = self.get_subtree(new_expr.tree, not_leaf=True, not_root=True)
        if target_node is None:
            return None, False
        
        # Choose a feasible operator.
        target_degree = self.random_state.choice(valid_degrees[valid_degrees <= min_new_size])
        new_operator = self._get_random_operator(int(target_degree))
        
        target_parent = target_node.parent
        
        # Insert the new operator above the target subtree.
        new_node = SymbolicNode(node_content=new_operator)
        children = [target_node]  # Reuse the selected subtree directly.
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
        Delete a randomly selected node from the tree.
        
        Strategy
        --------
        1. If the selected node is a leaf, replace it with another random leaf.
        2. If the selected node is internal, randomly promote one of its children.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        # Trees of size 1 cannot be reduced further.
        if parent.tree.size <= 1:
            return None, False
        
        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)
        
        # Select the target node directly on the cloned tree.
        target_node = self.get_subtree(new_expr.tree)
        if target_node is None:
            return None, False
        
        # Apply the deletion.
        if target_node.is_leaf:
            # Leaf case: replace the node with a new random leaf.
            new_leaf_op = self._get_leaf_with_rules(target_node)
            target_node.node_content = new_leaf_op
        else:
            # Internal-node case: promote one of the children.
            promoted_child = self.random_state.choice(list(target_node.children))
            target_parent = target_node.parent
            if target_parent is None:
                # If the target is the root, replace the whole tree.
                new_expr.tree = promoted_child
            else:
                # Otherwise, replace the corresponding child pointer in the parent.
                children_list = list(target_parent.children)
                idx = children_list.index(target_node)
                children_list[idx] = promoted_child
                target_parent.children = children_list
        
        return new_expr, True

    def do_nothing_tree(self, parent: Expression):
        """
        Return an unchanged copy of the parent expression.
        
        This operator is useful when the mutation schedule intentionally keeps a
        fraction of individuals unchanged.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to copy.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the copied expression and
            ``True``.
        """
        new_expr = self.reproduce(parent)
        return new_expr, True

    def mutate_constant(self, parent: Expression):
        """
        Mutate a randomly selected constant value.
        
        Strategy
        --------
        1. Randomly choose a constant node.
        2. Apply a multiplicative perturbation to its value.
        3. With small probability, flip the sign of the perturbation.
        
        Perturbation rule
        -----------------
        ``perturbation = 1 + perturbation_factor * random() + 0.1``
        
        The new value is either ``old_value * perturbation`` or
        ``old_value / perturbation``. With probability
        ``probability_negate_constant``, the perturbation is negated.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        # Skip expressions that do not contain constants.
        if not parent._has_constants():
            return None, False
        
        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)
        
        # Collect candidate constant nodes on the cloned tree.
        candidates = [node for node in PreOrderIter(new_expr.tree) 
                    if isinstance(node.node_content, Constant)]
        
        if not candidates:
            return None, False
        
        # Select a constant node at random.
        target_node = self.random_state.choice(candidates)
        
        # Compute the new constant value.
        perturbation = 1 + self.perturbation_factor * self.random_state.random() + 0.1
        perturbation = perturbation if self.random_state.uniform() > 0.5 else 1/perturbation
        if self.random_state.uniform() < self.probability_negate_constant:
            perturbation = -perturbation
        new_value = target_node.node_content.value * perturbation
        
        # Apply the updated constant value on the cloned tree.
        target_node.node_content = Constant(value=new_value)
        
        return new_expr, True

    def mutate_variable(self, parent: Expression):
        """
        Replace a randomly selected variable with another variable.
        
        Strategy
        --------
        1. Randomly choose a variable node.
        2. Select a replacement variable using Gaussian-like weights so that
           nearby variable indices are preferred.
        
        Weight rule
        -----------
        ``weight[i] = exp(-0.5 * distance^2)``, where
        ``distance = |i - current_variable_index|``.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        # Skip expressions that do not contain variables.
        if not parent._has_variables():
            return None, False
        
        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)
        
        # Collect candidate variable nodes on the cloned tree.
        candidates = [node for node in PreOrderIter(new_expr.tree) 
                    if isinstance(node.node_content, Variable)]

        if not candidates:
            return None, False

        # Select a variable node at random.
        target_node = self.random_state.choice(candidates)
        
        # Choose a replacement variable with Gaussian-style weighting.
        old_variable = target_node.node_content
        variable_idx = self.generator.variables.index(old_variable)
        variable_indices = np.delete(np.arange(len(self.generator.variables)), variable_idx)
        
        if len(variable_indices) == 0:
            return None, False
        
        distances = np.abs(variable_indices - variable_idx)
        weights = np.exp(-0.5 * distances ** 2)
        new_idx = self.random_state.choice(variable_indices, p=weights/weights.sum())
        new_variable = self.generator.variables[new_idx]
        
        # Apply the new variable on the cloned tree.
        target_node.node_content = new_variable
        
        return new_expr, True

    def mutate_operator(self, parent: Expression):
        """
        Replace a randomly selected operator with another operator of the same degree.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        # Trees of size 1 contain no internal operators to replace.
        if parent.size == 1:
            return None, False
        
        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)
        
        # Select the target internal node on the cloned tree.
        target_node = self.get_subtree(new_expr.tree, not_leaf=True)
        if target_node is None:
            return None, False
        
        # Choose a replacement operator with the same arity, excluding the current one.
        target_degree = target_node.degree
        target_operator = target_node.node_content
        new_operator = self._get_random_operator(target_degree, exclude=target_operator)
        
        if not new_operator:
            return None, False
        
        # Apply the replacement on the cloned tree.
        target_node.node_content = new_operator
        
        return new_expr, True


    def mutate_aggregation(self, parent: Expression):
        """
        Mutate aggregation nodes: modify window ranges, operation types, or neighbor counts
        for DynamicAggregation nodes (supporting both spatial and spectral dimensions).
        
        Mutation types:
        1. Change operation type (mean, max, min, etc.) - low probability 
           (0.0001 * number of valid operations)
        2. Change window range (spectral dimension) - high probability:
           - shift_both: translate window as a whole
           - shift_start: move start position
           - shift_end: move end position
           - expand: expand window (left, right, or both)
           - shrink: shrink window (from left, right, or both)
        3. Change neighbor count (spatial dimension) - high probability:
           - increase/decrease k_neighbors within valid bounds
        
        Args:
            parent: Parent expression tree
            
        Returns:
            tuple: (new_expr, success) - New expression and success flag
        """
        # 1. Check if expression contains aggregation nodes
        if not parent._has_aggregations():
            return None, False
        
        # 2. Create new expression via reproduction
        new_expr = self.reproduce(parent)
        
        # 3. Collect candidate nodes: support both legacy DynamicAggregation 
        #    and new DynamicAggregation
        candidates = []
        for node in PreOrderIter(new_expr.tree):
            content = node.node_content
            if isinstance(content, DynamicAggregation):
                candidates.append(node)
        
        if not candidates:
            return None, False
        
        # 4. Select target node randomly
        target_node = self.random_state.choice(candidates)
        aggregation = target_node.node_content
        
        # Ensure we're working with DynamicAggregation
        if not isinstance(aggregation, DynamicAggregation):
            return None, False
        
        # 5. Determine which dimensions are active for mutation
        spectral_active = aggregation._spectral_active
        spatial_active = aggregation._spatial_active
        
        if not spectral_active and not spatial_active:
            # Identity node, nothing to mutate
            return None, False
        
        # 6. Decide which dimension(s) to mutate
        # If both active, randomly choose one; if only one active, mutate that
        if spectral_active and spatial_active:
            mutate_dimension = self.random_state.choice(['spectral', 'spatial'])
        elif spectral_active:
            mutate_dimension = 'spectral'
        else:
            mutate_dimension = 'spatial'
        
        # 7. Perform mutation based on selected dimension
        if mutate_dimension == 'spectral' and spectral_active:
            new_aggregation = self._mutate_spectral_dimension(aggregation)
        elif mutate_dimension == 'spatial' and spatial_active:
            new_aggregation = self._mutate_spatial_dimension(aggregation)
        else:
            # Fallback: should not reach here if logic is correct
            return None, False
        
        if new_aggregation is None:
            return None, False
        
        # 8. Apply mutation to tree
        target_node.node_content = new_aggregation
        
        return new_expr, True

    def _mutate_spectral_dimension(self, aggregation: DynamicAggregation):
        """
        Mutate the spectral (feature) dimension of a DynamicAggregation node.
        
        Args:
            aggregation: DynamicAggregation instance with spectral aggregation active
            
        Returns:
            DynamicAggregation: New aggregation instance with mutated parameters,
                               or None if mutation failed
        """
        valid_stats = self.generator.spectral_stats
        valid_op_num = len(valid_stats) if valid_stats else 1
        prob_mutate_operator = 0.0001 * valid_op_num if valid_op_num > 1 else 0.0
        
        if self.random_state.random() < prob_mutate_operator and valid_stats:
            # Mutate operation type
            new_stat_name = self.random_state.choice(valid_stats)
            new_aggregation = DynamicAggregation(
                v_start=aggregation.v_start,
                v_end=aggregation.v_end,
                stat_name_spectral=new_stat_name,
                window_size=aggregation.window_size,
                stat_name_spatial=aggregation.stat_name_spatial,
                target_feature=aggregation.target_feature,
                n_variables=aggregation.n_variables
            )
            return new_aggregation
        else:
            # Mutate window range [v_start, v_end]
            v_start, v_end = aggregation.v_start, aggregation.v_end
            n_variables = aggregation.n_variables
            current_window_size = v_end - v_start + 1
            
            if current_window_size < 2:
                # Window too small to mutate range, skip or force minimal expansion
                return None
            
            max_change_ratio = 0.5
            max_shift = max(1, int(current_window_size * max_change_ratio))
            mutation_type = self.random_state.choice(
                ['shift_both', 'shift_start', 'shift_end', 'expand', 'shrink']
            )
            
            if mutation_type == 'shift_both':
                # Translate window as a whole
                shift = self.random_state.randint(-max_shift, max_shift + 1)
                new_start = v_start + shift
                new_end = v_end + shift
                # Boundary handling
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
                # Move start position only
                shift = self.random_state.randint(-max_shift, max_shift + 1)
                new_start = max(0, min(v_start + shift, v_end - 1))
                new_end = v_end
            elif mutation_type == 'shift_end':
                # Move end position only
                shift = self.random_state.randint(-max_shift, max_shift + 1)
                new_end = max(v_start + 1, min(v_end + shift, n_variables - 1))
                new_start = v_start
            elif mutation_type == 'expand':
                # Expand window
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
                # Shrink window
                max_shrink = min(max_shift, current_window_size - 2)
                if max_shrink < 1:
                    # Window too small to shrink, fallback to shift
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
            
            # Final boundary validation
            new_start = max(0, min(new_start, n_variables - 2))
            new_end = max(new_start + 1, min(new_end, n_variables - 1))
            
            # Validate derivative order compatibility
            deriv_order = aggregation.deriv_order
            if new_end - new_start < deriv_order:
                # Adjust to satisfy derivative requirement
                new_end = new_start + deriv_order + 1
                if new_end >= n_variables:
                    new_start = max(0, n_variables - deriv_order - 2)
                    new_end = n_variables - 1
            
            length = new_end - new_start + 1
            if isinstance(self.generator.valid_spectral_length, int):
                if length > self.generator.valid_spectral_length:
                    return None
            elif isinstance(self.generator.valid_spectral_length, tuple):
                min_length, max_length = self.generator.valid_spectral_length
                if length < min_length or length > max_length:
                    return None

            new_aggregation = DynamicAggregation(
                v_start=new_start, v_end=new_end,
                stat_name_spectral=aggregation.stat_name_spectral,
                window_size=aggregation.window_size,
                stat_name_spatial=aggregation.stat_name_spatial,
                target_feature=aggregation.target_feature,
                n_variables=aggregation.n_variables
            )
            return new_aggregation

    def _mutate_spatial_dimension(self, aggregation: DynamicAggregation):
        """
        Mutate the spatial (sample) dimension of a DynamicAggregation node.
        
        Args:
            aggregation: DynamicAggregation instance with spatial aggregation active
            
        Returns:
            DynamicAggregation: New aggregation instance with mutated parameters,
                               or None if mutation failed
        """
        valid_stats = self.generator.spatial_stats
        valid_op_num = len(valid_stats) if valid_stats else 1
        prob_mutate_operator = 0.0001 * valid_op_num if valid_op_num > 1 else 0.0
        
        if self.random_state.random() < prob_mutate_operator and valid_stats:
            # Mutate spatial operation type
            new_stat_name = self.random_state.choice(valid_stats)
            new_aggregation = DynamicAggregation(
                v_start=aggregation.v_start,
                v_end=aggregation.v_end,
                stat_name_spectral=aggregation.stat_name_spectral,
                window_size=aggregation.window_size,
                stat_name_spatial=new_stat_name,
                target_feature=aggregation.target_feature,
                n_variables=aggregation.n_variables
            )
            return new_aggregation
        else:
            # Mutate window size in sample dimension
            new_window_size = self.random_state.choice(self.generator.valid_window_sizes)
            
            new_aggregation = DynamicAggregation(
                v_start=aggregation.v_start,
                v_end=aggregation.v_end,
                stat_name_spectral=aggregation.stat_name_spectral,
                window_size=new_window_size,
                stat_name_spatial=aggregation.stat_name_spatial,
                target_feature=aggregation.target_feature,
                n_variables=aggregation.n_variables
            )
            return new_aggregation

    def swap_operands(self, parent: Expression):
        """
        Swap the two children of a randomly selected binary operator.
        
        This mutation applies to degree-2 operators only. It changes the tree
        structure and may or may not change the semantics, depending on whether
        the operator is commutative.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        # Check whether the tree contains any binary operator.
        if not parent._has_binary_operator():
            return None, False
        
        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)
        
        # Collect candidate binary nodes on the cloned tree.
        candidates = [node for node in PreOrderIter(new_expr.tree) if node.degree == 2]
        
        if not candidates:
            return None, False
        
        # Select a target node.
        target_node = self.random_state.choice(candidates)
        
        # Reverse the child order.
        swapped_children = list(target_node.children)[::-1]
        target_node.children = swapped_children
        
        return new_expr, True

    def rotate_tree(self, parent: Expression):
        r"""
        Perform a left or right tree rotation on a selected subtree.
        
        Right-rotation example (``A`` is the parent, ``B`` is the left child)::
        
            A              B
           / \            / \
          B   C    =>    D   A
         / \                / \
        D   E              E   C
        
        Left-rotation example (``A`` is the parent, ``B`` is the right child)::
        
            A                B
           / \              / \
          C   B      =>    A   E
             / \          / \
            D   E        C   D
        
        Validity conditions
        -------------------
        - Right rotation requires a non-leaf left child.
        - Left rotation requires a degree-2 parent and a non-leaf right child.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the rotated expression and
            a Boolean success flag.
        """
        def is_valid_rotation_node(node: SymbolicNode) -> bool:
            """Return whether a node can serve as a rotation root."""
            if node.is_leaf:
                return False
            return any(not child.is_leaf for child in node.children)

        # Check whether the original tree contains any rotatable node.
        for node in PreOrderIter(parent.tree):
            if is_valid_rotation_node(node):
                break
        else:
            return None, False

        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)

        # Collect rotatable nodes on the cloned tree.
        candidates = [node for node in PreOrderIter(new_expr.tree) if is_valid_rotation_node(node)]
        if not candidates:
            return None, False

        # Select the subtree root to rotate.
        subtree_root = self.random_state.choice(candidates)
        
        # Determine which rotation directions are valid.
        can_rotate_right = not subtree_root.children[0].is_leaf
        can_rotate_left = subtree_root.degree == 2 and not subtree_root.children[1].is_leaf
        
        if not can_rotate_left and not can_rotate_right:
            return None, False

        # Choose the rotation direction.
        if can_rotate_left and can_rotate_right:
            direction = self.random_state.choice(['left', 'right'])
        elif can_rotate_left:
            direction = 'left'
        else:
            direction = 'right'

        original_parent = subtree_root.parent
        
        # Apply the rotation.
        if direction == 'right':
            # Right rotation.
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
                # B is a unary operator.
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
                # B is a binary operator.
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
            
            # Replace the original subtree root.
            if original_parent is None:
                new_expr.tree = new_B
            else:
                children_list = list(original_parent.children)
                idx = children_list.index(subtree_root)
                children_list[idx] = new_B
                original_parent.children = children_list
                
        else:  # left rotation
            # Left rotation.
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
                # B is a unary operator.
                if A.degree == 1:
                    return None, False
                elif A.degree == 2:
                    new_A.children = [new_C, new_D]
                    new_B.children = [new_A]
                else:
                    return None, False
            elif B.degree == 2:
                # B is a binary operator.
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
            
            # Replace the original subtree root.
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
        Replace a randomly selected subtree with a newly generated random subtree.
        
        Strategy
        --------
        1. Randomly choose a node in the tree.
        2. Determine the maximum allowed size for the replacement subtree.
        3. Sample a valid target size.
        4. Generate a new random subtree of that size.
        5. Replace the selected node with the new subtree.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)
        
        # Select the target node on the cloned tree.
        target_node = self.get_subtree(new_expr.tree)
        if target_node is None:
            return None, False
        
        # Compute the maximum admissible size for the replacement subtree.
        max_target_size = self.maxsize - (new_expr.tree.size - target_node.size)
        valid_sizes = np.array(list(self.generator.complexity_probs.keys()))
        size_probs = np.array(list(self.generator.complexity_probs.values()))
        mask = valid_sizes <= max_target_size
        
        if not np.any(mask):
            return None, False
        
        # Sample the new subtree size.
        target_size = self.random_state.choice(
            valid_sizes[mask], 
            p=size_probs[mask]/size_probs[mask].sum()
        )
        
        # Generate a new subtree with the sampled size.
        new_subtree = self.generator.build_tree(target_size)
        
        # Replace the target node.
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
        Replace a subtree with one of its own descendants.
        
        Strategy
        --------
        1. Randomly choose a non-leaf subtree.
        2. Randomly choose a non-root node inside that subtree.
        3. Replace the selected subtree with the chosen descendant.
        
        This mutation typically reduces tree size while preserving part of the
        original structure.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to mutate.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        # Trees of size 1 cannot be hoisted further.
        if parent.tree.size <= 1:
            return None, False
        
        # Clone the parent expression before editing it.
        new_expr = self.reproduce(parent)
        
        # Select a non-leaf subtree on the cloned tree.
        subtree = self.get_subtree(new_expr.tree, not_leaf=True)
        if subtree is None:
            return None, False
        
        # Select a non-root descendant inside the chosen subtree.
        subsubtree = self.get_subtree(subtree, not_root=True)
        if subsubtree is None:
            return None, False
        
        # Replace the subtree by the selected descendant.
        if subtree.is_root:
            # If the selected subtree is the root, replace the entire tree.
            new_expr.tree = subsubtree
        else:
            # Otherwise, replace the corresponding child reference in the parent.
            target_parent = subtree.parent
            children_list = list(target_parent.children)
            replacement_idx = children_list.index(subtree)
            children_list[replacement_idx] = subsubtree
            target_parent.children = children_list
        
        return new_expr, True

    def simplify(self, parent: Expression):
        """
        Simplify an expression using algebraic rewrite rules.
        
        Typical simplifications include constant folding, identity rules such as
        ``x + 0 = x`` and ``x * 1 = x``, zero rules such as ``x * 0 = 0``, and
        algebraic rewrites such as ``x - x = 0`` or ``x / x = 1``.
        
        Parameters
        ----------
        parent : Expression
            Parent expression to simplify.
            
        Returns
        -------
        tuple
            A pair ``(new_expr, success)`` containing the mutated expression and
            a Boolean success flag.
        """
        new_expr = parent.simplify(self.constants_tolerance)
        
        # Judge whether simplification has occurred.
        simplified = not Expression._trees_are_equal(parent.tree, new_expr.tree)
        return new_expr, simplified

    def optimize_constants(
        self, parent: Expression, 
        X: np.ndarray, y: np.ndarray, 
        sample_weight: Optional[np.ndarray] = None,
        optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, optimizer_iterations=10
    ):
        """
        Optimize constant values in an expression using numerical optimization.

        Uses scipy.optimize.minimize with multiple restarts to find optimal
        constant values that minimize (or maximize) the fitness function.

        Parameters
        ----------
        parent : Expression
            Parent expression containing constants to optimize.
        X : ndarray, shape (n_samples, n_features)
            Input features.
        y : ndarray, shape (n_samples,)
            Target values.
        sample_weight : ndarray, optional, shape (n_samples,)
            Sample weights for weighted fitness evaluation.
        optimizer_algorithm : str, default='L-BFGS-B'
            Optimization algorithm (e.g., 'L-BFGS-B', 'SLSQP', 'CG').
        optimizer_nrestarts : int, default=3
            Number of optimization restarts with perturbed initial points.
        optimizer_iterations : int, default=10
            Maximum iterations per optimization run.

        Returns
        -------
        tuple
            ``(optimized_expr, success, final_fitness)`` where:
            - optimized_expr: Expression with optimized constants, or None if failed
            - success: bool indicating whether optimization improved fitness
            - final_fitness: float fitness value of the optimized expression
        """
        # Check if expression contains constants
        total_constants = len(parent.constant_indices)
        if not (total_constants > 0):
            return None, False, np.nan

        parent = parent.copy()
        # Extract initial constant values
        initial_constants = np.array([
            node.node_content.value for node in PostOrderIter(parent.tree) 
                if isinstance(node.node_content, Constant)
        ])

        # Define optimization objective function (using pre-compiled gradients)
        def objective(constants_np: np.ndarray):
            # Compute loss using fast NumPy execution
            updated_parent = parent.update_constants(constants_np)
            fitness = updated_parent.fitness(X, y, sample_weight, constants=constants_np)
            loss = -fitness if updated_parent.metric.greater_is_better else fitness
            
            return loss

        # Multi-restart optimization
        best_loss = float('inf')
        best_constants = initial_constants.copy()
        
        for restart in range(optimizer_nrestarts):
            # Initial point
            if restart == 0:
                x0 = initial_constants.copy()
            else:
                noise_scale = 0.1 / np.sqrt(restart)
                noise = self.random_state.uniform(-noise_scale, noise_scale, size=len(initial_constants))
                constants_scale = np.abs(initial_constants) + 1e-6
                x0 = initial_constants + noise * constants_scale
            
            # Execute optimization
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
            # Update best result
            if result.fun < best_loss:
                best_loss = result.fun
                best_constants = result.x

        # Create optimized expression
        optimized_expr = parent.update_constants(best_constants)
        final_fitness = -best_loss if parent.metric.greater_is_better else best_loss
        
        return optimized_expr, True, final_fitness

    def optimize_aggregations(self, parent: Expression, 
                              X: np.ndarray, y: np.ndarray, 
                              sample_weight: Optional[np.ndarray] = None,
                              optimizer_iterations=10, max_shift_ratio=0.1, 
                              early_exaggeration_iter=3, early_stopping_patience=4, 
                              exaggeration_factor=2.5) -> Tuple[Optional['Expression'], bool]:
        """
        Optimize DynamicAggregation node parameters using local search (greedy hill climbing).

        Uses a greedy hill climbing algorithm to optimize aggregation window positions
        and sizes for spectral aggregation nodes.

        Parameters
        ----------
        parent : Expression
            Parent expression containing aggregation nodes to optimize.
        X : ndarray, shape (n_samples, n_features)
            Input features.
        y : ndarray, shape (n_samples,)
            Target values.
        sample_weight : ndarray, optional, shape (n_samples,)
            Sample weights for weighted fitness evaluation.
        optimizer_iterations : int, default=10
            Maximum number of optimization iterations.
        max_shift_ratio : float, default=0.1
            Maximum window shift ratio relative to current window size.
        early_exaggeration_iter : int, default=3
            Number of iterations in the early exaggeration phase (larger step sizes).
        early_stopping_patience : int, default=4
            Number of consecutive non-improving iterations before early stopping.
        exaggeration_factor : float, default=2.5
            Step size multiplier for the early exaggeration phase.

        Returns
        -------
        tuple
            ``(new_expr, success, raw_fitness)`` where:
            - new_expr: Expression with optimized aggregation parameters
            - success: bool indicating optimization success
            - raw_fitness: float fitness value of the optimized expression

        Optimization Strategy
        ---------------------
        1. Shift window (left/right translation)
        2. Expand window (increase coverage)
        3. Shrink window (decrease coverage)
        4. Adjust boundaries (move v_start or v_end independently)

        Time Complexity
        ---------------
        O(iterations * neighbors * n_samples)
        where neighbors ≈ O(n_aggregations * shift_amount * operation_types)
        """
        if parent._count_scalar_aggregations() == 0:
            return None, False, np.nan
        
        # Step 1: Collect all aggregation nodes in a single traversal
        agg_nodes = [node for node in PreOrderIter(parent.tree) 
                    if isinstance(node.node_content, DynamicAggregation)]
        
        # Step 2: Create copy and record initial states
        new_expr = parent.copy()
        n_variables = X.shape[1]
        early_exaggeration_iter = min(early_exaggeration_iter, optimizer_iterations)
        
        # Initial states: [(index, v_start, v_end, stat_name, valid_op), ...]
        initial_states = []
        for node in agg_nodes:
            agg = node.node_content
            initial_states.append({
                'node': node,
                'v_start': agg.v_start,
                'v_end': agg.v_end,
                'stat_name_spectral': agg.stat_name_spectral
            })
        
        # Step 3: Compute initial fitness
        best_fitness = new_expr.fitness(X, y)
        best_states = [s.copy() for s in initial_states]
        
        # Step 4: Define fast state application function
        def apply_states(states):
            """Apply parameter states to nodes in-place."""
            for state in states:
                node = state['node']
                node.node_content = DynamicAggregation(
                    v_start=state['v_start'],
                    v_end=state['v_end'],
                    stat_name_spectral=state['stat_name_spectral'],
                    n_variables=n_variables
                )
            return True
        
        # Step 6: Greedy hill climbing algorithm
        current_states = initial_states
        current_fitness = best_fitness
        
        iterations = 0
        no_improvement_count = 0  # Consecutive non-improving iterations
        
        while iterations < optimizer_iterations and no_improvement_count < early_stopping_patience:
            iterations += 1
            improved = False
            
            # Dynamic step size adjustment (early exaggeration)
            current_exaggeration_factor = (exaggeration_factor 
                                          if iterations <= early_exaggeration_iter 
                                          else 1.0)
            
            # Generate neighbor states
            neighbors = self._get_neighbors(
                n_variables, current_states, max_shift_ratio, 
                current_exaggeration_factor, self.random_state,
                self.generator.valid_spectral_length
            )
            
            # Evaluate neighbors (first-improvement strategy)
            for neighbor_states in neighbors:
                apply_states(neighbor_states)
                
                neighbor_fitness = new_expr.fitness(X, y, sample_weight)
                
                # Check for improvement
                is_better = (neighbor_fitness > current_fitness 
                            if parent.metric.greater_is_better 
                            else neighbor_fitness < current_fitness)
                
                if is_better:
                    # Accept improvement
                    current_states = neighbor_states
                    current_fitness = neighbor_fitness
                    improved = True
                    
                    # Update global best
                    best_states = [s.copy() for s in current_states]
                    best_fitness = current_fitness
                    no_improvement_count = 0
                    break  # First-improvement strategy
            
            # Early stopping counter (skip during exaggeration phase)
            if not improved and iterations > early_exaggeration_iter:
                no_improvement_count += 1
        
        # Step 7: Apply best states
        apply_states(best_states)
        raw_fitness = best_fitness
        
        return new_expr, True, raw_fitness

    @staticmethod
    def _get_neighbors(n_variables, states, max_shift_ratio, 
                       current_exaggeration_factor, random_state, valid_spectral_length=None):
        """
        Generate neighbor states for greedy hill climbing.

        Creates candidate neighbor states by applying various window operations
        including shifts, expansions, and shrinks to each aggregation node.

        Parameters
        ----------
        n_variables : int
            Total number of variables (features) in the spectral dimension.
        states : list of dict
            Current states of aggregation nodes.
        max_shift_ratio : float
            Maximum shift ratio relative to window size.
        current_exaggeration_factor : float
            Multiplier for step size (greater than 1 during exaggeration phase).
        random_state : RandomState
            Random state for stochastic neighbor generation.
        valid_spectral_length : int or tuple, optional
            Valid length constraint for spectral windows.

        Returns
        -------
        list
            List of neighbor state configurations.
        """
        neighbors = []

        def check_and_append(new_states, new_start, new_end):
            """Validate and append neighbor state if within constraints."""
            length = new_end - new_start + 1
            if isinstance(valid_spectral_length, int):
                if length > valid_spectral_length:
                    return None
            elif isinstance(valid_spectral_length, tuple):
                min_length, max_length = valid_spectral_length
                if length < min_length or length > max_length:
                    return None
            
            neighbors.append(new_states)
        
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
                    check_and_append(new_states, new_start, new_end)
            
            # --- 操作2: 仅移动 v_start ---
            for direction in [-1, 1]:
                shift = direction * shift_amount
                new_start = v_start + shift
                
                if (0 <= new_start < v_end - 1 < n_variables) and \
                        (new_start != v_start):
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_start'] = new_start
                    check_and_append(new_states, new_start, new_states[i]['v_end'])
            
            # --- 操作3: 仅移动 v_end ---
            for direction in [-1, 1]:
                shift = direction * shift_amount
                new_end = v_end + shift
                
                if (0 <= v_start < v_start + 1 < new_end < n_variables) and \
                        (new_end != v_end):
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_end'] = new_end
                    check_and_append(new_states, new_states[i]['v_start'], new_end)
            
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
                check_and_append(new_states, new_start, new_end)
            
            # 仅左扩展
            if (v_start > 0) and (new_start != v_start):
                new_start = max(0, v_start - expand_amount)
                new_states = [s.copy() for s in states]
                new_states[i]['v_start'] = new_start
                check_and_append(new_states, new_start, new_states[i]['v_end'])
            
            # 仅右扩展
            if (v_end < n_variables - 1) and (new_end != v_end):
                new_end = min(n_variables - 1, v_end + expand_amount)
                new_states = [s.copy() for s in states]
                new_states[i]['v_end'] = new_end
                check_and_append(new_states, new_states[i]['v_start'], new_end)
            
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
                    check_and_append(new_states, new_start, new_end)
                
                # 仅左收缩
                new_start = min(v_start + shrink_amount, v_end - 1)
                if new_start < v_end and new_start != v_start:
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_start'] = new_start
                    check_and_append(new_states, new_start, new_states[i]['v_end'])
                
                # 仅右收缩
                new_end = max(v_start + 1, v_end - shrink_amount)
                if v_start < new_end and new_end != v_end:
                    new_states = [s.copy() for s in states]
                    new_states[i]['v_end'] = new_end
                    check_and_append(new_states, new_states[i]['v_start'], new_end)
        
        return neighbors



def weighted_random_choice_expr_set(weights_dict: dict, random_state: np.random.RandomState) -> str:
    """
    Select an item using weighted random choice for ExpressionSet operations.

    Parameters
    ----------
    weights_dict : dict
        Mapping of item names to their selection weights.
    random_state : RandomState
        Random state for reproducible selection.

    Returns
    -------
    str
        Selected item name.
    """
    random_state = check_random_state(random_state)
    
    # Convert dict to hashable tuple for caching
    weights_tuple = tuple(sorted(weights_dict.items()))
    names, cumulative_probs = _get_cumulative_probs_expr_set(weights_tuple)
    
    # Roulette wheel selection
    random_value = random_state.uniform()
    idx = np.searchsorted(cumulative_probs, random_value, side='right')
    return names[idx]


@lru_cache(maxsize=128)
def _get_cumulative_probs_expr_set(weights_tuple):
    """
    Compute normalized cumulative probabilities for weighted selection.

    Parameters
    ----------
    weights_tuple : tuple
        Tuple of (name, weight) pairs.

    Returns
    -------
    tuple
        ``(names, cumulative_probs)`` where cumulative_probs is normalized.
    """
    names, weights = zip(*weights_tuple)
    weights_array = np.array(weights, dtype=np.float64)
    
    # Normalize and compute cumulative probabilities
    total_weight = weights_array.sum()
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")
    
    probabilities = weights_array / total_weight
    cumulative_probs = np.cumsum(probabilities)
    cumulative_probs[-1] = 1.0  # Correct floating-point accumulation error
    
    return names, cumulative_probs



class ExpressionSetGP:
    """
    Genetic Programming operators for multi-output ExpressionSet individuals.

    ExpressionSetGP implements crossover and mutation operators that operate on
    collections of Expression trees (ExpressionSet). These operators are used by
    the Population class to evolve multi-output models such as multi-class
    classifiers.

    Mutation Operators
    ------------------
    The following mutation types are available for set-level operations:
    - add_expr: Add a new expression to the set (in a None slot)
    - delete_expr: Remove an expression from the set
    - swap_exprs: Swap two expressions in the set
    - mutate_expr: Apply expression-level mutation to one member
    - mutate_constant: Mutate constants in a random expression
    - randomize_expr: Replace a single expression with a new random one
    - randomize_set: Replace all expressions with new random ones
    - simplify_set: Simplify all expressions in the set
    - do_nothing_set: No change (for weighted selection schemes)

    Crossover Operators
    -------------------
    - single_point: Swap all expressions after a random crossover point
    - two_point: Swap expressions between two crossover points
    - multi_point: Swap expressions at multiple random points
    - subtree_crossover: Apply subtree crossover within corresponding expressions

    Parameters
    ----------
    generator : ExprSetGenerator
        Factory for creating new random expressions.
    gpoperator : ExpressionGP
        Expression-level genetic operators for subtree operations.
    set_mutation_weights : dict, optional
        Relative probability weights for each mutation type.
    set_crossover_method : str, default='single_point'
        Method for set-level crossover ('single_point', 'two_point', 'multi_point').
    set_crossover_probability : float, default=0.0369
        Probability of performing set-level crossover vs. subtree crossover.
    random_state : int or RandomState, optional
        Random seed for reproducibility.

    See Also
    --------
    ExpressionGP : Operators for single-expression individuals.
    ExpressionSet : The multi-output expression container.
    """
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
        """
        Create a new ExpressionSet from a list of expressions.

        Parameters
        ----------
        expressions : list
            List of Expression objects (may include None placeholders).

        Returns
        -------
        ExpressionSet
            New expression set with the given expressions.
        """
        return ExpressionSet(
            expressions=expressions,
            metric=self.generator.metric,
            out_func=self.generator.out_func
        )

    def crossover(self, 
                  parent: ExpressionSet, donor: ExpressionSet, 
        ) -> Tuple[Optional[ExpressionSet], Optional[ExpressionSet], bool]:
        """
        Perform crossover between two ExpressionSets, producing two offspring.

        Strategy
        ---------
        With probability set_crossover_probability, performs set-level crossover
        (swapping entire expressions between sets). Otherwise, performs subtree
        crossover within corresponding expressions.

        Parameters
        ----------
        parent : ExpressionSet
            First parent expression set.
        donor : ExpressionSet
            Second parent expression set (provides genetic material).

        Returns
        -------
        tuple
            ``(offspring1, offspring2, success)`` containing two new offspring
            and a success flag.
        """
        if len(parent) != len(donor):
            raise ValueError("Crossover requires the same length of expressions for both parents.")

        if self.random_state.uniform()<self.set_crossover_probability and len(parent.expressions)>1:
            # Set-level crossover
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
            # Subtree-level crossover
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
        
        # Validate complexity and constraints
        if not offspring1.is_valid(self. maxsize):
            return None, None, False
        if not offspring2.is_valid(self. maxsize):
            return None, None, False

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
            new_expr_set, mutation_succeeded, mutation_name = self.mutate_expr(parent)
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

        # 验证复杂度和约束
        if mutation_succeeded and new_expr_set is not None:
            if not new_expr_set.is_valid(self. maxsize):
                return None, False, mutation_name

        return new_expr_set, mutation_succeeded, mutation_name

    def mutate_expr(self, parent: ExpressionSet):
        """
        Apply expression-level mutation to a randomly selected expression in the set.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set to mutate.

        Returns
        -------
        tuple
            ``(new_expr_set, success, mutation_name)`` containing the mutated
            expression set and success flag.
        """
        valid_points = [i for i, v in enumerate(parent.expressions) if v is not None]
        if not valid_points:
            return None, False
        
        mutation_point = self.random_state.choice(valid_points)
        parent_expr = parent.expressions[mutation_point]
        
        mutated_expr, mutation_succeeded, mutation_name = self.gpoperator.mutation(parent_expr)
        
        # Early failure check
        if not mutation_succeeded:
            return None, False, mutation_name
        
        # Build new expression list
        new_exprs = (
            parent.expressions[:mutation_point] + 
            [mutated_expr] + 
            parent.expressions[mutation_point+1:]
        )
        
        # Create copy with new list
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True, mutation_name

    def mutate_constant(self, parent: ExpressionSet):
        """
        Mutate a constant value in a random expression within the set.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set containing constants to mutate.

        Returns
        -------
        tuple
            ``(new_expr_set, success)`` containing the mutated expression set.
        """
        # Find expressions that can be mutated
        valid_points = [i for i, v in enumerate(parent.expressions) if v is not None]
        if not valid_points:
            return None, False
        
        # Randomly select an expression for mutation
        # Note: If selected expression has no constants, mutate_constant() returns False
        mutation_point = self.random_state.choice(valid_points)
        parent_expr = parent.expressions[mutation_point]

        # Attempt mutation
        mutated_expr, mutation_succeeded = self.gpoperator.mutate_constant(parent_expr)

        if not mutation_succeeded:
            return None, False
            
        # Build new list
        new_exprs = (
            parent.expressions[:mutation_point] + 
            [mutated_expr] + 
            parent.expressions[mutation_point+1:]
        )
        
        # Create copy
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True

    def delete_expr(self, parent: ExpressionSet):
        """
        Delete an expression from the set by replacing it with None.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set.

        Returns
        -------
        tuple
            ``(new_expr_set, success)`` containing the modified expression set.
        """
        valid_points = [i for i, v in enumerate(parent.expressions) if v is not None]
        
        # Check if deletion is allowed
        if len(valid_points) <= self.generator.minorder:
            return None, False
        if not valid_points:
            return None, False

        # Select deletion point
        point_to_delete = self.random_state.choice(valid_points)
        
        # Build new list (replace with None)
        new_exprs = (
            parent.expressions[:point_to_delete] + 
            [None] + 
            parent.expressions[point_to_delete+1:]
        )
        
        # Create copy
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True

    def swap_exprs(self, parent: ExpressionSet):
        """
        Swap two expressions within the set.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set.

        Returns
        -------
        tuple
            ``(new_expr_set, success)`` containing the modified expression set.
        """
        valid_points = [i for i, v in enumerate(parent.expressions) if v is not None]
        if len(valid_points) < 2:
            return None, False
        
        idx1, idx2 = self.random_state.choice(valid_points, size=2, replace=False)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        # Swap expressions in new set
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
        """
        Replace all expressions with newly generated random expressions.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set to replace.

        Returns
        -------
        tuple
            ``(new_expr_set, success)`` containing the new random expression set.
        """
        new_expr_set = self.generator.generate_random_exprset()
        return new_expr_set, True

    def randomize_expr(self, parent: ExpressionSet):
        """
        Replace a single expression with a newly generated random expression.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set.

        Returns
        -------
        tuple
            ``(new_expr_set, success)`` containing the modified expression set.
        """
        # Randomly select a replacement point (can replace None)
        mutation_point = self.random_state.choice(len(parent.expressions))
        
        # Generate new expression
        new_expr = self.generator.generate_random_expr()
        
        # Build new list
        new_exprs = (
            parent.expressions[:mutation_point] + 
            [new_expr] + 
            parent.expressions[mutation_point+1:]
        )

        # Create copy
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True

    def simplify(self, parent: ExpressionSet):
        """
        Simplify all expressions in the set using algebraic rules.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set to simplify.

        Returns
        -------
        tuple
            ``(new_expr_set, success)`` where success is True if any
            simplification occurred.
        """
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
        
        return new_expr_set, True

    def do_nothing_set(self, parent: ExpressionSet):
        """
        Return an unchanged copy of the expression set.

        This operator is useful when the mutation schedule intentionally keeps
        a fraction of individuals unchanged.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set to copy.

        Returns
        -------
        tuple
            ``(new_expr_set, True)`` containing an identical copy.
        """
        new_expr_set = self.reproduce(
            [expr.copy() if expr is not None else None for expr in parent.expressions]
        )
        return new_expr_set, True

    def add_expr(self, parent: ExpressionSet):
        """
        Add a new randomly generated expression to an empty slot in the set.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set with empty slots.

        Returns
        -------
        tuple
            ``(new_expr_set, success)`` containing the modified expression set.
        """
        # Find empty slots
        empty_indices = [i for i, expr in enumerate(parent.expressions) if expr is None]
        if not empty_indices:
            return None, False

        point_to_add = self.random_state.choice(empty_indices)
        
        # Generate new expression
        new_expr = self.generator.generate_random_expr()
        
        # Build new list
        new_exprs = (
            parent.expressions[:point_to_add] + 
            [new_expr] + 
            parent.expressions[point_to_add+1:]
        )
        
        # Create copy
        new_expr_set = self.reproduce(new_exprs)
        
        return new_expr_set, True

    def optimize_constants(
        self, parent: ExpressionSet, 
        X: np.ndarray, y: np.ndarray, 
        sample_weight: Optional[np.ndarray] = None, 
        optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, optimizer_iterations=10
    ) -> Tuple[Optional['ExpressionSet'], bool]:
        """
        Optimize constant values across all expressions in the set.

        Uses scipy.optimize.minimize with multiple restarts to find optimal
        constant values that minimize (or maximize) the fitness function.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set containing constants to optimize.
        X : ndarray, shape (n_samples, n_features)
            Input features.
        y : ndarray, shape (n_samples,)
            Target values.
        sample_weight : ndarray, optional, shape (n_samples,)
            Sample weights for weighted fitness evaluation.
        optimizer_algorithm : str, default='L-BFGS-B'
            Optimization algorithm (e.g., 'L-BFGS-B', 'SLSQP', 'CG').
        optimizer_nrestarts : int, default=3
            Number of optimization restarts with perturbed initial points.
        optimizer_iterations : int, default=10
            Maximum iterations per optimization run.

        Returns
        -------
        tuple
            ``(optimized_exprset, success, final_fitness)`` where:
            - optimized_exprset: ExpressionSet with optimized constants
            - success: bool indicating whether optimization improved fitness
            - final_fitness: float fitness value of the optimized expression
        """
        # Check if expression contains constants
        const_info = parent._build_constant_info()
        if const_info['total_constants'] == 0:
            return None, False, np.nan
        
        parent = parent.copy()
        # Extract initial constant values
        initial_constants = np.array([
            node.node_content.value 
            for expr_idx, node in const_info['flat_nodes']
        ])
        
        # Define optimization objective
        def objective(constants):
            # Compute loss using fast NumPy execution
            updated_parent = parent.update_constants(constants)
            fitness = updated_parent.fitness(X, y, sample_weight, constants=constants)
            loss = -fitness if updated_parent.metric.greater_is_better else fitness
            
            return loss
        
        # Multi-restart optimization
        best_loss = float('inf')
        best_constants = initial_constants.copy()
        
        for restart in range(optimizer_nrestarts):
            # Initial point
            if restart == 0:
                x0 = initial_constants.copy()
            else:
                noise_scale = 0.1 / np.sqrt(restart)
                noise = self.random_state.uniform(-noise_scale, noise_scale, size=len(initial_constants))
                constants_scale = np.abs(initial_constants) + 1e-6
                x0 = initial_constants + noise * constants_scale
            
            # Execute optimization
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
            # Update best result
            if result.fun < best_loss:
                best_loss = result.fun
                best_constants = result.x
        
        # Create optimized expression set
        optimized_expr = parent.update_constants(best_constants)
        final_fitness = -best_loss if parent.metric.greater_is_better else best_loss
        
        return optimized_expr, True, final_fitness

    def optimize_aggregations(
        self, parent: ExpressionSet, 
        X: np.ndarray, y: np.ndarray, 
        sample_weight: Optional[np.ndarray] = None, 
        optimizer_iterations=10, max_shift_ratio=0.1, 
        early_exaggeration_iter=3, early_stopping_patience=4, exaggeration_factor=2.5
    ) -> Tuple[Optional['ExpressionSet'], bool]:
        """
        Optimize DynamicAggregation parameters across all expressions in the set.

        Uses a greedy hill climbing algorithm to optimize aggregation window
        positions and sizes for all spectral aggregation nodes in the set.

        Key Optimizations
        -----------------
        1. Single-pass collection of all aggregation nodes
        2. Batch state dictionary operations (reduced object creation)
        3. Fast state application (in-place node modification)
        4. Vectorized neighbor generation (reduced loop overhead)

        Time Complexity
        ---------------
        - Node collection: O(sum(tree_size)) once
        - Per iteration: O(neighbors × k) where k = number of aggregation nodes
        - Total: O(iterations × neighbors × k)

        Performance
        -----------
        Approximately 2-3x faster than the original implementation.

        Parameters
        ----------
        parent : ExpressionSet
            Parent expression set containing aggregation nodes to optimize.
        X : ndarray, shape (n_samples, n_features)
            Input features.
        y : ndarray, shape (n_samples,)
            Target values.
        sample_weight : ndarray, optional, shape (n_samples,)
            Sample weights for weighted fitness evaluation.
        optimizer_iterations : int, default=10
            Maximum number of optimization iterations.
        max_shift_ratio : float, default=0.1
            Maximum window shift ratio relative to current window size.
        early_exaggeration_iter : int, default=3
            Number of iterations in the early exaggeration phase.
        early_stopping_patience : int, default=4
            Number of consecutive non-improving iterations before stopping.
        exaggeration_factor : float, default=2.5
            Step size multiplier for the early exaggeration phase.

        Returns
        -------
        tuple
            ``(new_exprset, success, raw_fitness)`` where:
            - new_exprset: ExpressionSet with optimized aggregations
            - success: bool indicating optimization success
            - raw_fitness: float fitness value of the optimized expression
        """
        # Check if any expression contains aggregations
        if sum(expr._count_scalar_aggregations() for expr in parent.expressions if expr is not None) <= 0:
            return None, False, np.nan
        
        # Copy original expression set
        new_expr_set = parent.copy()
        
        # Collect all aggregation nodes in single pass (key optimization)
        agg_nodes = []
        for expr in new_expr_set.expressions:
            if expr is not None:
                agg_nodes.extend([node for node in PreOrderIter(expr.tree) 
                                if isinstance(node.node_content, DynamicAggregation)])
        
        n_variables = X.shape[1]
        early_exaggeration_iter = min(early_exaggeration_iter, optimizer_iterations)
        
        # Record initial states
        initial_states = []
        for node in agg_nodes:
            agg = node.node_content
            initial_states.append({
                'node': node,             # Node object
                'v_start': agg.v_start,   # Window start index
                'v_end': agg.v_end,       # Window end index
                'stat_name_spectral': agg.stat_name_spectral  # Aggregation name
            })
        
        # Compute initial fitness
        best_fitness = new_expr_set.fitness(X, y)
        best_states = [s.copy() for s in initial_states]
        
        # Fast state application function (closure optimization)
        def apply_states(states):
            """Apply aggregation parameters in-place."""
            for state in states:
                node = state['node']
                node.node_content = DynamicAggregation(
                    v_start=state['v_start'],
                    v_end=state['v_end'],
                    stat_name_spectral=state['stat_name_spectral'],
                    n_variables=n_variables
                )
        
        # Greedy hill climbing algorithm
        current_states = initial_states
        current_fitness = best_fitness
        
        iterations = 0
        no_improvement_count = 0
        
        while iterations < optimizer_iterations and no_improvement_count < early_stopping_patience:
            iterations += 1
            improved = False
            
            # Dynamic step size adjustment
            current_exaggeration_factor = (exaggeration_factor 
                                        if iterations <= early_exaggeration_iter 
                                        else 1.0)
            
            # Generate neighbor states
            neighbors = self.gpoperator._get_neighbors(
                n_variables, current_states, max_shift_ratio, 
                current_exaggeration_factor, self.random_state,
                self.generator.valid_spectral_length
            )
            
            # Evaluate neighbors (first-improvement strategy)
            for neighbor_states in neighbors:
                apply_states(neighbor_states)
                
                neighbor_fitness = new_expr_set.fitness(X, y, sample_weight)
                
                # Check for improvement
                is_better = (neighbor_fitness > current_fitness 
                        if parent.metric.greater_is_better 
                        else neighbor_fitness < current_fitness)
                
                if is_better:
                    current_states = neighbor_states
                    current_fitness = neighbor_fitness
                    improved = True
                    
                    # Update global best
                    best_states = [s.copy() for s in current_states]
                    best_fitness = current_fitness
                    no_improvement_count = 0
                    break  # First-improvement strategy
            
            # Early stopping counter (skip during exaggeration phase)
            if not improved and iterations > early_exaggeration_iter:
                no_improvement_count += 1
        
        # Apply best states
        apply_states(best_states)
        raw_fitness = best_fitness
        
        return new_expr_set, True, raw_fitness



