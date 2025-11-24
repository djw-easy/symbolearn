import pandas as pd


from src.utils import _calculate_scores
from src.expression import Expression, ExpressionSet


class HallOfFame:
    """
    A data structure to store and maintain the Pareto frontier of expressions
    found during symbolic regression evolution.
    """
    def __init__(self, greater_is_better, constants_tolerance=1e-5):
        """
        Initializes the HallOfFame.

        Parameters
        ----------
        greater_is_better : bool
            Whether a higher fitness score is better. This is needed to correctly
            determine dominance (we always want to minimize error).
        constants_tolerance : float, optional
            Tolerance for simplifying constants (default: 1e-5)
        """
        # self.entries[complexity] = (expression, error)
        self.entries = {}
        # If greater_is_better is true, we need to flip the sign of the fitness
        # for dominance comparison, as Pareto fronts assume minimization of objectives.
        self.greater_is_better = greater_is_better
        self._multiplier = -1.0 if greater_is_better else 1.0
        self.constants_tolerance = constants_tolerance

    def add(self, candidate: Expression | ExpressionSet, raw_fitness: float, _is_simplified: bool = False):
        """
        Tries to add a new candidate to the hall of fame, maintaining the
        Pareto frontier. If the candidate can be added, it will be simplified first
        and then the add process is executed again with the simplified version.

        Parameters
        ----------
        candidate : Expression or ExpressionSet
            An individual that has been evaluated and has .size, 
            and optionally .__len__() attributes.
        raw_fitness : float
            The raw fitness value of the candidate.
        _is_simplified : bool, optional
            Internal flag to prevent infinite recursion. Should not be set by users.
        """
        # 如果未简化,先简化再递归调用
        if not _is_simplified:
            if isinstance(candidate, Expression):
                simplified = candidate.simplify(constants_tolerance=self.constants_tolerance)
                # 递归添加简化后的版本
                self.add(simplified, raw_fitness, _is_simplified=True)
                return
            elif isinstance(candidate, ExpressionSet):
                simplified = candidate.simplify(constants_tolerance=self.constants_tolerance)
                # 递归添加简化后的版本
                self.add(simplified, raw_fitness, _is_simplified=True)
                return
        
        # 以下代码只处理已简化的候选者
        # Determine objectives (always minimize)
        error = raw_fitness * self._multiplier
        if isinstance(candidate, ExpressionSet):
            objectives = (candidate.order, candidate.size, error)
            candidate_key = (candidate.order, candidate.size)
        else:
            objectives = (candidate.size, error)
            candidate_key = candidate.size

        # 检查是否已存在相同或更好的解
        if candidate_key in self.entries:
            existing_objectives = self.entries[candidate_key][1]
            # 注意: existing_objectives 最后一个元素是 raw_fitness, 需要转换为 error
            existing_error = existing_objectives[-1] * self._multiplier
            
            # 重建完整的 objectives 用于比较(都使用 error)
            if isinstance(candidate, ExpressionSet):
                existing_obj_for_comparison = (existing_objectives[0], existing_objectives[1], existing_error)
            else:
                existing_obj_for_comparison = (existing_objectives[0], existing_error)
            
            # 如果现有解在所有目标上都不劣于新候选者,则不添加
            if all(e <= o for e, o in zip(existing_obj_for_comparison, objectives)):
                return

        # Check for dominance
        dominated_keys = []
        is_dominated = False
        
        for key, (expr, stored_objectives) in self.entries.items():
            # 将存储的 raw_fitness 转换为 error 进行比较
            stored_error = stored_objectives[-1] * self._multiplier
            
            if isinstance(candidate, ExpressionSet):
                existing_obj_for_comparison = (stored_objectives[0], stored_objectives[1], stored_error)
            else:
                existing_obj_for_comparison = (stored_objectives[0], stored_error)
            
            # 新候选者被现有解支配
            if all(e <= o for e, o in zip(existing_obj_for_comparison, objectives)):
                is_dominated = True
                break
            
            # 新候选者支配现有解
            if all(o <= e for o, e in zip(objectives, existing_obj_for_comparison)):
                dominated_keys.append(key)

        if is_dominated:
            return

        # Remove dominated entries
        for key in dominated_keys:
            del self.entries[key]

        # Add the new non-dominated solution
        # 存储时最后一个元素保持为 raw_fitness
        objectives_to_store = tuple(list(objectives)[:-1] + [raw_fitness])
        self.entries[candidate_key] = (candidate.copy(), objectives_to_store)

    def __len__(self) -> int:
        """
        Returns the number of entries in the hall of fame.

        Returns
        -------
        int
            The number of entries in the hall of fame.
        """
        return len(self.entries)

    def get_pareto_front(self) -> list:
        """
        Returns the current Pareto front, sorted by objectives.

        Returns
        -------
        list
            A list of tuples representing the Pareto front.
            For Expression: [(complexity, raw_fitness, expression_string), ...]
            For ExpressionSet: [(num_expressions, complexity, raw_fitness, expression_string), ...]
        """
        pareto_front = []
        # Sort by the objectives tuple, which handles multi-dimensional sorting
        sorted_keys = sorted(self.entries.keys())
        
        for key in sorted_keys:
            expr, objectives = self.entries[key]
            raw_fitness = objectives[-1]
            
            if len(objectives) == 3: # ExpressionSet
                num_expr, complexity, _ = objectives
                pareto_front.append((num_expr, complexity, raw_fitness, expr))
            else: # Expression
                complexity, _ = objectives
                pareto_front.append((complexity, raw_fitness, expr))
        
        
        # Check the dimensionality of the front to determine columns
        first_entry = pareto_front[0]
        is_multi_output = len(first_entry) == 4

        # Convert expressions to strings for display
        front_for_df = []
        for entry in pareto_front:
            if is_multi_output:
                order, complexity, error, expr_obj = entry
                front_for_df.append((order, complexity, expr_obj, error))
            else:
                complexity, error, expr_obj = entry
                front_for_df.append((complexity, expr_obj, error))

        if is_multi_output:
            columns = ['order', 'complexity', 'expression', 'error']
        else:
            columns = ['complexity', 'expression', 'error']

        hof_df = pd.DataFrame(front_for_df, columns=columns)
        if front_for_df:
            # 计算并添加score列
            scores = _calculate_scores(hof_df, self.greater_is_better)
            hof_df.insert(3, 'score', scores) if is_multi_output else hof_df.insert(2, 'score', scores)
            if is_multi_output:
                hof_df = hof_df.sort_values(by=['order', 'complexity'])
            else:
                hof_df = hof_df.sort_values(by=['complexity'])
        
        return hof_df

