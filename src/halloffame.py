import pandas as pd


from src.utils import _calculate_scores
from src.expression import Expression, ExpressionSet



class HallOfFame:
    """
    Stores and maintains the Pareto frontier of non-dominated symbolic expressions.

    The HallOfFame implements a multi-objective archive that keeps track of
    the best solutions found during evolution according to the complexity-error
    tradeoff. It uses Pareto dominance to determine which solutions should be
    retained.

    A solution is Pareto-optimal if no other solution has both:
    1. Equal or better error
    2. Equal or lower complexity

    In other words, a solution is dominated if there exists another solution
    that is no worse in all objectives and strictly better in at least one.

    Parameters
    ----------
    greater_is_better : bool
        Whether higher fitness scores are better. This affects the internal
        optimization direction for dominance comparisons.
    constants_tolerance : float, default=1e-5
        Tolerance for constant simplification when adding new candidates.
    keep_dominated : bool, default=True
        Whether to maintain an archive of dominated solutions for analysis.

    Attributes
    ----------
    entries : dict
        Dictionary of Pareto-optimal solutions keyed by complexity.
        Format: {complexity_key: (expression, objectives_tuple, raw_fitness)}
    complexity_archive : dict or None
        Archive of best solution per complexity (only when keep_dominated=True).
    keep_dominated : bool
        Whether dominated solutions are archived.

    Methods
    -------
    add(candidate, raw_fitness, _is_simplified=False)
        Add a new candidate expression to the Hall of Fame.
    get_pareto_front(include_dominated=False)
        Return a DataFrame of solutions on the Pareto front.

    Examples
    --------
    >>> from src.halloffame import HallOfFame
    >>> hof = HallOfFame(greater_is_better=False)
    >>> # Add some expressions
    >>> hof.add(expr_simple, error=0.1)  # Simple but higher error
    >>> hof.add(expr_complex, error=0.05)  # Complex but lower error
    >>> # Both may be kept if neither dominates the other
    >>> pareto_df = hof.get_pareto_front()
    >>> print(pareto_df)
       complexity  error  expression
    0           5    0.10   x0 + x1
    1          12    0.05  sin(x0) * cos(x1)

    Notes
    -----
    The HallOfFame uses a complexity-keyed structure where each complexity
    value maintains at most one expression. When a new expression with the
    same complexity but better fitness is added, it replaces the existing one.

    Pareto dominance is computed in the (complexity, error) space for single-
    output problems and (order, complexity, error) for multi-output problems.
    Error is always treated as "minimize" internally regardless of whether
    the metric is "greater_is_better".

    The _is_simplified flag controls whether the candidate is simplified
    before checking dominance. If not simplified, it will be simplified first.

    See Also
    --------
    BaseSymbolic : Uses HallOfFame to maintain global best solutions.
    Population : Maintains a local HallOfFame for each island.
    """
    def __init__(self, greater_is_better, constants_tolerance=1e-5, keep_dominated=True):
        """
        Initializes the HallOfFame.

        Parameters
        ----------
        greater_is_better : bool
            Whether a higher fitness score is better. This is needed to correctly
            determine dominance (we always want to minimize error).
        constants_tolerance : float, optional
            Tolerance for simplifying constants (default: 1e-5)
        keep_dominated : bool, optional
            Whether to keep dominated solutions for analysis. Default: True.
        """
        # Pareto optimal solutions: self.entries[complexity_key] = (expression, objectives, raw_fitness)
        self.entries = {}
        # Archive of best solution for each unique complexity configuration
        self.complexity_archive = {} if keep_dominated else None
        
        self.greater_is_better = greater_is_better
        self._multiplier = -1.0 if greater_is_better else 1.0
        self.constants_tolerance = constants_tolerance
        self.keep_dominated = keep_dominated

    def _get_complexity_key(self, candidate):
        """Get unique key for complexity configuration"""
        if isinstance(candidate, ExpressionSet):
            return (candidate.order,  candidate.complexity)
        return  candidate.complexity

    def _get_objectives(self, candidate, raw_fitness):
        """Get objectives tuple (always minimize)"""
        error = raw_fitness * self._multiplier
        if isinstance(candidate, ExpressionSet):
            return (candidate.order,  candidate.complexity, error)
        return ( candidate.complexity, error)

    def _dominates(self, obj1, obj2):
        """
        Check if obj1 dominates obj2 in Pareto sense.
        obj1 dominates obj2 if:
        - obj1 is <= obj2 in all dimensions (minimization)
        - obj1 is strictly < obj2 in at least one dimension
        """
        is_less_equal = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        is_strictly_less = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
        return is_less_equal and is_strictly_less

    def _update_archive(self, complexity_key, expression, raw_fitness, is_pareto):
        """Update archive with new solution if better than existing"""
        if not self.keep_dominated:
            return
        
        # Determine if this solution is better than existing for this complexity
        archive_entry = self.complexity_archive.get(complexity_key)
        
        # If no existing entry or new solution is better
        is_better = (
            archive_entry is None or
            (self.greater_is_better and raw_fitness > archive_entry['error']) or
            (not self.greater_is_better and raw_fitness < archive_entry['error'])
        )
        
        if is_better:
            self.complexity_archive[complexity_key] = {
                'expression': expression.copy(),
                'error': raw_fitness,
                'is_pareto': is_pareto
            }
        # If same complexity but worse fitness, don't store
        # (we only keep the best solution per complexity configuration)

    def add(self, candidate: Expression | ExpressionSet, raw_fitness: float, _is_simplified: bool = False):
        """
        Tries to add a new candidate to the hall of fame, maintaining the
        Pareto frontier. The candidate will be simplified first if not already simplified.
        """
        # Handle simplification first
        if not _is_simplified:
            simplified = candidate.simplify(constants_tolerance=self.constants_tolerance)
            # Only re-add if simplification changed the expression
            if simplified != candidate:
                self.add(simplified, raw_fitness, _is_simplified=True)
                # Original candidate is dominated by its simplified version
                # Only update archive if original has DIFFERENT complexity than simplified
                if self.keep_dominated:
                    orig_key = self._get_complexity_key(candidate)
                    simp_key = self._get_complexity_key(simplified)
                    # Only archive the unsimplified version if it has different complexity
                    if orig_key != simp_key:
                        self._update_archive(orig_key, candidate, raw_fitness, False)
                return
            # If no change after simplification, continue with original
        
        # Get objectives and complexity key
        objectives = self._get_objectives(candidate, raw_fitness)
        complexity_key = self._get_complexity_key(candidate)
        
        # Check if dominated by existing solution with same complexity
        if complexity_key in self.entries:
            existing_objectives = self.entries[complexity_key][1]
            # Compare using internal minimization form
            if self._dominates(existing_objectives, objectives):
                # Dominated by existing solution with same complexity
                self._update_archive(complexity_key, candidate, raw_fitness, False)
                return
        
        # Check Pareto dominance against all current Pareto optimal solutions
        dominated_keys = []
        is_dominated = False
        
        for key, entry in self.entries.items():
            existing_objectives = entry[1]  # Now correctly extract objectives from tuple
            
            # Candidate is dominated by existing solution
            if self._dominates(existing_objectives, objectives):
                is_dominated = True
                break
                
            # Candidate dominates existing solution
            if self._dominates(objectives, existing_objectives):
                dominated_keys.append(key)
        
        if is_dominated:
            self._update_archive(complexity_key, candidate, raw_fitness, False)
            return
        
        # Remove dominated solutions
        for key in dominated_keys:
            expr = self.entries[key][0]
            # Update archive: mark as non-Pareto if it was the archived version
            if self.keep_dominated and key in self.complexity_archive:
                archive_entry = self.complexity_archive[key]
                if archive_entry['expression'] == expr:
                    archive_entry['is_pareto'] = False
            del self.entries[key]
        
        # Add new Pareto optimal solution
        # Store objectives in internal form (for dominance comparisons) and raw_fitness
        self.entries[complexity_key] = (candidate.copy(), objectives, raw_fitness)
        
        # Update archive - this solution is Pareto optimal
        self._update_archive(complexity_key, candidate, raw_fitness, True)

    def __len__(self) -> int:
        return len(self.entries)

    def get_pareto_front(self, include_dominated=False) -> pd.DataFrame:
        """
        Returns the Pareto front, optionally including dominated solutions.
        
        Parameters
        ----------
        include_dominated : bool, optional (default=False)
            If True, includes the best solution for each complexity configuration,
            including those not in the Pareto front.
            Requires HallOfFame to be initialized with keep_dominated=True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing solutions with columns:
            - Single output: ['complexity', 'expression', 'score', 'error', 'is_pareto']
            - Multi-output: ['order', 'complexity', 'expression', 'score', 'error', 'is_pareto']
        """
        if include_dominated and not self.keep_dominated:
            raise ValueError("include_dominated=True requires HallOfFame to be initialized with keep_dominated=True")
        
        # Prepare data
        all_entries = []
        
        if include_dominated and self.complexity_archive:
            # Get best solution for each complexity configuration
            for complexity_key, archive_entry in self.complexity_archive.items():
                expr = archive_entry['expression']
                error = archive_entry['error']
                is_pareto = archive_entry['is_pareto']
                
                if isinstance(complexity_key, tuple):  # ExpressionSet: (order, complexity)
                    order, complexity = complexity_key
                    all_entries.append((order, complexity, expr, error, is_pareto))
                else:  # Expression: complexity
                    complexity = complexity_key
                    all_entries.append((complexity, expr, error, is_pareto))
        else:
            # Standard Pareto front (only non-dominated solutions)
            for complexity_key, entry in self.entries.items():
                expr = entry[0]
                raw_fitness = entry[2]  # Extract raw_fitness from the tuple
                
                if isinstance(complexity_key, tuple):  # ExpressionSet
                    order, complexity = complexity_key
                    all_entries.append((order, complexity, expr, raw_fitness, True))
                else:  # Expression
                    complexity = complexity_key
                    all_entries.append((complexity, expr, raw_fitness, True))
        
        if not all_entries:
            return pd.DataFrame()
        
        # Determine if multi-output based on first entry
        first_entry = all_entries[0]
        is_multi_output = len(first_entry) == 5  # (order, complexity, expr, error, is_pareto)
        
        # Build DataFrame
        if is_multi_output:
            columns = ['order', 'complexity', 'expression', 'error', 'is_pareto']
        else:
            columns = ['complexity', 'expression', 'error', 'is_pareto']
        
        df = pd.DataFrame(all_entries, columns=columns)
        
        # Calculate scores
        scores = _calculate_scores(df, self.greater_is_better)
        score_col_idx = 2 if is_multi_output else 1
        df.insert(score_col_idx + 1, 'score', scores)
        
        # Reset index for clean display
        df = df.reset_index(drop=True)
        
        # Sort by complexity and Pareto status
        if is_multi_output:
            df = df.sort_values(
                by=['order', 'complexity', 'is_pareto', 'error'],
                ascending=[True, True, False, not self.greater_is_better]
            )
        else:
            df = df.sort_values(
                by=['complexity', 'is_pareto', 'error'],
                ascending=[True, False, not self.greater_is_better]
            )
        
        df = df.reset_index(drop=True)
        return df