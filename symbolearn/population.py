from typing import List, Union, Tuple, Optional, Dict
from collections import Counter, defaultdict
from scipy import sparse
import numpy as np
import pandas as pd
import math



from symbolearn.node import Constant
from symbolearn.tree import PostOrderIter
from symbolearn.halloffame import HallOfFame
from symbolearn.utils import check_random_state
from symbolearn.log import EvolutionLogger, LogAnalyzer
from symbolearn.expression import Expression, ExpressionSet
from symbolearn.gpoperator import ExpressionGP, ExpressionSetGP
from symbolearn.generator import ExprGenerator, ExprSetGenerator




class Population:
    """
    Represents a single island (sub-population) in the parallel island model genetic algorithm.

    The Population class encapsulates all evolutionary operations for a single
    island, including:
    - Initialization of random individuals
    - Tournament selection
    - Crossover and mutation reproduction
    - Fitness evaluation
    - Replacement (steady-state evolution)
    - Hall of Fame maintenance for the island

    In the island model, multiple Population instances evolve independently and
    periodically exchange individuals (migration) to share genetic material.
    Each population maintains its own Hall of Fame which is later merged into
    a global Hall of Fame by the BaseSymbolic class.

    Parameters
    ----------
    population_size : int
        Maximum number of individuals (expressions) in this population.
    generator : ExprGenerator or ExprSetGenerator
        Factory for creating new random expressions conforming to constraints.
    gpoperator : ExpressionGP or ExpressionSetGP
        Genetic programming operators for crossover and mutation.
    annealing : bool, default=True
        Whether to use simulated annealing for offspring acceptance.
        When True, worse offspring may be accepted with probability
        decreasing with temperature.
    alpha : float, default=3.17
        Temperature decay rate for simulated annealing. Lower values
        make the acceptance criterion more stringent over time.
    adaptive_parsimony_scaling : float, default=0.6
        Scaling factor for adaptive parsimony pressure. Higher values
        increase selection pressure toward simpler individuals.
    should_simplify : bool, default=True
        Whether to apply algebraic simplification to expressions
        after each generation.
    should_optimize_constants : bool, default=True
        Whether to optimize constant values using numerical optimization
        (L-BFGS-B) after each generation.
    should_optimize_aggregations : bool, default=True
        Whether to optimize spatial/spectral aggregation parameters.
    optimizer_algorithm : str, default='L-BFGS-B'
        Optimization algorithm for constant optimization.
    optimizer_nrestarts : int, default=3
        Number of restarts for the constant optimizer to find global optimum.
    optimizer_probability : float, default=0.14
        Probability that any individual will have its constants optimized
        in a given generation.
    optimizer_iterations : int, default=8
        Maximum iterations per constant optimization run.
    topn : int, default=12
        Number of top individuals used for constant pool updates.
    batching : bool, default=False
        Whether to use mini-batch training for large datasets.
    batch_size : int, default=256
        Number of samples per batch when batching is enabled.
    enable_logging : bool, default=False
        Whether to record detailed evolution logs for analysis.

    Attributes
    ----------
    individuals : list
        List of Expression or ExpressionSet objects representing the current population.
    fitnesses : np.ndarray
        Array of fitness values for each individual.
    complexitys : np.ndarray
        Array of complexity values for each individual.
    hall_of_fame : HallOfFame
        Local Hall of Fame containing the best individuals found in this population.
    niteration : int
        Number of generations this population has evolved.

    Examples
    --------
    >>> from symbolearn.population import Population
    >>> from symbolearn.generator import ExprGenerator
    >>> from symbolearn.gpoperator import ExpressionGP
    >>> generator = ExprGenerator(maxsize=21, operators=['add', 'sub', 'mul', 'div'], ...)
    >>> gpoperator = ExpressionGP(generator, mutation_weights={...})
    >>> pop = Population(population_size=50, generator=generator, gpoperator=gpoperator)
    >>> pop.init_population(X_train, y_train, seed=42)
    >>> # Evolve for several cycles
    >>> pop.evolve(X_train, y_train, seed=43, ncycles=100, crossover_probability=0.02, ...)
    >>> print(f"Best fitness: {max(pop.fitnesses) if pop.greater_is_better else min(pop.fitnesses)}")

    Notes
    -----
    The Population class implements steady-state evolution where:
    1. Each cycle selects parents via tournament selection
    2. Offspring are created via crossover (with probability crossover_probability)
       or mutation
    3. Offspring fitness is evaluated
    4. The oldest individuals are replaced if offspring are accepted

    Simulated annealing controls acceptance of offspring. The acceptance probability
    is: P(accept) = exp(-delta / (temperature * alpha)), where delta is the
    change in fitness. This allows worse individuals to occasionally be accepted,
    promoting exploration.

    See Also
    --------
    BaseSymbolic : Orchestrates multiple Population instances with migration.
    HallOfFame : Stores Pareto-optimal solutions across all populations.
    """
    def __init__(self,
                 population_size: int,
                 generator: ExprGenerator | ExprSetGenerator,
                 gpoperator: ExpressionGP | ExpressionSetGP,
                 annealing: bool = True,
                 alpha: float = 3.17,
                 adaptive_parsimony_scaling: float = 0.6, 
                 should_simplify: bool = True,
                 should_optimize_constants: bool = True,
                 should_optimize_aggregations: bool = True,
                 optimizer_algorithm: str = 'L-BFGS-B',
                 optimizer_nrestarts: int = 3,
                 optimizer_probability: float = 0.14,
                 optimizer_iterations: int = 8,
                 topn: int = 12, 
                 batching: bool = False, 
                 batch_size: int = 256,
                 enable_logging: bool = False):
        """
        初始化一个种群。

        Parameters
        ----------
        population_size : int
            种群大小
        generator : ExprGenerator | ExprSetGenerator
            表达式生成器
        gpoperator : ExpressionGP | ExpressionSetGP
            遗传编程操作器
        enable_logging : bool, default=False
            是否启用进化日志记录
        ... (其他参数)
        """
        self.population_size = population_size
        self.generator = generator
        self.gpoperator = gpoperator
        self.annealing = annealing
        self.alpha, self.topn = alpha, topn
        self.adaptive_parsimony_scaling = adaptive_parsimony_scaling
        self.should_simplify = should_simplify
        self.should_optimize_constants = should_optimize_constants
        self.should_optimize_aggregations = should_optimize_aggregations
        self.optimizer_algorithm = optimizer_algorithm
        self.optimizer_nrestarts = optimizer_nrestarts
        self.optimizer_probability = optimizer_probability
        self.optimizer_iterations = optimizer_iterations
        
        self.maxsize = generator.maxsize
        self.greater_is_better = generator.metric.greater_is_better
        
        # Hall of Fame
        self.hall_of_fame = HallOfFame(self.greater_is_better)
        self.niteration = 0
        
        # 批处理相关
        self.batch_pool = []
        self.batching = batching
        self.batch_size = batch_size
        
        # 日志记录器
        self.enable_logging = enable_logging
        if self.enable_logging:
            self.logger = EvolutionLogger(
                enabled=True,
                greater_is_better=self.greater_is_better
            )
        else:
            self.logger = None

    def __getitem__(self, key):
        return self.individuals[key]

    def __setitem__(self, key, value):
        self.individuals[key] = value

    def __len__(self):
        return len(self.individuals)

    def __repr__(self):
        """返回一个能代表种群状态的字符串。"""
        best_fitness = np.max(self.fitnesses) if self.greater_is_better else np.min(self.fitnesses)
        return (f"Population(size={len(self)}, "
                f"best_fitness={best_fitness:.4f}, "
                f"avg_fitness={np.mean(self.fitnesses):.4f})")

    def _init_batch_pool(self, X, y, random_state, sample_weight=None):
        """
        Pre-generate a pool of data batches for training.
        
        Supports both 2D tabular data and 3D spatial data:
        - 2D: X shape (n_samples, n_features) - traditional sample-wise batching
        - 3D: X shape (height, width, n_features) - spatial window batching
        
        For 3D spatial data, batch_size can be:
        - int: square window size (win_size, win_size, n_features)
        - tuple[int, int]: rectangular window size (win_h, win_w, n_features)
        
        Parameters
        ----------
        X : np.ndarray
            Input data. Either:
            - 2D array of shape (n_samples, n_features) for tabular data
            - 3D array of shape (height, width, n_features) for spatial data
        y : np.ndarray
            Target values. Shape must be compatible with X:
            - For 2D X: shape (n_samples,) or (n_samples, n_targets)
            - For 3D X: shape (height, width) or (height*width,) 
              or (height, width, n_targets) for multi-target per location
        random_state : np.random.RandomState
            Random number generator instance for reproducibility.
        sample_weight : np.ndarray, optional
            Weights for individual samples/spatial locations. 
            Shape must match y.
            
        Returns
        -------
        batch_pool : list of tuples
            Each tuple contains (X_batch, y_batch, sample_weight_batch).
            For 3D spatial data, X_batch shape is (win_h, win_w, n_features).
            For 2D tabular data, X_batch shape is (batch_size, n_features).
        """
        # Dispatch based on input dimensionality
        if X.ndim == 2:
            # Traditional 2D tabular data: (n_samples, n_features)
            return self._init_batch_pool_2d(X, y, random_state, sample_weight)
        elif X.ndim == 3:
            # 3D spatial data: (height, width, n_features)
            return self._init_batch_pool_3d(X, y, random_state, sample_weight)
        else:
            raise ValueError(
                f"X must be 2D (n_samples, n_features) or 3D (height, width, n_features), "
                f"got shape {X.shape}"
            )
    
    def _init_batch_pool_2d(self, X, y, random_state, sample_weight=None):
        """
        Initialize batch pool for 2D tabular data (traditional sample-wise batching).
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        random_state : np.random.RandomState
            Random number generator for reproducibility.
        sample_weight : np.ndarray, shape (n_samples,), optional
            Weights for individual samples.
            
        Returns
        -------
        batch_pool : list of tuples
            List of (X_batch, y_batch, sample_weight_batch) tuples.
        """
        n_samples, n_features = X.shape
        
        # Validate batch_size for 2D data (must be integer)
        if isinstance(self.batch_size, (tuple, list)):
            raise ValueError(
                f"batch_size must be an integer for 2D tabular data, "
                f"got {self.batch_size} of type {type(self.batch_size)}"
            )
        
        use_full_data = self.batch_size >= n_samples
        sample_size = n_samples if use_full_data else self.batch_size
        
        # Calculate batch pool size
        if use_full_data:
            batch_pool_size = 1
        else:
            # Generate extra batches to increase diversity during training
            batch_pool_size = (n_samples // self.batch_size + 1) * 2
        
        batch_pool = []
        for _ in range(batch_pool_size):
            if use_full_data:
                # Use all data but shuffle order for stochasticity
                indices = np.arange(n_samples)
                random_state.shuffle(indices)
            else:
                # Randomly select batch_size samples without replacement
                indices = random_state.choice(n_samples, sample_size, replace=False)
            
            # Extract batch data
            X_batch = X[indices].copy()
            y_batch = y[indices].copy()
            
            # Extract sample weights if provided
            if sample_weight is not None:
                sw_batch = sample_weight[indices].copy()
            else:
                sw_batch = None
            
            batch_pool.append((X_batch, y_batch, sw_batch))
        
        return batch_pool
    
    def _init_batch_pool_3d(self, X, y, random_state, sample_weight=None):
        """
        Initialize batch pool for 3D spatial data (window-based spatial sampling).
        
        Extracts random spatial windows from the input volume.
        If window size >= input spatial dimensions, uses full data as single batch.
        
        Note: This function assumes y and sample_weight have already been validated 
        to have shape (height, width) matching X's spatial dimensions.
        
        Parameters
        ----------
        X : np.ndarray, shape (height, width, n_features)
            Spatial data volume where first two dimensions represent spatial 
            coordinates (e.g., image height/width, grid coordinates) and the 
            last dimension represents feature channels.
        y : np.ndarray, shape (height, width)
            Target values for each spatial location.
            Shape validation is performed upstream before calling this method.
        random_state : np.random.RandomState
            Random number generator for reproducibility.
        sample_weight : np.ndarray, shape (height, width), optional
            Weights for individual spatial locations.
            Shape validation is performed upstream before calling this method.
            
        Returns
        -------
        batch_pool : list of tuples
            Each tuple contains (X_batch, y_batch, sample_weight_batch).
            X_batch shape: (win_h, win_w, n_features) or (height, width, n_features) 
            if using full data.
            y_batch shape: (win_h, win_w) or (height, width) if using full data.
            sample_weight_batch shape: same as y_batch or None if not provided.
        """
        height, width, n_features = X.shape
        
        # Parse and validate batch_size for 3D spatial data
        if isinstance(self.batch_size, int):
            # Square window: (win_size, win_size)
            win_h = win_w = self.batch_size
        elif isinstance(self.batch_size, (tuple, list)) and len(self.batch_size) == 2:
            # Rectangular window: (win_h, win_w)
            win_h, win_w = self.batch_size
            # Ensure integer values
            win_h = int(win_h)
            win_w = int(win_w)
        else:
            raise ValueError(
                f"batch_size for 3D spatial data must be either:\n"
                f"  - int: for square windows (win_size, win_size)\n"
                f"  - tuple/list of 2 ints: for rectangular windows (win_h, win_w)\n"
                f"Got {self.batch_size} of type {type(self.batch_size)}"
            )
        
        # Validate window dimensions are positive
        if win_h <= 0 or win_w <= 0:
            raise ValueError(
                f"Window dimensions must be positive integers, got ({win_h}, {win_w})"
            )
        
        # Check if window size >= input spatial dimensions -> use full data mode
        use_full_data = (win_h >= height) and (win_w >= width)
        
        if use_full_data:
            # Full data mode: return entire volume as single batch
            # This is analogous to batch_size >= n_samples in 2D case
            
            # Extract full data with copies to prevent unintended modifications
            X_batch = X.copy()
            y_batch = y.copy()
            
            # Handle sample_weight if provided (already validated to match y's shape)
            if sample_weight is not None:
                sw_batch = sample_weight.copy()
            else:
                sw_batch = None
            
            # Return single-batch pool (full data mode)
            return [(X_batch, y_batch, sw_batch)]
        
        else:
            # Window sampling mode: extract random sub-windows
            # Validate that window fits within input bounds
            if win_h > height or win_w > width:
                raise ValueError(
                    f"Window size ({win_h}, {win_w}) exceeds input spatial dimensions "
                    f"({height}, {width}). Window must fit within input bounds, or use "
                    f"full data mode by setting window size >= input dimensions."
                )
            
            # Calculate valid starting positions for window sampling
            # h_start can range from 0 to (height - win_h), inclusive
            max_h_start = height - win_h
            max_w_start = width - win_w
            
            # Calculate total number of unique window positions
            total_possible_windows = (max_h_start + 1) * (max_w_start + 1)
            
            # Determine batch pool size strategy:
            # - For small search spaces: use all possible windows for maximum coverage
            # - For large search spaces: sample a representative fraction to balance 
            #   diversity with memory constraints
            min_pool_size = 20    # Minimum diversity guarantee
            max_pool_size = 100   # Maximum memory constraint
            pool_fraction = 0.15  # Sample 15% of possible windows for large spaces
            
            if total_possible_windows <= min_pool_size:
                # Small space: enumerate all possible windows
                batch_pool_size = total_possible_windows
            else:
                # Large space: sample fraction with bounds
                batch_pool_size = max(
                    min_pool_size,
                    min(max_pool_size, int(total_possible_windows * pool_fraction))
                )
            
            batch_pool = []
            
            for _ in range(batch_pool_size):
                # Randomly select window top-left corner coordinates
                h_start = random_state.randint(0, max_h_start + 1)
                w_start = random_state.randint(0, max_w_start + 1)
                h_end = h_start + win_h
                w_end = w_start + win_w
                
                # Extract spatial window from X: shape (win_h, win_w, n_features)
                X_batch = X[h_start:h_end, w_start:w_end, :].copy()
                
                # Extract corresponding y values: shape (win_h, win_w)
                # Note: y shape validation is performed upstream
                y_batch = y[h_start:h_end, w_start:w_end].copy()
                
                # Extract sample weights if provided: shape (win_h, win_w)
                # Note: sample_weight shape validation is performed upstream
                if sample_weight is not None:
                    sw_batch = sample_weight[h_start:h_end, w_start:w_end].copy()
                else:
                    sw_batch = None
                
                batch_pool.append((X_batch, y_batch, sw_batch))
            
            return batch_pool

    def _get_batch(self, X: np.ndarray, y: np.ndarray, random_state, 
                   sample_weight: Optional[np.ndarray] = None):
        """
        Get a batch of data from the pre-generated pool.
        
        Supports both 2D tabular and 3D spatial data formats.
        Automatically dispatches to appropriate batching strategy based on X.ndim.
        
        Parameters
        ----------
        X : np.ndarray
            Input data, either:
            - 2D: shape (n_samples, n_features) for tabular data
            - 3D: shape (height, width, n_features) for spatial data
        y : np.ndarray
            Target values, shape compatible with X as described in _init_batch_pool.
        random_state : np.random.RandomState
            Random number generator instance for batch selection.
        sample_weight : np.ndarray, optional
            Sample weights matching y's shape.
            
        Returns
        -------
        tuple : (X_batch, y_batch, sample_weight_batch)
            Batched data ready for fitness evaluation.
            Shapes depend on input dimensionality and batch_size configuration.
        """
        # If batching is disabled, return full data with all components
        if not self.batching:
            return X, y, sample_weight
        
        # Initialize batch pool on first call (lazy initialization)
        if not self.batch_pool:
            self.batch_pool = self._init_batch_pool(
                X, y, random_state, sample_weight=sample_weight
            )
        
        # Randomly select a batch from the pool (sampling with replacement)
        # This allows the same batch to be selected multiple times across iterations
        batch_idx = random_state.randint(0, len(self.batch_pool))
        batch = self.batch_pool[batch_idx]
        
        return batch

    def init_population(self, X: np.array, y: np.array, 
                        seed: int | np.random.RandomState, 
                        sample_weight: Optional[np.ndarray] = None) -> None:
        """初始化种群。"""
        self.individuals = []
        random_state = check_random_state(seed)
        self.complexitys = np.zeros(self.population_size)
        self.fitnesses = np.zeros(self.population_size, dtype=np.float32)
        for i in range(self.population_size):
            X_batch, y_batch, sample_weight_batch = self._get_batch(
                X, y, random_state, sample_weight
            )
            if isinstance(self.generator, ExprSetGenerator):
                new_individual = self.generator.generate_random_exprset()
            else:
                new_individual = self.generator.generate_random_expr()
            raw_fitness_ = new_individual.fitness(
                X_batch, y_batch, sample_weight=sample_weight_batch
            )
            self.individuals.append(new_individual)
            self.complexitys[i] = new_individual.complexity
            self.fitnesses[i] = raw_fitness_
            self.hall_of_fame.add(new_individual, raw_fitness_)
        sorted_indices = np.argsort(-self.fitnesses) if self.greater_is_better else np.argsort(self.fitnesses)
        self.birthes = np.empty_like(sorted_indices)
        self.birthes[sorted_indices] = np.arange(len(self.fitnesses))
        
        return self

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def find_oldest_n(self, n: int) -> np.ndarray:
        """查找最老的 n 个个体（O(n) 使用 argpartition）"""
        if n <= 0:
            return np.array([], dtype=int)
        n = min(n, len(self.individuals))
        return np.argpartition(self.birthes, -n)[-n:]
    
    def find_top_n(self, n: int, find_best: bool = True) -> np.ndarray:
        """查找最佳/最差的 n 个个体（O(n) 使用 argpartition）"""
        if n == 0:
            return np.array([], dtype=int)
        
        if self.greater_is_better:
            if find_best:
                return np.argpartition(self.fitnesses, -n)[-n:]
            else:
                return np.argpartition(self.fitnesses, n)[:n]
        else:
            if find_best:
                return np.argpartition(self.fitnesses, n)[:n]
            else:
                return np.argpartition(self.fitnesses, -n)[-n:]

    # =========================================================================
    # 优化的锦标赛选择（向量化版本）
    # =========================================================================

    def tournament_selection(self, random_state, tournament_selection_n,
                            tournament_selection_p, find_best: bool = True,
                            n_select: int = 1, ensure_unique: bool = True):
        """
        优化的锦标赛选择（支持批量选择）
        """
        if n_select == 1:
            return self._tournament_selection_single(
                random_state, tournament_selection_n, 
                tournament_selection_p, find_best
            )
        else:
            return self._tournament_selection_batch(
                random_state, tournament_selection_n,
                tournament_selection_p, find_best,
                n_select, ensure_unique
            )
    
    def _tournament_selection_single(self, random_state, tournament_selection_n,
                                     tournament_selection_p, find_best):
        """单个锦标赛选择（内部方法）"""
        contender_indices = random_state.randint(0, len(self.individuals), tournament_selection_n)
        fitnesses = self.fitnesses[contender_indices].copy()
        sorted_indices = np.argsort(fitnesses)
        if self.greater_is_better == find_best:
            sorted_indices = sorted_indices[::-1]
        
        for idx in sorted_indices:
            if random_state.random() < tournament_selection_p:
                parent_index = contender_indices[idx]
                return self.individuals[parent_index], parent_index
        
        best_idx = sorted_indices[0]
        parent_index = contender_indices[best_idx]
        return self.individuals[parent_index], parent_index
    
    def _tournament_selection_batch(self, random_state, tournament_selection_n,
                                    tournament_selection_p, find_best,
                                    n_select, ensure_unique):
        """批量锦标赛选择（内部方法）"""
        selected_individuals = []
        selected_indices = []
        selected_indices_set = set()
        
        max_attempts = n_select * 10
        attempts = 0
        
        while len(selected_individuals) < n_select and attempts < max_attempts:
            attempts += 1
            individual, index = self._tournament_selection_single(
                random_state, tournament_selection_n,
                tournament_selection_p, find_best
            )
            
            if ensure_unique:
                if index in selected_indices_set:
                    continue
                selected_indices_set.add(index)
            
            selected_individuals.append(individual)
            selected_indices.append(index)
        
        if len(selected_individuals) < n_select:
            remaining = n_select - len(selected_individuals)
            available_indices = [i for i in range(len(self.individuals)) 
                               if i not in selected_indices_set]
            
            if available_indices:
                additional_indices = random_state.choice(
                    available_indices, 
                    size=min(remaining, len(available_indices)),
                    replace=False
                )
                
                for idx in additional_indices:
                    selected_individuals.append(self.individuals[idx])
                    selected_indices.append(idx)
        
        return selected_individuals, selected_indices
    
    def tournament_selection_pair(self, random_state, tournament_selection_n,
                                  tournament_selection_p):
        """专门用于交叉的配对选择（语法糖）"""
        individuals, indices = self.tournament_selection(
            random_state, tournament_selection_n,
            tournament_selection_p, find_best=True,
            n_select=2, ensure_unique=True
        )
        
        return (individuals[0], individuals[1]), (indices[0], indices[1])

    # =========================================================================
    # 优化的个体替换（批量更新）
    # =========================================================================

    def _replace_individual(self, index: int, offspring: Union[Expression, ExpressionSet], 
                            raw_fitness: float, update_birth: bool = True, is_simplified: bool = False):
        """替换个体（带延迟更新优化）"""
        new_size = offspring.complexity
        
        self.individuals[index] = offspring
        self.fitnesses[index] = raw_fitness
        if update_birth:
            self.birthes[index] = 0
            self.birthes += 1
        self.complexitys[index] = new_size
        
        self.hall_of_fame.add(offspring, raw_fitness, is_simplified or (not self.should_simplify))

    # =========================================================================
    # 优化的进化循环
    # =========================================================================

    def evolve(self, X, y, seed, ncycles, crossover_probability, 
               tournament_selection_n, tournament_selection_p, 
               sample_weight: Optional[np.ndarray] = None) -> 'Population':
        """单个岛屿内部的稳态进化过程。"""
        max_temp, min_temp = (1.0, 0.0) if self.annealing else (1.0, 1.0)
        all_temperatures = np.linspace(max_temp, min_temp, ncycles) if ncycles > 1 else [max_temp]
        
        random_state = check_random_state(seed)
        n_evol_cycles = math.ceil(len(self) / tournament_selection_n)
        
        # 每次循环都根据优化概率决定是否重新初始化批量池
        if self.batching and random_state.uniform() < self.optimizer_probability:
            self.batch_pool = self._init_batch_pool(X, y, random_state)
        
        for cycle_num, temperature in enumerate(all_temperatures):
            X_batch, y_batch, sample_weight_batch = self._get_batch(
                X, y, random_state, sample_weight
            )
            
            for _ in range(n_evol_cycles):
                if random_state.uniform() < crossover_probability:
                    # 1. 选择父母并繁殖
                    (parent1, parent2), (idx1, idx2) = self.tournament_selection_pair(
                        random_state, tournament_selection_n, tournament_selection_p
                    )
                    
                    # 2. 执行交叉
                    offspring1, offspring2, crossover_accepted = self._crossover(
                        parent1, parent2
                    )
                    
                    # 3. 替换种群中的两个个体
                    if crossover_accepted:
                        replace_idx1, replace_idx2 = self.find_oldest_n(2)
                        
                        # 计算适应度
                        offspring1_fitness = offspring1.fitness(
                            X_batch, y_batch, sample_weight=sample_weight_batch
                        )
                        offspring2_fitness = offspring2.fitness(
                            X_batch, y_batch, sample_weight=sample_weight_batch
                        )
                        
                        # 记录日志
                        if self.enable_logging:
                            # 获取order（如果是ExpressionSet）
                            parent1_order = getattr(parent1, 'order', None)
                            parent2_order = getattr(parent2, 'order', None)
                            offspring1_order = getattr(offspring1, 'order', None)
                            offspring2_order = getattr(offspring2, 'order', None)
                            
                            self.logger.log_operation(
                                generation=self.niteration,
                                operation_type='crossover',
                                operation_name='subtree_crossover',
                                parent_fitness=self.fitnesses[idx1],
                                parent_complexity=self.complexitys[idx1],
                                offspring_fitness=offspring1_fitness,
                                offspring_complexity=offspring1.complexity,
                                accepted=True,
                                parent2_fitness=self.fitnesses[idx2],
                                parent2_complexity=self.complexitys[idx2],
                                offspring2_fitness=offspring2_fitness,
                                offspring2_complexity=offspring2.complexity,
                                batch_used=self.batching,
                                parent_order=parent1_order,
                                offspring_order=offspring1_order,
                                parent2_order=parent2_order,
                                offspring2_order=offspring2_order
                            )
                        
                        self._replace_individual(replace_idx1, offspring1, offspring1_fitness)
                        self._replace_individual(replace_idx2, offspring2, offspring2_fitness)
                else:
                    # 1. 选择父母并繁殖
                    parent1, parent1_index = self.tournament_selection(
                        random_state, tournament_selection_n,
                        tournament_selection_p, find_best=True, n_select=1
                    )
                    
                    # 2. 执行突变
                    offspring, mutation_accepted, offspring_fitness, mutation_name, accept_prob = self._mutation(
                        X_batch, y_batch, parent1, parent1_index, temperature, random_state, 
                        sample_weight=sample_weight_batch
                    )
                    
                    # 3. 记录日志（无论是否接受）
                    if self.enable_logging and mutation_name is not None:
                        # 获取order（如果是ExpressionSet）
                        parent_order = getattr(parent1, 'order', None)
                        offspring_order = getattr(offspring, 'order', None) if offspring is not None else None
                        
                        self.logger.log_operation(
                            generation=self.niteration,
                            operation_type='mutation',
                            operation_name=mutation_name,
                            parent_fitness=self.fitnesses[parent1_index],
                            parent_complexity=self.complexitys[parent1_index],
                            offspring_fitness=offspring_fitness if offspring is not None else self.fitnesses[parent1_index],
                            offspring_complexity=offspring.complexity if offspring is not None else self.complexitys[parent1_index],
                            accepted=mutation_accepted,
                            temperature=temperature,
                            probability=accept_prob,
                            batch_used=self.batching,
                            parent_order=parent_order,
                            offspring_order=offspring_order
                        )
                    
                    # 4. 替换种群中的一个个体
                    if mutation_accepted:
                        replace_idx = self.find_oldest_n(1)[0]
                        self._replace_individual(replace_idx, offspring, offspring_fitness)
        
        # 执行常量优化
        if self.generator.use_constants and self.should_optimize_constants:
            self._optimize_constants_in_population(
                X, y, random_state, self.optimizer_algorithm, 
                self.optimizer_probability, self.optimizer_nrestarts, 
                self.optimizer_iterations, sample_weight
            )
            self._update_constants_in_population(self.topn, random_state)
        
        if self.generator.use_aggregation and self.should_optimize_aggregations:
            self._optimize_aggregations_in_population(
                X, y, random_state, self.optimizer_probability, 
                self.optimizer_iterations, sample_weight
            )
        
        # 应用简化和常量突变
        if self.should_simplify:
            for i, ind in enumerate(self.individuals):
                simplified_ind, simplify_accepted = self.gpoperator.simplify(ind)
                X_batch, y_batch, sample_weight_batch = self._get_batch(
                    X, y, random_state, sample_weight
                )
                if simplify_accepted:
                    raw_fitness = simplified_ind.fitness(
                        X_batch, y_batch, sample_weight=sample_weight_batch
                    )
                else:
                    raw_fitness = None
                
                # 记录简化日志
                if self.enable_logging:
                    # 获取order（如果是ExpressionSet）
                    parent_order = getattr(ind, 'order', None)
                    offspring_order = getattr(simplified_ind, 'order', None)
                    
                    self.logger.log_operation(
                        generation=self.niteration,
                        operation_type='simplify',
                        operation_name='simplify_tree' if isinstance(ind, Expression) else 'simplify_set',
                        parent_fitness=self.fitnesses[i],
                        parent_complexity=self.complexitys[i],
                        offspring_fitness=raw_fitness if simplify_accepted else self.fitnesses[i],
                        offspring_complexity=simplified_ind.complexity if simplify_accepted else self.complexitys[i],
                        accepted=simplify_accepted,
                        batch_used=self.batching,
                        parent_order=parent_order,
                        offspring_order=offspring_order
                    )
                
                if simplify_accepted:
                    self._replace_individual(
                        i, simplified_ind, raw_fitness, update_birth=False, is_simplified=True
                    )
        
        self.niteration += 1
        return self
    
    def _mutation(self, X: np.ndarray, y: np.ndarray, parent: Expression | ExpressionSet, 
                  parent_idx: int, temperature: float, random_state: np.random.RandomState, 
                  sample_weight: Optional[np.ndarray] = None):
        """执行突变操作"""
        # 1. 记录父代适应度
        parent_fitness = self.fitnesses[parent_idx]
        
        # 2. Attempt to generate a valid mutated offspring
        max_attempts = 10
        for _ in range(max_attempts):
            offspring, mutation_succeeded, mutation_name = self.gpoperator.mutation(parent)
            if mutation_succeeded:
                break
        
        # 3. Handle special case for simplification
        if mutation_name in ['simplify_tree', 'simplify_set']:
            if self.batching:
                return offspring, mutation_succeeded, parent_fitness, mutation_name, 1.0

        if not mutation_succeeded:
            return None, False, None, mutation_name, 0.0

        # 4. Evaluate the offspring's fitness
        offspring_fitness = offspring.fitness(
            X, y, sample_weight=sample_weight
        )
        
        # 5. Calculate cost change (delta) for acceptance probability
        before_cost = self.fitnesses[parent_idx]
        after_cost = offspring_fitness
        
        delta = before_cost - after_cost if self.greater_is_better else after_cost - before_cost

        # 6. Calculate acceptance probability (probChange)
        probChange = 1.0

        if self.annealing and temperature > 0:
            probChange *= math.exp(-delta / (temperature * self.alpha))

        # 7. Decide whether to accept the mutation
        accepted = random_state.uniform() <= probChange
        
        return (
            offspring if accepted else None, 
            accepted, 
            offspring_fitness,
            mutation_name,
            probChange
        )
    
    def _crossover(self, parent1: Expression | ExpressionSet, parent2: Expression | ExpressionSet):
        max_attempts = 10
        for _ in range(max_attempts):
            offspring1, offspring2, crossover_succeeded = self.gpoperator.crossover(parent1, parent2)
            if crossover_succeeded:
                break
        if not crossover_succeeded:
            return None, None, False
        
        return offspring1, offspring2, True

    def _optimize_constants_in_population(self, X, y, random_state, optimizer_algorithm, 
                                          optimize_probability, optimizer_nrestarts, optimizer_iterations, 
                                          sample_weight: Optional[np.ndarray] = None):
        """Optimizes the constants in the population."""
        for i in range(len(self.individuals)):
            if random_state.uniform() < optimize_probability:
                parent, parent_index = self.individuals[i], i
                X_batch, y_batch, sample_weight_batch = self._get_batch(
                    X, y, random_state, sample_weight
                )
                offspring, optimize_accepted, raw_fitness = self.gpoperator.optimize_constants(
                    parent, X_batch, y_batch, sample_weight_batch, 
                    optimizer_algorithm, optimizer_nrestarts, optimizer_iterations
                )
                
                # 记录常量优化日志
                if self.enable_logging:
                    # 获取order（如果是ExpressionSet）
                    parent_order = getattr(parent, 'order', None)
                    offspring_order = getattr(offspring, 'order', None) if optimize_accepted else None
                    
                    self.logger.log_operation(
                        generation=self.niteration,
                        operation_type='optimize_constants',
                        operation_name='constant_optimization',
                        parent_fitness=self.fitnesses[i],
                        parent_complexity=self.complexitys[i],
                        offspring_fitness=raw_fitness,
                        offspring_complexity=offspring.complexity if optimize_accepted else self.complexitys[i],
                        accepted=optimize_accepted,
                        batch_used=self.batching,
                        parent_order=parent_order,
                        offspring_order=offspring_order
                    )
                
                if optimize_accepted:
                    self._replace_individual(i, offspring, raw_fitness, update_birth=False)
    
    def _update_constants_in_population(self, topn, random_state):
        optimize_constant_indexes = self.find_top_n(topn)
        individuals = [self.individuals[index] for index in optimize_constant_indexes]
        constant_values = []
        for individual in individuals:
            if isinstance(individual, Expression):
                constant_values += [
                    node.node_content.value for node in PostOrderIter(individual.tree) 
                        if isinstance(node.node_content, Constant)
                ]
            else:
                for expr in individual.expressions:
                    if expr is not None:
                        constant_values += [
                            node.node_content.value for node in PostOrderIter(expr.tree) 
                                if isinstance(node.node_content, Constant)
                        ]
        
        old_constants = self.generator.constants
        representative_values = self._find_top_n_frequent_floats(constant_values, len(old_constants))
        all_constants = [Constant(value=value) for value in representative_values]
        if len(all_constants) > 0:
            new_constants = list(
                random_state.choice(all_constants, size=min(len(old_constants), len(all_constants)))
            )
            if len(new_constants) == len(old_constants):
                self.generator._update_constants(new_constants)
            else:
                replace_positions = np.random.choice(len(old_constants), len(new_constants), replace=False)
                for i, pos in enumerate(replace_positions):
                    old_constants[pos] = new_constants[i]
                self.generator._update_constants(old_constants)
    
    def _find_top_n_frequent_floats(self, numbers, n, tolerance=0.01):
        """找出高频浮点数"""
        bins = defaultdict(list)
        if len(numbers) < n:
            return numbers
        
        for num in numbers:
            bin_id = math.floor(num / tolerance) * tolerance
            bins[bin_id].append(num)
        
        bin_frequencies = {bin_id: len(nums) for bin_id, nums in bins.items()}
        sorted_bins = sorted(bin_frequencies.items(), key=lambda item: item[1], reverse=True)
        top_n_bins = sorted_bins[:n]
        
        representative_values = []
        for bin_id, freq in top_n_bins:
            representative_value = sum(bins[bin_id]) / len(bins[bin_id])
            representative_values.append(representative_value)
        
        return representative_values

    def _optimize_aggregations_in_population(self, X, y, random_state, 
                                             optimize_probability, optimizer_iterations, 
                                             sample_weight: Optional[np.ndarray] = None):
        """Optimizes the aggregations in the population."""
        for i in range(len(self.individuals)):
            if random_state.uniform() < optimize_probability:
                parent, parent_index = self.individuals[i], i
                X_batch, y_batch, sample_weight_batch = self._get_batch(
                    X, y, random_state, sample_weight
                )
                offspring, optimize_accepted, raw_fitness = self.gpoperator.optimize_aggregations(
                    parent, X_batch, y_batch, sample_weight_batch, optimizer_iterations
                )
                
                # 记录聚合优化日志
                if self.enable_logging:
                    # 获取order（如果是ExpressionSet）
                    parent_order = getattr(parent, 'order', None)
                    offspring_order = getattr(offspring, 'order', None) if optimize_accepted else None
                    
                    self.logger.log_operation(
                        generation=self.niteration,
                        operation_type='optimize_aggregations',
                        operation_name='aggregation_optimization',
                        parent_fitness=self.fitnesses[i],
                        parent_complexity=self.complexitys[i],
                        offspring_fitness=raw_fitness,
                        offspring_complexity=offspring.complexity if optimize_accepted else self.complexitys[i],
                        accepted=optimize_accepted,
                        batch_used=self.batching,
                        parent_order=parent_order,
                        offspring_order=offspring_order
                    )
                
                if optimize_accepted:
                    self._replace_individual(i, offspring, raw_fitness, update_birth=False)

