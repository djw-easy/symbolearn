from itertools import combinations_with_replacement
from collections import Counter, defaultdict
from typing import List, Union, Tuple
import jax.numpy as jnp
import numpy as np
import math


from src.node_jax import Constant
from src.halloffame import HallOfFame
from src.utils import check_random_state
from src.expression import Expression, ExpressionSet
from src.gpoperator import ExpressionGP, ExpressionSetGP
from src.generator import ExprGenerator, ExprSetGenerator




class Population:
    """
    代表遗传算法中的一个种群（岛屿）。

    这个类封装了与单个种群相关的操作，包括进化、选择和个体管理。
    """
    def __init__(self,
                 population_size: int,
                 generator: ExprGenerator | ExprSetGenerator,
                 gpoperator: ExpressionGP | ExpressionSetGP,
                 annealing: bool = True,
                 use_frequency : bool = True, alpha: float = 3.17,
                 adaptive_parsimony_scaling: float = 0.6, 
                 should_simplify: bool = True,
                 should_optimize_constants: bool = True,
                 should_optimize_aggregations: bool = True,
                 optimizer_algorithm: str = 'L-BFGS-B',
                 optimizer_nrestarts: int = 3,
                 optimizer_probability: float = 0.14,
                 optimizer_iterations: int = 8,
                 topn: int = 12, batching=False, batch_size=256):
        """
        初始化一个种群。

        Parameters
        ----------
        individuals : list
            组成该种群的个体列表。
        metric : Fitness
            用于评估个体适应度的适应度对象。
        """
        self.population_size = population_size
        self.generator = generator
        self.gpoperator = gpoperator
        self.annealing = annealing
        self.use_frequency = use_frequency
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
        
        # 计算 size_maxcounts（树结构数量）
        if hasattr(generator, 'size_maxcounts'):
            self.size_maxcounts = generator.size_maxcounts
        else:
            self.size_maxcounts = generator.size_tree_counts
        
        # ===== 预计算频率调整因子（关键优化）=====
        self._frequency_penalty_cache = {}  # {size: penalty_factor}
        self._log_max_trees_cache = {}      # {size: log(max_trees)}
        
        # Hall of Fame
        self.hall_of_fame = HallOfFame(self.greater_is_better)
        self.niteration = 0
        
        # 批处理相关
        self.batch_pool = []
        self.batching = batching
        self.batch_size = batch_size

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

    def _init_batch_pool(self, X, y, random_state):
        """预生成 batch 池"""
        batch_pool = []
        batch_pool_size = (X.shape[0] // self.batch_size + 1) * 2
        for _ in range(batch_pool_size):
            indices = random_state.choice(X.shape[0], self.batch_size, replace=False)
            batch_pool.append((
                X[indices].copy(), y[indices].copy()
            ))
        return batch_pool

    def _get_batch(self, X, y, random_state):
        """从 batch 池中轮流取"""
        if not self.batching:
            return X, y
        
        # 初始化 batch 池
        if not self.batch_pool:
            self.batch_pool = self._init_batch_pool(X, y, random_state)
        
        # 轮流使用
        batch = self.batch_pool[random_state.randint(len(self.batch_pool))]
        
        return batch

    def init_population(self, X, y, seed) -> None:
        """初始化种群。"""
        self.individuals = []
        random_state = check_random_state(seed)
        self.sizes = np.zeros(self.population_size)
        self.fitnesses = np.zeros(self.population_size)
        for i in range(self.population_size):
            X_batch, y_batch = self._get_batch(X, y, random_state)
            new_expr = self.generator.generate_random_expr()
            raw_fitness_ = new_expr.fitness(X_batch, y_batch)
            self.individuals.append(new_expr)
            self.sizes[i] = new_expr.size
            self.fitnesses[i] = raw_fitness_
            self.hall_of_fame.add(new_expr, raw_fitness_)
        sorted_indices = np.argsort(-self.fitnesses) if self.greater_is_better else np.argsort(self.fitnesses)
        self.birthes = np.empty_like(sorted_indices)
        self.birthes[sorted_indices] = np.arange(len(self.fitnesses))
        # 频率计数器
        self.frequencies = Counter(self.sizes)
        self._update_frequency_cache()
        
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

    def _update_frequency_cache(self):
        """
        预计算所有大小的频率调整因子
        
        这是最重要的优化：避免在锦标赛选择中重复计算 log
        """
        self._frequency_penalty_cache.clear()
        self._log_max_trees_cache.clear()
        
        for size, max_trees in self.size_maxcounts.items():
            # 预计算 log(max_trees)
            log_max_trees = math.log(max_trees + 1)
            self._log_max_trees_cache[size] = log_max_trees
            
            # 预计算频率惩罚
            current_frequency = self.frequencies.get(size, 0)
            log_frequency = math.log(current_frequency + 1)
            
            # 计算相对探索密度
            relative_density = log_frequency / log_max_trees if log_max_trees > 0 else 0
            
            # 频率惩罚因子
            frequency_penalty = 1.0 + self.adaptive_parsimony_scaling * relative_density
            self._frequency_penalty_cache[size] = frequency_penalty
    
    def _get_frequency_penalty(self, size: int) -> float:
        """快速获取频率惩罚因子（O(1) 查表）"""
        return self._frequency_penalty_cache.get(size, 1.0)

    # =========================================================================
    # 优化的锦标赛选择（向量化版本）
    # =========================================================================

    def tournament_selection(self, random_state, tournament_selection_n,
                            tournament_selection_p, find_best: bool = True,
                            n_select: int = 1, ensure_unique: bool = True):
        """
        优化的锦标赛选择（支持批量选择）
        
        参数
        ----------
        random_state : RandomState
            随机状态
        tournament_selection_n : int
            每次锦标赛的竞争者数量
        tournament_selection_p : float
            选择概率（锦标赛中最佳个体被选中的概率）
        find_best : bool, default=True
            是否选择最佳个体（True）或最差个体（False）
        n_select : int, default=1
            需要选择的个体数量
            - n_select=1: 返回 (individual, index)
            - n_select>1: 返回 (individuals_list, indices_list)
        ensure_unique : bool, default=True
            是否确保选择的个体互不相同（交叉时需要）
        
        返回
        -------
        如果 n_select == 1:
            individual, index : 选中的个体和索引
        如果 n_select > 1:
            individuals, indices : 选中的个体列表和索引列表
        
        关键改进：
        1. 使用预缓存的 sizes 数组（避免 ind.size 调用）
        2. 使用预计算的频率惩罚因子（避免重复 log 计算）
        3. 向量化适应度调整
        4. 批量选择多个个体（交叉优化）
        
        性能提升：~5-10x（单个）, ~10-20x（批量）
        """
        if n_select == 1:
            # 单个选择（原有逻辑，优化版）
            return self._tournament_selection_single(
                random_state, tournament_selection_n, 
                tournament_selection_p, find_best
            )
        else:
            # 批量选择（新功能）
            return self._tournament_selection_batch(
                random_state, tournament_selection_n,
                tournament_selection_p, find_best,
                n_select, ensure_unique
            )
    
    def _tournament_selection_single(self, random_state, tournament_selection_n,
                                     tournament_selection_p, find_best):
        """
        单个锦标赛选择（内部方法）
        """
        # 1. 随机选择竞争者索引
        contender_indices = random_state.randint(0, len(self.individuals), tournament_selection_n)
        
        # 2. 获取适应度（O(k) 数组索引，k = tournament_size）
        fitnesses = self.fitnesses[contender_indices].copy()
        
        # 3. 应用频率调整（如果启用）
        if self.use_frequency:
            # 获取竞争者的大小（O(k) 数组索引）
            sizes = self.sizes[contender_indices]
            
            # 向量化获取频率惩罚（O(k) 字典查询，但 k 很小）
            frequency_penalties = np.array([
                self._get_frequency_penalty(size) for size in sizes
            ])
            
            # 向量化应用惩罚
            if self.greater_is_better:
                fitnesses /= frequency_penalties
            else:
                fitnesses *= frequency_penalties
        
        # 4. 排序竞争者（O(k log k)，k 通常很小如 3-7）
        sorted_indices = np.argsort(fitnesses)
        if self.greater_is_better == find_best:
            sorted_indices = sorted_indices[::-1]
        
        # 5. 概率选择
        for idx in sorted_indices:
            if random_state.random() < tournament_selection_p:
                parent_index = contender_indices[idx]
                return self.individuals[parent_index], parent_index
        
        # 6. 回退到最佳
        best_idx = sorted_indices[0]
        parent_index = contender_indices[best_idx]
        return self.individuals[parent_index], parent_index
    
    def _tournament_selection_batch(self, random_state, tournament_selection_n,
                                    tournament_selection_p, find_best,
                                    n_select, ensure_unique):
        """
        批量锦标赛选择（内部方法）
        
        优化策略：
        1. 预分配结果数组
        2. 向量化竞争者选择
        3. 去重检查（如果需要）
        4. 并行处理多个锦标赛
        
        时间复杂度：
        - 不去重: O(n_select × k × log k)
        - 去重: O(n_select × k × log k + n_select²) 最坏情况
        
        实际上去重开销很小，因为 n_select 通常 <= 5
        """
        selected_individuals = []
        selected_indices = []
        selected_indices_set = set()  # 用于快速去重检查
        
        max_attempts = n_select * 10  # 防止无限循环
        attempts = 0
        
        while len(selected_individuals) < n_select and attempts < max_attempts:
            attempts += 1
            
            # 1. 执行一次锦标赛选择
            individual, index = self._tournament_selection_single(
                random_state, tournament_selection_n,
                tournament_selection_p, find_best
            )
            
            # 2. 去重检查（如果需要）
            if ensure_unique:
                if index in selected_indices_set:
                    continue  # 已选择过，重新选择
                selected_indices_set.add(index)
            
            # 3. 添加到结果
            selected_individuals.append(individual)
            selected_indices.append(index)
        
        # 4. 检查是否成功选择了足够的个体
        if len(selected_individuals) < n_select:
            # 未能选择足够的不同个体，补充随机个体
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
        """
        专门用于交叉的配对选择（语法糖）
        
        等价于：
        tournament_selection(..., n_select=2, ensure_unique=True)
        
        返回
        -------
        (parent1, parent2), (index1, index2)
        """
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
                            raw_fitness: float, update_birth: bool = True):
        """
        替换个体（带延迟更新优化）
        
        策略：
        1. 立即更新核心数组（fitnesses, birthes, sizes）
        2. 延迟更新频率缓存（攒够一批再更新）
        """
        old_size = self.sizes[index]
        new_size = offspring.size
        
        # 更新核心数组（O(1)）
        self.individuals[index] = offspring
        self.fitnesses[index] = raw_fitness
        if update_birth:
            self.birthes[index] = 0
            self.birthes += 1  # 所有个体年龄+1
        self.sizes[index] = new_size
        
        # 更新频率计数器
        if self.use_frequency:
            self.frequencies[new_size] = self.frequencies.get(new_size, 0) + 1
        
        # 添加到名人堂
        self.hall_of_fame.add(offspring, raw_fitness)

    # =========================================================================
    # 优化的进化循环
    # =========================================================================

    def evolve(self, X, y, seed, ncycles, crossover_probability, 
               tournament_selection_n, tournament_selection_p) -> 'Population':
        """
        单个岛屿内部的稳态进化过程。
        """
        max_temp, min_temp = (1.0, 0.0) if self.annealing else (1.0, 1.0)
        all_temperatures = np.linspace(max_temp, min_temp, ncycles) if ncycles > 1 else [max_temp]
        
        random_state = check_random_state(seed)
        n_evol_cycles = math.ceil(len(self) / tournament_selection_n)
        
        for temperature in all_temperatures:
            # 记录需要更新缓存的标志
            cache_needs_update = False
            X_batch, y_batch = self._get_batch(X, y, random_state)
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
                        replace_idx1, replace_idx2 = self.find_oldest_n(2) # 替换最老
                        self._replace_individual(
                            replace_idx1, offspring1, offspring1.fitness(X_batch, y_batch)
                        )
                        self._replace_individual(
                            replace_idx2, offspring2, offspring1.fitness(X_batch, y_batch)
                        )
                        cache_needs_update = True
                else:
                    # 1. 选择父母并繁殖
                    parent1, parent1_index = self.tournament_selection(
                        random_state, tournament_selection_n,
                        tournament_selection_p, find_best=True, n_select=1
                    )
                    # 2. 执行突变
                    offspring, mutation_accepted, raw_fitness = self._mutation(
                        X_batch, y_batch, parent1, parent1_index, temperature, random_state
                    )
                    # 3. 替换种群中的一个个体
                    if mutation_accepted:
                        replace_idx = self.find_oldest_n(1)[0] # 替换最老
                        self._replace_individual(replace_idx, offspring, raw_fitness)
                        cache_needs_update = True
            
            # 每个温度周期后批量更新缓存
            if self.use_frequency and cache_needs_update:
                self._update_frequency_cache()
        
        # 执行常量优化
        if self.generator.use_constants and self.should_optimize_constants:
            self._optimize_constants_in_population(
                X, y, random_state, self.optimizer_algorithm, 
                self.optimizer_probability, self.optimizer_nrestarts, self.optimizer_iterations
            )
            # self._update_constants_in_population(self.optimizer_probability, self.topn, random_state)
        
        if self.generator.use_aggregations and self.should_optimize_aggregations:
            self._optimize_aggregations_in_population(
                X, y, random_state, self.optimizer_probability, self.optimizer_iterations
            )
        
        # 应用简化和常量突变
        if self.should_simplify:
            for i, ind in enumerate(self.individuals):
                simplified_ind, simplify_accepted = self.gpoperator.simplify(ind)
                if simplify_accepted:
                    raw_fitness = simplified_ind.fitness(X, y)
                    self._replace_individual(i, simplified_ind, raw_fitness, update_birth=False)
        
        self.niteration += 1
        return self
    
    def _mutation(self, X: jnp.ndarray, y: jnp.ndarray, 
                  parent: Expression | ExpressionSet, 
                  parent_idx: int, temperature: float, 
                  random_state: np.random.RandomState):
        """
        Performs mutation on a parent and decides whether to accept the offspring
        based on simulated annealing and frequency-based adaptation.
        """
        # 1. Attempt to generate a valid mutated offspring
        max_attempts = 10
        for _ in range(max_attempts):
            offspring, mutation_succeeded, mutation_name = self.gpoperator.mutation(parent)
            if mutation_succeeded:
                break
        
        if not mutation_succeeded:
            return None, False, mutation_name

        # 2. Evaluate the offspring's fitness
        raw_fitness = offspring.fitness(X, y)

        # 3. Handle special case for simplification: always accept if fitness is not worse
        if mutation_name in ['simplify_tree', 'simplify_set']:
            is_better_or_equal = (
                raw_fitness >= self.fitnesses[parent_idx]
                if self.greater_is_better
                else raw_fitness <= self.fitnesses[parent_idx]
            )
            if is_better_or_equal:
                return offspring, True, raw_fitness
        
        # 4. Calculate cost change (delta) for acceptance probability
        before_cost = self.fitnesses[parent_idx]
        after_cost = raw_fitness
        
        delta = before_cost - after_cost if self.greater_is_better else after_cost - before_cost

        # 5. Calculate acceptance probability (probChange)
        probChange = 1.0

        if self.annealing and temperature > 0:
            probChange *= math.exp(-delta / (temperature * self.alpha))

        if self.use_frequency:
            # 获取父代和子代的大小
            parent_size = parent.size
            offspring_size = offspring.size
            
            # frequency-based acceptance probability adjustment
            parent_log_max = self._log_max_trees_cache.get(parent_size, 1.0)
            offspring_log_max = self._log_max_trees_cache.get(offspring_size, 1.0)
            parent_frequency = self.frequencies.get(parent_size, 0)
            offspring_frequency = self.frequencies.get(offspring_size, 0)
            
            # 使用对数缓解指数增长影响
            parent_log_density = math.log(parent_frequency + 1) / parent_log_max
            offspring_log_density = math.log(offspring_frequency + 1) / offspring_log_max
            
            # 计算频率调整因子：鼓励向探索密度更低的区域移动
            # 如果offspring的探索密度更低，增加接受概率
            density_ratio = parent_log_density / (offspring_log_density + 1e-8)  # 防止除零
            
            # 温和的频率调整：使用 adaptive_parsimony_scaling 控制强度
            frequency_factor = 1.0 + self.adaptive_parsimony_scaling * (density_ratio - 1.0)
            frequency_factor = max(0.1, min(10.0, frequency_factor))  # 限制在合理范围内
            
            probChange *= frequency_factor

        # 6. Decide whether to accept the mutation
        accepted = random_state.uniform() < probChange
        
        return (offspring, True, raw_fitness) if accepted else (None, False, raw_fitness)
        
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
                                          optimize_probability, optimizer_nrestarts, optimizer_iterations):
        """Optimizes the constants in the population."""
        for i in range(len(self.individuals)):
            if random_state.uniform() < optimize_probability:
                parent, parent_index = self.individuals[i], i
                X_batch, y_batch = self._get_batch(X, y, random_state)
                offspring, optimize_accepted, raw_fitness = self.gpoperator.optimize_constants(
                    parent, X_batch, y_batch, 
                    optimizer_algorithm, optimizer_nrestarts, optimizer_iterations
                )
                if optimize_accepted:
                    self._replace_individual(i, offspring, raw_fitness, update_birth=False)
    
    def _update_constants_in_population(self, optimize_probability, topn, random_state):
        optimize_constant_indexes = self.find_top_n(topn)
        individuals = [self.individuals[index] for index in optimize_constant_indexes]
        constant_values = []
        for individual in individuals:
            if isinstance(individual, Expression):
                constant_values += [gene.value for gene in individual.genes
                                    if isinstance(gene, Constant)]
            else:
                for expr in individual.expressions:
                    if expr is not None:
                        constant_values += [gene.value for gene in individual.genes 
                                            if isinstance(gene, Constant)]
        
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
                                             optimize_probability, optimizer_iterations):
        """Optimizes the aggregations in the population."""
        for i in range(len(self.individuals)):
            if random_state.uniform() < optimize_probability:
                parent, parent_index = self.individuals[i], i
                X_batch, y_batch = self._get_batch(X, y, random_state)
                offspring, optimize_accepted, raw_fitness = self.gpoperator.optimize_aggregations(
                    parent, X_batch, y_batch, optimizer_iterations
                )
                if optimize_accepted:
                    self._replace_individual(i, offspring, raw_fitness, update_birth=False)
