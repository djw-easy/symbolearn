from typing import List, Union, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class EvolutionLogEntry:
    """
    A single log entry recording one evolutionary operation.

    This dataclass captures all relevant information about a genetic programming
    operation (mutation, crossover, simplification, constant optimization) including
    the before/after state of the individuals and whether the offspring was
    accepted into the population.

    Fields
    ------
    generation : int
        The evolutionary generation number (0-indexed).
    timestamp : float
        Unix timestamp when this operation occurred.
    operation_type : str
        Category of operation performed. One of:
        - 'mutation': Single-parent genetic modification
        - 'crossover': Two-parent recombination
        - 'simplify': Algebraic simplification
        - 'optimize_constants': Numerical constant optimization
        - 'optimize_aggregations': Spatial/spectral aggregation optimization
    operation_name : str
        Specific name of the operation. Examples:
        - 'add_node', 'delete_node', 'mutate_constant', 'mutate_operator'
        - 'subtree_crossover', 'hoist_tree', 'swap_operands'
        - 'simplify_tree', 'constant_optimization'
    parent_fitness : float
        Fitness value of the parent individual(s).
    parent_complexity : int
        Complexity of the parent individual.
    offspring_fitness : float
        Fitness value of the offspring individual.
    offspring_complexity : int
        Complexity of the offspring individual.
    fitness_delta : float
        Change in fitness (offspring - parent).
    complexity_delta : int
        Change in complexity (offspring - parent).
    accepted : bool
        Whether the offspring was accepted into the population.
    temperature : float, optional
        Current annealing temperature (for simulated annealing).
    probability : float, optional
        Acceptance probability (for simulated annealing).
    parent2_fitness : float, optional
        Fitness of second parent (for crossover operations).
    parent2_complexity : int, optional
        Complexity of second parent (for crossover operations).
    offspring2_fitness : float, optional
        Fitness of second offspring (for crossover operations).
    offspring2_complexity : int, optional
        Complexity of second offspring (for crossover operations).
    batch_used : bool
        Whether mini-batch data was used for evaluation.
    parent_order : int, optional
        Number of expressions in parent (for ExpressionSet).
    offspring_order : int, optional
        Number of expressions in offspring (for ExpressionSet).
    order_delta : int, optional
        Change in expression count (for ExpressionSet).
    parent2_order : int, optional
        Expression count of second parent (for ExpressionSet crossover).
    offspring2_order : int, optional
        Expression count of second offspring (for ExpressionSet crossover).
    """

    generation: int                    # Evolutionary generation number
    timestamp: float                   # Unix timestamp of the operation
    operation_type: str                # Operation category: 'mutation', 'crossover', 'simplify', 'optimize_constants', 'optimize_aggregations'
    operation_name: str                # Specific operation name: 'add_node', 'subtree_crossover', etc.
    parent_fitness: float              # Parent fitness value
    parent_complexity: int             # Parent complexity (size)
    offspring_fitness: float           # Offspring fitness value
    offspring_complexity: int           # Offspring complexity
    fitness_delta: float              # Fitness change (offspring - parent)
    complexity_delta: int             # Complexity change
    accepted: bool                    # Whether offspring was accepted
    temperature: Optional[float]       # Annealing temperature (when applicable)
    probability: Optional[float]       # Acceptance probability (when applicable)
    parent2_fitness: Optional[float]   # Second parent fitness (for crossover)
    parent2_complexity: Optional[int]    # Second parent complexity (for crossover)
    offspring2_fitness: Optional[float] # Second offspring fitness (for crossover)
    offspring2_complexity: Optional[int] # Second offspring complexity (for crossover)
    batch_used: bool                  # Whether mini-batch data was used
    parent_order: Optional[int]        # Parent expression count (for ExpressionSet)
    offspring_order: Optional[int]     # Offspring expression count (for ExpressionSet)
    order_delta: Optional[int]         # Expression count change (for ExpressionSet)
    parent2_order: Optional[int]        # Second parent expression count (for ExpressionSet crossover)
    offspring2_order: Optional[int]    # Second offspring expression count (for ExpressionSet crossover)

    def to_dict(self) -> dict:
        """Convert the entry to a dictionary representation."""
        return asdict(self)


class EvolutionLogger:
    """
    Records detailed logs of genetic programming evolutionary operations.

    The EvolutionLogger captures every mutation, crossover, simplification,
    and optimization operation performed during evolution, enabling post-hoc
    analysis of the evolutionary process.

    This logger is designed to be lightweight when disabled (no overhead)
    and comprehensive when enabled, tracking:
    - All genetic operations with before/after state
    - Simulated annealing temperatures and acceptance probabilities
    - Multi-output ExpressionSet order changes
    - Batch data usage patterns

    Parameters
    ----------
    enabled : bool, default=True
        Whether logging is active. When False, all logging methods are no-ops.
    greater_is_better : bool, default=True
        Whether higher fitness values are better. This affects how fitness_delta
        is computed in the log entries.

    Attributes
    ----------
    logs : list of EvolutionLogEntry
        The recorded log entries.
    enabled : bool
        Whether logging is active.
    greater_is_better : bool
        Optimization direction flag.

    Methods
    -------
    log_operation(...)
        Record a single evolutionary operation.
    to_dataframe() -> pd.DataFrame
        Convert all logs to a pandas DataFrame.
    clear()
        Remove all logged entries and reset the timer.
    get_summary() -> dict
        Compute summary statistics from the logs.
    merge_logs(loggers) -> EvolutionLogger
        Merge multiple loggers into one.

    Examples
    --------
    >>> logger = EvolutionLogger(enabled=True, greater_is_better=False)
    >>> logger.log_operation(
    ...     generation=0,
    ...     operation_type='mutation',
    ...     operation_name='add_node',
    ...     parent_fitness=0.5,
    ...     parent_complexity=10,
    ...     offspring_fitness=0.3,
    ...     offspring_complexity=12,
    ...     accepted=True
    ... )
    >>> df = logger.to_dataframe()
    >>> print(df.shape)
    (1, 20)

    Notes
    -----
    The logger stores timestamps as relative offsets from the first logged
    operation, not absolute timestamps. This makes temporal analysis cleaner
    within a single run.

    See Also
    --------
    LogAnalyzer : Provides analysis methods for evolution logs.
    Population : Uses EvolutionLogger when enable_logging=True.
    """

    def __init__(self, enabled: bool = True, greater_is_better: bool = True):
        self.enabled = enabled
        self.greater_is_better = greater_is_better
        self.logs: List[EvolutionLogEntry] = []
        self._start_time = datetime.now().timestamp()
    
    def log_operation(self, 
                     generation: int,
                     operation_type: str,
                     operation_name: str,
                     parent_fitness: float,
                     parent_complexity: int,
                     offspring_fitness: float,
                     offspring_complexity: int,
                     accepted: bool,
                     temperature: Optional[float] = None,
                     probability: Optional[float] = None,
                     parent2_fitness: Optional[float] = None,
                     parent2_complexity: Optional[int] = None,
                     offspring2_fitness: Optional[float] = None,
                     offspring2_complexity: Optional[int] = None,
                     batch_used: bool = False,
                     parent_order: Optional[int] = None,
                     offspring_order: Optional[int] = None,
                     parent2_order: Optional[int] = None,
                     offspring2_order: Optional[int] = None):
        """
        记录一次进化操作
        
        Parameters
        ----------
        generation : int
            当前代数
        operation_type : str
            操作类型
        operation_name : str
            具体操作名称
        parent_fitness : float
            父代适应度
        parent_complexity : int
            父代复杂度
        offspring_fitness : float
            子代适应度
        offspring_complexity : int
            子代复杂度
        accepted : bool
            是否被接受
        temperature : float, optional
            当前温度
        probability : float, optional
            接受概率
        parent2_fitness : float, optional
            第二个父代适应度（交叉时）
        parent2_complexity : int, optional
            第二个父代复杂度（交叉时）
        offspring2_fitness : float, optional
            第二个子代适应度（交叉时）
        offspring2_complexity : int, optional
            第二个子代复杂度（交叉时）
        batch_used : bool, default=False
            是否使用了批量数据
        parent_order : int, optional
            父代order（ExpressionSet时）
        offspring_order : int, optional
            子代order（ExpressionSet时）
        parent2_order : int, optional
            第二个父代order（交叉且ExpressionSet时）
        offspring2_order : int, optional
            第二个子代order（交叉且ExpressionSet时）
        """
        if not self.enabled:
            return
        
        # 计算适应度变化（考虑优化方向）
        if self.greater_is_better:
            fitness_delta = offspring_fitness - parent_fitness
        else:
            fitness_delta = parent_fitness - offspring_fitness
        
        # 复杂度变化
        complexity_delta = offspring_complexity - parent_complexity
        
        # order变化（如果是ExpressionSet）
        order_delta = None
        if parent_order is not None and offspring_order is not None:
            order_delta = offspring_order - parent_order
        
        # 创建日志条目
        entry = EvolutionLogEntry(
            generation=generation,
            timestamp=datetime.now().timestamp() - self._start_time,
            operation_type=operation_type,
            operation_name=operation_name,
            parent_fitness=parent_fitness,
            parent_complexity=parent_complexity,
            offspring_fitness=offspring_fitness,
            offspring_complexity=offspring_complexity,
            fitness_delta=fitness_delta,
            complexity_delta=complexity_delta,
            accepted=accepted,
            temperature=temperature,
            probability=probability,
            parent2_fitness=parent2_fitness,
            parent2_complexity=parent2_complexity,
            offspring2_fitness=offspring2_fitness,
            offspring2_complexity=offspring2_complexity,
            batch_used=batch_used,
            parent_order=parent_order,
            offspring_order=offspring_order,
            order_delta=order_delta,
            parent2_order=parent2_order,
            offspring2_order=offspring2_order
        )
        
        self.logs.append(entry)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        将日志转换为DataFrame
        
        Returns
        -------
        pd.DataFrame
            包含所有日志条目的DataFrame
        """
        if not self.logs:
            return pd.DataFrame()
        
        return pd.DataFrame([entry.to_dict() for entry in self.logs])
    
    def clear(self):
        """清空日志"""
        self.logs.clear()
        self._start_time = datetime.now().timestamp()
    
    def get_summary(self) -> dict:
        """
        获取日志摘要统计
        
        Returns
        -------
        dict
            包含各种统计信息的字典
        """
        if not self.logs:
            return {}
        
        df = self.to_dataframe()
        
        summary = {
            'total_operations': len(df),
            'accepted_operations': df['accepted'].sum(),
            'acceptance_rate': df['accepted'].mean(),
            'operations_by_type': df['operation_type'].value_counts().to_dict(),
            'operations_by_name': df['operation_name'].value_counts().to_dict(),
            'avg_fitness_delta': df['fitness_delta'].mean(),
            'avg_complexity_delta': df['complexity_delta'].mean(),
            'avg_fitness_delta_accepted': df[df['accepted']]['fitness_delta'].mean(),
            'avg_complexity_delta_accepted': df[df['accepted']]['complexity_delta'].mean(),
            'total_runtime': df['timestamp'].max() if len(df) > 0 else 0
        }
        
        return summary
    
    @staticmethod
    def merge_logs(loggers: List['EvolutionLogger']) -> 'EvolutionLogger':
        """
        合并多个日志记录器
        
        Parameters
        ----------
        loggers : list of EvolutionLogger
            要合并的日志记录器列表
        
        Returns
        -------
        EvolutionLogger
            合并后的新日志记录器
        """
        if not loggers:
            return EvolutionLogger(enabled=False)
        
        merged = EvolutionLogger(
            enabled=True,
            greater_is_better=loggers[0].greater_is_better
        )
        
        # 合并所有日志条目
        for logger in loggers:
            if logger.enabled:
                merged.logs.extend(logger.logs)
        
        # 按时间戳排序
        merged.logs.sort(key=lambda x: x.timestamp)
        
        return merged


class LogAnalyzer:
    """
    Static utility class for analyzing evolution logs.

    LogAnalyzer provides a suite of static methods for examining the
    evolutionary process captured by EvolutionLogger. It can answer questions like:
    - Which mutation operators are most effective?
    - How does temperature affect acceptance rates?
    - What operations lead to the biggest fitness improvements?
    - How do order and complexity change during multi-output evolution?

    All methods accept a pandas DataFrame as the first argument and return
    various analysis results. The DataFrame is expected to have columns
    matching the fields of EvolutionLogEntry.

    Methods
    -------
    analyze_operation_effectiveness(df) -> pd.DataFrame
        Compute per-operator statistics (acceptance rate, fitness/complexity deltas).
    analyze_order_changes(df) -> dict
        Analyze ExpressionSet order changes during evolution.
    analyze_order_vs_complexity(df) -> dict
        Study the relationship between order and complexity in multi-output problems.
    analyze_temporal_trends(df, window_size=100) -> pd.DataFrame
        Compute rolling statistics over time using a sliding window.
    analyze_fitness_improvement_paths(df, min_improvement=None, threshold_percentile=75.0) -> pd.DataFrame
        Identify operations that lead to significant fitness improvements.
    analyze_complexity_vs_fitness(df) -> dict
        Examine the tradeoff between complexity and fitness.
    analyze_temperature_impact(df) -> pd.DataFrame
        Study how simulated annealing temperature affects acceptance.
    generate_full_report(df, output_file=None) -> str
        Produce a comprehensive text report of all analyses.

    Examples
    --------
    >>> df = logger.to_dataframe()
    >>> effectiveness = LogAnalyzer.analyze_operation_effectiveness(df)
    >>> print(effectiveness[['acceptance_rate', 'avg_fitness_delta_accepted']])
    >>> report = LogAnalyzer.generate_full_report(df, output_file='analysis.txt')

    Notes
    -----
    Many analysis methods return empty DataFrames or dictionaries when
    the input is empty or lacks the required columns. Check the return
    type and handle empty results appropriately.

    The analyze_fitness_improvement_paths method uses adaptive thresholding
    by default, computing the threshold as the 75th percentile of positive
    fitness deltas among accepted operations.
    """
    
    @staticmethod
    def analyze_operation_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
        """
        分析各种操作的有效性
        
        Parameters
        ----------
        df : pd.DataFrame
            日志DataFrame
        
        Returns
        -------
        pd.DataFrame
            操作有效性分析结果
        """
        if df.empty:
            return pd.DataFrame()
        
        # 按操作名称分组统计
        analysis = df.groupby('operation_name').agg({
            'accepted': ['count', 'sum', 'mean'],
            'fitness_delta': ['mean', 'std', 'min', 'max'],
            'complexity_delta': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # 重命名列
        analysis.columns = [
            'total_attempts', 'total_accepted', 'acceptance_rate',
            'avg_fitness_delta', 'std_fitness_delta', 'min_fitness_delta', 'max_fitness_delta',
            'avg_complexity_delta', 'std_complexity_delta', 'min_complexity_delta', 'max_complexity_delta'
        ]
        
        # 只统计被接受的操作
        accepted_df = df[df['accepted']].groupby('operation_name').agg({
            'fitness_delta': 'mean',
            'complexity_delta': 'mean'
        }).round(4)
        accepted_df.columns = ['avg_fitness_delta_accepted', 'avg_complexity_delta_accepted']
        
        analysis = analysis.join(accepted_df)
        
        # 如果有order数据，添加order统计
        if 'order_delta' in df.columns and df['order_delta'].notna().any():
            order_stats = df[df['accepted']].groupby('operation_name')['order_delta'].agg(['mean', 'std']).round(4)
            order_stats.columns = ['avg_order_delta_accepted', 'std_order_delta_accepted']
            analysis = analysis.join(order_stats)
        
        # 按接受率排序
        analysis = analysis.sort_values(['acceptance_rate', 'avg_fitness_delta_accepted'], ascending=False)
        
        return analysis
    
    @staticmethod
    def analyze_order_changes(df: pd.DataFrame) -> Dict:
        """
        分析ExpressionSet的order变化模式（新增）
        
        Parameters
        ----------
        df : pd.DataFrame
            日志DataFrame
        
        Returns
        -------
        dict
            order变化分析结果
        """
        if df.empty or 'order_delta' not in df.columns:
            return {'error': 'No order data available'}
        
        # 筛选有order数据的记录
        order_df = df[df['order_delta'].notna()].copy()
        
        if order_df.empty:
            return {'error': 'No order data available'}
        
        # 基本统计
        analysis = {
            'total_operations_with_order': len(order_df),
            'accepted_operations_with_order': order_df['accepted'].sum(),
            'acceptance_rate': order_df['accepted'].mean(),
            'avg_order_delta_all': order_df['order_delta'].mean(),
            'std_order_delta_all': order_df['order_delta'].std(),
            'avg_order_delta_accepted': order_df[order_df['accepted']]['order_delta'].mean(),
            'std_order_delta_accepted': order_df[order_df['accepted']]['order_delta'].std(),
        }
        
        # order变化分布
        order_df['order_change_type'] = order_df['order_delta'].apply(
            lambda x: 'increase' if x > 0 else ('decrease' if x < 0 else 'no_change')
        )
        
        change_dist = order_df.groupby('order_change_type').agg({
            'accepted': ['count', 'sum', 'mean'],
            'fitness_delta': 'mean'
        }).round(4)
        
        analysis['order_change_distribution'] = change_dist.to_dict('index')
        
        # order与适应度的关系
        accepted_order = order_df[order_df['accepted']]
        if len(accepted_order) > 1:
            correlation = accepted_order[['order_delta', 'fitness_delta']].corr().iloc[0, 1]
            analysis['order_fitness_correlation'] = float(correlation)
        else:
            analysis['order_fitness_correlation'] = None
        
        # 按操作类型分析order变化
        order_by_op = order_df[order_df['accepted']].groupby('operation_name').agg({
            'order_delta': ['mean', 'std', 'min', 'max'],
            'fitness_delta': 'mean'
        }).round(4)
        
        analysis['order_by_operation'] = order_by_op.to_dict('index')
        
        return analysis
    
    @staticmethod
    def analyze_order_vs_complexity(df: pd.DataFrame) -> Dict:
        """
        分析order与complexity的关系（新增）
        
        Parameters
        ----------
        df : pd.DataFrame
            日志DataFrame
        
        Returns
        -------
        dict
            order与complexity关系分析
        """
        if df.empty or 'order_delta' not in df.columns:
            return {'error': 'No order data available'}
        
        order_df = df[(df['order_delta'].notna()) & (df['accepted'])].copy()
        
        if order_df.empty or len(order_df) < 2:
            return {'error': 'Insufficient order data'}
        
        # 计算相关系数
        correlations = {
            'order_complexity_correlation': order_df[['order_delta', 'complexity_delta']].corr().iloc[0, 1],
            'order_fitness_correlation': order_df[['order_delta', 'fitness_delta']].corr().iloc[0, 1],
        }
        
        # 按order变化分组分析complexity和fitness
        order_df['order_bin'] = pd.cut(
            order_df['order_delta'],
            bins=[-np.inf, -1, 0, 1, np.inf],
            labels=['decrease', 'no_change_neg', 'no_change_pos', 'increase']
        )
        
        binned_analysis = order_df.groupby('order_bin', observed=True).agg({
            'complexity_delta': ['mean', 'std'],
            'fitness_delta': ['mean', 'std'],
            'accepted': 'count'
        }).round(4)
        
        correlations['binned_analysis'] = binned_analysis.to_dict('index')
        
        return correlations
    
    @staticmethod
    def analyze_temporal_trends(df: pd.DataFrame, window_size: int = 100) -> pd.DataFrame:
        """
        分析时间趋势（使用滑动窗口）
        
        Parameters
        ----------
        df : pd.DataFrame
            日志DataFrame
        window_size : int, default=100
            滑动窗口大小
        
        Returns
        -------
        pd.DataFrame
            时间趋势分析结果
        """
        if df.empty or len(df) < window_size:
            return pd.DataFrame()
        
        df = df.sort_values('timestamp').copy()
        
        # 计算滑动窗口统计
        df['acceptance_rate_rolling'] = df['accepted'].rolling(window=window_size).mean()
        df['avg_fitness_delta_rolling'] = df['fitness_delta'].rolling(window=window_size).mean()
        df['avg_complexity_delta_rolling'] = df['complexity_delta'].rolling(window=window_size).mean()
        
        return df[['generation', 'timestamp', 
                  'acceptance_rate_rolling', 
                  'avg_fitness_delta_rolling', 
                  'avg_complexity_delta_rolling']].dropna()
    
    @staticmethod
    def analyze_fitness_improvement_paths(df: pd.DataFrame, min_improvement: Optional[float] = None, 
                                         threshold_percentile: float = 75.0) -> pd.DataFrame:
        """
        分析导致适应度显著提升的操作路径
        
        Parameters
        ----------
        df : pd.DataFrame
            日志DataFrame
        min_improvement : float, optional
            最小显著提升阈值。如果为None，则自动根据数据计算
        threshold_percentile : float, default=75.0
            当min_improvement为None时，使用此百分位数作为阈值
            例如：75.0表示fitness_delta超过75%分位数的操作被视为显著提升
        
        Returns
        -------
        pd.DataFrame
            显著提升操作的统计
        
        Notes
        -----
        自适应阈值计算策略：
        1. 如果指定了min_improvement，直接使用
        2. 否则，计算已接受操作中fitness_delta的指定百分位数
        3. 如果所有fitness_delta都为负（适应度下降），使用接近0的小正值
        """
        if df.empty:
            return pd.DataFrame()
        
        # 筛选已接受的操作
        accepted_df = df[df['accepted']].copy()
        
        if accepted_df.empty:
            return pd.DataFrame()
        
        # 计算自适应阈值
        if min_improvement is None:
            fitness_deltas = accepted_df['fitness_delta']
            
            # 策略1：如果有正的fitness_delta，使用百分位数
            if (fitness_deltas > 0).any():
                positive_deltas = fitness_deltas[fitness_deltas > 0]
                min_improvement = np.percentile(positive_deltas, threshold_percentile)
            else:
                # 策略2：如果所有都是负值或0，使用一个接近最大值的阈值
                max_delta = fitness_deltas.max()
                if max_delta <= 0:
                    # 所有都是负值，使用最大值的90%（最接近0的10%）
                    min_improvement = max_delta * 0.9
                else:
                    # 有0值，使用一个很小的正值
                    min_improvement = 1e-6
            
            # 确保阈值不会太小
            if abs(min_improvement) < 1e-10:
                min_improvement = 1e-6
        
        # 筛选出接受且有显著提升的操作
        significant_df = accepted_df[accepted_df['fitness_delta'] > min_improvement].copy()
        
        if significant_df.empty:
            return pd.DataFrame()
        
        # 统计各操作类型的贡献
        analysis = significant_df.groupby('operation_name').agg({
            'fitness_delta': ['count', 'sum', 'mean', 'max'],
            'complexity_delta': 'mean'
        }).round(4)
        
        analysis.columns = [
            'count', 'total_improvement', 'avg_improvement', 'max_improvement',
            'avg_complexity_change'
        ]
        
        # 按平均改进量排序
        analysis = analysis.sort_values('avg_improvement', ascending=False)
        
        # 添加使用的阈值信息作为DataFrame的属性（可选）
        analysis.attrs['threshold_used'] = min_improvement
        analysis.attrs['threshold_percentile'] = threshold_percentile
        
        return analysis
    
    @staticmethod
    def analyze_complexity_vs_fitness(df: pd.DataFrame) -> Dict:
        """
        分析复杂度与适应度的关系
        
        Parameters
        ----------
        df : pd.DataFrame
            日志DataFrame
        
        Returns
        -------
        dict
            相关性分析结果
        """
        if df.empty:
            return {}
        
        accepted_df = df[df['accepted']].copy()
        
        if accepted_df.empty:
            return {}
        
        # 计算相关系数
        correlation = accepted_df[['fitness_delta', 'complexity_delta']].corr().iloc[0, 1]
        
        # 分析不同复杂度变化区间的适应度变化
        accepted_df['complexity_bin'] = pd.cut(
            accepted_df['complexity_delta'],
            bins=[-np.inf, -5, -1, 0, 1, 5, np.inf],
            labels=['large_decrease', 'small_decrease', 'no_change_neg', 
                   'no_change_pos', 'small_increase', 'large_increase']
        )
        
        binned_analysis = accepted_df.groupby('complexity_bin', observed=True)['fitness_delta'].agg([
            'count', 'mean', 'std'
        ]).round(4)
        
        return {
            'correlation': float(correlation),
            'binned_analysis': binned_analysis.to_dict('index')
        }
    
    @staticmethod
    def analyze_temperature_impact(df: pd.DataFrame) -> pd.DataFrame:
        """
        分析温度对接受率的影响（退火时）
        
        Parameters
        ----------
        df : pd.DataFrame
            日志DataFrame
        
        Returns
        -------
        pd.DataFrame
            温度影响分析结果
        """
        if df.empty or 'temperature' not in df.columns:
            return pd.DataFrame()
        
        # 筛选有温度记录的数据
        temp_df = df[df['temperature'].notna()].copy()
        
        if temp_df.empty:
            return pd.DataFrame()
        
        # 按温度区间分组
        temp_df['temp_bin'] = pd.cut(
            temp_df['temperature'],
            bins=10,
            labels=[f'bin_{i}' for i in range(10)]
        )
        
        analysis = temp_df.groupby('temp_bin', observed=True).agg({
            'accepted': ['count', 'mean'],
            'fitness_delta': 'mean',
            'temperature': 'mean'
        }).round(4)
        
        analysis.columns = ['count', 'acceptance_rate', 'avg_fitness_delta', 'avg_temperature']
        
        return analysis
    
    @staticmethod
    def generate_full_report(df: pd.DataFrame, output_file: Optional[str] = None) -> str:
        """
        生成完整的分析报告
        
        Parameters
        ----------
        df : pd.DataFrame
            日志DataFrame
        output_file : str, optional
            输出文件路径
        
        Returns
        -------
        str
            报告文本
        """
        if df.empty:
            return "No data to analyze."
        
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("进化过程日志分析报告".center(100))
        report_lines.append("=" * 100)
        report_lines.append("")
        
        # 1. 基本统计
        report_lines.append("【1. 基本统计】")
        report_lines.append(f"  总操作数: {len(df)}")
        report_lines.append(f"  接受操作数: {df['accepted'].sum()}")
        report_lines.append(f"  总体接受率: {df['accepted'].mean():.2%}")
        report_lines.append(f"  平均适应度变化: {df['fitness_delta'].mean():.6f}")
        report_lines.append(f"  平均复杂度变化: {df['complexity_delta'].mean():.2f}")
        
        # 添加order统计（如果有）
        if 'order_delta' in df.columns and df['order_delta'].notna().any():
            report_lines.append(f"  平均order变化: {df['order_delta'].mean():.4f}")
        
        report_lines.append("")
        
        # 2. 操作类型统计
        report_lines.append("【2. 操作类型统计】")
        op_type_stats = df['operation_type'].value_counts()
        for op_type, count in op_type_stats.items():
            percentage = count / len(df) * 100
            report_lines.append(f"  {op_type}: {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # 3. 操作有效性分析（精简版）
        report_lines.append("【3. 操作有效性分析（Top 15）】")
        effectiveness = LogAnalyzer.analyze_operation_effectiveness(df).head(15)
        
        # 只显示关键列
        key_columns = ['total_attempts', 'acceptance_rate', 
                      'avg_fitness_delta_accepted', 'avg_complexity_delta_accepted']
        
        # 如果有order数据，添加order列
        if 'avg_order_delta_accepted' in effectiveness.columns:
            key_columns.append('avg_order_delta_accepted')
        
        effectiveness_display = effectiveness[key_columns]
        
        # 格式化输出
        if 'avg_order_delta_accepted' in key_columns:
            report_lines.append(f"{'Operation':<20}  {'Attempts':>8}  {'Accept_rate':>13}  {'Fitness_change':>20}  {'Complexity_change':>17}  {'Order_change':>12}")
            report_lines.append("-" * 100)
        else:
            report_lines.append(f"{'Operation':<20}  {'Attempts':>8}  {'Accept_rate':>13}  {'Fitness_change':>20}  {'Complexity_change':>17}")
            report_lines.append("-" * 86)
        
        for op_name, row in effectiveness_display.iterrows():
            line = f"{op_name:<20}  {int(row['total_attempts']):>8}  {row['acceptance_rate']:>13.1%}  "
            line += f"{row['avg_fitness_delta_accepted']:>20.4f}  {row['avg_complexity_delta_accepted']:>17.2f}"
            
            if 'avg_order_delta_accepted' in key_columns:
                line += f"  {row['avg_order_delta_accepted']:>12.3f}"
            
            report_lines.append(line)
        
        report_lines.append("")
        
        # 4. 显著改进操作分析（精简版）
        report_lines.append("【4. 显著适应度提升操作（Top 10）】")
        improvements = LogAnalyzer.analyze_fitness_improvement_paths(df)
        
        if not improvements.empty:
            report_lines.append(f"{'Operation':<20} {'Changes':>8} {'Total_changes':>15} {'Average_change':>15} {'Max_change':>13}")
            report_lines.append("-" * 75)
            
            for op_name, row in improvements.head(10).iterrows():
                report_lines.append(
                    f"{op_name:<20} {int(row['count']):>8} "
                    f"{row['total_improvement']:>15.4f} "
                    f"{row['avg_improvement']:>15.4f} "
                    f"{row['max_improvement']:>13.4f}"
                )
        else:
            report_lines.append("  无显著提升操作")
        
        report_lines.append("")
        report_lines.append("=" * 100)
        
        report = "\n".join(report_lines)
        
        # 保存到文件（如果指定）
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
