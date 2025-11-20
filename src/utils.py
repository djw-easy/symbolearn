import numbers
import numpy as np
import pandas as pd
import numpy.ma as ma
from joblib import cpu_count
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from numpy.random import Generator, PCG64, SeedSequence


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def check_random_generator(random_state):
    """
    将输入转换为一个新的 np.random.Generator 实例。
    - None: 返回一个新的、无种子的 Generator。
    - int: 返回一个以此为种子的 Generator。
    - Generator: 直接返回该实例。
    - RandomState: (可选) 从旧版转换为新版。
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(random_state)
    if isinstance(random_state, Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        # 从旧版提取种子信息并创建新版
        # 这是一个简化的转换，可能需要更鲁棒的处理
        state = random_state.get_state()
        seed = state[1][0] 
        return np.random.default_rng(seed)
    
    raise ValueError(f"无法将 {type(random_state)} 转换为 np.random.Generator")


def poisson_sample(lambda_val: float, random_state: np.random.RandomState) -> int:
    """
    Generates a Poisson-distributed random number using Knuth's algorithm.

    Args:
        lambda_val (float): The mean (λ) of the Poisson distribution.

    Returns:
        int: A random integer sampled from the Poisson distribution.
    """
    k = 0
    p = 1.0
    L = np.exp(-lambda_val)

    while p > L:
        k += 1
        p *= random_state.uniform()
    
    return k - 1


def _calculate_scores(df: pd.DataFrame, greater_is_better: bool) -> pd.Series:
    """根据损失和复杂度为每个方程计算分数。

    分数被定义为对数损失相对于复杂度的负导数。
    更高的分数意味着方程在略微增加复杂度的情况下，获得了显著更好的损失。
    此版本修复了当误差未改善时分数可能为负的问题。
    """
    df_sorted = df.sort_values('complexity').reset_index()
    
    scores = np.zeros(df_sorted.shape[0])
    last_error = None
    last_complexity = 0

    for _, row in df_sorted.iterrows():
        cur_error = row["error"]
        cur_complexity = row["complexity"]
        cur_score = 0.0

        if last_error is not None and cur_complexity > last_complexity:
            if greater_is_better:
                # 仅当误差增加（改善）时，才计算正分
                if cur_error > last_error:
                    if last_error > 0.0:
                        cur_score = np.log(cur_error / last_error) / (cur_complexity - last_complexity)
                    else:  # 从0或负误差改善是无限好的
                        cur_score = np.inf
            else:  # 越小越好
                # 仅当误差减少（改善）时，才计算正分
                if cur_error < last_error:
                    if cur_error > 0.0:
                        cur_score = -np.log(cur_error / last_error) / (cur_complexity - last_complexity)
                    else:  # 改善到0误差是无限好的
                        cur_score = np.inf
        
        scores[row['index']] = cur_score
        last_error = cur_error
        last_complexity = cur_complexity
    
    return scores


def _idx_model_selection(hof_df: pd.DataFrame, model_selection: str, greater_is_better: bool):
    """Select an expression and return its index."""

    # We must default to "accuracy" if no score column is present (like in the case of linear loss_scale)
    model_selection = model_selection if "score" in hof_df.columns else "accuracy"
    
    if model_selection == 'accuracy':
        # 选择损失最低（准确度最高）的候选模型
        # 根据 greater_is_better 判断 error 越大越好还是越小越好
        if greater_is_better:
            # error 越大越好，选择 error 最大的模型
            chosen_idx = hof_df['error'].idxmax()
        else:
            # error 越小越好，选择 error 最小的模型
            chosen_idx = hof_df['error'].idxmin()
    elif model_selection == 'score':
        # 选择分数最高的候选模型
        chosen_idx = hof_df['score'].idxmax()
    elif model_selection == 'best':
        # 'best' 选择损失比最准确模型至少好 1.5 倍的表达式中分数最高的候选模型。
        if greater_is_better:
            # 如果 error 越大越好，则 min_error 实际上是最大的 error
            min_error = hof_df['error'].max()
            # “好 1.5 倍”意味着 error 应该更大，即 error >= min_error * 1.5
            # 但这里是筛选，所以是 error >= min_error * 1.5
            threshold_error = min_error * 1.5
            
            if min_error == 0: # 理论上 greater_is_better 情况下 min_error 不会是 0
                filtered_df = hof_df[hof_df['error'] == 0]
            else:
                filtered_df = hof_df[hof_df['error'] >= threshold_error]
        else:
            # 如果 error 越小越好，则 min_error 是最小的 error
            min_error = hof_df['error'].min()
            # “好 1.5 倍”意味着 error 应该更小，即 error <= min_error / 1.5
            threshold_error = min_error / 1.5
            if min_error == 0:
                filtered_df = hof_df[hof_df['error'] == 0]
            else:
                filtered_df = hof_df[hof_df['error'] <= threshold_error]

        if filtered_df.empty:
            # 如果没有符合条件的模型，则退回到选择分数最高的模型
            chosen_idx = hof_df['score'].idxmax()
        else:
            # 在筛选出的模型中选择分数最高的
            chosen_idx = filtered_df['score'].idxmax()
    else:
        raise ValueError(f"Invalid model_selection strategy: {model_selection}. "
                            f"Choose from 'accuracy', 'best', or 'score'.")

    return chosen_idx


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()






