import numbers
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import cpu_count
from numpy.random import Generator
from collections import defaultdict
from typing import Union, Tuple, Optional, Any, Literal
from numpy.lib.stride_tricks import sliding_window_view



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
    """Convert input to a new np.random.Generator instance.

    Parameters
    ----------
    random_state : None, int, Generator, or RandomState
        Input random state to convert.

    Returns
    -------
    np.random.Generator
        - None: Returns a new, unseeded Generator.
        - int: Returns a Generator seeded with the given value.
        - Generator: Returns the instance directly.
        - RandomState: Converts from legacy RandomState to new Generator.
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(random_state)
    if isinstance(random_state, Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        # Extract seed from legacy RandomState and create new Generator
        # This is a simplified conversion; may need more robust handling
        state = random_state.get_state()
        seed = state[1][0]
        return np.random.default_rng(seed)

    raise ValueError(f"Cannot convert {type(random_state)} to np.random.Generator")


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
    """Calculate a score for each equation based on loss and complexity.

    The score is defined as the negative log-likelihood ratio per unit complexity
    increase. Higher scores indicate that an equation achieves significantly
    better loss with only a modest increase in complexity.

    This version fixes the issue where scores could be negative when error
    did not improve.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'complexity' and 'error' columns.
    greater_is_better : bool
        Whether higher error values are better (True) or worse (False).

    Returns
    -------
    pd.Series
        Score for each equation in the same order as the input DataFrame.
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
                # Only award positive score when error increases (improves)
                if cur_error > last_error:
                    if last_error > 0.0:
                        cur_score = np.log(cur_error / last_error) / (cur_complexity - last_complexity)
                    else:  # Improvement from 0 or negative error is infinitely good
                        cur_score = np.inf
            else:  # lower is better
                # Only award positive score when error decreases (improves)
                if cur_error < last_error:
                    if cur_error > 0.0:
                        cur_score = -np.log(cur_error / last_error) / (cur_complexity - last_complexity)
                    else:  # Improvement to 0 error is infinitely good
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
        # Select the candidate with lowest error (highest accuracy)
        # Interpret 'error' based on greater_is_better flag
        if greater_is_better:
            # Higher error is better: select model with maximum error
            chosen_idx = hof_df['error'].idxmax()
        else:
            # Lower error is better: select model with minimum error
            chosen_idx = hof_df['error'].idxmin()
    elif model_selection == 'score':
        # Select the candidate with the highest score
        chosen_idx = hof_df['score'].idxmax()
    elif model_selection == 'best':
        # 'best' selects the highest-scoring model among those whose error
        # is at least 1.5x better than the most accurate model
        if greater_is_better:
            # Higher error is better: min_error is actually the maximum error
            min_error = hof_df['error'].max()
            # "1.5x better" means error should be larger: error >= min_error * 1.5
            threshold_error = min_error * 1.5

            if min_error == 0:  # theoretically should not happen in greater_is_better case
                filtered_df = hof_df[hof_df['error'] == 0]
            else:
                filtered_df = hof_df[hof_df['error'] >= threshold_error]
        else:
            # Lower error is better: min_error is the minimum error
            min_error = hof_df['error'].min()
            # "1.5x better" means error should be smaller: error <= min_error / 1.5
            threshold_error = min_error / 1.5
            if min_error == 0:
                filtered_df = hof_df[hof_df['error'] == 0]
            else:
                filtered_df = hof_df[hof_df['error'] <= threshold_error]

        if filtered_df.empty:
            # Fallback: select the model with the highest score
            chosen_idx = hof_df['score'].idxmax()
        else:
            # Select the highest-scoring model from the filtered set
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



def otsu_threshold(data, bins=256):
    """
    通用大津法阈值分割 - 支持任意范围的浮点数数据
    
    参数:
        data: 输入数据(numpy数组,任意维度,任意数值范围)
        bins: 直方图分箱数量,默认256
    
    返回:
        threshold: 最优阈值
        binary_data: 二值化后的数据(布尔数组)
    """
    # 展平数据
    data_flat = data.flatten()
    
    # 移除NaN和Inf
    data_flat = data_flat[np.isfinite(data_flat)]
    
    if len(data_flat) == 0:
        raise ValueError("数据中没有有效值")
    
    # 计算直方图
    hist, bin_edges = np.histogram(data_flat, bins=bins)
    hist = hist.astype(float)
    
    # 计算每个bin的中心值
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 总像素数
    total = np.sum(hist)
    
    # 归一化直方图
    hist_norm = hist / total
    
    # 初始化变量
    max_variance = 0
    optimal_threshold = bin_centers[0]
    
    # 遍历所有可能的阈值
    for i in range(1, bins):
        # 前景(类别0: 0~i)
        w0 = np.sum(hist_norm[:i])
        if w0 == 0 or w0 == 1:
            continue
        
        # 背景(类别1: i~end)
        w1 = 1 - w0
        
        # 前景加权平均值
        mu0 = np.sum(bin_centers[:i] * hist_norm[:i]) / w0
        
        # 背景加权平均值
        mu1 = np.sum(bin_centers[i:] * hist_norm[i:]) / w1
        
        # 类间方差
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        # 更新最大类间方差和对应阈值
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = bin_centers[i]
    
    # 根据阈值进行二值化
    binary_data = data > optimal_threshold
    
    return optimal_threshold, binary_data



def stratified_train_test_split(
    X: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    train_size: int,
    per_class: bool = True,
    balanced: bool = True,
    ignore_label: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    preserve_shape: bool = True,
    allow_insufficient: bool = False  # New parameter: allow insufficient samples in some classes
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    General stratified train/test split function.

    Automatically identifies data mode:
    1. Tabular Mode: When y is 1D (N,). X should be (N, D).
       Returns: X_train, X_test, y_train, y_test (subsets of samples)
    2. Map Mode: When y is 2D/3D (H, W). X should be (H, W, D) or (H, W).
       Behavior depends on `preserve_shape`:
       - True: Returns full-size mask maps, unselected positions are NaN.
       - False: Returns flattened arrays containing only selected valid pixels.

    Parameters
    ----------
    X : array-like
        Feature data.
        - Tabular Mode: shape (n_samples, n_features)
        - Map Mode: shape (H, W, D) or (H, W)
    y : array-like
        Target labels.
        - Tabular Mode: shape (n_samples,)
        - Map Mode: shape (H, W)
    train_size : int
        Controls the number of training samples. Interpretation depends on `per_class`:
        - per_class=True  (default): Number of samples per class to include in
          the training set. All remaining valid samples go to the test set.
        - per_class=False: Total number of training samples across all classes.
          How these are distributed across classes is controlled by `balanced`:
            * balanced=True  (default): Allocate samples proportionally to each
              class's size in y (stratified by class frequency). Remainders after
              flooring are distributed one-by-one to classes with the largest
              fractional parts.
            * balanced=False: Draw `train_size` samples uniformly at random from
              all valid samples, ignoring class boundaries entirely.
    per_class : bool, default=True
        Determines how `train_size` is interpreted:
        - True:  `train_size` is the per-class quota.
        - False: `train_size` is the total quota, split across all classes
                 according to the `balanced` flag.
    balanced : bool, default=True
        Only relevant when per_class=False.
        - True:  Proportional allocation — each class receives samples in
                 proportion to its frequency in y (stratified sampling).
        - False: Pure random sampling — `train_size` samples are drawn uniformly
                 at random from all valid samples without class constraints.
    ignore_label : int or float, optional
        Label value to ignore (e.g., background 0 in map mode).
        - None: All labels participate in sampling.
        - Value: Samples with this label value are excluded from sampling.
    random_state : int, optional
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    preserve_shape : bool, default=True
        Only affects Map Mode.
        - If True: Return features with shape (H, W, D) and y with shape (H, W),
          where invalid/unselected positions are filled with np.nan.
        - If False: Return only valid/selected pixels with shape (N_selected, D)
          and (N_selected,). Original dtypes are preserved (no NaN conversion).
    allow_insufficient : bool, default=False
        Only relevant when per_class=True.
        - False (default): Raise ValueError if any class has fewer samples than
          train_size.
        - True: If a class has fewer samples than train_size, use all available
          samples for that class. The class samples will appear in BOTH train
          and test sets (overlap allowed for insufficient classes).

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        - Tabular Mode: Split subsets (N_train, D), (N_test, D), etc.
        - Map Mode (preserve_shape=True): Arrays with same shape as input,
          unselected positions filled with np.nan (dtype=float).
        - Map Mode (preserve_shape=False): Flattened arrays containing only
          the selected train/test pixels.

    Raises
    ------
    ValueError
        If shapes mismatch, train_size is non-positive, or any class lacks
        sufficient samples (when allow_insufficient=False).

    Examples
    --------
    >>> # Per-class mode: 10 samples per class in training set
    >>> X_tr, X_te, y_tr, y_te = stratified_train_test_split(
    ...     X, y, train_size=10, per_class=True)

    >>> # Per-class mode with allow_insufficient: use all samples if < train_size
    >>> X_tr, X_te, y_tr, y_te = stratified_train_test_split(
    ...     X, y, train_size=50, per_class=True, allow_insufficient=True)

    >>> # Total mode, proportional: 100 samples total, proportional to class size
    >>> X_tr, X_te, y_tr, y_te = stratified_train_test_split(
    ...     X, y, train_size=100, per_class=False, balanced=True)
    """
    X = np.array(X)
    y = np.array(y)

    # --- 1. Mode Identification and Shape Validation ---
    is_map_mode = y.ndim >= 2

    if is_map_mode:
        # Map mode validation
        if X.ndim < 2:
            raise ValueError("In map mode, X must have at least 2 dimensions (H, W) or 3 (H, W, D).")

        # Check if spatial dimensions match (H, W)
        if X.shape[:2] != y.shape:
            raise ValueError(f"Spatial dimensions of X {X.shape[:2]} do not match shape of y {y.shape}.")

        original_shape = y.shape
        y_flat = y.ravel()
        n_pixels = len(y_flat)
        indices = np.arange(n_pixels)
    else:
        # Tabular mode validation
        if y.ndim != 1:
            raise ValueError("In tabular mode, y must be a 1D array (N,).")
        if len(X) != len(y):
            raise ValueError("In tabular mode, X and y must have the same number of samples.")

        y_flat = y
        indices = np.arange(len(y))

    if isinstance(train_size, float):
        if not (0 < train_size < 1):
            raise ValueError("train_size must be a positive integer or a float in (0, 1).")
        train_size = max(1, int(train_size * len(y)))
    elif train_size <= 0:
        raise ValueError("train_size must be a positive integer.")

    # --- 2. Filter Valid Samples (Handle ignore_label) ---
    if ignore_label is not None:
        valid_mask = y_flat != ignore_label
    else:
        valid_mask = np.ones_like(y_flat, dtype=bool)

    valid_indices = indices[valid_mask]
    valid_labels = y_flat[valid_mask]

    if len(valid_indices) == 0:
        raise ValueError("No valid samples found (all labels filtered by ignore_label).")

    # --- 3. Group Indices by Class ---
    class_to_indices = defaultdict(list)
    for idx, label in zip(valid_indices, valid_labels):
        class_to_indices[label].append(idx)

    n_classes = len(class_to_indices)

    # --- 4. Compute Per-Class Training Quota ---
    if per_class:
        # Track which classes have insufficient samples (for allow_insufficient)
        insufficient_classes = set()
        
        # Check for insufficient classes
        for label, idx_list in class_to_indices.items():
            if len(idx_list) < train_size:
                insufficient_classes.add(label)
        
        # Handle insufficient classes based on allow_insufficient flag
        if len(insufficient_classes) > 0:
            if not allow_insufficient:
                # Raise error for the first insufficient class
                label = list(insufficient_classes)[0]
                raise ValueError(
                    f"Class '{label}' has only {len(class_to_indices[label])} valid samples, "
                    f"but a training quota of {train_size} is required. "
                    f"(train_size={train_size}, per_class={per_class}). "
                    f"Set allow_insufficient=True to use all available samples."
                )
            else:
                # Warning about insufficient classes
                import warnings
                warnings.warn(
                    f"{len(insufficient_classes)} class(es) have fewer samples than train_size={train_size}. "
                    f"All available samples for these classes will be used in BOTH train and test sets. "
                    f"Affected classes: {sorted(insufficient_classes)}",
                    UserWarning
                )
        
        # Set quota: use train_size or all available samples (whichever is smaller)
        quota_per_class = {
            label: min(train_size, len(idxs)) 
            for label, idxs in class_to_indices.items()
        }
    else:
        # New behavior: distribute `train_size` total samples across all classes.
        if train_size > len(valid_indices):
            raise ValueError(
                f"train_size={train_size} exceeds the total number of valid samples "
                f"({len(valid_indices)})."
            )

        if balanced:
            # Proportional allocation: each class receives a quota proportional
            # to its share of the total valid pool (stratified by class frequency).
            class_sizes = {label: len(idxs) for label, idxs in class_to_indices.items()}
            total_valid = sum(class_sizes.values())

            # Compute raw (float) quotas then floor them.
            raw_quotas = {
                label: (size / total_valid) * train_size
                for label, size in class_sizes.items()
            }
            floored_quotas = {label: int(q) for label, q in raw_quotas.items()}
            allocated = sum(floored_quotas.values())
            remainder = train_size - allocated

            # Distribute remaining samples by largest fractional parts.
            if remainder > 0:
                fractional_parts = sorted(
                    class_to_indices.keys(),
                    key=lambda lbl: raw_quotas[lbl] - floored_quotas[lbl],
                    reverse=True
                )
                for label in fractional_parts[:remainder]:
                    floored_quotas[label] += 1

            quota_per_class = floored_quotas
        else:
            # Pure random sampling: draw `train_size` indices uniformly at random
            # from all valid samples, then count how many fall in each class.
            # This does NOT guarantee any per-class minimum.
            rng_random = np.random.RandomState(random_state)
            chosen = rng_random.choice(valid_indices, size=train_size, replace=False)
            chosen_set = set(chosen.tolist())

            # Re-route: override the normal quota path by directly assigning
            # train/test indices here and skipping Step 7's per-class loop.
            train_indices = chosen
            test_indices = np.array(
                [idx for idx in valid_indices if idx not in chosen_set]
            )

            # Jump straight to output construction (Step 8).
            quota_per_class = None  # sentinel: signals that indices are already set

    # --- 5. Check Sample Count per Class Against Computed Quotas ---
    # (Skipped in pure random mode where quota_per_class is None)
    # (Skipped when allow_insufficient=True as quotas are already adjusted)
    if quota_per_class is not None and not (per_class and allow_insufficient):
        for label, idx_list in class_to_indices.items():
            quota = quota_per_class[label]
            if len(idx_list) < quota:
                raise ValueError(
                    f"Class '{label}' has only {len(idx_list)} valid samples, "
                    f"but a training quota of {quota} is required. "
                    f"(train_size={train_size}, per_class={per_class}"
                    + (f", balanced={balanced}" if not per_class else "")
                    + ")"
                )

    # --- 6. Set Random State ---
    rng = np.random.RandomState(random_state)

    # --- 7. Perform Stratified Split Using Per-Class Quotas ---
    # Skipped in pure random mode (balanced=False) where indices are already assigned.
    if quota_per_class is not None:
        train_indices = []
        test_indices = []

        for label, idx_list in class_to_indices.items():
            idx_array = np.array(idx_list)
            if shuffle:
                rng.shuffle(idx_array)

            quota = quota_per_class[label]
            actual_count = len(idx_list)
            
            # Check if this is an insufficient class (when allow_insufficient=True)
            is_insufficient = (per_class and allow_insufficient and actual_count < train_size)
            
            if is_insufficient:
                # For insufficient classes: all samples go to BOTH train and test
                train_indices.extend(idx_array.tolist())
                test_indices.extend(idx_array.tolist())
            else:
                # Normal split: first `quota` to train, rest to test
                train_indices.extend(idx_array[:int(quota)].tolist())
                test_indices.extend(idx_array[int(quota):].tolist())

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

    # --- 8. Construct Output ---
    if not is_map_mode:
        # === Tabular Mode: Direct Index Split ===
        # preserve_shape is ignored in tabular mode as shape is already (N, D)
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y_flat[train_indices]
        y_test = y_flat[test_indices]
    else:
        # === Map Mode ===
        if preserve_shape:
            # --- Option 1: Full Shape with NaNs (Original Behavior) ---
            # Must convert to float to support NaN
            X_dtype = float if np.issubdtype(X.dtype, np.integer) else X.dtype
            y_dtype = float

            # Initialize full NaN arrays
            X_train = np.full(X.shape, np.nan, dtype=X_dtype)
            X_test = np.full(X.shape, np.nan, dtype=X_dtype)
            y_train = np.full(original_shape, np.nan, dtype=y_dtype)
            y_test = np.full(original_shape, np.nan, dtype=y_dtype)

            # Flattened views for assignment
            y_train_flat = y_train.ravel()
            y_test_flat = y_test.ravel()
            y_flat_orig = y_flat

            # Assign Training Labels
            y_train_flat[train_indices] = y_flat_orig[train_indices]

            # Assign Training Features (X)
            if X.ndim == 3:
                # Use advanced indexing: X_train[row, col, :] = ...
                train_coords = np.unravel_index(train_indices, original_shape)
                X_train[train_coords[0], train_coords[1], :] = X[train_coords[0], train_coords[1], :]

                test_coords = np.unravel_index(test_indices, original_shape)
                X_test[test_coords[0], test_coords[1], :] = X[test_coords[0], test_coords[1], :]
            else:
                # X is (H, W)
                X_train_flat = X_train.ravel()
                X_test_flat = X_test.ravel()
                X_flat = X.ravel()

                X_train_flat[train_indices] = X_flat[train_indices]
                X_test_flat[test_indices] = X_flat[test_indices]

            # Assign Test Labels
            y_test_flat[test_indices] = y_flat_orig[test_indices]
        else:
            # --- Option 2: Flattened Valid Pixels Only (No NaNs) ---
            # Preserves original dtype
            if X.ndim == 3:
                # Reshape (H, W, D) -> (H*W, D)
                H, W, D = X.shape
                X_flat = X.reshape(-1, D)
            else:
                # Reshape (H, W) -> (H*W,)
                X_flat = X.ravel()

            # Direct indexing on flattened arrays
            # Note: train_indices and test_indices refer to positions in the flattened array
            X_train = X_flat[train_indices]
            X_test = X_flat[test_indices]
            y_train = y_flat[train_indices]
            y_test = y_flat[test_indices]

    return X_train, X_test, y_train, y_test





"""
Optimized spatial patch extraction and aggregation for hyperspectral / remote-sensing data.

Key improvements over the original:
  1. Fixed the sliding_window_view axis ordering — the view shape for a (H, W, D) input
     is (H, W, D, K, K), so we must move the two trailing window axes to positions 3 & 4,
     producing (H, W, K, K, D).  The original moveaxis call was wrong.
  2. Vectorised batch aggregation — removed the inner per-pixel Python loop entirely;
     all pixels in a batch are reduced in a single NumPy call along axes (1, 2) instead
     of axis (0, 1) on individual patches.
  3. Consistent label dtype — y_valid_compact is always cast to float64 so that NaN
     sentinels can be stored regardless of preserve_shape mode.
  4. Robust tqdm guard — uses a try/except import instead of the fragile globals() check.
  5. 'range' method integrated into the dispatch table for cleaner code.
  6. Minor: copy=False on the compact label slice avoids an unnecessary allocation.
"""

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


def extract_and_aggregate_spatial(
    X_3d: np.ndarray,
    y_2d: Optional[np.ndarray] = None,
    window_size: int = 5,
    method: Literal['mean', 'max', 'min', 'sum', 'range', 'median', 'std'] = 'mean',
    padding_mode: str = 'reflect',
    ignore_label: Optional[Union[int, float]] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    skip_nan: bool = True,
    preserve_shape: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract spatial patches and aggregate features on-the-fly to minimise memory usage.

    Avoids materialising all patches simultaneously by processing valid pixels
    incrementally.  Ideal for large hyperspectral / remote-sensing datasets.

    Parameters
    ----------
    X_3d : np.ndarray
        Input 3-D array of shape (H, W, D):
          H — height (rows), W — width (columns), D — feature depth (spectral bands).
    y_2d : np.ndarray, optional
        2-D label / validity mask of shape (H, W).
        When provided, only pixels that are (a) not NaN AND (b) not equal to
        ``ignore_label`` (if given) will be processed.
        When None, ALL pixels are processed and only the feature array is returned.
    window_size : int, default 5
        Side length K of the square spatial window (K × K).  Must be odd.
    method : str, default 'mean'
        Aggregation method that reduces a (K, K, D) patch to a (D,) vector.
        Choices: 'mean', 'max', 'min', 'sum', 'range', 'median', 'std'.
    padding_mode : str, default 'reflect'
        Padding mode forwarded to ``np.pad``.
    ignore_label : int or float, optional
        Pixels whose label equals this value are excluded (requires y_2d).
    batch_size : int, optional
        Number of valid pixels to aggregate per iteration.
        None → process all valid pixels in one vectorised call.
    verbose : bool, default False
        Show a tqdm progress bar (requires tqdm to be installed).
    skip_nan : bool, default True
        Use nan-safe reduction functions (nanmean, nanmax, …).
        When False, NaN values propagate to the output.
    preserve_shape : bool, default True
        True  → return arrays of shape (H, W, D) and (H, W); invalid pixels = NaN.
        False → return only the N_valid valid pixels: (N_valid, D) and (N_valid,).

    Returns
    -------
    features : np.ndarray
        Aggregated feature array.
    y_valid : np.ndarray
        Label array (only when y_2d is provided).
    """

    # ------------------------------------------------------------------
    # 1.  Input validation
    # ------------------------------------------------------------------
    H, W, D = X_3d.shape

    if y_2d is not None:
        if y_2d.shape != (H, W):
            raise ValueError(
                f"y_2d shape {y_2d.shape} does not match X_3d spatial dims ({H}, {W})."
            )

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for symmetric padding.")

    valid_methods = ('mean', 'max', 'min', 'sum', 'range', 'median', 'std')
    if method not in valid_methods:
        raise ValueError(f"Unsupported method '{method}'. Choose from {valid_methods}.")

    # ------------------------------------------------------------------
    # 2.  Dtype promotion — output must be float to hold NaN sentinels
    # ------------------------------------------------------------------
    output_dtype = np.promote_types(X_3d.dtype, np.float32)

    # ------------------------------------------------------------------
    # 3.  Pad and build strided window view
    # ------------------------------------------------------------------
    pad = window_size // 2

    X_padded = np.pad(
        X_3d.astype(output_dtype, copy=False),
        pad_width=((pad, pad), (pad, pad), (0, 0)),
        mode=padding_mode,
    )

    # sliding_window_view on (H+2p, W+2p, D) with window axes (0,1)
    # produces shape: (H, W, D, K, K)
    windows_view = sliding_window_view(
        X_padded,
        window_shape=(window_size, window_size),
        axis=(0, 1),
    )

    # Verify and reorder to (H, W, K, K, D)
    # Expected raw shape: (H, W, D, K, K)
    assert windows_view.shape == (H, W, D, window_size, window_size), (
        f"Unexpected windows_view shape {windows_view.shape}; "
        f"expected ({H}, {W}, {D}, {window_size}, {window_size})."
    )
    # Move D from axis-2 to the last axis: (H, W, D, K, K) -> (H, W, K, K, D)
    windows_view = np.moveaxis(windows_view, 2, -1)
    # windows_view is now a *view* (no copy), shape (H, W, K, K, D)

    # ------------------------------------------------------------------
    # 4.  Determine valid pixel coordinates
    # ------------------------------------------------------------------
    return_labels = y_2d is not None

    if y_2d is None:
        # All pixels are valid
        valid_rows, valid_cols = np.indices((H, W))
        valid_rows = valid_rows.ravel()
        valid_cols = valid_cols.ravel()
        y_valid_full = None
    else:
        # Build boolean validity mask
        if np.issubdtype(y_2d.dtype, np.integer):
            # Integer arrays cannot contain NaN — all pixels initially valid
            valid_mask = np.ones((H, W), dtype=bool)
        else:
            valid_mask = ~np.isnan(y_2d)

        if ignore_label is not None:
            valid_mask &= (y_2d != ignore_label)

        valid_rows, valid_cols = np.where(valid_mask)

        # Allocate the full label output; invalid pixels stay NaN
        y_valid_full = np.full((H, W), np.nan, dtype=np.float64)
        if valid_rows.size > 0:
            y_valid_full[valid_rows, valid_cols] = (
                y_2d[valid_rows, valid_cols].astype(np.float64, copy=False)
            )

    N_valid = valid_rows.size

    # ------------------------------------------------------------------
    # 5.  Handle the empty-valid-set edge case
    # ------------------------------------------------------------------
    if preserve_shape:
        features_full = np.full((H, W, D), np.nan, dtype=output_dtype)
    else:
        features_full = None  # populated later

    if N_valid == 0:
        if preserve_shape:
            return (features_full, y_valid_full) if return_labels else features_full
        empty_feats = np.empty((0, D), dtype=output_dtype)
        empty_labels = np.empty((0,), dtype=np.float64)
        return (empty_feats, empty_labels) if return_labels else empty_feats

    # ------------------------------------------------------------------
    # 6.  Build vectorised aggregation function
    #
    #     Patches delivered to agg_func have shape (B, K, K, D).
    #     Reduction axes are (1, 2) — the two spatial window dimensions.
    #     The result has shape (B, D).
    # ------------------------------------------------------------------
    if skip_nan:
        _agg_map = {
            'mean':   lambda p: np.nanmean(p,   axis=(1, 2)),
            'max':    lambda p: np.nanmax(p,    axis=(1, 2)),
            'min':    lambda p: np.nanmin(p,    axis=(1, 2)),
            'sum':    lambda p: np.nansum(p,    axis=(1, 2)),
            'std':    lambda p: np.nanstd(p,    axis=(1, 2)),
            'median': lambda p: np.nanmedian(p, axis=(1, 2)),
            'range':  lambda p: (
                np.nanmax(p, axis=(1, 2)) - np.nanmin(p, axis=(1, 2))
            ),
        }
    else:
        _agg_map = {
            'mean':   lambda p: np.mean(p,   axis=(1, 2)),
            'max':    lambda p: np.max(p,    axis=(1, 2)),
            'min':    lambda p: np.min(p,    axis=(1, 2)),
            'sum':    lambda p: np.sum(p,    axis=(1, 2)),
            'std':    lambda p: np.std(p,    axis=(1, 2)),
            'median': lambda p: np.median(p, axis=(1, 2)),
            'range':  lambda p: (
                np.max(p, axis=(1, 2)) - np.min(p, axis=(1, 2))
            ),
        }

    agg_func = _agg_map[method]

    # ------------------------------------------------------------------
    # 7.  Compact feature buffer
    # ------------------------------------------------------------------
    features_compact = np.empty((N_valid, D), dtype=output_dtype)

    # ------------------------------------------------------------------
    # 8.  Batched vectorised aggregation (no Python loop over pixels)
    # ------------------------------------------------------------------
    effective_batch = N_valid if batch_size is None else batch_size

    indices = range(0, N_valid, effective_batch)
    if verbose and _TQDM_AVAILABLE:
        total_batches = (N_valid + effective_batch - 1) // effective_batch
        indices = _tqdm(indices, desc="Aggregating spatial features", total=total_batches)
    elif verbose:
        import warnings
        warnings.warn("tqdm is not installed; verbose progress bar unavailable.", stacklevel=2)

    for start in indices:
        end = min(start + effective_batch, N_valid)

        batch_rows = valid_rows[start:end]
        batch_cols = valid_cols[start:end]

        # Advanced index → shape (B, K, K, D)  — still no full materialisation
        batch_patches = windows_view[batch_rows, batch_cols]

        # Single vectorised reduction: (B, K, K, D) → (B, D)
        features_compact[start:end] = agg_func(batch_patches)

    # ------------------------------------------------------------------
    # 9.  Write results to the requested output layout
    # ------------------------------------------------------------------
    if preserve_shape:
        features_full[valid_rows, valid_cols] = features_compact
        return (features_full, y_valid_full) if return_labels else features_full

    # preserve_shape=False — return compact arrays
    if return_labels:
        # Cast to float64 for NaN consistency (same as preserve_shape=True path)
        y_valid_compact = y_2d[valid_rows, valid_cols].astype(np.float64, copy=False)
        return features_compact, y_valid_compact

    return features_compact


if __name__ == '__main__':
    """
    Simple unit tests for extract_and_aggregate_spatial.

    Tests cover:
    - Basic mean aggregation (known values, manually verifiable)
    - All aggregation methods produce correct output shapes
    - preserve_shape=False returns compact arrays
    - y_2d masking with ignore_label
    - NaN propagation when skip_nan=False
    - Edge case: no valid pixels
    - Boundary pixels (padding correctness)
    """


    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"

    results = []

    def check(name: str, condition: bool, detail: str = ""):
        status = PASS if condition else FAIL
        print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
        results.append(condition)

    print("=" * 60)
    print("Test 1: mean aggregation on constant array")
    print("=" * 60)
    # 3x3 image, 2 bands, all values = 1.0  → every window mean = 1.0
    X = np.ones((3, 3, 2), dtype=np.float32)
    out = extract_and_aggregate_spatial(X, window_size=3, method='mean', preserve_shape=True)
    check("output shape", out.shape == (3, 3, 2), str(out.shape))
    check("all values == 1.0", np.allclose(out, 1.0))

    print()
    print("=" * 60)
    print("Test 2: mean aggregation — centre pixel window manually verified")
    print("=" * 60)
    # 5x5 image, 1 band; values = row index (0..4)
    X = np.arange(25, dtype=np.float32).reshape(5, 5, 1)
    out = extract_and_aggregate_spatial(X, window_size=3, method='mean', preserve_shape=True)
    # Centre pixel (2,2): 3x3 window rows [1,2,3] cols [1,2,3]
    # values: 6,7,8, 11,12,13, 16,17,18  → mean = 108/9 = 12.0
    check("centre pixel mean", np.isclose(out[2, 2, 0], 12.0), f"got {out[2,2,0]:.4f}")

    print()
    print("=" * 60)
    print("Test 3: all methods — shape and dtype")
    print("=" * 60)
    X = np.random.rand(8, 8, 4).astype(np.float32)
    for m in ('mean', 'max', 'min', 'sum', 'range', 'median', 'std'):
        out = extract_and_aggregate_spatial(X, window_size=3, method=m, preserve_shape=True)
        check(f"method='{m}' shape", out.shape == (8, 8, 4), str(out.shape))
        check(f"method='{m}' dtype is float", np.issubdtype(out.dtype, np.floating))

    print()
    print("=" * 60)
    print("Test 4: preserve_shape=False returns compact arrays")
    print("=" * 60)
    X = np.ones((4, 4, 3), dtype=np.float32)
    y = np.array([
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ], dtype=np.float32)
    # ignore_label=0 → 10 valid pixels
    feats, labels = extract_and_aggregate_spatial(
        X, y_2d=y, window_size=3, method='mean',
        ignore_label=0, preserve_shape=False
    )
    n_valid = int((y == 1).sum())
    check("compact features shape", feats.shape == (n_valid, 3), str(feats.shape))
    check("compact labels shape",   labels.shape == (n_valid,),   str(labels.shape))
    check("all labels == 1.0",      np.all(labels == 1.0))
    check("all feature values == 1.0", np.allclose(feats, 1.0))

    print()
    print("=" * 60)
    print("Test 5: preserve_shape=True — invalid pixels are NaN")
    print("=" * 60)
    X = np.ones((4, 4, 2), dtype=np.float32)
    y = np.zeros((4, 4), dtype=np.float32)
    y[1, 1] = 1.0   # only one valid pixel
    feats, y_out = extract_and_aggregate_spatial(
        X, y_2d=y, window_size=3, method='mean',
        ignore_label=0, preserve_shape=True
    )
    check("valid pixel is not NaN", not np.isnan(feats[1, 1, 0]))
    check("invalid pixel is NaN",   np.isnan(feats[0, 0, 0]))
    check("y_out valid pixel == 1.0", y_out[1, 1] == 1.0)
    check("y_out invalid pixel is NaN", np.isnan(y_out[0, 0]))

    print()
    print("=" * 60)
    print("Test 6: skip_nan=False — NaN propagates into output")
    print("=" * 60)
    X = np.ones((5, 5, 2), dtype=np.float32)
    X[2, 2, 0] = np.nan   # inject NaN in band 0 at centre
    out = extract_and_aggregate_spatial(X, window_size=3, method='mean', skip_nan=False)
    # Pixels whose 3x3 window covers (2,2) will have NaN in band 0
    check("centre pixel band-0 is NaN", np.isnan(out[2, 2, 0]))
    check("centre pixel band-1 is not NaN", not np.isnan(out[2, 2, 1]))

    print()
    print("=" * 60)
    print("Test 7: skip_nan=True — NaN ignored in aggregation")
    print("=" * 60)
    X = np.ones((5, 5, 1), dtype=np.float32)
    X[2, 2, 0] = np.nan
    out = extract_and_aggregate_spatial(X, window_size=3, method='mean', skip_nan=True)
    # nanmean of (8×1.0 + nan) / 8 = 1.0
    check("centre pixel nanmean == 1.0", np.isclose(out[2, 2, 0], 1.0), f"got {out[2,2,0]:.4f}")

    print()
    print("=" * 60)
    print("Test 8: edge case — no valid pixels")
    print("=" * 60)
    X = np.ones((3, 3, 2), dtype=np.float32)
    y = np.zeros((3, 3), dtype=np.float32)   # all zeros → all ignored
    feats, labels = extract_and_aggregate_spatial(
        X, y_2d=y, window_size=3, method='mean',
        ignore_label=0, preserve_shape=False
    )
    check("empty features shape", feats.shape == (0, 2), str(feats.shape))
    check("empty labels shape",   labels.shape == (0,),   str(labels.shape))

    print()
    print("=" * 60)
    print("Test 9: batch_size produces same result as no batching")
    print("=" * 60)
    np.random.seed(42)
    X = np.random.rand(10, 10, 6).astype(np.float32)
    out_full  = extract_and_aggregate_spatial(X, window_size=5, method='mean')
    out_batch = extract_and_aggregate_spatial(X, window_size=5, method='mean', batch_size=7)
    check("batched == non-batched", np.allclose(out_full, out_batch, equal_nan=True))

    print()
    print("=" * 60)
    print("Test 10: range method = max - min")
    print("=" * 60)
    X = np.random.rand(6, 6, 3).astype(np.float32)
    out_range = extract_and_aggregate_spatial(X, window_size=3, method='range')
    out_max   = extract_and_aggregate_spatial(X, window_size=3, method='max')
    out_min   = extract_and_aggregate_spatial(X, window_size=3, method='min')
    check("range == max - min", np.allclose(out_range, out_max - out_min))

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    passed = sum(results)
    total  = len(results)
    color  = "\033[92m" if passed == total else "\033[91m"
    print(f"{color}Results: {passed}/{total} checks passed\033[0m")
    print("=" * 60)

