import numbers
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import cpu_count
from sklearn.preprocessing import StandardScaler
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


def standardize_spatial_image_from_training(
    X: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    output_dtype=np.float32,
):
    """Standardize a spatial feature cube using training pixels only.

    Parameters
    ----------
    X : ndarray of shape (H, W, D)
        Raw spatial feature cube.
    train_mask : ndarray of shape (H, W)
        Boolean mask identifying pixels allowed to fit the scaler.
    valid_mask : ndarray of shape (H, W), optional
        Pixels to transform in the returned cube. By default, pixels whose
        complete feature vector is finite are considered valid.
    output_dtype : numpy dtype, default=np.float32
        Data type of the transformed cube.

    Returns
    -------
    X_scaled : ndarray of shape (H, W, D)
        Standardized cube with NaN outside ``valid_mask``.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted training-only scaler.
    """
    X = np.asarray(X)
    train_mask = np.asarray(train_mask, dtype=bool)

    if X.ndim != 3:
        raise ValueError(f'X must have shape (H, W, D), got ndim={X.ndim}.')
    if train_mask.shape != X.shape[:2]:
        raise ValueError(
            f'train_mask shape {train_mask.shape} does not match '
            f'X spatial shape {X.shape[:2]}.'
        )

    if valid_mask is None:
        valid_mask = np.all(np.isfinite(X), axis=-1)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != X.shape[:2]:
            raise ValueError(
                f'valid_mask shape {valid_mask.shape} does not match '
                f'X spatial shape {X.shape[:2]}.'
            )

    if np.any(train_mask & ~valid_mask):
        raise ValueError('train_mask must be a subset of valid_mask.')
    if not np.any(train_mask):
        raise ValueError('train_mask must contain at least one training pixel.')

    scaler = StandardScaler()
    scaler.fit(X[train_mask])

    X_scaled = np.full(X.shape, np.nan, dtype=output_dtype)
    X_scaled[valid_mask] = scaler.transform(X[valid_mask]).astype(
        output_dtype,
        copy=False,
    )
    return X_scaled, scaler


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



def otsu_threshold_float(data, bins=256):
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
    allow_insufficient: bool = False,
    min_test_samples: Optional[int] = None,
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
    min_test_samples : int, optional
        Minimum number of test samples required for each class. If a class has
        fewer test samples after the standard split, additional samples from
        its training portion are also used for testing (creating train/test
        overlap for that class). By default (None), no minimum is enforced.
        Only effective when per_class=True or per_class=False with balanced=True
        (not effective in pure random sampling mode).

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

    >>> # Per-class mode with min_test_samples: supplement test set if needed
    >>> X_tr, X_te, y_tr, y_te = stratified_train_test_split(
    ...     X, y, train_size=10, per_class=True, min_test_samples=5)

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

    if train_size <= 0:
        raise ValueError("train_size must be a positive integer.")

    if min_test_samples is not None and (
        isinstance(min_test_samples, (bool, np.bool_))
        or not isinstance(min_test_samples, (int, np.integer))
        or min_test_samples <= 0
    ):
        raise ValueError("min_test_samples must be a positive integer or None.")

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

            # Resolve train_size: interpret values <= 1 as a proportion of the
            # valid pool, otherwise as an absolute sample count.
            if train_size <= 1.0:
                total_train = int(round(train_size * total_valid))
            else:
                total_train = int(round(train_size))

            if total_train > total_valid:
                raise ValueError(
                    f"train_size={train_size} (resolved to {total_train} samples) "
                    f"exceeds the total number of valid samples ({total_valid})."
                )

            # Compute raw (float) quotas then floor them.
            raw_quotas = {
                label: (size / total_valid) * total_train
                for label, size in class_sizes.items()
            }
            floored_quotas = {label: int(q) for label, q in raw_quotas.items()}
            allocated = sum(floored_quotas.values())
            # Remaining samples to distribute (integer count).
            remainder = total_train - allocated

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
        supplemented_classes = []

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
                if min_test_samples is not None and actual_count < min_test_samples:
                    supplemented_classes.append(label)
            else:
                # Normal split: first `quota` to train, rest to test
                train_indices.extend(idx_array[:quota].tolist())
                test_indices.extend(idx_array[quota:].tolist())
                
                if min_test_samples is not None:
                    test_count = actual_count - quota
                    if test_count < min_test_samples:
                        deficit = min_test_samples - test_count
                        n_to_take = min(deficit, quota)
                        if n_to_take > 0:
                            supplement = idx_array[:n_to_take].tolist()
                            test_indices.extend(supplement)
                            supplemented_classes.append(label)

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if supplemented_classes:
            import warnings
            warnings.warn(
                f"The following classes had fewer than min_test_samples={min_test_samples} "
                "test samples after the stratified split. Additional samples from their "
                "training set were also used as test samples, creating train/test overlap "
                f"for these classes: {supplemented_classes}.",
                UserWarning,
            )

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



def _choose_spatial_block(
    candidate_utilities,
    rng,
):
    """Choose one block from pre-computed coverage/adjacency utilities.

    Adjacency has already been incorporated into each candidate's utility.
    Consequently, ties must be resolved without applying a second, implicit
    adjacency preference; otherwise any positive penalty behaves almost like a
    hard on/off switch instead of a continuous trade-off.
    """
    best_utility = max(
        utility
        for _, _, _, utility in candidate_utilities
    )

    best_candidates = [
        (block_id, coverage_gain, adjacent_count)
        for block_id, coverage_gain, adjacent_count, utility
        in candidate_utilities
        if np.isclose(utility, best_utility, rtol=1e-12, atol=1e-12)
    ]

    best_block_ids = [
        block_id
        for block_id, _, _ in best_candidates
    ]

    return best_block_ids[rng.randint(len(best_block_ids))]


def spatially_disjoint_train_test_split(
    X,
    y,
    train_size,
    per_class=True,
    block_size=10,
    buffer_size=3,
    ignore_label=None,
    random_state=None,
    preserve_shape=True,
    allow_insufficient=False,
    adjacency_penalty=0.0,
    min_test_samples=None,
):
    """
    Split a labeled image into spatially disjoint training and testing samples.

    The image is partitioned into non-overlapping spatial blocks. Complete
    blocks are selected as the training region, and a buffer zone around the
    selected region is excluded from testing. Training pixels are sampled from
    the selected blocks, whereas valid pixels outside the training region and
    its buffer zone form the test set.

    Parameters
    ----------
    X : np.ndarray
        Feature image with shape (H, W, D) or (H, W).
    y : np.ndarray
        Label map with shape (H, W).
    train_size : int
        Desired number of training samples. When ``per_class=True``, this is
        the target number of training samples for each class. Otherwise, this
        is the total training-sample budget, allocated proportionally to class
        frequencies.
    per_class : bool, default=True
        Whether ``train_size`` denotes a per-class quota.
    block_size : int, default=10
        Side length of each square spatial block.
    buffer_size : int, default=3
        Width of the buffer zone excluded around selected training blocks.
    ignore_label : int or float, optional
        Label value excluded from both training and testing.
    random_state : int, optional
        Random seed for reproducible block selection and sample selection.
    preserve_shape : bool, default=True
        If True, return full-size arrays with unselected positions filled with
        NaN. Otherwise, return flattened selected samples only.
    allow_insufficient : bool, default=False
        If False, raise an error if a class cannot meet its requested training
        quota while retaining spatially disjoint testing samples. If True, use
        all eligible pixels from the selected training region for classes that
        cannot meet their quotas. Training and testing samples never overlap.
    adjacency_penalty : float, default=0.0
        Relative penalty for selecting blocks adjacent to already selected
        training blocks. A candidate block with ``n`` selected 8-neighbors has
        its coverage utility divided by
        ``1 + adjacency_penalty * n``. Set to 0 to select blocks only according
        to class-coverage gain. Larger values favor more dispersed training
        blocks, but may reduce the remaining test area because more buffer
        regions are created.
    min_test_samples : int, optional
        Minimum number of test samples required for each class. If a class has
        fewer test samples than this threshold in the spatially disjoint region,
        additional samples from the training region of that class are also used
        for testing (creating train/test overlap for that class). By default
        (None), no minimum is enforced.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Spatially disjoint training and testing subsets. Their format is
        controlled by ``preserve_shape``.

    Notes
    -----
    A valid split requires every class to retain at least one labeled test
    pixel outside the selected training region and its buffer zone.
    When ``min_test_samples`` is set, classes with insufficient spatially
    disjoint test samples may have train/test overlap.
    """
    import warnings

    X = np.asarray(X)
    y = np.asarray(y)

    if y.ndim != 2:
        raise ValueError(
            "spatially_disjoint_train_test_split requires a 2D label map "
            "with shape (H, W)."
        )

    if X.ndim not in (2, 3):
        raise ValueError(
            "X must have shape (H, W) or (H, W, D) for spatial splitting."
        )

    if X.shape[:2] != y.shape:
        raise ValueError(
            f"Spatial dimensions of X {X.shape[:2]} do not match y {y.shape}."
        )

    if (
        isinstance(train_size, (bool, np.bool_))
        or not isinstance(train_size, (int, np.integer))
        or train_size <= 0
    ):
        raise ValueError("train_size must be a positive integer.")

    if (
        isinstance(block_size, (bool, np.bool_))
        or not isinstance(block_size, (int, np.integer))
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer.")

    if (
        isinstance(buffer_size, (bool, np.bool_))
        or not isinstance(buffer_size, (int, np.integer))
        or buffer_size < 0
    ):
        raise ValueError("buffer_size must be a non-negative integer.")

    if (
        isinstance(adjacency_penalty, (bool, np.bool_))
        or not isinstance(
            adjacency_penalty,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(adjacency_penalty)
        or adjacency_penalty < 0
    ):
        raise ValueError(
            "adjacency_penalty must be a finite non-negative number."
        )

    if min_test_samples is not None and (
        isinstance(min_test_samples, (bool, np.bool_))
        or not isinstance(min_test_samples, (int, np.integer))
        or min_test_samples <= 0
    ):
        raise ValueError("min_test_samples must be a positive integer or None.")

    height, width = y.shape
    flat_y = y.ravel()

    if ignore_label is None:
        valid_mask = np.ones_like(y, dtype=bool)
    else:
        valid_mask = y != ignore_label

    if not np.any(valid_mask):
        raise ValueError("No valid samples remain after applying ignore_label.")

    labels = np.unique(y[valid_mask])
    n_classes = len(labels)

    if n_classes == 0:
        raise ValueError("No valid classes were found.")

    label_to_position = {
        label: position
        for position, label in enumerate(labels)
    }

    total_counts = np.array(
        [np.sum(valid_mask & (y == label)) for label in labels],
        dtype=int,
    )

    if per_class:
        quotas = np.full(n_classes, train_size, dtype=int)
    else:
        total_valid = int(total_counts.sum())

        if train_size > total_valid:
            raise ValueError(
                f"train_size={train_size} exceeds the number of valid samples "
                f"({total_valid})."
            )

        raw_quotas = total_counts / total_valid * train_size
        quotas = np.floor(raw_quotas).astype(int)
        remainder = int(train_size - quotas.sum())

        if remainder > 0:
            fractional_order = np.argsort(-(raw_quotas - quotas))
            for position in fractional_order[:remainder]:
                quotas[position] += 1

    # Each class needs at least one sample for training and one for testing.
    if np.any(total_counts < 2):
        invalid_labels = labels[total_counts < 2].tolist()
        raise ValueError(
            "Spatially disjoint splitting requires at least two valid samples "
            f"per class. Insufficient classes: {invalid_labels}."
        )

    if not allow_insufficient:
        insufficient = total_counts <= quotas

        if np.any(insufficient):
            invalid_labels = labels[insufficient].tolist()
            raise ValueError(
                "The following classes cannot satisfy the requested training "
                "quota while retaining at least one spatially disjoint test "
                f"sample: {invalid_labels}. Set allow_insufficient=True to "
                "use fewer training samples for these classes."
            )

    # With allow_insufficient=True, reserve at least one sample in principle
    # for the spatially disjoint test region.
    target_region_counts = quotas.copy()
    if allow_insufficient:
        target_region_counts = np.minimum(
            target_region_counts,
            total_counts - 1,
        )

    blocks = []
    grid_to_block = {}

    for row_start in range(0, height, block_size):
        row_end = min(row_start + block_size, height)
        row_idx = row_start // block_size

        for col_start in range(0, width, block_size):
            col_end = min(col_start + block_size, width)
            col_idx = col_start // block_size

            block_valid = valid_mask[row_start:row_end, col_start:col_end]
            if not np.any(block_valid):
                continue

            block_labels = y[row_start:row_end, col_start:col_end]
            class_counts = np.zeros(n_classes, dtype=int)

            for label, position in label_to_position.items():
                class_counts[position] = np.sum(
                    block_valid & (block_labels == label)
                )

            block_id = len(blocks)
            blocks.append(
                {
                    "row_start": row_start,
                    "row_end": row_end,
                    "col_start": col_start,
                    "col_end": col_end,
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "counts": class_counts,
                }
            )
            grid_to_block[(row_idx, col_idx)] = block_id

    if not blocks:
        raise ValueError("No valid spatial blocks were found.")

    def count_adjacent_selected(block_id, selected_set):
        """Count selected 8-neighbor blocks of a candidate block."""
        block = blocks[block_id]
        row_idx = block["row_idx"]
        col_idx = block["col_idx"]
        adjacent_count = 0

        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue

                neighbor_id = grid_to_block.get(
                    (row_idx + row_offset, col_idx + col_offset)
                )

                if neighbor_id is not None and neighbor_id in selected_set:
                    adjacent_count += 1

        return adjacent_count

    def dilate_mask(mask, radius):
        """Expand a Boolean mask by a Chebyshev-distance square kernel."""
        if radius == 0:
            return mask.copy()

        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        expanded = np.zeros_like(mask, dtype=bool)

        for row_offset in range(2 * radius + 1):
            for col_offset in range(2 * radius + 1):
                expanded |= padded[
                    row_offset:row_offset + height,
                    col_offset:col_offset + width,
                ]

        return expanded

    def build_output(train_indices, test_indices):
        """Build either full-shape maps or flattened sample arrays."""
        if not preserve_shape:
            if X.ndim == 3:
                feature_dim = X.shape[2]
                X_flat = X.reshape(-1, feature_dim)
            else:
                X_flat = X.ravel()

            return (
                X_flat[train_indices],
                X_flat[test_indices],
                flat_y[train_indices],
                flat_y[test_indices],
            )

        output_dtype = np.result_type(X.dtype, np.float64)
        X_train = np.full(X.shape, np.nan, dtype=output_dtype)
        X_test = np.full(X.shape, np.nan, dtype=output_dtype)
        y_train = np.full(y.shape, np.nan, dtype=float)
        y_test = np.full(y.shape, np.nan, dtype=float)

        train_rows, train_cols = np.unravel_index(train_indices, y.shape)
        test_rows, test_cols = np.unravel_index(test_indices, y.shape)

        y_train[train_rows, train_cols] = y[train_rows, train_cols]
        y_test[test_rows, test_cols] = y[test_rows, test_cols]

        if X.ndim == 3:
            X_train[train_rows, train_cols, :] = X[train_rows, train_cols, :]
            X_test[test_rows, test_cols, :] = X[test_rows, test_cols, :]
        else:
            X_train[train_rows, train_cols] = X[train_rows, train_cols]
            X_test[test_rows, test_cols] = X[test_rows, test_cols]

        return X_train, X_test, y_train, y_test

    max_attempts = 100
    last_failure = None

    for attempt in range(max_attempts):
        seed = None if random_state is None else random_state + attempt
        rng = np.random.RandomState(seed)

        selected_set = set()
        selected_mask = np.zeros(y.shape, dtype=bool)
        available_counts = np.zeros(n_classes, dtype=int)
        remaining_block_ids = list(range(len(blocks)))

        # Select blocks that most efficiently cover unresolved class quotas.
        while np.any(available_counts < target_region_counts):
            deficits = np.maximum(
                target_region_counts - available_counts,
                0,
            )
            candidate_utilities = []

            for block_id in remaining_block_ids:
                block_counts = blocks[block_id]["counts"]

                # Count only samples that contribute to unresolved quotas.
                coverage_gain = int(np.minimum(block_counts, deficits).sum())
                if coverage_gain <= 0:
                    continue

                adjacent_count = count_adjacent_selected(
                    block_id,
                    selected_set,
                )

                # A relative penalty remains meaningful across block sizes and
                # datasets, unlike subtracting a fixed number of samples.
                utility = coverage_gain / (
                    1.0 + adjacency_penalty * adjacent_count
                )

                candidate_utilities.append(
                    (block_id, coverage_gain, adjacent_count, utility)
                )

            if not candidate_utilities:
                break

            chosen_block_id = _choose_spatial_block(
                candidate_utilities,
                rng,
            )
            chosen_block = blocks[chosen_block_id]

            selected_set.add(chosen_block_id)
            selected_mask[
                chosen_block["row_start"]:chosen_block["row_end"],
                chosen_block["col_start"]:chosen_block["col_end"],
            ] = True
            available_counts += chosen_block["counts"]
            remaining_block_ids.remove(chosen_block_id)

        if not selected_set:
            last_failure = "No training blocks could be selected."
            continue

        if not allow_insufficient and np.any(available_counts < quotas):
            invalid_labels = labels[available_counts < quotas].tolist()
            last_failure = (
                "Selected training blocks could not meet the requested quotas "
                f"for classes {invalid_labels}."
            )
            continue

        # Buffered pixels are excluded from testing but are not training samples.
        exclusion_mask = dilate_mask(selected_mask, buffer_size)
        test_candidate_mask = valid_mask & ~exclusion_mask

        test_counts = np.array(
            [
                np.sum(test_candidate_mask & (y == label))
                for label in labels
            ],
            dtype=int,
        )

        # --- min_test_samples: supplement test set from training region ---
        supplemented_classes = []
        if min_test_samples is not None:
            for position, label in enumerate(labels):
                deficit = min_test_samples - test_counts[position]
                if deficit > 0:
                    class_train_mask = selected_mask & (y == label)
                    class_train_indices = np.flatnonzero(class_train_mask.ravel())
                    rng.shuffle(class_train_indices)
                    n_to_take = min(deficit, len(class_train_indices))
                    if n_to_take > 0:
                        supplement = class_train_indices[:n_to_take]
                        rows, cols = np.unravel_index(supplement, y.shape)
                        test_candidate_mask[rows, cols] = True
                        test_counts[position] += n_to_take
                        supplemented_classes.append(label)

        zero_test_classes = None
        if np.any(test_counts == 0):
            invalid_labels = labels[test_counts == 0].tolist()
            if not allow_insufficient:
                last_failure = (
                    "Selected training blocks and buffer zone removed all test "
                    f"samples for classes {invalid_labels}."
                )
                continue
            zero_test_classes = labels[test_counts == 0]

        train_indices = []
        insufficient_labels = []

        for position, label in enumerate(labels):
            candidate_indices = np.flatnonzero(
                selected_mask.ravel() & (flat_y == label)
            )
            rng.shuffle(candidate_indices)

            requested = quotas[position]

            if len(candidate_indices) < requested:
                if not allow_insufficient:
                    last_failure = (
                        f"Class {label} has only {len(candidate_indices)} "
                        f"eligible training samples, but {requested} are "
                        "required."
                    )
                    break

                insufficient_labels.append(label)
                requested = len(candidate_indices)

            train_indices.extend(candidate_indices[:requested].tolist())

        else:
            train_indices = np.asarray(train_indices, dtype=int)

            if zero_test_classes is not None:
                overlap_mask = test_candidate_mask.copy()
                for label in zero_test_classes:
                    overlap_mask |= (y == label)
                test_indices = np.flatnonzero(overlap_mask.ravel())
            else:
                test_indices = np.flatnonzero(test_candidate_mask.ravel())

            if len(train_indices) == 0 or len(test_indices) == 0:
                last_failure = "The generated split contains an empty subset."
                continue

            if insufficient_labels:
                warnings.warn(
                    "Some classes could not meet their requested training quota "
                    "under the spatially disjoint constraint. All eligible "
                    "samples from the selected training region were used for "
                    f"these classes: {insufficient_labels}.",
                    UserWarning,
                )

            if zero_test_classes is not None:
                warnings.warn(
                    "All pixels of some classes fall inside the excluded "
                    "training+buffer region. Their full set of pixels is used "
                    "for testing, creating train/test overlap for these classes: "
                    f"{zero_test_classes.tolist()}.",
                    UserWarning,
                )

            if supplemented_classes:
                warnings.warn(
                    f"The following classes had fewer than min_test_samples={min_test_samples} "
                    "spatially disjoint test samples. Additional samples from their training "
                    "region were used as test samples, creating train/test overlap for these "
                    f"classes: {supplemented_classes}.",
                    UserWarning,
                )

            return build_output(train_indices, test_indices)

    raise ValueError(
        "Unable to construct a valid spatially disjoint split after "
        f"{max_attempts} attempts. Last failure: {last_failure} "
        "Consider reducing block_size or buffer_size, changing random_state, "
        "reducing adjacency_penalty, or setting allow_insufficient=True."
    )



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
    # 3a.  Short-circuit: window_size == 1 — no spatial aggregation needed
    # ------------------------------------------------------------------
    if window_size == 1:
        features = X_3d.astype(output_dtype, copy=False)

        if y_2d is None:
            if preserve_shape:
                return features
            return features.reshape(-1, D)

        # y_2d is provided — apply validity mask
        if np.issubdtype(y_2d.dtype, np.integer):
            valid_mask = np.ones((H, W), dtype=bool)
        else:
            valid_mask = ~np.isnan(y_2d)
        if ignore_label is not None:
            valid_mask &= (y_2d != ignore_label)

        if preserve_shape:
            out = np.full((H, W, D), np.nan, dtype=output_dtype)
            out[valid_mask] = features[valid_mask]
            y_out = np.full((H, W), np.nan, dtype=np.float64)
            y_out[valid_mask] = y_2d[valid_mask].astype(np.float64, copy=False)
            return out, y_out

        # preserve_shape=False
        valid_rows, valid_cols = np.where(valid_mask)
        return (
            features[valid_rows, valid_cols],
            y_2d[valid_rows, valid_cols].astype(np.float64, copy=False),
        )

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

