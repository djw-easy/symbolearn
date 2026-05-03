import numpy as np
from numba import njit, jit


# ==================== Binary Cross Entropy ====================

@njit(cache=True)
def _binary_cross_entropy_core(y_true, y_pred, sample_weight, epsilon=1e-9):
    """
    Core computation for binary cross-entropy loss (Numba JIT accelerated).

    Args:
        y_true        : int32 array, shape (n,)   — ground-truth labels in {0, 1}
        y_pred        : float64 array, shape (n,)  — predicted probabilities for the positive class
        sample_weight : float64 array, shape (n,)  — weights for each sample
        epsilon       : float — small constant for numerical stability

    Returns:
        float — weighted mean binary cross-entropy loss
    """
    n = len(y_true)
    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        p = min(max(y_pred[i], epsilon), 1.0 - epsilon)
        ce = -(y_true[i] * np.log(p) + (1 - y_true[i]) * np.log(1.0 - p))
        loss_sum += sample_weight[i] * ce
        weight_sum += sample_weight[i]
    if weight_sum == 0.0:
        return 0.0
    return loss_sum / weight_sum


def _binary_cross_entropy_loss(y_true, y_pred, sample_weight=None):
    """Binary cross-entropy loss dispatcher."""
    if len(y_true) == 0:
        return 0.0
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim != 1 or len(sample_weight) != n:
            raise ValueError(
                f"sample_weight must be a 1-D array of length {n}, "
                f"got shape {sample_weight.shape}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")
    return _binary_cross_entropy_core(y_true, y_pred, sample_weight)


# ==================== Multiclass Cross Entropy ====================

@njit(cache=True)
def _multiclass_cross_entropy_core(y_true, y_pred, sample_weight, epsilon=1e-9):
    """
    Core computation for multiclass cross-entropy loss (Numba JIT accelerated).

    Args:
        y_true        : int32 array, shape (n,)    — ground-truth class indices
        y_pred        : float64 array, shape (n, C) — predicted class probabilities (softmax output)
        sample_weight : float64 array, shape (n,)   — weights for each sample
        epsilon       : float — small constant for numerical stability

    Returns:
        float — weighted mean cross-entropy loss, or -1.0 if a label is out of range
    """
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]

    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0  # sentinel: invalid label

    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        p = min(max(y_pred[i, y_true[i]], epsilon), 1.0 - epsilon)
        loss_sum += sample_weight[i] * (-np.log(p))
        weight_sum += sample_weight[i]
    if weight_sum == 0.0:
        return 0.0
    return loss_sum / weight_sum


def _multiclass_cross_entropy_loss(y_true, y_pred, sample_weight=None):
    """Multiclass cross-entropy loss dispatcher."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be a 2-D array, got ndim={y_pred.ndim}")

    n_samples, n_classes = y_pred.shape
    if len(y_true) == 0:
        return 0.0
    if len(y_true) != n_samples:
        raise ValueError(
            f"y_true and y_pred sample count mismatch: {len(y_true)} vs {n_samples}"
        )

    y_true = np.asarray(y_true, dtype=np.int32)
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim != 1 or len(sample_weight) != n_samples:
            raise ValueError(
                f"sample_weight must be a 1-D array of length {n_samples}, "
                f"got shape {sample_weight.shape}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

    loss = _multiclass_cross_entropy_core(y_true, y_pred, sample_weight)
    if loss == -1.0:
        raise ValueError(f"y_true contains out-of-range labels. Valid range: [0, {n_classes - 1}]")
    return loss


# ==================== Binary NLL Loss ====================

@njit(cache=True)
def _binary_nll_core(y_true, y_pred, sample_weight):
    """
    Core computation for binary negative log-likelihood loss (Numba JIT accelerated).

    Expects y_pred to be log-probabilities of the positive class (i.e. log p(y=1|x)).

    Args:
        y_true        : int32 array, shape (n,)  — ground-truth labels in {0, 1}
        y_pred        : float64 array, shape (n,) — log-probabilities for the positive class
        sample_weight : float64 array, shape (n,) — weights for each sample

    Returns:
        float — weighted mean NLL loss
    """
    n = len(y_true)
    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        log_p_pos = min(y_pred[i], 0.0)          # clamp to (-inf, 0]
        log_p_neg = np.log1p(-np.exp(log_p_pos))  # log(1 - p)
        nll = -(y_true[i] * log_p_pos + (1 - y_true[i]) * log_p_neg)
        loss_sum += sample_weight[i] * nll
        weight_sum += sample_weight[i]
    if weight_sum == 0.0:
        return 0.0
    return loss_sum / weight_sum


def _binary_nll_loss(y_true, y_pred, sample_weight=None):
    """Binary NLL loss dispatcher."""
    if len(y_true) == 0:
        return 0.0
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim != 1 or len(sample_weight) != n:
            raise ValueError(
                f"sample_weight must be a 1-D array of length {n}, "
                f"got shape {sample_weight.shape}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")
    return _binary_nll_core(y_true, y_pred, sample_weight)


# ==================== Multiclass NLL Loss ====================

@njit(cache=True)
def _multiclass_nll_core(y_true, y_pred, sample_weight):
    """
    Core computation for multiclass NLL loss (Numba JIT accelerated).

    Expects y_pred to be log-probabilities (i.e. log-softmax output).

    Args:
        y_true        : int32 array, shape (n,)    — ground-truth class indices
        y_pred        : float64 array, shape (n, C) — log-probabilities for each class
        sample_weight : float64 array, shape (n,)   — weights for each sample

    Returns:
        float — weighted mean NLL loss, or -1.0 if a label is out of range
    """
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]

    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0  # sentinel: invalid label

    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        loss_sum += sample_weight[i] * (-y_pred[i, y_true[i]])
        weight_sum += sample_weight[i]
    if weight_sum == 0.0:
        return 0.0
    return loss_sum / weight_sum


def _multiclass_nll_loss(y_true, y_pred, sample_weight=None):
    """Multiclass NLL loss dispatcher."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be a 2-D array, got ndim={y_pred.ndim}")

    n_samples, n_classes = y_pred.shape
    if len(y_true) == 0:
        return 0.0
    if len(y_true) != n_samples:
        raise ValueError("y_true and y_pred sample count mismatch")

    y_true = np.asarray(y_true, dtype=np.int32)
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim != 1 or len(sample_weight) != n_samples:
            raise ValueError(
                f"sample_weight must be a 1-D array of length {n_samples}, "
                f"got shape {sample_weight.shape}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

    loss = _multiclass_nll_core(y_true, y_pred, sample_weight)
    if loss == -1.0:
        raise ValueError("y_true contains out-of-range labels")
    return loss


# ==================== Binary Focal Loss ====================

@njit(cache=True)
def _binary_focal_core(y_true, y_pred, sample_weight, alpha, gamma, epsilon=1e-9):
    """
    Core computation for binary focal loss (Numba JIT accelerated).

    Args:
        y_true        : int32 array, shape (n,)  — ground-truth labels in {0, 1}
        y_pred        : float64 array, shape (n,) — predicted probabilities for the positive class
        sample_weight : float64 array, shape (n,) — weights for each sample
        alpha         : float — positive-class weight; negative value means alpha is disabled
        gamma         : float — focusing exponent
        epsilon       : float — small constant for numerical stability

    Returns:
        float — weighted mean focal loss
    """
    n = len(y_true)
    loss_sum = 0.0
    weight_sum = 0.0

    for i in range(n):
        p = min(max(y_pred[i], epsilon), 1.0 - epsilon)
        ce = -(y_true[i] * np.log(p) + (1 - y_true[i]) * np.log(1.0 - p))

        # p_t: model confidence on the correct class
        p_t = p * y_true[i] + (1.0 - p) * (1 - y_true[i])
        focal_factor = (1.0 - p_t) ** gamma

        focal_val = focal_factor * ce
        if alpha >= 0.0:
            alpha_t = alpha * y_true[i] + (1.0 - alpha) * (1 - y_true[i])
            focal_val = alpha_t * focal_val

        loss_sum += sample_weight[i] * focal_val
        weight_sum += sample_weight[i]

    if weight_sum == 0.0:
        return 0.0
    return loss_sum / weight_sum


def _binary_focal_loss(y_true, y_pred, sample_weight=None, alpha=0.25, gamma=2.0):
    """Binary focal loss dispatcher."""
    if alpha is not None and not (0.0 < alpha < 1.0):
        raise ValueError("For binary focal loss, alpha must be in (0, 1).")
    if len(y_true) == 0:
        return 0.0

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim != 1 or len(sample_weight) != n:
            raise ValueError(
                f"sample_weight must be a 1-D array of length {n}, "
                f"got shape {sample_weight.shape}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

    alpha_val = float(alpha) if alpha is not None else -1.0
    return _binary_focal_core(y_true, y_pred, sample_weight, alpha_val, float(gamma))


# ==================== Multiclass Focal Loss ====================

@njit(cache=True)
def _multiclass_focal_core(y_true, y_pred, sample_weight, alpha, use_alpha, gamma, epsilon=1e-9):
    """
    Core computation for multiclass focal loss (Numba JIT accelerated).

    Args:
        y_true        : int32 array, shape (n,)    — ground-truth class indices
        y_pred        : float64 array, shape (n, C) — predicted class probabilities
        sample_weight : float64 array, shape (n,)   — weights for each sample
        alpha         : float64 array, shape (C,)   — per-class alpha weights
        use_alpha     : bool — whether to apply alpha weighting
        gamma         : float — focusing exponent
        epsilon       : float — small constant for numerical stability

    Returns:
        float — weighted mean focal loss, or -1.0 if a label is out of range
    """
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]

    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0  # sentinel: invalid label

    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        p_t = min(max(y_pred[i, y_true[i]], epsilon), 1.0 - epsilon)
        ce = -np.log(p_t)
        focal_factor = (1.0 - p_t) ** gamma
        focal_val = focal_factor * ce
        if use_alpha:
            focal_val = alpha[y_true[i]] * focal_val
        loss_sum += sample_weight[i] * focal_val
        weight_sum += sample_weight[i]

    if weight_sum == 0.0:
        return 0.0
    return loss_sum / weight_sum


def _multiclass_focal_loss(y_true, y_pred, sample_weight=None, alpha=None, gamma=2.0):
    """Multiclass focal loss dispatcher."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.ndim != 2:
        raise ValueError("y_pred must be a 2-D array")

    n_samples, n_classes = y_pred.shape
    if len(y_true) == 0:
        return 0.0
    if len(y_true) != n_samples:
        raise ValueError("y_true and y_pred sample count mismatch")

    y_true = np.asarray(y_true, dtype=np.int32)
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim != 1 or len(sample_weight) != n_samples:
            raise ValueError(
                f"sample_weight must be a 1-D array of length {n_samples}, "
                f"got shape {sample_weight.shape}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

    if alpha is not None:
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        if len(alpha_arr) != n_classes:
            raise ValueError("alpha length must equal number of classes")
        use_alpha = True
    else:
        alpha_arr = np.zeros(n_classes, dtype=np.float64)  # placeholder, not used
        use_alpha = False

    loss = _multiclass_focal_core(y_true, y_pred, sample_weight, alpha_arr, use_alpha, float(gamma))
    if loss == -1.0:
        raise ValueError("y_true contains out-of-range labels")
    return loss


# ==================== Binary Hinge Loss ====================

@njit(cache=True)
def _binary_hinge_core(y_true, y_pred, sample_weight):
    """
    Core computation for binary hinge loss (Numba JIT accelerated).

    Args:
        y_true        : int32 array, shape (n,)  — ground-truth labels in {0, 1}
        y_pred        : float64 array, shape (n,) — decision function values (raw scores)
        sample_weight : float64 array, shape (n,) — weights for each sample

    Returns:
        float — weighted mean hinge loss
    """
    n = len(y_true)
    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        w = sample_weight[i]
        y_signed = 2 * y_true[i] - 1   # map {0, 1} -> {-1, +1}
        margin = y_signed * y_pred[i]
        loss_sum += w * max(0.0, 1.0 - margin)
        weight_sum += w
    
    if weight_sum == 0.0:
        return 0.0
    return loss_sum / weight_sum


def _binary_hinge_loss(y_true, y_pred, sample_weight=None):
    """Binary hinge loss dispatcher."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    n = len(y_true)
    if n == 0:
        return 0.0

    # Process sample_weight
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    
    if sample_weight.ndim != 1:
        raise ValueError("sample_weight must be 1-D array")
        
    if len(sample_weight) != n:
        raise ValueError(
            f"sample_weight length mismatch: {len(sample_weight)} vs {n}"
        )
    
    if np.any(sample_weight < 0):
        raise ValueError("sample_weight cannot contain negative values")

    # Validate labels
    unique = np.unique(y_true)
    if not all(lbl in (0, 1) for lbl in unique):
        raise ValueError("Binary hinge loss requires labels in {0, 1}")

    return _binary_hinge_core(y_true, y_pred, sample_weight)


# ==================== Multiclass Hinge Loss ====================

@njit(cache=True)
def _multiclass_hinge_core(y_true, y_pred, sample_weight):
    """
    Core computation for multiclass hinge loss (Numba JIT accelerated).

    Uses the Crammer-Singer formulation: for each sample, sum violations
    over all incorrect classes and normalise by (n_classes - 1).

    Args:
        y_true        : int32 array, shape (n,)    — ground-truth class indices
        y_pred        : float64 array, shape (n, C) — decision function scores
        sample_weight : float64 array, shape (n,)   — weights for each sample

    Returns:
        float — weighted mean hinge loss, or -1.0 if a label is out of range
    """
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]

    # Check label validity first (to avoid index errors in loop)
    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0  # sentinel: invalid label

    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        w = sample_weight[i]
        true_cls = y_true[i]
        true_score = y_pred[i, true_cls]
        total_violation = 0.0
        for j in range(n_classes):
            if j != true_cls:
                margin = true_score - y_pred[i, j]
                total_violation += max(0.0, 1.0 - margin)
        
        sample_loss = total_violation / (n_classes - 1)
        loss_sum += w * sample_loss
        weight_sum += w

    if weight_sum == 0.0:
        return 0.0
    return loss_sum / weight_sum


def _multiclass_hinge_loss(y_true, y_pred, sample_weight=None):
    """Multiclass hinge loss dispatcher."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be a 2-D array, got ndim={y_pred.ndim}")

    n_samples, n_classes = y_pred.shape
    if len(y_true) == 0:
        return 0.0
    if len(y_true) != n_samples:
        raise ValueError(
            f"y_true and y_pred sample count mismatch: {len(y_true)} vs {n_samples}"
        )

    y_true = np.asarray(y_true, dtype=np.int32)
    
    # Process sample_weight
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    
    if sample_weight.ndim != 1:
        raise ValueError("sample_weight must be 1-D array")

    if len(sample_weight) != n_samples:
        raise ValueError(
            f"sample_weight length mismatch: {len(sample_weight)} vs {n_samples}"
        )
        
    if np.any(sample_weight < 0):
        raise ValueError("sample_weight cannot contain negative values")

    loss = _multiclass_hinge_core(y_true, y_pred, sample_weight)

    if loss == -1.0:
        raise ValueError(f"y_true contains out-of-range labels. Valid range: [0, {n_classes - 1}]")
    return loss


# ==================== Accuracy ====================

@njit(cache=True)
def _accuracy_binary_core(y_true, y_pred, sample_weight):
    """
    Core computation for binary accuracy (Numba JIT accelerated).

    Args:
        y_true        : int64 array, shape (n,)  — ground-truth labels in {0, 1}
        y_pred        : float64 array, shape (n,) — predicted probabilities for the positive class
        sample_weight : float64 array, shape (n,) — weights for each sample

    Returns:
        float — weighted accuracy in [0, 1]
    """
    n = len(y_true)
    correct_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        pred = 1 if y_pred[i] >= 0.5 else 0
        if pred == y_true[i]:
            correct_sum += sample_weight[i]
        weight_sum += sample_weight[i]
    if weight_sum == 0.0:
        return 0.0
    return correct_sum / weight_sum


@njit(cache=True)
def _accuracy_multiclass_core(y_true, y_pred, sample_weight):
    """
    Core computation for multiclass accuracy (Numba JIT accelerated).

    Args:
        y_true        : int64 array, shape (n,)    — ground-truth class indices
        y_pred        : float64 array, shape (n, C) — predicted class probabilities
        sample_weight : float64 array, shape (n,)   — weights for each sample

    Returns:
        float — weighted accuracy in [0, 1]
    """
    n = len(y_true)
    n_classes = y_pred.shape[1]
    correct_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        best_j = 0
        best_p = y_pred[i, 0]
        for j in range(1, n_classes):
            if y_pred[i, j] > best_p:
                best_p = y_pred[i, j]
                best_j = j
        if best_j == y_true[i]:
            correct_sum += sample_weight[i]
        weight_sum += sample_weight[i]
    if weight_sum == 0.0:
        return 0.0
    return correct_sum / weight_sum


# ==================== Public API ====================

def cross_entropy_loss(y_true, y_pred, sample_weight=None):
    """
    Compute cross-entropy loss with numerical stability.
    Supports both binary and multiclass settings.

    Args:
        y_true : array-like, shape (n_samples,)
            Integer ground-truth labels.
        y_pred : array-like, shape (n_samples,) or (n_samples, n_classes)
            Predicted probabilities.
            - Binary      : shape (n_samples,), probability of the positive class.
            - Multiclass  : shape (n_samples, n_classes), softmax probabilities.
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights. If None, all samples are weighted equally.

    Returns:
        float — weighted mean cross-entropy loss value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 1:
        return _binary_cross_entropy_loss(y_true, y_pred, sample_weight)
    else:
        return _multiclass_cross_entropy_loss(y_true, y_pred, sample_weight)


def nll_loss(y_true, y_pred, sample_weight=None):
    """
    Compute negative log-likelihood (NLL) loss with numerical stability.
    Supports both binary and multiclass settings.

    Args:
        y_true : array-like, shape (n_samples,)
            Integer ground-truth labels.
        y_pred : array-like, shape (n_samples,) or (n_samples, n_classes)
            Log-probabilities (log-softmax / log-sigmoid output).
            - Binary      : shape (n_samples,), log P(y=1|x).
            - Multiclass  : shape (n_samples, n_classes), log P(y=k|x).
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights. If None, all samples are weighted equally.

    Returns:
        float — weighted mean NLL loss value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 1:
        return _binary_nll_loss(y_true, y_pred, sample_weight)
    else:
        return _multiclass_nll_loss(y_true, y_pred, sample_weight)


def focal_loss(y_true, y_pred, sample_weight=None, alpha=None, gamma=2.0):
    """
    Compute focal loss to address class imbalance and hard/easy sample imbalance.

    Args:
        y_true : array-like, shape (n_samples,)
            Integer ground-truth labels.
        y_pred : array-like, shape (n_samples,) or (n_samples, n_classes)
            Predicted probabilities.
            - Binary      : shape (n_samples,), probability of the positive class.
            - Multiclass  : shape (n_samples, n_classes), softmax probabilities.
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights. If None, all samples are weighted equally.
        alpha : float or array-like, optional
            Class balance weight.
            - Binary      : float in (0, 1), weight for the positive class.
            - Multiclass  : array-like of shape (n_classes,), per-class weights.
        gamma : float, default=2.0
            Focusing parameter controlling the down-weighting of easy samples.

    Returns:
        float — weighted mean focal loss value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 1:
        return _binary_focal_loss(y_true, y_pred, sample_weight, alpha, gamma)
    else:
        return _multiclass_focal_loss(y_true, y_pred, sample_weight, alpha, gamma)


def hinge_loss(y_true, y_pred, sample_weight=None):
    """
    Compute hinge loss.

    Args:
        y_true : array-like, shape (n_samples,)
            Integer ground-truth labels.
        y_pred : array-like, shape (n_samples,) or (n_samples, n_classes)
            Decision function values (raw scores, not probabilities).
            - Binary      : shape (n_samples,).
            - Multiclass  : shape (n_samples, n_classes).
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights. If None, all samples are weighted equally.

    Returns:
        float — weighted mean hinge loss value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 1:
        return _binary_hinge_loss(y_true, y_pred, sample_weight)
    else:
        return _multiclass_hinge_loss(y_true, y_pred, sample_weight)


def accuracy(y_true, y_pred, sample_weight=None):
    """
    Compute overall accuracy, optionally weighted by sample_weight.

    Args:
        y_true : array-like, shape (n_samples,)
            Integer ground-truth labels.
        y_pred : array-like, shape (n_samples,) or (n_samples, n_classes)
            Predicted probabilities.
            - Binary      : shape (n_samples,), probability of the positive class.
            - Multiclass  : shape (n_samples, n_classes), softmax probabilities.
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights. If None, all samples are weighted equally.

    Returns:
        float — weighted accuracy in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    if n == 0:
        return 0.0

    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim != 1 or len(sample_weight) != n:
            raise ValueError(
                f"sample_weight must be a 1-D array of length {n}, "
                f"got shape {sample_weight.shape}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

    if y_pred.ndim == 1:
        return _accuracy_binary_core(y_true, y_pred, sample_weight)
    elif y_pred.ndim == 2:
        return _accuracy_multiclass_core(y_true, y_pred, sample_weight)
    else:
        raise ValueError("y_pred must be 1-D (binary) or 2-D (multiclass)")


# ==================== Quick smoke-test ====================

if __name__ == "__main__":
    print("=== Binary classification (uniform weights) ===")
    y_true_bin = np.array([0, 1, 1, 0, 1])
    y_pred_bin = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    print(f"Cross-entropy : {cross_entropy_loss(y_true_bin, y_pred_bin):.4f}")
    print(f"Focal loss    : {focal_loss(y_true_bin, y_pred_bin, alpha=0.25, gamma=2.0):.4f}")
    print(f"Hinge loss    : {hinge_loss(y_true_bin, y_pred_bin):.4f}")
    print(f"Accuracy      : {accuracy(y_true_bin, y_pred_bin):.4f}")

    print("\n=== Binary classification (custom weights) ===")
    w_bin = np.array([1.0, 2.0, 2.0, 1.0, 2.0])
    print(f"Cross-entropy : {cross_entropy_loss(y_true_bin, y_pred_bin, sample_weight=w_bin):.4f}")
    print(f"Focal loss    : {focal_loss(y_true_bin, y_pred_bin, sample_weight=w_bin, alpha=0.25, gamma=2.0):.4f}")
    print(f"Hinge loss    : {hinge_loss(y_true_bin, y_pred_bin, sample_weight=w_bin):.4f}")
    print(f"Accuracy      : {accuracy(y_true_bin, y_pred_bin, sample_weight=w_bin):.4f}")

    print("\n=== Multiclass classification (uniform weights) ===")
    y_true_mc = np.array([0, 1, 2, 1, 0])
    y_pred_mc = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.3, 0.6, 0.1],
        [0.9, 0.05, 0.05],
    ])
    print(f"Cross-entropy : {cross_entropy_loss(y_true_mc, y_pred_mc):.4f}")
    print(f"Focal loss    : {focal_loss(y_true_mc, y_pred_mc, gamma=2.0):.4f}")
    print(f"Hinge loss    : {hinge_loss(y_true_mc, y_pred_mc):.4f}")
    print(f"Accuracy      : {accuracy(y_true_mc, y_pred_mc):.4f}")

    print("\n=== Multiclass classification (custom weights) ===")
    w_mc = np.array([1.0, 3.0, 1.0, 3.0, 1.0])
    print(f"Cross-entropy : {cross_entropy_loss(y_true_mc, y_pred_mc, sample_weight=w_mc):.4f}")
    print(f"Focal loss    : {focal_loss(y_true_mc, y_pred_mc, sample_weight=w_mc, gamma=2.0):.4f}")
    print(f"Hinge loss    : {hinge_loss(y_true_mc, y_pred_mc, sample_weight=w_mc):.4f}")
    print(f"Accuracy      : {accuracy(y_true_mc, y_pred_mc, sample_weight=w_mc):.4f}")
