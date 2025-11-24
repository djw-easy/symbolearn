import numpy as np
from numba import njit, jit


# ==================== Binary Cross Entropy ====================
@njit
def _binary_cross_entropy_core(y_true, y_pred, w, epsilon=1e-9):
    """
    二分类交叉熵的核心计算（numba加速）
    
    Args:
        y_true: int32 数组
        y_pred: float64 数组
        w: float64 数组或 None
    """
    n = len(y_true)
    
    # Clip 预测值
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # 计算交叉熵
    loss_sum = 0.0
    for i in range(n):
        log_likelihood = -(y_true[i] * np.log(y_pred_clipped[i]) + 
                          (1 - y_true[i]) * np.log(1 - y_pred_clipped[i]))
        loss_sum += log_likelihood
    
    return loss_sum / n


@njit
def _binary_cross_entropy_weighted_core(y_true, y_pred, w, epsilon=1e-9):
    """
    加权二分类交叉熵的核心计算（numba加速）
    """
    n = len(y_true)
    
    # Clip 预测值
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # 计算加权交叉熵
    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        log_likelihood = -(y_true[i] * np.log(y_pred_clipped[i]) + 
                          (1 - y_true[i]) * np.log(1 - y_pred_clipped[i]))
        loss_sum += w[i] * log_likelihood
        weight_sum += w[i]
    
    if weight_sum == 0.0:
        return 0.0
    
    return loss_sum / weight_sum


def _binary_cross_entropy_loss(y_true, y_pred, w=None):
    """二分类交叉熵损失"""
    # 输入验证
    if len(y_true) == 0:
        return 0.0
    
    # 转换为合适的类型
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    # 调用 numba 加速的核心函数
    if w is not None:
        w = np.asarray(w, dtype=np.float32)
        return _binary_cross_entropy_weighted_core(y_true, y_pred, w)
    else:
        return _binary_cross_entropy_core(y_true, y_pred, None)


# ==================== Multiclass Cross Entropy ====================
@njit
def _multiclass_cross_entropy_core(y_true, y_pred, w, epsilon=1e-9):
    """
    多分类交叉熵的核心计算（numba加速）
    
    Args:
        y_true: int32 数组 (n,)
        y_pred: float64 数组 (n, m)
        w: float64 数组 (n,) 或 None
    """
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]
    
    # 验证标签范围
    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0  # 错误标记
    
    # Clip 预测值
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    loss_sum = 0.0
    for i in range(n):
        log_likelihood = -np.log(y_pred_clipped[i, y_true[i]])
        loss_sum += log_likelihood
    
    return loss_sum / n


@njit
def _multiclass_cross_entropy_weighted_core(y_true, y_pred, w, epsilon=1e-9):
    """加权多分类交叉熵的核心计算（numba加速）"""
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]
    
    # 验证标签范围
    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0  # 错误标记
    
    # 验证权重
    for i in range(n):
        if w[i] < 0:
            return -2.0  # 负权重错误
    
    # Clip 预测值
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        log_likelihood = -np.log(y_pred_clipped[i, y_true[i]])
        loss_sum += w[i] * log_likelihood
        weight_sum += w[i]
    
    if weight_sum == 0.0:
        return 0.0
    
    return loss_sum / weight_sum


def _multiclass_cross_entropy_loss(y_true, y_pred, w=None):
    """多分类交叉熵损失"""
    # 输入验证
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred 必须是 2D 数组，当前维度: {y_pred.ndim}")
    
    n_samples, n_classes = y_pred.shape
    
    if len(y_true) == 0:
        return 0.0
    
    if len(y_true) != n_samples:
        raise ValueError(f"y_true 和 y_pred 样本数不匹配: {len(y_true)} vs {n_samples}")
    
    if w is not None and len(w) != n_samples:
        raise ValueError(f"权重 w 和样本数不匹配: {len(w)} vs {n_samples}")
    
    # 转换类型
    y_true = np.asarray(y_true, dtype=np.int32)
    
    # 调用 numba 加速的核心函数
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        loss = _multiclass_cross_entropy_weighted_core(y_true, y_pred, w)
    else:
        loss = _multiclass_cross_entropy_core(y_true, y_pred, None)
    
    # 错误处理
    if loss == -1.0:
        raise ValueError(
            f"y_true 包含超出范围的标签。有效范围: [0, {n_classes-1}]"
        )
    elif loss == -2.0:
        raise ValueError("权重 w 不能包含负值")
    
    return loss


# ==================== Binary NLL Loss ====================
@njit
def _binary_nll_core(y_true, y_pred, w, epsilon=1e-9):
    """二分类 NLL 的核心计算（numba加速）"""
    n = len(y_true)
    
    loss_sum = 0.0
    for i in range(n):
        log_p_pos = min(y_pred[i], 0.0)  # 确保 <= 0
        log_p_neg = np.log1p(-np.exp(log_p_pos))
        nll = -(y_true[i] * log_p_pos + (1 - y_true[i]) * log_p_neg)
        loss_sum += nll
    
    return loss_sum / n


@njit
def _binary_nll_weighted_core(y_true, y_pred, w, epsilon=1e-9):
    """加权二分类 NLL 的核心计算（numba加速）"""
    n = len(y_true)
    
    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        log_p_pos = min(y_pred[i], 0.0)
        log_p_neg = np.log1p(-np.exp(log_p_pos))
        nll = -(y_true[i] * log_p_pos + (1 - y_true[i]) * log_p_neg)
        loss_sum += w[i] * nll
        weight_sum += w[i]
    
    if weight_sum == 0.0:
        return 0.0
    
    return loss_sum / weight_sum


def _binary_nll_loss(y_true, y_pred, w=None):
    """二分类负对数似然损失"""
    if len(y_true) == 0:
        return 0.0
    
    # 转换类型
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        return _binary_nll_weighted_core(y_true, y_pred, w)
    else:
        return _binary_nll_core(y_true, y_pred, None)


# ==================== Multiclass NLL Loss ====================
@njit
def _multiclass_nll_core(y_true, y_pred, w):
    """多分类 NLL 的核心计算（numba加速）"""
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]
    
    # 验证标签范围
    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0
    
    loss_sum = 0.0
    for i in range(n):
        nll = -y_pred[i, y_true[i]]
        loss_sum += nll
    
    return loss_sum / n


@njit
def _multiclass_nll_weighted_core(y_true, y_pred, w):
    """加权多分类 NLL 的核心计算（numba加速）"""
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]
    
    # 验证标签范围和权重
    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0
        if w[i] < 0:
            return -2.0
    
    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        nll = -y_pred[i, y_true[i]]
        loss_sum += w[i] * nll
        weight_sum += w[i]
    
    if weight_sum == 0.0:
        return 0.0
    
    return loss_sum / weight_sum


def _multiclass_nll_loss(y_true, y_pred, w=None):
    """多分类负对数似然损失"""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred 必须是 2D 数组，当前维度: {y_pred.ndim}")
    
    n_samples, n_classes = y_pred.shape
    
    if len(y_true) == 0:
        return 0.0
    
    if len(y_true) != n_samples:
        raise ValueError(f"y_true 和 y_pred 样本数不匹配")
    
    if w is not None and len(w) != n_samples:
        raise ValueError(f"权重 w 和样本数不匹配")
    
    # 转换类型
    y_true = np.asarray(y_true, dtype=np.int32)
    
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        loss = _multiclass_nll_weighted_core(y_true, y_pred, w)
    else:
        loss = _multiclass_nll_core(y_true, y_pred, None)
    
    if loss == -1.0:
        raise ValueError(f"y_true 包含超出范围的标签")
    elif loss == -2.0:
        raise ValueError("权重 w 不能包含负值")
    
    return loss


# ==================== Binary Focal Loss ====================
@njit
def _binary_focal_core(y_true, y_pred, w, alpha, gamma, epsilon=1e-9):
    """二分类 Focal Loss 的核心计算（numba加速）"""
    n = len(y_true)
    
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    loss_sum = 0.0
    for i in range(n):
        # 交叉熵项
        log_likelihood = -(y_true[i] * np.log(y_pred_clipped[i]) + 
                          (1 - y_true[i]) * np.log(1 - y_pred_clipped[i]))
        
        # p_t (模型对正确类别的预测概率)
        p_t = y_pred_clipped[i] * y_true[i] + (1 - y_pred_clipped[i]) * (1 - y_true[i])
        
        # 调制因子
        modulating_factor = (1.0 - p_t) ** gamma
        
        # alpha 权重
        focal_val = modulating_factor * log_likelihood
        if alpha >= 0:  # alpha < 0 表示不使用 alpha
            alpha_t = alpha * y_true[i] + (1 - alpha) * (1 - y_true[i])
            focal_val = alpha_t * focal_val
        
        loss_sum += focal_val
    
    return loss_sum / n


@njit
def _binary_focal_weighted_core(y_true, y_pred, w, alpha, gamma, epsilon=1e-9):
    """加权二分类 Focal Loss 的核心计算（numba加速）"""
    n = len(y_true)
    
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        log_likelihood = -(y_true[i] * np.log(y_pred_clipped[i]) + 
                          (1 - y_true[i]) * np.log(1 - y_pred_clipped[i]))
        
        p_t = y_pred_clipped[i] * y_true[i] + (1 - y_pred_clipped[i]) * (1 - y_true[i])
        modulating_factor = (1.0 - p_t) ** gamma
        
        focal_val = modulating_factor * log_likelihood
        if alpha >= 0:
            alpha_t = alpha * y_true[i] + (1 - alpha) * (1 - y_true[i])
            focal_val = alpha_t * focal_val
        
        loss_sum += w[i] * focal_val
        weight_sum += w[i]
    
    if weight_sum == 0.0:
        return 0.0
    
    return loss_sum / weight_sum


def _binary_focal_loss(y_true, y_pred, w=None, alpha=0.25, gamma=2.0):
    """二分类 Focal Loss"""
    if alpha is not None and not (0 < alpha < 1):
        raise ValueError("对于二分类，alpha 必须在 (0, 1) 范围内。")
    
    if len(y_true) == 0:
        return 0.0
    
    # 转换类型
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # 使用负数表示不使用 alpha
    alpha_val = alpha if alpha is not None else -1.0
    
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        return _binary_focal_weighted_core(y_true, y_pred, w, alpha_val, gamma)
    else:
        return _binary_focal_core(y_true, y_pred, None, alpha_val, gamma)


# ==================== Multiclass Focal Loss ====================
@njit
def _multiclass_focal_core(y_true, y_pred, alpha, gamma, epsilon=1e-9):
    """多分类 Focal Loss 的核心计算（numba加速）"""
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]
    
    # 验证标签范围
    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0
    
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    loss_sum = 0.0
    for i in range(n):
        # 正确类别的预测概率
        p_t = y_pred_clipped[i, y_true[i]]
        
        # 交叉熵项
        log_likelihood = -np.log(p_t)
        
        # 调制因子
        modulating_factor = (1.0 - p_t) ** gamma
        
        # alpha 权重
        focal_val = modulating_factor * log_likelihood
        if alpha is not None:
            focal_val = alpha[y_true[i]] * focal_val
        
        loss_sum += focal_val
    
    return loss_sum / n


@njit
def _multiclass_focal_weighted_core(y_true, y_pred, w, alpha, gamma, epsilon=1e-9):
    """加权多分类 Focal Loss 的核心计算（numba加速）"""
    n = y_pred.shape[0]
    n_classes = y_pred.shape[1]
    
    # 验证标签范围和权重
    for i in range(n):
        if y_true[i] < 0 or y_true[i] >= n_classes:
            return -1.0
        if w[i] < 0:
            return -2.0
    
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    loss_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        p_t = y_pred_clipped[i, y_true[i]]
        log_likelihood = -np.log(p_t)
        modulating_factor = (1.0 - p_t) ** gamma
        
        focal_val = modulating_factor * log_likelihood
        if alpha is not None:
            focal_val = alpha[y_true[i]] * focal_val
        
        loss_sum += w[i] * focal_val
        weight_sum += w[i]
    
    if weight_sum == 0.0:
        return 0.0
    
    return loss_sum / weight_sum


def _multiclass_focal_loss(y_true, y_pred, w=None, alpha=None, gamma=2.0):
    """多分类 Focal Loss"""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred 必须是 2D 数组")
    
    n_samples, n_classes = y_pred.shape
    
    if len(y_true) == 0:
        return 0.0
    
    if len(y_true) != n_samples:
        raise ValueError(f"y_true 和 y_pred 样本数不匹配")
    
    if w is not None and len(w) != n_samples:
        raise ValueError(f"权重 w 和样本数不匹配")
    
    if alpha is not None:
        alpha = np.asarray(alpha, dtype=np.float64)
        if len(alpha) != n_classes:
            raise ValueError(f"alpha 的长度必须等于类别数")
    
    # 转换类型
    y_true = np.asarray(y_true, dtype=np.int32)
    
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        loss = _multiclass_focal_weighted_core(y_true, y_pred, w, alpha, gamma)
    else:
        loss = _multiclass_focal_core(y_true, y_pred, alpha, gamma)
    
    if loss == -1.0:
        raise ValueError(f"y_true 包含超出范围的标签")
    elif loss == -2.0:
        raise ValueError("权重 w 不能包含负值")
    
    return loss


# ==================== 公共接口函数 ====================
def cross_entropy_loss(y_true, y_pred, w=None):
    """
    计算（可加权的）交叉熵损失，确保数值稳定性。
    支持多分类和二分类场景。
    
    Args:
        y_true: array-like, shape (n_samples,)
            真实标签，整数类型
        y_pred: array-like, shape (n_samples,) 或 (n_samples, n_classes)
            预测概率
            - 二分类: shape (n_samples,), 表示正类概率
            - 多分类: shape (n_samples, n_classes), 表示各类概率
        w: array-like, shape (n_samples,), optional
            样本权重
    
    Returns:
        float: 交叉熵损失值
    
    Note:
        调用者需要在外部处理 MaskedArray，只传入有效样本
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if w is not None:
        w = np.asarray(w, dtype=float)
    
    is_binary = y_pred.ndim == 1
    
    if is_binary:
        return _binary_cross_entropy_loss(y_true, y_pred, w)
    else:
        return _multiclass_cross_entropy_loss(y_true, y_pred, w)


def nll_loss(y_true, y_pred, w=None):
    """
    计算（可加权的）负对数似然损失，确保数值稳定性。
    支持多分类和二分类场景。
    
    Args:
        y_true: array-like, shape (n_samples,)
            真实标签，整数类型
        y_pred: array-like, shape (n_samples,) 或 (n_samples, n_classes)
            对数概率（log probabilities）
            - 二分类: shape (n_samples,), 表示正类的对数概率
            - 多分类: shape (n_samples, n_classes), 表示各类的对数概率
        w: array-like, shape (n_samples,), optional
            样本权重
    
    Returns:
        float: 负对数似然损失值
    
    Note:
        调用者需要在外部处理 MaskedArray，只传入有效样本
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if w is not None:
        w = np.asarray(w, dtype=float)
    
    is_binary = y_pred.ndim == 1
    
    if is_binary:
        return _binary_nll_loss(y_true, y_pred, w)
    else:
        return _multiclass_nll_loss(y_true, y_pred, w)


def focal_loss(y_true, y_pred, w=None, alpha=None, gamma=2.0):
    """
    计算（可加权的）Focal Loss，旨在解决类别不平衡和难易样本不均衡问题。
    函数结构与 cross_entropy_loss 兼容，并增加了 alpha 和 gamma 参数。
    
    Args:
        y_true: array-like, shape (n_samples,)
            真实标签，整数类型
        y_pred: array-like, shape (n_samples,) 或 (n_samples, n_classes)
            预测概率
            - 二分类: shape (n_samples,), 表示正类概率
            - 多分类: shape (n_samples, n_classes), 表示各类概率
        w: array-like, shape (n_samples,), optional
            样本权重
        alpha: float 或 array-like, optional
            类别平衡权重
            - 二分类: float, 范围 (0, 1), 正类的权重
            - 多分类: array-like, shape (n_classes,), 各类的权重
        gamma: float, default=2.0
            聚焦参数，用于调节难易样本的权重
    
    Returns:
        float: Focal Loss 损失值
    
    Note:
        调用者需要在外部处理 MaskedArray，只传入有效样本
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if w is not None:
        w = np.asarray(w, dtype=float)
    
    is_binary = y_pred.ndim == 1
    
    if is_binary:
        return _binary_focal_loss(y_true, y_pred, w, alpha, gamma)
    else:
        return _multiclass_focal_loss(y_true, y_pred, w, alpha, gamma)


@jit(nopython=True, cache=True)
def _accuracy_binary_numba(y_true, y_pred, w):
    """二分类准确率计算（Numba加速）"""
    n = len(y_true)
    correct = 0.0
    total = 0.0
    
    for i in range(n):
        pred_label = 1 if y_pred[i] >= 0.5 else 0
        weight = w[i] if w is not None else 1.0
        if pred_label == y_true[i]:
            correct += weight
        total += weight
    
    return correct / total if total > 0 else 0.0

@jit(nopython=True, cache=True)
def _accuracy_multiclass_numba(y_true, y_pred, w):
    """多分类准确率计算（Numba加速）"""
    n = len(y_true)
    n_classes = y_pred.shape[1]
    correct = 0.0
    total = 0.0
    
    for i in range(n):
        # 找到最大概率的类别
        max_prob = y_pred[i, 0]
        pred_label = 0
        for j in range(1, n_classes):
            if y_pred[i, j] > max_prob:
                max_prob = y_pred[i, j]
                pred_label = j
        
        weight = w[i] if w is not None else 1.0
        if pred_label == y_true[i]:
            correct += weight
        total += weight
    
    return correct / total if total > 0 else 0.0


def accuracy(y_true, y_pred, w=None):
    """
    计算overall accuracy
    
    Args:
        y_true: array-like, shape (n_samples,)
            真实标签，整数类型
        y_pred: array-like, shape (n_samples,) 或 (n_samples, n_classes)
            预测概率
            - 二分类: shape (n_samples,), 表示正类概率
            - 多分类: shape (n_samples, n_classes), 表示各类概率
        w: array-like, shape (n_samples,), optional
            样本权重
    
    Returns:
        float: overall accuracy
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        if len(w) != len(y_true):
            raise ValueError("权重数组长度必须与样本数量一致")
    
    is_binary = y_pred.ndim == 1
    
    if is_binary:
        # 二分类
        return _accuracy_binary_numba(y_true, y_pred, w)
    else:
        # 多分类
        if y_pred.ndim != 2:
            raise ValueError("多分类预测必须是2维数组")
        return _accuracy_multiclass_numba(y_true, y_pred, w)





# 使用示例
if __name__ == "__main__":
    import numpy as np
    
    # 二分类示例
    print("=== 二分类示例 ===")
    y_true_binary = np.array([0, 1, 1, 0, 1])
    y_pred_binary = np.array([0.1, 0.9, 0.8, 0.2, 0.7])  # sigmoid 输出
    loss_binary = cross_entropy_loss(y_true_binary, y_pred_binary)
    print(f"二分类交叉熵损失: {loss_binary:.4f}")
    
    # 多分类示例
    print("\n=== 多分类示例 ===")
    y_true_multi = np.array([0, 1, 2, 1, 0])
    y_pred_multi = np.array([
        [0.8, 0.1, 0.1],  # 预测类别 0，实际类别 0
        [0.2, 0.7, 0.1],  # 预测类别 1，实际类别 1
        [0.1, 0.2, 0.7],  # 预测类别 2，实际类别 2
        [0.3, 0.6, 0.1],  # 预测类别 1，实际类别 1
        [0.9, 0.05, 0.05] # 预测类别 0，实际类别 0
    ])  # softmax 输出
    loss_multi = cross_entropy_loss(y_true_multi, y_pred_multi)
    print(f"多分类交叉熵损失: {loss_multi:.4f}")
    
    # 带权重的示例
    print("\n=== 带权重的二分类示例 ===")
    weights = np.array([1.0, 2.0, 1.5, 1.0, 2.5])
    loss_weighted = cross_entropy_loss(y_true_binary, y_pred_binary, w=weights)
    print(f"加权二分类交叉熵损失: {loss_weighted:.4f}")