import jax.numpy as jnp
from jax import jit

# ==================== 辅助函数 & 核心 JIT (仅处理 2D 输入) ====================

@jit
def _safe_mean(loss_sum, w):
    """JIT-safe 加权平均计算。"""
    weight_sum = jnp.sum(w)
    # 处理 weight_sum = 0 的情况
    loss = jnp.where(weight_sum > 0, loss_sum / weight_sum, 0.0)
    return loss


@jit
def _cross_entropy_loss_core_jit_2d(y_true, y_pred_2d, w):
    """
    JIT-safe 交叉熵核心计算 (2D 输入)
    y_pred_2d: shape (n_samples, n_classes) 的概率
    """
    n_samples = y_pred_2d.shape[0]
    epsilon = 1e-9
    
    y_pred_clipped = jnp.clip(y_pred_2d, epsilon, 1.0 - epsilon)
    
    # 交叉熵项: -log(p_t)
    p_t = y_pred_clipped[jnp.arange(n_samples), y_true]
    log_likelihood = -jnp.log(p_t)
    
    weighted_loss_sum = jnp.sum(w * log_likelihood)
    return _safe_mean(weighted_loss_sum, w)


@jit
def _nll_loss_core_jit_2d(y_true, y_pred_log_2d, w):
    """
    JIT-safe 负对数似然核心计算 (2D 输入)
    y_pred_log_2d: shape (n_samples, n_classes) 的对数概率
    """
    n_samples = y_pred_log_2d.shape[0]
    
    # NLL 项: -log(p_t) = -log_p_t
    log_p_t = y_pred_log_2d[jnp.arange(n_samples), y_true]
    nll = -log_p_t
    
    weighted_loss_sum = jnp.sum(w * nll)
    return _safe_mean(weighted_loss_sum, w)


@jit
def _focal_loss_core_jit_2d(y_true, y_pred_2d, w, alpha_vals, gamma):
    """
    JIT-safe Focal Loss 核心计算 (2D 输入)
    alpha_vals: shape (n_classes,) 或 (2,) 数组
    """
    n_samples = y_pred_2d.shape[0]
    epsilon = 1e-9
    
    y_pred_clipped = jnp.clip(y_pred_2d, epsilon, 1.0 - epsilon)
    
    # 1. p_t (模型对正确类别的预测概率)
    p_t = y_pred_clipped[jnp.arange(n_samples), y_true]
    
    # 2. 交叉熵项: -log(p_t)
    log_likelihood = -jnp.log(p_t)
    
    # 3. 调制因子: (1 - p_t) ** gamma
    modulating_factor = (1.0 - p_t) ** gamma
    focal_val = modulating_factor * log_likelihood
    
    # 4. alpha 权重
    # alpha_vals 必须是 (n_classes,) 或 (2,) 数组
    alpha_t = alpha_vals[y_true]
    
    final_focal_val = alpha_t * focal_val
    
    weighted_loss_sum = jnp.sum(w * final_focal_val)
    return _safe_mean(weighted_loss_sum, w)


# ==================== 公共接口函数 (JIT 外部兼容包装) ====================

def cross_entropy_loss(y_true, y_pred, w=None):
    """
    计算（可加权的）交叉熵损失 (JAX JIT 兼容版本)
    Args:
        y_true: 真实标签 (n,)
        y_pred: 预测概率 (n,) 或 (n, class_num)
        w: 样本权重 (n,), 可选 None。
    """
    y_true = jnp.asarray(y_true, dtype=jnp.int32)
    y_pred = jnp.asarray(y_pred, dtype=jnp.float32)
    n_samples = y_pred.shape[0]

    if n_samples == 0:
        return 0.0

    # 1. JIT-safe 权重处理 (None -> ones)
    if w is None:
        w_safe = jnp.ones(n_samples, dtype=jnp.float32)
    else:
        w_safe = jnp.asarray(w, dtype=jnp.float32)
    
    # 2. 核心修复: 将 1D (二分类) 转换为 2D 格式 (N, 2)
    if y_pred.ndim == 1:
        # y_pred 是正类概率 p
        p = y_pred
        p_neg = 1.0 - p
        y_pred_2d = jnp.stack([p_neg, p], axis=1) # shape (N, 2)
        return _cross_entropy_loss_core_jit_2d(y_true, y_pred_2d, w_safe)
    
    elif y_pred.ndim == 2:
        return _cross_entropy_loss_core_jit_2d(y_true, y_pred, w_safe)

    else:
        raise ValueError(f"y_pred 维度必须是 1 或 2，当前维度: {y_pred.ndim}")


def nll_loss(y_true, y_pred, w=None):
    """
    计算（可加权的）负对数似然损失 (JAX JIT 兼容版本)
    Args:
        y_true: 真实标签 (n,)
        y_pred: 对数概率 (n,) 或 (n, class_num)
        w: 样本权重 (n,), 可选 None。
    """
    y_true = jnp.asarray(y_true, dtype=jnp.int32)
    y_pred_log = jnp.asarray(y_pred, dtype=jnp.float64)
    n_samples = y_pred_log.shape[0]

    if n_samples == 0:
        return 0.0

    # 1. JIT-safe 权重处理 (None -> ones)
    if w is None:
        w_safe = jnp.ones(n_samples, dtype=jnp.float64)
    else:
        w_safe = jnp.asarray(w, dtype=jnp.float64)

    # 2. 核心修复: 将 1D (二分类) 转换为 2D 格式 (N, 2)
    if y_pred_log.ndim == 1:
        # y_pred_log 是正类对数概率 log(p)
        log_p = y_pred_log
        # log(1-p) = log(1 - exp(log(p)))。使用 log1p(-exp(z))
        log_p_neg = jnp.log1p(-jnp.exp(log_p))
        y_pred_log_2d = jnp.stack([log_p_neg, log_p], axis=1) # shape (N, 2)
        
        return _nll_loss_core_jit_2d(y_true, y_pred_log_2d, w_safe)
    
    elif y_pred_log.ndim == 2:
        return _nll_loss_core_jit_2d(y_true, y_pred_log, w_safe)
    
    else:
        raise ValueError(f"y_pred 维度必须是 1 或 2，当前维度: {y_pred_log.ndim}")


def focal_loss(y_true, y_pred, w=None, alpha=None, gamma=2.0):
    """
    计算（可加权的）Focal Loss (JAX JIT 兼容版本)
    Args:
        y_true: 真实标签 (n,)
        y_pred: 预测概率 (n,) 或 (n, class_num)
        w: 样本权重 (n,), 可选 None。
        alpha: float (二分类) 或 array-like (多分类), 类别平衡权重。
        gamma: float, 聚焦参数。
    """
    y_true = jnp.asarray(y_true, dtype=jnp.int32)
    y_pred = jnp.asarray(y_pred, dtype=jnp.float64)
    gamma_val = jnp.asarray(gamma, dtype=jnp.float64)
    n_samples = y_pred.shape[0]

    if n_samples == 0:
        return 0.0

    # 1. JIT-safe 权重处理
    if w is None:
        w_safe = jnp.ones(n_samples, dtype=jnp.float64)
    else:
        w_safe = jnp.asarray(w, dtype=jnp.float64)

    # 2. 核心修复: 将 1D (二分类) 转换为 2D 格式 (N, 2)
    if y_pred.ndim == 1:
        # y_pred 是正类概率 p
        p = y_pred
        p_neg = 1.0 - p
        y_pred_2d = jnp.stack([p_neg, p], axis=1) # shape (N, 2)

        # alpha 转换为 shape (2,) 数组
        alpha_val = alpha if alpha is not None else 0.5
        if not (0 <= alpha_val <= 1):
             raise ValueError("对于二分类，alpha 必须在 [0, 1] 范围内。")
             
        # alpha_0 (负类) = 1 - alpha_val, alpha_1 (正类) = alpha_val
        alpha_vals = jnp.asarray([1.0 - alpha_val, alpha_val], dtype=jnp.float64)
        
        return _focal_loss_core_jit_2d(y_true, y_pred_2d, w_safe, alpha_vals, gamma_val)
    
    elif y_pred.ndim == 2:
        n_classes = y_pred.shape[1]
        
        # alpha 转换为 shape (n_classes,) 数组
        if alpha is None:
            # 默认均匀权重
            alpha_vals = jnp.ones(n_classes, dtype=jnp.float64)
        else:
            alpha_vals = jnp.asarray(alpha, dtype=jnp.float64)
            if alpha_vals.ndim == 0:
                 alpha_vals = jnp.full((n_classes,), alpha_vals.item())
            elif alpha_vals.shape[0] != n_classes:
                 raise ValueError(f"多分类 alpha 长度必须等于类别数: {alpha_vals.shape[0]} vs {n_classes}")

        return _focal_loss_core_jit_2d(y_true, y_pred, w_safe, alpha_vals, gamma_val)
    
    else:
        raise ValueError(f"y_pred 维度必须是 1 或 2，当前维度: {y_pred.ndim}")



