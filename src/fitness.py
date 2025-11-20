import jax
import optax
import numpy as np
from jax import jit
from numba import njit
import jax.numpy as jnp
from typing import Optional, Union
from joblib import wrap_non_picklable_objects





__all__ = ['make_fitness']


class Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with Jax vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better, name):
        self.name = name
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)




@jax.jit
def mean_square_error(y, y_pred):
    """Calculate the mean square error."""
    return jnp.average(((y_pred - y) ** 2))


@jit
def cross_entropy_loss_simple(y_true: jnp.ndarray, y_pred: jnp.ndarray, 
                             w: Optional[jnp.ndarray] = None, epsilon: float = 1e-9) -> jnp.ndarray:
    """
    简化的交叉熵损失函数，避免复杂的错误处理，适合在JIT环境下使用。
    
    注意：调用者需要确保输入的有效性（标签在有效范围内，权重非负等）
    """
    # 确定是二分类还是多分类
    is_binary = y_pred.ndim == 1
    
    if is_binary:
        # 二分类
        y_pred_clipped = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
        log_likelihood = -(y_true * jnp.log(y_pred_clipped) + 
                          (1 - y_true) * jnp.log(1 - y_pred_clipped))
    else:
        # 多分类
        y_pred_clipped = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true_one_hot = jax.nn.one_hot(y_true, y_pred.shape[1])
        log_likelihood = -jnp.sum(y_true_one_hot * jnp.log(y_pred_clipped), axis=1)
    
    # 应用权重
    if w is not None:
        weighted_loss = w * log_likelihood
        loss_sum = jnp.sum(weighted_loss)
        weight_sum = jnp.sum(w)
        return jax.lax.cond(weight_sum == 0.0,
                           lambda: jnp.array(0.0),
                           lambda: loss_sum / weight_sum)
    else:
        return jnp.mean(log_likelihood)


_fitness_map = {
    # Regression
    'mse': Fitness(function=mean_square_error, greater_is_better=False, name='mse'), 
    # Classification
    'cross_entropy': Fitness(function=cross_entropy_loss_simple, 
                             greater_is_better=False, name='cross_entropy')
}



