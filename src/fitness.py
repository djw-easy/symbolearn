import jax
import numpy as np
from jax import jit
from numba import njit
import jax.numpy as jnp
from typing import Optional, Union
from joblib import wrap_non_picklable_objects


from src.metrics_np.regression import (
    mean_square_error as mean_square_error_np,
    mean_absolute_error as mean_absolute_error_np,
    root_mean_square_error as root_mean_square_error_np
)
from src.metrics_np.classification import (
    cross_entropy_loss as cross_entropy_loss_np, 
    focal_loss as focal_loss_np, 
    nll_loss as nll_loss_np, 
    accuracy as accuracy_np
)
from src.metrics_np.transformer import (
    weighted_pearson as weighted_pearson_np, 
    weighted_spearman as weighted_spearman_np
)
from src.metrics_np.transformer import (
    silhouette_loss as silhouette_loss_np,
    davies_bouldin_loss as davies_bouldin_loss_np, 
    calinski_harabasz_loss as calinski_harabasz_loss_np
)
from src.metrics_np.transformer import (
    fisher_loss as fisher_loss_np, 
    compactness_loss as compactness_loss_np, 
    f_statistic_loss as f_statistic_loss_np, 
    hellinger_loss as hellinger_loss_np, 
    wasserstein_loss as wasserstein_loss_np, 
    bhattacharyya_loss as bhattacharyya_loss_np, 
    js_divergence_loss as js_divergence_loss_np,
    separability_loss as separability_loss_np
)



from src.metrics_jax.regression import (
    mean_square_error as mean_square_error_jax,
    mean_absolute_error as mean_absolute_error_jax,
    root_mean_square_error as root_mean_square_error_jax
)
from src.metrics_jax.classification import (
    cross_entropy_loss as cross_entropy_loss_jax, 
    focal_loss as focal_loss_jax, 
    nll_loss as nll_loss_jax
)



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





_fitness_map = {
    # Regression
    'mae': Fitness(function=mean_absolute_error_np, greater_is_better=False, name='mae'), 
    'mse': Fitness(function=mean_square_error_np, greater_is_better=False, name='mse'), 
    'rmse': Fitness(function=root_mean_square_error_np, greater_is_better=False, name='rmse'), 
    # Classification
    'cross_entropy': Fitness(function=cross_entropy_loss_np, greater_is_better=False, name='cross_entropy'), 
    'nll_loss': Fitness(function=nll_loss_np, greater_is_better=False, name='nll_loss'), 
    'focal_loss': Fitness(function=focal_loss_np, greater_is_better=False, name='focal_loss'), 
    'accuracy': Fitness(function=accuracy_np, greater_is_better=True, name='accuracy'), 
    # Transformer - for continuous value
    'pearson': Fitness(function=weighted_pearson_np, greater_is_better=True, name='pearson'), 
    'spearman': Fitness(function=weighted_spearman_np, greater_is_better=True, name='spearman'), 
    # Transformer - for discrete value
    ## multi-dimension
    'silhouette': Fitness(function=silhouette_loss_np, greater_is_better=True, name='silhouette'), 
    'davies_bouldin': Fitness(function=davies_bouldin_loss_np, greater_is_better=False, name='davies_bouldin'), 
    'calinski_harabasz': Fitness(function=calinski_harabasz_loss_np, greater_is_better=True, name='calinski_harabasz'), 
    ## single-dimension
    'hellinger': Fitness(function=hellinger_loss_np, greater_is_better=True, name='hellinger'), 
    'bhattacharyya': Fitness(function=bhattacharyya_loss_np, greater_is_better=True, name='bhattacharyya'), 
    'js_divergence': Fitness(function=js_divergence_loss_np, greater_is_better=True, name='js_divergence'), 
    'wasserstein': Fitness(function=wasserstein_loss_np, greater_is_better=True, name='wasserstein'), 
    'earth_movers': Fitness(function=wasserstein_loss_np, greater_is_better=True, name='earth_movers'), 
    'fisher': Fitness(function=fisher_loss_np, greater_is_better=True, name='fisher'), 
    'f_statistic': Fitness(function=f_statistic_loss_np, greater_is_better=True, name='f_statistic'), 
    'compactness': Fitness(function=compactness_loss_np, greater_is_better=True, name='compactness'),
    'separability': Fitness(function=separability_loss_np, greater_is_better=True, name='separability')
}



_fitness_jax_map = {
    # Regression
    'mae': Fitness(function=mean_absolute_error_jax, greater_is_better=False, name='mae'), 
    'mse': Fitness(function=mean_square_error_jax, greater_is_better=False, name='mse'), 
    'rmse': Fitness(function=root_mean_square_error_jax, greater_is_better=False, name='rmse'), 
    # Classification
    # 'cross_entropy': Fitness(function=cross_entropy_loss_jax, greater_is_better=False, name='cross_entropy'), 
    'nll_loss': Fitness(function=nll_loss_jax, greater_is_better=False, name='nll_loss'), 
    # 'focal_loss': Fitness(function=focal_loss_jax, greater_is_better=False, name='focal_loss')
}


