import jax
import optax
import numpy as np
from numba import njit
import jax.numpy as jnp
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
def mean_square_error_jax(y, y_pred):
    """Calculate the mean square error."""
    return jnp.average(((y_pred - y) ** 2))



_fitness_map = {
    # Regression
    'mse': Fitness(function=mean_square_error_jax, greater_is_better=False, name='mse'), 
    # Classification
    'cross_entropy': Fitness(function=optax.softmax_cross_entropy_with_integer_labels, 
                             greater_is_better=False, name='cross_entropy')
}



