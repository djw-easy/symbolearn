import numbers
import numpy as np
from numba import njit
from scipy.stats import rankdata



@njit(cache=True)
def mean_absolute_error(y, y_pred, sample_weight=None):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=sample_weight)


@njit(cache=True)
def mean_square_error(y, y_pred, sample_weight=None):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=sample_weight)


@njit(cache=True)
def root_mean_square_error(y, y_pred, sample_weight=None):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=sample_weight))





