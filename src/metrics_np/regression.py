import numbers
import numpy as np
from numba import njit
from scipy.stats import rankdata



@njit
def mean_absolute_error(y, y_pred, w=None):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


@njit
def mean_square_error(y, y_pred, w=None):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


@njit
def root_mean_square_error(y, y_pred, w=None):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))





