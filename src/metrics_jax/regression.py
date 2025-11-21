from jax import jit
import jax.numpy as jnp



@jit
def mean_absolute_error(y, y_pred, w=None):
    """Calculate the mean absolute error."""
    return jnp.average(jnp.abs(y_pred - y), weights=w)


@jit
def mean_square_error(y, y_pred, w=None):
    """Calculate the mean square error."""
    return jnp.average(((y_pred - y) ** 2), weights=w)


@jit
def root_mean_square_error(y, y_pred, w=None):
    """Calculate the root mean square error."""
    return jnp.sqrt(jnp.average(((y_pred - y) ** 2), weights=w))









