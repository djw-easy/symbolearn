import numpy as np
from typing import Literal, Union, Optional, TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from symbolearn.expression import Expression, ExpressionSet

from symbolearn.metrics.regression import (
    mean_square_error,
    mean_absolute_error,
    root_mean_square_error
)
from symbolearn.metrics.classification import (
    cross_entropy_loss,
    focal_loss,
    nll_loss,
    accuracy,
    hinge_loss
)
from symbolearn.metrics.transformer import (
    weighted_pearson,
    weighted_spearman,
    silhouette_loss,
    davies_bouldin_loss,
    calinski_harabasz_loss,
    fisher_loss,
    compactness_loss,
    f_statistic_loss,
    hellinger_loss,
    wasserstein_loss,
    bhattacharyya_loss,
    js_divergence_loss,
    separability_loss
)




class Fitness(object):
    """A metric class to measure the fitness of a program.

    This class wraps a loss function and its metadata (e.g., whether greater is
    better), providing a unified interface to calculate the fitness score of
    a symbolic expression on a given dataset.

    An optional regularization penalty can be applied to the constants
    (coefficients) of the expression to penalize complexity and prevent
    overfitting. The penalty is scaled by ``1 / C``, following the convention
    used in scikit-learn: a larger ``C`` weakens the regularization, while a
    smaller ``C`` strengthens it.

    Parameters
    ----------
    loss_function : callable
        The loss function with signature ``loss_function(y_true, y_pred, **kwargs)``.
        ``y_true`` is the target vector, and ``y_pred`` is the predicted values
        from the expression. It should return a floating point score.

    greater_is_better : bool
        Whether a higher value indicates a better fit. Typically ``False`` for
        error-based metrics (e.g., MSE) and ``True`` for correlation-based or
        accuracy metrics. This flag also controls the sign of the regularization
        term: the penalty is subtracted when ``True`` and added when ``False``,
        so that it always discourages large coefficients regardless of the
        optimization direction.

    penalty : {'l1', 'l2', 'elasticnet'} or None, optional
        The type of regularization penalty applied to the expression constants:

        - ``'l1'``         — Lasso penalty: ``sum(|c_i|)``. Encourages sparsity
                             by driving small coefficients toward zero.
        - ``'l2'``         — Ridge penalty: ``sum(c_i^2)``. Discourages large
                             coefficients without enforcing sparsity.
        - ``'elasticnet'`` — Elastic Net penalty: equal-weight combination of
                             L1 and L2, i.e. ``0.5 * sum(|c_i|) + 0.5 * sum(c_i^2)``.
                             Balances sparsity and coefficient shrinkage.
        - ``None``         — No regularization is applied (default).

    C : float, optional
        Inverse regularization strength. Must be a positive float. Smaller
        values produce stronger regularization; larger values reduce its
        influence on the fitness score. Defaults to ``1.0``.

    function_kwargs : dict, optional
        Additional keyword arguments to be passed to the ``loss_function``.
        Defaults to an empty dict.

    Attributes
    ----------
    loss_function : callable
        The wrapped loss function.
    greater_is_better : bool
        Optimization direction flag.
    penalty : str or None
        The active regularization type.
    C : float
        Inverse regularization strength.
    function_kwargs : dict
        Extra keyword arguments forwarded to ``loss_function``.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> fitness = Fitness(mean_squared_error, greater_is_better=False,
    ...                   penalty='l2', C=0.5)
    >>> score = fitness(expr, X_train, y_train)

    Notes
    -----
    The regularization term is computed as ``penalty_value / (C * n_coeffs)``
    where ``n_coeffs`` is the number of constants in the expression. Dividing
    by ``n_coeffs`` normalizes the penalty so that expressions with different
    numbers of constants are penalized on a per-parameter basis, avoiding an
    inherent bias against expressions with more constants.
    """

    def __init__(self, loss_function: Callable, greater_is_better: bool,
                 penalty: Literal['l1', 'l2', 'elasticnet'] | None = None, 
                 C: float = 1.0, function_kwargs={}):
        self.C = C
        self.penalty = penalty
        self.loss_function = loss_function
        self.function_kwargs = function_kwargs
        self.greater_is_better = greater_is_better

    def __call__(self, expr: Union['Expression', 'ExpressionSet'],
                 X: np.ndarray, y: np.ndarray,
                 constants: Optional[np.ndarray] = None, 
                 sample_weight: Optional[np.ndarray] = None):
        """Execute the expression and calculate its fitness score.

        Parameters
        ----------
        expr : Expression or ExpressionSet
            Symbolic expression to evaluate.
        X : np.ndarray
            Input features (N, D) or (H, W, D).
        y : np.ndarray
            Target values (N,) or (H, W).
        constants : np.ndarray, optional
            Constants used in the expression.
        sample_weight : np.ndarray, optional
            Weights for valid samples.

        Returns
        -------
        raw_fitness : float
            The calculated fitness score, including regularization penalty if configured.
        """
        # Determine data dimensionality
        is_3d = X.ndim == 3
        
        # Create validity mask for execution (NaNs invalid in 3D, all valid in 2D)
        exec_mask = (~np.isnan(y)) if is_3d else None
        
        # Execute expression to get predictions
        y_pred_valid = expr.execute(X, exec_mask, constants)
        
        # Flatten data and mask for loss calculation
        if is_3d:
            y_flat = y.ravel()
            loss_mask = (~np.isnan(y_flat)) if is_3d else np.ones_like(y_flat, dtype=bool)
            y_valid = y_flat[loss_mask]
        else:
            y_valid = y
        
        # Process sample weights
        if sample_weight is not None:
            if is_3d:
                w_flat = sample_weight.ravel()
                # Use weights directly if already filtered, otherwise apply mask
                if w_flat.shape[0] == y_valid.shape[0]:
                    sample_weight_valid = w_flat
                else:
                    sample_weight_valid = w_flat[loss_mask]
        else:
            sample_weight_valid = None
        
        # Calculate and return fitness
        raw_fitness = self.loss_function(
            y_valid, y_pred_valid, sample_weight_valid, **self.function_kwargs
        )

        if self.penalty is not None:
            # Resolve constants: prefer explicit argument, fall back to expression attribute
            coeffs = constants if constants is not None else getattr(expr, 'constants', None)

            if coeffs is not None and len(coeffs) > 0:
                coeffs = np.asarray(coeffs, dtype=float)

                if self.penalty == 'l1':
                    # L1 (Lasso): penalizes sum of absolute values of coefficients
                    reg_term = np.sum(np.abs(coeffs))
                elif self.penalty == 'l2':
                    # L2 (Ridge): penalizes sum of squared coefficients
                    reg_term = np.sum(coeffs ** 2)
                elif self.penalty == 'elasticnet':
                    # Elastic Net: convex combination of L1 and L2 penalties (equal weight)
                    l1_term = np.sum(np.abs(coeffs))
                    l2_term = np.sum(coeffs ** 2)
                    reg_term = 0.5 * l1_term + 0.5 * l2_term
                else:
                    raise ValueError(f"Unsupported penalty type: '{self.penalty}'. "
                                     f"Choose from 'l1', 'l2', or 'elasticnet'.")

                # C controls regularization strength: smaller C -> stronger regularization.
                # The penalty is scaled by 1/(C * n_coeffs) so that the penalty is
                # invariant to the number of constants in the expression, consistent
                # with the per-parameter regularization convention.
                penalty_value = reg_term / (self.C * len(coeffs))

                # Add or subtract penalty depending on optimization direction
                if self.greater_is_better:
                    # Higher fitness is better: subtract penalty to discourage large coefficients
                    raw_fitness -= penalty_value
                else:
                    # Lower fitness is better (e.g. MSE): add penalty to increase the loss
                    raw_fitness += penalty_value

        return raw_fitness



# {function_name: (function : Callable, greater_is_better : bool)}
_loss_function_map = {
    # Regression metrics
    'mae': (mean_absolute_error, False),
    'mse': (mean_square_error, False),
    'rmse': (root_mean_square_error, False),
    # Classification metrics
    'cross_entropy': (cross_entropy_loss, False),
    'nll_loss': (nll_loss, False),
    'focal_loss': (focal_loss, False),
    'hinge_loss': (hinge_loss, False),
    'accuracy': (accuracy, True),
    # Transformer-based metrics - Continuous
    'pearson': (weighted_pearson, True),
    'spearman': (weighted_spearman, True),
    # Transformer-based metrics - Discrete/Clustering
    ## Multi-dimension
    'silhouette': (silhouette_loss, True),
    'davies_bouldin': (davies_bouldin_loss, False),
    'calinski_harabasz': (calinski_harabasz_loss, True),
    ## Single-dimension/Distribution comparison
    'hellinger': (hellinger_loss, True),
    'bhattacharyya': (bhattacharyya_loss, True),
    'js_divergence': (js_divergence_loss, True),
    'wasserstein': (wasserstein_loss, True),
    'earth_movers': (wasserstein_loss, True),
    'fisher': (fisher_loss, True),
    'f_statistic': (f_statistic_loss, True),
    'compactness': (compactness_loss, True),
    'separability': (separability_loss, True)
}

