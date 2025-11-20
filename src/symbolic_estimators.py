import warnings
from typing import Optional, List, Union


import numpy as np
import jax.numpy as jnp
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, _check_sample_weight



from src.expression import ExpressionSet
from src.utils import check_random_state
from src.node import softmax, Operator
from src.halloffame import HallOfFame
from src.base import BaseSymbolic



class SymbolicRegressor(BaseSymbolic, RegressorMixin):

    """
    A Genetic Programming symbolic regressor.

    This class implements a symbolic regressor using genetic programming.
    It evolves mathematical expressions to fit a given dataset.

    Parameters
    ----------
    metric : str, default='mae'
        The fitness metric to use for regression. Supported metrics are
        'mae', 'mse', and 'rmse'.

    out_func : str, Operator, or callable, optional
        The output function to apply to the result of the expression.
        For regression, this is typically None.

    **kwargs
        Additional keyword arguments to be passed to the BaseSymbolic
        constructor.
    """
    def __init__(self, *, metric='mse', out_func=None, **kwargs):
        self.typical_metrics = (
            'mae', 'mse', 'rmse'
        )
        super().__init__(metric=metric, out_func=out_func, **kwargs)
        
    def fit(self, 
            X: np.ndarray | jnp.ndarray, 
            y: np.ndarray | jnp.ndarray, 
            variable_names: Optional[List[str]] = None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        variable_names : list of str, optional
            The names of the features.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.is_multi_output_:
            if self.order is None:
                 raise ValueError('`order` must be specified for multi-output problems.')
            if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
                X, y = check_X_y(X, y, y_numeric=True, multi_output=False)
            else:
                X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        else:
            if y.ndim != 1 and not (y.ndim == 2 and y.shape[1] == 1):
                X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
            else:
                X, y = check_X_y(X, y.ravel(), y_numeric=True)
        
        # X, y = jnp.array(X), jnp.array(y)
        return self._run(X, y, variable_names)
    
    def predict(self, X):
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted values for X.

        """
        check_is_fitted(
            self, 
            attributes=['equations_', 'n_features_in_', 
                        'feature_names_in_', 'hall_of_fame_']
        )
        
        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self.get_best().expression.execute(X)

        return y


class SymbolicClassifier(BaseSymbolic, ClassifierMixin):

    """
    A Genetic Programming symbolic classifier.

    This class implements a symbolic classifier using genetic programming.
    It evolves a set of expressions to separate classes.

    Parameters
    ----------
    metric : str, default='cross_entropy'
        The fitness metric to use for classification. The primary supported
        metric is 'cross_entropy'.

    out_func : str, Operator, or callable, default='softmax'
        The output function to apply to the result of the expression.
        For classification, this is typically 'softmax' for multi-class
        problems and 'sigmoid' for binary problems.

    **kwargs
        Additional keyword arguments to be passed to the BaseSymbolic
        constructor.
    """

    def __init__(self, *, metric='cross_entropy', out_func='softmax', **kwargs):
        self.typical_metrics = (
            'cross_entropy', 'nll_loss', 'focal_loss', 'accuracy'
        )
        super().__init__(metric=metric, out_func=out_func, **kwargs)

    def fit(self, X, y, sample_weight=None,
            variable_names: Optional[List[str]] = None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        variable_names : list of str, optional
            The names of the features.

        Returns
        -------
        self : object
            Returns self.
        """
        check_classification_targets(y)
        if self.is_multi_output_:
            if self.order is None:
                 raise ValueError('`order` must be specified for multi-output problems.')
            if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
                X, y = check_X_y(X, y, multi_output=True)
            else:
                X, y = check_X_y(X, y, multi_output=True)
        else:
            if y.ndim != 1 and not (y.ndim == 2 and y.shape[1] == 1):
                X, y = check_X_y(X, y, multi_output=True)
            else:
                X, y = check_X_y(X, y.ravel())
        # Check arrays
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        
        self.classes_, y_new = np.unique(y, return_inverse=True)
        n_classes_ = self.classes_.shape[0]

        if n_classes_ == 2:
            if self.out_func == 'sigmoid':
                self.order = None
                self.is_multi_output_ = False
            else:
                self.order = 2
                self.is_multi_output_ = True
        elif n_classes_ > 2:
            if self.out_func not in ['softmax', 'log_softmax']:
                warnings.warn("The 'out_func' should be 'softmax' or 'log_softmax' for "
                              "multi-class problems. It has been set to " "'softmax'.", UserWarning)
                self.out_func = 'softmax'
                self._out_func = softmax
            if self.order is None:
                self.order = n_classes_
                self.is_multi_output_ = True
            else:
                if not isinstance(self.order, int):
                    raise ValueError("The 'order' should be equal to 'n_classes_' "
                                     "for multi-class problems.")
                if self.order != n_classes_:
                    self.order = n_classes_
                    self.is_multi_output_ = True
        else:
            raise ValueError("The number of classes should be greater than 1.")
        
        return self._run(X, y_new, sample_weight, variable_names)

    def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data.

        Returns
        -------
        y_pred : array-like, shape = [n_samples]
            Predicted target values for X.
        """
        check_is_fitted(
            self, 
            attributes=['equations_', 'n_features_in_', 
                        'feature_names_in_', 'hall_of_fame_']
        )

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))
        
        y_pred = self.get_best().expression.execute(X)
        if self.out_func in ['softmax', 'log_softmax']:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
        
        return self.classes_.take(y_pred.astype(np.intp))

    def predict_proba(self, X):
        """Predict probabilities for X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data.

        Returns
        -------
        proba : array-like, shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by arithmetical order.
        """
        check_is_fitted(
            self, 
            attributes=['equations_', 'n_features_in_', 
                        'feature_names_in_', 'hall_of_fame_']
        )

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))
        
        proba = self.get_best().expression.execute(X)
        
        return proba



class SymbolicTransformer(BaseSymbolic, TransformerMixin):

    """
    A Genetic Programming symbolic transformer.

    This class implements a symbolic transformer using genetic programming.
    It evolves expressions to create new features from existing ones.

    Parameters
    ----------
    metric : str, default='pearson'
        The fitness metric to use for transformation. Supported metrics
        include 'pearson', 'spearman', and others related to distribution
        divergence or clustering quality.

    out_func : str, Operator, or callable, optional
        The output function to apply to the result of the expression.

    **kwargs
        Additional keyword arguments to be passed to the BaseSymbolic
        constructor.
    """

    def __init__(self, *, metric='pearson', out_func=None, **kwargs):
        self.typical_metrics = (
            'pearson', 'spearman', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 
            'separability', 'compactness', 'fisher', 'f_statistic', 'hellinger', 'bhattacharyya', 'js_divergence', 'wasserstein', 'earth_movers'
        )
        super().__init__(metric=metric, out_func=out_func, **kwargs)

    def fit(self, X, y=None, sample_weight=None,
            variable_names: Optional[List[str]] = None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples], optional
            Target values. For the transformer, `y` is often the same as `X`
            or a subset of its columns, but can be any data that the new
            features should be correlated with.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        variable_names : list of str, optional
            The names of the features.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        # Check arrays
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        
        return self._run(X, y, sample_weight, variable_names)

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data.

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_features_new]
            Transformed data.
        """
        check_is_fitted(
            self, 
            attributes=['equations_', 'n_features_in_', 
                        'feature_names_in_', 'hall_of_fame_']
        )

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))
        
        return self.get_best().expression.execute(X).reshape(-1, 1)


