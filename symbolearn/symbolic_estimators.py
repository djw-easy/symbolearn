import warnings
from typing import Optional, List, Union


import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, _check_sample_weight



from symbolearn.node import softmax
from symbolearn.base import BaseSymbolic
from symbolearn.utils import extract_and_aggregate_spatial



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
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None,
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
        
        return self._run(X=X, y=y, sample_weight=sample_weight,
                         variable_names=variable_names)
    
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
            attributes=['n_features_in_', 'feature_names_in_', 'hall_of_fame_']
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
    A Genetic Programming symbolic classifier with Fisher discriminant support.

    This class implements a symbolic classifier using genetic programming.
    It evolves a set of expressions to separate classes, with special support
    for Fisher discriminant analysis in symbolic feature space.

    Parameters
    ----------
    metric : str, default='hinge_loss'
        The fitness metric to use for classification.
        Supported metrics: 'cross_entropy', 'nll_loss', 'focal_loss',
        'hinge_loss', 'nca_loss', 'prototype_loss', 'accuracy'

    out_func : str, Operator, or callable, default='softmax'
        The output function to apply to the result of the expression.
        For the 'hinge_loss' metric, it is recommended to use 'identity' 
        (i.e., no transformation) to ensure correct calculation.

    **kwargs
        Additional keyword arguments passed to the BaseSymbolic constructor.
    """

    def __init__(self, *, metric='hinge_loss', out_func='identity', **kwargs):
        self.typical_metrics = (
            'cross_entropy', 'nll_loss', 'focal_loss', 'hinge_loss', 'accuracy'
        )
        super().__init__(metric=metric, out_func=out_func, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _is_spatial_mode(self):
        """True when predict / score should operate on 3-D spatial arrays."""
        return self._use_spatial_aggregation or self._spatial_aggregation_bfit

    def _check_X_for_predict(self, X):
        """Validate X for predict / predict_proba / decision_function.

        Accepts 3-D ``(H, W, D)`` arrays in spatial mode and 2-D
        ``(n_samples, n_features)`` arrays in tabular mode.
        In pre-fit spatial aggregation mode (``_spatial_aggregation_bfit``),
        the array is transformed before being returned.
        """
        check_is_fitted(
            self,
            attributes=['n_features_in_', 'feature_names_in_', 'hall_of_fame_']
        )

        X = np.asarray(X)
        n_features = X.shape[-1]

        if self._use_spectral_aggregation and n_features < 2:
            raise ValueError(
                'Spectral aggregation requires at least two features.'
            )

        if self._is_spatial_mode and X.ndim != 3:
            raise ValueError(
                f'Spatial aggregation requires a 3D array (H, W, D), got ndim={X.ndim}.'
            )

        if X.ndim == 3:
            if X.shape[2] != self.n_features_in_:
                raise ValueError(
                    f'Feature dimension D must match n_features_in_ '
                    f'({self.n_features_in_}), got {X.shape[2]}.'
                )
            X = check_array(
                X, allow_nd=True, ensure_2d=False,
                dtype='numeric', ensure_all_finite='allow-nan'
            )
            if self._spatial_aggregation_bfit:
                pixel_mask = np.int32(~np.any(np.isnan(X), axis=-1))
                X, _ = extract_and_aggregate_spatial(
                    X, pixel_mask,
                    method=self.spatial_stats,
                    window_size=self.valid_window_sizes,
                    ignore_label=0
                )
        else:
            # Tabular mode: standard 2-D validation
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f'Feature count must match n_features_in_ '
                    f'({self.n_features_in_}), got {X.shape[1]}.'
                )
            X = check_array(X)

        return X

    def _check_X_y_for_fit(self, X, y, sample_weight, is_classification=False):
        """Validate and reshape X / y / sample_weight before fitting.

        Supports 2-D tabular ``(n_samples, n_features)`` and 3-D raster
        ``(H, W, D)`` inputs.  Pixels where y is ``np.nan`` are treated as
        invalid and excluded from sklearn validation.

        Returns
        -------
        X : np.ndarray
            ``(H, W, D)`` in spatial-aggregation mode, ``(N_valid, D)`` otherwise.
        y : np.ndarray
            ``(H, W)`` in spatial-aggregation mode, ``(N_valid,)`` otherwise.
        sample_weight : np.ndarray or None
        """
        n_features = X.shape[-1]

        if self._use_spectral_aggregation and n_features < 2:
            raise ValueError(
                'Spectral aggregation requires at least two features.'
            )

        if (self._use_spatial_aggregation or self._spatial_aggregation_bfit) \
                and X.ndim != 3:
            raise ValueError(
                f'Spatial aggregation requires a 3D input array (H, W, D), got ndim={X.ndim}.'
            )

        # --- Build valid-pixel mask and extract flat arrays for validation ---
        if X.ndim == 3:
            if y.ndim != 2:
                raise ValueError(
                    f'For 3D X (H, W, D), y must be 2D (H, W). Got y.ndim={y.ndim}.'
                )
            if X.shape[:2] != y.shape:
                raise ValueError(
                    f'Spatial dimensions of X {X.shape[:2]} and y {y.shape} must match.'
                )

            valid_mask = ~np.isnan(y)
            if not np.any(valid_mask):
                raise ValueError('No valid samples found in y (all NaN).')

            X_valid = X[valid_mask].reshape(-1, X.shape[2])
            y_valid = y[valid_mask].ravel()

            if sample_weight is not None:
                if np.ndim(sample_weight) != 2 or sample_weight.shape != y.shape:
                    raise ValueError(
                        f'For 3D input, sample_weight must be 2D {y.shape}, '
                        f'got shape {np.shape(sample_weight)}.'
                    )
                sample_weight = sample_weight[valid_mask].ravel()

        else:  # 2-D tabular
            if y.ndim > 2:
                raise ValueError(
                    f'y must be 1D or 2D for 2D input X. Got y.ndim={y.ndim}.'
                )
            valid_mask = (
                ~np.isnan(y) if y.ndim == 1
                else ~np.any(np.isnan(y), axis=1)
            )
            if not np.any(valid_mask):
                raise ValueError('No valid samples found in y (all NaN).')

            X_valid = X[valid_mask]
            y_valid = y[valid_mask].ravel()

            if sample_weight is not None:
                if sample_weight.ndim != 1:
                    raise ValueError('sample_weight must be 1D for 2D input X.')
                sample_weight = sample_weight[valid_mask]

        # --- Sklearn array validation on valid samples only ---
        X_valid, y_valid = check_X_y(
            X_valid, y_valid, multi_output=self.is_multi_output_
        )

        if np.any(np.isnan(X_valid)):
            n_bad = np.any(np.isnan(X_valid), axis=1).sum()
            raise ValueError(
                f'Data integrity check failed: {n_bad} samples have valid y '
                'but NaN values in X.'
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X_valid)

        if is_classification:
            check_classification_targets(y_valid)
            if np.unique(y_valid).shape[0] < 2:
                raise ValueError('The number of classes must be greater than 1.')

        # --- Assemble the arrays forwarded to _run ---
        if self._use_spatial_aggregation or self._spatial_aggregation_bfit:
            y_out = np.full_like(y, fill_value=np.nan, dtype=np.float64)
            y_out[valid_mask] = y_valid
            if self._use_spatial_aggregation:
                X_out = X
            else:
                X_out, _ = extract_and_aggregate_spatial(
                    X, y,
                    method=self.spatial_stats,
                    window_size=self.valid_window_sizes
                )
            return X_out, y_out, sample_weight

        return X_valid, y_valid, sample_weight

    @staticmethod
    def _scores_to_class_indices(scores, metric):
        """Convert raw expression scores to integer class indices.

        Parameters
        ----------
        scores : np.ndarray, shape (N,) or (N, n_classes)
        metric : str

        Returns
        -------
        indices : np.ndarray of int, shape (N,)
        """
        if metric == 'hinge_loss':
            # Binary: sign as decision boundary; multi-class: argmax
            return (
                (scores > 0).astype(np.intp) if scores.ndim == 1
                else np.argmax(scores, axis=1)
            )
        # softmax / sigmoid: threshold 0.5 for scalar, argmax for vector
        return (
            (scores > 0.5).astype(np.intp) if scores.ndim == 1
            else np.argmax(scores, axis=1)
        )

    @staticmethod
    def _scores_to_proba(scores, metric, out_func):
        """Convert raw expression scores to a probability matrix.

        Parameters
        ----------
        scores : np.ndarray, shape (N,) or (N, n_classes)
        metric : str
        out_func : str or callable

        Returns
        -------
        proba : np.ndarray, shape (N, n_classes)

        Note
        ----
        For ``hinge_loss`` the values are pseudo-probabilities (not calibrated).
        """
        from scipy.special import expit
        from scipy.special import softmax as scipy_softmax

        def _to_binary_proba(s):
            pos = expit(s).reshape(-1, 1)
            return np.hstack([1 - pos, pos])

        if metric == 'hinge_loss' and out_func in ['identity', None]:
            return (
                _to_binary_proba(scores) if scores.ndim == 1
                else scipy_softmax(scores, axis=1)
            )

        if out_func in ['softmax', 'log_softmax']:
            # Scores are already probabilities; ensure 2-D shape
            if scores.ndim == 1:
                return np.hstack([1 - scores.reshape(-1, 1), scores.reshape(-1, 1)])
            return scores

        # sigmoid or other activations
        return (
            _to_binary_proba(scores) if scores.ndim == 1
            else scipy_softmax(scores, axis=1)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray = None,
            variable_names: Optional[List[str]] = None):
        """Fit the Genetic Program according to X, y.

        Supports 2-D tabular ``(n_samples, n_features)`` and 3-D raster
        ``(H, W, D)`` inputs.  Invalid pixels must be marked with ``np.nan``
        in y and will be excluded.  If y is valid at a pixel, X must be
        finite at that pixel too.

        When spatial aggregation is active, the full spatial arrays
        (with NaN-masked pixels) are forwarded to ``_run`` so that spatial
        context is preserved.  ``predict`` and related methods then also
        expect 3-D input.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (H, W, D)
        y : array-like, shape (n_samples,) or (H, W)
        sample_weight : array-like, shape (n_samples,) or (H, W), optional
        variable_names : list of str, optional

        Returns
        -------
        self : object
        """
        X, y, sample_weight = self._check_X_y_for_fit(X, y, sample_weight, True)

        # In pre-fit aggregation mode the spatial layout is no longer needed
        if self._spatial_aggregation_bfit:
            finite_mask = ~np.isnan(y)
            X, y = X[finite_mask], y[finite_mask]

        # Encode class labels, preserving NaN pixels in spatial mode
        finite_mask = ~np.isnan(y)
        self.classes_, y_encoded = np.unique(y[finite_mask], return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        y[finite_mask] = y_encoded

        if self.metric == 'hinge_loss' and self.out_func not in ['identity', None]:
            warnings.warn(
                "For hinge_loss, 'out_func' should be 'identity'. "
                "The current setting may affect performance.",
                UserWarning
            )

        if self.n_classes_ == 2:
            if self.out_func == 'sigmoid':
                self.order = None
                self.is_multi_output_ = False
            else:
                self.order = 2
                self.is_multi_output_ = True
        elif self.n_classes_ > 2:
            if self.metric != 'hinge_loss' and \
                    self.out_func not in ['softmax', 'log_softmax']:
                warnings.warn(
                    "For multi-class problems 'out_func' should be 'softmax' "
                    "or 'log_softmax'. It has been set to 'softmax'.",
                    UserWarning
                )
                self.out_func = 'softmax'
                self._out_func = softmax
            
            if self.order is None:
                self.order = self.n_classes_
                self.is_multi_output_ = True
            else:
                if not isinstance(self.order, int):
                    raise ValueError(
                        "'order' must be an integer equal to 'n_classes_' "
                        "for multi-class problems."
                    )
                if self.order != self.n_classes_:
                    self.order = self.n_classes_
                    self.is_multi_output_ = True

        return self._run(X=X, y=y, sample_weight=sample_weight,
                         variable_names=variable_names)

    def predict(self, X, index: Optional[int] = None, include_dominated=False):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (H, W, D)
            Must be 3-D in spatial mode.

        index : int, optional
            Hall-of-fame index. If None, the best model is used.

        include_dominated : bool, default=False

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,) or (H, W)
            In spatial mode, invalid pixels (all-NaN in X) are set to np.nan.
        """
        X = self._check_X_for_predict(X)
        scores = self.get_best(index, include_dominated).expression.execute(X)

        if self._is_spatial_mode:
            valid_mask = ~np.any(np.isnan(X), axis=2)          # (H, W)
            class_indices = self._scores_to_class_indices(
                scores[valid_mask], self.metric
            )

            # float64 supports NaN for numeric classes; use object for string classes
            out_dtype = (
                np.float64 if np.issubdtype(self.classes_.dtype, np.number)
                else object
            )
            out = np.full(X.shape[:2], fill_value=np.nan, dtype=out_dtype)
            out[valid_mask] = self.classes_.take(class_indices)
            return out

        return self.classes_.take(
            self._scores_to_class_indices(scores, self.metric)
        )

    def predict_proba(self, X, index: Optional[int] = None, include_dominated=False):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (H, W, D)
            Must be 3-D in spatial mode.

        index : int, optional
            Hall-of-fame index. If None, the best model is used.

        include_dominated : bool, default=False

        Returns
        -------
        proba : np.ndarray
            Tabular: ``(n_samples, n_classes)``.
            Spatial: ``(H, W, n_classes)``; NaN pixels have probability NaN.

        Note
        ----
        For ``hinge_loss`` the values are pseudo-probabilities (not calibrated).
        """
        X = self._check_X_for_predict(X)
        scores = self.get_best(index, include_dominated).expression.execute(X)

        if self._is_spatial_mode:
            valid_mask = ~np.any(np.isnan(X), axis=2)          # (H, W)
            proba_valid = self._scores_to_proba(
                scores[valid_mask], self.metric, self.out_func
            )                                                   # (N_valid, n_classes)

            H, W = X.shape[:2]
            out = np.full(
                (H, W, proba_valid.shape[1]), fill_value=np.nan, dtype=np.float64
            )
            out[valid_mask] = proba_valid
            return out

        return self._scores_to_proba(scores, self.metric, self.out_func)

    def score(self, X, y, sample_weight=None):
        """Return mean accuracy, ignoring NaN pixels in spatial mode.

        Overrides ``ClassifierMixin.score`` so that NaN entries in y are
        excluded before the accuracy calculation rather than raising an error.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (H, W, D)
        y : array-like, shape (n_samples,) or (H, W)
            True labels. NaN values are ignored in spatial mode.
        sample_weight : array-like, optional

        Returns
        -------
        score : float
        """
        from sklearn.metrics import accuracy_score

        y = np.asarray(y)
        y_pred = self.predict(X)

        if self._is_spatial_mode:
            valid_mask = ~np.isnan(y)                           # (H, W)
            weights = (
                sample_weight[valid_mask].ravel()
                if sample_weight is not None else None
            )
            return accuracy_score(
                y[valid_mask].ravel(),
                y_pred[valid_mask].ravel(),
                sample_weight=weights
            )

        return accuracy_score(
            y[~np.isnan(y)].ravel(), y_pred, sample_weight=sample_weight
        )



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
        
        return self._run(X=X, y=y, sample_weight=sample_weight,
                         variable_names=variable_names)

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
            attributes=['n_features_in_', 'feature_names_in_', 'hall_of_fame_']
        )

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))
        
        return self.get_best().expression.execute(X).reshape(-1, 1)


