import re
import numpy as np
from scipy import sparse
from typing import Union, Optional, List
from numba import vectorize, njit, float32, float64


# ----------------------------------------------------------------------
# Base Class
# ----------------------------------------------------------------------

class NodeContent(object):
    """Base class for the content of a SymbolicNode."""
    def __call__(self, X, valid_mask=None):
        """
        Execute the node.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features) or (H, W, C).
        valid_mask : np.ndarray of bool, shape (H, W), optional
            When X is 3-D and valid_mask is provided, the output is
            flattened to 2-D retaining only valid (True) pixels.
        """
        pass


# ----------------------------------------------------------------------
# Terminal Nodes (Variable, Constant, DynamicAggregation)
# ----------------------------------------------------------------------


class Variable(NodeContent):
    __slots__ = ['variable', '__name']
    def __init__(self, variable: int, name: str = None):
        self.variable = variable
        self.__name = name
    
    def __call__(self, X, valid_mask: Optional[np.ndarray] = None):
        """
        Extract the variable's feature column from X.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features) or (H, W, C).
        valid_mask : np.ndarray of bool, shape (H, W), optional
            Only used when X is 3-D.  When provided, the spatial map
            X[..., self.variable] is boolean-indexed by valid_mask,
            returning a 1-D array of shape (n_valid,) containing only
            the values at valid pixel locations.
            When None and X is 3-D, the full (H, W) slice is returned
            unchanged (no flattening).

        Returns
        -------
        np.ndarray
            - X is 2-D, valid_mask ignored  → shape (n_samples,)
            - X is 3-D, valid_mask provided → shape (n_valid,)   [1-D]
            - X is 3-D, valid_mask is None  → shape (H, W)
        """
        if X.ndim == 3 and valid_mask is not None:
            # Extract the single channel and flatten to valid pixels only
            channel = X[..., self.variable]           # (H, W)
            return channel[valid_mask]                 # (n_valid,)
        return X[..., self.variable]
    
    @property
    def name(self):
        return self.__name if self.__name is not None else f'x{self.variable}'
    
    @property
    def degree(self):
        return 0

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.variable == other.variable
        return False


class Constant(NodeContent):
    __slots__ = ['value']
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, X):
        return self.value
    
    @property
    def name(self):
        return str(self.value)
    
    @property
    def degree(self):
        return 0

    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value
        return False


"""
node.py
=======
Defines DynamicAggregation, an operator that combines:
  - Spatial aggregation  : rectangular sliding-window statistics over a 2-D
                           spatial grid, ignoring NaN values inside the window.
  - Spectral aggregation : reduction along the feature (band) axis.

Both steps are optional and can be used independently or together.

Input conventions
-----------------
* When spatial aggregation is **active**, X must be 3-D:
      shape (H, W, C)   — height × width × n_channels/features

* When spatial aggregation is **inactive**, X must be 2-D:
      shape (n_samples, n_features)

Mask-gated spatial computation
--------------------------------
``__call__`` accepts an optional ``y`` array of shape **(H, W)** that carries
per-pixel labels (or any reference values).  When supplied, the spatial step
**only computes window statistics for pixels where y is not NaN**.  All other
output pixels are left as NaN without touching the inner window loop.

This can reduce computation dramatically when labelled pixels are sparse
(e.g. a classification mask where most pixels are background / unlabelled).

Window convention
-----------------
``window_size`` is the **half-size** of the rectangular kernel along each
spatial axis.  The full window spans:

    rows : [i - window_size, i + window_size]   (inclusive, clamped to grid)
    cols : [j - window_size, j + window_size]   (inclusive, clamped to grid)

So ``window_size=1`` → 3×3, ``window_size=2`` → 5×5, etc.
NaN pixels *inside* the window are silently skipped; if the entire window
contains only NaN the output pixel is also NaN.
"""



# ---------------------------------------------------------------------------
# Numba-accelerated window aggregation kernels
# ---------------------------------------------------------------------------
#
# Design note — two separate functions
# -------------------------------------
# Numba's @njit performs static type inference at *compile time* and cannot
# unify a function that returns ndim=2 in one branch and ndim=3 in another.
# We therefore provide two kernels with fixed return shapes:
#
#   _window_agg_single : operates on ONE feature channel  → output (H, W)
#   _window_agg_all    : operates on ALL feature channels → output (H, W, C)
#
# Design note — valid_mask parameter
# ------------------------------------
# Both kernels accept a boolean ``valid_mask`` of shape (H, W).
# At the very start of the per-pixel loop body we test:
#
#     if not valid_mask[i, j]: continue
#
# This single branch skips the entire inner window scan for unmasked pixels,
# which eliminates the dominant cost when labelled pixels are sparse.
# The output is initialised to NaN, so skipped pixels automatically carry NaN.
#
# stat_code encoding
# ------------------
#   0 = nanmean
#   1 = nanmax
#   2 = nanmin
#   3 = nansum
#   4 = nanmedian  (collected into a temp buffer, then insertion-sorted)
#   5 = nanrange   (nanmax - nanmin)


@njit(fastmath=True, cache=True)
def _window_agg_single(X, valid_mask, window_size, stat_code, feat_idx):
    """
    Sliding rectangular-window aggregation for a **single** feature channel,
    restricted to pixels selected by ``valid_mask``.

    Parameters
    ----------
    X : float64 array, shape (H, W, C)
        Input spatial-spectral data cube.
    valid_mask : bool array, shape (H, W)
        Pixels where ``valid_mask[i, j]`` is True will have their window
        statistic computed.  All other pixels are left as NaN in the output.
        Pass an all-True mask to process every pixel unconditionally.
    window_size : int
        Half-width/height of the rectangular window (>= 1).
        Full window extent along each axis = 2*window_size + 1.
    stat_code : int
        Aggregation operation: 0=mean, 1=max, 2=min, 3=sum, 4=median, 5=range.
    feat_idx : int
        Index of the feature channel to aggregate (0 <= feat_idx < C).

    Returns
    -------
    result : float64 array, shape (H, W)
        Per-pixel window statistics for the selected channel.
        Pixels excluded by ``valid_mask`` are NaN.
    """
    H, W, C = X.shape
    result = np.full((H, W), np.nan, dtype=np.float64)

    max_win = (2 * window_size + 1) ** 2
    buf = np.empty(max_win, dtype=np.float64)

    for i in range(H):
        r_lo = max(0, i - window_size)
        r_hi = min(H - 1, i + window_size)

        for j in range(W):
            if not valid_mask[i, j]:
                continue

            c_lo = max(0, j - window_size)
            c_hi = min(W - 1, j + window_size)

            count = 0
            for r in range(r_lo, r_hi + 1):
                for c in range(c_lo, c_hi + 1):
                    v = X[r, c, feat_idx]
                    if not np.isnan(v):
                        buf[count] = v
                        count += 1

            if count == 0:
                continue

            if stat_code == 0:
                s = 0.0
                for k in range(count):
                    s += buf[k]
                result[i, j] = s / count

            elif stat_code == 1:
                mx = -np.inf
                for k in range(count):
                    if buf[k] > mx:
                        mx = buf[k]
                result[i, j] = mx

            elif stat_code == 2:
                mn = np.inf
                for k in range(count):
                    if buf[k] < mn:
                        mn = buf[k]
                result[i, j] = mn

            elif stat_code == 3:
                s = 0.0
                for k in range(count):
                    s += buf[k]
                result[i, j] = s

            elif stat_code == 4:
                for a in range(1, count):
                    key = buf[a]
                    b = a - 1
                    while b >= 0 and buf[b] > key:
                        buf[b + 1] = buf[b]
                        b -= 1
                    buf[b + 1] = key
                mid = count // 2
                if count % 2 == 1:
                    result[i, j] = buf[mid]
                else:
                    result[i, j] = (buf[mid - 1] + buf[mid]) / 2.0

            elif stat_code == 5:
                mn = np.inf
                mx = -np.inf
                for k in range(count):
                    if buf[k] < mn:
                        mn = buf[k]
                    if buf[k] > mx:
                        mx = buf[k]
                result[i, j] = mx - mn

    return result


@njit(fastmath=True, cache=True)
def _window_agg_all(X, valid_mask, window_size, stat_code):
    """
    Sliding rectangular-window aggregation across **all** feature channels,
    restricted to pixels selected by ``valid_mask``.

    Parameters
    ----------
    X : float64 array, shape (H, W, C)
    valid_mask : bool array, shape (H, W)
    window_size : int
    stat_code : int

    Returns
    -------
    result : float64 array, shape (H, W, C)
    """
    H, W, C = X.shape
    result = np.full((H, W, C), np.nan, dtype=np.float64)

    max_win = (2 * window_size + 1) ** 2
    buf = np.empty(max_win, dtype=np.float64)

    for i in range(H):
        r_lo = max(0, i - window_size)
        r_hi = min(H - 1, i + window_size)

        for j in range(W):
            if not valid_mask[i, j]:
                continue

            c_lo = max(0, j - window_size)
            c_hi = min(W - 1, j + window_size)

            for f in range(C):
                count = 0
                for r in range(r_lo, r_hi + 1):
                    for c in range(c_lo, c_hi + 1):
                        v = X[r, c, f]
                        if not np.isnan(v):
                            buf[count] = v
                            count += 1

                if count == 0:
                    continue

                if stat_code == 0:
                    s = 0.0
                    for k in range(count):
                        s += buf[k]
                    result[i, j, f] = s / count

                elif stat_code == 1:
                    mx = -np.inf
                    for k in range(count):
                        if buf[k] > mx:
                            mx = buf[k]
                    result[i, j, f] = mx

                elif stat_code == 2:
                    mn = np.inf
                    for k in range(count):
                        if buf[k] < mn:
                            mn = buf[k]
                    result[i, j, f] = mn

                elif stat_code == 3:
                    s = 0.0
                    for k in range(count):
                        s += buf[k]
                    result[i, j, f] = s

                elif stat_code == 4:
                    for a in range(1, count):
                        key = buf[a]
                        b = a - 1
                        while b >= 0 and buf[b] > key:
                            buf[b + 1] = buf[b]
                            b -= 1
                        buf[b + 1] = key
                    mid = count // 2
                    if count % 2 == 1:
                        result[i, j, f] = buf[mid]
                    else:
                        result[i, j, f] = (buf[mid - 1] + buf[mid]) / 2.0

                elif stat_code == 5:
                    mn = np.inf
                    mx = -np.inf
                    for k in range(count):
                        if buf[k] < mn:
                            mn = buf[k]
                        if buf[k] > mx:
                            mx = buf[k]
                    result[i, j, f] = mx - mn

    return result


def _spatial_data_mask(X, valid_mask=None):
    """Return the NaN-free spatial mask, scanning only requested rows.

    During training, hyperspectral labels are usually very sparse.  Building
    ``~np.any(np.isnan(X), axis=-1)`` would scan the entire cube for every
    candidate expression.  When a label/ROI mask is available, checking only
    those rows is equivalent and reduces the work from ``H * W * C`` to
    ``n_valid * C``.
    """
    if X.ndim != 3:
        return None

    spatial_shape = X.shape[:2]
    if valid_mask is None:
        return np.ascontiguousarray(~np.any(np.isnan(X), axis=-1))

    if valid_mask.shape != spatial_shape:
        raise ValueError(
            f"valid_mask.shape {valid_mask.shape} does not match "
            f"the spatial dimensions of X {spatial_shape}."
        )

    valid_mask = np.ascontiguousarray(valid_mask, dtype=bool)
    data_mask = np.zeros(spatial_shape, dtype=bool)
    if np.any(valid_mask):
        data_mask[valid_mask] = ~np.any(np.isnan(X[valid_mask]), axis=-1)
    return data_mask


@njit(fastmath=True, cache=True)
def _window_agg_single_masked(
    X, flat_indices, window_size, stat_code, feat_idx
):
    """Compact counterpart of ``_window_agg_single`` for sparse masks."""
    H, W, _ = X.shape
    result = np.full(flat_indices.size, np.nan, dtype=np.float64)
    max_win = (2 * window_size + 1) ** 2
    buf = np.empty(max_win, dtype=np.float64)

    for point_idx in range(flat_indices.size):
        flat_idx = flat_indices[point_idx]
        i = flat_idx // W
        j = flat_idx - i * W
        r_lo = max(0, i - window_size)
        r_hi = min(H - 1, i + window_size)
        c_lo = max(0, j - window_size)
        c_hi = min(W - 1, j + window_size)

        count = 0
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                value = X[r, c, feat_idx]
                if not np.isnan(value):
                    buf[count] = value
                    count += 1

        if count == 0:
            continue
        if stat_code == 0 or stat_code == 3:
            total = 0.0
            for k in range(count):
                total += buf[k]
            result[point_idx] = total / count if stat_code == 0 else total
        elif stat_code == 1:
            value = -np.inf
            for k in range(count):
                if buf[k] > value:
                    value = buf[k]
            result[point_idx] = value
        elif stat_code == 2:
            value = np.inf
            for k in range(count):
                if buf[k] < value:
                    value = buf[k]
            result[point_idx] = value
        elif stat_code == 4:
            for a in range(1, count):
                key = buf[a]
                b = a - 1
                while b >= 0 and buf[b] > key:
                    buf[b + 1] = buf[b]
                    b -= 1
                buf[b + 1] = key
            mid = count // 2
            result[point_idx] = (
                buf[mid] if count % 2 == 1
                else (buf[mid - 1] + buf[mid]) / 2.0
            )
        elif stat_code == 5:
            minimum = np.inf
            maximum = -np.inf
            for k in range(count):
                if buf[k] < minimum:
                    minimum = buf[k]
                if buf[k] > maximum:
                    maximum = buf[k]
            result[point_idx] = maximum - minimum

    return result


@njit(fastmath=True, cache=True)
def _window_agg_range_masked(
    X, flat_indices, window_size, stat_code, feat_start, feat_end
):
    """Aggregate a contiguous feature range directly into compact rows."""
    H, W, _ = X.shape
    n_features = feat_end - feat_start
    result = np.full(
        (flat_indices.size, n_features), np.nan, dtype=np.float64
    )
    max_win = (2 * window_size + 1) ** 2
    buf = np.empty(max_win, dtype=np.float64)

    for point_idx in range(flat_indices.size):
        flat_idx = flat_indices[point_idx]
        i = flat_idx // W
        j = flat_idx - i * W
        r_lo = max(0, i - window_size)
        r_hi = min(H - 1, i + window_size)
        c_lo = max(0, j - window_size)
        c_hi = min(W - 1, j + window_size)

        for output_feature in range(n_features):
            feature = feat_start + output_feature
            count = 0
            for r in range(r_lo, r_hi + 1):
                for c in range(c_lo, c_hi + 1):
                    value = X[r, c, feature]
                    if not np.isnan(value):
                        buf[count] = value
                        count += 1

            if count == 0:
                continue
            if stat_code == 0 or stat_code == 3:
                total = 0.0
                for k in range(count):
                    total += buf[k]
                result[point_idx, output_feature] = (
                    total / count if stat_code == 0 else total
                )
            elif stat_code == 1:
                value = -np.inf
                for k in range(count):
                    if buf[k] > value:
                        value = buf[k]
                result[point_idx, output_feature] = value
            elif stat_code == 2:
                value = np.inf
                for k in range(count):
                    if buf[k] < value:
                        value = buf[k]
                result[point_idx, output_feature] = value
            elif stat_code == 4:
                for a in range(1, count):
                    key = buf[a]
                    b = a - 1
                    while b >= 0 and buf[b] > key:
                        buf[b + 1] = buf[b]
                        b -= 1
                    buf[b + 1] = key
                mid = count // 2
                result[point_idx, output_feature] = (
                    buf[mid] if count % 2 == 1
                    else (buf[mid - 1] + buf[mid]) / 2.0
                )
            elif stat_code == 5:
                minimum = np.inf
                maximum = -np.inf
                for k in range(count):
                    if buf[k] < minimum:
                        minimum = buf[k]
                    if buf[k] > maximum:
                        maximum = buf[k]
                result[point_idx, output_feature] = maximum - minimum

    return result


# ---------------------------------------------------------------------------
# DynamicAggregation
# ---------------------------------------------------------------------------

class DynamicAggregation(NodeContent):
    """
    Combined spatial (rectangular-window) and spectral (feature-axis) aggregation.

    Spatial Aggregation
    -------------------
    Input must be **(H, W, C)**:
      - A rectangular window of half-size ``window_size`` is centred on each
        pixel.  Full extent: (2*window_size+1) × (2*window_size+1).
      - NaN pixels *inside* the window are silently ignored per channel.
      - If ``target_feature`` is given, only that channel is aggregated and
        the output is **(H, W)** — no spectral step can follow.
      - Otherwise all channels are aggregated simultaneously and the output
        remains **(H, W, C)** (or **(H, W, n_bands)** when restricted to the
        spectral slice before aggregation).
      - When ``valid_mask`` (shape ``(H, W)``) is passed to ``__call__``,
        only pixels where ``valid_mask`` is **True** are processed; all other
        output pixels are NaN.  Passing ``valid_mask=None`` processes every
        pixel unconditionally (inferred from non-NaN data presence).

    Spectral Aggregation
    --------------------
    Operates on the last axis of a 2-D or 3-D array:
      - Band slice [v_start, v_end] (inclusive, 0-based) is extracted.
      - Optional finite-difference derivative of order ``deriv_order`` is
        applied (e.g. d1 = first differences, d2 = second differences).
      - A reduction statistic collapses the last axis → output loses one dim.

    Combined mode (both active, no target_feature)
    -----------------------------------------------
    1. Restrict input to the spectral slice: (H, W, C) → (H, W, n_bands).
    2. Spatial window aggregation on the slice → (H, W, n_bands).
    3. Spectral aggregation collapses bands → (H, W).

    3-D input with valid_mask — flattening behaviour
    -------------------------------------------------
    When X is 3-D **and** ``valid_mask`` is provided, the aggregation
    pipeline runs as normal but the final result is boolean-indexed by
    ``valid_mask`` before returning.  This flattens the spatial axes and
    returns only the valid-pixel rows:

      - spatial-only, target_feature set → (n_valid,)
      - spatial-only, no target_feature  → (n_valid, C)
      - spatial + spectral               → (n_valid,)
      - spectral-only on 3-D input       → (n_valid,)
      - identity on 3-D input            → (n_valid, C)

    When X is 2-D or ``valid_mask`` is None, the original shapes are returned.

    Identity mode (both inactive)
    ------------------------------
    X is returned unchanged unless X is 3-D and valid_mask is provided,
    in which case the valid rows are extracted.

    Parameters
    ----------
    v_start : int, optional
    v_end : int, optional
    stat_name_spectral : str, optional
    window_size : int, optional
    stat_name_spatial : str, optional
    target_feature : int, optional
    n_variables : int
    """

    __slots__ = [
        'v_start', 'v_end', 'stat_name_spectral', 'n_variables',
        'percentile_q', 'deriv_order', 'actual_stat_name_spectral',
        'window_size', 'stat_name_spatial', 'stat_code_spatial',
        'target_feature', '_spatial_active', '_spectral_active',
    ]

    _spectral_stat_map: dict = {
        'mean':   np.mean,
        'max':    np.max,
        'min':    np.min,
        'median': np.median,
        'std':    np.std,
        'var':    np.var,
        'sum':    np.sum,
    }

    @staticmethod
    def _range_func(arr, axis=None):
        """
        Compute range (max - min) along given axis.
        
        Parameters
        ----------
        arr : np.ndarray
            Input array.
        axis : int, optional
            Axis along which to compute range.
        
        Returns
        -------
        result : np.ndarray
            Range values (max - min).
        """
        return np.max(arr, axis=axis) - np.min(arr, axis=axis)
    
    @staticmethod
    def _slope_func(arr, axis=None):
        """
        Compute linear regression slope along spectral dimension.
        Uses band indices (0, 1, 2, ..., n-1) as independent variable.
        Formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        
        Parameters
        ----------
        arr : np.ndarray
            Input array of shape (n_samples, n_bands).
        axis : int, optional
            Axis along which to compute slope (ignored, always axis=1).
        
        Returns
        -------
        slope : np.ndarray
            Slope values of shape (n_samples,).
        """
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Slope calculation requires 2D array with at least 2 bands")
        n_bands = arr.shape[1]
        x = np.arange(n_bands, dtype=np.float64)
        sum_x = x.sum()
        sum_x2 = (x * x).sum()
        denominator = n_bands * sum_x2 - sum_x * sum_x
        if np.isclose(denominator, 0.0):
            return np.zeros(arr.shape[0], dtype=np.float64)
        sum_y = arr.sum(axis=1)
        sum_xy = arr @ x
        return (n_bands * sum_xy - sum_x * sum_y) / denominator

    _spectral_stat_map.update({
        'range': _range_func.__func__,
        'slope': _slope_func.__func__,
    })
    _base_spectral_ops: list = list(_spectral_stat_map.keys())

    _spatial_stat_map: dict = {
        'mean':   0,
        'max':    1,
        'min':    2,
        'sum':    3,
        'median': 4,
        'range':  5,
    }
    _base_spatial_ops: list = list(_spatial_stat_map.keys())

    def __init__(
        self,
        v_start:            Optional[int] = None,
        v_end:              Optional[int] = None,
        stat_name_spectral: Optional[str] = None,
        window_size:        Optional[int] = None,
        stat_name_spatial:  Optional[str] = None,
        target_feature:     Optional[int] = None,
        n_variables:        int = 0,
    ):
        self.n_variables  = n_variables
        self.percentile_q = None
        self.deriv_order  = 0

        self.window_size       = None
        self.stat_name_spatial = None
        self.stat_code_spatial = None
        self.target_feature    = None
        self._spatial_active   = False

        if stat_name_spatial is not None:
            if window_size is None or window_size < 1:
                raise ValueError(
                    "window_size must be an integer >= 1 when spatial "
                    "aggregation is enabled."
                )
            if stat_name_spatial not in self._spatial_stat_map:
                raise ValueError(
                    f"Unsupported spatial operator '{stat_name_spatial}'. "
                    f"Choose from {self._base_spatial_ops}."
                )
            self.window_size       = window_size
            self.stat_name_spatial = stat_name_spatial
            self.stat_code_spatial = self._spatial_stat_map[stat_name_spatial]
            self.target_feature    = target_feature
            self._spatial_active   = True

        self.v_start                   = None
        self.v_end                     = None
        self.stat_name_spectral        = None
        self.actual_stat_name_spectral = None
        self._spectral_active          = False

        if stat_name_spectral is not None:
            if v_start is None or v_end is None:
                raise ValueError(
                    "v_start and v_end must be provided when spectral "
                    "aggregation is enabled."
                )

            deriv_match = re.match(r'^d(\d+)_(.+)$', stat_name_spectral)
            if deriv_match:
                self.deriv_order               = int(deriv_match.group(1))
                self.actual_stat_name_spectral = deriv_match.group(2)
            else:
                self.deriv_order               = 0
                self.actual_stat_name_spectral = stat_name_spectral

            if v_start < 0 or v_end >= n_variables or v_end <= v_start:
                raise ValueError(
                    f"Invalid band range [{v_start}, {v_end}] for "
                    f"n_variables={n_variables}. "
                    f"Requires 0 <= v_start < v_end < n_variables."
                )

            n_bands = v_end - v_start + 1

            if n_bands - self.deriv_order < 1:
                raise ValueError(
                    f"Band range [{v_start}, {v_end}] ({n_bands} bands) is "
                    f"too narrow for a {self.deriv_order}-order derivative "
                    f"(need at least {self.deriv_order + 1} bands)."
                )

            if self.actual_stat_name_spectral == 'slope':
                bands_after_deriv = n_bands - self.deriv_order
                if bands_after_deriv < 2:
                    raise ValueError(
                        f"Band range [{v_start}, {v_end}] with a "
                        f"{self.deriv_order}-order derivative leaves only "
                        f"{bands_after_deriv} band(s); slope requires >= 2."
                    )

            is_percentile = False
            if self.actual_stat_name_spectral.startswith('percentile'):
                pct_match = re.match(r'^percentile(\d+(?:\.\d+)?)$',
                                     self.actual_stat_name_spectral)
                if not pct_match:
                    raise ValueError(
                        f"Invalid percentile format "
                        f"'{self.actual_stat_name_spectral}'. "
                        f"Expected 'percentile<q>' where q ∈ [0, 100]."
                    )
                q = float(pct_match.group(1))
                if not (0.0 <= q <= 100.0):
                    raise ValueError(
                        f"Percentile q must be in [0, 100], got {q}."
                    )
                self.percentile_q = q
                is_percentile = True

            if not is_percentile and \
                    self.actual_stat_name_spectral not in self._base_spectral_ops:
                raise ValueError(
                    f"Unsupported spectral operator "
                    f"'{self.actual_stat_name_spectral}'. "
                    f"Choose from {self._base_spectral_ops} or 'percentile<q>'."
                )

            self.v_start            = v_start
            self.v_end              = v_end
            self.stat_name_spectral = stat_name_spectral
            self._spectral_active   = True

        if self.target_feature is not None:
            if not self._spatial_active:
                raise ValueError(
                    "target_feature is only meaningful when spatial "
                    "aggregation is enabled."
                )
            if self._spectral_active:
                raise ValueError(
                    "target_feature cannot be combined with spectral "
                    "aggregation: single-channel spatial output (H, W) "
                    "has no band axis to reduce over."
                )
            if not (0 <= self.target_feature < n_variables):
                raise ValueError(
                    f"target_feature={self.target_feature} is out of range "
                    f"[0, {n_variables - 1}]."
                )

    def __call__(
        self,
        X: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        data_mask: Optional[np.ndarray] = None,
        flat_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Execute the aggregation pipeline on X.

        Parameters
        ----------
        X : np.ndarray
            - Shape **(H, W, C)** when spatial aggregation is active, or when
            a 3-D cube is passed for spectral-only / identity processing.
            - Shape **(n_samples, n_features)** when only spectral aggregation
            is active or in identity (pass-through) mode.
        valid_mask : np.ndarray of bool, shape (H, W), optional
            Caller-supplied label / ROI mask indicating which pixels to include.

            **Mask construction:**
            Internally a ``kernel_mask`` is derived as follows:

            - ``valid_mask`` provided → ``kernel_mask = valid_mask & data_mask``
                where ``data_mask = ~np.any(np.isnan(X), axis=-1)``.
                Only pixels that are both labelled *and* NaN-free enter the
                computation.
            - ``valid_mask=None``    → ``kernel_mask = data_mask``.
                All NaN-free pixels are processed unconditionally.
            - X is 2-D              → ``kernel_mask = None`` (unused).

            **Spatial aggregation (``_spatial_active=True``):**
            ``kernel_mask`` is forwarded to the Numba window kernels so that
            the O(w²) scan is skipped for masked-out pixels.
            The spatial output retains the full (H, W[, C]) shape (NaN at
            invalid locations).

            **Spectral aggregation (``_spectral_active=True``):**
            ``band_data[kernel_mask]`` is applied *before* the reduction,
            collapsing (H, W, n_bands) → (n_valid, n_bands) first, then
            reducing along axis=-1 to (n_valid,).  This avoids computing
            statistics over NaN-filled rows.

            When ``valid_mask`` was supplied, the (n_valid,) result is
            returned directly (flattened, valid pixels only).
            When ``valid_mask=None``, the result is written back into a
            full (H, W) NaN array at the positions given by ``kernel_mask``
            and the spatial-shaped array is returned.

            **3-D output shapes summary:**

            +----------------------------------------------+------------------------+------------------+
            | Active steps                                 | valid_mask provided    | valid_mask=None  |
            +----------------------------------------------+------------------------+------------------+
            | spatial only, target_feature set             | (n_valid,)             | (H, W)           |
            | spatial only, no target_feature              | (n_valid, C)           | (H, W, C)        |
            | spatial + spectral                           | (n_valid,)             | (H, W)           |
            | spectral only / spatial+spectral, 3-D input  | (n_valid,)             | (H, W)           |
            | identity on 3-D input                        | (n_valid, C)           | (H, W, C)        |
            | any mode, 2-D input                          | shape unchanged        | shape unchanged  |
            +----------------------------------------------+------------------------+------------------+

        Returns
        -------
        np.ndarray
            See the table above for output shapes.
        """
        input_is_3d = X.ndim == 3

        # ------------------------------------------------------------------
        # Build kernel_mask: the boolean map of pixels that will actually be
        # processed.  For 3-D inputs we always compute data_mask so NaN rows
        # are excluded even when no valid_mask is supplied.
        # ------------------------------------------------------------------
        if input_is_3d:
            H, W = X.shape[:2]
            if valid_mask is not None:
                if valid_mask.shape != (H, W):
                    raise ValueError(
                        f"valid_mask.shape {valid_mask.shape} does not match "
                        f"the spatial dimensions of X ({H}, {W})."
                    )
                valid_mask  = np.ascontiguousarray(valid_mask, dtype=bool)

            if data_mask is None:
                data_mask = _spatial_data_mask(X, valid_mask)
            else:
                if data_mask.shape != (H, W):
                    raise ValueError(
                        f"data_mask.shape {data_mask.shape} does not match "
                        f"the spatial dimensions of X ({H}, {W})."
                    )
                data_mask = np.ascontiguousarray(data_mask, dtype=bool)

            kernel_mask = (
                data_mask if valid_mask is None or valid_mask is data_mask
                else valid_mask & data_mask
            )
        else:
            # 2-D path: kernel_mask is unused; valid_mask is silently ignored.
            kernel_mask = None

        current        = X
        already_sliced = False

        # ==================================================================
        # Step 1 — Spatial aggregation (rectangular window, NaN-aware,
        #          mask-gated via Numba kernel)
        # ==================================================================
        if self._spatial_active:
            if current.ndim != 3:
                raise ValueError(
                    f"Spatial aggregation requires a 3-D input (H, W, C), "
                    f"got shape {current.shape}."
                )

            compact_input = current
            if (current.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
                    or not current.flags.c_contiguous):
                compact_input = np.ascontiguousarray(current, dtype=np.float64)

            # Sparse training masks should produce compact output directly.
            # This avoids allocating an H x W x bands float64 cube only to
            # immediately discard almost all rows.
            if valid_mask is not None:
                if flat_indices is None:
                    flat_indices = np.flatnonzero(kernel_mask).astype(np.int64)
                else:
                    flat_indices = np.asarray(flat_indices, dtype=np.int64)

            if self.target_feature is not None:
                if valid_mask is not None:
                    return _window_agg_single_masked(
                        compact_input, flat_indices, self.window_size,
                        self.stat_code_spatial, self.target_feature,
                    )
                spatial_input = np.ascontiguousarray(current, dtype=np.float64)
                return _window_agg_single(
                    spatial_input, kernel_mask, self.window_size,
                    self.stat_code_spatial, self.target_feature,
                )
            else:
                if valid_mask is not None:
                    feature_start = self.v_start if self._spectral_active else 0
                    feature_end = (
                        self.v_end + 1 if self._spectral_active
                        else current.shape[-1]
                    )
                    current = _window_agg_range_masked(
                        compact_input, flat_indices, self.window_size,
                        self.stat_code_spatial, feature_start, feature_end,
                    )
                    already_sliced = self._spectral_active
                else:
                    if self._spectral_active:
                        spatial_input = np.ascontiguousarray(
                            current[:, :, self.v_start: self.v_end + 1],
                            dtype=np.float64,
                        )
                        already_sliced = True
                    else:
                        spatial_input = np.ascontiguousarray(
                            current, dtype=np.float64
                        )
                    current = _window_agg_all(
                        spatial_input, kernel_mask, self.window_size,
                        self.stat_code_spatial,
                    )

        # ==================================================================
        # Step 2 — Spectral aggregation (last-axis reduction)
        # ==================================================================
        if self._spectral_active:
            if already_sliced:
                # Spatial step already restricted last axis to the band slice.
                band_data = current
            else:
                # Extract band slice; works for (H, W, C) and (n_samples, C).
                band_data = current[..., self.v_start: self.v_end + 1]

            # ------------------------------------------------------------------
            # Pre-reduction flattening for 3-D inputs
            # ------------------------------------------------------------------
            # Boolean-index BEFORE computing the statistic so that:
            #   (a) NaN-filled rows (invalid pixels) are excluded, preventing
            #       them from polluting nanmean / nanpercentile etc.
            #   (b) The reduction operates on a compact (n_valid, n_bands) array
            #       rather than the sparse (H, W, n_bands) cube, which is both
            #       faster and avoids allocating a full (H, W) intermediate.
            # After indexing, band_data is 2-D: (n_valid, n_bands).
            if kernel_mask is not None and current.ndim == 3:
                band_data = band_data[kernel_mask]   # (n_valid, n_bands)

            # Optional finite-difference derivative along the band axis.
            if self.deriv_order > 0:
                band_data = np.diff(band_data, n=self.deriv_order, axis=-1)

            # Reduce along the last (band) axis.
            if self.percentile_q is not None:
                result = np.nanpercentile(band_data, self.percentile_q, axis=-1)
            else:
                op = self._spectral_stat_map[self.actual_stat_name_spectral]
                result = op(band_data, axis=-1)
            # result is now (n_valid,) for 3-D inputs, (n_samples,) for 2-D.

            # ------------------------------------------------------------------
            # Write-back when valid_mask was NOT supplied
            # ------------------------------------------------------------------
            # The caller did not provide a label mask, so they expect the output
            # to have the same (H, W) spatial layout as the input rather than a
            # flattened (n_valid,) array.  Write the compact result back into a
            # full NaN array at the positions given by kernel_mask.
            if valid_mask is None and kernel_mask is not None:
                H, W = kernel_mask.shape
                result_out      = np.full((H, W), np.nan, dtype=np.float64)
                result_out[kernel_mask] = result
                return result_out   # (H, W)

            # valid_mask was supplied → return the compact (n_valid,) array.
            return result

        # ==================================================================
        # Step 3 — Return (spatial-only all-channels, or identity)
        # ==================================================================
        # Reaching here means:
        #   (a) Spatial-only (no target_feature) → current is (H, W, C).
        #   (b) Identity (both inactive)         → current is the original X.
        # In both cases the shape has spatial axes as the leading dimensions.
        if valid_mask is not None and current.ndim >= 3:
            # Flatten: (H, W, C) → (n_valid, C)  or  (H, W) → (n_valid,)
            return current[kernel_mask]   # (n_valid, C) or (n_valid,)
        return current

    # ------------------------------------------------------------------
    # Properties and utilities
    # ------------------------------------------------------------------

    @property
    def degree(self) -> int:
        return 0

    @property
    def name(self) -> str:
        parts = []
        if self._spatial_active:
            if self.target_feature is not None:
                parts.append(
                    f"{self.stat_name_spatial}_sp"
                    f"(w{self.window_size},f{self.target_feature})"
                )
            else:
                parts.append(
                    f"{self.stat_name_spatial}_sp(w{self.window_size})"
                )
        if self._spectral_active:
            parts.append(
                f"{self.stat_name_spectral}"
                f"(v{self.v_start + 1}-{self.v_end + 1})"
            )
        return "_".join(parts) if parts else "identity"

    def __repr__(self) -> str:
        return f"DynamicAggregation(name='{self.name}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DynamicAggregation):
            return False

        if self._spatial_active != other._spatial_active:
            return False
        if self._spatial_active:
            if (self.window_size       != other.window_size or
                    self.stat_name_spatial != other.stat_name_spatial or
                    self.target_feature    != other.target_feature):
                return False

        if self._spectral_active != other._spectral_active:
            return False
        if self._spectral_active:
            if (self.v_start            != other.v_start or
                    self.v_end              != other.v_end or
                    self.stat_name_spectral != other.stat_name_spectral):
                return False

        return True

    def _get_all_valid_spectral_ops(self) -> list:
        ops = self._base_spectral_ops.copy()
        ops.extend(['percentile25', 'percentile50', 'percentile75', 'percentile95'])
        return ops


# ----------------------------------------------------------------------
# Function Nodes (Math Functions)
# ----------------------------------------------------------------------


class Operator(NodeContent):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    degree : int
        The number of arguments that the ``function`` takes.

    """
    __slots__ = ['function', 'name', 'degree']
    def __init__(self, function, name, degree):
        self.function = function
        self.name = name
        self.degree = degree

    def __call__(self, *args):
        return self.function(*args)

    def __eq__(self, other):
        if isinstance(other, Operator):
            return self.name == other.name and self.degree == other.degree
        return False






def _protected_addition(x1, x2):
    """Protected addition with overflow handling.

    Computes x1 + x2 while clamping results that exceed a reasonable
    magnitude to zero, preventing numerical overflow in downstream operations.

    Parameters
    ----------
    x1, x2 : array-like
        Input arrays to add.

    Returns
    -------
    np.ndarray
        Element-wise sum, with values whose magnitude exceeds 1e10 replaced by 0.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.add(x1, x2)
        # Detect overflow: check if result is within reasonable magnitude range
        safe_mask = np.isfinite(result) & (np.abs(result) < 1e10)
        return np.where(safe_mask, result, 0.)


def _protected_subtraction(x1, x2):
    """Protected subtraction with overflow handling.

    Computes x1 - x2 while clamping results that exceed a reasonable
    magnitude to zero, preventing numerical overflow in downstream operations.

    Parameters
    ----------
    x1, x2 : array-like
        Input arrays for subtraction.

    Returns
    -------
    np.ndarray
        Element-wise difference, with values whose magnitude exceeds 1e10 replaced by 0.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.subtract(x1, x2)
        # Detect overflow: check if result is within reasonable magnitude range
        safe_mask = np.isfinite(result) & (np.abs(result) < 1e10)
        return np.where(safe_mask, result, 0.)


def _protected_multiplication(x1, x2):
    """Protected multiplication with overflow handling.

    Computes x1 * x2 with multi-stage overflow protection:
    1. Pre-check if either operand has magnitude exceeding 1e10
    2. Verify the resulting product is finite and within bounds

    Parameters
    ----------
    x1, x2 : array-like
        Input arrays to multiply.

    Returns
    -------
    np.ndarray
        Element-wise product, with values that overflow replaced by 0.
    """
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        # Pre-check: skip multiplication if either operand may cause overflow
        safe_mask = (np.abs(x1) < 1e10) & (np.abs(x2) < 1e10)
        # For potentially overflowing cases, further validate the product
        result = np.where(safe_mask, np.multiply(x1, x2), 0.)
        # Re-validate: ensure result is finite and within reasonable bounds
        result = np.where(np.isfinite(result) & (np.abs(result) < 1e10), result, 0.)
        return result

def _protected_division(x1, x2):
    """Protected division optimized for frequent invocation.

    Computes x1 / x2 with protection against division by zero and overflow.
    Returns 1.0 (the neutral element for multiplication) for invalid operations.

    Parameters
    ----------
    x1, x2 : array-like
        Numerator and denominator arrays.

    Returns
    -------
    np.ndarray
        Element-wise quotient. Returns 1.0 where x2 is near-zero, result is
        non-finite, or magnitude exceeds 1e10.
    """
    # Direct NumPy operations avoid unnecessary type checks and conditionals
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = np.divide(x1, x2)
        # Bitwise AND for efficiency instead of logical 'and'
        safe_mask = (np.abs(x2) > 1e-10) & np.isfinite(result) & (np.abs(result) < 1e10)
        return np.where(safe_mask, result, 1.0)


def _protected_sqrt(x1):
    """Protected square root for all real numbers.

    Computes sqrt(x) by first taking the absolute value of x, ensuring
    a real-valued result even for negative inputs.

    Parameters
    ----------
    x1 : array-like
        Input values (can be negative).

    Returns
    -------
    np.ndarray
        Element-wise square root of |x|, avoiding NaN from sqrt of negatives.
    """
    return np.sqrt(np.abs(x1))


def _protected_power(x1, x2):
    """Protected power function: safely computes x1 ** x2.

    Guards against overflow, complex numbers, and invalid values through
    multi-stage validation:
    1. Both base and exponent must be finite
    2. Negative bases require integer exponents (to avoid complex results)
    3. Magnitude constraints prevent intermediate overflow

    Parameters
    ----------
    x1 : array-like
        Base values.
    x2 : array-like
        Exponent values.

    Returns
    -------
    np.ndarray
        Element-wise power result. Returns 1.0 (identity) for invalid operations.
    """
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)

        # Initialize result to 1.0 (neutral element, also default for invalid ops)
        result = np.ones_like(x1, dtype=np.float64)

        # Condition 1: Both base and exponent must be finite
        finite_mask = np.isfinite(x1) & np.isfinite(x2)

        # Condition 2: Avoid negative bases with non-integer exponents (would yield complex)
        # Check if x2 is close to an integer (with floating-point tolerance)
        x2_is_integer = np.abs(x2 - np.rint(x2)) < 1e-10
        valid_base = (x1 >= 0) | x2_is_integer

        # Condition 3: Constrain magnitudes to prevent intermediate overflow
        # Empirical limits: |x1| < 1e5, |x2| < 100 (adjustable)
        magnitude_safe = (np.abs(x1) < 1e5) & (np.abs(x2) < 100)

        # Combine all safety conditions
        safe_mask = finite_mask & valid_base & magnitude_safe

        # Execute power only in safe region
        safe_result = np.power(x1, x2)

        # Re-validate: result must be finite and within reasonable bounds
        result_is_safe = np.isfinite(safe_result) & (np.abs(safe_result) < 1e10)

        # Final result: use computed value only when both input and output are safe
        result = np.where(safe_mask & result_is_safe, safe_result, 1.0)

        return result


def _protected_log(x1):
    """Protected natural logarithm handling zero and negative inputs.

    Computes log(|x|) for all real values, returning 0 where |x| is below
    the threshold (0.001) to prevent -inf and NaN propagation.

    Parameters
    ----------
    x1 : array-like
        Input values (can be negative or zero).

    Returns
    -------
    np.ndarray
        Element-wise natural log of |x|, with 0 returned for |x| <= 0.001.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Protected reciprocal handling zero inputs.

    Computes 1/x with special handling for values near zero, avoiding
    division by zero and overflow by returning 0 for |x| <= 0.001.

    Parameters
    ----------
    x1 : array-like
        Input values (should not be zero).

    Returns
    -------
    np.ndarray
        Element-wise reciprocal. Returns 0 for |x| <= 0.001.
    """
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = np.zeros_like(x1, dtype=np.float64)
        mask = np.abs(x1) > 0.001
        np.divide(1.0, x1, out=result, where=mask)
        return result


def _protected_exp(x1):
    """Protected exponential function with overflow clipping.

    Computes exp(x) after clipping x to the range where exp is still
    finite (x <= 700). Values above this threshold would overflow to inf.

    Parameters
    ----------
    x1 : array-like
        Input exponent values.

    Returns
    -------
    np.ndarray
        Element-wise exponential. Values are clipped at 700 to prevent overflow.
    """
    with np.errstate(over='ignore', under='ignore'):
        clipped_x1 = np.clip(x1, a_min=None, a_max=700.)
        return np.exp(clipped_x1)


def _protected_expsq(x1):
    """Protected Gaussian / RBF function: exp(-x^2).

    Computes exp(-x^2) with input clipping to [-30, 30] to prevent
    overflow in the exponentiation step (since -30^2 = -900 would
    still be within exp's range, but values outside would underflow).

    Parameters
    ----------
    x1 : array-like
        Input values for the Gaussian function.

    Returns
    -------
    np.ndarray
        Element-wise Gaussian kernel values, clipped to prevent overflow/underflow.
    """
    with np.errstate(over='ignore', under='ignore'):
        # Clip input range to prevent exponent overflow
        clipped_x1 = np.clip(x1, a_min=-30., a_max=30.)
        return np.exp(-np.square(clipped_x1))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    Computes sigmoid(x) = 1 / (1 + exp(-x)) using a piecewise strategy
    to maintain numerical stability:
    - For x >= 0: use 1 / (1 + exp(-x))
    - For x < 0: use exp(x) / (1 + exp(x))

    This avoids overflow for large positive inputs (exp(-x) -> 0) and
    precision loss for large negative inputs.

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape.

    Returns
    -------
    np.ndarray
        Element-wise sigmoid values in [0, 1].

    Examples
    --------
    >>> x = np.array([-1, 0, 1])
    >>> _sigmoid(x)
    array([0.26894142, 0.5, 0.73105858], dtype=float32)
    """
    EXP_LOWER_BOUND = -88.0
    EXP_UPPER_BOUND = 88.0

    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        x_clipped = np.clip(x, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
        pos_mask = (x_clipped >= 0)
        neg_mask = ~pos_mask

        result = np.empty_like(x, dtype=np.float32)

        # Positive branch: standard sigmoid formula
        result[pos_mask] = 1.0 / (1.0 + np.exp(-x_clipped[pos_mask]))

        # Negative branch: numerically stable alternative
        z = np.exp(x_clipped[neg_mask])
        result[neg_mask] = z / (1.0 + z)

        # Handle extreme values beyond clipping bounds
        result[x <= EXP_LOWER_BOUND] = 0.0
        result[x >= EXP_UPPER_BOUND] = 1.0

        return result.astype(np.float32)


def _softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation function: f(x) = ln(1 + exp(x)).

    A smooth approximation to the ReLU activation function that avoids
    the "dying ReLU" problem. Values are clipped at +/-88 to prevent
    overflow in the exp computation.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Element-wise softplus activation values.

    Examples
    --------
    >>> x = np.array([-2, -1, 0, 1, 2])
    >>> _softplus(x)
    array([0.12692801, 0.31326169, 0.69314718, 1.31326169, 2.12692801])
    """
    with np.errstate(over='ignore', under='ignore'):
        x_clipped = np.clip(x, -88, 88)
        return np.log(1.0 + np.exp(x_clipped))


# Softmax core implementation (handles 2D arrays only)
# Optimized for float32, also compatible with float64
@njit(fastmath=True, cache=True)
def _softmax_2d_impl(x):
    rows, cols = x.shape
    result = np.empty_like(x)
    
    for i in range(rows):
        row_max = -np.inf
        for j in range(cols):
            if x[i, j] > row_max:
                row_max = x[i, j]
        
        row_sum = 0.0
        for j in range(cols):
            val = x[i, j] - row_max
            # Clipping protection to prevent exp overflow
            if val < -700.0: val = -700.0
            elif val > 700.0: val = 700.0
            
            e_val = np.exp(val)
            result[i, j] = e_val
            row_sum += e_val
        
        if row_sum == 0.0:
            row_sum = 1.0
            
        factor = 1.0 / row_sum
        for j in range(cols):
            result[i, j] *= factor
            
    return result

# Softmax Python wrapper: handles scalars and arbitrary dimensions
def _softmax(x, axis=1):
    # Case 1: Scalar input
    # Mathematically, softmax of a single value is 1.0
    # If x is float32, return float32(1.0)
    if np.ndim(x) == 0:
        return np.array(1.0, dtype=np.asarray(x).dtype)

    x_arr = np.asarray(x)

    # Case 2: 1D array (treated as single sample) -> reshape to 2D
    if x_arr.ndim == 1:
        x_2d = x_arr.reshape(1, -1)
        res = _softmax_2d_impl(x_2d)
        return res.reshape(-1)  # Restore to 1D

    # Case 3: 2D array (standard case)
    if x_arr.ndim == 2:
        return _softmax_2d_impl(x_arr)

    # Case 4: Higher dimensions: fall back to NumPy implementation
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        e_x = np.exp(x_arr - np.max(x_arr, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)


class ZScore:
    name = "zscore"
    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            if x.ndim <= 1:
                m, s = np.mean(x), np.std(x)
                return np.where(s > 1e-8, (x - m) / s, 0.0).astype(np.float32)
            m = np.mean(x, axis=0, keepdims=True)
            s = np.std(x, axis=0, keepdims=True)
            return np.where(s > 1e-8, (x - m) / s, 0.0).astype(np.float32)

zscore = ZScore()

def _identity(x): return x
identity = Operator(function=_identity, name="identity", degree=1)
add2 = Operator(function=_protected_addition, name='add', degree=2)
sub2 = Operator(function=_protected_subtraction, name='sub', degree=2)
mul2 = Operator(function=_protected_multiplication, name='mul', degree=2)
div2 = Operator(function=_protected_division, name='div', degree=2)
sqrt1 = Operator(function=_protected_sqrt, name='sqrt', degree=1)
pow2 = Operator(function=_protected_power, name='power', degree=2)
log1 = Operator(function=_protected_log, name='log', degree=1)
neg1 = Operator(function=np.negative, name='neg', degree=1)
inv1 = Operator(function=_protected_inverse, name='inv', degree=1)
abs1 = Operator(function=np.abs, name='abs', degree=1)
maximum2 = Operator(function=np.maximum, name='maximum', degree=2)
minimum2 = Operator(function=np.minimum, name='minimum', degree=2)
sin1 = Operator(function=np.sin, name='sin', degree=1)
cos1 = Operator(function=np.cos, name='cos', degree=1)
tan1 = Operator(function=np.tan, name='tan', degree=1)
sinh1 = Operator(function=np.sinh, name='sinh', degree=1)
cosh1 = Operator(function=np.cosh, name='cosh', degree=1)
tanh1 = Operator(function=np.tanh, name='tanh', degree=1)
exp1 = Operator(function=_protected_exp, name='exp', degree=1)
expsq1 = Operator(function=_protected_expsq, name='expsq', degree=1)

sigmoid = Operator(function=_sigmoid, name='sigmoid', degree=1)
softplus = Operator(function=_softplus, name='softplus', degree=1)
softmax = Operator(function=_softmax, name='softmax', degree=2)


_operator_map = {
    'identity': identity,
    '+': add2, 
    'add': add2, 
    '-': sub2,
    'sub': sub2,
    '*': mul2,
    'mul': mul2,
    '/': div2,
    'div': div2,
    'sqrt': sqrt1,
    'pow': pow2,
    'power': pow2,
    'log': log1,
    'exp': exp1,
    'abs': abs1,
    'neg': neg1,
    'inv': inv1,
    'maximum': maximum2,
    'minimum': minimum2,
    'sin': sin1,
    'cos': cos1,
    'tan': tan1,
    'sinh': sinh1,
    'cosh': cosh1,
    'tanh': tanh1,
    'sigmoid': sigmoid, 
    'softmax': softmax,
    'expsq': expsq1,
    'softplus': softplus,
    'zscore': zscore
}


op_name_alias = {
    '+': 'add',
    '-': 'sub',
    '*': 'mul',
    '/': 'div',
    'pow': 'power'
}



