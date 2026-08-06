import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    roc_auc_score, 
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score
)
from scipy.stats import wasserstein_distance
from itertools import combinations


from sklearn.preprocessing import MinMaxScaler


def weighted_pearson(y, y_pred, w=None):
    """
    Calculate the weighted Pearson correlation coefficient.
    If 'w' is None, it defaults to the unweighted Pearson correlation.
    """
    if w is None:
        w = np.ones_like(y) # If no weights, use equal weights

    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)

        # Handle cases where all weights sum to zero or other problematic scenarios
        sum_w = np.sum(w)
        if sum_w == 0:
            return 0.0

        numerator = np.sum(w * y_pred_demean * y_demean)
        
        denominator_term1 = np.sum(w * y_pred_demean ** 2)
        denominator_term2 = np.sum(w * y_demean ** 2)
        
        # Avoid division by zero in the square root
        if denominator_term1 == 0 or denominator_term2 == 0:
            return 0.0

        corr = (numerator / sum_w) / np.sqrt((denominator_term1 * denominator_term2) / (sum_w ** 2))

    if np.isfinite(corr):
        return np.abs(corr)
    return 0.0


def weighted_spearman(y, y_pred, w=None):
    """
    Calculate the weighted Spearman correlation coefficient.
    If 'w' is None, it defaults to the unweighted Spearman correlation.
    """
    # Rank data
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)

    # Call weighted_pearson with the ranked data and the provided weights
    return weighted_pearson(y_pred_ranked, y_ranked, w)



# ==================== Pairwise Metric Functions (Binary Only) ====================


def separation_from_auc(x0: np.ndarray, x1: np.ndarray) -> float:
    """
    Computes a directional separation score using ROC AUC.

    This score is positive if class 1 values tend to be greater than class 0,
    and is clipped at a small positive value if the direction is reversed.

    Args:
        x0 (np.ndarray): An array of prediction values for class 0.
        x1 (np.ndarray): An array of prediction values for class 1.

    Returns:
        float: A directional separation score, typically between 0.001 and 1.0.
               Returns 0.0 for edge cases with no variance.
    """
    if len(x0) == 0 or len(x1) == 0:
        return 1.0 # Max score if one class is empty (perfectly separable)

    y_true = np.concatenate([np.zeros_like(x0), np.ones_like(x1)])
    y_score = np.concatenate([x0, x1])

    try:
        auc = roc_auc_score(y_true, y_score)
        # Directional score: positive if AUC > 0.5
        s_directional = 2.0 * (auc - 0.5)
        # Penalize wrong direction heavily by clipping score near zero
        return s_directional
    except ValueError:
        # Fallback for rare edge cases where AUC is not defined
        return 0.0




def _bhattacharyya_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Computes Bhattacharyya distance between binary class distributions (0 and 1).
    Assumes Gaussian distributions. Multiplied by separation_from_auc score.

    Args:
        prediction (np.ndarray): 1D array of continuous predictions.
        target (np.ndarray): 1D array of binary class labels (0 and 1).

    Returns:
        float: Bhattacharyya distance multiplied by AUC separation score. 
               Returns -np.inf on failure or if not exactly 2 classes.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target)

    unique_classes = np.unique(target)
    # Check if we have exactly 2 classes and they are 0 and 1
    if len(unique_classes) != 2 or not (set(unique_classes) == {0, 1}):
        return -np.inf

    # Extract predictions for each class
    preds_0 = prediction[target == 0]
    preds_1 = prediction[target == 1]

    if preds_0.size == 0 or preds_1.size == 0:
        return -np.inf

    # Calculate means and variances
    mu_0 = np.mean(preds_0)
    mu_1 = np.mean(preds_1)

    var_0 = np.var(preds_0, ddof=1) if preds_0.size > 1 else 0.0
    var_1 = np.var(preds_1, ddof=1) if preds_1.size > 1 else 0.0

    if var_0 <= 0.0 or var_1 <= 0.0:
        return -np.inf

    # Compute Bhattacharyya distance
    mean_diff_term = (mu_0 - mu_1) ** 2 / (var_0 + var_1)
    variance_ratio = var_0 / var_1 + var_1 / var_0
    variance_diff_term = 0.25 * np.log(0.25 * (variance_ratio + 2.0))

    distance = 0.25 * mean_diff_term + variance_diff_term

    if distance <= 0:
        return -np.inf

    # Calculate AUC separation score
    auc_separation = separation_from_auc(preds_0, preds_1)
    
    return distance * auc_separation


def _hellinger_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Computes Hellinger distance between binary class distributions (0 and 1).
    Assumes Gaussian distributions. Multiplied by separation_from_auc score.

    Args:
        prediction (np.ndarray): 1D array of continuous predictions.
        target (np.ndarray): 1D array of binary class labels (0 and 1).

    Returns:
        float: Hellinger distance multiplied by AUC separation score. 
               Returns -np.inf on failure or if not exactly 2 classes.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target)

    unique_classes = np.unique(target)
    # Check if we have exactly 2 classes and they are 0 and 1
    if len(unique_classes) != 2 or not (set(unique_classes) == {0, 1}):
        return -np.inf

    # Extract predictions for each class
    preds_0 = prediction[target == 0]
    preds_1 = prediction[target == 1]

    if preds_0.size == 0 or preds_1.size == 0:
        return -np.inf

    # Calculate means and variances
    mu_0 = np.mean(preds_0)
    mu_1 = np.mean(preds_1)

    var_0 = np.var(preds_0, ddof=1) if preds_0.size > 1 else 0.0
    var_1 = np.var(preds_1, ddof=1) if preds_1.size > 1 else 0.0

    std_0 = np.sqrt(var_0)
    std_1 = np.sqrt(var_1)

    if (var_0 + var_1) <= 0.0:
        return -np.inf

    # Compute Hellinger distance
    term1 = np.sqrt(2 * std_0 * std_1 / (var_0 + var_1))
    term2 = np.exp(-0.25 * (mu_0 - mu_1) ** 2 / (var_0 + var_1))
    bc = term1 * term2

    val_to_sqrt = max(0.0, 1.0 - bc)
    distance = np.sqrt(val_to_sqrt)

    # Calculate AUC separation score
    auc_separation = separation_from_auc(preds_0, preds_1)
    
    return distance * auc_separation


def _js_divergence_loss(prediction: np.ndarray, target: np.ndarray, n_bins: int = 50) -> float:
    """
    Computes Jensen-Shannon divergence between binary class distributions (0 and 1).
    Uses histogram-based estimation (non-parametric). Multiplied by separation_from_auc score.

    Args:
        prediction (np.ndarray): 1D array of continuous predictions.
        target (np.ndarray): 1D array of binary class labels (0 and 1).
        n_bins (int): Number of histogram bins.

    Returns:
        float: JS divergence ∈ [0,1] multiplied by AUC separation score. 
               Returns -np.inf on failure or if not exactly 2 classes.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target)

    unique_classes = np.unique(target)
    # Check if we have exactly 2 classes and they are 0 and 1
    if len(unique_classes) != 2 or not (set(unique_classes) == {0, 1}):
        return -np.inf

    min_val, max_val = np.min(prediction), np.max(prediction)
    if min_val == max_val:
        # Calculate AUC separation score even for constant predictions
        preds_0 = prediction[target == 0]
        preds_1 = prediction[target == 1]
        auc_separation = separation_from_auc(preds_0, preds_1)
        return 1.0 * auc_separation

    # Extract predictions for each class
    preds_0 = prediction[target == 0]
    preds_1 = prediction[target == 1]

    if preds_0.size == 0 or preds_1.size == 0:
        return -np.inf

    # Create histograms
    hist_0, _ = np.histogram(preds_0, bins=n_bins, range=(min_val, max_val))
    hist_1, _ = np.histogram(preds_1, bins=n_bins, range=(min_val, max_val))

    # Normalize to probabilities
    p_0 = hist_0 / preds_0.size
    p_1 = hist_1 / preds_1.size

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_0 += eps
    p_1 += eps
    p_0 /= np.sum(p_0)
    p_1 /= np.sum(p_1)

    # Compute JS divergence
    m = 0.5 * (p_0 + p_1)

    kl_0_m = np.sum(np.where(p_0 > 0, p_0 * np.log2(p_0 / m), 0))
    kl_1_m = np.sum(np.where(p_1 > 0, p_1 * np.log2(p_1 / m), 0))

    js_div = 0.5 * kl_0_m + 0.5 * kl_1_m
    js_div = np.clip(js_div, 0.0, 1.0)

    # Calculate AUC separation score
    auc_separation = separation_from_auc(preds_0, preds_1)
    
    return js_div * auc_separation


def _earth_movers_distance_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the Earth Mover's Distance (Wasserstein Distance) between binary class 
    distributions (0 and 1). Multiplied by separation_from_auc score.

    Args:
        prediction (np.ndarray): 1D array of continuous model predictions.
        target (np.ndarray): 1D array of binary class labels (0 and 1).

    Returns:
        float: Wasserstein distance multiplied by AUC separation score.
               Returns -np.inf if not exactly 2 classes or no valid computation.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target)

    unique_classes = np.unique(target)
    # Check if we have exactly 2 classes and they are 0 and 1
    if len(unique_classes) != 2 or not (set(unique_classes) == {0, 1}):
        return -np.inf

    # Extract predictions for each class
    preds_0 = prediction[target == 0]
    preds_1 = prediction[target == 1]

    # Skip if either class has no samples
    if preds_0.size == 0 or preds_1.size == 0:
        return -np.inf

    # Compute Wasserstein distance between the two 1D empirical distributions
    emd_distance = wasserstein_distance(u_values=preds_0, v_values=preds_1)
    
    # Calculate AUC separation score
    auc_separation = separation_from_auc(preds_0, preds_1)
    
    return emd_distance * auc_separation


def fisher_linear_discriminant(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Computes Fisher's Linear Discriminant criterion for binary classification (0 and 1).
    (between-class scatter / within-class scatter). Multiplied by separation_from_auc score.
    Higher values indicate better class separation.
    
    Args:
        prediction (np.ndarray): 1D array of continuous predictions.
        target (np.ndarray): 1D array of binary class labels (0 and 1).
    Returns:
        float: Fisher's Linear Discriminant score multiplied by AUC separation score. 
               Returns -np.inf on failure or if not exactly 2 classes.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target)
    unique_classes = np.unique(target)
    
    # Check if we have exactly 2 classes and they are 0 and 1
    if len(unique_classes) != 2 or not (set(unique_classes) == {0, 1}):
        return -np.inf
    
    # Extract predictions for each class
    preds_0 = prediction[target == 0]
    preds_1 = prediction[target == 1]
    
    if preds_0.size == 0 or preds_1.size == 0:
        return -np.inf
    
    # Overall mean
    overall_mean = np.mean(prediction)
    n_total = len(prediction)
    
    # Class means
    mean_0 = np.mean(preds_0)
    mean_1 = np.mean(preds_1)
    n_0 = len(preds_0)
    n_1 = len(preds_1)
    
    # Between-class scatter: sum of n_k * (mu_k - mu)^2
    between_scatter = n_0 * (mean_0 - overall_mean) ** 2 + n_1 * (mean_1 - overall_mean) ** 2
    
    # Within-class scatter: sum of (x_i - mu_k)^2 for each class
    within_scatter = np.sum((preds_0 - mean_0) ** 2) + np.sum((preds_1 - mean_1) ** 2)
    
    # Avoid division by zero
    if within_scatter <= 0.0:
        fisher_score = np.inf if between_scatter > 0.0 else -np.inf
    else:
        # Fisher's criterion: SB / SW
        fisher_score = between_scatter / within_scatter
    
    if fisher_score == -np.inf or fisher_score == np.inf:
        return fisher_score
    
    # Calculate AUC separation score
    auc_separation = separation_from_auc(preds_0, preds_1)
    
    return fisher_score * auc_separation


def f_statistic_anova(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Computes F-statistic from one-way ANOVA for binary classification (0 and 1).
    F = MSB / MSW where MSB is mean square between groups and MSW is mean square within groups.
    Multiplied by separation_from_auc score.
    
    Args:
        prediction (np.ndarray): 1D array of continuous predictions.
        target (np.ndarray): 1D array of binary class labels (0 and 1).
    Returns:
        float: F-statistic value multiplied by AUC separation score. 
               Returns -np.inf on failure or if not exactly 2 classes.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target)
    unique_classes = np.unique(target)
    
    # Check if we have exactly 2 classes and they are 0 and 1
    if len(unique_classes) != 2 or not (set(unique_classes) == {0, 1}):
        return -np.inf
    
    # Extract predictions for each class
    preds_0 = prediction[target == 0]
    preds_1 = prediction[target == 1]
    
    if preds_0.size == 0 or preds_1.size == 0:
        return -np.inf
    
    # Overall mean
    overall_mean = np.mean(prediction)
    n_total = len(prediction)
    k = 2  # number of groups (binary classification)
    
    # Class means
    mean_0 = np.mean(preds_0)
    mean_1 = np.mean(preds_1)
    n_0 = len(preds_0)
    n_1 = len(preds_1)
    
    # Sum of squares between groups (SSB)
    ssb = n_0 * (mean_0 - overall_mean) ** 2 + n_1 * (mean_1 - overall_mean) ** 2
    
    # Sum of squares within groups (SSW)
    ssw = np.sum((preds_0 - mean_0) ** 2) + np.sum((preds_1 - mean_1) ** 2)
    
    # Degrees of freedom
    df_between = k - 1  # degrees of freedom between groups = 1 for binary
    df_within = n_total - k  # degrees of freedom within groups
    
    # Avoid division by zero
    if df_between <= 0 or df_within <= 0 or ssw <= 0.0:
        return -np.inf
    
    # Mean squares
    msb = ssb / df_between  # Mean square between
    msw = ssw / df_within   # Mean square within
    
    # F-statistic
    f_stat = msb / msw
    
    # Calculate AUC separation score
    auc_separation = separation_from_auc(preds_0, preds_1)
    
    return f_stat * auc_separation



def _compactness_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Compute compactness score based on within-class standard deviation
    relative to pooled standard deviation for binary classification (0 and 1).
    Multiplied by separation_from_auc score.

    Args:
        prediction (np.ndarray): 1D array of continuous predictions.
        target (np.ndarray): 1D array of binary class labels (0 and 1).

    Returns:
        float: Compactness score C ∈ (0,1] multiplied by AUC separation score. 
               Returns -np.inf if not exactly 2 classes.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target)
    
    unique_classes = np.unique(target)
    # Check if we have exactly 2 classes and they are 0 and 1
    if len(unique_classes) != 2 or not (set(unique_classes) == {0, 1}):
        return -np.inf
    
    # Extract predictions for each class
    preds_0 = prediction[target == 0]
    preds_1 = prediction[target == 1]

    if preds_0.size == 0 or preds_1.size == 0:
        return -np.inf
    
    # Calculate within-class standard deviations
    std0 = np.std(preds_0, ddof=1) if len(preds_0) > 1 else 0.0
    std1 = np.std(preds_1, ddof=1) if len(preds_1) > 1 else 0.0
    avg_std = 0.5 * (std0 + std1)

    # Calculate pooled standard deviation
    pooled = np.concatenate([preds_0, preds_1])
    if len(pooled) <= 1:
        # Calculate AUC separation score even for edge cases
        auc_separation = separation_from_auc(preds_0, preds_1)
        return 1.0 * auc_separation

    pooled_std = np.std(pooled, ddof=1)
    if pooled_std == 0:
        # Calculate AUC separation score for constant predictions
        auc_separation = separation_from_auc(preds_0, preds_1)
        return 1.0 * auc_separation

    # Compute compactness ratio
    # ratio = avg_std / pooled_std
    # compactness_score = 1.0 / (1.0 + ratio)
    compactness_score = pooled_std / avg_std
    
    # Calculate AUC separation score
    auc_separation = separation_from_auc(preds_0, preds_1)
    
    return compactness_score * auc_separation



def label_conditioned_eta(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return binary label-conditioned normalized between-class variance.

    The two groups are defined by the supplied labels rather than by searching
    over response thresholds.  This is the ``eta`` component used by
    :func:`directional_otsu_separability`, exposed separately so experiments
    can distinguish the supervised projection objective from deployment-time
    Otsu threshold estimation.
    """
    prediction = np.asarray(prediction, dtype=np.float64).ravel()
    target = np.asarray(target).ravel()

    if prediction.shape[0] != target.shape[0]:
        return -np.inf
    if len(np.unique(target)) != 2 or set(np.unique(target)) != {0, 1}:
        return -np.inf
    if not np.all(np.isfinite(prediction)):
        return -np.inf

    z_0 = prediction[target == 0]
    z_1 = prediction[target == 1]
    if z_0.size < 2 or z_1.size < 2:
        return -np.inf

    total_variance = np.var(prediction)
    if total_variance <= 0:
        return -np.inf

    n = prediction.size
    w_0 = z_0.size / n
    w_1 = z_1.size / n
    between_variance = w_0 * w_1 * (np.mean(z_0) - np.mean(z_1)) ** 2
    return float(np.clip(between_variance / total_variance, 0.0, 1.0))


def directional_otsu_separability(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Calculates the Directional Otsu Separability Fitness for spectral index evaluation.
    
    This fitness function is designed for genetic programming-based spectral index discovery,
    specifically tailored for automated land cover extraction using threshold segmentation.
    
    The fitness combines two key components:
    1. **Directional Control Factor (S_dir)**: Based on ROC AUC, ensures target class has 
       systematically higher values than background. Computed as:
       S_dir = 2 * (AUC - 0.5), range: [-1, 1]
       
    2. **Normalized Between-Class Variance (eta_max)**: Inspired by Otsu's method, this measures
       the separability of the two classes. Computed as:
       eta_max = sigma_B^2 / sigma_T^2, range: [0, 1]
       where sigma_B^2 is the between-class variance and sigma_T^2 is the total variance.
    
    Final fitness: F_Otsu-dir = S_dir * eta_max, range: [-1, 1]
    
    A high positive score indicates:
    - Clear separation between classes (high between-class variance)
    - Target class values systematically higher than background
    - Ideal for automated Otsu-based threshold extraction
    
    Args:
        prediction (np.ndarray): 1D array of spectral index values (continuous features)
        target (np.ndarray): 1D array of binary labels (0: background, 1: target)
        
    Returns:
        float: Directional Otsu separability fitness score in range [-1, 1].
               Higher positive values indicate better fitness for extraction.
               Returns -inf for invalid inputs.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target)
    
    # --- Input Validation ---
    unique_classes = np.unique(target)
    if len(unique_classes) != 2 or not set(unique_classes) == {0, 1}:
        # Target must be binary {0, 1}
        return -np.inf
    
    # Separate predictions by class
    Z_0 = prediction[target == 0]  # Background samples
    Z_1 = prediction[target == 1]  # Target samples
    
    if Z_0.size < 2 or Z_1.size < 2:
        # Need sufficient samples per class for stable metrics
        return -np.inf
    
    # Check for invalid values
    if not np.all(np.isfinite(prediction)):
        return -np.inf
    
    # --- Component 1: Directional Control Factor (S_dir) ---
    S_dir = separation_from_auc(Z_0, Z_1)

    # --- Component 2: Normalized Between-Class Variance (eta_max) ---
    # Inspired by Otsu's method: maximize between-class variance
    
    # Calculate class weights
    N = len(prediction)
    w_0 = len(Z_0) / N  # Weight of background class
    w_1 = len(Z_1) / N  # Weight of target class
    
    # Calculate class means
    mean_0 = np.mean(Z_0)  # Mean of background class
    mean_1 = np.mean(Z_1)  # Mean of target class
    
    # Calculate between-class variance (sigma_B^2)
    # sigma_B^2 = w_0 * w_1 * (mean_0 - mean_1)^2
    sigma_B_squared = w_0 * w_1 * (mean_0 - mean_1) ** 2
    
    # Calculate total variance (sigma_T^2)
    sigma_T_squared = np.var(prediction)
    
    # Handle edge case: no variance
    if sigma_T_squared == 0:
        return -np.inf
    
    # Normalized between-class variance (eta_max)
    # eta_max = sigma_B^2 / sigma_T^2, range: [0, 1]
    eta_max = sigma_B_squared / sigma_T_squared
    
    # --- Final Fitness Score ---
    # F_Otsu-dir = S_dir * eta_max
    # Combines directional control with class separability
    fitness = S_dir * eta_max
    
    return fitness



scaler = MinMaxScaler()
# Multi dimensional
def davies_bouldin_loss(y_true, y_pred, sample_weight):
    """计算Davies-Bouldin损失函数。"""
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return davies_bouldin_score(y_pred, y_true.ravel())

def calinski_harabasz_loss(y_true, y_pred, sample_weight):
    """计算Calinski and Harabasz损失函数。"""
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return calinski_harabasz_score(y_pred, y_true.ravel())

def silhouette_loss(y_true, y_pred, sample_weight):
    """计算轮廓系数损失函数。"""
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return silhouette_score(y_pred, y_true.ravel())

# Single dimension
def bhattacharyya_loss(y_true, y_pred, sample_weight) -> float:
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if np.all(y_pred == y_pred[0]):
        return -np.inf
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return _bhattacharyya_loss(y_pred, y_true.ravel())

def hellinger_loss(y_true, y_pred, sample_weight) -> float:
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if np.all(y_pred == y_pred[0]):
        return -np.inf
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return _hellinger_loss(y_pred, y_true.ravel())

def js_divergence_loss(y_true, y_pred, sample_weight) -> float:
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if np.all(y_pred == y_pred[0]):
        return -np.inf
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return _js_divergence_loss(y_pred, y_true.ravel())

def wasserstein_loss(y_true, y_pred, sample_weight) -> float:
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if np.all(y_pred == y_pred[0]):
        return -np.inf
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return _earth_movers_distance_loss(y_pred, y_true.ravel())

def compactness_loss(y_true, y_pred, sample_weight) -> float:
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if np.all(y_pred == y_pred[0]):
        return -np.inf
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return _compactness_loss(y_pred, y_true.ravel())

def separability_loss(y_true, y_pred, sample_weight) -> float:
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if np.all(y_pred == y_pred[0]):
        return -np.inf
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return directional_otsu_separability(y_pred, y_true.ravel())


def eta_separability_loss(y_true, y_pred, sample_weight) -> float:
    """Fitness wrapper for label-conditioned ``eta`` without AUC direction."""
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if y_pred.ndim == 0 or np.all(y_pred == y_pred.flat[0]):
        return -np.inf
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return label_conditioned_eta(y_pred, y_true.ravel())

def fisher_loss(y_true, y_pred, sample_weight) -> float:
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if np.all(y_pred == y_pred[0]):
        return -np.inf
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return fisher_linear_discriminant(y_pred, y_true.ravel())

def f_statistic_loss(y_true, y_pred, sample_weight) -> float:
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if np.all(y_pred == y_pred[0]):
        return -np.inf
    y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    return f_statistic_anova(y_pred, y_true.ravel())
