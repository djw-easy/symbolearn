from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.random import check_random_state
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import numpy as np
import os


from symbolearn.fitness import Fitness
from symbolearn.utils import stratified_train_test_split, extract_and_aggregate_spatial
from symbolearn.symbolic_estimators import SymbolicRegressor, SymbolicClassifier, SymbolicTransformer


datasets_info = {
    'Houston': ['Houston', 'houston', 'Houston_gt', 'houston_gt', (349, 1905, 144), 15], 
    'Pavia University': ['PaviaU', 'paviaU', 'PaviaU_gt', 'paviaU_gt', (610, 340, 103), 9], 
    'Salinas': ['Salinas_corrected', 'salinas_corrected', 'Salinas_gt', 'salinas_gt', (514, 217, 204), 16], 
    'KSC': ['KSC', 'KSC', 'KSC_gt', 'KSC_gt', (512, 614, 176), 13],
    'Indian Pines': ['Indian_pines_corrected', 'indian_pines_corrected', 'Indian_pines_gt', 'indian_pines_gt', (145, 145, 200), 16], 
    'Salinas-A': ['SalinasA_corrected', 'salinasA_corrected', 'SalinasA_gt', 'salinasA_gt', (86, 83, 204), 6], 
    'WHU_Hi_LongKou': ['WHU_Hi_LongKou', 'WHU_Hi_LongKou', 'WHU_Hi_LongKou_gt', 'WHU_Hi_LongKou_gt', (550, 400, 270), 9], 
    'WHU_Hi_HanChuan': ['WHU_Hi_HanChuan', 'WHU_Hi_HanChuan', 'WHU_Hi_HanChuan_gt', 'WHU_Hi_HanChuan_gt', (1217, 303, 274), 16],
    'WHU_Hi_HongHu': ['WHU_Hi_HongHu', 'WHU_Hi_HongHu', 'WHU_Hi_HongHu_gt', 'WHU_Hi_HongHu_gt', (940, 475, 270), 22],
    'Trento': ['Italy_hsi', 'data', 'allgrd', 'mask_test', (166, 600, 63), 6],
    'Pavia Centre': ['Pavia', 'pavia', 'Pavia_gt', 'pavia_gt', (1096, 715, 102), 9],
    'Botswana': ['Botswana', 'Botswana', 'Botswana_gt', 'Botswana_gt', (1476, 256, 145), 14]
}


def load_dataset(dataset_name: str):
    """Loads hyperspectral data and applies standard scaling to valid pixels."""
    info = datasets_info[dataset_name]
    
    # Load raw image and ground truth
    img_dict = loadmat(f'./example_data/hyperspectral/{dataset_name}/{info[0]}.mat')
    gt_dict = loadmat(f'./example_data/hyperspectral/{dataset_name}/{info[2]}.mat')
    
    image = img_dict[info[1]]
    image_gt = gt_dict[info[3]]
    
    # Squeeze 3D ground truth if necessary
    if image_gt.ndim == 3:
        image_gt = image_gt.squeeze()

    # Define valid pixels (GT > 0)
    valid_mask = image_gt > 0

    # Perform spectral scaling
    scaler = StandardScaler()
    image_valid = image[valid_mask]
    image_valid_scaled = scaler.fit_transform(image_valid)
    
    # Reconstruct 3D cube with NaNs in background to avoid calculation on invalid pixels
    image_scaled_3d = np.full_like(image, fill_value=np.nan, dtype=np.float32)
    image_scaled_3d[valid_mask] = image_valid_scaled
    
    return image_scaled_3d, image_gt, image_valid_scaled, image_gt[valid_mask]



if __name__ == '__main__':
    print(f"{'-'*34} Testing ExpressionSet for Symbolic Classifier {'-'*35}")

    dataset_name = 'Pavia University'
    image_scaled, image_gt, X, y = load_dataset(dataset_name)
    _, _, y_train, y_test = stratified_train_test_split(
        image_scaled, image_gt, train_size=100, preserve_shape=True,
        ignore_label=0, random_state=42, shuffle=True, 
        per_class=True, balanced=True,
        allow_insufficient=True
    )
    
    sc_classifier = SymbolicClassifier(
        maxsize=15,
        niterations=10,
        populations=31,
        population_size=27,
        use_constant=True,
        use_variable=True,
        penalty=None, C=1.0,
        enable_logging=True,
        ncycles_per_iteration=38,
        valid_spectral_length=(5, 10),
        batching=False, batch_size=512,
        should_optimize_constants=False,
        should_optimize_aggregations=True,
        n_jobs=8, verbose=1, random_state=42,
        # metric='accuracy', out_func='softmax',
        # metric='focal_loss', out_func='softmax',
        metric='hinge_loss', out_func='identity',
        spatial_stats='mean', valid_window_sizes=3,
        spectral_stats=('mean', 'min', 'max', 'slope'),
        operators=('+', '-', '*', '/', 'sigmoid', 'tanh', 'softplus'),
        constraints={'/': (5, 5)}, nested_constraints={
            'sigmoid': {'sigmoid': 0, 'tanh': 1, 'softplus': 1}, 
            'tanh': {'sigmoid': 1, 'tanh': 0, 'softplus': 1}, 
            'softplus': {'sigmoid': 1, 'tanh': 1, 'softplus': 0}
        }
    )
    sc_classifier.fit(image_scaled, y_train)

    print("TrainSet Accuracy: ", sc_classifier.score(image_scaled, y_train))
    print("TestSet Accuracy: ", sc_classifier.score(image_scaled, y_test))
    print("\nPareto Front:")
    df = sc_classifier.get_hof()
    print(df)


