from symbolearn.utils import (
    spatially_disjoint_train_test_split,
    standardize_spatial_image_from_training,
)
from symbolearn.symbolic_estimators import SymbolicClassifier


import numpy as np
from scipy.io import loadmat


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
    """Load raw hyperspectral data while preserving a NaN background."""
    info = datasets_info[dataset_name]
    
    # Load raw image and ground truth
    img_dict = loadmat(f'./data/hyperspectral/{dataset_name}/{info[0]}.mat')
    gt_dict = loadmat(f'./data/hyperspectral/{dataset_name}/{info[2]}.mat')
    
    image = img_dict[info[1]]
    image_gt = gt_dict[info[3]]
    
    # Squeeze 3D ground truth if necessary
    if image_gt.ndim == 3:
        image_gt = image_gt.squeeze()

    # Define valid pixels (GT > 0)
    valid_mask = image_gt > 0

    # Reconstruct the raw 3D cube with NaNs in the background. Standardization
    # is deliberately deferred until after the train/test masks are known.
    image_raw_3d = np.full_like(image, fill_value=np.nan, dtype=np.float32)
    image_raw_3d[valid_mask] = image[valid_mask]

    return image_raw_3d, image_gt, image[valid_mask], image_gt[valid_mask]



if __name__ == '__main__':
    print(f"{'-'*34} Testing ExpressionSet for Symbolic Classifier {'-'*35}")

    dataset_name = 'Salinas'
    image_raw, image_gt, X, y = load_dataset(dataset_name)
    _, _, y_train, y_test = spatially_disjoint_train_test_split(
        image_raw, image_gt, train_size=100, preserve_shape=True,
        per_class=True, block_size=4, buffer_size=2,
        ignore_label=0, random_state=42,
        allow_insufficient=True,
        adjacency_penalty=0.05,
        min_test_samples=20
    )
    image_scaled_train, _ = standardize_spatial_image_from_training(
        image_raw, ~np.isnan(y_train), image_gt>0
    )
    image_scaled_test, _ = standardize_spatial_image_from_training(
        image_raw, ~np.isnan(y_test), image_gt>0
    )
    
    sc_classifier = SymbolicClassifier(
        maxsize=15,
        niterations=10,
        populations=31,
        population_size=27,
        use_constant=True,
        use_variable=True,
        penalty=None, C=1.0,
        enable_logging=False,
        ncycles_per_iteration=38,
        operators=('+', '-', '*'),
        batching=False, batch_size=512,
        should_optimize_constants=False,
        should_optimize_aggregations=True,
        n_jobs=8, verbose=1, random_state=42,
        metric='hinge_loss', out_func='zscore',
        spatial_stats='mean', valid_window_sizes=1,
        valid_spectral_length=(5, 30), spectral_stats=('mean',),
        constraints={'/': (5, 5)}, nested_constraints={
            'sigmoid': {'sigmoid': 0, 'tanh': 1, 'softplus': 1}, 
            'tanh': {'sigmoid': 1, 'tanh': 0, 'softplus': 1}, 
            'softplus': {'sigmoid': 1, 'tanh': 1, 'softplus': 0}
        }
    )
    sc_classifier.fit(image_scaled_train, y_train)

    print("TrainSet Accuracy: ", sc_classifier.score(image_scaled_train, y_train))
    print("TestSet Accuracy: ", sc_classifier.score(image_scaled_test, y_test))
    print("\nPareto Front:")
    df = sc_classifier.get_hof()
    print(df)


