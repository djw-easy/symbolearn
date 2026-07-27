"""
Symbolic Classifier Training Script for Hyperspectral Image Classification.

This module provides a command-line interface for training SymbolicClassifier
models on various hyperspectral image datasets. It handles data loading,
preprocessing, evolutionary training, evaluation, and result persistence.

The script supports:
- Multiple hyperspectral datasets (Houston, Pavia University, Salinas, etc.)
- Configurable evolutionary parameters (population size, generations, operators)
- Spatial and spectral aggregation features
- Model checkpointing and best model tracking
- Comprehensive reporting with training logs

Usage
-----
    python train_sc.py --dataset Houston --niterations 100 --n_jobs 16

    # Run with custom operators
    python train_sc.py --dataset "Pavia University" --operators + - * / sin cos

    # Run with spatial features
    python train_sc.py --dataset Salinas --spatial_stats mean --valid_window_sizes 3

Command-Line Arguments
--------------------
Dataset:
    --dataset : Dataset name from DATASETS_INFO (default: Houston)
    --train_size : Training samples per class (default: 30)
    --per_class : Interpret train_size as per-class count (default: True)

Symbolic Engine:
    --maxsize : Maximum expression complexity (default: 25)
    --niterations : Number of generations (default: 100)
    --populations : Number of islands (default: 31)
    --population_size : Individuals per island (default: 27)
    --ncycles_per_iteration : Evolution cycles per generation (default: 380)
    --operators : List of mathematical operators (default: ['+', '-', '*', '/', 'sigmoid', 'tanh', 'softplus'])
    --use_constant : Include constants in expressions (default: True)
    --use_variable : Include variables in expressions (default: True)
    --constraints : Operator argument complexity limits
    --nested_constraints : Limits on operator nesting depth

Spatial-Spectral Features:
    --spatial_stats : Spatial aggregation function (default: 'mean')
    --valid_window_sizes : Window size for spatial aggregation (default: 3)
    --spectral_stats : Spectral aggregation functions (default: ['mean', 'min', 'max', 'slope'])

System:
    --n_jobs : Parallel jobs (default: 16)
    --verbose : Verbosity level (default: 0)
    --random_state : Random seed (default: 42)
    --batching : Use mini-batch training (default: False)
    --batch_size : Samples per batch (default: 512)
    --metric : Fitness metric (default: 'hinge_loss')
    --out_func : Output transformation (default: 'identity')

Outputs
-------
The script generates the following output files:
- best_hof_<timestamp>.csv : Pareto front at best test accuracy
- best_hof_<timestamp>.joblib : Serialized best model
- final_hof_<timestamp>.csv : Final Pareto front
- final_hof_<timestamp>.joblib : Serialized final model
- evolution_log_<timestamp>.csv : Detailed evolution log (if enabled)
- report_<timestamp>.txt : Comprehensive training report

Examples
--------
    # Train on Houston dataset with default settings
    python train_sc.py --dataset Houston

    # Train with more iterations and custom operators
    python train_sc.py --dataset Salinas --niterations 200 --operators + - * / sin

    # Use spatial aggregation features
    python train_sc.py --dataset "WHU_Hi_HongHu" --spatial_stats mean --valid_window_sizes 5

Notes
-----
The script uses a subprocess-based approach for running multiple datasets sequentially,
ensuring complete memory cleanup between datasets.

See Also
--------
SymbolicClassifier : The classifier being trained.
train.py : Generic symbolic regression trainer.
"""

import os
import io
import json
import sys
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Custom imports
from symbolearn.base import seconds_to_readable
from symbolearn.utils import stratified_train_test_split, spatially_disjoint_train_test_split
from symbolearn.symbolic_estimators import SymbolicClassifier

# --- Dataset Configuration ---
# Format: {Name: [File, Data_Key, GT_File, GT_Key, Dimensions, Num_Classes]}
DATASETS_INFO = {
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
    info = DATASETS_INFO[dataset_name]
    
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

    # Perform spectral scaling
    scaler = StandardScaler()
    image_valid = image[valid_mask]
    image_valid_scaled = scaler.fit_transform(image_valid)
    
    # Reconstruct 3D cube with NaNs in background to avoid calculation on invalid pixels
    image_scaled_3d = np.full_like(image, fill_value=np.nan, dtype=np.float32)
    image_scaled_3d[valid_mask] = image_valid_scaled
    
    return image_scaled_3d, image_gt, image_valid_scaled, image_gt[valid_mask]


def str2bool(v):
    """Utility to parse boolean arguments from CLI."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Defines command line interface parameters."""
    parser = argparse.ArgumentParser(description='Symbolic Evolutionary Classifier for HSI')
    
    # Dataset Settings
    group_ds = parser.add_argument_group('Dataset')
    group_ds.add_argument('--dataset', type=str, default='Pavia University', choices=DATASETS_INFO.keys())
    group_ds.add_argument('--train_size', type=float, default=100, 
                          help='Controls the number of training samples')
    group_ds.add_argument('--per_class', type=float, default=True, 
                          help='Determines how `train_size` is interpreted')
    group_ds.add_argument('--min_test_samples', type=int, default=30,
                          help='Minimum test samples per class; supplements from training region if needed (default: None)')
    
    # Evolutionary Settings
    group_evo = parser.add_argument_group('Symbolic Engine')
    group_evo.add_argument('--maxsize', type=int, default=25)
    group_evo.add_argument('--niterations', type=int, default=100)
    group_evo.add_argument('--populations', type=int, default=31)
    group_evo.add_argument('--population_size', type=int, default=27)
    group_evo.add_argument('--ncycles_per_iteration', type=int, default=380)
    group_evo.add_argument('--operators', type=str, nargs='+', default=[
        '+', '-', '*', '/', 'sigmoid', 'tanh', 'softplus'
    ])
    group_evo.add_argument('--use_constant', type=str2bool, default=True)
    group_evo.add_argument('--use_variable', type=str2bool, default=True)
    group_evo.add_argument('--constraints', type=dict, default={'/': (5, 5)})
    group_evo.add_argument('--nested_constraints', type=dict, default={
        'sigmoid': {'sigmoid': 0, 'tanh': 1, 'softplus': 1}, 
        'tanh': {'sigmoid': 1, 'tanh': 0, 'softplus': 1}, 
        'softplus': {'sigmoid': 1, 'tanh': 1, 'softplus': 0}
    })

    # Optimization Settings
    group_opt = parser.add_argument_group('Optimization')
    group_opt.add_argument('--penalty', type=str, default=None)
    group_opt.add_argument('--C', type=float, default=1.0)
    group_opt.add_argument('--should_optimize_constants', type=str2bool, default=False)
    group_opt.add_argument('--should_optimize_aggregations', type=str2bool, default=True)
    
    # Spatial/Spectral Settings
    group_spat = parser.add_argument_group('Spatial-Spectral Features')
    group_spat.add_argument('--spatial_stats', type=str, default='mean')
    # window_size=1 means no spatial filter process
    group_spat.add_argument('--valid_window_sizes', type=int, default=1)
    group_spat.add_argument('--spectral_stats', type=str, nargs='+', default=[
        'mean', 'min', 'max', 'slope'
    ])
    group_spat.add_argument('--valid_spectral_length', type=tuple, default=(5, 50))
    
    # Computational Settings
    group_sys = parser.add_argument_group('System')
    group_sys.add_argument('--n_jobs', type=int, default=16)
    group_sys.add_argument('--verbose', type=int, default=0)
    group_sys.add_argument('--ndigits', type=int, default=10)
    group_sys.add_argument('--random_state', type=int, default=42)
    group_sys.add_argument('--batching', type=str2bool, default=False)
    group_sys.add_argument('--batch_size', type=int, default=512)
    group_sys.add_argument('--metric', type=str, default='hinge_loss')
    group_sys.add_argument('--out_func', type=str, default='zscore')
    group_sys.add_argument('--enable_logging', type=str2bool, default=False)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# A tee-style stream that writes to both the terminal and an in-memory buffer.
# This ensures fit() output is visible on the console AND captured for the report.
# ---------------------------------------------------------------------------
class _TeeStream:
    """Mirrors every write to *both* a real stream and a StringIO buffer."""

    def __init__(self, real_stream, buffer: io.StringIO):
        self._real   = real_stream
        self._buffer = buffer

    def write(self, data):
        self._real.write(data)
        self._buffer.write(data)

    def flush(self):
        self._real.flush()
        self._buffer.flush()

    # Proxy any other attribute access (e.g. .encoding, .fileno) to the real stream
    def __getattr__(self, name):
        return getattr(self._real, name)


def verbose_reporter(estimator, run_details, callback_results):
    """Print a progress report for the current generation.

    Parameters
    ----------
    run_details : dict, optional
        A dictionary containing evolution statistics. When ``None``,
        only the column header is printed.
    """
    # Estimate remaining time
    gen = run_details['generation'][-1]
    generation_time = run_details['generation_time'][-1]
    remaining_time = seconds_to_readable(
        (estimator.niterations - gen) * generation_time
    )
    used_time = seconds_to_readable(run_details['total_time'][-1])
    if gen == 1:
        # Print table header
        print(
            '      |{:^37}|{:^37}|{:^33}|{:^22}'.format(
                'Population Average', 'Best Individual', 'Progress', "Accuracy"
            )
        )
        print('-' * 6 + '|' + '-' * 37 + '|' + '-' * 37 + '|' + '-' * 33 + '|' + '-' * 22)
        line_format = (
            ' {:>4} |{:>7} {:>12} {:>15} '
            '|{:>7} {:>12} {:>15} |{:>15}  {:>15} |{:>11}  {:>8} '
        )
        print(
            line_format.format(
                'Iter', 'Order', 'Complexity', 'Error',
                'Order', 'Complexity', 'Error',
                'Time Left', 'Time Used', 
                'Train acc', 'Test acc'
            )
        )

    line_format = (
        ' {:4d} |{:7.2f} {:12.2f} {:15g} '
        '|{:7d} {:12d} {:15g} |{:>15}  {:>15} |  {:9.6f}  {:8.6f}'
    )
    print(
        line_format.format(
            run_details['generation'][-1],
            run_details['average_order'][-1],
            run_details['average_size'][-1],
            run_details['average_fitness'][-1],
            run_details['best_order'][-1],
            run_details['best_complexity'][-1],
            run_details['best_fitness'][-1],
            remaining_time, used_time,
            callback_results['train_acc'],
            callback_results['test_acc']
        )
    )



def main():
    args = parse_args()
    selected_ds = args.dataset
    
    # 1. Data Preparation
    print(f">>> Loading Dataset: {selected_ds}")
    image_scaled, image_gt, X_valid, y_valid = load_dataset(selected_ds)
    
    if args.per_class:
        _, _, y_train, y_test = spatially_disjoint_train_test_split(
            image_scaled, image_gt, train_size=args.train_size,
            ignore_label=0, random_state=args.random_state,
            min_test_samples=args.min_test_samples,
            preserve_shape=True, per_class=True,
            block_size=4, buffer_size=2,
            allow_insufficient=True,
        )
    else:
        _, _, y_train, y_test = spatially_disjoint_train_test_split(
            image_scaled, image_gt, train_size=args.train_size*DATASETS_INFO[selected_ds][-1],
            ignore_label=0, random_state=args.random_state,
            min_test_samples=args.min_test_samples,
            preserve_shape=True, per_class=False,
            block_size=4, buffer_size=2,
            allow_insufficient=True,
        )
    
    unique_classes, counts = np.unique(y_valid, return_counts=True)
    class_dist = dict(zip(unique_classes, counts))

    # 2. Prepare output directory and paths ahead of time (used inside callback)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        f'outputs/models{args.train_size}/SC',
        selected_ds,
        f'run_{args.random_state}',
        timestamp
    )
    os.makedirs(output_dir, exist_ok=True)
    best_hof_path = os.path.join(output_dir, f'best_hof_{timestamp}.csv')
    best_model_path = os.path.join(output_dir, f'best_hof_{timestamp}.joblib')

    # 3. Track best state via closure to avoid global variables
    best_state = {'train_acc': -1.0, 'test_acc': -1.0, 'generation': -1}

    def eval_test_score(estimator: SymbolicClassifier, run_details: dict):
        gen       = run_details['generation'][-1]
        train_acc = estimator.score(image_scaled, y_train)
        test_acc  = estimator.score(image_scaled, y_test)

        # Overwrite best HOF whenever test accuracy improves
        if test_acc > best_state['test_acc']:
            best_state['train_acc']   = train_acc
            best_state['test_acc']   = test_acc
            best_state['generation'] = gen
            estimator.get_hof(include_dominated=True).to_csv(
                best_hof_path, index=False
            )
            estimator.save_model(best_model_path, compress=9)
        
        callback_results = {'train_acc': train_acc, 'test_acc': test_acc}
        verbose_reporter(estimator, run_details, callback_results)

    # 4. Model Initialization
    sc_classifier = SymbolicClassifier(
        maxsize=args.maxsize,
        ndigits=args.ndigits,
        niterations=args.niterations,
        populations=args.populations,
        population_size=args.population_size,
        use_constant=args.use_constant,
        use_variable=args.use_variable,
        constraints=args.constraints,
        nested_constraints=args.nested_constraints,
        spatial_stats=args.spatial_stats,
        valid_window_sizes=args.valid_window_sizes,
        valid_spectral_length=args.valid_spectral_length,
        penalty=args.penalty, C=args.C,
        ncycles_per_iteration=args.ncycles_per_iteration,
        should_optimize_constants=args.should_optimize_constants,
        should_optimize_aggregations=args.should_optimize_aggregations,
        n_jobs=args.n_jobs, verbose=args.verbose,
        random_state=args.random_state,
        metric=args.metric, out_func=args.out_func,
        batching=args.batching, batch_size=args.batch_size,
        operators=tuple(args.operators),
        spectral_stats=tuple(args.spectral_stats),
        enable_logging=args.enable_logging,
        callbacks=eval_test_score, callback_every=1
    )

    # 5. Training
    # Redirect stdout to a tee stream so that all output produced by fit()
    # (including verbose_reporter lines printed inside the callback) is both
    # displayed on the terminal and stored in `fit_log_buffer` for the report.
    fit_log_buffer = io.StringIO()
    _original_stdout = sys.stdout
    sys.stdout = _TeeStream(_original_stdout, fit_log_buffer)
    try:
        print(f">>> Starting Training on {selected_ds}...")
        start_time = time.time()
        sc_classifier.fit(image_scaled, y_train)
        duration = time.time() - start_time
    finally:
        # Always restore the real stdout, even if fit() raises an exception
        sys.stdout = _original_stdout

    # 6. Prediction & Evaluation (using the final trained model)
    print(">>> Generating Predictions...")
    y_pred_map = sc_classifier.predict(image_scaled)

    train_mask = ~np.isnan(y_train)
    test_mask  = ~np.isnan(y_test)

    y_train_true = y_train[train_mask].astype(np.intp)
    y_test_true  = y_test[test_mask].astype(np.intp)
    y_train_pred = y_pred_map[train_mask].astype(np.intp)
    y_test_pred  = y_pred_map[test_mask].astype(np.intp)

    train_acc = accuracy_score(y_train_true, y_train_pred)
    test_acc  = accuracy_score(y_test_true,  y_test_pred)

    # 7. Save final HOF and evolution log
    final_hof_path = os.path.join(output_dir, f'final_hof_{timestamp}.csv')
    sc_classifier.get_hof(include_dominated=True).to_csv(
        final_hof_path, index=False
    )
    final_model_path = os.path.join(output_dir, f'final_hof_{timestamp}.joblib')
    sc_classifier.save_model(final_model_path, compress=9)
    if args.enable_logging:
        sc_classifier.save_evolution_log(
            os.path.join(output_dir, f'evolution_log_{timestamp}.csv')
        )

    # 8. Generate comprehensive report
    report_path = os.path.join(output_dir, f'report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\nSYMBOLIC CLASSIFICATION REPORT\n{'='*60}\n")
        f.write(f"Dataset:            {selected_ds}\n")
        f.write(f"Duration:           {seconds_to_readable(duration)}\n\n")
        f.write(f"Best Generation:    {best_state['generation']}\n")
        f.write(f"Best Train Acc:     {best_state['train_acc']:.6f}\n")
        f.write(f"Best Test Acc:      {best_state['test_acc']:.6f}\n")
        f.write(f"Best HOF Path:      {best_hof_path}\n\n")
        f.write(f"Final Train Acc:    {train_acc:.6f}\n")
        f.write(f"Final Test Acc:     {test_acc:.6f}\n")
        f.write(f"Final HOF Path:     {final_hof_path}\n\n")

        f.write(f"{'-'*20} DATA INFO {'-'*20}\n")
        f.write(f"Samples (Train/Test): {len(y_train_true)} / {len(y_test_true)}\n")
        f.write(f"Features:      {image_scaled.shape[-1]}\n\n")

        f.write(f"{'-'*20} CLASSIFICATION DETAILS {'-'*20}\n")
        f.write("TRAIN SET REPORT:\n")
        f.write(classification_report(y_train_true, y_train_pred, zero_division=0, digits=6))
        f.write("\nTEST SET REPORT:\n")
        f.write(classification_report(y_test_true,  y_test_pred,  zero_division=0, digits=6))

        # Write all stdout lines produced during fit() into the report
        f.write(f"\n{'-'*20} TRAINING LOG {'-'*20}\n")
        f.write(fit_log_buffer.getvalue())

        f.write(f"\n{'-'*20} CONFIGURATION {'-'*20}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    # 9. Write a structured run summary for easy aggregation across seeds
    run_summary = {
        'dataset': selected_ds,
        'seed': args.random_state,
        'best_train_acc': best_state['train_acc'],
        'best_test_acc': best_state['test_acc'],
        'best_generation': best_state['generation'],
        'final_train_acc': train_acc,
        'final_test_acc': test_acc,
        'duration_seconds': duration,
    }
    with open(os.path.join(output_dir, 'run_summary.json'), 'w') as f:
        json.dump(run_summary, f, indent=2)

    # 10. Final console summary
    print(f"    - Best Generation:   {best_state['generation']}")
    print(f"    - Best Train Acc:    {best_state['train_acc']:.6f}")
    print(f"    - Best Test Acc:     {best_state['test_acc']:.6f}")
    print(f"    - Final Train Acc:   {train_acc:.6f}")
    print(f"    - Final Test Acc:    {test_acc:.6f}")
    print(f"    - Results saved to:  {output_dir}")


if __name__ == '__main__':
    main()