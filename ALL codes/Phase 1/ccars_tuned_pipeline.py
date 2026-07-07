#!/usr/bin/env python
"""
ccars_tuned_pipeline.py

CCARS Pipeline with Tuned Hyperparameters and Checkerboard Spatial Split.

Uses Nicola's original CARS methodology with:
- Tuned classifier hyperparameters (SVM-RBF: C=100, gamma=0.01; RF: n_estimators=500)
- Checkerboard spatial split for fair evaluation
- Permutation tests for statistical validation
- Learning curves for overfitting detection

Usage:
    python ccars_tuned_pipeline.py --dataset salinas --components 4 --cars_runs 500 \\
        --use_calibration --n_permutations 200 --compute_learning_curves \\
        --preprocessing snv_only --output salinas_ccars_tuned
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

# Local imports (modules in same directory)
from multiclass_cars import MultiClassCARS
from multiclass_plsda import MultiClassPLSDA
from hsi_preprocessing import preprocess_hsi_data
from permutation_test import PermutationTest
from learning_curve import LearningCurve

# Paths
SCRIPT_DIR = Path(__file__).parent
DATASET_BASE = SCRIPT_DIR / "dataset"
RANDOM_SEED = 42

# ===========================================================================
# TUNED HYPERPARAMETERS (from checkerboard grid search)
# ===========================================================================
TUNED_PARAMS = {
    "SVM-RBF": {"C": 100, "gamma": 0.01},       # Was C=10, gamma='scale'
    "Random Forest": {"n_estimators": 500, "max_depth": None},  # Was 200, 20
    "SVM-Linear": {"C": 1}
}

DATASETS = {
    "indian_pines": {
        "cube_path": DATASET_BASE / "indian_pines_corrected.mat",
        "gt_path": DATASET_BASE / "indian_pines_gt.mat",
        "cube_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "block_size": 8,
        "wavelengths_csv": SCRIPT_DIR / "indianpines_wavelengths_200.csv"
    },
    "salinas": {
        "cube_path": DATASET_BASE / "salinas_corrected.mat",
        "gt_path": DATASET_BASE / "salinas_gt.mat",
        "cube_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "block_size": 16,
        "wavelengths_csv": SCRIPT_DIR / "wavelengths_salinas_corrected_204.csv"
    }
}


# ===========================================================================
# DATA LOADING AND CHECKERBOARD SPLIT
# ===========================================================================
def load_cube_and_gt(dataset_name):
    """Load hyperspectral cube and ground truth."""
    config = DATASETS[dataset_name]
    cube_mat = loadmat(str(config["cube_path"]))
    gt_mat = loadmat(str(config["gt_path"]))
    
    # Find cube data
    cube_key = config["cube_key"]
    for key in [cube_key, cube_key.lower(), cube_key.replace("_", "")]:
        if key in cube_mat:
            cube = cube_mat[key].astype(np.float32)
            break
    else:
        arrays = {k: v for k, v in cube_mat.items() if isinstance(v, np.ndarray) and v.ndim == 3}
        cube = arrays[max(arrays, key=lambda k: arrays[k].size)].astype(np.float32)
    
    # Find ground truth
    gt_key = config["gt_key"]
    for key in [gt_key, gt_key.lower(), gt_key.replace("_", "")]:
        if key in gt_mat:
            gt = gt_mat[key].astype(np.int32)
            break
    else:
        arrays = {k: v for k, v in gt_mat.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        gt = arrays[max(arrays, key=lambda k: arrays[k].size)].astype(np.int32)
    
    return cube, gt


def load_wavelengths(dataset_name):
    """Load wavelength values from CSV file."""
    config = DATASETS[dataset_name]
    csv_path = config.get("wavelengths_csv")
    
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        wavelengths = df.iloc[:, 0].values  # First column contains wavelengths
        print(f"  Loaded {len(wavelengths)} wavelengths from {csv_path.name}")
        return wavelengths
    else:
        # Fallback: generate sequential indices
        print(f"  ⚠️ Wavelength CSV not found, using sequential indices")
        return None


def optimized_checkerboard_split(cube, gt, block_size, max_retries=100):
    """Checkerboard spatial split with class balance optimization."""
    H, W, B = cube.shape
    all_classes = set(np.unique(gt)) - {0}
    
    best_split = None
    best_score = -1
    
    for offset in range(max_retries):
        offset_r = offset % block_size
        offset_c = (offset // block_size) % block_size
        
        block_rows = (np.arange(H) + offset_r) // block_size
        block_cols = (np.arange(W) + offset_c) // block_size
        
        row_matrix = block_rows[:, np.newaxis]
        col_matrix = block_cols[np.newaxis, :]
        
        checkerboard = (row_matrix + col_matrix) % 2 == 0
        
        train_mask = checkerboard & (gt > 0)
        test_mask = ~checkerboard & (gt > 0)
        
        train_classes = set(np.unique(gt[train_mask])) - {0}
        test_classes = set(np.unique(gt[test_mask])) - {0}
        
        common = train_classes & test_classes
        score = len(common)
        
        if score > best_score:
            best_score = score
            best_split = (train_mask, test_mask)
        
        if train_classes == all_classes and test_classes == all_classes:
            break
    
    train_mask, test_mask = best_split
    
    return (cube[train_mask], cube[test_mask], 
            gt[train_mask], gt[test_mask])



# ===========================================================================
# TUNED CLASSIFIER FRAMEWORK
# ===========================================================================
class TunedClassifierFramework:
    """Multi-classifier framework with TUNED hyperparameters."""
    
    def __init__(self, n_components=3, random_state=42):
        self.random_state = random_state
        self.n_components = n_components
        
        # TUNED classifiers
        self.classifiers = {
            'PLS-DA': MultiClassPLSDA(n_components=n_components),
            'SVM-Linear': SVC(kernel='linear', C=1, class_weight='balanced', 
                             random_state=random_state),
            'SVM-RBF': SVC(kernel='rbf', C=100, gamma=0.01, class_weight='balanced',
                          random_state=random_state),
            'Random Forest': RandomForestClassifier(
                n_estimators=500, max_depth=None, class_weight='balanced_subsample',
                n_jobs=-1, random_state=random_state
            )
        }
    
    def get_classifier(self, name):
        """Get a fresh classifier instance (not fitted yet)."""
        if name == 'PLS-DA':
            return MultiClassPLSDA(n_components=self.n_components)
        elif name == 'SVM-Linear':
            return SVC(kernel='linear', C=1, class_weight='balanced', 
                       random_state=self.random_state)
        elif name == 'SVM-RBF':
            return SVC(kernel='rbf', C=100, gamma=0.01, class_weight='balanced',
                       random_state=self.random_state)
        elif name == 'Random Forest':
            return RandomForestClassifier(
                n_estimators=500, max_depth=None, class_weight='balanced_subsample',
                n_jobs=-1, random_state=self.random_state
            )
        return self.classifiers.get(name)
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, 
                          classifier_name, wavelength_type='unknown'):
        """Train and evaluate a classifier."""
        clf = self.get_classifier(classifier_name)
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        start_time = time.time()
        clf.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        return {
            'classifier': classifier_name,
            'wavelength_type': wavelength_type,
            'n_features': X_train.shape[1],
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'train_time': train_time,
            'predictions': y_pred,
            'model': clf
        }


# ===========================================================================
# CCARS WITH CALIBRATION (Using existing MultiClassCARS)
# ===========================================================================
def run_ccars_wavelength_selection(X_train, y_train, wavelengths, 
                                   n_components, cars_runs=500, cars_iterations=100,
                                   use_calibration=True, calibration_fraction=0.5,
                                   output_dir=None):
    """Run CCARS wavelength selection."""
    
    if output_dir is None:
        output_dir = Path("ccars_results")
    output_dir = Path(output_dir)
    cars_results_path = output_dir / 'cars_results'
    
    # Initialize MultiClassCARS
    cars = MultiClassCARS(
        output_path=cars_results_path,
        n_components=n_components,
        use_calibration=use_calibration,
        random_state=RANDOM_SEED
    )
    
    # Prepare calibration split if enabled
    if use_calibration:
        # Split train into calibration and final sets
        n_train = len(y_train)
        n_calib = int(n_train * calibration_fraction)
        
        indices = np.random.default_rng(RANDOM_SEED).permutation(n_train)
        calib_indices = indices[:n_calib]
        final_indices = indices[n_calib:]
        
        X_calib = X_train[calib_indices]
        y_calib = y_train[calib_indices]
        X_final = X_train[final_indices]
        y_final = y_train[final_indices]
        
        # Further split calibration into train/test
        X_cal_train, X_cal_test, y_cal_train, y_cal_test = train_test_split(
            X_calib, y_calib, test_size=0.2, stratify=y_calib, random_state=RANDOM_SEED
        )
        
        print(f"  CCARS Calibration Split:")
        print(f"    Calibration: {len(y_calib)} ({calibration_fraction*100:.0f}%)")
        print(f"      - Train: {len(y_cal_train)}, Test: {len(y_cal_test)}")
        print(f"    Final (held-out): {len(y_final)} ({(1-calibration_fraction)*100:.0f}%)")
        
        # Fit CARS with calibration data only
        cars.fit(X_cal_train, y_cal_train, X_cal_test, y_cal_test, wavelengths)
    else:
        # Use all data
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
        )
        cars.fit(X_tr, y_tr, X_te, y_te, wavelengths)
    
    # Run CARS
    print(f"\n  Running CCARS ({cars_runs} runs × {cars_iterations} iterations)...")
    cars.run_cars(
        n_runs=cars_runs,
        n_iterations=cars_iterations,
        mc_samples=0.8,  # Nicola's parameter
        use_ars=True
    )
    
    return cars


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def run_tuned_ccars_pipeline(
    dataset_name='salinas',
    wavelength_counts=[10, 20, 30, 50],
    classifiers=['PLS-DA', 'SVM-RBF', 'Random Forest'],
    cars_runs=500,
    cars_iterations=100,
    pls_components=3,
    output_dir=None,
    use_calibration=True,
    calibration_fraction=0.5,
    n_permutations=1000,
    skip_permutation=False,
    compute_learning_curves=False,
    lc_train_sizes=None,
    preprocessing_method='log10_snv',
    adaptive_permutations=False,
    random_state=42
):
    """
    Comprehensive CCARS pipeline with tuned hyperparameters.
    """
    print("\n" + "="*80)
    print(f"CCARS TUNED PIPELINE - {dataset_name.upper()}")
    print("="*80)
    print(f"TUNED HYPERPARAMETERS:")
    print(f"  SVM-RBF: C=100, gamma=0.01 (was C=10, gamma='scale')")
    print(f"  Random Forest: n_estimators=500, max_depth=None (was 200, 20)")
    print(f"\nWavelength counts: {wavelength_counts}")
    print(f"Classifiers: {classifiers}")
    print(f"CCARS: {cars_runs} runs × {cars_iterations} iterations")
    print(f"PLS Components: {pls_components}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(f"{dataset_name}_ccars_tuned/components_{pls_components}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # =========================================================================
    # Step 1: Load and Preprocess Data with CHECKERBOARD SPLIT
    # =========================================================================
    print("\n" + "="*80)
    print("Step 1: Data Preparation (CHECKERBOARD SPATIAL SPLIT)")
    print("="*80)
    
    cube, gt = load_cube_and_gt(dataset_name)
    H, W, B = cube.shape
    block_size = DATASETS[dataset_name]["block_size"]
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Cube shape: {H}×{W}×{B} bands")
    print(f"  Block size: {block_size}×{block_size}")
    
    # Checkerboard split
    X_train_full, X_test_full, y_train, y_test = optimized_checkerboard_split(
        cube, gt, block_size
    )
    
    print(f"  Train samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    # Preprocess using hsi_preprocessing module
    print(f"\n  Preprocessing: {preprocessing_method}")
    X_train_df = pd.DataFrame(X_train_full)
    X_test_df = pd.DataFrame(X_test_full)
    X_train_processed = preprocess_hsi_data(X_train_df, method=preprocessing_method)
    X_test_processed = preprocess_hsi_data(X_test_df, method=preprocessing_method)
    X_train_full = X_train_processed.values
    X_test_full = X_test_processed.values
    
    # Load actual wavelength values from CSV
    wavelengths = load_wavelengths(dataset_name)
    if wavelengths is None:
        wavelengths = np.arange(B)  # Fallback to indices
    
    # =========================================================================
    # Step 2: Run CCARS Wavelength Selection
    # =========================================================================
    print("\n" + "="*80)
    print("Step 2: CCARS Wavelength Selection")
    print("="*80)
    
    cars = run_ccars_wavelength_selection(
        X_train_full, y_train, wavelengths,
        n_components=pls_components,
        cars_runs=cars_runs,
        cars_iterations=cars_iterations,
        use_calibration=use_calibration,
        calibration_fraction=calibration_fraction,
        output_dir=output_dir
    )
    
    print("\n✓ CCARS wavelength selection complete")
    
    # =========================================================================
    # Step 3: Evaluate with Tuned Classifiers
    # =========================================================================
    print("\n" + "="*80)
    print("Step 3: Classification with TUNED Hyperparameters")
    print("="*80)
    
    framework = TunedClassifierFramework(n_components=pls_components, random_state=random_state)
    all_results = []
    
    # Full spectrum baseline
    print("\nEvaluating FULL SPECTRUM baseline...")
    for clf_name in classifiers:
        result = framework.train_and_evaluate(
            X_train_full, y_train, X_test_full, y_test,
            classifier_name=clf_name, wavelength_type='full_spectrum'
        )
        result['n_wavelengths'] = B
        result['method'] = 'ALL_BANDS'
        all_results.append(result)
        print(f"  {clf_name}: Accuracy={result['accuracy']:.4f}, F1={result['f1_macro']:.4f}")
    
    # Selected wavelengths
    for n_wl in wavelength_counts:
        print(f"\n  Testing {n_wl} wavelengths...")
        
        try:
            selected_wl, wl_freq_df = cars.get_selected_wavelengths(top_n=n_wl)
            
            if len(selected_wl) == 0:
                print(f"    ⚠️ No wavelengths selected for n={n_wl}")
                continue
            
            # Map wavelength values back to band indices
            selected_wl_indices = []
            for wl_val in selected_wl:
                # Find the index where this wavelength value appears
                idx_matches = np.where(np.isclose(wavelengths, wl_val, atol=0.1))[0]
                if len(idx_matches) > 0:
                    selected_wl_indices.append(idx_matches[0])
                else:
                    # Fallback: find closest wavelength
                    closest_idx = np.argmin(np.abs(wavelengths - wl_val))
                    selected_wl_indices.append(closest_idx)
            selected_wl_indices = list(set(selected_wl_indices))  # Remove duplicates
            
            X_train_sel = X_train_full[:, selected_wl_indices]
            X_test_sel = X_test_full[:, selected_wl_indices]
            
            for clf_name in classifiers:
                result = framework.train_and_evaluate(
                    X_train_sel, y_train, X_test_sel, y_test,
                    classifier_name=clf_name, wavelength_type=f'{n_wl}_selected'
                )
                result['n_wavelengths'] = n_wl
                result['method'] = f'CCARS_{n_wl}'
                all_results.append(result)
                print(f"    {clf_name}: Accuracy={result['accuracy']:.4f}")
                
                # =========================================================
                # Permutation Test
                # =========================================================
                if not skip_permutation:
                    if adaptive_permutations:
                        # User-specified counts per classifier type
                        if 'PLS' in clf_name:
                            n_perms = 100
                        elif 'SVM' in clf_name:
                            n_perms = 40
                        elif 'Random Forest' in clf_name:
                            n_perms = 80
                        else:
                            n_perms = 100
                    else:
                        n_perms = n_permutations
                    
                    if n_perms > 0:
                        print(f"      Running permutation test ({n_perms} permutations)...")
                        perm_test = PermutationTest(n_permutations=n_perms, random_state=random_state)
                        p_values = perm_test.run_test(
                            result['model'], X_train_sel, y_train, X_test_sel, y_test,
                            metrics=['accuracy']
                        )
                        result['p_value_accuracy'] = p_values.get('accuracy', None)
                        
                        perm_dir = output_dir / 'permutation_tests'
                        perm_dir.mkdir(exist_ok=True)
                        perm_test.save_results(perm_dir / f'perm_{clf_name}_{n_wl}wl.csv')
                
                # =========================================================
                # Learning Curve
                # =========================================================
                if compute_learning_curves:
                    print(f"      Computing learning curve...")
                    lc = LearningCurve(random_state=random_state)
                    lc_results = lc.compute_curve(
                        result['model'], X_train_sel, y_train,
                        train_sizes=lc_train_sizes
                    )
                    
                    lc_dir = output_dir / 'learning_curves'
                    lc_dir.mkdir(exist_ok=True)
                    lc.save_results(lc_dir / f'lc_{clf_name}_{n_wl}wl.csv')
                    lc.plot_curve(lc_dir / f'lc_{clf_name}_{n_wl}wl.png')
                    
        except Exception as e:
            print(f"    ✗ Error for {n_wl} wavelengths: {e}")
    
    # =========================================================================
    # Step 4: Save Results
    # =========================================================================
    print("\n" + "="*80)
    print("Step 4: Saving Results")
    print("="*80)
    
    # Create results DataFrame
    results_df = pd.DataFrame([{
        'dataset': dataset_name,
        'method': r['method'],
        'classifier': r['classifier'],
        'n_wavelengths': r['n_wavelengths'],
        'accuracy': r['accuracy'],
        'f1_macro': r['f1_macro'],
        'f1_weighted': r['f1_weighted'],
        'kappa': r['kappa'],
        'train_time': r['train_time'],
        'p_value': r.get('p_value_accuracy', None)
    } for r in all_results])
    
    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    # Save
    results_path = output_dir / 'comprehensive_results.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"\n✓ Results saved to: {results_path}")
    print("\nTop 10 Results:")
    print(results_df.head(10).to_string(index=False))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nTUNED HYPERPARAMETERS USED:")
    print("  SVM-RBF: C=100, gamma=0.01")
    print("  Random Forest: n_estimators=500, max_depth=None")
    
    return {'results': results_df, 'cars': cars, 'framework': framework}


# ===========================================================================
# MAIN CLI
# ===========================================================================
def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='CCARS Pipeline with Tuned Hyperparameters and Checkerboard Split'
    )
    
    parser.add_argument('--dataset', type=str, default='salinas',
                       choices=['salinas', 'indian_pines'],
                       help='Dataset name')
    parser.add_argument('--wavelengths', type=int, nargs='*', default=None,
                       help='Wavelength counts to test (default: [10,20,30,50])')
    parser.add_argument('--classifiers', type=str, nargs='+',
                       default=['PLS-DA', 'SVM-RBF', 'Random Forest'],
                       help='Classifiers to evaluate')
    parser.add_argument('--cars_runs', type=int, default=500,
                       help='CCARS Monte Carlo runs (default: 500)')
    parser.add_argument('--cars_iterations', type=int, default=100,
                       help='CCARS iterations per run (default: 100)')
    parser.add_argument('--components', type=int, nargs='+', default=[2, 3, 4],
                       help='PLS components to test (default: 2, 3, 4)')
    parser.add_argument('--use_calibration', action='store_true', default=True,
                       help='Use calibration/final split (CCARS mode)')
    parser.add_argument('--no_use_calibration', dest='use_calibration', action='store_false',
                       help='Disable calibration split')
    parser.add_argument('--calibration_fraction', type=float, default=0.5,
                       help='Fraction for calibration set (default: 0.5)')
    parser.add_argument('--n_permutations', type=int, default=1000,
                       help='Number of permutations (default: 1000)')
    parser.add_argument('--skip_permutation', action='store_true',
                       help='Skip permutation tests')
    parser.add_argument('--compute_learning_curves', action='store_true',
                       help='Compute learning curves')
    parser.add_argument('--lc_train_sizes', type=int, nargs='+', default=None,
                       help='Training sizes for learning curves')
    parser.add_argument('--preprocessing', type=str, default='log10_snv',
                       choices=['snv_only', 'log10_snv', 'none'],
                       help="Preprocessing method (default: 'log10_snv' - Nicola's exact approach)")
    parser.add_argument('--adaptive_permutations', action='store_true',
                       help='Use adaptive permutation counts')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Wavelength counts
    wavelength_counts = args.wavelengths if args.wavelengths else [10, 20, 30, 50]
    
    # Components list
    components_list = args.components if isinstance(args.components, list) else [args.components]
    
    all_results = {}
    
    # Run for each PLS component
    for pls_comp in components_list:
        print(f"\n{'='*80}")
        print(f"Testing with PLS Components: {pls_comp}")
        print(f"{'='*80}\n")
        
        if args.output:
            comp_output = str(Path(args.output) / f"component_{pls_comp}")
        else:
            comp_output = f"{args.dataset}_ccars_tuned/component_{pls_comp}"
        
        results = run_tuned_ccars_pipeline(
            dataset_name=args.dataset,
            wavelength_counts=wavelength_counts,
            classifiers=args.classifiers,
            cars_runs=args.cars_runs,
            cars_iterations=args.cars_iterations,
            pls_components=pls_comp,
            output_dir=comp_output,
            use_calibration=args.use_calibration,
            calibration_fraction=args.calibration_fraction,
            n_permutations=args.n_permutations,
            skip_permutation=args.skip_permutation,
            compute_learning_curves=args.compute_learning_curves,
            lc_train_sizes=args.lc_train_sizes,
            preprocessing_method=args.preprocessing,
            adaptive_permutations=args.adaptive_permutations,
            random_state=args.random_state
        )
        
        all_results[f'components_{pls_comp}'] = results
    
    print("\n" + "="*80)
    print("✓ CCARS TUNED PIPELINE COMPLETE")
    print("="*80)
    
    return all_results


if __name__ == '__main__':
    results = main()
