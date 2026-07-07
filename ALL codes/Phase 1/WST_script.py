#!/usr/bin/env python
"""
This script runs the spectral data processing pipeline for all the following combinations:
  For N_PLS_COMPONENTS in [2, 3] and for the following processing pairs:
    - ("SVN", "SG")
    - ("MSC", "SG")
    - ("SG", "SVN")
    - ("SG", "MSC")

For each combination, the pipeline:
  - Loads and pivots the spectral data,
  - Applies the two-step preprocessing (first_op then second_op),
  - Performs PCA‐based outlier detection,
  - Saves three diagnostic plots (outlier detection, PCA scores with ellipse, and cleaned spectra)
    into a folder named "imgs",
  - Runs the CARS model.

Adjust file paths or parameters as needed.
"""

import sys, os
import time
sys.path.insert(0, "/Users/nicoladilillo/Projects_mac/lettuce_spectral_signature/")

import pandas as pd
import numpy as np
from WST import WST
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from scipy.stats import chi2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.io import loadmat
import argparse

# ------------------------------
# Data Loading Functions for HSI
# ------------------------------
def load_hsi_dataset(dataset_name):
    """Load HSI dataset (Indian Pines or Salinas) with real AVIRIS wavelengths."""
    base_dir = r"C:\Users\hosse\Desktop\Thesis_Phase1_completed\HSI_WST_Pipeline\dataset"
    wavelengths_dir = r"C:\Users\hosse\Desktop\Thesis_Phase1_completed\CCARS_Tuned_Hyperparameters"
    
    if dataset_name == 'indian_pines':
        cube_path = os.path.join(base_dir, 'Indian_pines_corrected.mat')
        gt_path = os.path.join(base_dir, 'Indian_pines_gt.mat')
        cube_key, gt_key = 'indian_pines_corrected', 'indian_pines_gt'
        wavelengths_file = os.path.join(wavelengths_dir, 'indianpines_wavelengths_200.csv')
    elif dataset_name == 'salinas':
        cube_path = os.path.join(base_dir, 'Salinas_corrected.mat')
        gt_path = os.path.join(base_dir, 'Salinas_gt.mat')
        cube_key, gt_key = 'salinas_corrected', 'salinas_gt'
        wavelengths_file = os.path.join(wavelengths_dir, 'wavelengths_salinas_corrected_204.csv')
    else:
        raise ValueError("Dataset must be 'indian_pines' or 'salinas'")
        
    X_cube = loadmat(cube_path)[cube_key]
    y_map = loadmat(gt_path)[gt_key]
    
    rows, cols, bands = X_cube.shape
    X_flat = X_cube.reshape(-1, bands)
    y_flat = y_map.reshape(-1)
    
    # Remove background (class 0)
    mask = y_flat > 0
    X = X_flat[mask]
    y = y_flat[mask]
    
    # Load real AVIRIS wavelengths from CSV file
    if os.path.exists(wavelengths_file):
        wavelengths_df = pd.read_csv(wavelengths_file)
        wavelengths = wavelengths_df.iloc[:, 0].values  # First column contains wavelength values
        print(f"Loaded {len(wavelengths)} real AVIRIS wavelengths from {os.path.basename(wavelengths_file)}")
    else:
        # Fallback to approximate wavelengths if file not found
        wavelengths = np.linspace(400, 2500, bands)
        print(f"Warning: Wavelength file not found, using approximate linspace values")
    
    return X, y, wavelengths, rows, cols

def subsample_stratified(X, y, max_samples=5000):
    """Subsample largely populated datasets to avoid memory issues."""
    if len(y) <= max_samples:
        return X, y
    
    # Stratified sampling
    from sklearn.model_selection import train_test_split
    # Use train_test_split as a quick hack for stratified subsampling
    # train_size is fraction
    fraction = max_samples / len(y)
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=fraction, stratify=y, random_state=42)
    return X_sub, y_sub

# ------------------------------
# Define Processing Functions
# ------------------------------
def msc_normalization(spectra):
    """Apply Multiplicative Scatter Correction (MSC) to each spectrum."""
    reference = np.mean(spectra, axis=0)
    def correct_spectrum(spectrum):
        slope, intercept = np.polyfit(reference, spectrum, 1)
        return (spectrum - intercept) / slope
    corrected = spectra.apply(lambda row: correct_spectrum(row.values), axis=1)
    return pd.DataFrame(corrected.tolist(), index=spectra.index, columns=spectra.columns)

def snv_normalization(spectra):
    """Apply Standard Normal Variate (SNV) normalization to each spectrum."""
    normalized = spectra.apply(lambda row: (row - np.mean(row)) / np.std(row), axis=1)
    # Return the DataFrame as is (do not use .tolist())
    return normalized

def sg_filtering(spectra, window_length=30, polyorder=2, deriv=0):
    """Apply Savitzky–Golay filtering along the wavelength axis for each spectrum."""
    filtered = spectra.apply(lambda row: savgol_filter(row, window_length=window_length, polyorder=polyorder, deriv=deriv), axis=1)
    return pd.DataFrame(filtered.tolist(), index=spectra.index, columns=spectra.columns)

def plot_confidence_ellipse(ax, x, y, confidence=0.95, **kwargs):
    """Plot a confidence ellipse based on the covariance of x and y."""
    if x.size != y.size:
        raise ValueError("x and y must have the same size.")
    cov_xy = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    vx, vy = eigvecs[:, 0]
    theta = np.degrees(np.arctan2(vy, vx))
    chi2_val = chi2.ppf(confidence, 2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    ellipse = patches.Ellipse((mean_x, mean_y), width, height, angle=theta,
                              fill=False, linestyle='--', linewidth=2, **kwargs)
    ax.add_patch(ellipse)

def mahalanobis_squared(score, center, inv_cov):
    diff = score - center
    return diff.T @ inv_cov @ diff

def plot_spectral(title, plot_name, all_classes, X_clean, img_folder):
    plt.figure(figsize=(12, 8))
    sns.set_context("paper", font_scale=2)
    sns.set_style("whitegrid")
    palette = sns.color_palette("viridis", n_colors=len(all_classes))
    color_mapping = {c: palette[i] for i, c in enumerate(all_classes)}
    
    wavelengths = X_clean.columns.values.astype(float)
    
    # Plot by class for efficiency
    for c in all_classes:
        # Select samples for this class
        # Assuming MultiIndex level 1 is 'Class'
        try:
            subset = X_clean[X_clean.index.get_level_values('Class') == c]
        except KeyError:
            # Fallback if index structure differs
             continue
             
        if subset.empty:
            continue
            
        # Transpose to plot: (n_wavelengths, n_samples)
        # Matplotlib plots columns as lines
        plt.plot(wavelengths, subset.values.T, color=color_mapping[c], alpha=0.3, linewidth=0.5)

    plt.gca().set_xlim([min(wavelengths), max(wavelengths)])
    plt.gca().set_ylim([X_clean.min().min(), X_clean.max().max()])
    
    legend_elements = [Line2D([0], [0], color=color_mapping[label], lw=2, label=label) for label in color_mapping]
    plt.legend(handles=legend_elements, title="", fontsize=20, title_fontsize=22)
    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.grid(False)
    
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(img_folder, plot_name), bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f'Plot saved: {plot_name}')

# ------------------------------
# Main Pipeline Function
# ------------------------------
def run_pipeline(n_components, preopocessing_ops, dataset_name, plots_only=False, skip_to=None):
    """Run the full pipeline. skip_to can be: 'cars', 'boss', 'ga_ipls', 'ga_ipls_boss'"""
    pipeline_name = f"{n_components}_" +  "_".join(preopocessing_ops).upper()
    print("--------------------------------------------------")
    print("\nRunning pipeline:", pipeline_name)
    if skip_to:
        print(f"Skipping to: {skip_to}")
    
    # --- Create images folder if not exists ---
    img_folder = f"imgs/{dataset_name}/{pipeline_name}/"
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    # Load and prepare data
    print(f"Loading {dataset_name} dataset...")
    X, y, wavelengths, _, _ = load_hsi_dataset(dataset_name)
    
    # Subsample if needed
    X, y = subsample_stratified(X, y, max_samples=5000)
    
    # Construct DataFrame to match Nicola's structure
    # Expected Index: ['Date', 'Class', 'Stress_weight', 'Position']
    # We will fake this:
    dates = ['2024-01-01'] * len(y)
    stress = ['HSI'] * len(y)
    positions = range(len(y))
    
    col_group = ['Date', 'Class', 'Stress_weight', 'Position']
    index = pd.MultiIndex.from_arrays([dates, y, stress, positions], names=col_group)
    
    # Wavelengths as columns (rounded string for consistency if needed, but float works usually)
    # Nicola's code assumed 'Wavelength' column in melted csv. Here we pivot directly.
    X_df = pd.DataFrame(X, index=index, columns=wavelengths)
    
    print("Samples per class:\n", X_df.index.get_level_values('Class').value_counts())
    
    unique_classes_raw = X_df.index.get_level_values('Class').unique()
    plot_spectral("Raw Data", f"all_raw_spectra.jpg", unique_classes_raw, X_df, img_folder)
    
    # --- Apply processing operations ---
    title = ""
    for preop in preopocessing_ops:
        if preop.upper() == "MSC":
            X_df = msc_normalization(X_df)
        elif preop.upper() in ["SVN", "SNV"]:
            X_df = snv_normalization(X_df)
        elif preop.upper() == "SG":
            X_df = sg_filtering(X_df)
        elif preop.upper() == "SG1":
            X_df = sg_filtering(X_df, deriv=1)
            
        else:
            raise ValueError("Invalid preprocessing operation. Use 'MSC', 'SVN', or 'SG'.")
        title += f"{preop.upper()} + "
    title = title.rstrip(" +")
    
    print(f"Pipeline applied!")
    X_processed = X_df
    
    # --- PCA-based Outlier Detection ---
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_processed)
    center = np.mean(scores, axis=0)
    cov = np.cov(scores, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    distances_squared = np.array([mahalanobis_squared(score, center, inv_cov) for score in scores])
    chi2_threshold = chi2.ppf(0.95, n_components)
    print("Chi-square threshold (95% confidence):", chi2_threshold)
    outlier_mask = distances_squared > chi2_threshold
    num_outliers = np.sum(outlier_mask)
    X_clean = X_processed[~outlier_mask]
    print(f"Number of outlier samples removed: {num_outliers}")
    print("Remaining samples:", X_clean.shape[0])
    
    # --- Remove classes with too few samples for stratified split ---
    MIN_SAMPLES_PER_CLASS = 10  # Minimum samples needed for reliable train/test split
    class_counts = X_clean.index.get_level_values('Class').value_counts()
    small_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index.tolist()
    if small_classes:
        print(f"\nWARNING: Removing {len(small_classes)} classes with <{MIN_SAMPLES_PER_CLASS} samples: {small_classes}")
        # Filter out small classes
        mask = ~X_clean.index.get_level_values('Class').isin(small_classes)
        X_clean = X_clean[mask]
        print(f"Samples after removing small classes: {X_clean.shape[0]}")
    
    # Print the number of samples per class
    print("Samples per class after filtering:\n", X_clean.index.get_level_values('Class').value_counts())
    
    # --- Plot 1: Outlier Detection ---
    plt.figure(figsize=(10, 6))
    colors = ['red' if d > chi2_threshold else 'blue' for d in distances_squared]
    plt.scatter(range(len(distances_squared)), distances_squared, c=colors, s=50, label='Samples')
    plt.axhline(chi2_threshold, color='green', linestyle='--', linewidth=2, label='Chi2 Threshold (95%)')
    plt.xlabel('Sample Index')
    plt.ylabel('Squared Mahalanobis Distance')
    # plt.title(f'Outlier Detection ({pipeline_name}, {n_components} comp)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, f"outlier_detection_{pipeline_name}.jpg"), dpi=150)
    plt.close()
    
    # --- Plot 2: PCA Scores with Confidence Ellipse ---
    plt.figure(figsize=(10, 8))
    inlier_mask = ~outlier_mask
    plt.scatter(scores[inlier_mask, 0], scores[inlier_mask, 1], c='blue', s=50, label='Inliers')
    plt.scatter(scores[outlier_mask, 0], scores[outlier_mask, 1], c='red', s=70, marker='X', label='Outliers')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA Scores ({pipeline_name}, {n_components} comp)')
    ax = plt.gca()
    plot_confidence_ellipse(ax, scores[inlier_mask, 0], scores[inlier_mask, 1], confidence=0.95, edgecolor='green')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, f"pca_scores_{pipeline_name}.jpg"), dpi=150)
    plt.close()
    
    # --- Plot 3: Cleaned and All Spectra using Seaborn ---
    unique_classes = X_clean.index.get_level_values('Class').unique()
    plot_spectral(title, f"cleaned_spectra_{pipeline_name}.jpg", unique_classes, X_clean, img_folder)
    
    unique_classes_processed = X_processed.index.get_level_values('Class').unique()
    plot_spectral("", f"all_spectra_{pipeline_name}.jpg", unique_classes_processed, X_processed, img_folder)
    
    if plots_only:
        print("Plots generated. Skipping model training (--plots_only).")
        return

    # --- Run the CARS Model ---
    if skip_to not in ['cars', 'boss', 'ga_ipls', 'ga_ipls_boss']:
        print("\nRunning CARS model...")
        start_time = time.time()
        path = os.path.join(os.path.abspath(os.getcwd()), pipeline_name, 'CARS')
        
        # Check for existing statistics to enable auto-resume
        stats_folder = os.path.join(path, 'statistics')
        start_run = 0
        total_runs = 500
        if os.path.exists(stats_folder):
            existing_files = [f for f in os.listdir(stats_folder) if f.startswith('statistics_') and f.endswith('.csv')]
            if existing_files:
                # Find highest completed run number
                completed_runs = [int(f.replace('statistics_', '').replace('.csv', '')) for f in existing_files]
                start_run = max(completed_runs) + 1
                if start_run >= total_runs:
                    print(f"CARS already completed ({start_run} runs found). Skipping.")
                    save_elapsed_time(start_time, path)
                else:
                    print(f"Resuming CARS from run {start_run}/{total_runs} ({len(existing_files)} runs completed)")
        
        if start_run < total_runs:
            c = WST(path, col_group, X_clean, MAX_COMPONENTS=n_components, CV_FOLD=5, test_percentage=0.3, cutoff=0.5)
            c.perform_pca()
            c.cars_model(R=total_runs - start_run, N=100, MC_SAMPLES=0.8, start=start_run)
            save_elapsed_time(start_time, path)
    else:
        print("Skipping CARS (already completed)")

    # --- Run the BOSS Model ---
    if skip_to not in ['boss', 'ga_ipls', 'ga_ipls_boss']:
        print("\nRunning BOSS model...")
        start_time = time.time()
        path = os.path.join(os.path.abspath(os.getcwd()), pipeline_name, 'BOSS')
        
        # Check if already completed
        if os.path.exists(os.path.join(path, 'elapsed_time.txt')):
            print("BOSS already completed (elapsed_time.txt found). Skipping.")
        else:
            c = WST(path, col_group, X_clean, MAX_COMPONENTS=n_components, CV_FOLD=5, test_percentage=0.3, cutoff=0.5)
            c.perform_pca()
            c.save_results()
            c.boss_model(speed=0)
            save_elapsed_time(start_time, path)
    else:
        print("Skipping BOSS (already completed)")

    # --- Run the GA-iPLS Model ---
    if skip_to not in ['ga_ipls', 'ga_ipls_boss']:
        print("\nRunning GA-iPLS model...")
        start_time = time.time()
        path = os.path.join(os.path.abspath(os.getcwd()), pipeline_name, 'GA-iPLS')
        
        # Check if already completed
        if os.path.exists(os.path.join(path, 'elapsed_time.txt')):
            print("GA-iPLS already completed (elapsed_time.txt found). Skipping.")
        else:
            c = WST(path, col_group, X_clean, MAX_COMPONENTS=n_components, CV_FOLD=5, test_percentage=0.3, cutoff=0.5)
            c.perform_pca()
            c.save_results()
            c.compute_ga_ipls(population_size=40, generations=100, crossover_prob=0.6, mutation_prob=0.1, n_intervals=100)
            save_elapsed_time(start_time, path)
    else:
        print("Skipping GA-iPLS (already completed)")

    # --- Run the GA-iPLS + BOSS Model ---
    print("\nRunning GA-iPLS + BOSS model...")
    start_time = time.time()
    
    # Select the only wavelengths that were selected by the GA-iPLS
    file_name = f'{pipeline_name}/GA-iPLS'
    path = os.path.join(os.path.abspath(os.getcwd()), file_name)
    c = WST(path, MAX_COMPONENTS=n_components, col_group=col_group, cutoff=0.5)
    w_str = c.compute_survived_wavelengths_best_score_single()
    
    # Convert wavelength strings to floats and find closest matching columns
    w_target = [float(i.split('.')[0]) for i in w_str]  # e.g., 611.0
    available_cols = X_clean.columns.values.astype(float)
    
    # Find the closest column for each target wavelength
    selected_cols = []
    for wt in w_target:
        closest_idx = np.argmin(np.abs(available_cols - wt))
        selected_cols.append(available_cols[closest_idx])
    selected_cols = list(set(selected_cols))  # Remove duplicates
    
    # Select only the wavelengths that were selected by the GA-iPLS
    X_clean = X_clean[selected_cols]
  
    
    # Run the BOSS model with the selected wavelengths
    path = os.path.join(os.path.abspath(os.getcwd()), pipeline_name, 'GA-iPLS_BOSS')
    c = WST(path, col_group, X_clean, MAX_COMPONENTS=n_components, CV_FOLD=5, test_percentage=0.3, cutoff=0.5)
    c.perform_pca()
    c.save_results()
    c.boss_model(speed=0)
    
    save_elapsed_time(start_time, path)
       
    print(f"Pipeline {pipeline_name} completed.\n")
    print("--------------------------------------------------")
    print()
    print()

def save_elapsed_time(start_time, path):
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed time: {int(hours)}:{int(minutes):02}:{int(seconds):02}")
    # Save the elapsed time to a file
    with open(os.path.join(path, f"elapsed_time.txt"), 'w') as f:
        f.write(f"Elapsed time: {int(hours)}:{int(minutes):02}:{int(seconds):02}\n")

# ------------------------------
# Main: Loop Through All Combinations
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description='Run WST Pipeline on HSI Data')
    parser.add_argument('--dataset', type=str, required=True, choices=['salinas', 'indian_pines'], help='Dataset to use')
    parser.add_argument('--plots_only', action='store_true', help='Generate plots only and skip model training')
    parser.add_argument('--skip_to', type=str, choices=['cars', 'boss', 'ga_ipls', 'ga_ipls_boss'], 
                        help='Skip to a specific step (cars, boss, ga_ipls, ga_ipls_boss)')
    parser.add_argument('--preprocessing', type=str, nargs=2, default=None,
                        help='Run only a specific preprocessing combo, e.g. --preprocessing SG SVN')
    args = parser.parse_args()

    # Define the four allowed processing pairs
    combinations = [
        ["SG", "SVN"],
        ["SG", "MSC"],
        ["SG1", "SVN"],
        ["SG1", "MSC"]
    ]
    
    # Filter to specific preprocessing if provided
    if args.preprocessing:
        combinations = [[args.preprocessing[0].upper(), args.preprocessing[1].upper()]]
        print(f"Running only: {combinations[0]}")

    n_components_max = 10
       
    img_folder = "imgs"
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    for combination in combinations:
        run_pipeline(n_components_max, combination, args.dataset, args.plots_only, args.skip_to)

if __name__ == "__main__":
    main()
