#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file cars.py
@brief Competitive Adaptive Reweighted Sampling (CARS) analysis module.

This module implements a CARS class that performs variable selection using 
the CARS algorithm. It supports both classification (using a PLS‐DA approach) 
and regression (using PLS) analyses.
"""

import os
import math
import random
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.svm import SVC
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (confusion_matrix, f1_score, mean_squared_error, precision_score, 
                             recall_score, accuracy_score, roc_curve, auc, r2_score)
from sklearn.model_selection import learning_curve, train_test_split
from scipy.signal import find_peaks
from sklearn.model_selection import StratifiedKFold

sns.set_context("paper", font_scale=2)
sns.set_style("whitegrid")
color_line = '#1f77b4'

# =============================================================================
# PLSDAClassifier
# =============================================================================
##
# @brief PLS Discriminant Analysis classifier based on PLSRegression.
#
# This classifier fits a PLS regression model on binary‐encoded labels and then 
# automatically determines an optimal cutoff (via Youden's J statistic on the ROC curve)
# to convert continuous predictions into class labels.
class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, cutoff=None):
        """
        Initialize the classifier.

        @param n_components: Number of latent components.
        @param cutoff: Initial cutoff value (will be optimized during fit).
        """
        self.n_components = n_components
        self.cutoff = cutoff
        self.pls = PLSRegression(n_components=self.n_components)

    def fit(self, X, y):
        """
        Fit the PLS-DA model on the training data and compute the optimal cutoff.

        @param X: Training feature matrix.
        @param y: Training target vector.
        @return: self.
        """
        # Determine the unique classes and binarize the target variable.
        self.classes_ = np.unique(y)
        # Assume the second unique value is the positive class.
        self.y_binary = (y == self.classes_[1]).astype(int)
        # Fit the underlying PLS regression model.
        self.pls.fit(X, self.y_binary)
        
        if self.cutoff is None:
            # Predict continuous outputs and flatten the result.
            y_pred_cont = self.pls.predict(X).ravel()
            
            # Compute the ROC curve.
            fpr, tpr, thresholds = roc_curve(self.y_binary, y_pred_cont)
            
            # Calculate Youden's J statistic and select the best threshold.
            J = tpr - fpr
            best_idx = np.argmax(J)
            self.cutoff = thresholds[best_idx]
        
    def predict(self, X):
        """
        Predict class labels for the given input using the computed cutoff.

        @param X: Input feature matrix.
        @return: Predicted class labels.
        """
        # Get continuous predictions from the underlying PLS model.
        y_pred_cont = self.pls.predict(X).ravel()
        # Apply the cutoff to obtain binary predictions.
        y_pred = (y_pred_cont >= self.cutoff).astype(int)
        return self.classes_[y_pred]

    @property
    def coef_(self):
        """
        Get the regression coefficients.
        """
        return self.pls.coef_

# =============================================================================
# WST Class
# =============================================================================
##
# @brief CARS (Competitive Adaptive Reweighted Sampling) analysis class.
#
# This class performs variable selection using the CARS algorithm. It supports 
# both classification (PLS-DA) and regression (PLS) tasks.
#
# @param path Directory path for saving results.
# @param col_group List of grouping columns.
# @param X_df Input DataFrame containing features. Its index must include the 
#             class/target column.
# @param MAX_LABEL Maximum number of labels for scree plots.
# @param MAX_COMPONENTS Maximum number of components to use in PCA/PLS.
# @param CV_FOLD Number of cross-validation folds.
# @param OPTIMAL_N Default optimal number of components.
# @param calibration If True, the data is “calibrated” (classes balanced).
# @param class_column Name of the target column.
# @param test_percentage Fraction of data to set aside for testing.
# @param scalar Optional scaler (e.g. StandardScaler) to preprocess data.
# @param task Task type: 'classification' or 'regression'.
# @param cutoff Optional cutoff value for classification.
class WST:
    def __init__(self, path, col_group=None, X_df=None, MAX_LABEL=10, MAX_COMPONENTS=10, 
                 FIX_COMPONENTS=None, CV_FOLD=5, class_column='Class', test_percentage=0.3,
                 random_state=42, scalar=None, task='classification', score_type='accuracy', cutoff=None):
        """
        Initialize the CARS analysis.

        @param path: Directory path for saving results.
        @param col_group: List of grouping columns.
        @param X_df: Input DataFrame with features (index must include class_column).
        @param MAX_LABEL: Maximum number of labels for scree plots.
        @param MAX_COMPONENTS: Maximum number of components for PCA/PLS.
        @param CV_FOLD: Number of cross-validation folds.
        @param OPTIMAL_N: Default optimal number of components.
        @param calibration: If True, balance classes.
        @param class_column: Name of the target column.
        @param test_percentage: Fraction of data set aside for testing.
        @param scalar: Optional scaler for preprocessing.
        @param task: Task type ('classification' or 'regression').
        """
        self.MAX_LABEL = MAX_LABEL
        self.MAX_COMPONENTS = MAX_COMPONENTS
        self.FIX_COMPONENTS = FIX_COMPONENTS
        self.CV_FOLD = CV_FOLD
        self.MAX_W = 10
        self.TEST = test_percentage
        self.random_state = random_state
        self.col_group = col_group
        self.col_class = class_column
        self.task = task
        self.score_type = score_type
        self.cutoff = cutoff

        # Setup result directories.
        self.path = self._check_exists(path)
        self.path_statistics = self._check_exists(os.path.join(path, 'statistics'))
        self.path_coefficients = self._check_exists(os.path.join(path, 'coefficients'))
        
        if X_df is not None:
            # Determine index position of the class column.
            self.col_class_i  = X_df.index.names.index(class_column)
            self.class_labels = X_df.index.get_level_values(class_column).unique()
            
            self.df_original = X_df
            self.X, self.y, self.index = self._random_input(self.df_original)
            self.X_train, self.X_test, self.y_train, self.y_test, self.index_train, self.index_testing = train_test_split(self.X, self.y, self.index, test_size=self.TEST, random_state=self.random_state)
            
            
            self.N_SAMPLES = self.X.shape[0]
            self.N_SAMPLES_TRAIN = self.X_train.shape[0]
            self.P = self.X.shape[1]
            self.columns = self.df_original.columns
            self.wavelengths = np.array([f"{w:.3f}" for w in self.columns])
        else:
            # If no data is provided, load previously saved results.
            self.load_results(class_column)
        
        self._split_train_test()
        
        print(f"Training/Testing split: {100 - self.TEST * 100:.1f}% training, {self.TEST * 100:.1f}% testing")

        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Testing samples: {self.X_test.shape[0]}")

        # Get all unique class and stress_weight combinations from train and test
        train_classes = self.index_train.get_level_values('Class').unique()
        train_stress = self.index_train.get_level_values('Stress_weight').unique()
        test_classes = self.index_testing.get_level_values('Class').unique()
        test_stress = self.index_testing.get_level_values('Stress_weight').unique()
        all_classes = sorted(set(train_classes).union(test_classes))
        all_stress = sorted(set(train_stress).union(test_stress))

        print("\nTrain/Test split by Class and Stress_weight:")
        for c in all_classes:
            for s in all_stress:
                train_mask = ((self.index_train.get_level_values('Class') == c) & (self.index_train.get_level_values('Stress_weight') == s))
                test_mask = ((self.index_testing.get_level_values('Class') == c) & (self.index_testing.get_level_values('Stress_weight') == s))
                n_train = train_mask.sum()
                n_test = test_mask.sum()
                if n_train > 0 or n_test > 0:
                    print(f"  Class {c}, Stress_weight {s}: {n_train} train, {n_test} test")

    def _split_train_test(self):
        # Balance the class division between 'Class' and 'Stress_weight'
        # Get the values for both columns from the index
        class_labels  = self.index.get_level_values('Class')
        stress_labels = self.index.get_level_values('Stress_weight')
        # Combine them into a single stratification label
        strat_labels = class_labels.astype(str) + "_" + stress_labels.astype(str)
        self.X_train, self.X_test, self.y_train, self.y_test, self.index_train, self.index_testing = train_test_split(
                self.X, self.y, self.index, test_size=self.TEST, random_state=self.random_state, stratify=strat_labels
            )
        
        self.N_SAMPLES_TRAIN = self.X_train.shape[0]

    def _random_input(self, dataset, shuffle=False):
        """
        Randomly shuffle the dataset and extract feature matrix and target vector.

        @param dataset: Input DataFrame.
        @param shuffle: If True, shuffle the data.
        @return: Tuple (X, y) where X is a NumPy array of features and y the labels.
        """
        df = dataset.sample(frac=1, random_state=self.random_state) if shuffle else dataset
        X = df.to_numpy()
        self.y_label = np.array([x[self.col_class_i] for x in df.index])
        self.label_binarizer = LabelBinarizer()
        if self.task == 'classification':
            y = self.label_binarizer.fit_transform(self.y_label)
            y = self.y_label
        else:
            y = np.array(self.y_label).reshape(-1, 1).astype(float)
        return X, y, df.index

    def _check_exists(self, path):
        """
        Check if a directory exists; if not, create it.

        @param path: Directory path.
        @return: Verified (or created) path.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def perform_pca(self, survived=False, wavelengths=None):
        """
        Perform PCA on the original data.

        @param survived: If True, use survived wavelengths from variable selection.
        @param wavelengths: Optional list of wavelengths to use.
        """
        index_row = self.df_original.index
        if survived:
            # Use only wavelengths that survived variable selection.
            survived_w = self.survived_df['Wavelengths'].value_counts()
            i = survived_w[survived_w >= 1].index
            X_p = self.df_original.fillna(0).values[:, i]
        elif wavelengths is not None:
            # Use provided wavelengths.
            vars_selected_i = [list(self.wavelengths).index(w) for w in wavelengths]
            X_p = self.df_original.fillna(0).values[:, vars_selected_i]
        else:
            X_p = self.df_original.fillna(0).values
        
        # Select the best number of components for PCA
        if self.FIX_COMPONENTS is not None:
            n_best = self.FIX_COMPONENTS
        else:
            n_best = self.MAX_COMPONENTS
        if n_best > self.P:
            n_best = self.P
        pca = PCA(n_components=n_best)
    
        X_pca = pca.fit_transform(X_p)
        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
        labels = [f'PC{i}' for i in range(1, len(per_var) + 1)]
        self.pca_screen_df = pd.DataFrame({'Components': labels, 'Variance': per_var})
        self.pca_df = pd.DataFrame(X_pca, index=index_row, columns=labels)
        self.PCA_1_labels = f'PC1 - {per_var[0]:.2f}%'
        self.PCA_2_labels = f'PC2 - {per_var[1]:.2f}%'
        if self.MAX_COMPONENTS > 2:
            self.PCA_3_labels = f'PC3 - {per_var[2]:.2f}%'
            self.df_PCA = pd.DataFrame(columns=[self.PCA_1_labels, self.PCA_2_labels, self.PCA_3_labels, *self.col_group])
        else:
            self.df_PCA = pd.DataFrame(columns=[self.PCA_1_labels, self.PCA_2_labels, *self.col_group])
        # Build a DataFrame for PCA plot with group information.
        for (i, sample) in enumerate(self.pca_df.index):
            new_data = pd.DataFrame({self.PCA_1_labels: [self.pca_df.iloc[i, 0]],
                                     self.PCA_2_labels: [self.pca_df.iloc[i, 1]]})
            if self.MAX_COMPONENTS > 2:
                new_data[self.PCA_3_labels] = self.pca_df.iloc[i, 2]
            for (j, n) in enumerate(self.col_group):
                new_data[n] = sample[j]
            self.df_PCA = pd.concat([self.df_PCA, new_data], ignore_index=True)

    def pca_3d_plot(self):
        """
        Plot a 3D PCA scatter plot.
        """
        if self.MAX_COMPONENTS > 2:
            fig = px.scatter_3d(self.df_PCA, x=self.PCA_1_labels, y=self.PCA_2_labels, z=self.PCA_3_labels,
                                color='Class', title='PCA 3D Plot', opacity=0.7, width=1000, height=1000, 
                                hover_data=self.col_group)
            fig.update_layout(scene=dict(aspectmode="cube"))
            fig.show()
        else:
            print("Insufficient components for a 3D plot.")

    def pca_2d_plot(self, W=8, L=6, suffix=''):
        """
        Plot 2D PCA scatter plots and save them.

        @param W: Plot width.
        @param L: Plot height.
        @param suffix: Optional suffix for saved file names.
        """
        fig = px.scatter(self.df_PCA, x=self.PCA_1_labels, y=self.PCA_2_labels, color='Class', 
                         title='PCA1 vs PCA2', opacity=0.7, width=500, height=500, hover_data=self.col_group)
        fig.show()
        plt.figure(figsize=(W, L))
        sns.scatterplot(x=self.PCA_1_labels, y=self.PCA_2_labels, data=self.df_PCA, 
                        hue='Class', palette='viridis')
        filename = f'PCA1_vs_PCA2_{suffix}.pdf' if suffix else 'PCA1_vs_PCA2.pdf'
        plt.savefig(os.path.join(self.path, filename), bbox_inches='tight')
        plt.show()
        plt.close()
        if self.MAX_COMPONENTS > 2:
            fig = px.scatter(self.df_PCA, x=self.PCA_1_labels, y=self.PCA_3_labels, color='Class', 
                             title='PCA1 vs PCA3', opacity=0.7, width=500, height=500, hover_data=self.col_group)
            fig.show()
            plt.figure(figsize=(W, L))
            sns.scatterplot(x=self.PCA_1_labels, y=self.PCA_3_labels, data=self.df_PCA, 
                            hue='Class', palette='viridis')
            filename = f'PCA1_vs_PCA3_{suffix}.pdf' if suffix else 'PCA1_vs_PCA3.pdf'
            plt.savefig(os.path.join(self.path, filename), bbox_inches='tight')
            plt.show()
            plt.close()
            fig = px.scatter(self.df_PCA, x=self.PCA_2_labels, y=self.PCA_3_labels, color='Class', 
                             title='PCA2 vs PCA3', opacity=0.7, width=500, height=500, hover_data=self.col_group)
            fig.show()
            plt.figure(figsize=(W, L))
            sns.scatterplot(x=self.PCA_2_labels, y=self.PCA_3_labels, data=self.df_PCA, 
                            hue='Class', palette='viridis')
            filename = f'PCA2_vs_PCA3_{suffix}.pdf' if suffix else 'PCA2_vs_PCA3.pdf'
            plt.savefig(os.path.join(self.path, filename), bbox_inches='tight')
            plt.show()
            plt.close()
        sns.set(font_scale=1)

    def cars_model(self, R=500, N=100, MC_SAMPLES=0.8, start=0):
        """
        Run the CARS model with Monte Carlo sampling iterations.

        @param R: Number of runs.
        @param N: Number of iterations per run.
        @param MC_SAMPLES: Fraction of samples for each Monte Carlo iteration.
        @param start: Starting run index.
        """
        self.set_data_df()
        K = int(MC_SAMPLES * self.N_SAMPLES_TRAIN)
        self._compute_cars(K, N, start, start + R)

    def _compute_cars(self, K, N, start, end):
        """
        Internal method to compute CARS over multiple runs.

        @param K: Number of samples per Monte Carlo iteration.
        @param N: Number of iterations per run.
        @param start: Starting run index.
        @param end: Ending run index.
        """
        a = (self.P / 2) ** (1 / (N - 1))
        k = math.log(self.P / 2) / (N - 1)
        
        for j in tqdm(range(start, end)):
            vars_selected_i = list(range(self.P))
            wavelength_weights = np.ones(self.P)/self.P
            num_LTs = np.zeros(N)
            errors  = np.zeros(N)
            ratio   = 1.0
            n_selected_variables = self.P
            
            # Compute metrics and store results using all samples
            self._cross_predict(self.X_train[:,vars_selected_i], self.y_train, model_type='PLS') 
            self._populate_partial(ratio, j, errors, num_LTs, 0, vars_selected_i, self.pls.coef_.flatten(), wavelength_weights, n_selected_variables)
            
            for i in range(1, N):
                # Randomly select K samples for training
                indices = np.random.choice(self.N_SAMPLES_TRAIN, size=K, replace=False)
                
                # Fit the model using the selected samples and the selected variables with cross-validation
                self._cross_predict(self.X_train[indices,:][:,vars_selected_i], self.y_train[indices], model_type='PLS')   
                
                # Extract coefficients and compute weights
                wavelength_weights, coefficients = self._compute_wavelength_weights(vars_selected_i)
                               
                # Find the ratio and select variabless
                ratio =  a * math.exp(-k * (i+1))
                
                # Compute the variable for the next iteration, always more than 2
                n_selected_variables = int(round(ratio * self.P, 0))
                if n_selected_variables < 2:
                    n_selected_variables = 2

                # Select the variables based on the computed weights 
                while True:                  
                    vars_selected_i = self._select_variables(n_selected_variables, wavelength_weights)
                    if len(vars_selected_i) >= 2:
                        break
                
                # Save the new model metrics
                self._cross_predict(self.X_train[:,vars_selected_i], self.y_train, model_type='PLS') 
                self._populate_partial(ratio, j, errors, num_LTs, i, vars_selected_i, coefficients, wavelength_weights, n_selected_variables)

            self._save_partial(j)
        self.save_results()

    def _save_partial(self, run):
        """
        Save partial results to disk for a given run.

        @param run: The current run index.
        """
        self.statiscs_df[self.statiscs_df.Run == run].to_csv(
            os.path.join(self.path_statistics, f'statistics_{run}.csv'), index=False)
        self.coefficients_df[self.coefficients_df.Run == run].to_csv(
            os.path.join(self.path_coefficients, f'coefficients_{run}.csv'), index=False)
        
        del self.statiscs_df
        del self.coefficients_df
        self.set_data_df()

    def _select_variables(self, n_selected_variables, wavelength_weights):
        """
        Select variables based on computed weights, at least two elements.

        @param n_selected_variables: Number of variables to select.
        @param wavelength_weights: Array of computed weights.
        @return: Sorted indices of selected variables.
        """
        # Remove the less used wavelengths
        sorted_indices  = np.argsort(wavelength_weights)[::-1]
        wavelength_weights[sorted_indices[n_selected_variables:]] = 0
        wavelength_weights /= np.sum(wavelength_weights)
        
        vars_selected_i = []
        while  len(vars_selected_i) < 2:
            vars_selected_i = np.array(list(set(np.random.choice(range(self.P), size=self.P, replace=True, p=wavelength_weights))))
                
        return np.sort(vars_selected_i)

    def _compute_wavelength_weights(self, var_selected_i):
        """
        Compute normalized wavelength weights from regression coefficients.

        @return: Array of normalized weights.
        """
        coefficients = np.zeros(self.P)
        coefficients[var_selected_i] = self.pls.coef_.flatten()
        tot = np.sum(np.abs(coefficients))
        return np.array([x / tot for x in np.abs(coefficients)]), coefficients

    def _compute_metrics(self, X, y, X_test, y_test, var_selected_i=None, model_type='PLS', learning_curve=False, confusion_matrix_f=False, pred_f=False):
        """
        Compute key performance metrics.
        
        For classification, returns:
          (accuracy, recall, precision, f1, cutoff)
        For regression, returns:
          (MSE, RMSE, R², Q²)
        
        Optionally plots a learning curve.

        @param X: Training features.
        @param y: Training targets.
        @param X_test: Testing features.
        @param y_test: Testing targets.
        @param var_selected_i: Selected variable indices.
        @param model_type: Model type ('PLS').
        @param learning_curve: If True, plot the learning curve.
        @param confusion_matrix: If True, plot the confusion matrix.
        @return: Tuple of performance metrics.
        """
        # For classification, use the predict method (which applies the cutoff internally).
        # print(f"Selected variables: {len(var_selected_i)}")
        y_pred = self.pls.predict(X_test)
        # print(f"Predicted shape: {y_pred.shape}, Test shape: {y_test.shape}")
        
        if self.task == 'classification':
            acc = accuracy_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics = {
                'accuracy': acc,
                'recall': rec,
                'precision': prec,
                'f1': f1,
            }
            if model_type == 'PLS':
                metrics['cutoff'] = self.pls.cutoff
                metrics['nLV'] = self.pls.n_components
            
            if confusion_matrix_f:
                cm = confusion_matrix(y_test, y_pred)
                
                # Calculate specificity - handle multi-class case
                if cm.shape[0] == 2:
                    # Binary classification
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    # Multi-class: compute macro-averaged specificity (one-vs-rest)
                    specificities = []
                    for i in range(cm.shape[0]):
                        # For class i, treat it as positive and rest as negative
                        tp = cm[i, i]
                        fp = cm[:, i].sum() - tp  # Other classes predicted as i
                        fn = cm[i, :].sum() - tp  # Class i predicted as others
                        tn = cm.sum() - tp - fp - fn  # True negatives
                        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        specificities.append(spec_i)
                    specificity = np.mean(specificities)
                
                metrics['specificity'] = float(specificity)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'{len(var_selected_i)} Wavelegths')
                plt.savefig(os.path.join(self.path, f'confusion_matrix_{len(var_selected_i)}.pdf'), bbox_inches='tight')
                # plt.show()
                plt.close()
                # Save confusion matrix as a DataFrame
                cm_df = pd.DataFrame(cm, index=self.class_labels, columns=self.class_labels)
                cm_df.to_csv(os.path.join(self.path, f'confusion_matrix_{len(var_selected_i)}.csv'), index=True)
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            SS_press = np.sum((y_test - y_pred) ** 2)
            SS_total = np.sum((y_test - np.mean(y)) ** 2)
            q2 = 1 - (SS_press / SS_total)
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'q2': q2
            }
            if model_type == 'PLS':
                metrics['nLV'] = self.pls.n_components
        
        if learning_curve:
            self._compute_learning_curve(X[:, var_selected_i], y, model_type=model_type)
        
        if pred_f:
            return metrics, y_pred
        else:
            return metrics

    def accuracy_survived_wavelenghts(self, thr=None, rdm=False, all=False, model_type='PLS', 
                                      wavelengths=None, learning_curve=False, roc_plot=False, 
                                      pls_plot=False, confusion_matrix_f=False, peak_detection=False, 
                                      peak_height=10, peak_distance=10):
        """
        Evaluate model performance using selected wavelength combinations.
        Performs train/test split inside this method for each combination.
        Optionally adds a combination based on peak-detection.
        Prints metrics for classification or regression.
        Args:
            thr: List of thresholds for variable occurrence.
            rdm: Shuffle input data.
            all: Include all features.
            model_type: Model type ('PLS').
            wavelengths: List of wavelength combinations to test.
            learning_curve: Plot learning curve.
            roc_plot: Plot ROC curve (classification only).
            pls_plot: Plot PLS components.
            confusion_matrix_f: Plot confusion matrix.
            peak_detection: Add peak-detected combination.
            peak_height: Minimum peak height for peak detection.
            peak_distance: Minimum peak distance for peak detection.
            test_size: Fraction for test split.
            random_state: Random seed for reproducibility.
        Returns:
            metrics_list, w_list, pred_y
        """
        var_combinations = []  # List of variable index combinations to test

        # Build combinations from provided wavelengths
        if wavelengths is not None:
            for e in wavelengths:
                e = sorted(e)
                # Convert all to string with 3 decimals for matching
                wavelengths_str = [f"{float(w):.3f}" for w in e]
                var_combinations.append([list(self.wavelengths).index(w) for w in wavelengths_str])

        # Add combinations from thresholds
        if thr is not None:
            for t in thr:
                vars_selected_i = self.extract_select_variable(t)
                vars_selected_i.sort()
                var_combinations.append(tuple(vars_selected_i))

        # Optionally add all features
        if all:
            var_combinations.append(tuple(range(self.P)))

        # Add peak-detected combination if requested
        if peak_detection and hasattr(self, 'plot_survived_df'):
            counts = self.plot_survived_df['count'].astype(float).values
            wavelengths_numeric = np.array([float(w) for w in self.plot_survived_df['Wavelengths']])
            peaks, properties = find_peaks(counts, height=peak_height, distance=peak_distance)
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths_numeric, counts, label="Counts")
            if len(peaks) > 0:
                plt.plot(wavelengths_numeric[peaks], counts[peaks], "x", label="Detected Peaks", markersize=10, color="red")
            plt.title("Survived Wavelength Counts and Detected Peaks")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Count")
            plt.legend()
            plt.show()
            if len(peaks) > 0:
                sorted_peak_indices = peaks[np.argsort(properties['peak_heights'])[::-1]]
                peak_wavelengths = self.plot_survived_df['Wavelengths'].iloc[sorted_peak_indices].tolist()
                peak_indices = [list(self.wavelengths).index(w) for w in peak_wavelengths if w in self.wavelengths]
                print("Peak-detected indices using SciPy:", peak_indices)
                var_combinations.append(tuple(peak_indices))
            else:
                print("No peaks detected in the survived wavelengths counts.")

        print(f'\nTotal combinations to test: {len(var_combinations)}')
        metrics_list = []
        w_list = []
        latex = ""

        # Use original data for splitting
        X_full = self.df_original.to_numpy()
        y_full = np.array([x[self.col_class_i] for x in self.df_original.index])
        
        for i in range(len(var_combinations) - 1, -1, -1):
            if not var_combinations[i]:
                print(f"Skipping empty combination at index {i}")
                continue
            
            var_combinations[i] = sorted(var_combinations[i])

            current_wavelengths = [self.wavelengths[w] for w in var_combinations[i]]
            w_list.append(current_wavelengths)
            print(f'\nCombination {i}: {current_wavelengths}')
            print(f'Number of wavelengths: {len(current_wavelengths)}')

            # Save current wavelengths into a file (one per line for easy reading)
            with open(os.path.join(self.path, f'wavelengths_{len(current_wavelengths)}.txt'), 'w') as f:
                for w in current_wavelengths:
                    f.write(f"{w}\n")

            # Perform train/test split for each combination
            X_train, X_test = self.X_train[:, var_combinations[i]], self.X_test[:, var_combinations[i]]

            # Fit and evaluate model
            self._cross_predict(X_train, self.y_train, model_type=model_type)
            if self.task == 'classification':
                metrics, pred_y = self._compute_metrics(X_train, self.y_train, X_test, self.y_test, pred_f=True,
                                                       var_selected_i=list(range(len(var_combinations[i]))), model_type=model_type,
                                                       learning_curve=learning_curve, confusion_matrix_f=confusion_matrix_f)
                print(f"Accuracy: {metrics['accuracy']:.2f}, Recall: {metrics['recall']:.2f}, Precision: {metrics['precision']:.2f}, Specifity: {metrics['specificity']:.2f}, F1: {metrics['f1']:.2f}", end=" ")
                if model_type == 'PLS':
                    print(f", nLV: {metrics['nLV']}, Cutoff: {metrics['cutoff']:.2f}")
                else:
                    print()
            else:
                metrics, pred_y = self._compute_metrics(X_train, self.y_train, X_test, self.y_test, pred_f=True,
                                                       var_selected_i=list(range(len(var_combinations[i]))), model_type=model_type, learning_curve=learning_curve)
                print(f"MSE: {metrics['mse']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.2f}, Q2: {metrics['q2']:.2f}")

            metrics_list.append(metrics)

            # Optional plots
            if roc_plot and self.task == 'classification':
                self.plot_roc(var_combinations[i])
            if pls_plot:
                self.plot_PLS(X_train, self.y_train, list(range(len(var_combinations[i]))))

            calibration = 'False' if hasattr(self, 'df_original_2') and self.df_original.equals(self.df_original_2) else 'True'
            if self.task == 'classification':
                latex += f"{calibration} & {len(var_combinations[i])} & {metrics['accuracy']:.2f} & {metrics['recall']:.2f} & {metrics['precision']:.2f} & {metrics['f1']:.2f}"
                if model_type == 'PLS':
                    latex += f"  & {metrics['nLV']:.2f}  & {metrics['cutoff']:.2f} \\ \\hline\n"
                else:
                    latex += "\\\hline\n"
            else:
                latex += f"{self.MAX_COMPONENTS} & {calibration} & {len(var_combinations[i])} & {metrics['mse']:.2f} & {metrics['rmse']:.2f} & {metrics['r2']:.2f} & {metrics['q2']:.2f} \\ \\hline\n"

            metrics['LV'] =  self.pls.n_components if model_type == 'PLS' else None
            metrics['Cutoff'] = self.pls.cutoff if model_type == 'PLS' else None

            # Save metrics into a file
            with open(os.path.join(self.path, f'metrics_{len(current_wavelengths)}.txt'), 'w') as f:
                f.write(metrics.__str__())

        print()
        print(latex)
        return metrics_list, w_list, pred_y

    def _compute_roc_auc(self, y_test, y_pred):
        """
        Compute ROC curve and AUC (for classification).

        @param y_test: True target values.
        @param y_pred: Predicted continuous outputs.
        @return: fpr, tpr, thresholds, and AUC.
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def plot_roc(self, vars_selected_i):
        """
        Plot ROC curve for classification.

        @param vars_selected_i: Selected variable indices.
        """
        self._cross_predict(self.X_2[:,vars_selected_i], self.y_2)
        y_pred_cv = self.pls.predict(self.X_test_2[:,vars_selected_i]).ravel()
        fpr, tpr, thresholds, roc_auc = self._compute_roc_auc(self.y_test_2, y_pred_cv)
        df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Thresholds': thresholds})
        fig = px.line(df, x='False Positive Rate', y='True Positive Rate', markers=True,
                      hover_data='Thresholds', title=f'ROC Curve - AUC {roc_auc:.2f}')
        fig.update_layout(showlegend=True)
        fig.show()    

    def plot_frequency(self, starting=500, separation=100, width=12, height=6, threshold=None):
        """
        Plot the frequency of wavelengths with custom x-ticks and an optional threshold line.

        @param starting: Starting x-value for ticks.
        @param separation: Spacing between ticks.
        @param width: Figure width.
        @param height: Figure height.
        @param threshold: Optional threshold value.
        """
        plt.figure(figsize=(width, height))
        ax = sns.barplot(data=self.plot_survived_df, x='Wavelengths', y='count')
        x_values = [float(x) for x in self.wavelengths]
        desired_ticks = [x for x in range(starting, int(max(x_values)), separation)]
        tick_positions = [str(min(x_values, key=lambda x: abs(x - tick))) for tick in desired_ticks]
        plt.xlabel('Wavelength (nm)')
        plt.ylabel("Occurrences")
        plt.xticks(tick_positions, labels=desired_ticks)
        for bar in ax.containers[0].get_children():
            bar.set_color(color_line)
        if threshold is not None:
            plt.axhline(y=threshold, color='r', linestyle='--')
            plt.text(200, threshold + 5, 'Threshold', color='red')
        plt.savefig(f'{self.path}/Frequency_of_Wavelengths.pdf')
        # plt.show()
        plt.close()
        fig = px.histogram(self.plot_survived_df, x='Wavelengths', y='count',
                           labels={'Wavelengths': 'Wavelengths', 'sum of count': 'Frequency'},
                           title='Frequency of Wavelengths')
        fig.show()

    def plot_selected_variables(self, run=0):
        """
        Plot the evolution of selected variables over sampling runs.

        @param run: Run index.
        """
        
        fig = px.line(self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Selected Variables',
                      labels={'Iteration': 'Iteration', 'Selected Variables': 'Selected Variables'},
                      hover_name='Selected Wavelengths',
                      title='Selected Variables Sampling Runs')
        fig.show()
        plt.figure(figsize=(14, 8))
        self.statiscs_df = pd.concat([self.statiscs_df, 
                                      pd.DataFrame({'Run': run, 'Iteration': 0, 'Ratio': 1, 
                                                    'Selected Variables': self.P, 
                                                    'Selected Wavelengths': 'All wavelengths', 
                                                    self.score_type: self.statiscs_df[self.statiscs_df['Run'] == run][self.score_type].values[0]},
                                                   index=[0])], ignore_index=True)
        sns.lineplot(data=self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Selected Variables', 
                     linewidth=3, color=color_line)
        plt.xlabel('Iteration')
        plt.ylabel('Selected Variables')
        plt.xlim(0, self.statiscs_df[self.statiscs_df['Run'] == run]['Iteration'].max())
        plt.title('')
        plt.savefig(os.path.join(self.path, 'Selected_Variables_sampling_runs.pdf'))
        plt.title('Selected Variables over Iterations')
        plt.savefig(os.path.join(self.path, 'Selected_Variables_sampling_runs_titled.pdf'))
        plt.close()

    def plot_ratio(self, run=0):
        """
        Plot the variable reduction ratio over sampling runs.

        @param run: Run index.
        """
        fig = px.line(self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Ratio',
                      labels={'Iteration': 'Iteration', 'Ratio': 'Ratio'},
                      hover_name='Selected Wavelengths',
                      title='Ratio Value Sampling Runs')
        fig.show()
        sns.lineplot(data=self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Ratio')
        plt.xlabel('Iteration')
        plt.ylabel('Ratio')
        plt.title('Ratio Value Sampling Runs')
        plt.savefig(os.path.join(self.path, 'Ratio_value_sampling_runs.pdf'))
        plt.close()

    def plot_rmsecv(self, run=0):
        """
        Plot RMSECV for regression over sampling runs.

        @param run: Run index.
        """
        fig = px.line(self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='RMSECV',
                      labels={'Iteration': 'Iteration', 'RMSECV': 'RMSECV'},
                      hover_name='Selected Wavelengths',
                      title='RMSECV Value Sampling Runs')
        fig.show()
        sns.lineplot(data=self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='RMSECV')
        plt.xlabel('Iteration')
        plt.ylabel('RMSECV')
        plt.title('RMSECV Value Sampling Runs')
        plt.savefig(os.path.join(self.path, 'RMSECV_value_sampling_runs.pdf'))
        plt.close()

    def plot_score(self, run=0):
        """
        Plot accuracy (for classification) over sampling runs.

        @param run: Run index.
        """
        if self.task == 'classification':
            fig = px.line(self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y=self.score_type,
                          labels={'Iteration': 'Iteration', self.score_type: self.score_type},
                          hover_name='Selected Wavelengths',
                          title='Accuracy Sampling Runs')
            fig.show()
            best_accuracy = self.statiscs_df[self.statiscs_df['Run'] == run][self.score_type].max()
            best_row = self.statiscs_df.sort_values(by='Iteration', ascending=False)[
                (self.statiscs_df['Run'] == run) & (self.statiscs_df[self.score_type] == best_accuracy)]
            plt.figure(figsize=(14, 8))
            self.statiscs_df['Score_app'] = self.statiscs_df[self.score_type].apply(lambda x: round(x, 2))
            sns.lineplot(data=self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Score_app', 
                         linewidth=3, color=color_line)
            plt.plot(best_row['Iteration'].values[0], round(best_accuracy, 2), 'ro')
            plt.text(best_row['Iteration'].values[0]+1, best_accuracy, f'{best_accuracy:.2f}', verticalalignment='top', color='r')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.xlim(0, self.statiscs_df[self.statiscs_df['Run'] == run]['Iteration'].max())
            plt.yticks(np.arange(0.75, 0.97, 0.05))
            plt.title('')
            plt.savefig(os.path.join(self.path, 'Accuracy_value_sampling_runs.pdf'))
            plt.title('Accuracy over iterations')
            plt.savefig(os.path.join(self.path, 'Accuracy_value_sampling_runs_titled.pdf'))
            plt.close()
        else:
            print("Accuracy plot is available only for classification tasks.")

    def plot_coefficients(self, run=0):
        """
        Plot evolution of coefficients over sampling runs.

        @param run: Run index.
        """
        fig = px.line(self.coefficients_df[self.coefficients_df['Run'] == run], x='Iteration', y='Cofficients',
                      labels={'Iteration': 'Iteration', 'Cofficients': 'Coefficients'},
                      title='Coefficients Sampling Runs', line_group='Wavelengths', color='Wavelengths')
        fig.show()
        sns.lineplot(data=self.coefficients_df[self.coefficients_df['Run'] == run], x='Iteration', y='Cofficients', hue='Wavelengths')
        plt.xlabel('Iteration')
        plt.ylabel('Coefficients')
        plt.title('Coefficients over Iterations')
        plt.savefig(os.path.join(self.path, 'Coefficients_sampling_runs.pdf'))
        plt.close()
    
    def plot_weight_coefficients(self, run=0):
        """
        Plot evolution of coefficient weights over sampling runs.

        @param run: Run index.
        """
        fig = px.line(self.coefficients_df[self.coefficients_df['Run'] == run], x='Iteration', y='Weights',
                      labels={'Iteration': 'Iteration', 'Weight': 'Weights'},
                      title='Coefficient Weights Sampling Runs', line_group='Wavelengths', color='Wavelengths')
        fig.show()
        sns.lineplot(data=self.coefficients_df[self.coefficients_df['Run'] == run], x='Iteration', y='Weights', hue='Wavelengths')
        plt.xlabel('Iteration')
        plt.ylabel('Coefficient Weights')
        plt.title('Coefficient Weights over Iterations')
        plt.savefig(os.path.join(self.path, 'Coefficients_sampling_runs.pdf'))
        plt.close()

    def save_results(self):
        """
        Save PCA and CARS results to disk.
        """
        self.df_PCA.to_csv(os.path.join(self.path, 'pca.csv'), index=False)
        self.pca_screen_df.to_csv(os.path.join(self.path, 'pca_screen.csv'), index=False)
        self.df_original.to_csv(os.path.join(self.path, 'X_df.csv'), index=True)
        np.savetxt(os.path.join(self.path, 'wavelengths.txt'), self.wavelengths, fmt='%s')
        with open(os.path.join(self.path, 'col_class_i.txt'), 'w') as f:
            f.write(f'{self.col_class_i}\n')

    def load_results_partial(self):
        """
        Load partial results from disk.
        """
        self.set_data_df()
        
        for file in os.listdir(self.path_statistics):
            self.statiscs_df = pd.concat([self.statiscs_df, pd.read_csv(os.path.join(self.path_statistics, file))], ignore_index=True)
        for file in os.listdir(self.path_coefficients):
            self.coefficients_df = pd.concat([self.coefficients_df, pd.read_csv(os.path.join(self.path_coefficients, file))], ignore_index=True)

    def load_results(self, class_column):
        """
        Load all results from disk.

        @param class_column: Name of the target column.
        """
        self.load_results_partial()
        self.df_PCA = pd.read_csv(os.path.join(self.path, 'pca.csv'))
        self.pca_screen_df = pd.read_csv(os.path.join(self.path, 'pca_screen.csv'))
        self.PCA_1_labels = self.df_PCA.columns[0]
        self.PCA_2_labels = self.df_PCA.columns[1]
        self.PCA_3_labels = self.df_PCA.columns[2] if len(self.df_PCA.columns) > 2 else None

        self.wavelengths = []
        with open(os.path.join(self.path, 'wavelengths.txt')) as f:
            for line in f:
                self.wavelengths.append(f"{float(line.strip()):.3f}")

        self.col_class_i = int(np.loadtxt(os.path.join(self.path, 'col_class_i.txt'), dtype=int))
        
        # Read pivot tables using _read_pivot.
        self.df_original, self.X, self.y, self.index = self._read_pivot(os.path.join(self.path, 'X_df.csv'))
        
        self.N_SAMPLES = self.df_original.shape[0]
        self.P = self.df_original.shape[1]

        self.class_labels = self.df_original.index.get_level_values(class_column).unique()
        self.col_class_i = self.df_original.index.names.index(class_column)

    def _read_pivot(self, path):
        """
        Read a pivoted DataFrame from CSV.

        @param path: File path.
        @return: Tuple (df, X, y).
        """
        if self.col_group is not None:
            df = pd.read_csv(path, index_col=self.col_group)
        else:
            df = pd.read_csv(path)
        X, y, index = self._random_input(df)
        return df, X, y, index

    def compute_survived_wavelengths_n_variables(self, n_variables=2):
        """
        Compute survived wavelengths using iterations with a fixed number of selected variables.
        
        For classification, chooses the iteration with maximum Accuracy.
        For regression, chooses the iteration with minimum RMSECV.
        
        @param n_variables: Number of variables to fix.
        """
        if self.task == 'classification':
            idx = self.statiscs_df[self.statiscs_df['Selected Variables'] == n_variables] \
                        .groupby('Run')['Accuracy'].idxmax().tolist()
        else:
            idx = self.statiscs_df[self.statiscs_df['Selected Variables'] == n_variables] \
                        .groupby('Run')['RMSECV'].idxmin().tolist()
        self._compute_survived_wavelengths(idx)

    def compute_survived_wavelengths_best_score(self):
        """
        Compute survived wavelengths using the best metric per run.
        
        For classification, selects the iteration with maximum Accuracy.
        For regression, selects the iteration with minimum RMSECV.
        """
        idx = self.statiscs_df.sort_values(by='Selected Variables') \
                        .groupby('Run')[self.score_type].idxmax().tolist()
        self._compute_survived_wavelengths(idx)
        
    def compute_survived_wavelengths_best_score_single(self):
        """
        Compute survived wavelengths using the best metric per run.
        
        For classification, selects the iteration with maximum Accuracy.
        For regression, selects the iteration with minimum RMSECV.
        """
        idx = self.statiscs_df.sort_values(by='Selected Variables')\
            [self.score_type].idxmax()
        self._compute_survived_wavelengths([idx])
        print(f"Best {self.score_type} for run {self.statiscs_df.iloc[idx]['Run']} and iteration {self.statiscs_df.iloc[idx]['Iteration']}")
        print(f"Best {self.score_type} = {self.statiscs_df.iloc[idx][self.score_type]}")
        return self.plot_survived_df[self.plot_survived_df['count'] > 0]['Wavelengths'].to_list()
    
    def _compute_survived_wavelengths(self, idx):
        """
        Compute the survived wavelengths from selected iterations.

        @param idx: List of indices (run/iteration) to consider.
        """
        survived_stats = self.statiscs_df.iloc[idx][['Run', 'Iteration']]
        self.survived_df = pd.merge(survived_stats, self.coefficients_df, on=['Run', 'Iteration'])[['Wavelengths', 'Run']]
        self.survived_df.sort_values(by=['Wavelengths', 'Run'], inplace=True)
        self.survived_df['Wavelengths'] = self.survived_df['Wavelengths'].apply(lambda x: f"{float(x):.3f}")
        count_survived = self.survived_df['Wavelengths'].value_counts().reset_index()
        count_survived.columns = ['Wavelengths', 'count']
        wavelengths_df = pd.DataFrame({'Wavelengths': [str(w) for w in self.wavelengths]})
        self.plot_survived_df = pd.merge(wavelengths_df, count_survived, on='Wavelengths', how='left').fillna(0)

    def plot_PLS(self, X, y, var_selected_i):
        """
        Plot PLS components.

        @param X: Feature matrix.
        @param y: Target vector.
        @param var_selected_i: Selected variable indices.
        """
        X_pls = self.pls.transform(X[:, var_selected_i])
        array = np.concatenate((X_pls, y), axis=1)
        columns = [f'PLS Component {i+1}' for i in range(self.MAX_COMPONENTS)]
        columns.append('Class')
        df_pls = pd.DataFrame(array, columns=columns)
        for i, c in enumerate(self.class_labels):
            df_pls.loc[df_pls['Class'] == i, 'Class'] = c
        sns.set_context("paper", font_scale=1.5)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_pls, x='PLS Component 1', y='PLS Component 2', hue='Class', palette='viridis', s=70)
        plt.xlabel("PLS Component 1")
        plt.ylabel("PLS Component 2")
        plt.grid(True)
        plt.savefig(os.path.join(self.path, f'PLS_Components_{len(var_selected_i)}.pdf'))
        plt.show()
        plt.close()
        plt.figure(figsize=(8, 10))
        g = sns.pairplot(df_pls, hue='Class', palette='viridis', corner=False)
        g.fig.subplots_adjust(left=0.2, wspace=0.1, hspace=0.1)
        for ax in g.axes[:, 0]:
            if ax is not None:
                ax.yaxis.set_label_coords(-0.4, 0.5)
        plt.savefig(os.path.join(self.path, f'PLS_Components_pp_{len(var_selected_i)}.pdf'))
        plt.show()
        plt.close()
        sns.set_context("paper", font_scale=2)

    def extract_select_variable(self, thr):
        """
        Extract variables that appear more than a given threshold.

        @param thr: Occurrence threshold.
        @return: List of selected variable indices.
        """
        survived_w = self.survived_df['Wavelengths'].value_counts()
        survived_w = survived_w[survived_w >= thr].index
        vars_selected_i = [list(self.wavelengths).index(f"{float(w):.3f}") for w in survived_w]
        return vars_selected_i

    def _cross_predict(self, X, y, model_type='PLS'):
        """
        Cross-validate the model and select the best number of components for PLS-DA.

        @param X: Training features.
        @param y: Training targets.
        @param model_type: Model type.
        @return: None. Sets self.pls to the best model.
        """
        if model_type == 'PLS' and self.task == 'classification':
            best_score = -np.inf
            best_n = 2
            max_components = min(self.MAX_COMPONENTS, X.shape[1])
            for nc in range(2, max_components + 1):
                cv = StratifiedKFold(n_splits=self.CV_FOLD, shuffle=True, random_state=42)
                scores = []
                for train_idx, test_idx in cv.split(X, y):
                    model = PLSDAClassifier(n_components=nc, cutoff=self.cutoff)
                    model.fit(X[train_idx], y[train_idx])
                    y_pred = model.predict(X[test_idx])
                    score = accuracy_score(y[test_idx], y_pred)
                    scores.append(score)
                mean_score = np.mean(scores)
                # print(f"n_components: {nc}, Mean Accuracy: {mean_score:.4f}")
                if mean_score > best_score:
                    best_score = mean_score
                    best_n = nc
            # Fit final model on all data with best_n
            self.pls = PLSDAClassifier(n_components=best_n, cutoff=self.cutoff)
            # print(X.shape, y.shape, best_n)
            self.pls.fit(X, y)
        elif model_type == 'PLS' and self.task == 'regression':
            # For regression, you can implement similar logic for RMSECV if needed
            self.pls = PLSRegression(n_components=min(self.MAX_COMPONENTS, X.shape[1]))
            self.pls.fit(X, y)
        elif model_type == 'SVM' and self.task == 'classification':
            self.pls = SVC(kernel='linear', probability=True, random_state=42)
            self.pls.fit(X, y)
        else:
            raise NotImplementedError('Only PLS model_type is supported in _cross_predict.')

    def _compute_learning_curve(self, X, y, model_type='PLS'):
        """
        Plot the learning curve.

        @param X: Feature matrix.
        @param y: Target vector.
        @param cutoff: Cutoff value (for classification; not used for regression).
        @param len_var: Number of selected variables.
        @param model_type: Model type.
        """
        len_var = len(X[1])
        if model_type == 'PLS':
            # For classification, instantiate PLSDAClassifier without externally providing cutoff.
            model = PLSDAClassifier(n_components=self.MAX_COMPONENTS, cutoff=self.cutoff) if self.task=='classification' else PLSRegression(n_components=self.MAX_COMPONENTS)
            train_sizes, train_scores, validation_scores = learning_curve(model, X, y,
                                                                           train_sizes=np.linspace(0.1, 1.0, 30), 
                                                                           scoring='accuracy' if self.task=='classification' else 'r2',
                                                                           random_state=42, cv=self.CV_FOLD)
        data = []
        n_train_sizes = train_sizes.shape[0]
        n_folds = train_scores.shape[1]
        for i in range(n_train_sizes):
            for j in range(n_folds):
                data.append({'Train_sizes': train_sizes[i], 'Score': train_scores[i, j], 'Set': 'Training'})
                data.append({'Train_sizes': train_sizes[i], 'Score': validation_scores[i, j], 'Set': 'Validation'})
        plot_data = pd.DataFrame(data)
        plot_data.to_csv(os.path.join(self.path, 'Learning_Curve_Data.csv'), index=False)
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=plot_data, x='Train_sizes', y='Score', hue='Set', palette='viridis', linewidth=3, marker='o')
        plt.ylim(0.49, 1)
        plt.xlabel("Training Samples", fontsize=26)
        plt.ylabel("Accuracy" if self.task=='classification' else "R2", fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(loc="lower right", fontsize=26)
        title = f"{len_var} Wavelengths"
        plt.title(title, fontsize=28)
        filename = f'Learning_Curve_{len_var}_titled.pdf' if len_var is not None else 'Learning_Curve_titled.pdf'
        plt.savefig(os.path.join(self.path, filename))
        plt.show()
        plt.close()

    def permutation_test(self, wavelengths=None, N=1000, save_file=True):
        """
        Perform a permutation test to assess model significance.

        @param wavelengths: List of wavelengths to test.
        @param N: Number of permutations.
        @param save_file: If True, save the permutation results.
        """
        return
        if wavelengths is None:
            var_combinations = range(self.P)
        else:
            wavelengths_str = [f"{w:.3f}" if isinstance(w, float) else w for w in wavelengths]
            var_combinations = [list(self.wavelengths).index(w) for w in wavelengths_str]
        n = len(var_combinations)
        
        # At position 0 put the right metrics
        permutation_scores = []
        
        X_train = self.X_train[:,var_combinations]
        y_train = self.y_train.copy()
        X_test = self.X_test[:,var_combinations]
        y_test = self.y_test
        self._cross_predict(X_train, y_train, model_type='PLS')

        if self.task == 'classification':
            metrics = self._compute_metrics(X_train, y_train, X_test, y_test, var_selected_i=var_combinations, model_type='PLS')
            print(f"Accuracy: {metrics['accuracy']:.2f}, Recall: {metrics['recall']:.2f}, Precision: {metrics['precision']:.2f}, F1: {metrics['f1']:.2f}")
            permutation_scores.append([metrics['accuracy'], metrics['recall'], metrics['precision'], metrics['f1']])
        else:
            metrics = self._compute_metrics(X_train, y_train, X_test, y_test, var_selected_i=var_combinations, model_type='PLS')
            print(f"MSE: {metrics['mse']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.2f}, Q2: {metrics['q2']:.2f}")
            permutation_scores.append([metrics['mse'], metrics['rmse'], metrics['r2'], metrics['q2']])
        
        for _ in tqdm(range(N)):
            np.random.shuffle(y_train)
            # np.random.shuffle(y_test)
            self._cross_predict(X_train, y_train, model_type='PLS')
            if self.task == 'classification':
                metrics = self._compute_metrics(X_train, y_train, X_test, y_test, var_selected_i=var_combinations, model_type='PLS')
                permutation_scores.append([metrics['accuracy'], metrics['recall'], metrics['precision'], metrics['f1']])
            else:
                metrics = self._compute_metrics(X_train, y_train, X_test, y_test, var_selected_i=var_combinations, model_type='PLS')
                permutation_scores.append([metrics['mse'], metrics['rmse'], metrics['r2'], metrics['q2']])
            
        if save_file:
            file_path = os.path.join(self.path, f'permutation_test_{n}.csv')
            print(f'Saving permutation test results to {file_path}')
            with open(file_path, 'w') as f:
                if self.task == 'classification':
                    f.write('Accuracy,Recall,Precision,F1\n')
                else:
                    f.write('MSE,RMSE,R2,Q2\n')
    
                for metrics in permutation_scores:
                    f.write(f"{metrics[0]:.2f},{metrics[1]:.2f},{metrics[2]:.2f},{metrics[3]:.2f}\n")
                        
    def boss_model(self, num_bootstrap=1000, speed=0, end=100):
        """
        Run the BOSS model.

        @param R: Number of runs.
        @param N: Number of iterations per run.
        @param MC_SAMPLES: Fraction of samples for each Monte Carlo iteration.
        @param rmsecv: If True, compute RMSECV (regression) or accuracy (classification).
        @param start: Starting run index.
        """
        if num_bootstrap >= 2000:
            ratio = 0.05
        elif 1000 <= num_bootstrap < 2000:
            ratio = 0.1
        elif 500 <= num_bootstrap < 1000:
            ratio = 0.2
        else:
            print("The number of bootstrap sampling ought to be larger than 500")
            return
        
        self.set_data_df()
        self._compute_boss(num_bootstrap, speed, ratio, end)
        # self._compute_boss(num_bootstrap, speed, ratio, end)

    def set_data_df(self):
        self.statiscs_df = pd.DataFrame(columns=['Run', 'Iteration', 'Ratio', 'Selected Variables', 'Selected Wavelengths', 'LV'])
        self.coefficients_df = pd.DataFrame(columns=['Run', 'Iteration', 'Wavelengths', 'Cofficients'])

    def _compute_boss(self, num_bootstrap, speed, ratio, end):
        """
        Internal method to compute BOSS over multiple runs.

        @param K: Number of samples per Monte Carlo iteration.
        @param N: Number of iterations per run.
        @param start: Starting run index.
        @param end: Ending run index.
        @param rmsecv: If True, compute RMSECV/accuracy.
        """
        # Number of variables
        n = self.P
        num_retained = n
        num_best = round(num_bootstrap * ratio)

        # Arrays to store selected variables, weights, and best errors for each outer iteration
        Variable = np.zeros((n, end))
        W = np.zeros((n, end))
        best_errors = np.zeros(end)
        Num_retained = np.zeros(end)
        Num_LV = np.zeros(end)
        
        # Outer loop: iterate for a fixed number of iterations ('end')
        for j in range(end):
            # Initialize arrays for the bootstrap iterations in this outer loop
            errors = np.zeros(num_bootstrap)
            num_LTs = np.zeros(num_bootstrap)
            variable_index = np.zeros((n, num_bootstrap))
            B = np.zeros((n, num_bootstrap))
            
            # Initialize equal weights for all variables
            w = np.ones(n) / n
            
            # Inner loop: bootstrap sampling iterations
            for i in tqdm(range(num_bootstrap), desc=f"Bootstrap Iteration {j+1}/{end} - Retaining {num_retained} variables"):
                # Weighted sampling of variable indices with replacement;
                # use np.unique to ensure each index appears only once adn that there are at least 2 wavelenghts
                while True:
                    Vsel = np.unique(np.random.choice(n, size=num_retained, replace=True, p=w))
                    if len(Vsel) >= 2:
                        break
                # print(Vsel, w)
                
                # Create a binary indicator vector for the selected variables
                indicator = np.zeros(n)
                indicator[Vsel] = 1
                variable_index[:, i] = indicator
                
                # Compute the model using the selected variables
                self._cross_predict(self.X_train[:,Vsel], self.y_train, model_type='PLS')
                
                # Extract the model coefficients and normalize them
                # Create a coefficient array corresponding to all variables; assign normalized coefficients to the selected ones
                b, coefficients = self._compute_wavelength_weights(Vsel)
                B[:, i] = coefficients / np.linalg.norm(coefficients)
                
                # Compute performance metrics for the model
                self._populate_partial(ratio, j, errors, num_LTs, i, Vsel, coefficients, b, num_retained)
            
            self._save_partial(j)
            
            # Sort the best values
            sorted_indices = self._compute_sorted_index(errors, variable_index)
            
            best_index = sorted_indices[0]
            best_errors[j] = errors[best_index]
            Num_LV[j] = num_LTs[best_index]
            Variable[:, j] = variable_index[:, best_index]
            # wavelength_weights = np.sum(B[:,sorted_indices[:num_best]], axis=0)
            # wavelength_weights /= np.linalg.norm(wavelength_weights)
            print(f"Best {self.score_type.upper()} for iteration {j}: {best_errors[j]:.4f}, selected variables: {np.sum(Variable[:, j])}")
            
            # Update weights: aggregate the coefficients of the top 'num_best' bootstrap samples
            top_indices = sorted_indices[:num_best]
            w = np.abs(np.sum(B[:, top_indices], axis=1))
            # Normalize the updated weights to sum to 1
            w =  w/np.linalg.norm(w)
            W[:, j] = w
            
            if speed == 0:
                num_retained = np.sum(variable_index) / num_bootstrap
            elif speed == 1:
                num_retained = np.sum(variable_index) / num_bootstrap * 1.1
            elif speed == 2:
                num_retained = np.sum(variable_index) / num_bootstrap * 1.2
            elif speed == 3:
                num_retained = np.sum(variable_index) / num_bootstrap * 1.3
            elif speed == 4:
                num_retained = np.sum(variable_index) / num_bootstrap * 1.4
            elif speed == 5:
                num_retained = np.sum(variable_index) / num_bootstrap * 1.5

            num_retained    = int(num_retained);
            Num_retained[j] = num_retained;
            if num_retained <= 1:
                break
        
        # Save the results
        self.save_results()
        
        # Best model selection
        if self.task == 'regression':
                sorted_indices = np.argsort(errors)
        elif self.task == 'classification':
            sorted_indices = np.argsort(errors)[::-1]
        best_index = sorted_indices[0]
        best_errors[j] = errors[best_index]

    def _populate_partial(self, ratio, run, errors, num_LTs, iteration, Vsel, coefficients, wavelength_weights, num_retained):
        
        X_train, X_test = self.X_train[:, Vsel], self.X_test[:, Vsel]
        
        # Compute the metrics        
        metrics = self._compute_metrics(X_train, self.y_train, X_test, self.y_test, var_selected_i=Vsel, model_type='PLS')
                
        # For regression, use the third metric (e.g. error/loss), while for classification use the first metric (e.g. accuracy)
        errors[iteration] = metrics['r2'] if self.task == 'regression' else metrics['accuracy']
                                        
        # Extract the number of components
        num_LTs[iteration] = int(self.pls.n_components)
        
        # Retrieve and normalize the model coefficients so that they sum to 1
        self.coefficients_df = pd.concat([self.coefficients_df, 
                                            pd.DataFrame({'Run': run, 'Iteration': iteration, 
                                                'Wavelengths': self.wavelengths[Vsel], 
                                                'Cofficients': coefficients[Vsel],
                                                'Weights': wavelength_weights[Vsel]})], ignore_index=True)
        
        n_selected_variables = len(Vsel)
        if n_selected_variables > self.MAX_W:
            selected_wavelengths = f"{n_selected_variables} wavelengths"
        else:
            selected_wavelengths = " - ".join([self.wavelengths[w_i] for w_i in Vsel])
            
        self.statiscs_df = pd.concat([self.statiscs_df,
                                              pd.DataFrame({'Run': run, 'Iteration': iteration, 
                                                            'Ratio': ratio, 
                                                            'Selected Variables': n_selected_variables, 
                                                            'Selected Wavelengths': selected_wavelengths,
                                                            'LV': num_LTs[iteration],
                                                            'Max Variables': num_retained,
                                                            **metrics}, index=[0])], ignore_index=True)
        
    def compute_ga_ipls(self, n_intervals=10, population_size=200, generations=100, 
                    crossover_prob=0.6, mutation_prob=0.1, MC_SAMPLES=0.8):
        """
        Perform GA-iPLS to select best spectral intervals.

        Parameters:
        - population_size: Number of candidate solutions.
        - generations: Number of generations to evolve.
        - crossover_prob: Probability of crossover.
        - mutation_prob: Probability of mutation.
        - n_intervals: Number of intervals for dividing spectral data.
        - MC_SAMPLES: Fraction of samples used for cross-validation.
        """

        # Initialize directories for results
        self.set_data_df()
        
        _, n_features = self.X.shape
        interval_length = n_features // n_intervals

        # Helper function to decode chromosome to feature indices
        def decode_intervals(chromosome):
            intervals = []
            for i, gene in enumerate(chromosome):
                if gene:
                    start = i * interval_length
                    end = (i+1)*interval_length if i < n_intervals - 1 else n_features
                    intervals.extend(range(start, end))
            return np.array(intervals)

        # Initialize population
        print((population_size, n_intervals))
        population = np.random.randint(0, 2, (population_size, n_intervals))

        best_score_global = -np.inf
        best_chromosome_global = None

        for generation in range(generations):
            scores = np.zeros(population_size)
            n_LTs  = np.zeros(population_size)
            variable_index = np.zeros((n_features, population_size))
            
            for i in tqdm(range(population_size), desc=f"Evaluating generation {generation}/{generations}"):
                Vsel = decode_intervals(population[i])
                variable_index[Vsel, i] = 1
                # print(len(Vsel), np.sum(variable_index[Vsel, i]))
                
                if len(Vsel) < 2:
                    scores[generation] = -np.inf
                    continue

                # Compute the model using the selected variables
                self._cross_predict(self.X_train[:,Vsel], self.y_train, model_type='PLS')
                
                # Extract the model coefficients and normalize them
                # Create a coefficient array corresponding to all variables; assign normalized coefficients to the selected ones
                b, coefficients = self._compute_wavelength_weights(Vsel)
                
                # Compute performance metrics for the model
                self._populate_partial(0, generation, scores, n_LTs, i, Vsel, coefficients, b, 0)
            
            self._save_partial(generation)
            
            # Select best individuals
            sorted_indices = self._compute_sorted_index(scores, variable_index)
            print(f'Best index: {sorted_indices[0]}, score = {scores[sorted_indices[0]]:.4f}, selected variables: {np.sum(variable_index[:, sorted_indices[0]])}')
            best_score_gen = scores[sorted_indices[0]]
            best_chromosome_gen = population[sorted_indices[0]]

            if best_score_gen > best_score_global:
                best_score_global = best_score_gen
                best_chromosome_global = best_chromosome_gen.copy()

            # Elitism: keep the best individual
            new_population = [population[sorted_indices[0]]]

            while len(new_population) < population_size:
                # Selection (tournament of size 2)
                idx1, idx2 = np.random.choice(population_size, 2, replace=False)
                parent1 = population[sorted_indices[min(idx1, idx2)]]

                idx3, idx4 = np.random.choice(population_size, 2, replace=False)
                parent2 = population[sorted_indices[min(idx3, idx4)]]

                child1, child2 = parent1.copy(), parent2.copy()

                # Crossover
                if np.random.rand() < crossover_prob:
                    cross_point = np.random.randint(1, n_intervals-1)
                    child1[:cross_point], child2[:cross_point] = parent2[:cross_point], parent1[:cross_point]

                # Mutation
                for child in [child1, child2]:
                    if np.random.rand() < mutation_prob:
                        mutate_point = np.random.randint(n_intervals)
                        child[mutate_point] = 1 - child[mutate_point]

                new_population.extend([child1, child2])

            population = np.array(new_population[:population_size])

        # Finalize and save results
        # self._save_partial('GA-iPLS-final')
        # self.save_results()

        best_intervals = ''
        tot_wavelenghts = 0
        for i, gene in enumerate(best_chromosome_global):
            if gene:
                start = i * interval_length
                end = (i+1)*interval_length if i < n_intervals - 1 else n_features
                best_intervals += f'from {self.wavelengths[start]} to {self.wavelengths[end-1]} - '
                tot_wavelenghts += end - start
        best_intervals = best_intervals[:-3]
        print(f"Best GA-iPLS Interval Selection: {best_intervals}")
        print(f"Total Wavelengths: {tot_wavelenghts}")
        
        best_intervals = decode_intervals(best_chromosome_global)
        best_wavelengths = [self.wavelengths[i] for i in best_intervals]

        print(f"Best Score: {best_score_global:.4f}")

        return best_wavelengths, best_score_global

    def _compute_sorted_index(self, scores, variable_index):
        # If more index have the same value, select the one with less variables
        df = pd.DataFrame({'errors': scores, 'num_variable': np.sum(variable_index, axis=0)})
        
        if self.task == 'regression':
            order = [True, True]
        elif self.task == 'classification':
            order = [False, True]
            
        return df.sort_values(by=['errors', 'num_variable'], ascending=order).index.tolist()