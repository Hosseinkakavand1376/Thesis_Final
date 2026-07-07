"""
Permutation Testing for Model Validation

Implements Nicola Dilillo's permutation test methodology to validate
that model performance is statistically significant and not due to chance.

Based on: Section 4.3 of Dilillo et al. (2025)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class PermutationTest:
    """
    Permutation test for classifier validation
    
    Tests the null hypothesis that the model has no predictive power
    by comparing performance on actual labels vs randomly shuffled labels.
    """
    
    def __init__(self, n_permutations=1000, random_state=42):
        """
        Initialize permutation test
        
        Args:
            n_permutations: Number of random permutations (default 1000)
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.results = None
    
    def run_test(self, model, X_train, y_train, X_test, y_test, 
                 metrics=['accuracy', 'precision', 'recall', 'f1']):
        """
        Run permutation test on a trained model
        
        Args:
            model: Fitted classifier with .predict() method
            X_train: Training features
            y_train: Training labels (will be permuted)
            X_test: Test features
            y_test: Test labels
            metrics: List of metrics to calculate
        
        Returns:
            Dictionary with p-values for each metric
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        print("\n" + "=" * 70)
        print("Permutation Test - Statistical Validation")
        print("=" * 70)
        print(f"Permutations: {self.n_permutations}")
        print(f"Training samples: {len(y_train)}")
        print(f"Test samples: {len(y_test)}")
        print("=" * 70 + "\n")
        
        # Step 1: Get baseline performance on ACTUAL labels
        y_pred = model.predict(X_test)
        
        baseline_scores = {}
        if 'accuracy' in metrics:
            baseline_scores['accuracy'] = accuracy_score(y_test, y_pred)
        if 'precision' in metrics:
            baseline_scores['precision'] = precision_score(
                y_test, y_pred, average='weighted', zero_division=0
            )
        if 'recall' in metrics:
            baseline_scores['recall'] = recall_score(
                y_test, y_pred, average='weighted', zero_division=0
            )
        if 'f1' in metrics:
            baseline_scores['f1'] = f1_score(
                y_test, y_pred, average='weighted', zero_division=0
            )
        
        print("Baseline performance (actual labels):")
        for metric, score in baseline_scores.items():
            print(f"  {metric}: {score:.4f}")
        print()
        
        # Step 2: Run permutations
        permuted_scores = {metric: [] for metric in metrics}
        
        for i in tqdm(range(self.n_permutations), desc="Permutations"):
            # Shuffle training labels
            y_train_shuffled = np.random.permutation(y_train)
            
            # Retrain model on shuffled labels
            try:
                # Clone model and retrain
                from sklearn.base import clone
                model_perm = clone(model)
                model_perm.fit(X_train, y_train_shuffled)
                
                # Predict on test set (with original test labels)
                y_pred_perm = model_perm.predict(X_test)
                
                # Calculate metrics
                if 'accuracy' in metrics:
                    permuted_scores['accuracy'].append(
                        accuracy_score(y_test, y_pred_perm)
                    )
                if 'precision' in metrics:
                    permuted_scores['precision'].append(
                        precision_score(y_test, y_pred_perm, average='weighted', zero_division=0)
                    )
                if 'recall' in metrics:
                    permuted_scores['recall'].append(
                        recall_score(y_test, y_pred_perm, average='weighted', zero_division=0)
                    )
                if 'f1' in metrics:
                    permuted_scores['f1'].append(
                        f1_score(y_test, y_pred_perm, average='weighted', zero_division=0)
                    )
            except Exception as e:
                print(f"\nWarning: Permutation {i} failed: {e}")
                # Use NaN for failed permutations
                for metric in metrics:
                    permuted_scores[metric].append(np.nan)
        
        # Step 3: Calculate p-values
        p_values = {}
        for metric in metrics:
            # Remove NaN values
            valid_scores = [s for s in permuted_scores[metric] if not np.isnan(s)]
            
            if len(valid_scores) == 0:
                p_values[metric] = np.nan
                continue
            
            # p-value = (# permuted scores >= baseline + 1) / (n_permutations + 1)
            # +1 ensures p-value is never exactly 0
            n_greater_or_equal = sum(s >= baseline_scores[metric] for s in valid_scores)
            p_values[metric] = (n_greater_or_equal + 1) / (len(valid_scores) + 1)
        
        # Step 4: Report results
        print("\n" + "=" * 70)
        print("Permutation Test Results")
        print("=" * 70)
        
        for metric in metrics:
            baseline = baseline_scores[metric]
            p_val = p_values[metric]
            
            if np.isnan(p_val):
                sig = "ERROR"
            elif p_val < 0.001:
                sig = "*** (p < 0.001)"
            elif p_val < 0.01:
                sig = "** (p < 0.01)"
            elif p_val < 0.05:
                sig = "* (p < 0.05)"
            else:
                sig = "NOT SIGNIFICANT"
            
            print(f"\n{metric.upper()}:")
            print(f"  Baseline: {baseline:.4f}")
            print(f"  p-value: {p_val:.4f}")
            print(f"  Significance: {sig}")
        
        print("\n" + "=" * 70)
        
        # Store results
        self.results = {
            'baseline_scores': baseline_scores,
            'permuted_scores': permuted_scores,
            'p_values': p_values,
            'n_permutations': self.n_permutations
        }
        
        return p_values
    
    def plot_permutation_distribution(self, output_path=None, metric='accuracy'):
        """
        Plot permutation test distribution
        
        Args:
            output_path: Path to save plot (optional)
            metric: Metric to plot (default 'accuracy')
        """
        if self.results is None:
            print(" No results to plot. Run test first.")
            return
        
        baseline = self.results['baseline_scores'][metric]
        permuted = self.results['permuted_scores'][metric]
        p_value = self.results['p_values'][metric]
        
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of permuted scores
        plt.hist(permuted, bins=50, alpha=0.7, color='gray', edgecolor='black', label='Permuted')
        
        # Plot baseline
        plt.axvline(baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline:.4f})')
        
        # Add p-value text
        plt.text(0.05, 0.95, f'p-value = {p_value:.4f}',
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xlabel(metric.capitalize(), fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Permutation Test Distribution - {metric.capitalize()}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {output_path}")
        
        plt.close()
    
    def save_results(self, output_path):
        """
        Save permutation test results to CSV
        
        Args:
            output_path: Path to save results
        """
        if self.results is None:
            print("No results to save. Run test first.")
            return
        
        # Create DataFrame
        results_data = []
        for metric in self.results['baseline_scores'].keys():
            results_data.append({
                'Metric': metric,
                'Baseline_Score': self.results['baseline_scores'][metric],
                'P_Value': self.results['p_values'][metric],
                'N_Permutations': self.n_permutations,
                'Significant': 'Yes' if self.results['p_values'][metric] < 0.05 else 'No'
            })
        
        df = pd.DataFrame(results_data)
        df.to_csv(output_path, index=False)
        print(f"✓ Results saved to: {output_path}")


# Example usage
if __name__ == '__main__':
    """Test permutation testing on synthetic data"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    print("\n" + "=" * 70)
    print("Testing Permutation Test Module")
    print("=" * 70)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=500, n_features=50, n_informative=30,
        n_classes=5, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train a classifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Run permutation test
    perm_test = PermutationTest(n_permutations=100, random_state=42)
    p_values = perm_test.run_test(
        model, X_train, y_train, X_test, y_test,
        metrics=['accuracy', 'f1']
    )
    
    # Plot distribution
    perm_test.plot_permutation_distribution(
        output_path='permutation_test_example.png',
        metric='accuracy'
    )
    
    # Save results
    perm_test.save_results('permutation_test_results.csv')
    
    print("\n" + "=" * 70)
    print("Permutation test module validation complete")
    print("=" * 70)
