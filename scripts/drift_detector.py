# drift_detector.py

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import sys
import argparse

# Add parent directory to path to import utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_data, save_data

def detect_numerical_drift(reference_data, current_data, column, threshold=0.05):
    """
    Detect drift in numerical columns using Kolmogorov-Smirnov test.
    Returns a dictionary with test results.
    """
    # Remove missing values
    ref_values = reference_data[column].dropna()
    cur_values = current_data[column].dropna()
    # Skip test if not enough data
    if len(ref_values) < 10 or len(cur_values) < 10:
        return {
            'column': column,
            'drift_detected': False,
            'p_value': None,
            'test': 'Not enough data for KS test'
        }
    # Perform KS test
    ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
    return {
        'column': column,
        'drift_detected': bool(p_value < threshold),
        'p_value': float(p_value),
        'test': 'Kolmogorov-Smirnov'
    }

def detect_categorical_drift(reference_data, current_data, column, threshold=0.05):
    """
    Detect drift in categorical columns using Chi-square test.
    Returns a dictionary with test results.
    """
    # Get normalized value counts for both datasets
    ref_counts = reference_data[column].value_counts(normalize=True)
    cur_counts = current_data[column].value_counts(normalize=True)
    # Combine all categories present in either dataset
    all_categories = set(ref_counts.index) | set(cur_counts.index)
    # Build arrays for observed frequencies
    ref_obs = np.array([ref_counts.get(cat, 0) for cat in all_categories])
    cur_obs = np.array([cur_counts.get(cat, 0) for cat in all_categories])
    # Skip test if not enough categories
    if len(all_categories) < 2:
        return {
            'column': column,
            'drift_detected': False,
            'p_value': None,
            'test': 'Not enough categories for Chi-square test'
        }
    # Perform Chi-square test
    try:
        chi2_stat, p_value = stats.chisquare(cur_obs, ref_obs)
        return {
            'column': column,
            'drift_detected': bool(p_value < threshold),
            'p_value': float(p_value),
            'test': 'Chi-square'
        }
    except:
        return {
            'column': column,
            'drift_detected': False,
            'p_value': None,
            'test': 'Chi-square test failed'
        }

def detect_data_drift(reference_data_path, current_data_path, output_dir=None, threshold=0.05):
    """
    Detect data drift between two datasets.
    Runs drift tests for all columns present in both datasets.
    Saves a JSON report if output_dir is provided.
    Returns a dictionary with drift results.
    """
    print(f"Detecting drift between {reference_data_path} and {current_data_path}")
    reference_data = load_data(reference_data_path)
    current_data = load_data(current_data_path)
    # Warn if columns are missing in current data
    missing_columns = set(reference_data.columns) - set(current_data.columns)
    if missing_columns:
        print(f"Warning: Current data is missing columns: {missing_columns}")
    # Only test columns present in both datasets
    common_columns = set(reference_data.columns) & set(current_data.columns)
    results = {
        'reference_data': reference_data_path,
        'current_data': current_data_path,
        'drift_detected': False,
        'threshold': threshold,
        'column_results': []
    }
    # Test each column for drift
    for column in common_columns:
        # Skip columns with only one unique value in either dataset
        if reference_data[column].nunique() <= 1 or current_data[column].nunique() <= 1:
            continue
        # Choose test based on column type
        if pd.api.types.is_numeric_dtype(reference_data[column]):
            test_result = detect_numerical_drift(reference_data, current_data, column, threshold)
        else:
            test_result = detect_categorical_drift(reference_data, current_data, column, threshold)
        results['column_results'].append(test_result)
        # Mark drift if detected in any column
        if test_result['drift_detected']:
            results['drift_detected'] = True
    # Ensure drift_detected is a bool
    results['drift_detected'] = bool(results['drift_detected'])
    # Save results to file if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/drift_report.json", "w") as f:
            json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Detect data drift between two datasets")
    parser.add_argument("--reference_data", required=True, help="Path to reference dataset")
    parser.add_argument("--current_data", required=True, help="Path to current dataset")
    parser.add_argument("--output_dir", help="Directory to save drift report")
    parser.add_argument("--threshold", type=float, default=0.05, help="p-value threshold for significance")
    args = parser.parse_args()
    # Run drift detection
    results = detect_data_drift(args.reference_data, args.current_data, args.output_dir, args.threshold)
    # Print summary to console
    print(f"\nDrift detected: {results['drift_detected']}")
    if results['drift_detected']:
        print("Columns with drift:")
        for result in results['column_results']:
            if result['drift_detected']:
                print(f"  - {result['column']} (p-value: {result['p_value']:.6f}, test: {result['test']})")