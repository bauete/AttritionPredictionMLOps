# run_experiment_1.py

import argparse
import os
import pandas as pd
import json
import datetime
import numpy as np
from scipy import stats
from tabulate import tabulate
from baseline_pipeline import run_baseline_pipeline
from mlops_pipeline import run_mlops_pipeline
import subprocess

def perform_wilcoxon_test(model_a_scores, model_b_scores, model_a_name="Model A", model_b_name="Model B"):
    """
    Perform a Wilcoxon signed-rank test on paired F1 scores from two models.
    Returns a dictionary with test results and summary statistics.
    """
    if len(model_a_scores) != len(model_b_scores):
        print(f"Warning: Different number of scores for {model_a_name} ({len(model_a_scores)}) and {model_b_name} ({len(model_b_scores)})")
        # Use the minimum length
        min_len = min(len(model_a_scores), len(model_b_scores))
        model_a_scores = model_a_scores[:min_len]
        model_b_scores = model_b_scores[:min_len]
    
    if len(model_a_scores) < 5:
        print(f"Warning: Small sample size ({len(model_a_scores)}) for Wilcoxon test. Results may not be reliable.")
    
    try:
        # Perform the Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(model_a_scores, model_b_scores)
        
        # Determine which model performed better on average
        mean_a = np.mean(model_a_scores)
        mean_b = np.mean(model_b_scores)
        better_model = model_a_name if mean_a > mean_b else model_b_name
        
        result = {
            "test_type": "wilcoxon_signed_rank",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "better_model": better_model if p_value < 0.05 else None,
            "mean_score_a": float(mean_a),
            "mean_score_b": float(mean_b),
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "sample_size": len(model_a_scores),
            "fold_scores_a": [float(score) for score in model_a_scores],
            "fold_scores_b": [float(score) for score in model_b_scores]
        }
        
        return result
    
    except Exception as e:
        print(f"Error performing Wilcoxon test: {e}")
        return {
            "test_type": "wilcoxon_signed_rank",
            "error": str(e),
            "model_a_name": model_a_name,
            "model_b_name": model_b_name
        }

def extract_cv_fold_metrics(training_metrics_path):
    """
    Extract cross-validation fold metrics (F1 scores) from a training_metrics.json file.
    Returns a list of F1 scores, or an empty list if not found.
    """
    if not os.path.exists(training_metrics_path):
        print(f"Warning: Training metrics file not found at {training_metrics_path}")
        return []
        
    try:
        with open(training_metrics_path, 'r') as f:
            training_data = json.load(f)
        
        if 'cross_validation' in training_data and 'folds' in training_data['cross_validation']:
            # Extract test F1 scores from each fold
            fold_scores = []
            for fold, scores in training_data['cross_validation']['folds'].items():
                if 'test_f1' in scores:
                    fold_scores.append(scores['test_f1'])
            return fold_scores
        return []
    except Exception as e:
        print(f"Error reading training metrics from {training_metrics_path}: {e}")
        return []

def run_experiment(ibm_dataset, kpmg_dataset, dirty_dataset, output_dir):
    """
    Run the entire experiment comparing baseline and MLOps pipelines on the IBM dataset.
    Saves all results and statistical comparisons to disk.
    """
    # Add timestamp to results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"run_{timestamp}")
    
    results = {
        "experiment_date": datetime.datetime.now().isoformat(),
        "timestamp": timestamp,
        "datasets": {
            "ibm": ibm_dataset,
            # "kpmg": kpmg_dataset, 
            # "dirty": dirty_dataset 
        },
        "baseline": {},
        "mlops": {},
        "comparison": {}
    }
    
    # Create output directories for both pipelines
    baseline_dir = f"{output_dir}/baseline"
    mlops_dir = f"{output_dir}/mlops"
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(mlops_dir, exist_ok=True)
    
    print(f"\n===== EXPERIMENT 1 STARTED AT {timestamp} =====")
    print(f"Results will be saved to: {output_dir}")
    
    # D1 & T1: Train both pipelines on IBM dataset
    print("\n===== STAGE D1 & T1: TRAINING ON IBM DATASET =====")
    
    print("\nRunning baseline pipeline on IBM dataset...")
    baseline_ibm_results = run_baseline_pipeline("ibm", f"{baseline_dir}/ibm")
    
    print("\nRunning MLOps pipeline on IBM dataset...")
    mlops_ibm_results = run_mlops_pipeline("ibm", f"{mlops_dir}/ibm")
    
    # M1: Evaluation of initial performance
    print("\n===== STAGE M1: EVALUATING INITIAL PERFORMANCE =====")
    
    results["baseline"]["ibm"] = baseline_ibm_results
    results["mlops"]["ibm"] = mlops_ibm_results

    # Helper function to safely get point_estimate from results
    def get_point_estimate(results_dict, metric_key):
        if not results_dict:
            return float('nan')
        metric_data = results_dict.get(metric_key, {})
        if not isinstance(metric_data, dict):
            return float('nan')
        point_estimate = metric_data.get("point_estimate")
        return float(point_estimate) if point_estimate is not None else float('nan')

    # Extract main metrics for both pipelines
    mlops_ibm_acc = get_point_estimate(mlops_ibm_results, "accuracy")
    baseline_ibm_acc = get_point_estimate(baseline_ibm_results, "accuracy")
    mlops_ibm_f1 = get_point_estimate(mlops_ibm_results, "f1_score")
    baseline_ibm_f1 = get_point_estimate(baseline_ibm_results, "f1_score")
    mlops_ibm_roc = get_point_estimate(mlops_ibm_results, "auc")
    baseline_ibm_roc = get_point_estimate(baseline_ibm_results, "auc")
    
    results["comparison"]["ibm"] = {
        "accuracy_diff": mlops_ibm_acc - baseline_ibm_acc,
        "f1_diff": mlops_ibm_f1 - baseline_ibm_f1,
        "roc_auc_diff": mlops_ibm_roc - baseline_ibm_roc
    }
    
    
    # Save experiment results
    with open(f"{output_dir}/experiment_1.json", "w") as f:
        # Convert NumPy types to Python native types
        json_safe_results = convert_numpy_types(results)
        json.dump(json_safe_results, f, indent=2)

    print("\n===== IBM DATASET EXPERIMENT COMPLETED =====")
    print(f"Results saved to {output_dir}/experiment_1.json") 

    # Print summary of main metrics
    print("\nSummary of results:")
    print(f"IBM Dataset - Baseline F1: {baseline_ibm_f1:.4f}, MLOps F1: {mlops_ibm_f1:.4f}")
    print(f"IBM Dataset - Baseline Accuracy: {baseline_ibm_acc:.4f}, MLOps Accuracy: {mlops_ibm_acc:.4f}")
    print(f"IBM Dataset - Baseline AUC: {baseline_ibm_roc:.4f}, MLOps AUC: {mlops_ibm_roc:.4f}")
    
    # Add statistical testing section using Wilcoxon test directly in the experiment
    print("\n===== RUNNING STATISTICAL SIGNIFICANCE TESTING =====")
    baseline_training_path = f"{baseline_dir}/ibm/artifacts/training_metrics.json"
    mlops_training_path = f"{mlops_dir}/ibm/artifacts/training_metrics.json"
    
    # Extract F1 scores from CV folds for both models
    baseline_f1_scores = extract_cv_fold_metrics(baseline_training_path)
    mlops_f1_scores = extract_cv_fold_metrics(mlops_training_path)
    
    # If both models have CV fold scores, perform statistical test
    if baseline_f1_scores and mlops_f1_scores:
        print(f"\nFound cross-validation fold metrics for both models:")
        print(f"  MLOps Pipeline: {len(mlops_f1_scores)} folds")
        print(f"  Baseline Pipeline: {len(baseline_f1_scores)} folds")
        
        # Create fold-by-fold comparison table
        fold_table = [["Fold", "MLOps Pipeline", "Baseline Pipeline", "Difference"]]
        for i, (mlops_score, baseline_score) in enumerate(zip(mlops_f1_scores, baseline_f1_scores)):
            fold_table.append([
                f"Fold {i+1}",
                f"{mlops_score:.4f}",
                f"{baseline_score:.4f}",
                f"{mlops_score - baseline_score:.4f}"
            ])
            
        # Add summary row
        fold_table.append([
            "Mean",
            f"{np.mean(mlops_f1_scores):.4f}",
            f"{np.mean(baseline_f1_scores):.4f}",
            f"{np.mean(mlops_f1_scores) - np.mean(baseline_f1_scores):.4f}"
        ])
        fold_table.append([
            "Std Dev",
            f"{np.std(mlops_f1_scores):.4f}",
            f"{np.std(baseline_f1_scores):.4f}",
            ""
        ])
        
        print("\nF1 Scores By Cross-Validation Fold:")
        print(tabulate(fold_table, headers="firstrow", tablefmt="grid"))
        
        # Perform Wilcoxon test
        wilcoxon_results = perform_wilcoxon_test(
            mlops_f1_scores, 
            baseline_f1_scores,
            "MLOps Pipeline", 
            "Baseline Pipeline"
        )
        
        # Display results
        print("\nWilcoxon Signed-Rank Test Results:")
        print(f"  Test Statistic: {wilcoxon_results['statistic']:.4f}")
        print(f"  p-value: {wilcoxon_results['p_value']:.4f}")
        
        if wilcoxon_results['p_value'] < 0.05:
            print(f"\n  Result: STATISTICALLY SIGNIFICANT DIFFERENCE (p < 0.05)")
            print(f"  Better Model: {wilcoxon_results['better_model']}")
        else:
            print(f"\n  Result: NO STATISTICALLY SIGNIFICANT DIFFERENCE (p ≥ 0.05)")
        
        # Add results to the output dictionary
        results["statistical_comparison"] = wilcoxon_results
    else:
        print("Insufficient data for statistical comparison. Cross-validation fold metrics not found.")
    
    # Save experiment results
    with open(f"{output_dir}/experiment_1.json", "w") as f:
        json_safe_results = convert_numpy_types(results)
        json.dump(json_safe_results, f, indent=2)
    
    print("\n===== IBM DATASET EXPERIMENT COMPLETED =====")
    print(f"Results saved to {output_dir}/experiment_1.json")

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

if __name__ == "__main__":
    # Parse command-line arguments for dataset paths and output directory
    parser = argparse.ArgumentParser(description="Run the experiment comparing baseline and MLOps pipelines")
    parser.add_argument("--ibm_dataset", default="data/ibm_dataset.csv", help="Path to IBM dataset")
    parser.add_argument("--kpmg_dataset", default="data/kpmg_dataset.csv", help="Path to KPMG dataset (not used in this version)")
    parser.add_argument("--dirty_dataset", default="data/ibm_dataset_dirty.csv", help="Path to dirty dataset (not used in this version)")
    parser.add_argument("--output_dir", default="results/experiment_1", help="Directory to save experiment results")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_experiment(args.ibm_dataset, args.kpmg_dataset, args.dirty_dataset, args.output_dir)