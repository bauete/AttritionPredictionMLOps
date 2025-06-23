"""
Experiment 1: Comparing Baseline and MLOps Pipelines on Generalizability

This script runs Experiment 2, which processes the KPMG dataset using the trained models and feature selectors from Experiment 1.
"""

import argparse
import os
import pandas as pd
import json
import datetime
import numpy as np
from baseline_pipeline import run_baseline_pipeline
from mlops_pipeline import run_mlops_pipeline

def run_experiment(ibm_dataset, kpmg_dataset, dirty_dataset, output_dir_exp2):
    """
    Args:
        ibm_dataset: Path to IBM dataset (used for MLOps reference_data_path).
        kpmg_dataset: Path to KPMG dataset (this will be processed).
        dirty_dataset: Path to dirty dataset (unused in this version of the script).
        output_dir_exp2: Root directory for experiment 2 results (e.g., 'results/experiment_2').
    """
    # Add timestamp to results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_exp2 = os.path.join(output_dir_exp2, f"run_{timestamp}")
    
    # Prepare results dictionary to collect all outputs and metadata
    results = {
        "experiment_date": datetime.datetime.now().isoformat(),
        "timestamp": timestamp,
        "experiment_focus": "KPMG Dataset Processing (Experiment 2)",
        "datasets_provided": {
            "ibm_reference_dataset": ibm_dataset,
            "kpmg": kpmg_dataset,
            "dirty": dirty_dataset 
        },
        "kpmg_processing_results": {
            "baseline": {},
            "mlops": {},
            "comparison": {}
        }
    }
    
    # Define locations of IBM artifacts from Experiment 1 (used as reference for this experiment)
    experiment_1_artifacts_root = "results/experiment_1/latest_artifacts"
    ibm_baseline_artifacts_dir = os.path.join(experiment_1_artifacts_root, "baseline", "ibm")
    ibm_mlops_artifacts_dir = os.path.join(experiment_1_artifacts_root, "mlops", "ibm")

    # Define output directories for this experiment's KPMG runs
    kpmg_baseline_output_dir = os.path.join(output_dir_exp2, "baseline", "kpmg")
    kpmg_mlops_output_dir = os.path.join(output_dir_exp2, "mlops", "kpmg")
    
    os.makedirs(kpmg_baseline_output_dir, exist_ok=True)
    os.makedirs(kpmg_mlops_output_dir, exist_ok=True)

    print(f"Expecting IBM artifacts from Experiment 1 in:")
    print(f"  Baseline: {ibm_baseline_artifacts_dir}")
    print(f"  MLOps: {ibm_mlops_artifacts_dir}")
    print(f"Outputs for this KPMG run (Experiment 2) will be in: {output_dir_exp2}")

    if not os.path.exists(ibm_baseline_artifacts_dir) or not os.path.exists(ibm_mlops_artifacts_dir):
        print(f"Warning: Reference IBM artifact directories from {experiment_1_artifacts_root} not found. This experiment stage assumes they exist from prior runs.")

    # -------------------------------
    # D2 & T2: Run pipelines on KPMG dataset
    # -------------------------------
    print("\n===== STAGE D2 & T2: TESTING ON KPMG DATASET =====")
    
    print("\nRunning baseline pipeline on KPMG dataset...")
    # Baseline pipeline uses IBM's trained model and feature selector from Experiment 1.
    baseline_kpmg_results = run_baseline_pipeline(
        source="kpmg",
        output_dir=kpmg_baseline_output_dir, 
        evaluate_only=True,
        reference_artifacts_dir=ibm_baseline_artifacts_dir
    )

    print("\nRunning MLOps pipeline on KPMG dataset...")
    # MLOps pipeline may retrain if drift is detected, using IBM's feature selector for consistency.
    # reference_data_for_drift is used for drift detection (should be the raw acquired IBM data from Exp1 MLOps run).
    reference_data_for_drift = os.path.join(ibm_mlops_artifacts_dir, "01_acquired_data.csv")
    if not os.path.exists(reference_data_for_drift):
        print(f"Warning: Reference data for drift detection not found at {reference_data_for_drift}. Using provided ibm_dataset path: {ibm_dataset} instead.")
        reference_data_for_drift = ibm_dataset

    mlops_kpmg_results = run_mlops_pipeline(
        source="kpmg",
        output_dir=kpmg_mlops_output_dir, 
        reference_data_path=reference_data_for_drift, 
        detect_drift=True,
        reference_artifacts_dir=ibm_mlops_artifacts_dir
    )
    
    # -------------------------------
    # M2: Evaluate and log performance for KPMG dataset
    # -------------------------------
    print("\n===== STAGE M2: EVALUATING KPMG PERFORMANCE =====")
    results["kpmg_processing_results"]["baseline"] = baseline_kpmg_results if baseline_kpmg_results else {}
    results["kpmg_processing_results"]["mlops"] = mlops_kpmg_results if mlops_kpmg_results else {}
    
    # -------------------------------
    # C2: Metrics comparison for KPMG dataset
    # -------------------------------
    print("\n===== STAGE C2: KPMG METRICS COMPARISON =====")
    baseline_res_safe = baseline_kpmg_results or {}
    mlops_res_safe = mlops_kpmg_results or {}

    # Helper function to safely get point_estimate from results
    def get_point_estimate(results_dict, metric_key):
        if not results_dict or results_dict.get("error"):
            return float('nan')
        metric_data = results_dict.get(metric_key, {})
        if not isinstance(metric_data, dict):
            return float('nan')
        point_estimate = metric_data.get("point_estimate")
        return float(point_estimate) if point_estimate is not None else float('nan')

    # Extract main metrics for both pipelines
    mlops_kpmg_acc = get_point_estimate(mlops_res_safe, "accuracy")
    baseline_kpmg_acc = get_point_estimate(baseline_res_safe, "accuracy")
    mlops_kpmg_f1 = get_point_estimate(mlops_res_safe, "f1_score")
    baseline_kpmg_f1 = get_point_estimate(baseline_res_safe, "f1_score")
    mlops_kpmg_roc = get_point_estimate(mlops_res_safe, "auc")
    baseline_kpmg_roc = get_point_estimate(baseline_res_safe, "auc")

    # Extract fairness metrics (demographic parity and equalized odds for gender and age)
    mlops_kpmg_dpd_gender = mlops_res_safe.get("fairness_metrics", {}).get("demographic_parity_difference_gender", float('nan'))
    baseline_kpmg_dpd_gender = baseline_res_safe.get("fairness_metrics", {}).get("demographic_parity_difference_gender", float('nan'))
    mlops_kpmg_eod_gender = mlops_res_safe.get("fairness_metrics", {}).get("equalized_odds_difference_gender", float('nan'))
    baseline_kpmg_eod_gender = baseline_res_safe.get("fairness_metrics", {}).get("equalized_odds_difference_gender", float('nan'))
    mlops_kpmg_dpd_age = mlops_res_safe.get("fairness_metrics", {}).get("demographic_parity_difference_age_binned", float('nan'))
    baseline_kpmg_dpd_age = baseline_res_safe.get("fairness_metrics", {}).get("demographic_parity_difference_age_binned", float('nan'))
    mlops_kpmg_eod_age = mlops_res_safe.get("fairness_metrics", {}).get("equalized_odds_difference_age_binned", float('nan'))
    baseline_kpmg_eod_age = baseline_res_safe.get("fairness_metrics", {}).get("equalized_odds_difference_age_binned", float('nan'))

    # Store metric differences in results
    results["kpmg_processing_results"]["comparison"] = {
        "accuracy_diff": mlops_kpmg_acc - baseline_kpmg_acc if not (pd.isna(mlops_kpmg_acc) or pd.isna(baseline_kpmg_acc)) else float('nan'),
        "f1_diff": mlops_kpmg_f1 - baseline_kpmg_f1 if not (pd.isna(mlops_kpmg_f1) or pd.isna(baseline_kpmg_f1)) else float('nan'),
        "roc_auc_diff": mlops_kpmg_roc - baseline_kpmg_roc if not (pd.isna(mlops_kpmg_roc) or pd.isna(baseline_kpmg_roc)) else float('nan'),
        "dpd_gender_diff": mlops_kpmg_dpd_gender - baseline_kpmg_dpd_gender if not (pd.isna(mlops_kpmg_dpd_gender) or pd.isna(baseline_kpmg_dpd_gender)) else float('nan'),
        "eod_gender_diff": mlops_kpmg_eod_gender - baseline_kpmg_eod_gender if not (pd.isna(mlops_kpmg_eod_gender) or pd.isna(baseline_kpmg_eod_gender)) else float('nan'),
        "dpd_age_binned_diff": mlops_kpmg_dpd_age - baseline_kpmg_dpd_age if not (pd.isna(mlops_kpmg_dpd_age) or pd.isna(baseline_kpmg_dpd_age)) else float('nan'),
        "eod_age_binned_diff": mlops_kpmg_eod_age - baseline_kpmg_eod_age if not (pd.isna(mlops_kpmg_eod_age) or pd.isna(baseline_kpmg_eod_age)) else float('nan'),
    }
    
    # Save experiment results for the KPMG stage (within Experiment 2's output_dir)
    results_filename = os.path.join(output_dir_exp2, "experiment_2_kpmg_results.json") 
    with open(results_filename, "w") as f:
        # Use a default converter for datetime and pandas types
        json.dump(results, f, indent=2, default=lambda x: str(x) if isinstance(x, (datetime.datetime, datetime.date, pd.Timestamp)) or hasattr(x, 'item') else x)
    
    print("\n===== KPMG STAGE (EXPERIMENT 2) COMPLETED =====")
    print(f"Results for KPMG stage saved to {results_filename}")
    
    # Print summary for KPMG results
    if baseline_kpmg_results and not baseline_kpmg_results.get("error") and \
       mlops_kpmg_results and not mlops_kpmg_results.get("error"):
        print("\nSummary of KPMG results (Experiment 2):")
        print(f"  Baseline F1: {baseline_kpmg_f1:.4f}, MLOps F1: {mlops_kpmg_f1:.4f}")
        print(f"  Baseline Accuracy: {baseline_kpmg_acc:.4f}, MLOps Accuracy: {mlops_kpmg_acc:.4f}")
        print(f"  Baseline ROC AUC: {baseline_kpmg_roc:.4f}, MLOps ROC AUC: {mlops_kpmg_roc:.4f}")
        print(f"  Baseline DPD Gender: {baseline_kpmg_dpd_gender:.4f}, MLOps DPD Gender: {mlops_kpmg_dpd_gender:.4f}")
        print(f"  Baseline EOD Gender: {baseline_kpmg_eod_gender:.4f}, MLOps EOD Gender: {mlops_kpmg_eod_gender:.4f}")
        print(f"  Baseline DPD Age: {baseline_kpmg_dpd_age:.4f}, MLOps DPD Age: {mlops_kpmg_dpd_age:.4f}")
        print(f"  Baseline EOD Age: {baseline_kpmg_eod_age:.4f}, MLOps EOD Age: {mlops_kpmg_eod_age:.4f}")
    else:
        print("\nCould not generate full summary for KPMG (Experiment 2) as some results are missing or contain errors.")
        if baseline_kpmg_results and baseline_kpmg_results.get("error"):
            print(f"  Baseline error: {baseline_kpmg_results.get('error')}")
        if mlops_kpmg_results and mlops_kpmg_results.get("error"):
            print(f"  MLOps error: {mlops_kpmg_results.get('error')}")

if __name__ == "__main__":
    # Parse command-line arguments for dataset paths and output directory
    parser = argparse.ArgumentParser(description="Run experiment stage D2, T2, M2, C2 (KPMG dataset processing). Assumes prerequisite model/data from IBM stages (Experiment 1) are present.")
    parser.add_argument("--ibm_dataset", default="data/ibm_dataset.csv", help="Path to the original clean IBM dataset CSV (used as reference for MLOps drift detection).")
    parser.add_argument("--kpmg_dataset", default="data/kpmg_dataset.csv", help="Path to KPMG dataset (this dataset will be processed).")
    parser.add_argument("--dirty_dataset", default="data/ibm_dataset_dirty.csv", help="Path to dirty dataset (unused in this script version).")
    parser.add_argument("--output_dir", default="results/experiment_2", help="Root directory to save Experiment 2 results.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True) 
    
    run_experiment(args.ibm_dataset, args.kpmg_dataset, args.dirty_dataset, args.output_dir)