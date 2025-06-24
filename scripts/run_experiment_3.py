"""
Experiment 3: Comparing Baseline and MLOps Pipelines on Dirty IBM Dataset

This script runs the third stage of the experiment, focusing on the dirty version of the IBM dataset.
"""

import argparse
import os
import pandas as pd
import json
import datetime
from baseline_pipeline import run_baseline_pipeline
from mlops_pipeline import run_mlops_pipeline

# Helper function to extract the point estimate from results
def get_point_estimate(results_dict, metric_key):
    if not results_dict or results_dict.get("error"):  # Check for error key
        return float('nan')
    metric_data = results_dict.get(metric_key, {})
    if not isinstance(metric_data, dict):
        # Fallback for older structures or direct float values if any script still returns that
        if isinstance(metric_data, (int, float)):
            return float(metric_data)
        return float('nan')
    point_estimate = metric_data.get("point_estimate")
    return float(point_estimate) if point_estimate is not None else float('nan')

def run_experiment_dirty_only(ibm_raw_dataset_path_for_drift_reference, dirty_dataset_path, output_dir_exp3):
    """
    Run only the third stage of the experiment (D3, T3, M3, C3)
    comparing baseline and MLOps pipelines on the dirty IBM dataset.
    Assumes prior training on clean IBM data (Experiment 1) has been completed.

    Args:
        ibm_raw_dataset_path_for_drift_reference: Path to the original raw clean IBM dataset CSV.
        dirty_dataset_path: Path to the dirty IBM dataset (used for informational purposes).
        output_dir_exp3: Root directory for Experiment 3 results (e.g., 'results/experiment_3').
    """
    # Add timestamp to results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_exp3 = os.path.join(output_dir_exp3, f"run_{timestamp}")
    
    # Prepare results dictionary to collect all outputs and metadata
    results = {
        "experiment_date": datetime.datetime.now().isoformat(),
        "timestamp": timestamp,
        "experiment_focus": "Dirty Dataset Comparison (Experiment 3: D3, T3, M3, C3)",
        "datasets_provided": {
            "ibm_raw_reference_csv": ibm_raw_dataset_path_for_drift_reference,
            "dirty_dataset_csv": dirty_dataset_path
        },
        "baseline": {"dirty_ibm": {}},
        "mlops": {"dirty_ibm": {}},
        "comparison": {"dirty_ibm": {}}
    }
    
    # Directories for IBM artifacts (assumed to be pre-existing from Experiment 1)
    experiment_1_artifacts_root = "results/experiment_1/latest_artifacts" 
    ibm_baseline_artifacts_dir = os.path.join(experiment_1_artifacts_root, "baseline", "ibm")
    ibm_mlops_artifacts_dir = os.path.join(experiment_1_artifacts_root, "mlops", "ibm")

    # Output directories for the current dirty data processing stage (within Experiment 3's output_dir)
    dirty_baseline_output_dir = os.path.join(output_dir_exp3, "baseline", "dirty_ibm")
    dirty_mlops_output_dir = os.path.join(output_dir_exp3, "mlops", "dirty_ibm")
    
    os.makedirs(dirty_baseline_output_dir, exist_ok=True)
    os.makedirs(dirty_mlops_output_dir, exist_ok=True)
    
    print(f"\n EXPERIMENT 3 STARTED AT {timestamp}")
    print(f"Results will be saved to: {output_dir_exp3}")

    print(f"Expecting reference IBM artifacts from Experiment 1 in:")
    print(f"  Baseline: {ibm_baseline_artifacts_dir}")
    print(f"  MLOps: {ibm_mlops_artifacts_dir}")
    print(f"Outputs for this Dirty IBM run (Experiment 3) will be in: {output_dir_exp3}")

    if not os.path.exists(ibm_baseline_artifacts_dir) or not os.path.exists(ibm_mlops_artifacts_dir):
        print(f"Warning: Reference IBM artifact directories from {experiment_1_artifacts_root} not found. This experiment stage assumes they exist from prior runs.")
        # Depending on strictness, you might want to exit or raise an error here.

    # D3 & T3: Test pipelines with dirty dataset

    print("\n STAGE D3 & T3: TESTING ON DIRTY IBM DATASET")
    
    print("\nRunning baseline pipeline on dirty IBM dataset (evaluating with IBM model)...")
    # Baseline for dirty data uses IBM's trained model and feature selector from Experiment 1.
    baseline_dirty_results = run_baseline_pipeline(
        source="dirty_ibm",
        output_dir=dirty_baseline_output_dir, 
        evaluate_only=True,
        reference_artifacts_dir=ibm_baseline_artifacts_dir
    )
    
    print("\nRunning MLOps pipeline on dirty IBM dataset (checking drift against clean IBM data)...")
    # MLOps for dirty data will check drift against the clean IBM data.
    # It uses IBM's MLOps feature selector and model as a reference from Experiment 1.
    # It will retrain if drift is detected.

    # Determine the reference data for drift detection (ideally, acquired data from Exp1 MLOps IBM run)
    reference_data_for_drift = os.path.join(ibm_mlops_artifacts_dir, "01_acquired_data.csv")
    if not os.path.exists(reference_data_for_drift):
        print(f"Warning: Reference ACQUIRED data for drift detection not found at {reference_data_for_drift}.")
        print(f"Falling back to using the provided RAW IBM dataset path for drift reference: {ibm_raw_dataset_path_for_drift_reference}.")
        reference_data_for_drift = ibm_raw_dataset_path_for_drift_reference

    mlops_dirty_results = run_mlops_pipeline(
        source="dirty_ibm",
        output_dir=dirty_mlops_output_dir,
        detect_drift=True,
        reference_data_path=reference_data_for_drift,
        reference_artifacts_dir=ibm_mlops_artifacts_dir
    )
    
    
    # M3: Evaluate and log performance
    
    print("\n STAGE M3: EVALUATING DIRTY IBM DATA PERFORMANCE")
    
    results["baseline"]["dirty_ibm"] = baseline_dirty_results if baseline_dirty_results else {}
    results["mlops"]["dirty_ibm"] = mlops_dirty_results if mlops_dirty_results else {}
    
    
    # C3: Metrics comparison for Dirty Dataset
    
    print("\n STAGE C3: DIRTY IBM DATA METRICS COMPARISON")
    
    baseline_res_safe_dirty = baseline_dirty_results or {}
    mlops_res_safe_dirty = mlops_dirty_results or {}

    # Extract main metrics for both pipelines
    mlops_dirty_acc = get_point_estimate(mlops_res_safe_dirty, "accuracy")
    baseline_dirty_acc = get_point_estimate(baseline_res_safe_dirty, "accuracy")
    mlops_dirty_f1 = get_point_estimate(mlops_res_safe_dirty, "f1_score")
    baseline_dirty_f1 = get_point_estimate(baseline_res_safe_dirty, "f1_score")
    mlops_dirty_roc = get_point_estimate(mlops_res_safe_dirty, "auc")
    baseline_dirty_roc = get_point_estimate(baseline_res_safe_dirty, "auc")

    # Extract fairness metrics for gender and age
    mlops_dirty_dpd_gender = mlops_res_safe_dirty.get("fairness_metrics", {}).get("demographic_parity_difference_gender", float('nan'))
    baseline_dirty_dpd_gender = baseline_res_safe_dirty.get("fairness_metrics", {}).get("demographic_parity_difference_gender", float('nan'))
    mlops_dirty_eod_gender = mlops_res_safe_dirty.get("fairness_metrics", {}).get("equalized_odds_difference_gender", float('nan'))
    baseline_dirty_eod_gender = baseline_res_safe_dirty.get("fairness_metrics", {}).get("equalized_odds_difference_gender", float('nan'))
    mlops_dirty_dpd_age = mlops_res_safe_dirty.get("fairness_metrics", {}).get("demographic_parity_difference_age_binned", float('nan'))
    baseline_dirty_dpd_age = baseline_res_safe_dirty.get("fairness_metrics", {}).get("demographic_parity_difference_age_binned", float('nan'))
    mlops_dirty_eod_age = mlops_res_safe_dirty.get("fairness_metrics", {}).get("equalized_odds_difference_age_binned", float('nan'))
    baseline_dirty_eod_age = baseline_res_safe_dirty.get("fairness_metrics", {}).get("equalized_odds_difference_age_binned", float('nan'))

    # Store metric differences in results
    results["comparison"]["dirty_ibm"] = {
        "accuracy_diff": mlops_dirty_acc - baseline_dirty_acc if not (pd.isna(mlops_dirty_acc) or pd.isna(baseline_dirty_acc)) else float('nan'),
        "f1_diff": mlops_dirty_f1 - baseline_dirty_f1 if not (pd.isna(mlops_dirty_f1) or pd.isna(baseline_dirty_f1)) else float('nan'),
        "roc_auc_diff": mlops_dirty_roc - baseline_dirty_roc if not (pd.isna(mlops_dirty_roc) or pd.isna(baseline_dirty_roc)) else float('nan'),
        "dpd_gender_diff": mlops_dirty_dpd_gender - baseline_dirty_dpd_gender if not (pd.isna(mlops_dirty_dpd_gender) or pd.isna(baseline_dirty_dpd_gender)) else float('nan'),
        "eod_gender_diff": mlops_dirty_eod_gender - baseline_dirty_eod_gender if not (pd.isna(mlops_dirty_eod_gender) or pd.isna(baseline_dirty_eod_gender)) else float('nan'),
        "dpd_age_binned_diff": mlops_dirty_dpd_age - baseline_dirty_dpd_age if not (pd.isna(mlops_dirty_dpd_age) or pd.isna(baseline_dirty_dpd_age)) else float('nan'),
        "eod_age_binned_diff": mlops_dirty_eod_age - baseline_dirty_eod_age if not (pd.isna(mlops_dirty_eod_age) or pd.isna(baseline_dirty_eod_age)) else float('nan'),
    }
    
    # Optionally, check for drift detection and pipeline duration in logs
    drift_detections_in_run = 0
    pipeline_duration_seconds = None 
    
    drift_status_path = os.path.join(dirty_mlops_output_dir, "00_drift_status.txt")
    if os.path.exists(drift_status_path):
        with open(drift_status_path, 'r') as f:
            status = f.read().strip()
            if status == "DRIFT_DETECTED":
                drift_detections_in_run = 1
    
    log_path = os.path.join(dirty_mlops_output_dir, "pipeline_log.txt")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_lines = f.readlines()
            for line in reversed(log_lines): 
                if "Pipeline run completed in" in line:
                    try:
                        pipeline_duration_seconds = float(line.split("in ")[1].split(" seconds")[0])
                        break
                    except (IndexError, ValueError) as e:
                        print(f"Could not parse pipeline duration from log line: {line.strip()} - Error: {e}")
                        pipeline_duration_seconds = None

    results["mlops_metrics_dirty_run"] = {
        "drift_detected_in_this_run": bool(drift_detections_in_run),
        "pipeline_duration_seconds": pipeline_duration_seconds
    }
    
    # Save experiment results for the dirty dataset stage (within Experiment 3's output_dir)
    results_file_path = os.path.join(output_dir_exp3, "experiment_3_dirty_results.json")
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: str(x) if isinstance(x, (datetime.datetime, datetime.date, pd.Timestamp)) or hasattr(x, 'item') else x)
    
    print("\n EXPERIMENT 3 (DIRTY IBM DATA ONLY) COMPLETED")
    print(f"Results saved to {results_file_path}")
    
    # Print summary for Dirty IBM results
    print("\nSummary of results for Dirty IBM Dataset (Experiment 3):")
    if baseline_dirty_results and not baseline_dirty_results.get("error") and \
       mlops_dirty_results and not mlops_dirty_results.get("error"):
        print(f"  Dirty IBM - Baseline F1: {baseline_dirty_f1:.4f}, MLOps F1: {mlops_dirty_f1:.4f}")
        print(f"  Dirty IBM - Baseline Accuracy: {baseline_dirty_acc:.4f}, MLOps Accuracy: {mlops_dirty_acc:.4f}")
        print(f"  Dirty IBM - Baseline ROC AUC: {baseline_dirty_roc:.4f}, MLOps ROC AUC: {mlops_dirty_roc:.4f}")
        print(f"  Dirty IBM - Baseline DPD Gender: {baseline_dirty_dpd_gender:.4f}, MLOps DPD Gender: {mlops_dirty_dpd_gender:.4f}")
        print(f"  Dirty IBM - Baseline EOD Gender: {baseline_dirty_eod_gender:.4f}, MLOps EOD Gender: {mlops_dirty_eod_gender:.4f}")
        print(f"  Dirty IBM - Baseline DPD Age Binned: {baseline_dirty_dpd_age:.4f}, MLOps DPD Age Binned: {mlops_dirty_dpd_age:.4f}")
        print(f"  Dirty IBM - Baseline EOD Age Binned: {baseline_dirty_eod_age:.4f}, MLOps EOD Age Binned: {mlops_dirty_eod_age:.4f}")

    else:
        print("  Could not generate full summary for Dirty IBM Dataset as some results are missing or contain errors.")
        if baseline_dirty_results and baseline_dirty_results.get("error"): 
            print(f"  Baseline (Dirty IBM) error: {baseline_dirty_results.get('error')}")
        if mlops_dirty_results and mlops_dirty_results.get("error"): 
            print(f"  MLOps (Dirty IBM) error: {mlops_dirty_results.get('error')}")

    if pipeline_duration_seconds is not None:
        print(f"  MLOps pipeline duration for dirty data: {pipeline_duration_seconds:.2f} seconds")

if __name__ == "__main__":
    # Parse command-line arguments for dataset paths and output directory
    parser = argparse.ArgumentParser(description="Run Experiment 3: Dirty IBM dataset processing.")
    parser.add_argument("--ibm_raw_dataset_reference", default="data/ibm_dataset.csv", help="Path to the original raw clean IBM dataset CSV (for MLOps drift reference).")
    parser.add_argument("--dirty_dataset", default="data/ibm_dataset_dirty.csv", help="Path to the dirty IBM dataset CSV (used for informational purposes by this script). The pipeline scripts will look for 'ibm_dataset_dirty.csv' in their 'data' dir based on the 'dirty_ibm' source type.")
    parser.add_argument("--output_dir", default="results/experiment_3", help="Root directory to save Experiment 3 results.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_experiment_dirty_only(
        ibm_raw_dataset_path_for_drift_reference=args.ibm_raw_dataset_reference,
        dirty_dataset_path=args.dirty_dataset,
        output_dir_exp3=args.output_dir
    )