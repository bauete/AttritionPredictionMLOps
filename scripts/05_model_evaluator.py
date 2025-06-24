"""
Step 5: Model Evaluation

This script evaluates a trained XGBoost model on a hold-out test set, computes point estimates, fairness metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
import argparse
import json 
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


# 0. Argument Parsing 

# Parse command-line arguments for input data, model, and output directory.
parser = argparse.ArgumentParser(description="Evaluate a trained XGBoost model.")
parser.add_argument("--input_file",type=str,required=True,help="Path to the engineered CSV data file (full dataset).")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.pkl file.")
parser.add_argument("--output_dir",type=str,required=True, help="Directory to save the evaluation_summary.json.")
args = parser.parse_args()


# 1. Load data, model, and scaler

print(f"Loading data from: {args.input_file}")
df = pd.read_csv(args.input_file)

# Define target and features
target_column = 'Attrition'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in the input data.")

y = df[target_column]
# Drop target and any potential ID or index columns
columns_to_drop_from_X = [target_column]
if 'EmployeeNumber' in df.columns:
    columns_to_drop_from_X.append('EmployeeNumber')
if 'Unnamed: 0' in df.columns:
    columns_to_drop_from_X.append('Unnamed: 0')

X = df.drop(columns=columns_to_drop_from_X, errors='ignore')

# Split data to isolate the hold-out test set (10% of data)
print("Splitting data into train/test to isolate the hold-out test set...")
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42  # Must match trainer
)
print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")

# Load scaler
# The scaler is expected to be in the same directory as the model.
scaler_path = os.path.join(os.path.dirname(args.model_path), "scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Ensure it was saved by the trainer.")
print(f"Loading scaler from: {scaler_path}")
scaler = joblib.load(scaler_path)

# List of numeric columns to scale
numeric_cols = ['Age', 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'PerformanceRating', 'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'StagnationScore', 'TenureRatio', 'PCA1', 'PCA2', 'PCA3']
if len(numeric_cols) > 0:
    print(f"Applying scaling to numeric columns: {numeric_cols}")
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
else:
    print("No numeric columns found in X_test for scaling.")
    X_test_scaled = X_test.copy()

# Load trained model
print(f"Loading model from: {args.model_path}")
model = joblib.load(args.model_path)


# 2. Compute point estimates

print("Computing point estimates...")
# Predict probabilities and binary labels
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Convert to numpy arrays for saving
y_pred_np = np.array(y_pred)
y_true_np = np.array(y_test)

# Define output paths based on the script's output directory
y_pred_path = os.path.join(args.output_dir, "y_pred.npy")
y_true_path = os.path.join(args.output_dir, "y_true.npy")

# Save the arrays for later use or analysis
np.save(y_pred_path, y_pred_np)
np.save(y_true_path, y_true_np)
print(f"Saved predictions to {y_pred_path}")
print(f"Saved true labels to {y_true_path}")

# Compute standard classification metrics
auc_point = roc_auc_score(y_test, y_pred_proba)
acc_point = accuracy_score(y_test, y_pred)
precision_point = precision_score(y_test, y_pred, zero_division=0)
recall_point = recall_score(y_test, y_pred, zero_division=0)
f1_point = f1_score(y_test, y_pred, zero_division=0, average="weighted")

print(f"Point estimates on the hold-out test set:")
print(f"  AUC:       {auc_point:.4f}")
print(f"  Accuracy:  {acc_point:.4f}")
print(f"  Precision: {precision_point:.4f}")
print(f"  Recall:    {recall_point:.4f}")
print(f"  F1‐Score:  {f1_point:.4f}")


# 2.5 Compute Fairness Metrics

print("\nComputing fairness metrics...")
fairness_metrics_results = {}

# Compute fairness metrics for Gender (if available)
if 'Gender_Female' in X_test.columns:
    sf_gender = X_test['Gender_Female']
    # Compute demographic parity and equalized odds differences
    dpd_gender = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_gender)
    eod_gender = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_gender)
    fairness_metrics_results["demographic_parity_difference_gender"] = dpd_gender
    fairness_metrics_results["equalized_odds_difference_gender"] = eod_gender
    print(f"  Demographic Parity Difference (Gender): {dpd_gender:.4f}")
    print(f"  Equalized Odds Difference (Gender):     {eod_gender:.4f}")
else:
    print("  Warning: 'Gender' column not found in X_test. Skipping Gender fairness metrics.")
    fairness_metrics_results["demographic_parity_difference_gender"] = None
    fairness_metrics_results["equalized_odds_difference_gender"] = None

# Compute fairness metrics for Age (binned into groups)
if 'Age' in X_test.columns:
    sf_age_original = X_test['Age']
    # Define bins and labels for age groups
    age_bins = [0, 29, 39, 49, sf_age_original.max() + 1] 
    age_labels = ['<30', '30-39', '40-49', '50+']
    sf_age_binned = pd.cut(sf_age_original, bins=age_bins, labels=age_labels, right=False)
    
    dpd_age = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_age_binned)
    eod_age = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_age_binned)
    fairness_metrics_results["demographic_parity_difference_age_binned"] = dpd_age
    fairness_metrics_results["equalized_odds_difference_age_binned"] = eod_age
    print(f"  Demographic Parity Difference (Age Binned): {dpd_age:.4f}")
    print(f"  Equalized Odds Difference (Age Binned):     {eod_age:.4f}")
else:
    print("  Warning: 'Age' column not found in X_test. Skipping Age fairness metrics.")
    fairness_metrics_results["demographic_parity_difference_age_binned"] = None
    fairness_metrics_results["equalized_odds_difference_age_binned"] = None


# 3. Compute 95% bootstrap confidence intervals

def bootstrap_metric(metric_func, y_true, y_probs, y_preds, n_boot=1000, alpha=0.95):
    """
    Compute the bootstrap confidence interval for a given metric function.
    - metric_func: function(y_true_boot, y_pred_or_proba_boot) -> float
    - y_true, y_probs, y_preds: full test set arrays
    - n_boot: number of bootstrap samples
    - alpha: confidence level
    Returns (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(42) # Ensure reproducibility for bootstrap
    stats = []

    n_samples = len(y_true)
    for _ in range(n_boot):
        idxs = rng.randint(0, n_samples, n_samples)
        y_true_boot = y_true.iloc[idxs] # Use .iloc for Series
        # If metric_func expects probabilities, pass y_probs; if expects labels, pass y_preds
        if metric_func.__name__ in ["roc_auc_score"]:
            y_prob_boot = y_probs.iloc[idxs] # Use .iloc for Series
            stats.append(metric_func(y_true_boot, y_prob_boot))
        else:
            y_pred_boot = y_preds.iloc[idxs] # Use .iloc for Series
            if metric_func.__name__ in ["precision_score", "recall_score", "f1_score"]:
                stats.append(metric_func(y_true_boot, y_pred_boot, zero_division=0, average="weighted"))
            else: # For accuracy_score or other metrics not needing zero_division
                stats.append(metric_func(y_true_boot, y_pred_boot))
    
    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + (1.0 - alpha) / 2.0) * 100
    lower = np.percentile(stats, lower_p)
    upper = np.percentile(stats, upper_p)
    return lower, upper

# Prepare arrays (ensure y_test is a Series for .iloc)
y_test_series = y_test.reset_index(drop=True) if isinstance(y_test, pd.DataFrame) else pd.Series(y_test).reset_index(drop=True)
y_probs_series = pd.Series(y_pred_proba).reset_index(drop=True)
y_preds_series = pd.Series(y_pred).reset_index(drop=True)

print("\nComputing bootstrap confidence intervals...")
# Compute 95% confidence intervals for each metric
auc_ci = bootstrap_metric(roc_auc_score, y_test_series, y_probs_series, y_preds_series)
acc_ci = bootstrap_metric(accuracy_score, y_test_series, y_probs_series, y_preds_series)
precision_ci = bootstrap_metric(precision_score, y_test_series, y_probs_series, y_preds_series)
recall_ci = bootstrap_metric(recall_score, y_test_series, y_probs_series, y_preds_series)
f1_ci = bootstrap_metric(f1_score, y_test_series, y_probs_series, y_preds_series)

print(f"\n95% Bootstrap confidence intervals (n_boot=1000):")
print(f"  AUC CI:       [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
print(f"  Accuracy CI:  [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
print(f"  Precision CI: [{precision_ci[0]:.4f}, {precision_ci[1]:.4f}]")
print(f"  Recall CI:    [{recall_ci[0]:.4f}, {recall_ci[1]:.4f}]")
print(f"  F1‐Score CI:  [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")


# 4. Save results to JSON

# Collect all results in a dictionary
evaluation_summary = {
    "auc": {"point_estimate": auc_point, "ci_lower": auc_ci[0], "ci_upper": auc_ci[1]},
    "accuracy": {"point_estimate": acc_point, "ci_lower": acc_ci[0], "ci_upper": acc_ci[1]},
    "precision": {"point_estimate": precision_point, "ci_lower": precision_ci[0], "ci_upper": precision_ci[1]},
    "recall": {"point_estimate": recall_point, "ci_lower": recall_ci[0], "ci_upper": recall_ci[1]},
    "f1_score": {"point_estimate": f1_point, "ci_lower": f1_ci[0], "ci_upper": f1_ci[1]},
    "fairness_metrics": fairness_metrics_results
}

output_summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
os.makedirs(args.output_dir, exist_ok=True) # Ensure output directory exists

# Save the evaluation summary as a JSON file
with open(output_summary_path, 'w') as f:
    json.dump(evaluation_summary, f, indent=4)

print(f"\nEvaluation summary saved to: {output_summary_path}")
print("--- Finished Step 5: Model Evaluation ---")