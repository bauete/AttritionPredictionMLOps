"""
Fairlearn Analysis Script

This script performs fairness analysis on a trained model using Fairlearn's MetricFrame.
"""
import os
import joblib
import json
import numpy as np
import pandas as pd
import argparse

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference


def main():
    ---------
    # 0. Argument Parsing 
    ---------
    parser = argparse.ArgumentParser(
        description="Analyze model fairness using Fairlearn's MetricFrame."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the engineered CSV data file (full dataset)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model.pkl file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the fairness_analysis.json."
    )
    args = parser.parse_args()

    ---------
    # 1. Load data, model, and scaler
    ---------
    print(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)

    # Define target and drop unneeded columns
    target_column = "Attrition"
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the input data.")

    y = df[target_column]
    columns_to_drop = [target_column]
    if "EmployeeNumber" in df.columns:
        columns_to_drop.append("EmployeeNumber")
    if "Unnamed: 0" in df.columns:
        columns_to_drop.append("Unnamed: 0")

    X = df.drop(columns=columns_to_drop, errors="ignore")

    # Split data to isolate the hold-out test set (10% of data)
    print("Splitting data into train/test to isolate the hold-out test set...")
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        stratify=y,
        random_state=42
    )
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")

    # Load scaler for numeric features
    scaler_path = os.path.join(os.path.dirname(args.model_path), "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Ensure it was saved by the trainer.")
    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Apply scaling to numeric features
    numeric_cols = [
        "Age", "DistanceFromHome", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
        "JobSatisfaction", "MonthlyIncome", "PerformanceRating", "TotalWorkingYears",
        "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "StagnationScore", "TenureRatio", "PCA1", "PCA2", "PCA3"
    ]
    X_test_scaled = X_test.copy()
    if all(col in X_test.columns for col in numeric_cols):
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Load trained model
    print(f"Loading model from: {args.model_path}")
    model = joblib.load(args.model_path)

    ---------
    # 2. Generate predictions
    ---------
    print("Generating predictions...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    ---------
    # 3. Prepare sensitive features for fairness analysis
    ---------
    sensitive_features = {}

    # Gender: handle both one-hot and single-column encodings
    if "Gender_Female" in X_test.columns and "Gender_Male" in X_test.columns:
        gender_series = pd.Series("Unknown", index=X_test.index)
        gender_series[X_test["Gender_Female"] == 1] = "Female"
        gender_series[X_test["Gender_Male"] == 1] = "Male"
        sensitive_features["Gender"] = gender_series
    elif "Gender_Female" in X_test.columns:
        gender_series = X_test["Gender_Female"].map({0: "Male", 1: "Female"})
        sensitive_features["Gender"] = gender_series
    elif "Gender_Male" in X_test.columns:
        gender_series = X_test["Gender_Male"].map({0: "Female", 1: "Male"})
        sensitive_features["Gender"] = gender_series

    # Age: bin into groups for group fairness analysis
    if "Age" in X_test.columns:
        raw_age_bins = pd.cut(
            X_test["Age"],
            bins=[0, 29, 39, 49, X_test["Age"].max() + 1],
            labels=["<30", "30-39", "40-49", "50+"],
            right=False
        )
        present_bins = raw_age_bins.dropna().unique().tolist()
        filtered_age_bins = raw_age_bins.where(raw_age_bins.isin(present_bins))
        sensitive_features["Age_Group"] = filtered_age_bins

    ---------
    # 4. Create MetricFrame for label-based metrics (accuracy, precision, recall, F1)
    ---------
    print("\nCreating MetricFrame with multiple label-based metrics...")

    metrics = {
        "accuracy": lambda y_t, y_p: accuracy_score(y_t, y_p),
        "precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        "recall": lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
        "f1_score": lambda y_t, y_p: f1_score(y_t, y_p, zero_division=0)
        # Note: AUC is computed separately per group
    }

    results = {}
    fairness_metrics = {}

    # Loop over each sensitive feature (e.g., Gender, Age_Group)
    for feature_name, feature_vals in sensitive_features.items():
        print(f"\n*** {feature_name} MetricFrame Summary ***")

        # Build MetricFrame for group-wise and overall metrics
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=feature_vals
        )

        # Print and store overall metrics
        print("Overall metrics:")
        for metric_name, metric_value in metric_frame.overall.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # Print and store group-specific metrics
        print(f"\nMetrics by {feature_name} groups:")
        for group_label in metric_frame.by_group.index:
            print(f"  {group_label}:")
            for metric_name in metrics.keys():
                group_value = metric_frame.by_group.loc[group_label, metric_name]
                print(f"    {metric_name}: {group_value:.4f}")

        # Disparities (difference between best and worst group)
        group_labels = metric_frame.by_group.index.tolist()
        if len(group_labels) > 1:
            print(f"\nDisparities for {feature_name}:")
            disparity_dict = metric_frame.difference(method="between_groups")
            for metric_name in metrics.keys():
                print(f"  {metric_name} difference: {disparity_dict[metric_name]:.4f}")
        else:
            print(f"\nDisparities for {feature_name}: N/A (only one group present)")
            disparity_dict = {k: None for k in metrics.keys()}

        # Ratios (worst/best group)
        if len(group_labels) > 1:
            print(f"\nRatios for {feature_name}:")
            ratio_dict = {}
            for metric_name in metrics.keys():
                try:
                    ratio_value = metric_frame.ratio(method="between_groups")[metric_name]
                    print(f"  {metric_name} ratio: {ratio_value:.4f}")
                    ratio_dict[metric_name] = ratio_value
                except ZeroDivisionError:
                    print(f"  {metric_name} ratio: N/A (division by zero)")
                    ratio_dict[metric_name] = None
        else:
            print(f"\nRatios for {feature_name}: N/A (only one group present)")
            ratio_dict = {k: None for k in metrics.keys()}

        # Fairness metrics: Demographic Parity Difference & Equalized Odds Difference
        dpd = None
        eod = None
        if len(group_labels) > 1:
            dpd = demographic_parity_difference(
                y_true=y_test, y_pred=y_pred, sensitive_features=feature_vals
            )
            eod = equalized_odds_difference(
                y_true=y_test, y_pred=y_pred, sensitive_features=feature_vals
            )
            print(f"\n  Demographic Parity Difference: {dpd:.4f}")
            print(f"  Equalized Odds Difference: {eod:.4f}")
        else:
            print(f"\n  Demographic Parity Difference: N/A (only one group present)")
            print(f"  Equalized Odds Difference: N/A (only one group present)")

        # Store all label-based results under this feature
        results[feature_name] = {
            "overall": {k: float(v) for k, v in metric_frame.overall.items()},
            "by_group": {
                grp: {k: float(metric_frame.by_group.loc[grp, k]) for k in metrics}
                for grp in metric_frame.by_group.index
            },
            "label_based_disparities": {k: (float(disparity_dict[k]) if disparity_dict[k] is not None else None) 
                                       for k in metrics},
            "label_based_ratios": {k: (float(ratio_dict[k]) if ratio_dict[k] is not None else None)
                                  for k in metrics},
            "demographic_parity_difference": (float(dpd) if dpd is not None else None),
            "equalized_odds_difference": (float(eod) if eod is not None else None)
        }

        ---------
        # 5. Compute per-group AUC manually
        ---------
        auc_by_group = {}
        for grp in feature_vals.dropna().unique():
            mask = (feature_vals == grp)
            y_true_grp = y_test[mask]
            y_proba_grp = y_pred_proba[mask]

            # Only compute AUC if there are at least two unique labels in the subgroup
            if len(y_true_grp) < 2 or len(np.unique(y_true_grp)) < 2:
                auc_by_group[grp] = None
            else:
                auc_by_group[grp] = float(roc_auc_score(y_true_grp, y_proba_grp))

        # Compute overall AUC on the full test set
        try:
            overall_auc = float(roc_auc_score(y_test, y_pred_proba))
        except ValueError:
            overall_auc = None

        # Compute AUC disparity & ratio if there are at least two valid group AUCs
        valid_auc_vals = [v for v in auc_by_group.values() if v is not None]
        if len(valid_auc_vals) > 1:
            auc_diff = float(max(valid_auc_vals) - min(valid_auc_vals))
            auc_ratio = float(min(valid_auc_vals) / max(valid_auc_vals))
        else:
            auc_diff = None
            auc_ratio = None

        print(f"\nAUC by {feature_name}:")
        for grp, auc_val in auc_by_group.items():
            if auc_val is None:
                print(f"  {grp}: N/A")
            else:
                print(f"  {grp}: {auc_val:.4f}")

        if overall_auc is not None:
            print(f"Overall AUC: {overall_auc:.4f}")
        else:
            print("Overall AUC: N/A")

        if auc_diff is not None:
            print(f"AUC disparity: {auc_diff:.4f}")
        else:
            print("AUC disparity: N/A")

        if auc_ratio is not None:
            print(f"AUC ratio: {auc_ratio:.4f}")
        else:
            print("AUC ratio: N/A")

        # Attach AUC info to the results structure
        results[feature_name].update({
            "overall_auc": overall_auc,
            "auc_by_group": {grp: (float(val) if val is not None else None) for grp, val in auc_by_group.items()},
            "auc_difference": auc_diff,
            "auc_ratio": auc_ratio
        })

        # Collect the named fairness metrics for summary
        if len(group_labels) > 1:
            fairness_metrics[f"demographic_parity_difference_{feature_name}"] = dpd
            fairness_metrics[f"equalized_odds_difference_{feature_name}"] = eod
        else:
            fairness_metrics[f"demographic_parity_difference_{feature_name}"] = None
            fairness_metrics[f"equalized_odds_difference_{feature_name}"] = None

        fairness_metrics[f"auc_difference_{feature_name}"] = auc_diff
        fairness_metrics[f"auc_ratio_{feature_name}"] = auc_ratio

    ---------
    # 6. Save results to JSON
    ---------
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fairness_analysis.json")

    with open(output_path, "w") as f:
        json.dump(
            {
                "metric_frame_results": results,
                "fairness_metrics": fairness_metrics
            },
            f,
            indent=4
        )

    print(f"\nFairness analysis saved to: {output_path}")
    print("--- Completed Fairlearn MetricFrame Analysis ---")


if __name__ == "__main__":
    main()
