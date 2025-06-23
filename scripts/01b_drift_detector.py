"""
Step 1b: Drift Detector
*MLOps Only

This script detects data drift by comparing current and reference datasets using statistical tests and Isolation Forest.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

def convert_to_serializable(obj):
    """Recursively converts numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

# Constants for drift detection thresholds
ALPHA = 0.05  # Significance level for statistical tests
STATISTICAL_DRIFT_FEATURE_THRESHOLD = 0.1  # Fraction of features that must drift to flag overall drift
ANOMALY_RATE_THRESHOLD = 0.05  # Fraction of anomalies needed to flag drift

def preprocess_for_isolation_forest(df, features, categorical_cols, fit_encoders=False, encoders=None):
    """
    Prepares data for Isolation Forest: fills missing values and encodes categoricals.
    """
    df_proc = df.copy()

    # Impute missing values for each feature
    for col in features:
        if col in df_proc:
            if df_proc[col].isnull().any():
                if df_proc[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_proc[col]):
                    mode_val = df_proc[col].mode()
                    df_proc[col] = df_proc[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
                else:
                    df_proc[col] = df_proc[col].fillna(df_proc[col].median())
        else:
            # If column missing, fill with zeros
            df_proc[col] = 0

    # Ensure all features exist
    for col in features:
        if col not in df_proc:
            df_proc[col] = 0

    df_subset = df_proc[features].copy()

    if fit_encoders:
        encoders = {}

    # Encode categorical columns
    for col in categorical_cols:
        if col in df_subset:
            if fit_encoders:
                le = LabelEncoder()
                df_subset[col] = le.fit_transform(df_subset[col].astype(str))
                encoders[col] = le
            else:
                if encoders and col in encoders:
                    le = encoders[col]
                    values = df_subset[col].astype(str)
                    mask_known = values.isin(le.classes_)
                    df_subset.loc[mask_known, col] = le.transform(values[mask_known])
                    df_subset.loc[~mask_known, col] = -1
                    df_subset[col] = df_subset[col].astype(int)
                else:
                    df_subset[col] = -1

    return (df_subset, encoders) if fit_encoders else df_subset

def detect_data_drift(
    current_csv,
    reference_csv,
    if_model_path,
    if_features_path,
    if_encoders_path,
    retrain_if=False,
    output_report_path=None
):
    """
    Compares current and reference data using statistical tests and Isolation Forest.
    Returns drift status and a detailed report.
    """
    stats = {"statistical_tests": {}, "anomaly_detection": {}}
    overall_drift = False

    # Load datasets
    try:
        cur_df = pd.read_csv(current_csv)
        ref_df = pd.read_csv(reference_csv)
    except Exception as e:
        stats["error"] = str(e)
        return True, stats

    # Use only columns present in both datasets
    common_cols = list(set(cur_df.columns) & set(ref_df.columns))
    if not common_cols:
        stats["error"] = "No common columns"
        return True, stats

    # Identify numeric and categorical columns
    num_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(ref_df[c])]
    cat_cols = [c for c in common_cols if not pd.api.types.is_numeric_dtype(ref_df[c])]

    # --- Statistical drift tests ---
    drifted_count = 0
    for col in common_cols:
        ref_series = ref_df[col].dropna()
        cur_series = cur_df[col].dropna()
        if ref_series.empty or cur_series.empty:
            stats["statistical_tests"][col] = {"status": "skipped_empty", "p_value": None}
            continue

        p_value = None
        test_type = None

        if col in num_cols:
            # Kolmogorov–Smirnov test for numeric columns
            ref_numeric = pd.to_numeric(ref_series, errors='coerce').dropna()
            cur_numeric = pd.to_numeric(cur_series, errors='coerce').dropna()
            if not ref_numeric.empty and not cur_numeric.empty and len(ref_numeric.unique()) > 1 and len(cur_numeric.unique()) > 1:
                _, p_value = ks_2samp(ref_numeric, cur_numeric)
                test_type = "KS"
            else:
                stats["statistical_tests"][col] = {"status": "skipped_mixed_or_empty_numeric", "p_value": None}
                continue
        elif col in cat_cols:
            # Chi-squared test for categorical columns
            categories = pd.concat([ref_series, cur_series]).unique()
            o_ref = ref_series.value_counts().reindex(categories, fill_value=0)
            o_cur = cur_series.value_counts().reindex(categories, fill_value=0)
            if o_ref.sum() > 0 and o_cur.sum() > 0:
                contingency = pd.DataFrame({"ref": o_ref, "cur": o_cur})
                try:
                    _, p_value, _, _ = chi2_contingency(contingency, lambda_="log-likelihood")
                    test_type = "ChiSq"
                except:
                    stats["statistical_tests"][col] = {"status": "skipped_test_error", "p_value": None}
                    continue
            else:
                stats["statistical_tests"][col] = {"status": "skipped_no_data", "p_value": None}
                continue

        if p_value is not None:
            drift_flag = p_value < ALPHA
            stats["statistical_tests"][col] = {"p_value": p_value, "drift": drift_flag, "test": test_type}
            if drift_flag:
                drifted_count += 1
        else:
            if col not in stats["statistical_tests"]:
                stats["statistical_tests"][col] = {"status": "skipped", "p_value": None}

    # Summarize statistical drift
    total = len(common_cols)
    pct_drift = drifted_count / total
    stats["stat_summary"] = {
        "total_features": total,
        "drifted_features": drifted_count,
        "drift_pct": pct_drift
    }
    if pct_drift > STATISTICAL_DRIFT_FEATURE_THRESHOLD:
        overall_drift = True

    # --- Anomaly detection with Isolation Forest ---
    if_features = num_cols + cat_cols

    if retrain_if:
        # Train new Isolation Forest on reference data
        if if_features:
            ref_if_df, encs = preprocess_for_isolation_forest(ref_df, if_features, cat_cols, fit_encoders=True)
            if not ref_if_df.empty:
                model = IsolationForest(contamination='auto', random_state=42)
                model.fit(ref_if_df)
                os.makedirs(os.path.dirname(if_model_path), exist_ok=True)
                joblib.dump(model, if_model_path)
                joblib.dump(if_features, if_features_path)
                joblib.dump(encs, if_encoders_path)
                stats["anomaly_detection"]["status"] = "trained_new_IF"
            else:
                stats["anomaly_detection"]["status"] = "empty_ref_for_IF"
        else:
            stats["anomaly_detection"]["status"] = "no_features_for_IF"
    else:
        # Load existing model and apply to current data
        if os.path.exists(if_model_path) and os.path.exists(if_features_path) and os.path.exists(if_encoders_path):
            model = joblib.load(if_model_path)
            trained_feats = joblib.load(if_features_path)
            trained_encs = joblib.load(if_encoders_path)
            cur_if_df = preprocess_for_isolation_forest(cur_df, trained_feats, [c for c in cat_cols if c in trained_feats], fit_encoders=False, encoders=trained_encs)
            if not cur_if_df.empty:
                preds = model.predict(cur_if_df)  # -1 = anomaly, 1 = normal
                anomaly_rate = np.mean(preds == -1)
                drift_flag = anomaly_rate > ANOMALY_RATE_THRESHOLD
                stats["anomaly_detection"] = {
                    "status": "used_loaded_IF",
                    "anomaly_rate": anomaly_rate,
                    "drift": drift_flag
                }
                if drift_flag:
                    overall_drift = True
            else:
                stats["anomaly_detection"]["status"] = "empty_cur_for_IF"
        else:
            stats["anomaly_detection"]["status"] = "skipped_IF_not_found"

    stats["overall_drift"] = overall_drift

    # Save report if requested
    if output_report_path:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        try:
            serializable_stats = convert_to_serializable(stats)
            with open(output_report_path, "w") as f:
                json.dump(serializable_stats, f, indent=2)
            stats["report_saved_to"] = output_report_path
        except Exception as e:
            stats["report_save_error"] = str(e)

    return overall_drift, stats

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Detect data drift using statistical tests and Isolation Forest.")
    parser.add_argument("--current_data", required=True, help="Path to the CSV file of the current data to check for drift.")
    parser.add_argument("--reference_data", default="data/processed/reference_data_acquired.csv", help="Path to the CSV file of the reference (baseline) data.")
    parser.add_argument("--if_model_path", default="artifacts/drift_detector/if_model.joblib", help="Path to save/load the Isolation Forest model (.joblib).")
    parser.add_argument("--if_features_path", default="artifacts/drift_detector/if_features.joblib", help="Path to save/load the list of features for the Isolation Forest model (.joblib).")
    parser.add_argument("--if_encoders_path", default="artifacts/drift_detector/if_encoders.joblib", help="Path to save/load the LabelEncoders for the Isolation Forest model (.joblib).")
    parser.add_argument("--retrain_if", action="store_true", help="If specified, a new Isolation Forest model will be trained on the reference data. Defaults to False.")
    parser.add_argument("--output_report", default=None, help="Optional: Path to save a JSON formatted drift details report.")
    parser.add_argument("--drift_status_file", default=None, help="Optional: Path to a file where the drift status ('DRIFT_DETECTED' or 'NO_DRIFT') will be written (e.g., 'drift_status.txt').")
    args = parser.parse_args()

    # Run drift detection
    drift, details = detect_data_drift(
        current_csv=args.current_data,
        reference_csv=args.reference_data,
        if_model_path=args.if_model_path,
        if_features_path=args.if_features_path,
        if_encoders_path=args.if_encoders_path,
        retrain_if=args.retrain_if,
        output_report_path=args.output_report
    )

    # Print or save drift status
    if not args.output_report:
        print("DRIFT_DETECTED" if drift else "NO_DRIFT")
    else:
        print(f"Drift analysis complete. Report: {args.output_report}")