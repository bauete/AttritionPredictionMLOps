"""
Step 4: Train XGBoost Model for Attrition Prediction

This script trains an XGBoost model to predict employee attrition using the preprocessed IBM dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd
import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

# ----------------------------------------
# 1. Load preprocessed data
# ----------------------------------------

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train an XGBoost model for attrition prediction.")
parser.add_argument(
    "--input_file",
    type=str,
    default="data/processed/03_IBM_engineered_features.csv", # Default path
    help="Path to the preprocessed CSV data file."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="models", # Default output directory
    help="Directory to save the trained model, scaler, and parameters."
)
parser.add_argument(
    "--use_smote",
    type=bool,
    default=True,
    help="Whether to use SMOTE for oversampling the minority class."
)
args = parser.parse_args()

DATA_PATH = args.input_file
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

print(f"Loading data from: {DATA_PATH}")
print(f"Saving artifacts to: {OUTPUT_DIR}")

df = pd.read_csv(DATA_PATH)
# Assume 'Attrition' is the binary target column (0/1) and drop any ID columns
columns_to_drop = ['Attrition']
if 'EmployeeNumber' in df.columns:
    columns_to_drop.append('EmployeeNumber')
if 'Unnamed: 0' in df.columns: # Often an index column from previous saves
    columns_to_drop.append('Unnamed: 0')

X = df.drop(columns=columns_to_drop, errors='ignore') # Use errors='ignore' for robustness
y = df["Attrition"]

# ----------------------------------------
# 2. Train/validation split (final hold-out)
# ----------------------------------------
# We keep a small hold-out test set apart; this script only trains on the remaining data.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.10,
    stratify=y,
    random_state=42
)

# Calculate scale_pos_weight for handling class imbalance
counts = y_trainval.value_counts()
scale_pos_weight_calculated = counts[0] / counts[1] if len(counts) == 2 and counts[1] > 0 else 1
print(f"Calculated scale_pos_weight: {scale_pos_weight_calculated}")

# ----------------------------------------
# 3. Preprocessing: Standardize numeric features
# ----------------------------------------
numeric_cols = X_trainval.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()

# Fit scaler on the training-validation set, then transform trainval (test is left unused here)
X_trainval[numeric_cols] = scaler.fit_transform(X_trainval[numeric_cols])

# Save the scaler for future use
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# ----------------------------------------
# 3b. Apply SMOTE to handle class imbalance
# ----------------------------------------
if args.use_smote:
    print("\nApplying SMOTE to balance classes in training data...")
    # Split for separate SMOTE application (only on training data)
    X_train_pre_smote, X_val, y_train_pre_smote, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.10,
        stratify=y_trainval,
        random_state=42
    )
    # Apply SMOTE to the training portion only
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train_pre_smote, y_train_pre_smote)
    print(f"Original class distribution in training data: {np.bincount(y_train_pre_smote.astype(int))}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train.astype(int))}")
    smote_info = {
        "applied": True,
        "before_balance_ratio": float(counts[0] / counts[1]) if counts[1] > 0 else float('inf'),
        "after_balance_ratio": 1.0  # SMOTE creates a perfectly balanced dataset by default
    }
else:
    # Regular train/val split without SMOTE
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.10,
        stratify=y_trainval,
        random_state=42
    )
    smote_info = {
        "applied": False
    }

# ----------------------------------------
# 4. Hyperparameter search with StratifiedKFold + RandomizedSearchCV
# ----------------------------------------
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    seed=42,
    verbosity=0
)

# Define parameter distributions for RandomizedSearch
param_dist = {
    "learning_rate": uniform(loc=0.01, scale=0.3),        # between 0.01 and 0.31
    "n_estimators": randint(100, 1000),                   # between 100 and 999
    "max_depth": randint(3, 11),                          # integers [3, 10]
    "min_child_weight": randint(1, 11),                   # integers [1, 10]
    "subsample": uniform(loc=0.5, scale=0.5),             # between 0.5 and 1.0
    "colsample_bytree": uniform(loc=0.5, scale=0.5),      # between 0.5 and 1.0
    "reg_alpha": uniform(loc=0.0, scale=1.0),             # L1 regularization [0, 1]
    "reg_lambda": uniform(loc=0.0, scale=1.0),            # L2 regularization [0, 1]
    # "gamma": uniform(loc=0.0, scale=1.0)                # Minimum loss reduction [0, 1]
}

# Only include scale_pos_weight when not using SMOTE
if not args.use_smote:
    param_dist["scale_pos_weight"] = [scale_pos_weight_calculated, 
                                     scale_pos_weight_calculated * 0.8, 
                                     scale_pos_weight_calculated * 1.2, 
                                     scale_pos_weight_calculated * 1.5]

# Use StratifiedKFold to preserve Attrition ratio in each fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Randomized search over configurations
rand_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=50,
    scoring="f1",
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    return_train_score=True  # Make sure to return training scores
)

# Fit on the train+val set
rand_search.fit(X_trainval, y_trainval)

print("Best hyperparameters found:")
print(rand_search.best_params_)
print(f"Best CV F1: {rand_search.best_score_:.4f}")

# ----------------------------------------
# 4.5 Extract and save all fold metrics for statistical testing
# ----------------------------------------
# Extract cross-validation results into a DataFrame
cv_results = pd.DataFrame(rand_search.cv_results_)

# Get the index of the best configuration
best_idx = rand_search.best_index_

# Extract the scores for each fold for the best configuration
best_config_scores = {}
for i in range(cv.n_splits):
    test_fold_score = cv_results.loc[best_idx, f"split{i}_test_score"]
    train_fold_score = cv_results.loc[best_idx, f"split{i}_train_score"]
    best_config_scores[f"fold_{i+1}"] = {
        "test_f1": float(test_fold_score),
        "train_f1": float(train_fold_score)
    }

print("\nF1 scores for each fold with best hyperparameters:")
for fold, scores in best_config_scores.items():
    print(f"  {fold}: Test F1 = {scores['test_f1']:.4f}, Train F1 = {scores['train_f1']:.4f}")

# ----------------------------------------
# 5. Retrain final model with early stopping
# ----------------------------------------
# Split the train+val further into train and a small early-stop validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.10,
    stratify=y_trainval,
    random_state=42
)

# Take the best parameters from the randomized search
best_params = rand_search.best_params_
final_model = XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="auc",
    seed=42,
    verbosity=0,
    **best_params
)

# Early stopping settings: stop if AUC does not improve over 10 rounds
final_model.fit(
    X_train,
    y_train,
    early_stopping_rounds=10,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ----------------------------------------
# 6. Calculate and save training and validation metrics
# ----------------------------------------
print("\nCalculating performance metrics across data splits...")

def calculate_metrics(model, X, y, split_name):
    """
    Calculate and print accuracy, precision, recall, F1, and AUC for a given split.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    # Convert to binary predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y, y_pred_proba))
    }
    print(f"{split_name} metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    return metrics

# Calculate metrics for each data split
training_metrics = calculate_metrics(final_model, X_train, y_train, "Training")
validation_metrics = calculate_metrics(final_model, X_val, y_val, "Validation")
trainval_metrics = calculate_metrics(final_model, X_trainval, y_trainval, "Train+Validation")
test_metrics = calculate_metrics(final_model, X_test, y_test, "Test (hold-out)")

# Combine all metrics into a single dictionary for saving
performance_metrics = {
    "training": training_metrics,
    "validation": validation_metrics,
    "train_validation_combined": trainval_metrics,
    "test": test_metrics,
    "best_cv_score": float(rand_search.best_score_),
    "cv_results_summary": {
        "mean_score": float(np.mean(rand_search.cv_results_["mean_test_score"])),
        "std_score": float(np.std(rand_search.cv_results_["mean_test_score"])),
        "max_score": float(np.max(rand_search.cv_results_["mean_test_score"])),
        "min_score": float(np.min(rand_search.cv_results_["mean_test_score"]))
    }
}

# Save metrics to a JSON file
with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), 'w') as f:
    json.dump({
        **performance_metrics,
        "smote": smote_info,
        "cross_validation": {
            "folds": best_config_scores,
            "mean_test_f1": float(np.mean([scores["test_f1"] for scores in best_config_scores.values()])),
            "std_test_f1": float(np.std([scores["test_f1"] for scores in best_config_scores.values()])),
            "mean_train_f1": float(np.mean([scores["train_f1"] for scores in best_config_scores.values()])),
            "std_train_f1": float(np.std([scores["train_f1"] for scores in best_config_scores.values()])),
        }
    }, f, indent=2)

print(f"\nTraining and validation metrics saved to {os.path.join(OUTPUT_DIR, 'training_metrics.json')}")

# Save the final model and best parameters to disk
joblib.dump(final_model, os.path.join(OUTPUT_DIR, "model.pkl"))
joblib.dump(best_params, os.path.join(OUTPUT_DIR, "xgb_best_params.pkl"))

print(f"Model training complete. Artifacts saved under '{OUTPUT_DIR}'.")
