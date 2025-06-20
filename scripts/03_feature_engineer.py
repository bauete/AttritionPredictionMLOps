# scripts/03_feature_engineer.py
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from utils import load_data, save_data, save_model_or_object, load_model_or_object
from sklearn.compose import ColumnTransformer
import os
import json

def handle_missing_columns(df, expected_cols_path, training_mode=False):
    """
    Align DataFrame columns to an expected list for consistency between training and inference.
    In training mode, save the columns; in application mode, align to the saved columns.
    """
    if training_mode:
        # Save the current columns (after OHE, before new features)
        current_columns = df.columns.tolist()
        try:
            os.makedirs(os.path.dirname(expected_cols_path), exist_ok=True)
            with open(expected_cols_path, 'w') as f:
                json.dump(current_columns, f, indent=4)
            print(f"Current feature columns (post-OHE) saved to {expected_cols_path}.")
        except Exception as e:
            print(f"Error saving expected columns to {expected_cols_path}: {e}.")
        return df
    else:
        # Application mode: align columns to expected
        if not os.path.exists(expected_cols_path):
            print(f"Warning: Expected columns file {expected_cols_path} not found. Column alignment cannot be performed. Returning DataFrame as is.")
            return df
        try:
            with open(expected_cols_path, 'r') as f:
                expected_columns = json.load(f)
            print(f"Expected columns (post-OHE) loaded from {expected_cols_path}.")
        except Exception as e:
            print(f"Error loading expected columns from {expected_cols_path}: {e}. Returning DataFrame as is.")
            return df

        current_columns = df.columns.tolist()
        aligned_df = pd.DataFrame(0, index=df.index, columns=expected_columns)
        cols_to_copy = [col for col in expected_columns if col in current_columns]
        aligned_df[cols_to_copy] = df[cols_to_copy]
        missing_cols = [col for col in expected_columns if col not in current_columns]
        if missing_cols:
            print(f"Added missing expected columns (filled with 0): {missing_cols}.")
        extra_cols = [col for col in current_columns if col not in expected_columns]
        if extra_cols:
            print(f"Dropped unexpected columns: {extra_cols}.")
        print(f"Columns aligned. Original shape: {df.shape}, Aligned shape: {aligned_df.shape}.")
        return aligned_df

def main(args):
    """
    Main function for the feature engineering step.
    Loads data, applies OHE, aligns columns, creates new features, and saves output.
    """
    print("--- Starting Step 3: Feature Engineering ---")
    df = load_data(args.input_file)

    if df is None or df.empty:
        print(f"Error: No data loaded from {args.input_file}. Exiting feature engineering.")
        return

    target = 'Attrition'
    if target in df.columns:
        # Map target to 1/0
        y = df[target].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)
        X = df.drop(target, axis=1)
    else:
        X = df.copy()
        y = None

    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        print(f"One-hot encoding categorical columns: {categorical_cols.tolist()}")
        X = pd.get_dummies(X, columns=categorical_cols, dummy_na=False, prefix=categorical_cols, prefix_sep='_')
        print(f"Shape after one-hot encoding: {X.shape}")
    else:
        print("No categorical columns found for one-hot encoding.")

    # Align columns using expected_cols.json for consistency
    if hasattr(args, 'expected_cols_path') and args.expected_cols_path:
        training_mode_for_expected_cols = (not args.apply_only) and (y is not None)
        print(f"Handling column standardization. Training mode for expected_cols.json: {training_mode_for_expected_cols}.")
        X = handle_missing_columns(X, args.expected_cols_path, training_mode=training_mode_for_expected_cols)
        print(f"Shape after handle_missing_columns: {X.shape}")
    else:
        print("Warning: --expected_cols_path not provided. Column consistency across runs is not guaranteed.")

    print("Creating new features...")
    X = create_new_features(X)
    print(f"Shape after create_new_features: {X.shape}")

    # Concatenate engineered features and target
    if y is not None:
        engineered_df = pd.concat([X, y.rename(target)], axis=1)
    else:
        engineered_df = X.copy()

    save_data(engineered_df, args.output_file)
    print(f"Engineered data shape: {engineered_df.shape}")
    print("--- Finished Step 3: Feature Engineering ---")

def create_new_features(X):
    """
    Engineer new features based on domain knowledge and exploratory analysis.
    """
    X_new = X.copy()

    # Stagnation Score: Proportion of tenure spent without promotion
    if 'YearsAtCompany' in X.columns and 'YearsSinceLastPromotion' in X.columns:
        X_new['StagnationScore'] = np.where(
            X['YearsAtCompany'] > 0,
            X['YearsSinceLastPromotion'] / X['YearsAtCompany'],
            0
        )
        scaler = MinMaxScaler()
        X_new['StagnationScore'] = scaler.fit_transform(X_new[['StagnationScore']])[:, 0]

    # Tenure Ratio: Fraction of career at current employer
    if 'YearsAtCompany' in X.columns and 'TotalWorkingYears' in X.columns:
        X_new['TenureRatio'] = np.where(
            X['TotalWorkingYears'] > 0,
            X['YearsAtCompany'] / X['TotalWorkingYears'],
            0
        )
        scaler = MinMaxScaler()
        X_new['TenureRatio'] = scaler.fit_transform(X_new[['TenureRatio']])[:, 0]

    # Workload Index: Combines overtime and job involvement
    if 'OverTime' in X.columns and 'JobInvolvement' in X.columns:
        overtime_numeric = X['OverTime'].map({'Yes': 1, 'No': 0}) if X['OverTime'].dtype == 'object' else X['OverTime']
        normalized_job_involvement = (X['JobInvolvement'] - 1) / 3
        X_new['WorkloadIndex'] = (0.7 * overtime_numeric) + (0.3 * normalized_job_involvement)
        X_new['WorkloadIndex'] = X_new['WorkloadIndex'].clip(lower=0, upper=1)

    # PCA components from numerical variables
    try:
        num_cols = X_new.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(num_cols) >= 3:
            X_new[num_cols] = X_new[num_cols].replace([np.inf, -np.inf], np.nan)
            for col in num_cols:
                if X_new[col].isnull().any():
                    col_median = X_new[col].median()
                    if pd.isna(col_median):
                        X_new[col] = X_new[col].fillna(0)
                    else:
                        X_new[col] = X_new[col].fillna(col_median)
            existing_num_cols = [col for col in num_cols if col in X_new.columns]
            if not existing_num_cols or len(existing_num_cols) < 1:
                print("No numerical columns suitable for PCA after NaN/Inf handling.")
            else:
                data_for_scaling = X_new[existing_num_cols].dropna(axis=1, how='all')
                data_for_scaling = data_for_scaling.fillna(0)
                if data_for_scaling.empty or data_for_scaling.shape[1] == 0:
                    print("No numerical columns left for PCA after final cleaning.")
                else:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(data_for_scaling)
                    n_pca_components = min(3, X_scaled.shape[1])
                    if n_pca_components > 0:
                        pca = PCA(n_components=n_pca_components)
                        pca_result = pca.fit_transform(X_scaled)
                        for i in range(pca_result.shape[1]):
                            X_new[f'PCA{i+1}'] = pca_result[:, i]
                        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
                    else:
                        print("Not enough features for PCA.")
    except Exception as e:
        print(f"Error in PCA calculation: {e}")

    # Final imputation for any remaining NaNs/Infs
    numerical_cols_final = X_new.select_dtypes(include=np.number).columns
    X_new[numerical_cols_final] = X_new[numerical_cols_final].replace([np.inf, -np.inf], np.nan)
    for col in numerical_cols_final:
        if X_new[col].isnull().any():
            col_median = X_new[col].median()
            if pd.isna(col_median):
                X_new[col] = X_new[col].fillna(0)
            else:
                X_new[col] = X_new[col].fillna(col_median)
        if np.isinf(X_new[col]).any():
             X_new[col] = X_new[col].replace([np.inf, -np.inf], 0)

    return X_new

def select_features(X, y, k=20):
    """
    Select the top 'k' features based on ANOVA F-value (SelectKBest).
    """
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)

    # Get the names of the selected columns.
    cols = X.columns[selector.get_support()]
    return X[cols], selector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering for attrition data.")
    parser.add_argument("--input_file", type=str, default="data/processed/02_IBM_preprocessed_data.csv", help="Path to processed data file.")
    parser.add_argument("--output_file", type=str, default="data/processed/03_IBM_engineered_features.csv", help="Path to save engineered data.")
    parser.add_argument("--feature_selector_path", type=str, default="artifacts/feature_selector.pkl", help="Path to save/load the feature selector object.")
    parser.add_argument("--expected_cols_path", type=str, default="artifacts/expected_cols_post_ohe.json", help="Path to save/load expected columns JSON file.")
    parser.add_argument("--apply_only", action="store_true", help="Flag to load and apply an existing feature selector and expected columns, rather than refitting.")

    args = parser.parse_args()
    main(args)