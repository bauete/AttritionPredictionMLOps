"""
Step 2a: Preprocess IBM Dataset

This script preprocesses the IBM dataset for further analysis.
"""
import os
import pandas as pd
import argparse
from utils import load_data, save_data, save_model_or_object

def main(args):
    # Define file paths from arguments
    input_file = args.input_file
    output_file = args.output_file
    preprocessor_output_path = args.preprocessor_artifact_path
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if preprocessor_output_path:
        os.makedirs(os.path.dirname(preprocessor_output_path), exist_ok=True)
    
    print(f"--- Starting Step 2: Pre-processing IBM dataset ---")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    if preprocessor_output_path:
        print(f"Preprocessor artifact path: {preprocessor_output_path}")
    
    # Load the raw data
    df = load_data(input_file)

    # Encode target variable as binary
    target = 'Attrition'
    y = df[target].apply(lambda x: 1 if x == 'Yes' else 0)
    X = df.drop(target, axis=1)


    # List of columns to remove
    cols_to_remove = [
        'YearsWithCurrManager',
        'TrainingTimesLastYear',
        'StockOptionLevel',
        'StandardHours',
        'RelationshipSatisfaction',
        'PercentSalaryHike',
        'Over18',
        'NumCompaniesWorked',
        'MonthlyRate',
        'HourlyRate',
        'EmployeeCount',
        'EducationField',
        'Education',
        'DailyRate',
        'BusinessTravel',
        'JobRole'
    ]
    # Only drop columns that exist in the data
    actual_cols_to_remove = [col for col in cols_to_remove if col in X.columns]
    X = X.drop(columns=actual_cols_to_remove)
    print(f"Removed columns: {actual_cols_to_remove}")

    # Simplify 'Department' values
    if 'Department' in X.columns:
        X['Department'] = X['Department'].replace('Research & Development', 'Other')
        print("Replaced 'Research & Development' with 'Other' in 'Department' column.")

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Check and fill missing values for numerical features
    for col in numerical_features:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"Filled {X[col].isnull().sum()} missing values in '{col}' with median: {median_val}")
    
    # Check and fill missing values for categorical features
    for col in categorical_features:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0]
            X[col] = X[col].fillna(mode_val)
            print(f"Filled {X[col].isnull().sum()} missing values in '{col}' with mode: {mode_val}")

    # Bin 'MonthlyIncome' into 10 categories
    if 'MonthlyIncome' in X.columns:
        X['MonthlyIncome'] = pd.to_numeric(X['MonthlyIncome'], errors='coerce')
        # Handle any NaNs that might have been introduced by to_numeric or were already there
        if X['MonthlyIncome'].isnull().any():
            median_income = X['MonthlyIncome'].median()
            X['MonthlyIncome'] = X['MonthlyIncome'].fillna(median_income)
            print(f"Filled missing 'MonthlyIncome' with median ({median_income}) before binning.")

        try:
            X['MonthlyIncome_Category'] = pd.qcut(X['MonthlyIncome'], q=10, labels=False, duplicates='drop') + 1
            # Replace original 'MonthlyIncome' with the new categories
            X['MonthlyIncome'] = X['MonthlyIncome_Category']
            X.drop(columns=['MonthlyIncome_Category'], inplace=True)
            print("Categorized 'MonthlyIncome' into 10 bins (1-10).")
            # Update numerical_features list if 'MonthlyIncome' was in it, as it's now categorical in nature (ordinal)
            if 'MonthlyIncome' in numerical_features:
                numerical_features.remove('MonthlyIncome')
                # Optionally, add it to a list of ordinal features if you treat them differently
        except ValueError as e:
            print(f"Could not categorize 'MonthlyIncome' into 10 bins due to: {e}. Keeping original 'MonthlyIncome'.")

    # Combine processed features and target for saving
    X_processed = X.copy() 
    # Combine processed X and y for saving
    processed_df = pd.concat([X_processed, y.rename(target)], axis=1)
    save_data(processed_df, output_file)

    print(f"Processed data shape: {X_processed.shape}")
    print(f"Processed data saved to: {output_file}")
    
    # Additional processing for KPMG requirements
    result_df = processed_df.copy()
    
    # 13. EmployeeNumber (employee_id -> EmployeeNumber)
    if 'employee_id' in result_df.columns:
        result_df = result_df.rename(columns={'employee_id': 'EmployeeNumber'})
    
    # Save the final processed data
    save_data(result_df, output_file)
    print(f"Final processed data saved to: {output_file}")
    
    print("--- Finished Step 2: Pre-processing ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IBM dataset.")
    parser.add_argument("--input_file", type=str, default="data/processed/01_IBM_acquired_data.csv", help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, default="data/processed/02_IBM_preprocessed_data.csv", help="Path to save the preprocessed CSV file.")
    parser.add_argument("--preprocessor_artifact_path", type=str, default="artifacts/IBM_preprocessor.pkl", help="Path to save the preprocessor object.")
    
    args = parser.parse_args()
    main(args)