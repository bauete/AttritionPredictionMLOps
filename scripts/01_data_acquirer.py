"""
Step 1: Data Acquirer

Loads a dataset, performs basic checks (missing values, duplicates, target column),
and saves the cleaned data for further processing.
"""

import argparse
import pandas as pd
from utils import load_data, save_data

def main(args):
    print("--- Starting Step 1: Data Loading ---")
    df = load_data(args.input_file)

    print("Initial dataset shape:", df.shape)
    print("Checking for missing values...")
    print(df.isnull().sum().sum())
    print("Checking for duplicate rows...")
    print(df.duplicated().sum())

    # For non-KPMG datasets, ensure the target column exists
    if "kpmg" not in args.input_file.lower():
        if 'Attrition' not in df.columns:
            raise ValueError("Target column 'Attrition' not found in the dataset.")
    else:
        print("Skipping 'Attrition' column check for KPMG dataset (assumed to be handled in KPMG preprocessing).")

    save_data(df, args.output_file)
    print("--- Finished Step 1: Data Loading ---")

if __name__ == "__main__":
    # Parse command-line arguments for input/output file paths
    parser = argparse.ArgumentParser(description="Load and perform initial checks on the dataset.")
    parser.add_argument("--input_file", type=str, default="data/ibm_dataset.csv", help="Path to the raw input CSV file.")
    parser.add_argument("--output_file", type=str, default="data/processed/01_IBM_acquired_data.csv", help="Path to save the loaded (and initially checked) CSV file.")
    parser.add_argument("--reference_data", default="data/ibm_dataset.csv", help="Path to reference data for drift detection")
    args = parser.parse_args()
    main(args)