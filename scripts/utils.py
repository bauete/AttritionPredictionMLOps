# scripts/utils.py
import pandas as pd
import joblib
import os

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def save_data(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Saving data to {file_path}...")
    df.to_csv(file_path, index=False)

def save_model_or_object(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Saving object to {file_path}...")
    joblib.dump(obj, file_path)

def load_model_or_object(file_path):
    print(f"Loading object from {file_path}...")
    return joblib.load(file_path)

