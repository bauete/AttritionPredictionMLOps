"""
McNemar's Test for Model Comparison

This script performs McNemar's test to compare the performance of two classification models
"""
import argparse
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar_test(y_true_path, y_pred_a_path, y_pred_b_path, model_a_name="Model A", model_b_name="Model B"):
    # Load the data (ground truth and predictions from both models)
    y_true = np.load(y_true_path)
    y_pred_a = np.load(y_pred_a_path)
    y_pred_b = np.load(y_pred_b_path)

    # Sanity check: Ensure all arrays are the same length
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("Input arrays must have the same length.")

    # Determine which predictions are correct for each model
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Build the 2x2 contingency table:
    # n11: both correct, n10: A correct, B wrong, n01: A wrong, B correct, n00: both wrong
    n11 = np.sum((correct_a == True) & (correct_b == True))    # Both correct
    n10 = np.sum((correct_a == True) & (correct_b == False))   # A correct, B wrong
    n01 = np.sum((correct_a == False) & (correct_b == True))   # A wrong, B correct
    n00 = np.sum((correct_a == False) & (correct_b == False))  # Both wrong
    
    table = [[n11, n10],
             [n01, n00]]

    # Print the contingency table for reference
    print("Contingency Table:")
    print(f"                 {model_b_name}")
    print(f"               Correct  |  Wrong")
    print(f"------------------------------------")
    print(f"{model_a_name: <12} Correct | {n11: >8} | {n10: >6}")
    print(f"{model_a_name: <12} Wrong   | {n01: >8} | {n00: >6}")
    print("\n")

    # Perform McNemar's test (exact version)
    # Focuses on the discordant pairs (n10 and n01)
    result = mcnemar(table, exact=True)

    # Print test statistic and p-value
    print("McNemar's Test Results:")
    print(f"Statistic (Chi-squared): {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.4f}")

    # Interpret the result
    if result.pvalue < 0.05:
        if n10 > n01:
            print(f"\nThe difference is STATISTICALLY SIGNIFICANT. {model_b_name} made more errors where {model_a_name} was correct.")
        else:
            print(f"\nThe difference is STATISTICALLY SIGNIFICANT. {model_a_name} made more errors where {model_b_name} was correct.")
    else:
        print("\nThere is NO STATISTICALLY SIGNIFICANT difference in the models' error rates.")

if __name__ == '__main__':
    # Parse command-line arguments for file paths and model names
    parser = argparse.ArgumentParser(description="Perform McNemar's test to compare two models.")
    parser.add_argument("--true_labels", required=True, help="Path to the .npy file with ground truth labels.")
    parser.add_argument("--model_a_preds", required=True, help="Path to the .npy file with predictions from Model A.")
    parser.add_argument("--model_b_preds", required=True, help="Path to the .npy file with predictions from Model B.")
    parser.add_argument("--model_a_name", default="Baseline", help="Name for Model A.")
    parser.add_argument("--model_b_name", default="MLOps", help="Name for Model B.")
    args = parser.parse_args()

    # Run the test and print results
    run_mcnemar_test(args.true_labels, args.model_a_preds, args.model_b_preds, args.model_a_name, args.model_b_name)


