"""
Dirty data synthesizer

This script introduces various realistic error patterns to create a dirty version of the dataset.
"""

import pandas as pd
import numpy as np

# Load the clean dataset
df_clean = pd.read_csv('data/ibm_dataset.csv')

# Create a copy to work on
df_dirty = df_clean.copy()
np.random.seed(42)


# 1. Simulate Missing & Partial Records


# 1a. For all JobLevel 1 employees, set 'EnvironmentSatisfaction' to NaN (simulate schema-driven missingness)
mask_entry_level = df_dirty['JobLevel'] == 1
df_dirty.loc[mask_entry_level, 'EnvironmentSatisfaction'] = np.nan

# 1b. For 5% of employees with JobLevel >= 3, set 'JobSatisfaction' and 'WorkLifeBalance' to NaN (simulate partial records)
mask_senior = df_dirty['JobLevel'] >= 3
senior_indices = df_dirty[mask_senior].sample(frac=0.05, random_state=24).index
df_dirty.loc[senior_indices, ['JobSatisfaction', 'WorkLifeBalance']] = np.nan


# 2. Inconsistent Formats & Typos in Categorical Fields


# 2a. Introduce department name typos/variants for 10% of each department
dept_typo_map = {
    'Sales': ['Sale', 'Sales ', 'Saless'],
    'Research & Development': ['R&D', 'Research and Dev', 'ResDev'],
    'Human Resources': ['HR', 'Human Resource', 'Hum Res'],
    'Marketing': ['Market', 'Mktg', 'Markting'],
    'Finance': ['Finanace', 'Fin', 'Finance '],
    'IT': ['Information Technology', 'It', 'I T']
}
for dept, variants in dept_typo_map.items():
    subset_idx = df_dirty[df_dirty['Department'] == dept].sample(frac=0.10, random_state=42).index
    replacements = np.random.choice(variants, size=len(subset_idx))
    df_dirty.loc[subset_idx, 'Department'] = replacements

# 2b. Change case/whitespace in 'MaritalStatus' for 5% of rows
ms_indices = df_dirty.sample(frac=0.05, random_state=51).index
df_dirty.loc[ms_indices, 'MaritalStatus'] = df_dirty.loc[ms_indices, 'MaritalStatus'].str.upper().str.strip()

# 2c. Introduce typos in 'Gender' for 2% of rows
gender_indices = df_dirty.sample(frac=0.02, random_state=63).index
gender_variants = ['M', 'F', 'Male', 'female', ' fem ', 'Mal']
for idx in gender_indices:
    df_dirty.at[idx, 'Gender'] = np.random.choice(gender_variants)

# 2d. Inconsistent 'OverTime' values for 3% of rows
ot_indices = df_dirty.sample(frac=0.03, random_state=17).index
for idx in ot_indices:
    val = df_dirty.at[idx, 'OverTime']
    if val == 'Yes':
        df_dirty.at[idx, 'OverTime'] = np.random.choice(['Y', 'yes', ' YES '])
    elif val == 'No':
        df_dirty.at[idx, 'OverTime'] = np.random.choice(['N', 'no', ' NO '])


# 3. Numeric Formatting & Unit Errors


# 3a. Swap comma/period in 'MonthlyIncome' for 4% of rows (simulate locale issues)
income_indices = df_dirty.sample(frac=0.04, random_state=33).index
for idx in income_indices:
    val = df_dirty.at[idx, 'MonthlyIncome']
    if pd.notnull(val):
        formatted = f"{val:,.2f}"
        df_dirty.at[idx, 'MonthlyIncome'] = formatted.replace(',', 'X').replace('.', ',').replace('X', '.')

# 3b. Multiply 5% of 'MonthlyIncome' by 1.2 (simulate currency/unit error)
mislabel_indices = df_dirty.sample(frac=0.05, random_state=44).index
for idx in mislabel_indices:
    val = df_dirty.at[idx, 'MonthlyIncome']
    if isinstance(val, (int, float)):
        df_dirty.at[idx, 'MonthlyIncome'] = val * 1.2

# 3c. Transpose digits in 'EmployeeNumber' for 2% of rows (simulate ID entry errors)
id_indices = df_dirty.sample(frac=0.02, random_state=55).index
for idx in id_indices:
    emp_id = str(df_dirty.at[idx, 'EmployeeNumber'])
    if len(emp_id) > 3:
        i = np.random.randint(1, len(emp_id) - 2)
        char_list = list(emp_id)
        char_list[i], char_list[i+1] = char_list[i+1], char_list[i]
        df_dirty.at[idx, 'EmployeeNumber'] = int(''.join(char_list))


# 4. Introduce Outliers & Noise in Numeric Fields


# 4a. Multiply 'Age' by 2.5–3.5 for 1% of rows (extreme outliers)
age_outlier_indices = df_dirty.sample(frac=0.01, random_state=71).index
df_dirty.loc[age_outlier_indices, 'Age'] = df_dirty.loc[age_outlier_indices, 'Age'] * np.random.uniform(2.5, 3.5, size=len(age_outlier_indices))

# 4b. Add Gaussian noise to 'YearsAtCompany' (σ=0.5), clip at 0
noise = np.random.normal(0, 0.5, len(df_dirty))
df_dirty['YearsAtCompany'] = df_dirty['YearsAtCompany'] + noise
df_dirty['YearsAtCompany'] = df_dirty['YearsAtCompany'].clip(lower=0)

# 4c. Replace 0.5% of 'TotalWorkingYears' with Pareto-distributed spikes (heavy-tailed outliers)
twy_indices = df_dirty.sample(frac=0.005, random_state=83).index
pareto_spikes = (np.random.pareto(a=2, size=len(twy_indices)) + 1) * df_dirty.loc[twy_indices, 'TotalWorkingYears'].median()
df_dirty.loc[twy_indices, 'TotalWorkingYears'] = pareto_spikes


# 5. Duplicates or Contradictory Records


# 5a. Add 3% exact duplicates, but change 'Department' and 'JobLevel' to create conflicts
dup_indices = df_dirty.sample(frac=0.03, random_state=91).index
duplicates = df_dirty.loc[dup_indices].copy()
dept_choices = list(dept_typo_map.keys())
for i, idx in enumerate(dup_indices):
    duplicates.at[idx, 'Department'] = np.random.choice(dept_choices)
    duplicates.at[idx, 'JobLevel'] = np.random.randint(1, 6)  # JobLevel assumed 1–5
df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)


# 6. Legacy or Deprecated Categorical Values


# 6a. Rename 10% of 'Research & Development' to 'R&D Lab'
mask_rd = df_dirty['Department'] == 'Research & Development'
rd_indices = df_dirty[mask_rd].sample(frac=0.10, random_state=101).index
df_dirty.loc[rd_indices, 'Department'] = 'R&D Lab'

# 6b. Rename 5% of 'Sales' to 'Sale Dept'
mask_sales = df_dirty['Department'] == 'Sales'
old_sale_indices = df_dirty[mask_sales].sample(frac=0.05, random_state=113).index
df_dirty.loc[old_sale_indices, 'Department'] = 'Sale Dept'


# 7. Invalid Values in Categorical Ratings


# 7a. Set 'EnvironmentSatisfaction' and 'JobInvolvement' to 0 or 5 (invalid) for 2% of rows
es_indices = df_dirty.sample(frac=0.02, random_state=127).index
df_dirty.loc[es_indices, 'EnvironmentSatisfaction'] = np.random.choice([0, 5], size=len(es_indices))
ji_indices = df_dirty.sample(frac=0.02, random_state=131).index
df_dirty.loc[ji_indices, 'JobInvolvement'] = np.random.choice([0, 5], size=len(ji_indices))


# 8. Contradictory Attrition Flags


# 8a. For 1% of active employees, set 'YearsSinceLastPromotion' > 'TotalWorkingYears' (logical contradiction)
active_no_indices = df_dirty[df_dirty['Attrition'] == 'No'].sample(frac=0.01, random_state=137).index
df_dirty.loc[active_no_indices, 'YearsSinceLastPromotion'] = df_dirty.loc[active_no_indices, 'TotalWorkingYears'] + np.random.randint(1, 5, size=len(active_no_indices))


# 9. Distribution Drift (Systematic Shifts)


print("Adding distribution drift to multiple numerical columns...")

# Define columns to drift and their multipliers
drift_columns = {
    'MonthlyIncome': 1.25,                # 25% increase in salaries
    'YearsSinceLastPromotion': 1.5,       # 50% longer since last promotion
    'Age': 1.08,                          # 8% older workforce
    'MonthlyRate': 0.85,                  # 15% decrease in monthly rates
    'DistanceFromHome': 1.4,              # 40% increase in commute distance
    'PercentSalaryHike': 0.7              # 30% smaller salary increases
}
# Apply drift to each column
for column, multiplier in drift_columns.items():
    if column not in df_dirty.columns:
        continue
    print(f"Applying drift to {column} (multiplier: {multiplier})")
    string_mask = df_dirty[column].apply(lambda x: isinstance(x, str))
    string_values = df_dirty.loc[string_mask, column].copy()
    df_dirty[column] = pd.to_numeric(df_dirty[column], errors='coerce') * multiplier
    df_dirty[column] = df_dirty[column].round(2)
    df_dirty.loc[string_mask, column] = string_values


# Save the dirty dataset

df_dirty.to_csv('data/ibm_dataset_dirty.csv', index=False)
print("Dirty dataset created using only the specified columns, with realistic error patterns.")
