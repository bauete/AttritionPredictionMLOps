import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

def preprocess_kpmg_data(input_file_path, output_file_path):
    """
    Preprocess KPMG dataset: Identify unique employees, combine their multiple records,
    and transform features to align with an IBM-style dataset.
    """
    # Try reading the CSV file with different delimiters if needed
    try:
        try:
            df = pd.read_csv(input_file_path)
        except pd.errors.ParserError:
            try:
                df = pd.read_csv(input_file_path, sep=';')
            except pd.errors.ParserError:
                df = pd.read_csv(input_file_path, sep='\t')
    except Exception as e:
        print(f"Error reading CSV file '{input_file_path}': {e}")
        return pd.DataFrame()

    # Convert relevant columns to string to avoid mixed type issues
    str_cols = [
        'employee_id', 'Date Of Birth', 'Rating Label (Picklist Label)', 
        'Rating Label (External Code)', 'Readiness for Promotion (Picklist Label)',
        'Current Salary (Adjusted) (Latest Month)', 'Number of previous employers',
        'Pay Grade (Pay Grade Name)', 'Function (Label)', 'Local Job Level (Local Job Level Title)',
        'Employee Status (Picklist Label)', 'Primary Reason for Leaving (Primary Reason)',
        'Sub Function (Sub Function Name)', 'Gender', 'Leeftijd', 'Marital Status (Picklist Label)'
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Ensure the unique employee ID column exists
    actual_employee_id_col = 'employee_id'
    if actual_employee_id_col not in df.columns:
        print(f"Error: Employee ID column '{actual_employee_id_col}' not found in the input CSV '{input_file_path}'.")
        return pd.DataFrame()

    result_df = df.copy()
    reference_date = pd.Timestamp(datetime.now().date())

    # --- Create KPMG-suffixed features for mapping ---
    result_df['Age_KPMG'] = result_df.get('Leeftijd')
    result_df['Gender_KPMG'] = result_df.get('Gender')
    result_df['MaritalStatus_KPMG'] = result_df.get('Marital Status (Picklist Label)')
    result_df['DistanceFromHome_KPMG'] = pd.to_numeric(result_df.get('Total Kilometers'), errors='coerce')
    
    # Parse date columns
    date_cols_kpmg_sources = {
        'OriginalHireDate_KPMG': 'Original Hire Date',
        'MostRecentHireDate_KPMG': 'Most Recent Hire Date',
        'JobEntryDate_KPMG': 'Job Entry Date',
        'LastPromotionDate_KPMG': 'Last Promotion Date',
        'DepartureDate_KPMG': 'Departure Date'
    }
    for kpmg_col, source_col in date_cols_kpmg_sources.items():
        if source_col in result_df.columns:
            result_df[kpmg_col] = pd.to_datetime(result_df[source_col], format='%d-%m-%Y', errors='coerce')
        else:
            result_df[kpmg_col] = pd.NaT

    # Map other KPMG columns to intermediate features
    result_df['Department_KPMG'] = result_df.get('Function (Label)')
    result_df['JobRole_KPMG_Title'] = result_df.get('Local Job Level (Local Job Level Title)')
    result_df['JobRole_Source_KPMG'] = result_df.get('Sub Function (Sub Function Name)')
    result_df['PerformanceRating_KPMG_Code'] = result_df.get('Rating Label (External Code)')
    result_df['MonthlyIncome_Source_KPMG'] = result_df.get('Pay Grade (Pay Grade Name)')
    result_df['NumCompaniesWorked_KPMG'] = pd.to_numeric(result_df.get('Number of previous employers'), errors='coerce')
    result_df['EmployeeStatus_KPMG'] = result_df.get('Employee Status (Picklist Label)')
    result_df['ReasonForLeaving_KPMG'] = result_df.get('Primary Reason for Leaving (Primary Reason)')
    result_df['EmployeeID_KPMG_Source'] = result_df[actual_employee_id_col]

    # --- Feature Mapping and Engineering to IBM Dataset ---
    # Age
    result_df['Age'] = pd.to_numeric(result_df['Age_KPMG'], errors='coerce')

    # Gender mapping
    gender_map = {'F': 'Female', 'M': 'Male', 'Man': 'Male', 'Vrouw': 'Female', 'Female': 'Female', 'Male': 'Male'}
    result_df['Gender'] = result_df['Gender_KPMG'].astype(str).map(gender_map).fillna(result_df['Gender_KPMG'])

    # MaritalStatus mapping
    marital_map = {
        'Single': 'Single', 'Alleenstaand': 'Single',
        'Married': 'Married', 'Gehuwd/Geregistreerd partnerschap': 'Married',
        'Divorced': 'Divorced', 'Gescheiden': 'Divorced',        
        'Widowed': 'Single', 'Weduwe/weduwnaar': 'Single',        
        'Other' : 'Single','Unknown': 'Single', '<NA>': 'Single', 'nan': 'Single', '': 'Single'
    }
    result_df['MaritalStatus'] = result_df['MaritalStatus_KPMG'].astype(str).map(marital_map).fillna('Single')

    # DistanceFromHome
    result_df['DistanceFromHome'] = result_df['DistanceFromHome_KPMG']

    # Date-based features: calculate years from reference date
    if 'MostRecentHireDate_KPMG' in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df['MostRecentHireDate_KPMG']):
        result_df['YearsAtCompany'] = np.ceil((reference_date - result_df['MostRecentHireDate_KPMG']).dt.days / 365.25)
    else:
        result_df['YearsAtCompany'] = np.nan
    if 'OriginalHireDate_KPMG' in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df['OriginalHireDate_KPMG']):
        result_df['TotalWorkingYears'] = np.ceil((reference_date - result_df['OriginalHireDate_KPMG']).dt.days / 365.25)
    else:
        result_df['TotalWorkingYears'] = np.nan
    if 'JobEntryDate_KPMG' in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df['JobEntryDate_KPMG']):
        result_df['YearsInCurrentRole'] = np.ceil((reference_date - result_df['JobEntryDate_KPMG']).dt.days / 365.25)
    else:
        result_df['YearsInCurrentRole'] = np.nan
    if 'LastPromotionDate_KPMG' in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df['LastPromotionDate_KPMG']):
        result_df['YearsSinceLastPromotion'] = np.ceil((reference_date - result_df['LastPromotionDate_KPMG']).dt.days / 365.25)
    else:
        result_df['YearsSinceLastPromotion'] = np.nan
    result_df['YearsWithCurrManager'] = np.nan

    # Convert year features to Int64
    cols_to_convert_to_int = ['YearsAtCompany', 'TotalWorkingYears', 'YearsInCurrentRole', 'YearsSinceLastPromotion']
    for col in cols_to_convert_to_int:
        if col in result_df.columns:
            result_df[col] = result_df[col].astype('Int64')

    # Department mapping
    department_map_kpmg = { 
        'Assurance': 'Other', 
        'Advisory': 'Sales',
        'Business Services': 'Human Resources', 
        'Tax & Legal': 'Other', 
        'KPMG INT (non NL)': 'Other', 
        'KPMG Meijburg': 'Other',
        'default': 'Other' 
    }
    result_df['Department'] = result_df['Department_KPMG'].astype(str).map(lambda x: department_map_kpmg.get(x, department_map_kpmg['default'])).fillna('Other')

    # JobLevel mapping
    job_level_mapping_kpmg = {
        'Assurance Associate HBO': 1, 'Assurance Associate WO': 1,
        'Consultant/Executive': 1, 'Specialist (BS)': 1, 'Other': 1, 'Associate': 1,
        'Sr. Consultant/Associate': 2, 'Sr. Assurance Associate': 2, 'Senior Associate': 2,
        'Manager': 3, 'Teamleader/Advisor (BS)': 3,
        'Sr. Manager/Assoc. Director': 4, 'Manager/Sr. Adv. (BS)': 4,
        'Advisory Associate Director FUS': 4, 'Senior Manager': 4,
        'Director (BS)': 5, 'Director': 5, 'Non Equity Partner': 5,
        'INT\'L Grade 6': 5, 'INT\'L Grade 5': 4,
    }
    result_df['JobLevel'] = result_df['JobRole_KPMG_Title'].astype(str).map(job_level_mapping_kpmg)
    fill_job_level = result_df['JobLevel'].median() if pd.notna(result_df['JobLevel'].median()) else 2 
    result_df['JobLevel'] = pd.to_numeric(result_df['JobLevel'], errors='coerce').fillna(fill_job_level)

    # JobRole placeholder (not available in KPMG)
    result_df['JobRole'] = 'Other'

    # PerformanceRating mapping
    performance_map_kpmg = {
        '1': 4, '2': 3, '3': 2, '4': 1, '0': 2, '6': 1, '8': 1, 
        '1.0': 4, '2.0': 3, '3.0': 2, '4.0': 1, '0.0': 2, '6.0': 1, '8.0': 1 
    }
    result_df['PerformanceRating'] = result_df['PerformanceRating_KPMG_Code'].astype(str).str.split('.').str[0].map(performance_map_kpmg)
    fill_perf_rating = result_df['PerformanceRating'].median() if pd.notna(result_df['PerformanceRating'].median()) else 3
    result_df['PerformanceRating'] = pd.to_numeric(result_df['PerformanceRating'], errors='coerce').fillna(fill_perf_rating)
    result_df['PerformanceRating'] = result_df['PerformanceRating'].astype('Int64')

    # MonthlyIncome binning based on pay grade
    ordered_titles = [
        'Working Intern', 'Trainee HBO', 'Trainee WO', 'Assistant', 
        'Junior Assurance Associate', 'Assurance Associate HBO', 
        'Assurance Associate WO', 'Associate', 'Consultant HBO', 'Consultant WO',
        'Senior Assurance Associate', 'Senior Consultant', 'Senior', 'Supervisor',
        'Assistant Manager', 'INT\'L Grade 2', 'INT\'L Grade 3', 'INT\'L Grade 3T',
        'Grade 4', 'INT\'L Grade 4', 'INT\'L Grade 4T', 'Grade 5', 'INT\'L Grade 5',
        'Grade 6', 'Grade 6 FUS', 'INT\'L Grade 6', 'Grade 7', 'Grade 7 FUS', 
        'INT\'L Grade 7', 'Grade 8', 'Manager', 'Grade 9', 'Grade 9 FUS',
        'Senior Manager', 'Senior Manager FUS', 'Associate Director', 
        'Advisory Associate Director FUS', 'Grade 10', 'Director', 'Grade 11', 
        'Grade 11 FUS', 'Non Equity Partner', 'Non Equity Partner FUS', 'Grade 12',
        'Partner/Director FUS', 'Partner', 'Grade 13', 'Executive'
    ]
    title_to_rank = {title: i for i, title in enumerate(ordered_titles)}
    if 'MonthlyIncome_Source_KPMG' in result_df.columns:
        result_df['PayGradeRank'] = result_df['MonthlyIncome_Source_KPMG'].map(title_to_rank)
        if not result_df['PayGradeRank'].empty and result_df['PayGradeRank'].isnull().any():
            median_rank = result_df['PayGradeRank'].median()
            result_df['PayGradeRank'] = result_df['PayGradeRank'].fillna(median_rank)
        if 'PayGradeRank' in result_df.columns and not result_df['PayGradeRank'].dropna().empty:
            try:
                result_df['MonthlyIncome'] = pd.qcut(result_df['PayGradeRank'], q=10, labels=False, duplicates='drop') + 1
            except ValueError:
                num_unique_ranks = result_df['PayGradeRank'].nunique()
                if num_unique_ranks > 1:
                    result_df['MonthlyIncome'] = pd.qcut(result_df['PayGradeRank'], q=min(10, num_unique_ranks), labels=False, duplicates='drop') + 1
                else:
                    result_df['MonthlyIncome'] = 5
        else:
            result_df['MonthlyIncome'] = 5
        if 'PayGradeRank' in result_df.columns:
            result_df.drop(columns=['PayGradeRank'], inplace=True)
    else:
        result_df['MonthlyIncome'] = np.nan
    result_df['MonthlyIncome'] = pd.to_numeric(result_df['MonthlyIncome'], errors='coerce').fillna(5).astype('Int64')

    # NumCompaniesWorked
    result_df['NumCompaniesWorked'] = result_df['NumCompaniesWorked_KPMG']

    # Attrition mapping
    result_df['Attrition'] = 1
    inactive_statuses = ['Inactive', 'Terminated', 'Former', 'Left', 'Resigned', 'Uit dienst', 'uit dienst']
    if 'EmployeeStatus_KPMG' in result_df.columns:
        result_df.loc[result_df['EmployeeStatus_KPMG'].astype(str).str.lower().isin([s.lower() for s in inactive_statuses]), 'Attrition'] = 0
    if 'DepartureDate_KPMG' in result_df.columns:
        result_df.loc[result_df['DepartureDate_KPMG'].notna(), 'Attrition'] = 'Yes'
    if 'ReasonForLeaving_KPMG' in result_df.columns:
        result_df.loc[result_df['ReasonForLeaving_KPMG'].notna() & (result_df['ReasonForLeaving_KPMG'].astype(str).str.lower() != 'nan') & (result_df['ReasonForLeaving_KPMG'].astype(str).str.strip() != ''), 'Attrition'] = 'Yes'
    result_df['Attrition'] = result_df['Attrition'].map({'Yes': 1, 'No': 0}).fillna(0)

    # EmployeeNumber
    result_df['EmployeeNumber'] = result_df['EmployeeID_KPMG_Source']

    # --- Define IBM Features for Output ---
    TARGET_IBM_FEATURES_PLUS_ATTRITION = [
        'Age', 'Department', 'DistanceFromHome', 'EmployeeNumber', 
        'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 
        'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'OverTime', 'PerformanceRating', 
        'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
        'YearsSinceLastPromotion', 'Attrition'
    ]
    final_df = pd.DataFrame(columns=TARGET_IBM_FEATURES_PLUS_ATTRITION)
    for col_name in TARGET_IBM_FEATURES_PLUS_ATTRITION:
        if col_name in result_df.columns:
            final_df[col_name] = result_df[col_name]
        else:
            # Fill missing IBM features with sensible defaults
            if col_name == 'EnvironmentSatisfaction': final_df[col_name] = 3 
            elif col_name == 'JobInvolvement': final_df[col_name] = 3
            elif col_name == 'JobSatisfaction': final_df[col_name] = 3
            elif col_name == 'OverTime': final_df[col_name] = 'No' 
            elif col_name == 'WorkLifeBalance': final_df[col_name] = 3
            else:
                final_df[col_name] = np.nan 
            print(f"Warning: IBM feature '{col_name}' not found/derived. Added as default/NaN column.")
            
    # Type conversions for final_df
    int_cols_ibm = [
        'Age', 'DistanceFromHome', 'JobLevel', 'PerformanceRating', 
        'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'WorkLifeBalance', 'Attrition', 'MonthlyIncome',
        'YearsAtCompany', 'TotalWorkingYears', 'YearsInCurrentRole', 'YearsSinceLastPromotion'
    ]
    for col in int_cols_ibm:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').astype('Int64')
    if 'EmployeeNumber' in final_df.columns:
        final_df['EmployeeNumber'] = final_df['EmployeeNumber'].astype(str)

    # --- Final Checks and Output ---
    yes_count = (final_df['Attrition'] == 1).sum()
    total_count = len(final_df)
    if total_count > 0:
        print(f"\nAttrition in processed KPMG data: {yes_count} 'Yes' ({int(yes_count)/total_count:.1%}) out of {total_count} employees")
    else:
        print("No data processed.")
    
    # Save the processed data
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    final_df.to_csv(output_file_path, index=False)
    print(f"Processed KPMG data saved to {output_file_path}")
        
    return final_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess KPMG dataset.")
    parser.add_argument("--input_file", type=str, default="data/processed/01_KPMG_acquired_data.csv", help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, default="data/processed/02_KPMG_preprocessed_data.csv", help="Path to save the preprocessed CSV file.")
    parser.add_argument("--preprocessor_artifact_path", type=str, default="artifacts/KPMG_preprocessor.pkl", help="Path to save the preprocessor object.")
    
    args = parser.parse_args()
    
    print(f"Starting KPMG data preprocessing from: {args.input_file}")
    processed_df = preprocess_kpmg_data(args.input_file, args.output_file)
    
    if not processed_df.empty:
        print(f"\nSuccessfully processed {len(processed_df)} employees.")
    else:
        print("No data was processed or an error occurred during preprocessing.")

