# This file will contain code for data cleaning, preprocessing, and feature engineering.

import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    This function takes a raw dataframe and performs the following preprocessing steps:
    1. Drops 'Churn Category' and 'Churn Reason' columns.
    2. Fills missing values in 'Internet Type' with 'No Internet Service'.
    3. Converts 'Total Charges' to a numeric type, filling NaNs with 0.
    4. Performs one-hot encoding on categorical variables.
    5. Drops the original categorical columns.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """

    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()

    # 1. Drop unnecessary columns
    df_processed = df_processed.drop(columns=['Churn Category', 'Churn Reason', 'Customer ID', 'Lat Long'], errors='ignore')

    # 2. Fill missing values
    # Based on EDA, 'Internet Type' has missing values. We can assume these customers do not have internet service.
    if 'Internet Type' in df_processed.columns:
        df_processed['Internet Type'].fillna('No Internet Service', inplace=True)

    # 3. Convert 'Total Charges' to numeric
    if 'Total Charges' in df_processed.columns:
        df_processed['Total Charges'] = pd.to_numeric(df_processed['Total Charges'], errors='coerce')
        df_processed['Total Charges'].fillna(0, inplace=True)

    # 4. Identify categorical columns for one-hot encoding
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

    # Also include some numerical columns that are actually categorical
    # For simplicity, we will treat all object columns as categorical for now.
    # We will exclude the target variable 'Churn' if it's in the list, as we don't want to encode it.
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')

    # 5. Perform one-hot encoding
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True, dummy_na=False)

    return df_processed
