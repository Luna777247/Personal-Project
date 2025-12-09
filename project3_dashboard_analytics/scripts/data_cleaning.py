import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """Load data from various formats"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def clean_data(df):
    """Clean and preprocess data"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna(subset=['user_id', 'revenue'])  # Critical columns
    
    # Standardize date formats
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Clean text fields
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].str.strip().str.lower()
    
    # Remove outliers (IQR method for revenue)
    if 'revenue' in df.columns:
        Q1 = df['revenue'].quantile(0.25)
        Q3 = df['revenue'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['revenue'] < (Q1 - 1.5 * IQR)) | (df['revenue'] > (Q3 + 1.5 * IQR)))]
    
    return df

def validate_data(df):
    """Data validation checks"""
    issues = []
    
    # Check for negative revenues
    if 'revenue' in df.columns and (df['revenue'] < 0).any():
        issues.append("Negative revenue values found")
    
    # Check date ranges
    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        if min_date < pd.Timestamp('2020-01-01') or max_date > pd.Timestamp.now():
            issues.append("Date range seems unrealistic")
    
    return issues

def save_cleaned_data(df, output_path):
    """Save cleaned data"""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")