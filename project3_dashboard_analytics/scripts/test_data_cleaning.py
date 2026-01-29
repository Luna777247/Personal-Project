import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_cleaning import load_data, clean_data, validate_data, save_cleaned_data

def test_data_cleaning():
    """Test the data cleaning functionality"""
    # Load sample data
    input_file = "D:/project/Personal Project/project3_dashboard_analytics/data/raw/sample_sales_data.csv"
    output_file = "D:/project/Personal Project/project3_dashboard_analytics/data/processed/cleaned_data.csv"

    print("Loading data...")
    df = load_data(input_file)
    print(f"Loaded {len(df)} rows")

    print("Cleaning data...")
    cleaned_df = clean_data(df)
    print(f"Cleaned data has {len(cleaned_df)} rows")

    print("Validating data...")
    issues = validate_data(cleaned_df)
    if issues:
        print("Issues found:", issues)
    else:
        print("No validation issues found")

    print("Saving cleaned data...")
    save_cleaned_data(cleaned_df, output_file)

    print("Data cleaning test completed!")

if __name__ == "__main__":
    test_data_cleaning()