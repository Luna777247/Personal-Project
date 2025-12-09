import pandas as pd
import json
from scripts.data_cleaning import load_data, clean_data
from scripts.kpi_calculation import calculate_all_kpis

def prepare_data_for_dashboard(input_path, output_dir):
    """Prepare cleaned data and KPIs for dashboard consumption"""
    
    # Load and clean data
    raw_df = load_data(input_path)
    cleaned_df = clean_data(raw_df)
    
    # Calculate KPIs
    kpis = calculate_all_kpis(cleaned_df)
    
    # Save cleaned data
    cleaned_df.to_csv(f"{output_dir}/cleaned_data.csv", index=False)
    
    # Save KPIs as JSON for dashboard
    with open(f"{output_dir}/kpis.json", 'w') as f:
        json.dump(kpis, f, indent=2, default=str)
    
    # Create summary tables for dashboard
    if 'date' in cleaned_df.columns:
        # Daily summary
        daily_summary = cleaned_df.groupby(pd.to_datetime(cleaned_df['date']).dt.date).agg({
            'revenue': 'sum',
            'user_id': 'nunique',
            'order_id': 'count' if 'order_id' in cleaned_df.columns else 'size'
        }).reset_index()
        daily_summary.columns = ['date', 'daily_revenue', 'unique_users', 'daily_orders']
        daily_summary.to_csv(f"{output_dir}/daily_summary.csv", index=False)
    
    if 'category' in cleaned_df.columns:
        # Category summary
        category_summary = cleaned_df.groupby('category').agg({
            'revenue': 'sum',
            'user_id': 'nunique'
        }).reset_index()
        category_summary.columns = ['category', 'category_revenue', 'category_users']
        category_summary.to_csv(f"{output_dir}/category_summary.csv", index=False)
    
    print(f"Data prepared for dashboard in {output_dir}")

if __name__ == "__main__":
    # Example usage
    input_file = "data/raw/sample_sales_data.csv"
    output_dir = "data/processed"
    prepare_data_for_dashboard(input_file, output_dir)