import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kpi_calculation import calculate_all_kpis
from data_cleaning import load_data

def test_kpi_calculation():
    """Test the KPI calculation functionality"""
    # Load cleaned data
    input_file = "D:/project/Personal Project/project3_dashboard_analytics/data/processed/cleaned_data.csv"

    print("Loading cleaned data...")
    df = load_data(input_file)
    print(f"Loaded {len(df)} rows")

    print("Calculating KPIs...")
    kpis = calculate_all_kpis(df)

    print("Revenue KPIs:")
    for key, value in kpis['revenue_kpis'].items():
        print(f"  {key}: {value}")

    print("\nUser KPIs:")
    for key, value in kpis['user_kpis'].items():
        print(f"  {key}: {value}")

    print("\nGrowth KPIs:")
    for key, value in kpis['growth_kpis'].items():
        print(f"  {key}: {value}")

    print("KPI calculation test completed!")

if __name__ == "__main__":
    test_kpi_calculation()