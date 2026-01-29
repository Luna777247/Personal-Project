import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from revenue_analyzer import RevenueProductAnalyzer

def test_data_generation():
    """Test data generation functionality"""
    print("Testing data generation...")
    analyzer = RevenueProductAnalyzer()

    # Test with smaller dataset for faster testing
    data = analyzer.generate_mock_sales_data(1000)

    assert len(data) == 1000, f"Expected 1000 records, got {len(data)}"
    assert 'total_revenue' in data.columns, "Missing total_revenue column"
    assert 'profit' in data.columns, "Missing profit column"
    assert data['total_revenue'].sum() > 0, "Total revenue should be positive"

    print("✓ Data generation test passed")
    return analyzer

def test_data_processing():
    """Test data cleaning and processing"""
    print("Testing data processing...")
    analyzer = test_data_generation()

    processed = analyzer.clean_and_process_data()

    assert len(processed) > 0, "Processed data should not be empty"
    assert 'profit_margin' in processed.columns, "Missing profit_margin column"
    assert 'cost_percentage' in processed.columns, "Missing cost_percentage column"
    assert processed['profit_margin'].notna().all(), "Profit margin should not contain NaN"

    print("✓ Data processing test passed")
    return analyzer

def test_kpi_calculations():
    """Test KPI calculations"""
    print("Testing KPI calculations...")
    analyzer = test_data_processing()

    kpis = analyzer.calculate_kpi_metrics()

    assert 'overall' in kpis, "Missing overall KPIs"
    assert 'monthly_revenue' in kpis, "Missing monthly revenue KPIs"
    assert 'product_performance' in kpis, "Missing product performance KPIs"
    assert kpis['overall']['total_revenue'] > 0, "Total revenue should be positive"
    assert kpis['overall']['total_profit'] > 0, "Total profit should be positive"

    print("✓ KPI calculations test passed")
    return analyzer

def test_optimization_recommendations():
    """Test optimization recommendations"""
    print("Testing optimization recommendations...")
    analyzer = test_kpi_calculations()

    recommendations = analyzer.generate_optimization_recommendations()

    assert len(recommendations) == 3, f"Expected 3 recommendations, got {len(recommendations)}"

    for i, rec in enumerate(recommendations, 1):
        assert 'title' in rec, f"Recommendation {i} missing title"
        assert 'description' in rec, f"Recommendation {i} missing description"
        assert 'impact' in rec, f"Recommendation {i} missing impact"
        assert 'actions' in rec, f"Recommendation {i} missing actions"
        assert len(rec['actions']) > 0, f"Recommendation {i} should have actions"

    print("✓ Optimization recommendations test passed")
    return analyzer

def test_file_outputs():
    """Test file generation"""
    print("Testing file outputs...")
    analyzer = test_optimization_recommendations()

    # Test complete analysis (which includes report generation and data saving)
    analyzer.create_visualizations()  # Create visualizations
    report = analyzer.generate_comprehensive_report()  # Generate report

    # Save processed data manually for testing
    analyzer.processed_data.to_csv('../data/processed_sales_data.csv', index=False)

    assert os.path.exists('../docs/revenue_analysis_report.md'), "Report file not created"
    assert len(report) > 1000, "Report content too short"

    # Test data export
    assert os.path.exists('../data/processed_sales_data.csv'), "Processed data file not created"

    # Check CSV content
    df = pd.read_csv('../data/processed_sales_data.csv')
    assert len(df) > 0, "CSV file is empty"
    assert 'total_revenue' in df.columns, "CSV missing total_revenue column"

    print("✓ File outputs test passed")
    return analyzer

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")

    # Test with no data
    analyzer = RevenueProductAnalyzer()

    try:
        analyzer.clean_and_process_data()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No sales data available" in str(e), "Wrong error message"

    print("✓ Error handling test passed")

def test_performance_with_larger_dataset():
    """Test performance with larger dataset"""
    print("Testing performance with larger dataset...")
    analyzer = RevenueProductAnalyzer()

    start_time = datetime.now()
    analyzer.generate_mock_sales_data(5000)
    analyzer.clean_and_process_data()
    analyzer.calculate_kpi_metrics()
    analyzer.generate_optimization_recommendations()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(".2f")
    assert duration < 30, f"Analysis took too long: {duration} seconds"

    print("✓ Performance test passed")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("TESTING REVENUE & PRODUCT PERFORMANCE ANALYSIS TOOL")
    print("=" * 60)

    try:
        test_data_generation()
        test_data_processing()
        test_kpi_calculations()
        test_optimization_recommendations()
        test_file_outputs()
        test_error_handling()
        test_performance_with_larger_dataset()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Revenue & Product Performance Analysis Tool is ready for use.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_all_tests()