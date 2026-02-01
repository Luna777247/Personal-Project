import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from financial_analyzer import FinancialAnalyzer

def test_data_generation():
    """Test financial data generation"""
    print("Testing financial data generation...")
    analyzer = FinancialAnalyzer()

    data = analyzer.generate_mock_financial_data(2)

    assert len(data) == 8, f"Expected 8 quarterly records, got {len(data)}"
    assert 'revenue' in data.columns, "Missing revenue column"
    assert 'net_profit' in data.columns, "Missing net_profit column"
    assert data['revenue'].sum() > 0, "Total revenue should be positive"

    print("✓ Data generation test passed")
    return analyzer

def test_financial_ratios():
    """Test financial ratios calculation"""
    print("Testing financial ratios calculation...")
    analyzer = test_data_generation()

    ratios = analyzer.calculate_financial_ratios()

    assert 'profitability' in ratios, "Missing profitability ratios"
    assert 'liquidity' in ratios, "Missing liquidity ratios"
    assert 'leverage' in ratios, "Missing leverage ratios"
    assert ratios['profitability']['avg_net_margin'] > 0, "Net margin should be positive"

    print("✓ Financial ratios test passed")
    return analyzer

def test_growth_analysis():
    """Test growth rate analysis"""
    print("Testing growth analysis...")
    analyzer = test_financial_ratios()

    growth = analyzer.analyze_growth_rates()

    assert 'quarterly' in growth, "Missing quarterly growth data"
    assert 'cagr' in growth, "Missing CAGR calculations"
    assert 'revenue_cagr' in growth['cagr'], "Missing revenue CAGR"

    print("✓ Growth analysis test passed")
    return analyzer

def test_financial_projections():
    """Test financial projections"""
    print("Testing financial projections...")
    analyzer = test_growth_analysis()

    projections = analyzer.generate_financial_projections(2)

    assert 'projections' in projections, "Missing projections data"
    assert len(projections['projections']) == 2, "Expected 2 projection periods"
    assert 'projected_revenue' in projections['projections'][0], "Missing projected revenue"

    print("✓ Financial projections test passed")
    return analyzer

def test_file_outputs():
    """Test file generation"""
    print("Testing file outputs...")
    analyzer = test_financial_projections()

    # Test PDF report generation
    analyzer.generate_pdf_report()

    # Save processed data manually for testing
    analyzer.processed_data.to_csv('../data/financial_data_processed.csv', index=False)

    assert os.path.exists('../reports/financial_analysis_report.pdf'), "PDF report not created"

    # Test data export
    assert os.path.exists('../data/financial_data_processed.csv'), "Processed data file not created"

    # Check CSV content
    df = pd.read_csv('../data/financial_data_processed.csv')
    assert len(df) > 0, "CSV file is empty"
    assert 'revenue' in df.columns, "CSV missing revenue column"

    print("✓ File outputs test passed")
    return analyzer

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")

    analyzer = FinancialAnalyzer()

    # Test with no data
    try:
        analyzer.calculate_financial_ratios()
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass  # Expected

    print("✓ Error handling test passed")

def test_comprehensive_workflow():
    """Test complete workflow with smaller dataset"""
    print("Testing comprehensive workflow...")
    analyzer = FinancialAnalyzer()

    start_time = datetime.now()

    analyzer.generate_mock_financial_data(2)
    analyzer.calculate_financial_ratios()
    analyzer.analyze_growth_rates()
    analyzer.generate_financial_projections(2)
    analyzer.create_financial_charts()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(".2f")
    assert duration < 10, f"Analysis took too long: {duration} seconds"

    print("✓ Comprehensive workflow test passed")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("TESTING BASIC FINANCIAL DATA ANALYSIS TOOL")
    print("=" * 60)

    try:
        test_data_generation()
        test_financial_ratios()
        test_growth_analysis()
        test_financial_projections()
        test_file_outputs()
        test_error_handling()
        test_comprehensive_workflow()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Basic Financial Data Analysis Tool is ready for use.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_all_tests()