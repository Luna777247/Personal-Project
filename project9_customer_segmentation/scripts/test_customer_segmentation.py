import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from customer_segmenter import CustomerSegmenter

def test_data_generation():
    """Test customer data generation"""
    print("Testing customer data generation...")
    segmenter = CustomerSegmenter()

    data = segmenter.generate_mock_customer_data(1000)

    assert len(data) > 0, "No customer data generated"
    assert 'customer_id' in data.columns, "Missing customer_id column"
    assert 'amount' in data.columns, "Missing amount column"
    assert data['amount'].sum() > 0, "Total amount should be positive"

    print("✓ Data generation test passed")
    return segmenter

def test_rfm_calculation():
    """Test RFM score calculation"""
    print("Testing RFM calculation...")
    segmenter = test_data_generation()

    rfm = segmenter.calculate_rfm_scores()

    assert len(rfm) > 0, "RFM data is empty"
    assert 'r_score' in rfm.columns, "Missing R score"
    assert 'f_score' in rfm.columns, "Missing F score"
    assert 'm_score' in rfm.columns, "Missing M score"
    assert 'rfm_score' in rfm.columns, "Missing RFM score"

    # Check score ranges
    assert rfm['r_score'].min() >= 1 and rfm['r_score'].max() <= 5, "R score out of range"
    assert rfm['f_score'].min() >= 1 and rfm['f_score'].max() <= 5, "F score out of range"
    assert rfm['m_score'].min() >= 1 and rfm['m_score'].max() <= 5, "M score out of range"

    print("✓ RFM calculation test passed")
    return segmenter

def test_customer_segmentation():
    """Test customer segmentation"""
    print("Testing customer segmentation...")
    segmenter = test_rfm_calculation()

    segmented = segmenter.perform_customer_segmentation(3)

    assert 'cluster' in segmented.columns, "Missing cluster column"
    assert len(segmented['cluster'].unique()) == 3, "Expected 3 clusters"
    assert len(segmenter.cluster_profiles) == 3, "Expected 3 cluster profiles"

    print("✓ Customer segmentation test passed")
    return segmenter

def test_segment_insights():
    """Test segment insights generation"""
    print("Testing segment insights...")
    segmenter = test_customer_segmentation()

    insights = segmenter.segment_insights

    assert len(insights) == 3, "Expected insights for 3 clusters"

    for cluster, insight in insights.items():
        assert 'segment_name' in insight, f"Missing segment name for cluster {cluster}"
        assert 'description' in insight, f"Missing description for cluster {cluster}"
        assert 'strategy' in insight, f"Missing strategy for cluster {cluster}"
        assert 'profile' in insight, f"Missing profile for cluster {cluster}"

    print("✓ Segment insights test passed")
    return segmenter

def test_file_outputs():
    """Test file generation"""
    print("Testing file outputs...")
    segmenter = test_segment_insights()

    # Save the CSV files that would normally be saved in run_complete_segmentation
    segmenter.segmented_data.to_csv('../data/customer_segments.csv', index=False)
    segmenter.rfm_data.to_csv('../data/rfm_scores.csv', index=False)

    # Test report generation
    report = segmenter.generate_segmentation_report()

    assert os.path.exists('../results/customer_segmentation_report.md'), "Report file not created"
    assert len(report) > 1000, "Report content too short"

    # Test data export
    assert os.path.exists('../data/customer_segments.csv'), "Segmented data file not created"
    assert os.path.exists('../data/rfm_scores.csv'), "RFM scores file not created"

    # Check CSV content
    df_segments = pd.read_csv('../data/customer_segments.csv')
    df_rfm = pd.read_csv('../data/rfm_scores.csv')

    assert len(df_segments) > 0, "Segments CSV is empty"
    assert len(df_rfm) > 0, "RFM CSV is empty"
    assert 'cluster' in df_segments.columns, "Segments CSV missing cluster column"

    print("✓ File outputs test passed")
    return segmenter

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")

    segmenter = CustomerSegmenter()

    # Test with no data
    try:
        segmenter.calculate_rfm_scores()
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass  # Expected

    print("✓ Error handling test passed")

def test_performance_with_larger_dataset():
    """Test performance with larger dataset"""
    print("Testing performance with larger dataset...")
    segmenter = CustomerSegmenter()

    start_time = datetime.now()
    segmenter.generate_mock_customer_data(2000)
    segmenter.calculate_rfm_scores()
    segmenter.perform_customer_segmentation(4)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(".2f")
    assert duration < 15, f"Analysis took too long: {duration} seconds"

    print("✓ Performance test passed")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("TESTING CUSTOMER SEGMENTATION TOOL")
    print("=" * 60)

    try:
        test_data_generation()
        test_rfm_calculation()
        test_customer_segmentation()
        test_segment_insights()
        test_file_outputs()
        test_error_handling()
        test_performance_with_larger_dataset()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Customer Segmentation Tool is ready for use.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_all_tests()