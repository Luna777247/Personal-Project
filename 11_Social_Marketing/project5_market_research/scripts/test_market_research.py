#!/usr/bin/env python3
"""
Test script for Market Research Analysis Tool
"""

import os
import sys
import pandas as pd
from market_research_analyzer import MarketResearchAnalyzer

def test_market_research_tool():
    """Test the market research analysis functionality"""
    print("Testing Market Research Analysis Tool...")

    # Initialize analyzer
    analyzer = MarketResearchAnalyzer()

    # Test data generation
    print("✓ Testing data generation...")
    analyzer.generate_mock_survey_data(50)
    assert len(analyzer.survey_df) == 50, "Data generation failed"
    assert 'age_group' in analyzer.survey_df.columns, "Missing demographic columns"
    print("  - Generated 50 survey responses")

    # Test demographic analysis
    print("✓ Testing demographic analysis...")
    demographics = analyzer.analyze_demographics()
    assert 'age' in demographics, "Demographic analysis failed"
    assert len(demographics['age']) > 0, "No age data found"
    print("  - Analyzed demographics successfully")

    # Test brand perception analysis
    print("✓ Testing brand perception analysis...")
    brand_perception = analyzer.analyze_brand_perception()
    assert 'nps' in brand_perception, "NPS analysis failed"
    assert 'awareness' in brand_perception, "Awareness analysis failed"
    print(f"  - NPS Score: {brand_perception['nps']['nps']:.1f}")

    # Test purchase behavior analysis
    print("✓ Testing purchase behavior analysis...")
    purchase_behavior = analyzer.analyze_purchase_behavior()
    assert 'spending' in purchase_behavior, "Spending analysis failed"
    assert 'channels' in purchase_behavior, "Channel analysis failed"
    print(f"  - Average spending: ${purchase_behavior['spending']['mean']:.2f}")

    # Test pain points identification
    print("✓ Testing pain points analysis...")
    pain_points = analyzer.identify_pain_points()
    assert len(pain_points) > 0, "Pain points analysis failed"
    print(f"  - Identified {len(pain_points)} pain points")

    # Test customer segmentation
    print("✓ Testing customer segmentation...")
    segmented_data, segment_analysis = analyzer.segment_customers(n_clusters=3)
    assert 'segment' in segmented_data.columns, "Segmentation failed"
    assert len(segment_analysis) == 3, "Wrong number of segments"
    print(f"  - Created {len(segment_analysis)} customer segments")

    # Test communication recommendations
    print("✓ Testing communication recommendations...")
    comm_recommendations = analyzer.generate_communication_recommendations()
    assert 'primary_channels' in comm_recommendations, "Communication analysis failed"
    print(f"  - Generated recommendations for {len(comm_recommendations['primary_channels'])} channels")

    # Test file outputs
    print("✓ Testing file outputs...")
    # Create fresh analyzer for final test
    fresh_analyzer = MarketResearchAnalyzer()
    results = fresh_analyzer.run_complete_analysis(num_respondents=25)

    # Check if files were created
    assert os.path.exists('results/market_research_report.md'), "Report not generated"
    assert os.path.exists('results/market_research_dashboard.png'), "Dashboard not generated"
    assert os.path.exists('data/processed_survey_data.csv'), "Survey data not saved"
    assert os.path.exists('data/customer_segments.csv'), "Segment data not saved"

    print("  - All output files generated successfully")

    # Verify data integrity
    survey_data = pd.read_csv('data/processed_survey_data.csv')
    segment_data = pd.read_csv('data/customer_segments.csv')

    assert len(survey_data) == 25, f"Survey data count mismatch: expected 25, got {len(survey_data)}"
    assert len(segment_data) == 25, f"Segment data count mismatch: expected 25, got {len(segment_data)}"
    assert 'segment' in segment_data.columns, "Segment column missing"

    print("  - Data integrity verified")

    print("\n🎉 All tests passed! Market Research Tool is working correctly.")
    return True

def test_error_handling():
    """Test error handling capabilities"""
    print("\nTesting error handling...")

    analyzer = MarketResearchAnalyzer()

    # Test with empty data
    try:
        analyzer.analyze_demographics()
        assert False, "Should have failed with empty data"
    except AttributeError:
        print("✓ Properly handles empty data scenario")

    print("✓ Error handling tests passed")

if __name__ == "__main__":
    try:
        # Change to scripts directory
        os.chdir(os.path.dirname(__file__))

        # Run tests
        test_market_research_tool()
        test_error_handling()

        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("Market Research Analysis Tool is ready for use.")
        print("="*50)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)