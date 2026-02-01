#!/usr/bin/env python3
"""
Test script for Communication Campaign Analysis Tool
"""

import os
import sys
import pandas as pd
import json
from communication_analyzer import CommunicationCampaignAnalyzer

def test_communication_analysis_tool():
    """Test the communication campaign analysis functionality"""
    print("Testing Communication Campaign Analysis Tool...")

    # Initialize analyzer
    analyzer = CommunicationCampaignAnalyzer()

    # Test data generation
    print("✓ Testing campaign data generation...")
    analyzer.generate_mock_campaign_data(8)
    assert len(analyzer.campaign_data) == 8, "Campaign data generation failed"
    assert 'campaign_id' in analyzer.campaign_data[0], "Missing campaign ID"
    assert 'performance_metrics' in analyzer.campaign_data[0], "Missing performance metrics"
    print("  - Generated 8 campaign records")

    # Test performance analysis
    print("✓ Testing performance analysis...")
    avg_metrics, type_performance = analyzer.analyze_campaign_performance()
    assert 'avg_reach' in avg_metrics, "Performance analysis failed"
    assert 'avg_roi' in avg_metrics, "ROI analysis failed"
    assert len(type_performance) > 0, "Campaign type analysis failed"
    print(f"  - Average ROI: {avg_metrics['avg_roi']:.1f}%")

    # Test channel effectiveness
    print("✓ Testing channel effectiveness analysis...")
    channel_summary = analyzer.analyze_channel_effectiveness()
    assert len(channel_summary) > 0, "Channel analysis failed"
    assert 'roi_contribution' in channel_summary.columns, "ROI contribution missing"
    assert 'cost_per_conversion' in channel_summary.columns, "Cost per conversion missing"
    print(f"  - Analyzed {len(channel_summary)} channels")

    # Test sentiment impact analysis
    print("✓ Testing sentiment impact analysis...")
    sentiment_df, correlations, model = analyzer.analyze_sentiment_impact()
    assert len(sentiment_df) == 8, "Sentiment data count mismatch"
    assert 'sentiment_score' in correlations.columns, "Sentiment correlations failed"
    assert hasattr(model, 'coef_'), "Sentiment model failed"
    print(f"  - Sentiment-effectiveness correlation: {correlations.loc['sentiment_score', 'effectiveness_score']:.3f}")

    # Test PR impact analysis
    print("✓ Testing PR impact analysis...")
    pr_df, pr_correlations = analyzer.analyze_pr_impact()
    assert len(pr_df) == 8, "PR data count mismatch"
    assert 'earned_media_value' in pr_df.columns, "Earned media value missing"
    assert 'awareness_lift' in pr_df.columns, "Awareness lift missing"
    print(f"  - Average earned media value: ${pr_df['earned_media_value'].mean():,.0f}")

    # Test campaign segmentation
    print("✓ Testing campaign segmentation...")
    campaign_segments, segment_analysis = analyzer.analyze_campaign_performance()[1], analyzer.segment_campaigns_by_performance()[1]  # Get segment analysis
    campaign_segments, segment_analysis = analyzer.segment_campaigns_by_performance()
    assert 'performance_segment' in campaign_segments.columns, "Segmentation failed"
    assert len(segment_analysis) >= 3, "Insufficient segments created"
    print(f"  - Created {len(segment_analysis)} performance segments")

    # Test recommendations generation
    print("✓ Testing recommendations generation...")
    recommendations = analyzer.generate_communication_recommendations()
    assert 'channel_strategy' in recommendations, "Channel strategy recommendations failed"
    assert 'content_strategy' in recommendations, "Content strategy recommendations failed"
    assert 'pr_strategy' in recommendations, "PR strategy recommendations failed"
    print(f"  - Generated recommendations for {len(recommendations['channel_strategy']['top_channels'])} top channels")

    # Test file outputs with fresh analyzer
    print("✓ Testing file outputs...")
    fresh_analyzer = CommunicationCampaignAnalyzer()
    results = fresh_analyzer.run_complete_analysis(num_campaigns=6)

    # Check if files were created
    assert os.path.exists('results/communication_campaign_report.md'), "Report not generated"
    assert os.path.exists('results/communication_analysis_dashboard.png'), "Dashboard not generated"
    assert os.path.exists('data/campaign_performance_data.csv'), "Campaign data not saved"
    assert os.path.exists('data/analysis_summary.json'), "Summary JSON not saved"

    print("  - All output files generated successfully")

    # Verify data integrity
    campaign_data = pd.read_csv('data/campaign_performance_data.csv')
    with open('data/analysis_summary.json', 'r') as f:
        summary_data = json.load(f)

    assert len(campaign_data) == 6, f"Campaign data count mismatch: expected 6, got {len(campaign_data)}"
    assert summary_data['total_campaigns'] == 6, "Summary data count mismatch"
    assert 'average_roi' in summary_data, "ROI summary missing"
    assert 'top_channels' in summary_data, "Top channels summary missing"

    print("  - Data integrity verified")

    # Test report content quality
    with open('results/communication_campaign_report.md', 'r', encoding='utf-8') as f:
        report_content = f.read()

    assert 'Executive Summary' in report_content, "Executive summary missing"
    assert 'Strategic Recommendations' in report_content, "Strategic recommendations missing"
    assert 'Channel Effectiveness Analysis' in report_content, "Channel analysis section missing"
    assert len(report_content) > 5000, "Report too short"

    print("  - Report content quality verified")

    print("\n🎉 All tests passed! Communication Campaign Analysis Tool is working correctly.")
    return True

def test_error_handling():
    """Test error handling capabilities"""
    print("\nTesting error handling...")

    analyzer = CommunicationCampaignAnalyzer()

    # Test with empty data
    try:
        analyzer.analyze_campaign_performance()
        assert False, "Should have failed with empty data"
    except (AttributeError, KeyError):
        print("✓ Properly handles empty data scenario")

    # Test with minimal data
    analyzer.generate_mock_campaign_data(1)
    try:
        results = analyzer.analyze_campaign_performance()
        assert results[0]['avg_reach'] > 0, "Minimal data analysis failed"
        print("✓ Handles minimal data correctly")
    except Exception as e:
        print(f"✗ Minimal data test failed: {e}")

    print("✓ Error handling tests completed")

def performance_test():
    """Test performance with larger dataset"""
    print("\nTesting performance with larger dataset...")

    analyzer = CommunicationCampaignAnalyzer()
    start_time = pd.Timestamp.now()

    # Test with larger dataset
    results = analyzer.run_complete_analysis(num_campaigns=20)

    end_time = pd.Timestamp.now()
    duration = (end_time - start_time).total_seconds()

    assert duration < 30, f"Analysis took too long: {duration:.1f} seconds"
    assert len(analyzer.campaign_data) == 20, "Large dataset generation failed"

    print(f"Analysis completed in {duration:.1f} seconds")
    print("✓ Performance test passed")

if __name__ == "__main__":
    try:
        # Change to scripts directory
        os.chdir(os.path.dirname(__file__))

        # Run tests
        test_communication_analysis_tool()
        test_error_handling()
        performance_test()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("Communication Campaign Analysis Tool is ready for use.")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)