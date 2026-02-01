# Project 9: Customer Segmentation Analysis Tool

## Overview
An advanced customer segmentation tool that uses RFM (Recency, Frequency, Monetary) analysis and K-means clustering to segment customers into actionable groups. This tool processes customer transaction data to identify high-value customers, at-risk segments, and targeted marketing opportunities.

## Features

### 👥 Customer Data Processing
- **Mock Data Generation**: Creates realistic customer transaction data
- **RFM Analysis**: Recency, Frequency, Monetary value calculations
- **Data Cleaning**: Handles missing values and outliers
- **Scalable Processing**: Supports thousands of customers and transactions

### 📊 RFM Scoring System
- **Recency Score**: Days since last purchase (1-5 scale, 5 = most recent)
- **Frequency Score**: Number of purchases (1-5 scale, 5 = most frequent)
- **Monetary Score**: Total spending (1-5 scale, 5 = highest value)
- **Combined RFM Score**: Three-digit code (e.g., 555 = best customers)

### 🎯 K-means Clustering
- **Automated Segmentation**: Unsupervised learning for customer grouping
- **Optimal Cluster Selection**: Elbow method and silhouette analysis
- **Cluster Profiling**: Detailed characteristics of each segment
- **Segment Stability**: Validation of clustering results

### 📈 Segment Insights & Recommendations
- **Segment Profiling**: Demographics, behavior, and value metrics
- **Targeted Strategies**: Segment-specific marketing recommendations
- **Retention Strategies**: Prevent customer churn in at-risk segments
- **Growth Opportunities**: Identify expansion potential

### 📋 Comprehensive Reporting
- **Executive Summary**: Key segmentation findings
- **Segment Analysis**: Detailed breakdown of each customer group
- **Strategic Recommendations**: Actionable marketing and retention plans
- **Performance Metrics**: ROI projections and success measurements

## Technical Stack
- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (K-means, preprocessing)
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy
- **Data Export**: CSV and Markdown reporting

## Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the segmentation**:
   ```bash
   cd scripts
   python customer_segmenter.py
   ```

## Usage

### Complete Customer Segmentation
```python
from customer_segmenter import CustomerSegmenter

segmenter = CustomerSegmenter()
segmenter.run_complete_segmentation()
```

### Custom Segmentation Analysis
```python
# Generate customer data
segmenter.generate_mock_customer_data(num_customers=1000)

# Calculate RFM scores
rfm_data = segmenter.calculate_rfm_scores()

# Perform clustering
segmented_data = segmenter.perform_customer_segmentation(n_clusters=4)

# Generate insights
insights = segmenter.generate_segment_insights()

# Create report
report = segmenter.generate_segmentation_report()
```

## Output Files

### Reports
- `results/customer_segmentation_report.md`: Comprehensive segmentation analysis
- `results/segment_insights.md`: Detailed segment recommendations

### Visualizations
- `results/customer_segmentation_analysis.png`: RFM and cluster visualizations
- `results/rfm_distributions.png`: RFM score distributions
- `results/cluster_profiles.png`: Cluster characteristic charts

### Data Exports
- `data/customer_segments.csv`: Segmented customer data with cluster assignments
- `data/rfm_scores.csv`: RFM scores for all customers
- `data/customer_transactions.csv`: Raw transaction data

## RFM Analysis Framework

### Recency (R) - How recently did they purchase?
- **R = 5**: Purchased within last 30 days
- **R = 4**: Purchased within last 31-90 days
- **R = 3**: Purchased within last 91-180 days
- **R = 2**: Purchased within last 181-365 days
- **R = 1**: Purchased more than 365 days ago

### Frequency (F) - How often do they purchase?
- **F = 5**: 10+ purchases
- **F = 4**: 6-9 purchases
- **F = 3**: 3-5 purchases
- **F = 2**: 2 purchases
- **F = 1**: 1 purchase

### Monetary (M) - How much do they spend?
- **M = 5**: $500+ total spent
- **M = 4**: $200-499 total spent
- **M = 3**: $100-199 total spent
- **M = 2**: $50-99 total spent
- **M = 1**: <$50 total spent

## Customer Segments (Typical Clusters)

### Champions (High Value, Recent, Frequent)
- **RFM Profile**: Recent, frequent, high-spending customers
- **Strategy**: Reward loyalty, seek referrals, premium services
- **Business Value**: 30-40% of revenue, low churn risk

### Loyal Customers (Regular Buyers)
- **RFM Profile**: Regular purchase frequency, moderate spending
- **Strategy**: Loyalty programs, cross-selling, personalized offers
- **Business Value**: Stable revenue, brand advocates

### Potential Loyalists (Growing Value)
- **RFM Profile**: Recent purchasers, increasing frequency/spending
- **Strategy**: Onboard to loyalty program, targeted promotions
- **Business Value**: Growth potential, responsive to offers

### At-Risk Customers (Declining Engagement)
- **RFM Profile**: Previously good customers, declining activity
- **Strategy**: Re-engagement campaigns, special offers, feedback
- **Business Value**: Retention opportunity, prevent churn

### Hibernating (Low Engagement)
- **RFM Profile**: Low recency, frequency, and monetary value
- **Strategy**: Win-back campaigns, re-engagement emails
- **Business Value**: Low current value, reactivation potential

## Business Applications

### For Marketing Teams
- **Targeted Campaigns**: Send relevant offers to each segment
- **Personalization**: Customize messaging based on customer behavior
- **Campaign Optimization**: Focus budget on high-value segments

### For Sales Teams
- **Lead Scoring**: Prioritize prospects based on RFM profiles
- **Account Management**: Tailor approach to customer segments
- **Upselling Opportunities**: Identify cross-sell and upsell potential

### For Customer Success
- **Retention Strategies**: Proactive intervention for at-risk customers
- **Customer Health Scoring**: Monitor customer engagement
- **Support Prioritization**: Focus resources on high-value customers

### For Product Teams
- **Feature Prioritization**: Understand which segments need what features
- **Product Development**: Design products for target segments
- **Pricing Strategy**: Segment-based pricing and packaging

## Testing
Run the comprehensive test suite:
```bash
python test_customer_segmentation.py
```

## Sample Output

```
============================================================
CUSTOMER SEGMENTATION ANALYSIS TOOL
============================================================

📊 STEP 1: Customer Data Generation
Generating 1000 mock customer records...
Generated 4067 transactions for 1000 customers

📈 STEP 2: RFM Score Calculation
Calculating RFM scores for all customers...
RFM scores calculated

🎯 STEP 3: Customer Segmentation
Performing customer segmentation with 4 clusters...
Calculating cluster profiles...
Generating segment insights...
Customer segmentation complete with 4 clusters

📋 STEP 4: Segmentation Report
Generating comprehensive segmentation report...
Segmentation report saved to: results/customer_segmentation_report.md

✅ CUSTOMER SEGMENTATION COMPLETE!
📊 Check 'results/' folder for visualizations and reports
📄 Check 'data/' folder for segmented customer data
============================================================
```

## Segmentation Quality Metrics

### Clustering Validation
- **Silhouette Score**: Measures cluster separation quality
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
- **Elbow Method**: Identifies optimal number of clusters

### RFM Score Distribution
- **Score Balance**: Ensures even distribution across score ranges
- **Segment Sizes**: Validates meaningful segment sizes
- **Business Logic**: Confirms segments align with business understanding

## Strategic Recommendations Framework

### Immediate Actions (Week 1-2)
- **Champions**: Implement loyalty rewards program
- **At-Risk**: Launch re-engagement email campaign
- **High-Value Prospects**: Personalized onboarding

### Short-term Initiatives (Month 1-3)
- **Segment-Specific Campaigns**: Tailored marketing for each cluster
- **CRM Integration**: Implement segment-based automation
- **Content Personalization**: Segment-specific messaging

### Long-term Strategies (Month 3-6)
- **Loyalty Program Enhancement**: Advanced rewards for top segments
- **Product Development**: Segment-driven feature development
- **Customer Experience**: Personalized journeys for each segment

## Future Enhancements
- Real-time segmentation updates
- Predictive churn modeling
- Integration with CRM systems
- A/B testing framework for campaigns
- Customer lifetime value integration
- Automated campaign triggering

## Validation & Testing
- **Data Integrity**: All RFM calculations validated
- **Clustering Stability**: Consistent results across runs
- **Business Logic**: Segments align with marketing best practices
- **Performance**: Optimized for large customer databases

## License
This project is part of a data engineering portfolio demonstrating customer analytics and machine learning capabilities.

## Contact
Portfolio project showcasing advanced customer segmentation, RFM analysis, and marketing strategy development skills.</content>
<parameter name="filePath">D:\project\Personal Project\project9_customer_segmentation\README.md