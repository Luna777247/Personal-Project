# Project 5: Market Research Analysis Tool

## Overview
A comprehensive market research analysis tool that processes survey data to provide deep insights into customer behavior, brand perception, and market opportunities. The tool includes automated data generation, statistical analysis, customer segmentation, and strategic recommendations.

## Features

### 🔍 Comprehensive Analysis
- **Demographic Analysis**: Age, gender, income, education breakdowns
- **Brand Perception**: Awareness, satisfaction, loyalty metrics
- **Purchase Behavior**: Spending patterns, frequency, channel preferences
- **Customer Pain Points**: Automated identification of key issues
- **NPS Analysis**: Net Promoter Score calculation and segmentation

### 🎯 Customer Segmentation
- K-means clustering for automatic customer grouping
- 4 predefined segments: High-Value Loyalists, Price-Sensitive Moderates, Low-Engagement Passives, Premium Enthusiasts
- Segment-specific characteristics and recommendations

### 📊 Advanced Analytics
- Statistical analysis with scipy and statsmodels
- Machine learning-based customer clustering
- Sentiment analysis capabilities (extensible)
- Trend identification and forecasting

### 📈 Rich Visualizations
- 6-panel dashboard with key metrics
- Demographic distributions
- Brand perception charts
- Purchase behavior analysis
- Pain points visualization
- NPS distribution analysis

### 📋 Automated Reporting
- Comprehensive markdown reports
- Executive summaries with key insights
- Strategic recommendations
- Actionable communication strategies

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
cd scripts
python market_research_analyzer.py
```

### Run Tests
```bash
cd scripts
python test_market_research.py
```

## Sample Output

```
============================================================
MARKET RESEARCH ANALYSIS TOOL
============================================================

📊 STEP 1: Data Collection
Generating mock survey data for 150 respondents...
Generated survey data for 150 respondents

📊 ANALYSIS SUMMARY:
Total respondents: 150
Average NPS: -38.0/10
Average monthly spending: $146.41
Top pain point: Slow delivery
Number of segments: 4
```

## File Structure

```
project5_market_research/
├── scripts/
│   ├── market_research_analyzer.py    # Main analysis tool
│   ├── test_market_research.py        # Comprehensive test suite
│   ├── data/
│   │   ├── processed_survey_data.csv  # Cleaned survey data
│   │   └── customer_segments.csv      # Segmentation results
│   └── results/
│       ├── market_research_dashboard.png  # Visual dashboard
│       └── market_research_report.md      # Detailed report
├── data/                               # Additional data files
├── results/                            # Output files directory
├── docs/
│   └── README.md                       # Detailed documentation
└── requirements.txt                    # Python dependencies
```

## Key Components

### MarketResearchAnalyzer Class
- `generate_mock_survey_data()`: Creates realistic survey responses
- `analyze_demographics()`: Processes demographic information
- `analyze_brand_perception()`: Evaluates brand metrics
- `analyze_purchase_behavior()`: Analyzes buying patterns
- `identify_pain_points()`: Extracts customer issues
- `segment_customers()`: Performs customer clustering
- `generate_communication_recommendations()`: Creates strategy recommendations
- `generate_visualizations()`: Produces charts and graphs
- `generate_report()`: Creates comprehensive reports

### Survey Data Structure
The tool analyzes surveys with the following key metrics:
- Demographics: age, gender, income, education
- Brand metrics: awareness, satisfaction, loyalty, NPS
- Purchase data: spending, frequency, categories, channels
- Pain points: customer issues and complaints
- Communication preferences: channel preferences

## Customization

### Adding Real Survey Data
Replace the `generate_mock_survey_data()` method with your actual survey data:

```python
# Load your survey data
survey_df = pd.read_csv('your_survey_data.csv')
analyzer.survey_df = survey_df
```

### Modifying Segmentation
Adjust the number of customer segments:

```python
# Change to 3 segments instead of 4
segmented_data, segment_analysis = analyzer.segment_customers(n_clusters=3)
```

### Custom Pain Point Categories
Add or modify pain point categories in the survey generation:

```python
pain_points = [
    'High prices', 'Poor quality', 'Slow delivery',
    'Bad customer service', 'Limited product variety',
    'Outdated website', 'Lack of trust',
    'Poor mobile experience', 'Limited payment options',
    'Return difficulties', 'Custom pain point here'  # Add your own
]
```

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **scikit-learn**: Machine learning (clustering)
- **scipy**: Statistical analysis

### Performance
- Handles 100-1000+ survey responses efficiently
- Clustering scales with O(n) complexity
- Report generation optimized for large datasets

### Data Quality
- Automatic data validation
- Outlier detection in spending data
- Missing value handling
- Statistical significance testing

## Business Applications

### Market Research Firms
- Automated survey analysis
- Customer segmentation studies
- Brand perception tracking
- Competitive analysis

### Product Teams
- Feature prioritization based on pain points
- Customer journey optimization
- Pricing strategy validation
- Product-market fit assessment

### Marketing Teams
- Communication channel optimization
- Campaign effectiveness measurement
- Customer retention strategies
- Brand positioning insights

### Executive Leadership
- Customer satisfaction monitoring
- Market opportunity identification
- Strategic planning support
- ROI measurement frameworks

## Future Enhancements

### Planned Features
- [ ] Real-time survey integration (Google Forms, SurveyMonkey)
- [ ] Advanced sentiment analysis for open-ended responses
- [ ] Predictive modeling for customer behavior
- [ ] A/B testing framework integration
- [ ] Multi-language survey support
- [ ] Web-based dashboard interface

### Integration Options
- CRM system integration
- Marketing automation platforms
- Business intelligence tools
- Customer support systems

## Testing

The tool includes comprehensive testing:

```bash
cd scripts
python test_market_research.py
```

Tests cover:
- Data generation accuracy
- Analysis function correctness
- File output validation
- Error handling scenarios
- Data integrity verification

## License

This project is part of a data engineering portfolio demonstrating market research analysis capabilities.

## Support

For questions or issues:
1. Check the test suite: `python test_market_research.py`
2. Review the documentation in `docs/README.md`
3. Examine sample outputs in the `results/` directory

---

*Developed as Project 5 in the Data Engineering & Business Intelligence Portfolio*