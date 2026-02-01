# Project 4: Social Media Trend Analysis

## Overview
This project analyzes social media trends and sentiment around brands or events. It collects data from social media sources, performs sentiment analysis, identifies trending topics, and generates comprehensive reports with actionable insights.

## Features

### 🔍 Data Collection
- Mock social media data generation (can be extended to real APIs)
- Support for multiple platforms (Twitter, Facebook, Instagram, LinkedIn, TikTok)
- Engagement metrics tracking (likes, shares, comments)

### 😊 Sentiment Analysis
- Automated sentiment classification (Positive/Negative/Neutral)
- TextBlob-based natural language processing
- Sentiment distribution visualization

### 📝 Topic Analysis
- Keyword extraction and frequency analysis
- Word cloud generation
- Topic trend identification

### 📈 Engagement Analysis
- Platform performance comparison
- Daily engagement trends
- Sentiment-engagement correlation analysis

### 📊 Reporting
- Comprehensive analysis dashboard
- Automated report generation
- Visual insights and recommendations

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. For Vietnamese text processing (optional):
```bash
python -m textblob.download_corpora
```

## Usage

### Basic Analysis
```python
from scripts.social_media_analyzer import SocialMediaAnalyzer

# Initialize analyzer
analyzer = SocialMediaAnalyzer()

# Run complete analysis
results = analyzer.run_complete_analysis(topic="climate_change", num_posts=100)
```

### Custom Analysis
```python
# Generate data
analyzer.generate_mock_social_data("your_topic", 200)

# Run individual analyses
sentiment = analyzer.analyze_sentiment()
keywords = analyzer.extract_topics_and_keywords()
engagement = analyzer.analyze_engagement_and_spread()

# Generate visualizations
analyzer.generate_visualizations()

# Create report
analyzer.generate_report()
```

## Output Files

### Data Files (`data/`)
- `processed_social_data.csv` - Complete processed dataset

### Results Files (`results/`)
- `social_media_analysis_dashboard.png` - Main analysis dashboard
- `wordcloud.png` - Keyword word cloud visualization
- `analysis_report.md` - Comprehensive analysis report

## Analysis Components

### 1. Sentiment Distribution
- Pie chart showing positive/negative/neutral sentiment ratios
- Percentage breakdown of public opinion

### 2. Platform Performance
- Average engagement by platform
- Comparative analysis of reach and interaction

### 3. Engagement Trends
- Daily engagement patterns
- Peak activity identification

### 4. Content Analysis
- Most discussed topics and keywords
- Content theme identification

## Key Metrics

- **Sentiment Score**: Polarity analysis (-1 to +1)
- **Engagement Rate**: Combined likes + shares×10 + comments×5
- **Topic Frequency**: Keyword occurrence analysis
- **Platform Reach**: Cross-platform performance comparison

## Business Applications

### Brand Monitoring
- Track brand sentiment in real-time
- Identify emerging issues and opportunities
- Monitor competitor mentions

### Crisis Management
- Early warning system for negative sentiment spikes
- Rapid response strategy development
- Reputation risk assessment

### Content Strategy
- Identify trending topics and hashtags
- Optimize posting times and platforms
- Content performance analysis

### Market Research
- Consumer sentiment analysis
- Trend identification and forecasting
- Competitive intelligence gathering

## Customization Options

### Data Sources
- Extend to real social media APIs (Twitter, Facebook, Instagram)
- Add news sources and RSS feeds
- Include web scraping capabilities

### Analysis Features
- Custom sentiment dictionaries
- Multi-language support
- Advanced NLP techniques (BERT, GPT models)

### Reporting
- Custom dashboard templates
- Automated email reports
- Integration with BI tools (Tableau, Power BI)

## Technical Requirements

- Python 3.8+
- Required libraries: pandas, numpy, matplotlib, seaborn, textblob, wordcloud
- Optional: NLTK for advanced text processing

## Future Enhancements

- Real-time data streaming
- Machine learning sentiment models
- Predictive trend analysis
- Multi-language support
- API integrations with major social platforms

## License
This project is for educational and research purposes.

## Contributing
Feel free to contribute improvements, bug fixes, or new features!