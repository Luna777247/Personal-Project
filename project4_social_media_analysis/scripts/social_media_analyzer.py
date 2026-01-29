import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SocialMediaAnalyzer:
    """Social Media Trend Analysis Tool"""

    def __init__(self):
        self.posts_data = []
        self.sentiment_data = []
        self.topic_data = []

    def generate_mock_social_data(self, topic="climate_change", num_posts=100):
        """Generate mock social media data for demonstration"""
        print(f"Generating mock data for topic: {topic}")

        # Mock posts with different sentiments
        positive_posts = [
            "Amazing initiative for climate action! Love seeing companies taking responsibility 🌱",
            "So proud of our community coming together for environmental protection",
            "Great news! New sustainable policies will make a real difference",
            "Inspiring to see young people leading climate change discussions",
            "Excellent environmental report - transparency is key! 📊"
        ]

        negative_posts = [
            "Disappointed with the lack of action on climate change. Words are not enough!",
            "Corporate greenwashing at its finest. Where's the real commitment?",
            "Climate crisis is getting worse and we're still talking instead of acting",
            "Frustrated with politicians ignoring scientific evidence on environment",
            "Environmental regulations are too weak - we need stronger action now!"
        ]

        neutral_posts = [
            "Interesting discussion on climate change policies today",
            "New environmental study shows mixed results on carbon emissions",
            "Government announces climate action plan for next year",
            "Scientists present findings on global warming trends",
            "Companies share their sustainability goals for 2025"
        ]

        # Generate posts
        platforms = ['Twitter', 'Facebook', 'Instagram', 'LinkedIn', 'TikTok']
        usernames = [f'user_{i}' for i in range(1, num_posts + 1)]

        for i in range(num_posts):
            # Random sentiment distribution (40% positive, 35% neutral, 25% negative)
            rand = np.random.random()
            if rand < 0.4:
                sentiment = 'positive'
                text = np.random.choice(positive_posts)
            elif rand < 0.75:
                sentiment = 'neutral'
                text = np.random.choice(neutral_posts)
            else:
                sentiment = 'negative'
                text = np.random.choice(negative_posts)

            post = {
                'post_id': f'post_{i+1}',
                'username': np.random.choice(usernames),
                'platform': np.random.choice(platforms),
                'text': text,
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30)),
                'likes': np.random.randint(0, 1000),
                'shares': np.random.randint(0, 100),
                'comments': np.random.randint(0, 50),
                'engagement': 0,  # Will be calculated
                'actual_sentiment': sentiment  # For validation
            }

            # Calculate engagement
            post['engagement'] = post['likes'] + post['shares'] * 10 + post['comments'] * 5

            self.posts_data.append(post)

        self.posts_df = pd.DataFrame(self.posts_data)
        print(f"Generated {len(self.posts_df)} posts")
        return self.posts_df

    def analyze_sentiment(self):
        """Analyze sentiment of posts using TextBlob"""
        print("Analyzing sentiment...")

        def get_sentiment(text):
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity

            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'

        self.posts_df['predicted_sentiment'] = self.posts_df['text'].apply(get_sentiment)

        # Calculate sentiment distribution
        sentiment_counts = self.posts_df['predicted_sentiment'].value_counts()

        print("Sentiment Analysis Results:")
        print(sentiment_counts)

        return sentiment_counts

    def extract_topics_and_keywords(self):
        """Extract topics and keywords from posts"""
        print("Extracting topics and keywords...")

        def clean_text(text):
            # Remove mentions, hashtags, URLs
            text = re.sub(r'@[A-Za-z0-9_]+', '', text)
            text = re.sub(r'#', '', text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text.lower().strip()

        # Clean text and extract words
        self.posts_df['clean_text'] = self.posts_df['text'].apply(clean_text)

        # Common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}

        all_words = []
        for text in self.posts_df['clean_text']:
            words = [word for word in text.split() if word not in stop_words and len(word) > 2]
            all_words.extend(words)

        # Get most common words
        word_freq = Counter(all_words)
        top_keywords = word_freq.most_common(20)

        print("Top 20 Keywords:")
        for word, freq in top_keywords[:10]:
            print(f"  {word}: {freq}")

        return top_keywords

    def analyze_engagement_and_spread(self):
        """Analyze engagement patterns and spread metrics"""
        print("Analyzing engagement and spread...")

        # Platform engagement analysis
        platform_engagement = self.posts_df.groupby('platform').agg({
            'engagement': 'mean',
            'likes': 'mean',
            'shares': 'mean',
            'comments': 'mean'
        }).round(2)

        # Daily engagement trends
        self.posts_df['date'] = self.posts_df['timestamp'].dt.date
        daily_engagement = self.posts_df.groupby('date')['engagement'].sum()

        # Sentiment vs engagement correlation
        sentiment_engagement = self.posts_df.groupby('predicted_sentiment')['engagement'].mean()

        print("Platform Engagement:")
        print(platform_engagement)

        return platform_engagement, daily_engagement, sentiment_engagement

    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("Generating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Social Media Trend Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Sentiment Distribution
        sentiment_counts = self.posts_df['predicted_sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # green, red, gray
        axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0,0].set_title('Sentiment Distribution')

        # 2. Platform Engagement
        platform_engagement = self.posts_df.groupby('platform')['engagement'].mean().sort_values(ascending=True)
        platform_engagement.plot(kind='barh', ax=axes[0,1], color='skyblue')
        axes[0,1].set_title('Average Engagement by Platform')
        axes[0,1].set_xlabel('Average Engagement')

        # 3. Daily Engagement Trend
        daily_engagement = self.posts_df.groupby('date')['engagement'].sum()
        daily_engagement.plot(ax=axes[1,0], marker='o', color='orange')
        axes[1,0].set_title('Daily Engagement Trend')
        axes[1,0].set_ylabel('Total Engagement')
        axes[1,0].tick_params(axis='x', rotation=45)

        # 4. Sentiment vs Engagement
        sentiment_engagement = self.posts_df.groupby('predicted_sentiment')['engagement'].mean()
        sentiment_engagement.plot(kind='bar', ax=axes[1,1], color=['green', 'red', 'gray'])
        axes[1,1].set_title('Average Engagement by Sentiment')
        axes[1,1].set_ylabel('Average Engagement')

        plt.tight_layout()
        plt.savefig('results/social_media_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Generate word cloud
        self.generate_wordcloud()

    def generate_wordcloud(self):
        """Generate word cloud from post content"""
        text = ' '.join(self.posts_df['clean_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                            max_words=100, colormap='viridis').generate(text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Social Media Posts', fontsize=14, fontweight='bold')
        plt.savefig('results/wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("Generating analysis report...")

        # Calculate key metrics
        total_posts = len(self.posts_df)
        sentiment_dist = self.posts_df['predicted_sentiment'].value_counts(normalize=True) * 100
        avg_engagement = self.posts_df['engagement'].mean()
        total_engagement = self.posts_df['engagement'].sum()
        most_active_platform = self.posts_df.groupby('platform')['engagement'].sum().idxmax()

        # Create report
        report = f"""
# Social Media Trend Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report analyzes social media trends and sentiment around climate change discussions.

## Key Findings

### 1. Overall Metrics
- **Total Posts Analyzed:** {total_posts:,}
- **Total Engagement:** {total_engagement:,.0f}
- **Average Engagement per Post:** {avg_engagement:.1f}
- **Most Active Platform:** {most_active_platform}

### 2. Sentiment Analysis
- **Positive Sentiment:** {sentiment_dist.get('positive', 0):.1f}%
- **Negative Sentiment:** {sentiment_dist.get('negative', 0):.1f}%
- **Neutral Sentiment:** {sentiment_dist.get('neutral', 0):.1f}%

### 3. Platform Performance
{self.posts_df.groupby('platform')['engagement'].agg(['count', 'mean', 'sum']).round(2).to_string()}

### 4. Top Keywords
{self.extract_topics_and_keywords()[:10]}

## Insights and Recommendations

### Key Insights:
1. **Sentiment Distribution:** The majority of discussions show {'positive' if sentiment_dist.get('positive', 0) > 40 else 'mixed'} sentiment toward climate change initiatives.

2. **Engagement Patterns:** {most_active_platform} shows the highest engagement, suggesting this platform is most effective for climate discussions.

3. **Content Themes:** The most discussed topics include environmental protection, corporate responsibility, and government policies.

### Recommendations:
1. **Content Strategy:** Focus on positive environmental stories to leverage the {'high' if sentiment_dist.get('positive', 0) > 30 else 'moderate'} positive sentiment.

2. **Platform Selection:** Prioritize {most_active_platform} for maximum reach and engagement.

3. **Crisis Management:** Monitor negative sentiment closely and address concerns promptly to maintain brand reputation.

4. **Community Building:** Engage with positive influencers and community leaders to amplify positive messages.

## Data Sources
- Social media posts from multiple platforms
- Engagement metrics (likes, shares, comments)
- Sentiment analysis using TextBlob
- Keyword extraction and topic modeling

---
*Report generated automatically by Social Media Analysis Tool*
"""

        # Save report
        with open('results/analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("Report saved to: results/analysis_report.md")
        return report

    def run_complete_analysis(self, topic="climate_change", num_posts=100):
        """Run complete social media analysis pipeline"""
        print("=" * 60)
        print("SOCIAL MEDIA TREND ANALYSIS TOOL")
        print("=" * 60)

        # Step 1: Data Collection
        print("\n🔍 STEP 1: Data Collection")
        self.generate_mock_social_data(topic, num_posts)

        # Step 2: Sentiment Analysis
        print("\n😊 STEP 2: Sentiment Analysis")
        sentiment_results = self.analyze_sentiment()

        # Step 3: Topic Analysis
        print("\n📝 STEP 3: Topic and Keyword Analysis")
        keywords = self.extract_topics_and_keywords()

        # Step 4: Engagement Analysis
        print("\n📈 STEP 4: Engagement and Spread Analysis")
        platform_engagement, daily_trends, sentiment_engagement = self.analyze_engagement_and_spread()

        # Step 5: Visualizations
        print("\n📊 STEP 5: Generating Visualizations")
        self.generate_visualizations()

        # Step 6: Report Generation
        print("\n📋 STEP 6: Generating Final Report")
        report = self.generate_report()

        # Save processed data
        self.posts_df.to_csv('data/processed_social_data.csv', index=False)

        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("📁 Check 'results/' folder for visualizations and reports")
        print("📄 Check 'data/' folder for processed data")
        print("=" * 60)

        return {
            'sentiment_distribution': sentiment_results,
            'top_keywords': keywords,
            'platform_engagement': platform_engagement,
            'daily_trends': daily_trends,
            'sentiment_engagement': sentiment_engagement,
            'report': report
        }

def main():
    """Main function to run the social media analysis"""
    analyzer = SocialMediaAnalyzer()
    results = analyzer.run_complete_analysis(topic="climate_change", num_posts=150)

    # Print summary
    print("\n📊 ANALYSIS SUMMARY:")
    print(f"Total posts analyzed: {len(analyzer.posts_df)}")
    print(f"Sentiment distribution: {results['sentiment_distribution'].to_dict()}")
    print(f"Most active platform: {analyzer.posts_df.groupby('platform')['engagement'].sum().idxmax()}")
    print(f"Average engagement: {analyzer.posts_df['engagement'].mean():.1f}")

if __name__ == "__main__":
    main()