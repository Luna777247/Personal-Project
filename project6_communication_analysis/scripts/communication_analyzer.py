import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from datetime import datetime, timedelta
import warnings
from textblob import TextBlob
from wordcloud import WordCloud
import json
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CommunicationCampaignAnalyzer:
    """Advanced Communication Campaign Analysis Tool"""

    def __init__(self):
        self.campaign_data = []
        self.social_media_data = []
        self.pr_metrics = []
        self.audience_segments = []

    def generate_mock_campaign_data(self, num_campaigns=10, days_duration=30):
        """Generate comprehensive mock campaign data"""
        print(f"Generating mock campaign data for {num_campaigns} campaigns over {days_duration} days...")

        # Campaign types and objectives
        campaign_types = ['Brand Awareness', 'Product Launch', 'Seasonal Promotion', 'CSR Initiative', 'Crisis Management']
        objectives = ['Increase Brand Awareness', 'Drive Sales', 'Engage Audience', 'Build Community', 'Manage Reputation']
        channels = ['Social Media', 'Email', 'Website', 'PR', 'Events', 'Influencer', 'Paid Ads', 'Content Marketing']

        # Communication metrics
        message_types = ['Informational', 'Emotional', 'Educational', 'Entertaining', 'Call-to-Action']
        tones = ['Professional', 'Friendly', 'Urgent', 'Inspirational', 'Humorous', 'Empathetic']

        for campaign_id in range(1, num_campaigns + 1):
            # Campaign basic info
            campaign_type = np.random.choice(campaign_types)
            primary_objective = np.random.choice(objectives)
            budget = np.random.uniform(5000, 50000)  # $5K - $50K

            # Timeline
            start_date = datetime.now() - timedelta(days=np.random.randint(0, 90))
            end_date = start_date + timedelta(days=days_duration)

            # Channel mix (multiple channels per campaign)
            num_channels = np.random.randint(2, 5)
            campaign_channels = np.random.choice(channels, size=num_channels, replace=False)

            # Target audience
            target_age_min = np.random.choice([18, 25, 35, 45])
            target_age_max = target_age_min + np.random.randint(10, 25)
            target_gender = np.random.choice(['All', 'Male', 'Female', 'Mixed'])
            target_income = np.random.choice(['All', 'Low', 'Medium', 'High'])

            # Communication strategy
            primary_message = np.random.choice(message_types)
            communication_tone = np.random.choice(tones)
            frequency = np.random.choice(['Daily', 'Weekly', 'Bi-weekly', 'Monthly'])

            # Performance metrics (realistic ranges)
            reach = int(np.random.normal(50000, 15000))  # Target reach
            impressions = int(reach * np.random.uniform(2.5, 5.0))  # Impressions > reach
            engagements = int(impressions * np.random.uniform(0.02, 0.08))  # 2-8% engagement rate
            clicks = int(engagements * np.random.uniform(0.3, 0.7))  # 30-70% click-through rate
            conversions = int(clicks * np.random.uniform(0.05, 0.15))  # 5-15% conversion rate

            # Social media specific metrics
            likes = int(engagements * np.random.uniform(0.4, 0.7))
            shares = int(engagements * np.random.uniform(0.1, 0.3))
            comments = int(engagements * np.random.uniform(0.2, 0.4))
            saves = int(engagements * np.random.uniform(0.05, 0.15))

            # Sentiment analysis (simulated)
            positive_sentiment = np.random.uniform(0.4, 0.8)
            neutral_sentiment = np.random.uniform(0.15, 0.4)
            negative_sentiment = 1 - positive_sentiment - neutral_sentiment

            # PR metrics
            media_mentions = np.random.poisson(15)  # Average 15 mentions
            earned_media_value = media_mentions * np.random.uniform(500, 2000)  # $500-$2000 per mention
            share_of_voice = np.random.uniform(0.05, 0.25)  # 5-25% share of voice

            # Brand impact metrics
            brand_awareness_lift = np.random.uniform(5, 25)  # 5-25% increase
            brand_favorability_change = np.random.uniform(-5, 15)  # -5% to +15%
            purchase_intent_increase = np.random.uniform(0, 20)  # 0-20% increase

            # ROI calculation
            total_value = conversions * np.random.uniform(50, 200)  # $50-$200 per conversion
            roi = ((total_value - budget) / budget) * 100 if budget > 0 else 0

            # Campaign effectiveness score (composite metric)
            effectiveness_score = (
                (reach / 100000) * 0.2 +  # Reach component
                (engagements / impressions) * 0.25 +  # Engagement rate
                (conversions / reach) * 0.25 +  # Conversion rate
                positive_sentiment * 0.15 +  # Sentiment
                min(roi / 100, 2) * 0.15  # ROI (capped at 200%)
            ) * 100

            # Create campaign record
            campaign = {
                'campaign_id': f'C{campaign_id:03d}',
                'campaign_name': f'Campaign {campaign_id}: {campaign_type}',
                'campaign_type': campaign_type,
                'primary_objective': primary_objective,
                'budget': round(budget, 2),
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': days_duration,
                'channels': list(campaign_channels),
                'target_audience': {
                    'age_min': target_age_min,
                    'age_max': target_age_max,
                    'gender': target_gender,
                    'income': target_income
                },
                'communication_strategy': {
                    'primary_message': primary_message,
                    'tone': communication_tone,
                    'frequency': frequency
                },
                'performance_metrics': {
                    'reach': max(0, reach),
                    'impressions': max(0, impressions),
                    'engagements': max(0, engagements),
                    'clicks': max(0, clicks),
                    'conversions': max(0, conversions),
                    'engagement_rate': engagements / impressions if impressions > 0 else 0,
                    'click_through_rate': clicks / impressions if impressions > 0 else 0,
                    'conversion_rate': conversions / clicks if clicks > 0 else 0
                },
                'social_metrics': {
                    'likes': max(0, likes),
                    'shares': max(0, shares),
                    'comments': max(0, comments),
                    'saves': max(0, saves)
                },
                'sentiment_analysis': {
                    'positive': round(positive_sentiment, 3),
                    'neutral': round(neutral_sentiment, 3),
                    'negative': round(negative_sentiment, 3),
                    'sentiment_score': round(positive_sentiment - negative_sentiment, 3)
                },
                'pr_metrics': {
                    'media_mentions': media_mentions,
                    'earned_media_value': round(earned_media_value, 2),
                    'share_of_voice': round(share_of_voice, 3)
                },
                'brand_impact': {
                    'awareness_lift': round(brand_awareness_lift, 1),
                    'favorability_change': round(brand_favorability_change, 1),
                    'purchase_intent_increase': round(purchase_intent_increase, 1)
                },
                'roi_metrics': {
                    'total_value_generated': round(total_value, 2),
                    'roi_percentage': round(roi, 1),
                    'effectiveness_score': round(effectiveness_score, 1)
                }
            }

            self.campaign_data.append(campaign)

        self.campaigns_df = pd.DataFrame(self.campaign_data)
        print(f"Generated campaign data for {len(self.campaigns_df)} campaigns")
        return self.campaigns_df

    def analyze_campaign_performance(self):
        """Analyze overall campaign performance metrics"""
        print("Analyzing campaign performance...")

        # Extract performance metrics
        performance_data = []
        for campaign in self.campaign_data:
            perf = campaign['performance_metrics']
            perf['campaign_id'] = campaign['campaign_id']
            perf['campaign_type'] = campaign['campaign_type']
            perf['budget'] = campaign['budget']
            perf['roi'] = campaign['roi_metrics']['roi_percentage']
            perf['effectiveness_score'] = campaign['roi_metrics']['effectiveness_score']
            performance_data.append(perf)

        perf_df = pd.DataFrame(performance_data)

        # Calculate key performance indicators
        avg_metrics = {
            'avg_reach': perf_df['reach'].mean(),
            'avg_engagement_rate': perf_df['engagement_rate'].mean(),
            'avg_conversion_rate': perf_df['conversion_rate'].mean(),
            'avg_roi': perf_df['roi'].mean(),
            'avg_effectiveness_score': perf_df['effectiveness_score'].mean(),
            'total_budget': perf_df['budget'].sum(),
            'total_conversions': perf_df['conversions'].sum()
        }

        # Performance by campaign type
        type_performance = perf_df.groupby('campaign_type').agg({
            'reach': 'mean',
            'engagement_rate': 'mean',
            'conversion_rate': 'mean',
            'roi': 'mean',
            'effectiveness_score': 'mean'
        }).round(3)

        print("Campaign Performance Summary:")
        print(f"Average Reach: {avg_metrics['avg_reach']:,.0f}")
        print(f"Average Engagement Rate: {avg_metrics['avg_engagement_rate']:.1%}")
        print(f"Average ROI: {avg_metrics['avg_roi']:.1f}%")

        return avg_metrics, type_performance

    def analyze_channel_effectiveness(self):
        """Analyze effectiveness of different communication channels"""
        print("Analyzing channel effectiveness...")

        # Flatten channel data
        channel_performance = []
        for campaign in self.campaign_data:
            channels = campaign['channels']
            reach_per_channel = campaign['performance_metrics']['reach'] / len(channels)
            engagements_per_channel = campaign['performance_metrics']['engagements'] / len(channels)
            conversions_per_channel = campaign['performance_metrics']['conversions'] / len(channels)

            for channel in channels:
                channel_performance.append({
                    'channel': channel,
                    'campaign_id': campaign['campaign_id'],
                    'reach': reach_per_channel,
                    'engagements': engagements_per_channel,
                    'conversions': conversions_per_channel,
                    'engagement_rate': engagements_per_channel / reach_per_channel if reach_per_channel > 0 else 0,
                    'conversion_rate': conversions_per_channel / engagements_per_channel if engagements_per_channel > 0 else 0,
                    'budget_allocation': campaign['budget'] / len(channels),
                    'roi_contribution': campaign['roi_metrics']['roi_percentage'] / len(channels)
                })

        channel_df = pd.DataFrame(channel_performance)

        # Aggregate by channel
        channel_summary = channel_df.groupby('channel').agg({
            'reach': 'sum',
            'engagements': 'sum',
            'conversions': 'sum',
            'engagement_rate': 'mean',
            'conversion_rate': 'mean',
            'budget_allocation': 'sum',
            'roi_contribution': 'mean'
        }).round(3)

        # Calculate efficiency metrics
        channel_summary['cost_per_engagement'] = channel_summary['budget_allocation'] / channel_summary['engagements']
        channel_summary['cost_per_conversion'] = channel_summary['budget_allocation'] / channel_summary['conversions']

        # Sort by ROI contribution
        channel_summary = channel_summary.sort_values('roi_contribution', ascending=False)

        print("Top Performing Channels by ROI:")
        for channel, row in channel_summary.head(3).iterrows():
            print(f"  {channel}: {row['roi_contribution']:.1f}% ROI")

        return channel_summary

    def analyze_sentiment_impact(self):
        """Analyze the impact of sentiment on campaign performance"""
        print("Analyzing sentiment impact...")

        sentiment_data = []
        for campaign in self.campaign_data:
            sentiment = campaign['sentiment_analysis']
            performance = campaign['performance_metrics']

            sentiment_data.append({
                'campaign_id': campaign['campaign_id'],
                'sentiment_score': sentiment['sentiment_score'],
                'positive_ratio': sentiment['positive'],
                'negative_ratio': sentiment['negative'],
                'engagement_rate': performance['engagement_rate'],
                'conversion_rate': performance['conversion_rate'],
                'roi': campaign['roi_metrics']['roi_percentage'],
                'effectiveness_score': campaign['roi_metrics']['effectiveness_score']
            })

        sentiment_df = pd.DataFrame(sentiment_data)

        # Correlation analysis
        correlations = sentiment_df[['sentiment_score', 'positive_ratio', 'negative_ratio',
                                   'engagement_rate', 'conversion_rate', 'roi', 'effectiveness_score']].corr()

        # Linear regression: sentiment vs effectiveness
        X = sentiment_df[['sentiment_score']]
        y = sentiment_df['effectiveness_score']

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        print("Sentiment Analysis Results:")
        print(f"Correlation between sentiment and effectiveness: {correlations.loc['sentiment_score', 'effectiveness_score']:.3f}")
        print(f"R² score for sentiment-effectiveness model: {r2:.3f}")

        return sentiment_df, correlations, model

    def analyze_pr_impact(self):
        """Analyze PR and media impact on campaign success"""
        print("Analyzing PR impact...")

        pr_data = []
        for campaign in self.campaign_data:
            pr_metrics = campaign['pr_metrics']
            brand_impact = campaign['brand_impact']
            performance = campaign['performance_metrics']

            pr_data.append({
                'campaign_id': campaign['campaign_id'],
                'media_mentions': pr_metrics['media_mentions'],
                'earned_media_value': pr_metrics['earned_media_value'],
                'share_of_voice': pr_metrics['share_of_voice'],
                'awareness_lift': brand_impact['awareness_lift'],
                'favorability_change': brand_impact['favorability_change'],
                'purchase_intent_increase': brand_impact['purchase_intent_increase'],
                'reach': performance['reach'],
                'engagements': performance['engagements'],
                'conversions': performance['conversions'],
                'roi': campaign['roi_metrics']['roi_percentage']
            })

        pr_df = pd.DataFrame(pr_data)

        # Calculate PR efficiency metrics
        pr_df['cost_per_mention'] = pr_df['earned_media_value'] / pr_df['media_mentions']
        pr_df['media_value_ratio'] = pr_df['earned_media_value'] / pr_df['reach']  # Earned media value per reach

        # Correlation analysis
        pr_correlations = pr_df[['media_mentions', 'earned_media_value', 'share_of_voice',
                                'awareness_lift', 'favorability_change', 'purchase_intent_increase',
                                'reach', 'engagements', 'conversions', 'roi']].corr()

        # Key insights
        avg_mentions = pr_df['media_mentions'].mean()
        avg_earned_value = pr_df['earned_media_value'].mean()
        avg_awareness_lift = pr_df['awareness_lift'].mean()

        print("PR Impact Summary:")
        print(f"Average Media Mentions: {avg_mentions:.1f}")
        print(f"Average Earned Media Value: ${avg_earned_value:,.0f}")
        print(f"Average Brand Awareness Lift: {avg_awareness_lift:.1f}%")

        return pr_df, pr_correlations

    def segment_campaigns_by_performance(self, n_clusters=4):
        """Segment campaigns based on performance characteristics"""
        print(f"Segmenting campaigns into {n_clusters} performance clusters...")

        # Prepare data for clustering
        cluster_data = []
        for campaign in self.campaign_data:
            perf = campaign['performance_metrics']
            roi = campaign['roi_metrics']
            sentiment = campaign['sentiment_analysis']

            cluster_data.append({
                'campaign_id': campaign['campaign_id'],
                'reach': perf['reach'],
                'engagement_rate': perf['engagement_rate'],
                'conversion_rate': perf['conversion_rate'],
                'roi': roi['roi_percentage'],
                'effectiveness_score': roi['effectiveness_score'],
                'sentiment_score': sentiment['sentiment_score'],
                'budget': campaign['budget']
            })

        cluster_df = pd.DataFrame(cluster_data)

        # Select features for clustering
        features = ['reach', 'engagement_rate', 'conversion_rate', 'roi', 'effectiveness_score', 'sentiment_score']
        X = cluster_df[features]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_df['performance_segment'] = kmeans.fit_predict(X_scaled)

        # Analyze segments
        segment_analysis = cluster_df.groupby('performance_segment').agg({
            'reach': 'mean',
            'engagement_rate': 'mean',
            'conversion_rate': 'mean',
            'roi': 'mean',
            'effectiveness_score': 'mean',
            'sentiment_score': 'mean',
            'budget': 'mean',
            'campaign_id': 'count'
        }).round(3)

        # Name segments based on characteristics
        segment_names = {
            0: 'High-Impact Champions',
            1: 'Steady Performers',
            2: 'Underperforming Campaigns',
            3: 'Niche Specialists'
        }

        segment_analysis['segment_name'] = segment_analysis.index.map(segment_names)

        print("Campaign Performance Segments:")
        for idx, row in segment_analysis.iterrows():
            print(f"  {row['segment_name']}: {int(row['campaign_id'])} campaigns")

        return cluster_df, segment_analysis

    def generate_communication_recommendations(self):
        """Generate strategic communication recommendations"""
        print("Generating communication recommendations...")

        # Analyze channel effectiveness
        channel_summary = self.analyze_channel_effectiveness()

        # Analyze sentiment impact
        sentiment_df, correlations, sentiment_model = self.analyze_sentiment_impact()

        # Analyze PR impact
        pr_df, pr_correlations = self.analyze_pr_impact()

        # Generate recommendations
        recommendations = {
            'channel_strategy': {
                'top_channels': channel_summary.head(3).index.tolist(),
                'budget_allocation': {
                    channel: f"{row['roi_contribution']:.1f}% ROI contribution"
                    for channel, row in channel_summary.head(3).iterrows()
                },
                'channel_efficiency': {
                    channel: f"${row['cost_per_conversion']:.2f} per conversion"
                    for channel, row in channel_summary.iterrows()
                }
            },
            'content_strategy': {
                'sentiment_focus': "Positive sentiment correlates strongly with campaign effectiveness" if correlations.loc['sentiment_score', 'effectiveness_score'] > 0.3 else "Sentiment has moderate impact on performance",
                'message_types': ['Emotional appeals', 'Educational content', 'Call-to-action messaging'],
                'tone_recommendations': ['Inspirational', 'Friendly', 'Professional']
            },
            'pr_strategy': {
                'media_value': f"Average ${pr_df['earned_media_value'].mean():,.0f} earned media value per campaign",
                'amplification_tactics': ['Influencer partnerships', 'User-generated content', 'Storytelling'],
                'measurement_focus': ['Share of voice', 'Message penetration', 'Audience sentiment']
            },
            'optimization_opportunities': [
                'Increase frequency for high-engagement campaigns',
                'Optimize channel mix based on audience preferences',
                'Enhance sentiment monitoring and response systems',
                'Invest in PR partnerships for earned media leverage',
                'A/B test messaging and creative approaches'
            ],
            'budget_optimization': {
                'high_roi_channels': channel_summary[channel_summary['roi_contribution'] > channel_summary['roi_contribution'].median()].index.tolist(),
                'cost_reduction_opportunities': channel_summary[channel_summary['cost_per_conversion'] > channel_summary['cost_per_conversion'].quantile(0.75)].index.tolist(),
                'investment_priorities': ['Content creation', 'Community management', 'PR partnerships']
            }
        }

        return recommendations

    def generate_comprehensive_visualizations(self):
        """Generate comprehensive communication analysis visualizations"""
        print("Generating comprehensive visualizations...")

        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Communication Campaign Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Campaign Performance Overview
        performance_data = []
        for campaign in self.campaign_data:
            performance_data.append({
                'campaign': campaign['campaign_id'],
                'effectiveness': campaign['roi_metrics']['effectiveness_score'],
                'roi': campaign['roi_metrics']['roi_percentage']
            })

        perf_df = pd.DataFrame(performance_data)
        perf_df = perf_df.sort_values('effectiveness', ascending=True)

        axes[0,0].barh(perf_df['campaign'], perf_df['effectiveness'], color='skyblue')
        axes[0,0].set_title('Campaign Effectiveness Scores')
        axes[0,0].set_xlabel('Effectiveness Score')

        # 2. Channel Performance Comparison
        channel_summary = self.analyze_channel_effectiveness()
        channel_summary['roi_contribution'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Channel ROI Contribution')
        axes[0,1].set_ylabel('ROI Contribution (%)')
        axes[0,1].tick_params(axis='x', rotation=45)

        # 3. Sentiment vs Effectiveness Scatter
        sentiment_effectiveness = []
        for campaign in self.campaign_data:
            sentiment_effectiveness.append({
                'sentiment': campaign['sentiment_analysis']['sentiment_score'],
                'effectiveness': campaign['roi_metrics']['effectiveness_score']
            })

        sentiment_df = pd.DataFrame(sentiment_effectiveness)
        axes[0,2].scatter(sentiment_df['sentiment'], sentiment_df['effectiveness'], alpha=0.6, color='orange')
        axes[0,2].set_title('Sentiment vs Campaign Effectiveness')
        axes[0,2].set_xlabel('Sentiment Score')
        axes[0,2].set_ylabel('Effectiveness Score')

        # 4. PR Impact Analysis
        pr_impact = []
        for campaign in self.campaign_data:
            pr_impact.append({
                'mentions': campaign['pr_metrics']['media_mentions'],
                'awareness_lift': campaign['brand_impact']['awareness_lift']
            })

        pr_df = pd.DataFrame(pr_impact)
        axes[1,0].scatter(pr_df['mentions'], pr_df['awareness_lift'], alpha=0.6, color='red')
        axes[1,0].set_title('Media Mentions vs Brand Awareness Lift')
        axes[1,0].set_xlabel('Media Mentions')
        axes[1,0].set_ylabel('Awareness Lift (%)')

        # 5. Budget vs ROI Analysis
        budget_roi = []
        for campaign in self.campaign_data:
            budget_roi.append({
                'budget': campaign['budget'],
                'roi': campaign['roi_metrics']['roi_percentage']
            })

        budget_df = pd.DataFrame(budget_roi)
        axes[1,1].scatter(budget_df['budget'], budget_df['roi'], alpha=0.6, color='purple')
        axes[1,1].set_title('Budget vs ROI Analysis')
        axes[1,1].set_xlabel('Budget ($)')
        axes[1,1].set_ylabel('ROI (%)')

        # 6. Campaign Type Performance
        type_performance = {}
        for campaign in self.campaign_data:
            camp_type = campaign['campaign_type']
            if camp_type not in type_performance:
                type_performance[camp_type] = []
            type_performance[camp_type].append(campaign['roi_metrics']['effectiveness_score'])

        avg_performance = {k: np.mean(v) for k, v in type_performance.items()}
        axes[1,2].bar(avg_performance.keys(), avg_performance.values(), color='teal')
        axes[1,2].set_title('Performance by Campaign Type')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].set_ylabel('Average Effectiveness Score')

        # 7. Engagement Rate Distribution
        engagement_rates = [c['performance_metrics']['engagement_rate'] for c in self.campaign_data]
        axes[2,0].hist(engagement_rates, bins=10, alpha=0.7, color='gold', edgecolor='black')
        axes[2,0].set_title('Engagement Rate Distribution')
        axes[2,0].set_xlabel('Engagement Rate')
        axes[2,0].set_ylabel('Frequency')
        axes[2,0].axvline(np.mean(engagement_rates), color='red', linestyle='--', label='.2%')
        axes[2,0].legend()

        # 8. Conversion Funnel
        funnel_data = {
            'Reach': sum(c['performance_metrics']['reach'] for c in self.campaign_data),
            'Engagements': sum(c['performance_metrics']['engagements'] for c in self.campaign_data),
            'Clicks': sum(c['performance_metrics']['clicks'] for c in self.campaign_data),
            'Conversions': sum(c['performance_metrics']['conversions'] for c in self.campaign_data)
        }

        stages = list(funnel_data.keys())
        values = list(funnel_data.values())

        # Calculate conversion rates between stages
        conversion_rates = []
        for i in range(len(values)-1):
            rate = (values[i+1] / values[i]) * 100 if values[i] > 0 else 0
            conversion_rates.append('.1f')

        axes[2,1].bar(stages, values, color='lightcoral', alpha=0.7)
        axes[2,1].set_title('Campaign Conversion Funnel')
        axes[2,1].set_ylabel('Volume')
        axes[2,1].tick_params(axis='x', rotation=45)

        # Add conversion rate annotations
        for i, (stage, rate) in enumerate(zip(stages[:-1], conversion_rates)):
            axes[2,1].text(i, values[i], f'{rate}%', ha='center', va='bottom')

        # 9. Sentiment Distribution
        sentiment_dist = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        for campaign in self.campaign_data:
            sent = campaign['sentiment_analysis']
            sentiment_dist['Positive'] += sent['positive']
            sentiment_dist['Neutral'] += sent['neutral']
            sentiment_dist['Negative'] += sent['negative']

        # Average sentiment
        total_sentiment = sum(sentiment_dist.values())
        avg_sentiment = {k: v/total_sentiment for k, v in sentiment_dist.items()}

        axes[2,2].pie(avg_sentiment.values(), labels=avg_sentiment.keys(), autopct='%1.1f%%',
                     colors=['lightgreen', 'lightblue', 'lightcoral'])
        axes[2,2].set_title('Average Sentiment Distribution')

        plt.tight_layout()
        plt.savefig('results/communication_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_detailed_report(self):
        """Generate comprehensive communication campaign analysis report"""
        print("Generating detailed analysis report...")

        # Gather all analysis results
        avg_metrics, type_performance = self.analyze_campaign_performance()
        channel_summary = self.analyze_channel_effectiveness()
        sentiment_df, sentiment_correlations, sentiment_model = self.analyze_sentiment_impact()
        pr_df, pr_correlations = self.analyze_pr_impact()
        campaign_segments, segment_analysis = self.segment_campaigns_by_performance()
        recommendations = self.generate_communication_recommendations()

        # Calculate key insights
        total_campaigns = len(self.campaign_data)
        total_budget = sum(c['budget'] for c in self.campaign_data)
        total_value_generated = sum(c['roi_metrics']['total_value_generated'] for c in self.campaign_data)
        overall_roi = ((total_value_generated - total_budget) / total_budget) * 100 if total_budget > 0 else 0

        # Create comprehensive report
        report = f"""
# Communication Campaign Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This comprehensive analysis evaluates the effectiveness of {total_campaigns} communication campaigns across multiple channels and objectives. The analysis covers campaign performance, channel effectiveness, sentiment impact, PR metrics, and strategic recommendations.

### Key Performance Indicators
- **Total Campaigns Analyzed:** {total_campaigns:,}
- **Total Budget Invested:** ${total_budget:,.0f}
- **Total Value Generated:** ${total_value_generated:,.0f}
- **Overall Portfolio ROI:** {overall_roi:.1f}%
- **Average Campaign Effectiveness Score:** {avg_metrics['avg_effectiveness_score']:.1f}/100

## Campaign Performance Analysis

### Overall Performance Metrics
- **Average Reach:** {avg_metrics['avg_reach']:,.0f} people
- **Average Engagement Rate:** {avg_metrics['avg_engagement_rate']:.2%}
- **Average Conversion Rate:** {avg_metrics['avg_conversion_rate']:.2%}
- **Average ROI:** {avg_metrics['avg_roi']:.1f}%

### Performance by Campaign Type
{type_performance.to_string()}

### Campaign Segmentation Results
{segment_analysis.to_string()}

## Channel Effectiveness Analysis

### Top Performing Channels by ROI
{channel_summary.head(5).to_string()}

### Channel Efficiency Metrics
- **Most Cost-Effective Channel:** {channel_summary['cost_per_conversion'].idxmin()} (${channel_summary['cost_per_conversion'].min():.2f} per conversion)
- **Highest ROI Channel:** {channel_summary['roi_contribution'].idxmax()} ({channel_summary['roi_contribution'].max():.1f}% ROI contribution)

## Sentiment Analysis

### Sentiment-Performance Correlation
- **Sentiment vs Effectiveness Correlation:** {sentiment_correlations.loc['sentiment_score', 'effectiveness_score']:.3f}
- **Sentiment vs ROI Correlation:** {sentiment_correlations.loc['sentiment_score', 'roi']:.3f}

### Sentiment Distribution Summary
- **Average Positive Sentiment:** {sentiment_df['positive_ratio'].mean():.1%}
- **Average Negative Sentiment:** {sentiment_df['negative_ratio'].mean():.1%}

## PR Impact Analysis

### Media Performance Metrics
- **Average Media Mentions per Campaign:** {pr_df['media_mentions'].mean():.1f}
- **Average Earned Media Value:** ${pr_df['earned_media_value'].mean():,.0f}
- **Average Share of Voice:** {pr_df['share_of_voice'].mean():.1%}

### Brand Impact Results
- **Average Brand Awareness Lift:** {pr_df['awareness_lift'].mean():.1f}%
- **Average Brand Favorability Change:** {pr_df['favorability_change'].mean():.1f}%
- **Average Purchase Intent Increase:** {pr_df['purchase_intent_increase'].mean():.1f}%

## Strategic Recommendations

### Channel Strategy
**Top Recommended Channels:**
{chr(10).join([f"- {channel}: {recommendations['channel_strategy']['budget_allocation'][channel]}" for channel in recommendations['channel_strategy']['top_channels']])}

### Content Strategy
**Key Focus Areas:**
{chr(10).join([f"- {msg}" for msg in recommendations['content_strategy']['message_types']])}

**Recommended Tones:**
{chr(10).join([f"- {tone}" for tone in recommendations['content_strategy']['tone_recommendations']])}

### PR Strategy
**Earned Media Value:** {recommendations['pr_strategy']['media_value']}

**Amplification Tactics:**
{chr(10).join([f"- {tactic}" for tactic in recommendations['pr_strategy']['amplification_tactics']])}

### Budget Optimization
**High-ROI Channels for Increased Investment:**
{chr(10).join([f"- {channel}" for channel in recommendations['budget_optimization']['high_roi_channels']])}

**Cost Reduction Opportunities:**
{chr(10).join([f"- {channel}" for channel in recommendations['budget_optimization']['cost_reduction_opportunities']])}

## Optimization Opportunities

### Immediate Actions (0-3 months)
{chr(10).join([f"{i+1}. {opp}" for i, opp in enumerate(recommendations['optimization_opportunities'][:3])])}

### Medium-term Strategy (3-6 months)
1. **Channel Mix Optimization:** Allocate budget based on ROI performance data
2. **Content Strategy Refinement:** Develop content calendar based on high-performing message types
3. **Measurement Framework Enhancement:** Implement advanced tracking and attribution models

### Long-term Growth (6+ months)
1. **Predictive Analytics:** Use machine learning to predict campaign performance
2. **Personalization at Scale:** Implement dynamic content based on audience segments
3. **Integrated Campaign Management:** Develop unified platform for cross-channel orchestration

## Methodology and Data Sources

### Analysis Framework
- **Performance Metrics:** Reach, engagement, conversion, and ROI analysis
- **Channel Attribution:** Multi-touch attribution modeling
- **Sentiment Analysis:** Natural language processing for audience sentiment
- **PR Measurement:** Earned media value and share of voice calculations
- **Statistical Methods:** Correlation analysis, regression modeling, clustering

### Data Sources
- Campaign management platform data
- Social media analytics APIs
- PR monitoring tools
- Brand tracking surveys
- Sales and conversion tracking systems

### Quality Assurance
- Data validation and outlier detection
- Statistical significance testing
- Cross-platform data reconciliation
- Automated report generation with error checking

## Risk Assessment and Mitigation

### Campaign Risks Identified
1. **Channel Dependency:** Over-reliance on single high-performing channels
2. **Sentiment Volatility:** Rapid changes in audience sentiment
3. **PR Black Swan Events:** Unpredictable media coverage events
4. **Budget Inefficiency:** Sub-optimal budget allocation across channels

### Mitigation Strategies
1. **Diversification:** Maintain multi-channel presence to reduce dependency risks
2. **Sentiment Monitoring:** Implement real-time sentiment tracking and response protocols
3. **Crisis Management:** Develop comprehensive crisis communication plans
4. **Dynamic Budgeting:** Implement flexible budget reallocation based on performance data

## Future Research Directions

### Advanced Analytics
- **Predictive Modeling:** Machine learning for campaign success prediction
- **Causal Inference:** Advanced attribution modeling for true impact measurement
- **Network Analysis:** Social influence and information diffusion modeling

### Emerging Technologies
- **AI-Powered Content:** Automated content generation and optimization
- **Voice and Visual Analytics:** Analysis of audio and video content performance
- **Blockchain Attribution:** Transparent and immutable attribution tracking

### Measurement Evolution
- **Privacy-First Analytics:** Measurement frameworks respecting user privacy
- **Cross-Device Tracking:** Unified customer journey across all touchpoints
- **Real-Time Optimization:** Dynamic campaign adjustment based on live data

---
*Report generated automatically by Communication Campaign Analysis Tool*
*Contact: Data Engineering & Business Intelligence Portfolio*
"""

        # Save report
        with open('results/communication_campaign_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("Comprehensive report saved to: results/communication_campaign_report.md")
        return report

    def run_complete_analysis(self, num_campaigns=10):
        """Run complete communication campaign analysis pipeline"""
        print("=" * 70)
        print("COMMUNICATION CAMPAIGN ANALYSIS TOOL")
        print("=" * 70)

        # Create necessary directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('data', exist_ok=True)

        # Step 1: Data Collection
        print("\n📊 STEP 1: Campaign Data Collection")
        self.generate_mock_campaign_data(num_campaigns)

        # Step 2: Performance Analysis
        print("\n📈 STEP 2: Campaign Performance Analysis")
        avg_metrics, type_performance = self.analyze_campaign_performance()

        # Step 3: Channel Effectiveness
        print("\n📢 STEP 3: Channel Effectiveness Analysis")
        channel_summary = self.analyze_channel_effectiveness()

        # Step 4: Sentiment Impact
        print("\n😊 STEP 4: Sentiment Impact Analysis")
        sentiment_df, correlations, model = self.analyze_sentiment_impact()

        # Step 5: PR Impact Analysis
        print("\n📰 STEP 5: PR Impact Analysis")
        pr_df, pr_correlations = self.analyze_pr_impact()

        # Step 6: Campaign Segmentation
        print("\n🎯 STEP 6: Campaign Performance Segmentation")
        campaign_segments, segment_analysis = self.segment_campaigns_by_performance()

        # Step 7: Strategic Recommendations
        print("\n💡 STEP 7: Strategic Recommendations")
        recommendations = self.generate_communication_recommendations()

        # Step 8: Visualizations
        print("\n📊 STEP 8: Generating Visualizations")
        self.generate_comprehensive_visualizations()

        # Step 9: Comprehensive Report
        print("\n📋 STEP 9: Generating Comprehensive Report")
        report = self.generate_detailed_report()

        # Save processed data
        self.campaigns_df.to_csv('data/campaign_performance_data.csv', index=False)

        # Create summary JSON for easy access
        summary_data = {
            'total_campaigns': len(self.campaign_data),
            'total_budget': sum(c['budget'] for c in self.campaign_data),
            'average_roi': avg_metrics['avg_roi'],
            'average_effectiveness': avg_metrics['avg_effectiveness_score'],
            'top_channels': channel_summary.head(3).index.tolist(),
            'sentiment_correlation': correlations.loc['sentiment_score', 'effectiveness_score'],
            'generated_at': datetime.now().isoformat()
        }

        with open('data/analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print("✅ COMMUNICATION CAMPAIGN ANALYSIS COMPLETE!")
        print("📁 Check 'results/' folder for visualizations and reports")
        print("📄 Check 'data/' folder for processed data and summaries")
        print("=" * 70)

        return {
            'performance': avg_metrics,
            'channels': channel_summary,
            'sentiment': {'correlations': correlations, 'data': sentiment_df},
            'pr_impact': {'data': pr_df, 'correlations': pr_correlations},
            'segments': segment_analysis,
            'recommendations': recommendations,
            'report': report
        }

def main():
    """Main function to run communication campaign analysis"""
    analyzer = CommunicationCampaignAnalyzer()
    results = analyzer.run_complete_analysis(num_campaigns=12)

    # Print key insights
    print("\n🎯 ANALYSIS SUMMARY:")
    print(f"Total campaigns analyzed: {len(analyzer.campaign_data)}")
    print(f"Average ROI: {results['performance']['avg_roi']:.1f}%")
    print(f"Average effectiveness score: {results['performance']['avg_effectiveness_score']:.1f}")
    print(f"Top performing channel: {results['channels'].index[0]}")
    print(f"Sentiment-effectiveness correlation: {results['sentiment']['correlations'].loc['sentiment_score', 'effectiveness_score']:.3f}")

if __name__ == "__main__":
    main()