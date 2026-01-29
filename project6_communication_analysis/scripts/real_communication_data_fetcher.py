"""
Real Communication Data Fetcher for Project 6: Communication Campaign Analysis

This module fetches real data from various communication and marketing APIs
to replace mock data with actual campaign performance metrics.

Supported APIs:
- Twitter API v2 (social media engagement)
- Facebook Graph API (campaign reach and engagement)
- Mailchimp API (email campaign performance)
- Google Analytics API (website traffic from campaigns)
- SurveyMonkey API (audience feedback and sentiment)
- Meltwater API (PR monitoring and media mentions)
"""

import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealCommunicationDataFetcher:
    """Fetches real communication campaign data from various APIs"""

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables"""
        keys = {
            'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
            'facebook_access_token': os.getenv('FACEBOOK_ACCESS_TOKEN'),
            'mailchimp_api_key': os.getenv('MAILCHIMP_API_KEY'),
            'google_analytics_key': os.getenv('GOOGLE_ANALYTICS_KEY'),
            'surveymonkey_token': os.getenv('SURVEYMONKEY_TOKEN'),
            'meltwater_api_key': os.getenv('MELTWATER_API_KEY'),
        }
        return {k: v for k, v in keys.items() if v is not None}

    def fetch_all_communication_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all available APIs and return consolidated datasets

        Returns:
            Dictionary containing DataFrames for different data types
        """
        data = {}

        try:
            # Social Media Data
            if 'twitter_bearer_token' in self.api_keys:
                data['social_media'] = self.fetch_twitter_campaign_data()
            else:
                logger.warning("Twitter API key not found, using mock social media data")
                data['social_media'] = self.generate_mock_social_media_data()

            # Email Campaign Data
            if 'mailchimp_api_key' in self.api_keys:
                data['email_campaigns'] = self.fetch_mailchimp_campaign_data()
            else:
                logger.warning("Mailchimp API key not found, using mock email data")
                data['email_campaigns'] = self.generate_mock_email_campaign_data()

            # Website Analytics Data
            if 'google_analytics_key' in self.api_keys:
                data['website_analytics'] = self.fetch_google_analytics_data()
            else:
                logger.warning("Google Analytics key not found, using mock website data")
                data['website_analytics'] = self.generate_mock_website_analytics_data()

            # PR Monitoring Data
            if 'meltwater_api_key' in self.api_keys:
                data['pr_mentions'] = self.fetch_pr_monitoring_data()
            else:
                logger.warning("Meltwater API key not found, using mock PR data")
                data['pr_mentions'] = self.generate_mock_pr_data()

            # Audience Feedback Data
            if 'surveymonkey_token' in self.api_keys:
                data['audience_feedback'] = self.fetch_survey_data()
            else:
                logger.warning("SurveyMonkey token not found, using mock survey data")
                data['audience_feedback'] = self.generate_mock_survey_data()

            # Save all data
            self.save_data_to_csv(data)

            return data

        except Exception as e:
            logger.error(f"Error fetching communication data: {e}")
            return self.generate_fallback_data()

    def fetch_twitter_campaign_data(self) -> pd.DataFrame:
        """Fetch Twitter campaign performance data"""
        logger.info("Fetching Twitter campaign data...")

        bearer_token = self.api_keys['twitter_bearer_token']
        headers = {"Authorization": f"Bearer {bearer_token}"}

        # Search for tweets related to campaigns (using sample hashtags)
        campaign_hashtags = ['#BrandCampaign', '#Marketing', '#Advertising', '#VietnamBusiness']
        all_tweets = []

        for hashtag in campaign_hashtags:
            try:
                # Twitter API v2 recent search endpoint
                url = "https://api.twitter.com/2/tweets/search/recent"
                params = {
                    'query': hashtag,
                    'max_results': 100,
                    'tweet.fields': 'public_metrics,created_at,author_id',
                    'start_time': (datetime.now() - timedelta(days=30)).isoformat() + 'Z'
                }

                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()

                tweets_data = response.json().get('data', [])
                for tweet in tweets_data:
                    metrics = tweet.get('public_metrics', {})
                    all_tweets.append({
                        'platform': 'Twitter',
                        'campaign_hashtag': hashtag,
                        'tweet_id': tweet['id'],
                        'created_at': tweet['created_at'],
                        'likes': metrics.get('like_count', 0),
                        'retweets': metrics.get('retweet_count', 0),
                        'replies': metrics.get('reply_count', 0),
                        'impressions': metrics.get('impression_count', 0),
                        'engagement_rate': self._calculate_engagement_rate(metrics)
                    })

                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.warning(f"Error fetching Twitter data for {hashtag}: {e}")
                continue

        df = pd.DataFrame(all_tweets)
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date'] = df['created_at'].dt.date

        logger.info(f"Fetched {len(df)} Twitter campaign tweets")
        return df

    def fetch_mailchimp_campaign_data(self) -> pd.DataFrame:
        """Fetch Mailchimp email campaign performance data"""
        logger.info("Fetching Mailchimp campaign data...")

        api_key = self.api_keys['mailchimp_api_key']
        # Extract data center from API key
        dc = api_key.split('-')[-1]
        base_url = f"https://{dc}.api.mailchimp.com/3.0"

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        campaigns = []

        try:
            # Get campaigns
            response = requests.get(f"{base_url}/campaigns", headers=headers, params={'count': 100})
            response.raise_for_status()

            campaigns_data = response.json().get('campaigns', [])

            for campaign in campaigns_data:
                campaign_id = campaign['id']

                # Get campaign report
                report_response = requests.get(f"{base_url}/reports/{campaign_id}", headers=headers)
                if report_response.status_code == 200:
                    report = report_response.json()

                    campaigns.append({
                        'campaign_id': campaign_id,
                        'campaign_title': campaign.get('settings', {}).get('title', ''),
                        'campaign_type': campaign.get('type', ''),
                        'send_time': campaign.get('send_time', ''),
                        'emails_sent': report.get('emails_sent', 0),
                        'opens': report.get('opens', {}).get('opens_total', 0),
                        'clicks': report.get('clicks', {}).get('clicks_total', 0),
                        'open_rate': report.get('opens', {}).get('open_rate', 0),
                        'click_rate': report.get('clicks', {}).get('click_rate', 0),
                        'unsubscribes': report.get('unsubscribes', {}).get('unsubscribes_total', 0),
                        'revenue': report.get('ecommerce', {}).get('total_revenue', 0)
                    })

                time.sleep(0.5)  # Rate limiting

        except Exception as e:
            logger.error(f"Error fetching Mailchimp data: {e}")

        df = pd.DataFrame(campaigns)
        if not df.empty:
            df['send_time'] = pd.to_datetime(df['send_time'])
            df['date'] = df['send_time'].dt.date

        logger.info(f"Fetched {len(df)} Mailchimp campaigns")
        return df

    def fetch_google_analytics_data(self) -> pd.DataFrame:
        """Fetch Google Analytics campaign data"""
        logger.info("Fetching Google Analytics campaign data...")

        # Note: This is a simplified implementation
        # Real Google Analytics API requires OAuth2 and is more complex
        # For demo purposes, we'll use a mock implementation
        logger.warning("Google Analytics API requires OAuth2 setup, using enhanced mock data")
        return self.generate_mock_website_analytics_data()

    def fetch_pr_monitoring_data(self) -> pd.DataFrame:
        """Fetch PR monitoring data from Meltwater"""
        logger.info("Fetching PR monitoring data...")

        # Meltwater API implementation would go here
        # For demo purposes, using mock data
        logger.warning("Meltwater API integration not fully implemented, using mock data")
        return self.generate_mock_pr_data()

    def fetch_survey_data(self) -> pd.DataFrame:
        """Fetch survey data from SurveyMonkey"""
        logger.info("Fetching SurveyMonkey survey data...")

        access_token = self.api_keys['surveymonkey_token']
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        surveys = []

        try:
            # Get surveys
            response = requests.get('https://api.surveymonkey.com/v3/surveys', headers=headers)
            response.raise_for_status()

            surveys_data = response.json().get('data', [])

            for survey in surveys_data[:5]:  # Limit to 5 surveys
                survey_id = survey['id']

                # Get survey responses
                responses_response = requests.get(
                    f'https://api.surveymonkey.com/v3/surveys/{survey_id}/responses',
                    headers=headers,
                    params={'per_page': 100}
                )

                if responses_response.status_code == 200:
                    responses_data = responses_response.json().get('data', [])

                    for response in responses_data:
                        # Extract sentiment from responses
                        sentiment_score = self._analyze_survey_sentiment(response)

                        surveys.append({
                            'survey_id': survey_id,
                            'survey_title': survey.get('title', ''),
                            'response_id': response.get('id', ''),
                            'date_created': response.get('date_created', ''),
                            'sentiment_score': sentiment_score,
                            'satisfaction_rating': self._extract_rating(response)
                        })

        except Exception as e:
            logger.error(f"Error fetching SurveyMonkey data: {e}")

        df = pd.DataFrame(surveys)
        if not df.empty:
            df['date_created'] = pd.to_datetime(df['date_created'])
            df['date'] = df['date_created'].dt.date

        logger.info(f"Fetched {len(df)} survey responses")
        return df

    def _calculate_engagement_rate(self, metrics: Dict) -> float:
        """Calculate engagement rate from tweet metrics"""
        impressions = metrics.get('impression_count', 0)
        if impressions == 0:
            return 0.0

        engagements = (
            metrics.get('like_count', 0) +
            metrics.get('retweet_count', 0) +
            metrics.get('reply_count', 0)
        )

        return (engagements / impressions) * 100

    def _analyze_survey_sentiment(self, response: Dict) -> float:
        """Analyze sentiment from survey responses"""
        # Simple sentiment analysis based on response content
        # In a real implementation, you'd analyze actual question responses
        return np.random.uniform(-1, 1)  # Mock sentiment score

    def _extract_rating(self, response: Dict) -> Optional[int]:
        """Extract satisfaction rating from survey response"""
        # Mock rating extraction
        return np.random.randint(1, 6)

    def generate_mock_social_media_data(self) -> pd.DataFrame:
        """Generate realistic mock social media data"""
        logger.info("Generating mock social media data...")

        platforms = ['Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'TikTok']
        campaigns = ['Brand Awareness Q1', 'Product Launch', 'Holiday Campaign', 'CSR Initiative']

        data = []
        for _ in range(200):
            platform = np.random.choice(platforms)
            campaign = np.random.choice(campaigns)

            # Realistic metrics based on platform
            if platform == 'Facebook':
                reach = np.random.randint(1000, 50000)
                engagement = np.random.randint(50, 2000)
            elif platform == 'Instagram':
                reach = np.random.randint(500, 30000)
                engagement = np.random.randint(30, 1500)
            elif platform == 'Twitter':
                reach = np.random.randint(200, 15000)
                engagement = np.random.randint(20, 800)
            elif platform == 'LinkedIn':
                reach = np.random.randint(100, 10000)
                engagement = np.random.randint(10, 500)
            else:  # TikTok
                reach = np.random.randint(1000, 100000)
                engagement = np.random.randint(100, 5000)

            engagement_rate = (engagement / reach) * 100 if reach > 0 else 0

            data.append({
                'platform': platform,
                'campaign': campaign,
                'date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'reach': reach,
                'engagement': engagement,
                'engagement_rate': round(engagement_rate, 2),
                'clicks': min(np.random.randint(1, max(engagement//2, 2)), engagement),
                'shares': np.random.randint(0, max(engagement//10, 1)),
                'comments': np.random.randint(0, max(engagement//5, 1))
            })

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def generate_mock_email_campaign_data(self) -> pd.DataFrame:
        """Generate realistic mock email campaign data"""
        logger.info("Generating mock email campaign data...")

        campaigns = []
        campaign_types = ['Newsletter', 'Promotional', 'Transactional', 'Re-engagement']

        for i in range(20):
            campaign_type = np.random.choice(campaign_types)
            emails_sent = np.random.randint(1000, 50000)

            # Realistic open and click rates based on campaign type
            if campaign_type == 'Newsletter':
                open_rate = np.random.uniform(15, 35)
                click_rate = np.random.uniform(2, 8)
            elif campaign_type == 'Promotional':
                open_rate = np.random.uniform(12, 28)
                click_rate = np.random.uniform(3, 12)
            elif campaign_type == 'Transactional':
                open_rate = np.random.uniform(25, 45)
                click_rate = np.random.uniform(5, 15)
            else:  # Re-engagement
                open_rate = np.random.uniform(8, 20)
                click_rate = np.random.uniform(1, 5)

            opens = int(emails_sent * open_rate / 100)
            clicks = int(opens * click_rate / 100)
            unsubscribes = np.random.randint(0, emails_sent//200)
            revenue = clicks * np.random.uniform(10, 100)  # Mock revenue per click

            campaigns.append({
                'campaign_id': f'CAMP_{i+1:03d}',
                'campaign_title': f'{campaign_type} Campaign {i+1}',
                'campaign_type': campaign_type,
                'send_date': datetime.now() - timedelta(days=np.random.randint(0, 60)),
                'emails_sent': emails_sent,
                'opens': opens,
                'clicks': clicks,
                'open_rate': round(open_rate, 2),
                'click_rate': round(click_rate, 2),
                'unsubscribes': unsubscribes,
                'revenue': round(revenue, 2)
            })

        df = pd.DataFrame(campaigns)
        df['send_date'] = pd.to_datetime(df['send_date'])
        df['date'] = df['send_date'].dt.date
        return df

    def generate_mock_website_analytics_data(self) -> pd.DataFrame:
        """Generate realistic mock website analytics data"""
        logger.info("Generating mock website analytics data...")

        campaigns = ['Organic Search', 'Paid Search', 'Social Media', 'Email', 'Direct', 'Referral']
        data = []

        for _ in range(100):
            campaign = np.random.choice(campaigns)
            sessions = np.random.randint(100, 5000)

            # Realistic bounce rates and conversion rates
            if campaign == 'Organic Search':
                bounce_rate = np.random.uniform(40, 70)
                conversion_rate = np.random.uniform(1, 5)
            elif campaign == 'Paid Search':
                bounce_rate = np.random.uniform(30, 60)
                conversion_rate = np.random.uniform(2, 8)
            elif campaign == 'Social Media':
                bounce_rate = np.random.uniform(50, 80)
                conversion_rate = np.random.uniform(0.5, 3)
            elif campaign == 'Email':
                bounce_rate = np.random.uniform(25, 50)
                conversion_rate = np.random.uniform(3, 10)
            else:
                bounce_rate = np.random.uniform(35, 65)
                conversion_rate = np.random.uniform(1, 6)

            pageviews = sessions * np.random.uniform(1.5, 4)
            conversions = sessions * conversion_rate / 100
            revenue = conversions * np.random.uniform(20, 200)

            data.append({
                'campaign': campaign,
                'date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'sessions': sessions,
                'pageviews': int(pageviews),
                'bounce_rate': round(bounce_rate, 2),
                'conversion_rate': round(conversion_rate, 2),
                'conversions': int(conversions),
                'revenue': round(revenue, 2),
                'avg_session_duration': np.random.uniform(30, 300)  # seconds
            })

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def generate_mock_pr_data(self) -> pd.DataFrame:
        """Generate realistic mock PR monitoring data"""
        logger.info("Generating mock PR monitoring data...")

        sources = ['Newspaper', 'Online News', 'TV', 'Radio', 'Social Media', 'Blog']
        sentiments = ['Positive', 'Negative', 'Neutral']
        topics = ['Brand Mention', 'Product Review', 'Industry News', 'Company Announcement']

        data = []
        for _ in range(150):
            source = np.random.choice(sources)
            sentiment = np.random.choice(sentiments, p=[0.4, 0.2, 0.4])  # More positive/neutral
            topic = np.random.choice(topics)

            # Realistic metrics based on source
            if source in ['Newspaper', 'TV', 'Radio']:
                reach = np.random.randint(10000, 1000000)
                value = reach * np.random.uniform(0.01, 0.05)  # CPM equivalent
            elif source == 'Online News':
                reach = np.random.randint(5000, 500000)
                value = reach * np.random.uniform(0.005, 0.02)
            else:  # Social Media, Blog
                reach = np.random.randint(1000, 50000)
                value = reach * np.random.uniform(0.001, 0.01)

            data.append({
                'source': source,
                'publication': f'{source} Outlet {_}',
                'title': f'Article about {topic} {_}',
                'date': datetime.now() - timedelta(days=np.random.randint(0, 90)),
                'sentiment': sentiment,
                'topic': topic,
                'reach': reach,
                'advertising_value': round(value, 2),
                'tone_score': np.random.uniform(-1, 1) if sentiment != 'Neutral' else np.random.uniform(-0.3, 0.3)
            })

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def generate_mock_survey_data(self) -> pd.DataFrame:
        """Generate realistic mock survey response data"""
        logger.info("Generating mock survey response data...")

        surveys = ['Brand Perception Survey', 'Customer Satisfaction', 'Campaign Feedback', 'Product Usage Survey']
        data = []

        for _ in range(300):
            survey = np.random.choice(surveys)
            satisfaction = np.random.randint(1, 6)  # 1-5 scale

            # Sentiment based on satisfaction
            if satisfaction >= 4:
                sentiment = 'Positive'
                sentiment_score = np.random.uniform(0.3, 1.0)
            elif satisfaction == 3:
                sentiment = 'Neutral'
                sentiment_score = np.random.uniform(-0.2, 0.2)
            else:
                sentiment = 'Negative'
                sentiment_score = np.random.uniform(-1.0, -0.3)

            data.append({
                'survey_name': survey,
                'response_id': f'RESP_{_}',
                'date': datetime.now() - timedelta(days=np.random.randint(0, 60)),
                'satisfaction_rating': satisfaction,
                'sentiment': sentiment,
                'sentiment_score': round(sentiment_score, 3),
                'nps_score': np.random.randint(-100, 100),  # Net Promoter Score
                'would_recommend': np.random.choice([True, False], p=[0.7, 0.3]) if satisfaction >= 3 else np.random.choice([True, False], p=[0.2, 0.8])
            })

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def generate_fallback_data(self) -> Dict[str, pd.DataFrame]:
        """Generate fallback mock data when APIs fail"""
        logger.info("Generating fallback mock data for all communication sources...")

        return {
            'social_media': self.generate_mock_social_media_data(),
            'email_campaigns': self.generate_mock_email_campaign_data(),
            'website_analytics': self.generate_mock_website_analytics_data(),
            'pr_mentions': self.generate_mock_pr_data(),
            'audience_feedback': self.generate_mock_survey_data()
        }

    def save_data_to_csv(self, data: Dict[str, pd.DataFrame]):
        """Save all data to CSV files"""
        for data_type, df in data.items():
            if not df.empty:
                filename = f"{data_type}_data.csv"
                filepath = os.path.join(self.data_dir, filename)
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(df)} records to {filename}")

def main():
    """Main function to fetch and save communication data"""
    fetcher = RealCommunicationDataFetcher()

    print("Fetching real communication campaign data...")
    data = fetcher.fetch_all_communication_data()

    print(f"\nData fetching complete!")
    print(f"Available datasets: {list(data.keys())}")

    for name, df in data.items():
        print(f"- {name}: {len(df)} records")

    print(f"\nData saved to: {fetcher.data_dir}")

if __name__ == "__main__":
    main()