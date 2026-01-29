#!/usr/bin/env python3
"""
Real Social Media Data Fetcher
==============================

Fetches real social media data from various platforms for trend analysis.
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import praw
import tweepy
import os
from textblob import TextBlob
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialMediaDataFetcher:
    """Fetch real social media data from multiple platforms"""

    def __init__(self):
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # API Keys from environment variables
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'SocialMediaAnalyzer/1.0')

    def fetch_twitter_data(self, query='AI OR machine learning OR data science', max_results=100):
        """Fetch real Twitter data using Twitter API v2"""
        logger.info(f"Fetching Twitter data for query: {query}")

        if not self.twitter_bearer_token:
            logger.warning("Twitter API key not found, using mock data")
            return self._create_mock_twitter_data()

        try:
            headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}

            # Twitter API v2 endpoint
            url = "https://api.twitter.com/2/tweets/search/recent"
            params = {
                'query': query,
                'max_results': min(max_results, 100),
                'tweet.fields': 'created_at,public_metrics,text,author_id,lang',
                'user.fields': 'username,name,verified'
            }

            response = requests.get(url, headers=headers, params=params)
            data = response.json()

            if response.status_code == 200 and 'data' in data:
                tweets = []
                for tweet in data['data']:
                    tweets.append({
                        'id': tweet['id'],
                        'text': tweet['text'],
                        'created_at': tweet['created_at'],
                        'author_id': tweet['author_id'],
                        'retweet_count': tweet['public_metrics']['retweet_count'],
                        'like_count': tweet['public_metrics']['like_count'],
                        'reply_count': tweet['public_metrics']['reply_count'],
                        'platform': 'twitter',
                        'query': query
                    })

                tweets_df = pd.DataFrame(tweets)
                tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'])

                # Add sentiment analysis
                tweets_df['sentiment'] = tweets_df['text'].apply(self._analyze_sentiment)

                tweets_df.to_csv(self.data_dir / 'twitter_data.csv', index=False)
                logger.info(f"✅ Fetched {len(tweets)} tweets")
                return tweets_df
            else:
                logger.error(f"Twitter API error: {data}")

        except Exception as e:
            logger.error(f"❌ Failed to fetch Twitter data: {e}")

        return self._create_mock_twitter_data()

    def fetch_reddit_data(self, subreddits=['technology', 'programming', 'datascience'], limit=100):
        """Fetch real Reddit data using PRAW"""
        logger.info(f"Fetching Reddit data from subreddits: {subreddits}")

        if not all([self.reddit_client_id, self.reddit_client_secret]):
            logger.warning("Reddit API credentials not found, using mock data")
            return self._create_mock_reddit_data()

        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )

            posts = []
            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)

                for post in subreddit.hot(limit=limit//len(subreddits)):
                    posts.append({
                        'id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'subreddit': subreddit_name,
                        'url': post.url,
                        'platform': 'reddit'
                    })

            posts_df = pd.DataFrame(posts)

            # Combine title and text for analysis
            posts_df['full_text'] = posts_df['title'] + ' ' + posts_df['text'].fillna('')

            # Add sentiment analysis
            posts_df['sentiment'] = posts_df['full_text'].apply(self._analyze_sentiment)

            posts_df.to_csv(self.data_dir / 'reddit_data.csv', index=False)
            logger.info(f"✅ Fetched {len(posts)} Reddit posts")
            return posts_df

        except Exception as e:
            logger.error(f"❌ Failed to fetch Reddit data: {e}")

        return self._create_mock_reddit_data()

    def fetch_github_trends(self, topics=['machine-learning', 'data-science', 'ai'], days=7):
        """Fetch GitHub trending repositories and activity"""
        logger.info(f"Fetching GitHub trends for topics: {topics}")

        try:
            all_repos = []

            for topic in topics:
                url = f"https://api.github.com/search/repositories"
                params = {
                    'q': f'topic:{topic}',
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': 50
                }

                response = requests.get(url, params=params)
                data = response.json()

                if response.status_code == 200 and 'items' in data:
                    for repo in data['items']:
                        # Check if repo was created/updated recently
                        created_date = pd.to_datetime(repo['created_at'])
                        updated_date = pd.to_datetime(repo['updated_at'])

                        if (datetime.now() - created_date).days <= days or \
                           (datetime.now() - updated_date).days <= days:

                            all_repos.append({
                                'id': repo['id'],
                                'name': repo['name'],
                                'full_name': repo['full_name'],
                                'description': repo['description'] or '',
                                'stars': repo['stargazers_count'],
                                'forks': repo['forks_count'],
                                'language': repo['language'],
                                'created_at': created_date,
                                'updated_at': updated_date,
                                'topic': topic,
                                'platform': 'github',
                                'url': repo['html_url']
                            })

                time.sleep(1)  # Rate limiting

            repos_df = pd.DataFrame(all_repos)

            # Add engagement score
            repos_df['engagement_score'] = repos_df['stars'] * 0.7 + repos_df['forks'] * 0.3

            repos_df.to_csv(self.data_dir / 'github_trends.csv', index=False)
            logger.info(f"✅ Fetched {len(all_repos)} trending GitHub repositories")
            return repos_df

        except Exception as e:
            logger.error(f"❌ Failed to fetch GitHub data: {e}")

        return self._create_mock_github_data()

    def fetch_youtube_trends(self, api_key=None, region='US'):
        """Fetch YouTube trending videos"""
        logger.info(f"Fetching YouTube trends for region: {region}")

        if not api_key:
            api_key = os.getenv('YOUTUBE_API_KEY')

        if not api_key:
            logger.warning("YouTube API key not found, using mock data")
            return self._create_mock_youtube_data()

        try:
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                'part': 'snippet,statistics',
                'chart': 'mostPopular',
                'regionCode': region,
                'maxResults': 50,
                'key': api_key
            }

            response = requests.get(url, params=params)
            data = response.json()

            if response.status_code == 200 and 'items' in data:
                videos = []
                for video in data['items']:
                    stats = video.get('statistics', {})
                    videos.append({
                        'id': video['id'],
                        'title': video['snippet']['title'],
                        'description': video['snippet']['description'][:500],  # Truncate
                        'channel_title': video['snippet']['channelTitle'],
                        'published_at': video['snippet']['publishedAt'],
                        'view_count': int(stats.get('viewCount', 0)),
                        'like_count': int(stats.get('likeCount', 0)),
                        'comment_count': int(stats.get('commentCount', 0)),
                        'platform': 'youtube',
                        'region': region
                    })

                videos_df = pd.DataFrame(videos)
                videos_df['published_at'] = pd.to_datetime(videos_df['published_at'])

                # Add engagement rate
                videos_df['engagement_rate'] = (
                    videos_df['like_count'] + videos_df['comment_count']
                ) / videos_df['view_count'].replace(0, 1)

                videos_df.to_csv(self.data_dir / 'youtube_trends.csv', index=False)
                logger.info(f"✅ Fetched {len(videos)} trending YouTube videos")
                return videos_df

        except Exception as e:
            logger.error(f"❌ Failed to fetch YouTube data: {e}")

        return self._create_mock_youtube_data()

    def fetch_instagram_hashtags(self, hashtags=['ai', 'machinelearning', 'datascience'], limit=50):
        """Fetch Instagram hashtag data (using web scraping as alternative)"""
        logger.info(f"Fetching Instagram hashtag data for: {hashtags}")

        # Note: Instagram API requires business account, so we'll use mock data
        # In production, you'd use Instagram Graph API
        logger.info("Instagram API requires business account, using enhanced mock data")
        return self._create_mock_instagram_data(hashtags, limit)

    def _analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity

            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'

    def _create_mock_twitter_data(self):
        """Create realistic mock Twitter data"""
        logger.info("Creating mock Twitter data")

        tweets = []
        for i in range(200):
            tweets.append({
                'id': f'tweet_{i}',
                'text': f'This is tweet number {i} about AI and machine learning #AI #ML',
                'created_at': pd.Timestamp.now() - pd.Timedelta(minutes=i*5),
                'author_id': f'user_{i%50}',
                'retweet_count': np.random.poisson(10),
                'like_count': np.random.poisson(25),
                'reply_count': np.random.poisson(5),
                'platform': 'twitter',
                'query': 'AI OR machine learning'
            })

        tweets_df = pd.DataFrame(tweets)
        tweets_df['sentiment'] = tweets_df['text'].apply(self._analyze_sentiment)
        tweets_df.to_csv(self.data_dir / 'twitter_data.csv', index=False)
        return tweets_df

    def _create_mock_reddit_data(self):
        """Create realistic mock Reddit data"""
        logger.info("Creating mock Reddit data")

        posts = []
        subreddits = ['technology', 'programming', 'datascience']

        for i in range(150):
            subreddit = subreddits[i % len(subreddits)]
            posts.append({
                'id': f'post_{i}',
                'title': f'Interesting discussion about {subreddit} topic {i}',
                'text': f'This is a detailed post about {subreddit} with some insights and questions.',
                'score': np.random.poisson(50),
                'num_comments': np.random.poisson(20),
                'created_utc': pd.Timestamp.now() - pd.Timedelta(hours=i),
                'subreddit': subreddit,
                'url': f'https://reddit.com/r/{subreddit}/post_{i}',
                'platform': 'reddit'
            })

        posts_df = pd.DataFrame(posts)
        posts_df['full_text'] = posts_df['title'] + ' ' + posts_df['text']
        posts_df['sentiment'] = posts_df['full_text'].apply(self._analyze_sentiment)
        posts_df.to_csv(self.data_dir / 'reddit_data.csv', index=False)
        return posts_df

    def _create_mock_github_data(self):
        """Create realistic mock GitHub data"""
        logger.info("Creating mock GitHub data")

        repos = []
        topics = ['machine-learning', 'data-science', 'ai']

        for i in range(100):
            topic = topics[i % len(topics)]
            repos.append({
                'id': i,
                'name': f'awesome-{topic}-project-{i}',
                'full_name': f'user{ i%20 }/awesome-{topic}-project-{i}',
                'description': f'An awesome {topic} project with great features',
                'stars': np.random.poisson(500),
                'forks': np.random.poisson(100),
                'language': ['Python', 'JavaScript', 'Java', 'Go'][i%4],
                'created_at': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365)),
                'updated_at': pd.Timestamp.now() - pd.Timedelta(hours=np.random.randint(1, 168)),
                'topic': topic,
                'platform': 'github',
                'url': f'https://github.com/user{i%20}/awesome-{topic}-project-{i}'
            })

        repos_df = pd.DataFrame(repos)
        repos_df['engagement_score'] = repos_df['stars'] * 0.7 + repos_df['forks'] * 0.3
        repos_df.to_csv(self.data_dir / 'github_trends.csv', index=False)
        return repos_df

    def _create_mock_youtube_data(self):
        """Create realistic mock YouTube data"""
        logger.info("Creating mock YouTube data")

        videos = []
        for i in range(50):
            videos.append({
                'id': f'video_{i}',
                'title': f'Amazing Tech Tutorial {i} - Learn Something New!',
                'description': f'This is a comprehensive tutorial about technology topic {i}',
                'channel_title': f'TechChannel_{i%10}',
                'published_at': pd.Timestamp.now() - pd.Timedelta(hours=i*2),
                'view_count': np.random.poisson(50000),
                'like_count': np.random.poisson(1000),
                'comment_count': np.random.poisson(100),
                'platform': 'youtube',
                'region': 'US'
            })

        videos_df = pd.DataFrame(videos)
        videos_df['engagement_rate'] = (
            videos_df['like_count'] + videos_df['comment_count']
        ) / videos_df['view_count'].replace(0, 1)
        videos_df.to_csv(self.data_dir / 'youtube_trends.csv', index=False)
        return videos_df

    def _create_mock_instagram_data(self, hashtags, limit):
        """Create realistic mock Instagram data"""
        logger.info("Creating mock Instagram data")

        posts = []
        for i in range(limit):
            hashtag = hashtags[i % len(hashtags)]
            posts.append({
                'id': f'insta_{i}',
                'caption': f'Beautiful photo about #{hashtag} #instagood #photooftheday',
                'hashtags': [hashtag, 'instagood', 'photooftheday'],
                'likes': np.random.poisson(500),
                'comments': np.random.poisson(50),
                'created_at': pd.Timestamp.now() - pd.Timedelta(minutes=i*30),
                'platform': 'instagram',
                'engagement_score': np.random.poisson(550)
            })

        posts_df = pd.DataFrame(posts)
        posts_df.to_csv(self.data_dir / 'instagram_data.csv', index=False)
        return posts_df

    def fetch_all_social_data(self):
        """Fetch data from all social media platforms"""
        logger.info("🚀 Starting comprehensive social media data fetching...")

        twitter_data = self.fetch_twitter_data()
        reddit_data = self.fetch_reddit_data()
        github_data = self.fetch_github_trends()
        youtube_data = self.fetch_youtube_trends()
        instagram_data = self.fetch_instagram_hashtags()

        # Combine all data
        all_data = pd.concat([
            twitter_data[['platform', 'text', 'created_at', 'sentiment']].rename(columns={'text': 'content'}),
            reddit_data[['platform', 'full_text', 'created_utc', 'sentiment']].rename(columns={'full_text': 'content', 'created_utc': 'created_at'}),
            github_data[['platform', 'description', 'created_at']].assign(sentiment='neutral').rename(columns={'description': 'content'}),
            youtube_data[['platform', 'title', 'published_at']].assign(sentiment='neutral').rename(columns={'title': 'content', 'published_at': 'created_at'}),
            instagram_data[['platform', 'caption', 'created_at']].assign(sentiment='neutral').rename(columns={'caption': 'content'})
        ], ignore_index=True)

        all_data.to_csv(self.data_dir / 'combined_social_data.csv', index=False)
        logger.info(f"✅ Combined {len(all_data)} social media posts from all platforms")

        logger.info("✅ Social media data fetching completed!")
        logger.info(f"📁 Check {self.data_dir} for downloaded data files")

        return all_data

if __name__ == "__main__":
    fetcher = SocialMediaDataFetcher()
    fetcher.fetch_all_social_data()