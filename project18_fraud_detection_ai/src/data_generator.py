"""
Data generation module for fraud detection
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataGenerator:
    """Generate synthetic fraud transaction data"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_transactions(self, 
                            n_samples: int = 10000,
                            fraud_ratio: float = 0.01,
                            start_date: str = '2024-01-01') -> pd.DataFrame:
        """
        Generate synthetic transaction data
        
        Args:
            n_samples: Total number of transactions
            fraud_ratio: Proportion of fraudulent transactions
            start_date: Start date for transactions
            
        Returns:
            DataFrame with transaction data
        """
        logger.info(f"Generating {n_samples} transactions with {fraud_ratio:.2%} fraud ratio...")
        
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        # Generate normal transactions
        df_normal = self._generate_normal_transactions(n_normal, start_date)
        
        # Generate fraudulent transactions
        df_fraud = self._generate_fraud_transactions(n_fraud, start_date, n_normal)
        
        # Combine and shuffle
        df = pd.concat([df_normal, df_fraud], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} transactions")
        logger.info(f"Fraud transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
        
        return df
    
    def _generate_normal_transactions(self, n: int, start_date: str) -> pd.DataFrame:
        """Generate normal transaction patterns"""
        
        # Normal business hours transactions
        hours = np.random.choice(range(8, 20), size=n, p=self._business_hours_probs())
        timestamps = pd.to_datetime(start_date) + pd.to_timedelta(
            np.random.randint(0, 90*24*60, n), unit='min'
        )
        timestamps = timestamps + pd.to_timedelta(hours - timestamps.hour, unit='h')
        
        data = {
            'transaction_id': [f'TXN_{i:08d}' for i in range(n)],
            'customer_id': np.random.randint(1000, 10000, n),
            'merchant_id': np.random.randint(100, 1000, n),
            'amount': np.abs(np.random.lognormal(mean=3.5, sigma=1.0, size=n)),
            'timestamp': timestamps,
            'merchant_category': np.random.choice(
                ['grocery', 'gas', 'restaurant', 'retail', 'pharmacy', 'entertainment'],
                size=n,
                p=[0.25, 0.15, 0.20, 0.20, 0.10, 0.10]
            ),
            'card_type': np.random.choice(['credit', 'debit'], size=n, p=[0.6, 0.4]),
            'transaction_type': np.random.choice(
                ['in-store', 'online', 'atm'],
                size=n,
                p=[0.6, 0.35, 0.05]
            ),
            'country_code': np.random.choice(
                ['US', 'GB', 'DE', 'FR', 'CA'],
                size=n,
                p=[0.7, 0.1, 0.1, 0.05, 0.05]
            ),
            'device_id': [f'DEV_{np.random.randint(1000, 9999):04d}' for _ in range(n)],
            'ip_address': [self._generate_ip() for _ in range(n)],
            'is_fraud': 0
        }
        
        return pd.DataFrame(data)
    
    def _generate_fraud_transactions(self, n: int, start_date: str, offset: int) -> pd.DataFrame:
        """Generate fraudulent transaction patterns"""
        
        # Fraud patterns: unusual hours, locations, amounts
        fraud_types = np.random.choice(
            ['high_amount', 'unusual_time', 'foreign_country', 'multiple_rapid', 'unusual_merchant'],
            size=n
        )
        
        timestamps = pd.to_datetime(start_date) + pd.to_timedelta(
            np.random.randint(0, 90*24*60, n), unit='min'
        )
        
        data = {
            'transaction_id': [f'TXN_{i:08d}' for i in range(offset, offset + n)],
            'customer_id': np.random.randint(1000, 10000, n),
            'merchant_id': np.random.randint(100, 1000, n),
            'amount': [],
            'timestamp': [],
            'merchant_category': [],
            'card_type': np.random.choice(['credit', 'debit'], size=n, p=[0.8, 0.2]),
            'transaction_type': [],
            'country_code': [],
            'device_id': [f'DEV_{np.random.randint(1000, 9999):04d}' for _ in range(n)],
            'ip_address': [self._generate_ip() for _ in range(n)],
            'is_fraud': 1
        }
        
        # Generate fraud-specific patterns
        for i, fraud_type in enumerate(fraud_types):
            if fraud_type == 'high_amount':
                # Unusually high amounts
                data['amount'].append(np.random.lognormal(mean=6.0, sigma=1.5))
                data['timestamp'].append(timestamps[i])
                data['merchant_category'].append(np.random.choice(['electronics', 'jewelry', 'luxury']))
                data['transaction_type'].append('online')
                data['country_code'].append(np.random.choice(['US', 'GB']))
                
            elif fraud_type == 'unusual_time':
                # Late night transactions
                hour = np.random.choice(range(0, 6))
                ts = timestamps[i].replace(hour=hour)
                data['amount'].append(np.random.lognormal(mean=4.5, sigma=1.2))
                data['timestamp'].append(ts)
                data['merchant_category'].append(np.random.choice(['online_gaming', 'casino', 'retail']))
                data['transaction_type'].append('online')
                data['country_code'].append(np.random.choice(['US', 'GB']))
                
            elif fraud_type == 'foreign_country':
                # Foreign country transactions
                data['amount'].append(np.random.lognormal(mean=5.0, sigma=1.3))
                data['timestamp'].append(timestamps[i])
                data['merchant_category'].append(np.random.choice(['retail', 'entertainment', 'travel']))
                data['transaction_type'].append('online')
                data['country_code'].append(np.random.choice(['CN', 'RU', 'NG', 'BR', 'IN']))
                
            elif fraud_type == 'multiple_rapid':
                # Multiple transactions in short time
                data['amount'].append(np.random.lognormal(mean=4.0, sigma=1.0))
                data['timestamp'].append(timestamps[i])
                data['merchant_category'].append(np.random.choice(['retail', 'online_services']))
                data['transaction_type'].append('online')
                data['country_code'].append(np.random.choice(['US', 'GB', 'CA']))
                
            else:  # unusual_merchant
                # Unusual merchant categories
                data['amount'].append(np.random.lognormal(mean=4.5, sigma=1.2))
                data['timestamp'].append(timestamps[i])
                data['merchant_category'].append(np.random.choice(['cryptocurrency', 'wire_transfer', 'gift_cards']))
                data['transaction_type'].append('online')
                data['country_code'].append(np.random.choice(['US', 'GB']))
        
        return pd.DataFrame(data)
    
    @staticmethod
    def _business_hours_probs():
        """Probability distribution for business hours"""
        probs = np.array([0.05, 0.08, 0.12, 0.15, 0.15, 0.15, 0.12, 0.10, 0.05, 0.02, 0.01, 0.00])
        return probs / probs.sum()
    
    @staticmethod
    def _generate_ip():
        """Generate random IP address"""
        return f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}." \
               f"{np.random.randint(0, 255)}.{np.random.randint(0, 255)}"


def save_data(df: pd.DataFrame, output_path: str):
    """Save data to CSV"""
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to {output_path}")


if __name__ == "__main__":
    # Generate sample data
    generator = FraudDataGenerator()
    df = generator.generate_transactions(n_samples=50000, fraud_ratio=0.02)
    
    print("\nData Summary:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print(f"\nFraud distribution:")
    print(df['is_fraud'].value_counts())
    
    # Save data
    save_data(df, "data/raw/transactions.csv")
