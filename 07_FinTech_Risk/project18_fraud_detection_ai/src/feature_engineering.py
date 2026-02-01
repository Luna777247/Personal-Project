"""
Feature Engineering Module for Fraud Detection
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionFeatureEngineer:
    """Feature engineering for transaction data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_time_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Create time-based features from timestamp
        
        Args:
            df: Input dataframe
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with added time features
        """
        logger.info("Creating time-based features...")
        
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                    bins=[0, 6, 12, 18, 24],
                                    labels=['night', 'morning', 'afternoon', 'evening'],
                                    include_lowest=True)
        
        # Business hours flag
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                   (df['day_of_week'] < 5)).astype(int)
        
        return df
    
    def create_amount_features(self, df: pd.DataFrame, amount_col: str = 'amount') -> pd.DataFrame:
        """
        Create amount-based features
        
        Args:
            df: Input dataframe
            amount_col: Name of amount column
            
        Returns:
            DataFrame with added amount features
        """
        logger.info("Creating amount-based features...")
        
        df = df.copy()
        
        # Log transformation
        df['amount_log'] = np.log1p(df[amount_col])
        
        # Z-score
        df['amount_zscore'] = (df[amount_col] - df[amount_col].mean()) / df[amount_col].std()
        
        # Amount categories
        df['amount_category'] = pd.cut(df[amount_col], 
                                       bins=[0, 50, 200, 1000, float('inf')],
                                       labels=['small', 'medium', 'large', 'very_large'])
        
        return df
    
    def create_velocity_features(self, df: pd.DataFrame, 
                                 customer_id_col: str = 'customer_id',
                                 timestamp_col: str = 'timestamp',
                                 amount_col: str = 'amount',
                                 time_windows: List[int] = [3600, 86400, 604800]) -> pd.DataFrame:
        """
        Create transaction velocity features
        
        Args:
            df: Input dataframe
            customer_id_col: Customer identifier column
            timestamp_col: Timestamp column
            amount_col: Amount column
            time_windows: Time windows in seconds (e.g., 3600 for 1 hour)
            
        Returns:
            DataFrame with added velocity features
        """
        logger.info("Creating velocity features...")
        
        df = df.copy()
        df = df.sort_values([customer_id_col, timestamp_col])
        
        for window in time_windows:
            window_label = self._get_window_label(window)
            
            # Count of transactions in time window
            df[f'txn_count_{window_label}'] = df.groupby(customer_id_col)[timestamp_col].transform(
                lambda x: self._count_in_window(x, window)
            )
            
            # Sum of amounts in time window
            df[f'amount_sum_{window_label}'] = df.groupby(customer_id_col).rolling(
                window=f'{window}s', on=timestamp_col
            )[amount_col].sum().reset_index(0, drop=True)
            
            # Average amount in time window
            df[f'amount_avg_{window_label}'] = df.groupby(customer_id_col).rolling(
                window=f'{window}s', on=timestamp_col
            )[amount_col].mean().reset_index(0, drop=True)
            
            # Std of amounts in time window
            df[f'amount_std_{window_label}'] = df.groupby(customer_id_col).rolling(
                window=f'{window}s', on=timestamp_col
            )[amount_col].std().reset_index(0, drop=True)
        
        # Fill NaN values
        velocity_cols = [col for col in df.columns if any(x in col for x in ['txn_count', 'amount_sum', 'amount_avg', 'amount_std'])]
        df[velocity_cols] = df[velocity_cols].fillna(0)
        
        return df
    
    def create_merchant_features(self, df: pd.DataFrame,
                                 merchant_col: str = 'merchant_id',
                                 amount_col: str = 'amount') -> pd.DataFrame:
        """
        Create merchant-based features
        
        Args:
            df: Input dataframe
            merchant_col: Merchant identifier column
            amount_col: Amount column
            
        Returns:
            DataFrame with added merchant features
        """
        logger.info("Creating merchant features...")
        
        df = df.copy()
        
        # Merchant transaction frequency
        merchant_freq = df[merchant_col].value_counts().to_dict()
        df['merchant_frequency'] = df[merchant_col].map(merchant_freq)
        
        # Merchant average amount
        merchant_avg = df.groupby(merchant_col)[amount_col].mean().to_dict()
        df['merchant_avg_amount'] = df[merchant_col].map(merchant_avg)
        
        # Deviation from merchant average
        df['amount_vs_merchant_avg'] = df[amount_col] / (df['merchant_avg_amount'] + 1e-6)
        
        return df
    
    def create_customer_features(self, df: pd.DataFrame,
                                 customer_id_col: str = 'customer_id',
                                 amount_col: str = 'amount') -> pd.DataFrame:
        """
        Create customer-based features
        
        Args:
            df: Input dataframe
            customer_id_col: Customer identifier column
            amount_col: Amount column
            
        Returns:
            DataFrame with added customer features
        """
        logger.info("Creating customer features...")
        
        df = df.copy()
        
        # Customer transaction frequency
        customer_freq = df[customer_id_col].value_counts().to_dict()
        df['customer_frequency'] = df[customer_id_col].map(customer_freq)
        
        # Customer average amount
        customer_avg = df.groupby(customer_id_col)[amount_col].mean().to_dict()
        df['customer_avg_amount'] = df[customer_id_col].map(customer_avg)
        
        # Deviation from customer average
        df['amount_vs_customer_avg'] = df[amount_col] / (df['customer_avg_amount'] + 1e-6)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                    categorical_cols: List[str],
                                    fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            categorical_cols: List of categorical column names
            fit: Whether to fit encoders or use existing ones
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ else -1
                    )
                    
        return df
    
    def scale_features(self, df: pd.DataFrame, 
                      numeric_cols: List[str],
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            df: Input dataframe
            numeric_cols: List of numeric column names
            fit: Whether to fit scaler or use existing one
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Scaling numeric features...")
        
        df = df.copy()
        
        # Filter existing columns
        cols_to_scale = [col for col in numeric_cols if col in df.columns]
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            
        return df
    
    def create_all_features(self, df: pd.DataFrame, 
                           fit: bool = True,
                           config: Dict = None) -> pd.DataFrame:
        """
        Create all features in pipeline
        
        Args:
            df: Input dataframe
            fit: Whether to fit transformers
            config: Configuration dictionary
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features...")
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create amount features
        df = self.create_amount_features(df)
        
        # Create velocity features
        if config and 'time_windows' in config:
            df = self.create_velocity_features(df, time_windows=config['time_windows'])
        else:
            df = self.create_velocity_features(df)
        
        # Create merchant features
        if 'merchant_id' in df.columns:
            df = self.create_merchant_features(df)
        
        # Create customer features
        if 'customer_id' in df.columns:
            df = self.create_customer_features(df)
        
        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            df = self.encode_categorical_features(df, categorical_cols, fit=fit)
        
        return df
    
    @staticmethod
    def _get_window_label(seconds: int) -> str:
        """Convert seconds to readable label"""
        if seconds < 3600:
            return f"{seconds // 60}min"
        elif seconds < 86400:
            return f"{seconds // 3600}hour"
        else:
            return f"{seconds // 86400}day"
    
    @staticmethod
    def _count_in_window(timestamps, window_seconds):
        """Count transactions in rolling time window"""
        result = []
        for i, ts in enumerate(timestamps):
            count = sum((timestamps[:i+1] >= ts - pd.Timedelta(seconds=window_seconds)))
            result.append(count)
        return pd.Series(result, index=timestamps.index)


def generate_sample_data(n_samples: int = 10000, fraud_ratio: float = 0.01) -> pd.DataFrame:
    """
    Generate sample transaction data for testing
    
    Args:
        n_samples: Number of samples to generate
        fraud_ratio: Ratio of fraudulent transactions
        
    Returns:
        DataFrame with sample transaction data
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Normal transactions
    normal_data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_normal)],
        'customer_id': np.random.randint(1000, 5000, n_normal),
        'merchant_id': np.random.randint(100, 500, n_normal),
        'amount': np.random.lognormal(mean=3, sigma=1, size=n_normal),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_normal, freq='5min'),
        'merchant_category': np.random.choice(['retail', 'food', 'entertainment', 'gas', 'grocery'], n_normal),
        'card_type': np.random.choice(['credit', 'debit'], n_normal),
        'transaction_type': np.random.choice(['online', 'in-store', 'atm'], n_normal),
        'country_code': np.random.choice(['US', 'GB', 'DE', 'FR'], n_normal),
        'is_fraud': 0
    }
    
    # Fraudulent transactions (different patterns)
    fraud_data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_normal, n_samples)],
        'customer_id': np.random.randint(1000, 5000, n_fraud),
        'merchant_id': np.random.randint(100, 500, n_fraud),
        'amount': np.random.lognormal(mean=5, sigma=1.5, size=n_fraud),  # Higher amounts
        'timestamp': pd.date_range(start='2024-01-01', periods=n_fraud, freq='5min'),
        'merchant_category': np.random.choice(['retail', 'online_casino', 'foreign_transfer'], n_fraud),
        'card_type': np.random.choice(['credit', 'debit'], n_fraud),
        'transaction_type': np.random.choice(['online', 'international'], n_fraud),
        'country_code': np.random.choice(['CN', 'RU', 'NG', 'BR'], n_fraud),  # Different countries
        'is_fraud': 1
    }
    
    # Combine and shuffle
    df_normal = pd.DataFrame(normal_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    # Test feature engineering
    df = generate_sample_data(n_samples=1000)
    
    print("Sample data generated:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
    
    # Create features
    feature_engineer = TransactionFeatureEngineer()
    df_features = feature_engineer.create_all_features(df)
    
    print(f"\nAfter feature engineering:")
    print(f"Shape: {df_features.shape}")
    print(f"Columns: {df_features.columns.tolist()}")
