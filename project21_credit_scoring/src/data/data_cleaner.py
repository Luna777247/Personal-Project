"""
Data cleaning and preprocessing module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess raw credit data"""
    
    def __init__(self, config: Dict):
        """
        Initialize data cleaner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.missing_strategy = config.get('missing_value_strategy', {})
        self.outlier_method = config.get('outlier_method', 'iqr')
        self.outlier_threshold = config.get('outlier_threshold', 1.5)
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete cleaning pipeline
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Starting data cleaning. Shape: {df.shape}")
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Fix data types
        df = self._fix_data_types(df)
        
        # Validate data
        df = self._validate_data(df)
        
        logger.info(f"Cleaning completed. Final shape: {df.shape}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        removed = initial_shape - df.shape[0]
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on strategy"""
        # Check missing values
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            logger.info(f"Found missing values in {len(missing_cols)} columns")
            
            # Numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_strategy = self.missing_strategy.get('numeric', 'median')
            
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    if numeric_strategy == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif numeric_strategy == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif numeric_strategy == 'mode':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    
                    logger.info(f"Filled missing values in {col} using {numeric_strategy}")
            
            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            categorical_strategy = self.missing_strategy.get('categorical', 'mode')
            
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    if categorical_strategy == 'mode':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        df[col].fillna('Unknown', inplace=True)
                    
                    logger.info(f"Filled missing values in {col}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Exclude target and ID columns
        exclude_cols = ['loan_status', 'customer_id', 'id']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols:
            if self.outlier_method == 'iqr':
                df = self._remove_outliers_iqr(df, col)
            elif self.outlier_method == 'zscore':
                df = self._remove_outliers_zscore(df, col)
        
        return df
    
    def _remove_outliers_iqr(
        self,
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        initial_count = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed = initial_count - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} outliers from {column}")
        
        return df
    
    def _remove_outliers_zscore(
        self,
        df: pd.DataFrame,
        column: str,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        
        initial_count = len(df)
        df = df[z_scores < threshold]
        removed = initial_count - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} outliers from {column} using z-score")
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix and optimize data types"""
        # Convert categorical columns to category dtype
        categorical_cols = ['employment_status', 'home_ownership', 'loan_purpose',
                          'education_level', 'marital_status']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Convert boolean columns
        bool_cols = [col for col in df.columns if df[col].nunique() == 2]
        for col in bool_cols:
            if col in ['has_cosigner', 'has_guarantor']:
                df[col] = df[col].astype(bool)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data for business rules"""
        # Age validation
        if 'age' in df.columns:
            df = df[(df['age'] >= 18) & (df['age'] <= 100)]
        
        # Income validation
        if 'income' in df.columns:
            df = df[df['income'] > 0]
        
        # Loan amount validation
        if 'loan_amount' in df.columns:
            df = df[df['loan_amount'] > 0]
        
        # Interest rate validation
        if 'interest_rate' in df.columns:
            df = df[(df['interest_rate'] >= 0) & (df['interest_rate'] <= 100)]
        
        return df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate data quality report
        
        Args:
            df: Dataframe to analyze
            
        Returns:
            Quality metrics dictionary
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        report['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        report['categorical_summary'] = {
            col: df[col].value_counts().to_dict()
            for col in categorical_cols
        }
        
        return report


class DataScaler:
    """Scale numerical features"""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize scaler
        
        Args:
            method: Scaling method (standard, minmax, robust)
        """
        self.method = method
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit(self, X: pd.DataFrame, columns: List[str]) -> 'DataScaler':
        """
        Fit scaler on training data
        
        Args:
            X: Training dataframe
            columns: Columns to scale
            
        Returns:
            Self
        """
        self.columns = columns
        self.scaler.fit(X[columns])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data
        
        Args:
            X: Dataframe to transform
            
        Returns:
            Transformed dataframe
        """
        X_copy = X.copy()
        X_copy[self.columns] = self.scaler.transform(X_copy[self.columns])
        return X_copy
    
    def fit_transform(self, X: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, columns).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data
        
        Args:
            X: Scaled dataframe
            
        Returns:
            Original scale dataframe
        """
        X_copy = X.copy()
        X_copy[self.columns] = self.scaler.inverse_transform(X_copy[self.columns])
        return X_copy


if __name__ == "__main__":
    # Test data cleaning
    from src.data.data_generator import generate_credit_data
    
    # Generate sample data
    df = generate_credit_data(n_samples=1000)
    
    # Clean data
    config = {
        'missing_value_strategy': {'numeric': 'median', 'categorical': 'mode'},
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5
    }
    
    cleaner = DataCleaner(config)
    df_cleaned = cleaner.clean(df)
    
    # Get quality report
    report = cleaner.get_data_quality_report(df_cleaned)
    
    print("\nData Quality Report:")
    print(f"Total Rows: {report['total_rows']}")
    print(f"Total Columns: {report['total_columns']}")
    print(f"Memory Usage: {report['memory_usage_mb']:.2f} MB")
