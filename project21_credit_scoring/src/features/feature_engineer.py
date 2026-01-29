"""
Feature engineering for credit scoring
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for credit scoring model"""
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer
        
        Args:
            config: Feature configuration dictionary
        """
        self.config = config
        self.label_encoders = {}
        self.target_encoder = None
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training data)
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Engineering features. Input shape: {df.shape}")
        
        df = df.copy()
        
        # 1. Financial Ratios
        df = self._create_financial_ratios(df)
        
        # 2. Payment History Features
        df = self._create_payment_features(df)
        
        # 3. Age-based Features
        df = self._create_age_features(df)
        
        # 4. Credit Utilization Features
        df = self._create_credit_features(df)
        
        # 5. Interaction Features
        df = self._create_interaction_features(df)
        
        # 6. Categorical Encoding
        df = self._encode_categorical(df, fit=fit)
        
        logger.info(f"Feature engineering completed. Output shape: {df.shape}")
        
        return df
    
    def _create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial ratio features
        
        Key ratios:
        - Loan to Income Ratio
        - Debt to Income Ratio (DTI)
        - Payment to Income Ratio
        """
        # Loan to Income Ratio
        df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
        
        # Debt to Income Ratio (DTI) - Critical for credit risk
        df['debt_to_income_ratio'] = df['total_debt'] / (df['income'] + 1)
        
        # Payment to Monthly Income Ratio
        monthly_income = df['income'] / 12
        df['payment_to_income_ratio'] = df['monthly_payment'] / (monthly_income + 1)
        
        # Total obligation ratio (debt + new loan payment)
        total_monthly_obligation = (df['total_debt'] / 12) + df['monthly_payment']
        df['total_obligation_ratio'] = total_monthly_obligation / (monthly_income + 1)
        
        # Available income after obligations
        df['available_income'] = monthly_income - total_monthly_obligation
        df['available_income_ratio'] = df['available_income'] / (monthly_income + 1)
        
        # Loan term efficiency (monthly payment relative to loan amount and term)
        df['loan_efficiency'] = df['monthly_payment'] / (df['loan_amount'] / df['loan_term'])
        
        logger.info("Created financial ratio features")
        
        return df
    
    def _create_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create payment history features
        """
        # Delinquency rate
        df['delinquency_rate'] = df['num_delinquencies'] / (df['credit_history_length'] * 12 + 1)
        
        # Late payment rate
        df['late_payment_rate'] = df['num_late_payments'] / (df['credit_history_length'] * 12 + 1)
        
        # Total negative events
        df['total_negative_events'] = df['num_late_payments'] + df['num_delinquencies'] * 2
        
        # Payment consistency score (inverse of negative events)
        df['payment_consistency_score'] = 1 / (df['total_negative_events'] + 1)
        
        # Years since last delinquency (approximation)
        df['delinquency_recency'] = np.where(
            df['num_delinquencies'] > 0,
            df['credit_history_length'] / (df['num_delinquencies'] + 1),
            df['credit_history_length']
        )
        
        logger.info("Created payment history features")
        
        return df
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-related features
        """
        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        
        # Income per age (earning potential)
        df['income_per_age'] = df['income'] / df['age']
        
        # Employment stability (employment length relative to age)
        df['employment_stability'] = df['employment_length'] / (df['age'] - 18 + 1)
        
        # Credit maturity (credit history relative to age)
        df['credit_maturity'] = df['credit_history_length'] / (df['age'] - 18 + 1)
        
        # Years to retirement (assuming 65)
        df['years_to_retirement'] = np.maximum(65 - df['age'], 0)
        
        # Loan term relative to years to retirement
        df['loan_term_retirement_ratio'] = (df['loan_term'] / 12) / (df['years_to_retirement'] + 1)
        
        logger.info("Created age-based features")
        
        return df
    
    def _create_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create credit utilization features
        """
        # Average debt per credit line
        df['avg_debt_per_line'] = df['total_debt'] / (df['num_credit_lines'] + 1)
        
        # Credit line utilization
        # Assuming average credit limit of $10,000 per line
        assumed_credit_limit = df['num_credit_lines'] * 10000
        df['credit_utilization'] = df['total_debt'] / (assumed_credit_limit + 1)
        
        # Active account ratio
        df['active_account_ratio'] = df['num_open_accounts'] / (df['num_credit_lines'] + 1)
        
        # Credit diversity (number of credit lines relative to income)
        df['credit_diversity'] = df['num_credit_lines'] / (df['income'] / 10000 + 1)
        
        # Debt concentration
        df['debt_concentration'] = df['total_debt'] / (df['num_credit_lines'] + 1)
        
        logger.info("Created credit utilization features")
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features
        """
        # Income and employment interaction
        df['income_employment_score'] = df['income'] * df['employment_stability']
        
        # Credit history and payment consistency
        df['credit_payment_score'] = df['credit_history_length'] * df['payment_consistency_score']
        
        # DTI and payment history interaction
        df['dti_payment_risk'] = df['debt_to_income_ratio'] * (df['total_negative_events'] + 1)
        
        # Loan amount and interest rate interaction
        df['loan_cost'] = df['loan_amount'] * df['interest_rate'] / 100
        
        # Total cost of loan
        df['total_loan_cost'] = df['monthly_payment'] * df['loan_term']
        df['loan_markup'] = (df['total_loan_cost'] - df['loan_amount']) / (df['loan_amount'] + 1)
        
        logger.info("Created interaction features")
        
        return df
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders
            
        Returns:
            DataFrame with encoded categories
        """
        categorical_cols = [
            'employment_status',
            'home_ownership',
            'loan_purpose',
            'education_level',
            'marital_status',
            'gender'
        ]
        
        # Target encoding for high-cardinality categories
        target_encode_cols = ['loan_purpose', 'education_level']
        
        # Label encoding for low-cardinality categories
        label_encode_cols = [col for col in categorical_cols if col not in target_encode_cols]
        
        # Label encoding
        for col in label_encode_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Target encoding (requires target variable)
        if 'loan_status' in df.columns and fit:
            self.target_encoder = TargetEncoder(cols=target_encode_cols)
            encoded_df = self.target_encoder.fit_transform(df[target_encode_cols], df['loan_status'])
            
            for col in target_encode_cols:
                if col in df.columns:
                    df[f'{col}_target_encoded'] = encoded_df[col]
        
        elif self.target_encoder is not None:
            encoded_df = self.target_encoder.transform(df[target_encode_cols])
            
            for col in target_encode_cols:
                if col in df.columns:
                    df[f'{col}_target_encoded'] = encoded_df[col]
        
        # Handle age_group separately
        if 'age_group' in df.columns:
            if fit:
                self.label_encoders['age_group'] = LabelEncoder()
                df['age_group_encoded'] = self.label_encoders['age_group'].fit_transform(df['age_group'].astype(str))
            else:
                df['age_group_encoded'] = self.label_encoders['age_group'].transform(df['age_group'].astype(str))
        
        # Boolean encoding
        bool_cols = ['has_cosigner', 'has_guarantor']
        for col in bool_cols:
            if col in df.columns:
                df[f'{col}_int'] = df[col].astype(int)
        
        logger.info("Encoded categorical features")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all engineered feature names
        
        Returns:
            List of feature names
        """
        features = [
            # Original numeric features
            'age', 'employment_length', 'income', 'credit_history_length',
            'num_credit_lines', 'num_open_accounts', 'total_debt',
            'loan_amount', 'loan_term', 'interest_rate', 'monthly_payment',
            'num_late_payments', 'num_delinquencies',
            
            # Financial ratios
            'loan_to_income_ratio', 'debt_to_income_ratio',
            'payment_to_income_ratio', 'total_obligation_ratio',
            'available_income', 'available_income_ratio', 'loan_efficiency',
            
            # Payment features
            'delinquency_rate', 'late_payment_rate', 'total_negative_events',
            'payment_consistency_score', 'delinquency_recency',
            
            # Age features
            'income_per_age', 'employment_stability', 'credit_maturity',
            'years_to_retirement', 'loan_term_retirement_ratio',
            
            # Credit features
            'avg_debt_per_line', 'credit_utilization', 'active_account_ratio',
            'credit_diversity', 'debt_concentration',
            
            # Interaction features
            'income_employment_score', 'credit_payment_score',
            'dti_payment_risk', 'loan_cost', 'total_loan_cost', 'loan_markup',
            
            # Encoded categorical
            'employment_status_encoded', 'home_ownership_encoded',
            'marital_status_encoded', 'gender_encoded',
            'loan_purpose_target_encoded', 'education_level_target_encoded',
            'age_group_encoded',
            
            # Boolean
            'has_cosigner_int', 'has_guarantor_int'
        ]
        
        return features


if __name__ == "__main__":
    from src.data import generate_credit_data
    
    # Generate sample data
    df = generate_credit_data(n_samples=1000)
    
    # Engineer features
    config = {}
    engineer = FeatureEngineer(config)
    df_engineered = engineer.engineer_features(df, fit=True)
    
    print("\nOriginal shape:", df.shape)
    print("Engineered shape:", df_engineered.shape)
    print("\nNew features:")
    new_cols = set(df_engineered.columns) - set(df.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
    
    print("\nSample features:")
    print(df_engineered[['debt_to_income_ratio', 'payment_to_income_ratio', 
                         'payment_consistency_score']].describe())
