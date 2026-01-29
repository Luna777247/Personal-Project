"""
Credit scoring predictor
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditScorePredictor:
    """Handle credit score predictions"""
    
    def __init__(self, model_path: str, feature_engineer=None, scaler=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            feature_engineer: Feature engineering instance
            scaler: Feature scaler instance
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_engineer = feature_engineer
        self.scaler = scaler
        self.model_type = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        try:
            import xgboost as xgb
            import lightgbm as lgb
            
            # Determine model type from file extension
            if self.model_path.suffix == '.json' or 'xgboost' in str(self.model_path):
                self.model = xgb.Booster()
                self.model.load_model(str(self.model_path))
                self.model_type = 'xgboost'
                logger.info(f"Loaded XGBoost model from {self.model_path}")
            
            elif self.model_path.suffix == '.txt' or 'lightgbm' in str(self.model_path):
                self.model = lgb.Booster(model_file=str(self.model_path))
                self.model_type = 'lightgbm'
                logger.info(f"Loaded LightGBM model from {self.model_path}")
            
            else:
                # Try loading as pickle (sklearn)
                self.model = joblib.load(self.model_path)
                self.model_type = 'sklearn'
                logger.info(f"Loaded sklearn model from {self.model_path}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_input(self, data: Dict) -> pd.DataFrame:
        """
        Preprocess input data
        
        Args:
            data: Input feature dictionary
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Apply feature engineering if available
        if self.feature_engineer is not None:
            df = self.feature_engineer.engineer_features(df, fit=False)
        
        # Apply scaling if available
        if self.scaler is not None:
            feature_cols = df.select_dtypes(include=[np.number]).columns
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
    
    def predict(self, data: Dict) -> Tuple[float, float]:
        """
        Make prediction
        
        Args:
            data: Input features
            
        Returns:
            Tuple of (score, probability)
        """
        # Preprocess
        df = self.preprocess_input(data)
        
        # Predict
        if self.model_type == 'xgboost':
            import xgboost as xgb
            dmatrix = xgb.DMatrix(df)
            score = self.model.predict(dmatrix)[0]
        
        elif self.model_type == 'lightgbm':
            score = self.model.predict(df)[0]
        
        else:  # sklearn
            score = self.model.predict_proba(df)[0, 1]
        
        # Convert to probability if needed
        probability = float(score)
        
        return score, probability
    
    def determine_risk_level(self, probability: float) -> str:
        """
        Determine risk level from probability
        
        Args:
            probability: Default probability
            
        Returns:
            Risk level string
        """
        if probability < 0.3:
            return "low"
        elif probability < 0.6:
            return "medium"
        elif probability < 0.8:
            return "high"
        else:
            return "very_high"
    
    def determine_approval(
        self,
        probability: float,
        threshold: float = 0.6
    ) -> str:
        """
        Determine approval decision
        
        Args:
            probability: Default probability
            threshold: Approval threshold
            
        Returns:
            Approval decision
        """
        if probability < threshold * 0.7:
            return "Approved"
        elif probability < threshold:
            return "Review"
        else:
            return "Denied"
    
    def get_key_factors(self, data: Dict, top_n: int = 5) -> list:
        """
        Get key decision factors
        
        Args:
            data: Input features
            top_n: Number of factors to return
            
        Returns:
            List of key factors
        """
        factors = []
        
        # Debt to income ratio
        dti = data['total_debt'] / (data['income'] + 1)
        if dti > 0.5:
            factors.append(f"High debt-to-income ratio ({dti:.2f})")
        
        # Late payments
        if data['num_late_payments'] > 0:
            factors.append(f"{data['num_late_payments']} late payment(s)")
        
        # Delinquencies
        if data['num_delinquencies'] > 0:
            factors.append(f"{data['num_delinquencies']} delinquency(ies)")
        
        # Credit history
        if data['credit_history_length'] < 2:
            factors.append("Limited credit history")
        
        # Loan to income
        lti = data['loan_amount'] / (data['income'] + 1)
        if lti > 0.5:
            factors.append(f"High loan-to-income ratio ({lti:.2f})")
        
        # Employment
        if data['employment_length'] < 1:
            factors.append("Limited employment history")
        
        # Credit utilization
        if data['num_credit_lines'] > 0:
            avg_debt_per_line = data['total_debt'] / data['num_credit_lines']
            if avg_debt_per_line > 5000:
                factors.append("High credit utilization")
        
        # Return top N factors
        return factors[:top_n] if factors else ["Standard risk factors"]


if __name__ == "__main__":
    # Example usage
    sample_data = {
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "dependents": 2,
        "employment_status": "Employed",
        "employment_length": 10,
        "income": 75000,
        "credit_history_length": 8,
        "num_credit_lines": 5,
        "num_open_accounts": 3,
        "total_debt": 25000,
        "loan_amount": 20000,
        "loan_term": 60,
        "loan_purpose": "home_improvement",
        "interest_rate": 7.5,
        "monthly_payment": 400,
        "num_late_payments": 1,
        "num_delinquencies": 0,
        "home_ownership": "Own",
        "education_level": "Bachelor",
        "has_cosigner": False,
        "has_guarantor": False
    }
    
    # Note: This requires a trained model file
    # predictor = CreditScorePredictor("models/xgboost_model.json")
    # score, prob = predictor.predict(sample_data)
    # print(f"Score: {score:.4f}, Probability: {prob:.4f}")
