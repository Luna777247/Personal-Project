"""
Generate synthetic credit scoring data for demonstration
"""
import pandas as pd
import numpy as np
from faker import Faker
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()
np.random.seed(42)


def generate_credit_data(
    n_samples: int = 10000,
    default_rate: float = 0.20,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate synthetic credit scoring dataset
    
    Args:
        n_samples: Number of samples to generate
        default_rate: Target default rate (0-1)
        save_path: Path to save CSV file
        
    Returns:
        Pandas DataFrame with credit data
    """
    logger.info(f"Generating {n_samples} synthetic credit records...")
    
    data = []
    
    for i in range(n_samples):
        # Demographics
        age = np.random.randint(18, 75)
        gender = np.random.choice(['Male', 'Female'])
        marital_status = np.random.choice(
            ['Single', 'Married', 'Divorced', 'Widowed'],
            p=[0.35, 0.50, 0.12, 0.03]
        )
        education_level = np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD'],
            p=[0.25, 0.50, 0.20, 0.05]
        )
        
        # Employment
        employment_status = np.random.choice(
            ['Employed', 'Self-employed', 'Unemployed'],
            p=[0.75, 0.20, 0.05]
        )
        
        if employment_status == 'Unemployed':
            income = np.random.uniform(0, 20000)
            employment_length = 0
        else:
            # Income based on age and education
            base_income = 30000
            if education_level == 'Bachelor':
                base_income = 50000
            elif education_level == 'Master':
                base_income = 70000
            elif education_level == 'PhD':
                base_income = 90000
            
            income = base_income + (age - 25) * 1000 + np.random.uniform(-10000, 20000)
            income = max(income, 0)
            employment_length = min(age - 18, np.random.randint(0, 30))
        
        # Housing
        home_ownership = np.random.choice(
            ['Rent', 'Own', 'Mortgage'],
            p=[0.35, 0.25, 0.40]
        )
        
        # Credit History
        credit_history_length = min(age - 18, np.random.randint(0, 35))
        num_credit_lines = np.random.randint(0, 15)
        num_open_accounts = np.random.randint(0, num_credit_lines + 1)
        
        # Debt
        total_debt = np.random.uniform(0, income * 2) if income > 0 else 0
        
        # Loan Details
        loan_purpose = np.random.choice(
            ['debt_consolidation', 'home_improvement', 'business', 
             'education', 'medical', 'car', 'other'],
            p=[0.30, 0.15, 0.10, 0.15, 0.10, 0.15, 0.05]
        )
        
        loan_amount = np.random.uniform(1000, min(income * 3, 100000)) if income > 0 else np.random.uniform(1000, 20000)
        loan_term = np.random.choice([12, 24, 36, 48, 60, 72], p=[0.05, 0.15, 0.30, 0.25, 0.20, 0.05])
        interest_rate = np.random.uniform(5, 25)
        
        # Monthly payment calculation
        r = interest_rate / 100 / 12
        n = loan_term
        if r > 0:
            monthly_payment = loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            monthly_payment = loan_amount / n
        
        # Payment History
        num_late_payments = np.random.poisson(1) if np.random.random() < 0.3 else 0
        num_delinquencies = np.random.poisson(0.5) if np.random.random() < 0.15 else 0
        
        # Credit Score (simulated, not used as feature to avoid data leakage)
        credit_score = int(np.random.normal(680, 80))
        credit_score = np.clip(credit_score, 300, 850)
        
        # Additional features
        has_cosigner = np.random.choice([True, False], p=[0.15, 0.85])
        has_guarantor = np.random.choice([True, False], p=[0.10, 0.90])
        
        # Calculate risk factors for target generation
        risk_score = 0
        
        # Income risk
        if income < 30000:
            risk_score += 3
        elif income < 50000:
            risk_score += 1
        
        # DTI risk
        dti = total_debt / income if income > 0 else 10
        if dti > 0.5:
            risk_score += 3
        elif dti > 0.4:
            risk_score += 2
        elif dti > 0.3:
            risk_score += 1
        
        # Payment history risk
        risk_score += num_late_payments
        risk_score += num_delinquencies * 2
        
        # Employment risk
        if employment_status == 'Unemployed':
            risk_score += 4
        elif employment_length < 1:
            risk_score += 2
        
        # Age risk
        if age < 25:
            risk_score += 1
        
        # Credit history risk
        if credit_history_length < 2:
            risk_score += 2
        
        # Loan amount risk
        loan_to_income = loan_amount / income if income > 0 else 100
        if loan_to_income > 3:
            risk_score += 3
        elif loan_to_income > 2:
            risk_score += 2
        elif loan_to_income > 1:
            risk_score += 1
        
        # Reduce risk with positive factors
        if has_cosigner:
            risk_score -= 1
        if has_guarantor:
            risk_score -= 1
        if home_ownership == 'Own':
            risk_score -= 1
        
        # Generate target (loan_status: 0=Good, 1=Default)
        # Higher risk_score = higher probability of default
        default_prob = min(0.95, max(0.05, risk_score / 20))
        
        # Adjust to match target default_rate
        if default_prob > default_rate:
            default_prob = default_prob * (default_rate / 0.5)
        
        loan_status = 1 if np.random.random() < default_prob else 0
        
        record = {
            'customer_id': f'CUST_{i:06d}',
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'education_level': education_level,
            'employment_status': employment_status,
            'employment_length': employment_length,
            'income': round(income, 2),
            'home_ownership': home_ownership,
            'credit_history_length': credit_history_length,
            'num_credit_lines': num_credit_lines,
            'num_open_accounts': num_open_accounts,
            'total_debt': round(total_debt, 2),
            'loan_purpose': loan_purpose,
            'loan_amount': round(loan_amount, 2),
            'loan_term': loan_term,
            'interest_rate': round(interest_rate, 2),
            'monthly_payment': round(monthly_payment, 2),
            'num_late_payments': num_late_payments,
            'num_delinquencies': num_delinquencies,
            'has_cosigner': has_cosigner,
            'has_guarantor': has_guarantor,
            'loan_status': loan_status  # Target: 0=Good, 1=Default
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Add some missing values randomly (5% missing rate)
    for col in ['employment_length', 'credit_history_length', 'num_late_payments']:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    
    logger.info(f"Generated {len(df)} records")
    logger.info(f"Default rate: {df['loan_status'].mean():.2%}")
    logger.info(f"Shape: {df.shape}")
    
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved to {save_path}")
    
    return df


if __name__ == "__main__":
    # Generate data
    df = generate_credit_data(n_samples=10000, default_rate=0.20)
    
    # Save to file
    df.to_csv("data/raw/credit_data.csv", index=False)
    
    print("\nDataset Summary:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nDefault Rate:", f"{df['loan_status'].mean():.2%}")
    print("\nMissing Values:")
    print(df.isnull().sum())
