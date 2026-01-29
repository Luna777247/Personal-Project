"""
Pydantic models for API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level categories"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CreditScoreRequest(BaseModel):
    """Credit score request model"""
    # Demographics
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    gender: str = Field(..., description="Gender (Male/Female)")
    marital_status: str = Field(..., description="Marital status")
    dependents: int = Field(..., ge=0, le=10, description="Number of dependents")
    
    # Employment
    employment_status: str = Field(..., description="Employment status")
    employment_length: int = Field(..., ge=0, le=50, description="Employment length in years")
    
    # Income
    income: float = Field(..., gt=0, description="Annual income")
    
    # Credit
    credit_history_length: int = Field(..., ge=0, le=50, description="Credit history in years")
    num_credit_lines: int = Field(..., ge=0, le=50, description="Number of credit lines")
    num_open_accounts: int = Field(..., ge=0, le=50, description="Number of open accounts")
    
    # Debt
    total_debt: float = Field(..., ge=0, description="Total debt amount")
    
    # Loan
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term: int = Field(..., gt=0, le=360, description="Loan term in months")
    loan_purpose: str = Field(..., description="Loan purpose")
    interest_rate: float = Field(..., gt=0, le=100, description="Interest rate (%)")
    monthly_payment: float = Field(..., gt=0, description="Monthly payment amount")
    
    # Payment History
    num_late_payments: int = Field(..., ge=0, description="Number of late payments")
    num_delinquencies: int = Field(..., ge=0, description="Number of delinquencies")
    
    # Home
    home_ownership: str = Field(..., description="Home ownership status")
    
    # Education
    education_level: str = Field(..., description="Education level")
    
    # Co-signer
    has_cosigner: bool = Field(False, description="Has co-signer")
    has_guarantor: bool = Field(False, description="Has guarantor")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class FeatureContribution(BaseModel):
    """Feature contribution in prediction"""
    feature: str
    value: float
    shap_value: float


class CreditScoreResponse(BaseModel):
    """Credit score response model"""
    score: float = Field(..., description="Credit score (0-1)")
    probability: float = Field(..., description="Default probability")
    risk_level: RiskLevel = Field(..., description="Risk level category")
    approval_decision: str = Field(..., description="Approval decision")
    key_factors: List[str] = Field(..., description="Key decision factors")


class ExplainResponse(BaseModel):
    """Explanation response model"""
    score: float
    probability: float
    risk_level: RiskLevel
    approval_decision: str
    base_value: float
    top_features: List[FeatureContribution]


class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    applications: List[CreditScoreRequest]


class BatchPredictResponse(BaseModel):
    """Batch prediction response"""
    results: List[CreditScoreResponse]
    summary: Dict[str, int]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """API metrics response"""
    total_predictions: int
    avg_response_time_ms: float
    approval_rate: float
    avg_risk_score: float
