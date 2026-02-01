"""
Feature selection for credit scoring
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select most important features for credit scoring"""
    
    def __init__(self, config: Dict):
        """
        Initialize feature selector
        
        Args:
            config: Feature selection configuration
        """
        self.config = config
        self.selected_features = None
        self.feature_scores = None
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'all',
        n_features: int = 30
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using multiple methods
        
        Args:
            X: Feature dataframe
            y: Target variable
            method: Selection method ('correlation', 'univariate', 'rfe', 'importance', 'all')
            n_features: Number of features to select
            
        Returns:
            Tuple of (selected features dataframe, feature names list)
        """
        logger.info(f"Selecting features. Input: {X.shape[1]} features")
        
        feature_scores = {}
        
        if method in ['correlation', 'all']:
            feature_scores['correlation'] = self._correlation_selection(X, y)
        
        if method in ['univariate', 'all']:
            feature_scores['univariate'] = self._univariate_selection(X, y)
        
        if method in ['rfe', 'all']:
            feature_scores['rfe'] = self._rfe_selection(X, y, n_features)
        
        if method in ['importance', 'all']:
            feature_scores['importance'] = self._importance_selection(X, y)
        
        # Combine scores
        combined_scores = self._combine_scores(feature_scores)
        
        # Select top N features
        selected_features = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_features]
        
        self.selected_features = [f[0] for f in selected_features]
        self.feature_scores = dict(selected_features)
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return X[self.selected_features], self.selected_features
    
    def _correlation_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Select features based on correlation with target
        """
        correlations = {}
        
        for col in X.columns:
            try:
                corr = abs(X[col].corr(y))
                correlations[col] = corr if not np.isnan(corr) else 0
            except:
                correlations[col] = 0
        
        logger.info("Computed correlation scores")
        return correlations
    
    def _univariate_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Select features using univariate statistical tests
        """
        # F-score
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X.fillna(0), y)
        
        f_scores = dict(zip(X.columns, selector.scores_))
        
        # Normalize scores to 0-1
        max_score = max(f_scores.values())
        f_scores = {k: v / max_score for k, v in f_scores.items()}
        
        logger.info("Computed univariate scores")
        return f_scores
    
    def _rfe_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int
    ) -> Dict[str, float]:
        """
        Recursive Feature Elimination
        """
        # Use RandomForest as base estimator
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        rfe = RFE(estimator=rf, n_features_to_select=n_features, step=1)
        rfe.fit(X.fillna(0), y)
        
        # Higher ranking = less important (1 is best)
        # Invert ranking to score
        max_rank = max(rfe.ranking_)
        rfe_scores = dict(zip(
            X.columns,
            [(max_rank - rank + 1) / max_rank for rank in rfe.ranking_]
        ))
        
        logger.info("Computed RFE scores")
        return rfe_scores
    
    def _importance_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Select features based on RandomForest importance
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10
        )
        
        rf.fit(X.fillna(0), y)
        
        importances = dict(zip(X.columns, rf.feature_importances_))
        
        logger.info("Computed feature importance scores")
        return importances
    
    def _combine_scores(
        self,
        feature_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Combine scores from different methods
        
        Args:
            feature_scores: Dictionary of method -> {feature: score}
            
        Returns:
            Combined scores
        """
        combined = {}
        
        # Get all features
        all_features = set()
        for scores in feature_scores.values():
            all_features.update(scores.keys())
        
        # Average scores across methods
        for feature in all_features:
            scores = [
                scores_dict.get(feature, 0)
                for scores_dict in feature_scores.values()
            ]
            combined[feature] = np.mean(scores)
        
        return combined
    
    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.9
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features
        
        Args:
            X: Feature dataframe
            threshold: Correlation threshold
            
        Returns:
            Tuple of (filtered dataframe, removed features list)
        """
        corr_matrix = X.corr().abs()
        
        # Upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > threshold)
        ]
        
        logger.info(f"Removing {len(to_drop)} highly correlated features")
        
        return X.drop(columns=to_drop), to_drop
    
    def get_feature_report(self) -> pd.DataFrame:
        """
        Generate feature selection report
        
        Returns:
            DataFrame with feature scores
        """
        if self.feature_scores is None:
            raise ValueError("No features selected yet")
        
        report = pd.DataFrame([
            {'feature': k, 'score': v}
            for k, v in self.feature_scores.items()
        ])
        
        report = report.sort_values('score', ascending=False)
        
        return report


if __name__ == "__main__":
    from src.data import generate_credit_data
    from src.features.feature_engineer import FeatureEngineer
    
    # Generate and engineer data
    df = generate_credit_data(n_samples=1000)
    
    engineer = FeatureEngineer({})
    df_engineered = engineer.engineer_features(df, fit=True)
    
    # Separate features and target
    feature_cols = [col for col in df_engineered.columns 
                    if col not in ['loan_status', 'customer_id']]
    X = df_engineered[feature_cols]
    y = df_engineered['loan_status']
    
    # Select features
    selector = FeatureSelector({})
    X_selected, selected_features = selector.select_features(
        X, y,
        method='all',
        n_features=30
    )
    
    print("\nFeature Selection Results:")
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {len(selected_features)}")
    
    print("\nTop 10 Features:")
    report = selector.get_feature_report()
    print(report.head(10))
    
    # Remove correlated features
    X_filtered, dropped = selector.remove_correlated_features(X_selected, threshold=0.9)
    print(f"\nRemoved {len(dropped)} correlated features")
    print(f"Final features: {X_filtered.shape[1]}")
