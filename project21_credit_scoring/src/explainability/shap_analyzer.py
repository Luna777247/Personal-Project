"""
SHAP-based model explainability for credit scoring
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """SHAP analysis for model explainability"""
    
    def __init__(self, model, model_type: str = 'xgboost'):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained model (XGBoost or LightGBM)
            model_type: Type of model ('xgboost' or 'lightgbm')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
    
    def create_explainer(
        self,
        X_background: pd.DataFrame,
        n_samples: int = 100
    ):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background data for SHAP
            n_samples: Number of background samples
        """
        logger.info(f"Creating SHAP explainer with {n_samples} background samples")
        
        # Sample background data
        if len(X_background) > n_samples:
            X_background = X_background.sample(n=n_samples, random_state=42)
        
        self.feature_names = X_background.columns.tolist()
        
        # Create explainer
        if self.model_type == 'xgboost':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'lightgbm':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info("SHAP explainer created")
    
    def compute_shap_values(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute SHAP values for data
        
        Args:
            X: Input features
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        logger.info(f"Computing SHAP values for {len(X)} samples")
        
        self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, shap_values might be a list
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Positive class
        
        return self.shap_values
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        save_path: str = None,
        max_display: int = 20
    ):
        """
        Plot SHAP summary plot (feature importance)
        
        Args:
            X: Input features
            save_path: Path to save plot
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot saved to {save_path}")
        
        plt.close()
    
    def plot_bar(
        self,
        X: pd.DataFrame,
        save_path: str = None,
        max_display: int = 20
    ):
        """
        Plot SHAP bar plot (mean absolute SHAP values)
        
        Args:
            X: Input features
            save_path: Path to save plot
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bar plot saved to {save_path}")
        
        plt.close()
    
    def plot_waterfall(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        save_path: str = None
    ):
        """
        Plot SHAP waterfall for single prediction
        
        Args:
            X: Input features
            sample_idx: Index of sample to explain
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.explainer.expected_value,
            data=X.iloc[sample_idx].values,
            feature_names=X.columns.tolist()
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {save_path}")
        
        plt.close()
    
    def plot_force(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        save_path: str = None
    ):
        """
        Plot SHAP force plot for single prediction
        
        Args:
            X: Input features
            sample_idx: Index of sample to explain
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Initialize JS for matplotlib
        shap.initjs()
        
        # Create force plot
        force_plot = shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[sample_idx],
            X.iloc[sample_idx],
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {save_path}")
        
        plt.close()
    
    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: str = None,
        save_path: str = None
    ):
        """
        Plot SHAP dependence plot
        
        Args:
            X: Input features
            feature: Feature to plot
            interaction_feature: Interaction feature (auto-detected if None)
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            interaction_index=interaction_feature,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dependence plot saved to {save_path}")
        
        plt.close()
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get feature importance based on SHAP values
        
        Args:
            X: Input features
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        top_n: int = 10
    ) -> Dict:
        """
        Explain single prediction
        
        Args:
            X: Input features
            sample_idx: Index of sample to explain
            top_n: Number of top features to return
            
        Returns:
            Explanation dictionary
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get SHAP values for sample
        sample_shap = self.shap_values[sample_idx]
        sample_features = X.iloc[sample_idx]
        
        # Base value
        base_value = self.explainer.expected_value
        
        # Prediction
        prediction = base_value + sample_shap.sum()
        
        # Top contributing features
        feature_contributions = pd.DataFrame({
            'feature': X.columns,
            'value': sample_features.values,
            'shap_value': sample_shap
        })
        
        feature_contributions['abs_shap'] = np.abs(feature_contributions['shap_value'])
        feature_contributions = feature_contributions.sort_values('abs_shap', ascending=False)
        
        top_features = feature_contributions.head(top_n)[['feature', 'value', 'shap_value']].to_dict('records')
        
        explanation = {
            'base_value': float(base_value),
            'prediction': float(prediction),
            'probability': float(1 / (1 + np.exp(-prediction))),  # Sigmoid for probability
            'top_features': top_features
        }
        
        return explanation
    
    def generate_report(
        self,
        X: pd.DataFrame,
        output_dir: str = 'models/explanations'
    ):
        """
        Generate comprehensive SHAP report
        
        Args:
            X: Input features
            output_dir: Output directory for plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating SHAP report in {output_dir}")
        
        # Compute SHAP values
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Generate plots
        self.plot_summary(X, save_path=str(output_dir / 'shap_summary.png'))
        self.plot_bar(X, save_path=str(output_dir / 'shap_importance.png'))
        
        # Sample predictions
        for i in [0, 1, 2]:
            if i < len(X):
                self.plot_waterfall(X, sample_idx=i, 
                                    save_path=str(output_dir / f'shap_waterfall_sample_{i}.png'))
        
        # Feature importance
        importance = self.get_feature_importance(X, top_n=20)
        importance.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        logger.info("SHAP report generated")


if __name__ == "__main__":
    from src.models import ModelTrainer
    from src.data import generate_credit_data
    from src.features import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Generate and prepare data
    df = generate_credit_data(n_samples=2000)
    engineer = FeatureEngineer({})
    df_engineered = engineer.engineer_features(df, fit=True)
    
    feature_cols = [col for col in df_engineered.columns 
                    if col not in ['loan_status', 'customer_id']]
    X = df_engineered[feature_cols].fillna(0)
    y = df_engineered['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train model
    config = {
        'model': {
            'type': 'xgboost',
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 50,
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            }
        }
    }
    
    trainer = ModelTrainer(config)
    trainer.train(X_train, y_train, X_test, y_test)
    
    # SHAP analysis
    analyzer = SHAPAnalyzer(trainer.model, model_type='xgboost')
    analyzer.create_explainer(X_train, n_samples=100)
    
    # Generate report
    analyzer.generate_report(X_test)
    
    # Explain sample prediction
    explanation = analyzer.explain_prediction(X_test, sample_idx=0, top_n=10)
    
    print("\nSample Prediction Explanation:")
    print(f"Base value: {explanation['base_value']:.4f}")
    print(f"Prediction: {explanation['prediction']:.4f}")
    print(f"Probability: {explanation['probability']:.4f}")
    print("\nTop Contributing Features:")
    for feat in explanation['top_features'][:5]:
        print(f"  {feat['feature']}: {feat['value']:.2f} (SHAP: {feat['shap_value']:.4f})")
