"""
Unit tests for data processing
"""
import pytest
import pandas as pd
import numpy as np
from src.data import generate_credit_data, DataCleaner, DataScaler


class TestDataGenerator:
    """Test data generation"""
    
    def test_generate_credit_data_shape(self):
        """Test generated data has correct shape"""
        df = generate_credit_data(n_samples=100, seed=42)
        assert df.shape[0] == 100
        assert df.shape[1] > 20  # Should have many features
    
    def test_generate_credit_data_target(self):
        """Test target variable exists and is binary"""
        df = generate_credit_data(n_samples=100, seed=42)
        assert 'loan_status' in df.columns
        assert df['loan_status'].isin([0, 1]).all()
    
    def test_generate_credit_data_default_rate(self):
        """Test default rate approximately matches parameter"""
        df = generate_credit_data(n_samples=1000, default_rate=0.3, seed=42)
        actual_rate = df['loan_status'].mean()
        assert 0.25 <= actual_rate <= 0.35  # Allow 5% variance


class TestDataCleaner:
    """Test data cleaning"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with issues"""
        data = {
            'age': [25, 30, 35, -1, 200],  # Negative and outlier
            'income': [50000, 60000, np.nan, 70000, 80000],  # Missing value
            'debt': [10000, 15000, 20000, 25000, 30000]
        }
        return pd.DataFrame(data)
    
    def test_clean_data_removes_invalid_age(self, sample_data):
        """Test cleaning removes invalid ages"""
        cleaner = DataCleaner({})
        df_clean, report = cleaner.clean_data(sample_data)
        assert (df_clean['age'] >= 0).all()
        assert (df_clean['age'] <= 150).all()
    
    def test_clean_data_handles_missing(self, sample_data):
        """Test missing value handling"""
        cleaner = DataCleaner({
            'missing_values_strategy': {'numeric': 'median'}
        })
        df_clean, report = cleaner.clean_data(sample_data)
        assert not df_clean['income'].isna().any()


class TestDataScaler:
    """Test data scaling"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
    
    def test_fit_transform_standard(self, sample_data):
        """Test standard scaling"""
        scaler = DataScaler(scaler_type='standard')
        df_scaled = scaler.fit_transform(sample_data, sample_data.columns.tolist())
        
        # Check mean approximately 0 and std approximately 1
        assert abs(df_scaled['feature1'].mean()) < 1e-10
        assert abs(df_scaled['feature1'].std() - 1.0) < 1e-10
    
    def test_scaler_persistence(self, sample_data, tmp_path):
        """Test scaler can be saved and loaded"""
        scaler = DataScaler(scaler_type='standard')
        scaler.fit_transform(sample_data, sample_data.columns.tolist())
        
        # Save
        filepath = tmp_path / "scaler.pkl"
        scaler.save_scaler(str(filepath))
        
        # Load
        loaded_scaler = DataScaler.load_scaler(str(filepath))
        
        # Transform should give same results
        df_scaled1 = scaler.transform(sample_data)
        df_scaled2 = loaded_scaler.transform(sample_data)
        
        pd.testing.assert_frame_equal(df_scaled1, df_scaled2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
