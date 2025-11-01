import pytest
import pandas as pd
import numpy as np
from process_data.features.feature_transform import (
    BMICalculator, 
    EthnicityEncoder, 
    GenderBinaryEncoder
)

class TestBMICalculator:
    """Test BMICalculator class"""
    
    def test_bmi_calculation(self):
        """Test BMI is calculated correctly"""
        data = pd.DataFrame({
            'height': [1.75, 1.80, 1.65],
            'weight': [70, 80, 60]
        })
        
        calculator = BMICalculator()
        result = calculator.transform(data)
        
        assert 'bmi' in result.columns
        expected_bmi = [70 / (1.75**2), 80 / (1.80**2), 60 / (1.65**2)]
        np.testing.assert_array_almost_equal(result['bmi'].values, expected_bmi, decimal=2)
    
    def test_get_feature_names(self):
        """Test feature names are returned correctly"""
        calculator = BMICalculator()
        assert calculator.get_feature_names() == ['bmi']
    
    def test_zero_height_handling(self):
        """Test that zero height is handled (replaced with NA)"""
        data = pd.DataFrame({
            'height': [0, 1.80],
            'weight': [70, 80]
        })
        
        calculator = BMICalculator()
        result = calculator.transform(data)
        
        assert pd.isna(result['bmi'].iloc[0])
        assert not pd.isna(result['bmi'].iloc[1])


class TestEthnicityEncoder:
    """Test EthnicityEncoder class"""
    
    def test_one_hot_encoding(self):
        """Test ethnicity is one-hot encoded correctly"""
        data = pd.DataFrame({
            'ethnicity': ['Asian', 'Caucasian', 'Asian', 'African']
        })
        
        encoder = EthnicityEncoder()
        result = encoder.transform(data)
        
        # Check that ethnicity_* columns are created
        ethnicity_cols = [col for col in result.columns if col.startswith('ethnicity_')]
        assert len(ethnicity_cols) == 3  # 3 unique ethnicities
        
        # Check values are binary
        for col in ethnicity_cols:
            assert result[col].isin([0, 1]).all()
    
    def test_get_feature_names(self):
        """Test feature names are stored correctly"""
        data = pd.DataFrame({
            'ethnicity': ['Asian', 'Caucasian']
        })
        
        encoder = EthnicityEncoder()
        encoder.transform(data)
        
        feature_names = encoder.get_feature_names()
        assert 'ethnicity_Asian' in feature_names
        assert 'ethnicity_Caucasian' in feature_names


class TestGenderBinaryEncoder:
    """Test GenderBinaryEncoder class"""
    
    def test_gender_encoding_default(self):
        """Test gender is encoded with default values"""
        data = pd.DataFrame({
            'gender': ['M', 'F', 'Male', 'Female', 'm', 'f']
        })
        
        encoder = GenderBinaryEncoder()
        result = encoder.transform(data)
        
        assert 'gender_M' in result.columns
        assert 'gender_F' in result.columns
        
        # Check correct encoding
        assert result['gender_M'].tolist() == [1, 0, 1, 0, 1, 0]
        assert result['gender_F'].tolist() == [0, 1, 0, 1, 0, 1]
    
    def test_gender_encoding_custom_values(self):
        """Test gender encoding with custom values"""
        data = pd.DataFrame({
            'gender': ['Boy', 'Girl', 'BOY', 'girl']
        })
        
        encoder = GenderBinaryEncoder(
            male_vals=['Boy', 'BOY'],
            female_vals=['Girl', 'GIRL']
        )
        result = encoder.transform(data)
        
        assert result['gender_M'].tolist() == [1, 0, 1, 0]
        assert result['gender_F'].tolist() == [0, 1, 0, 1]
    
    def test_get_feature_names(self):
        """Test feature names are returned correctly"""
        encoder = GenderBinaryEncoder()
        assert encoder.get_feature_names() == ['gender_M', 'gender_F']
    
    def test_missing_gender_values(self):
        """Test handling of missing or unknown gender values"""
        data = pd.DataFrame({
            'gender': ['M', None, 'Unknown', 'F']
        })
        
        encoder = GenderBinaryEncoder()
        result = encoder.transform(data)
        
        # Unknown values should be encoded as 0 for both
        assert result['gender_M'].iloc[2] == 0
        assert result['gender_F'].iloc[2] == 0