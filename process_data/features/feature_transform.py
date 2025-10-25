from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class FeatureTransformer(ABC):
    """Abstract base class for feature transformers"""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> list:
        """Return the names of transformed features"""
        pass

class BMICalculator(FeatureTransformer):
    """Calculate BMI from height and weight"""
    print(f"Encoded columns: ")
    print(f"\nBMI = Weight / Height^2")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate BMI = weight / height^2
        data['bmi'] = data['weight'] / (data['height'] ** 2)
        return data
    
    def get_feature_names(self) -> list:
        return ['bmi']

class EthnicityEncoder(FeatureTransformer):
    """Encode ethnicity using one-hot encoding"""
    
    def __init__(self):
        self.feature_names = []
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # One-hot encode ethnicity
        ethnicity_dummies = pd.get_dummies(data['ethnicity'], prefix='ethnicity')
        # Store feature names
        self.feature_names = ethnicity_dummies.columns.tolist()
        return pd.concat([data, ethnicity_dummies], axis=1)
    
    def get_feature_names(self) -> list:
        return self.feature_names
        

class GenderBinaryEncoder(FeatureTransformer):
    """Compact binary encoder for gender: creates gender_M and gender_F"""
    def __init__(self, male_vals=None, female_vals=None):
        self.feature_names = []
        self.male_vals = {v.upper() for v in (male_vals or ['M', 'MALE'])}
        self.female_vals = {v.upper() for v in (female_vals or ['F', 'FEMALE'])}

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        s = data['gender'].astype(str).str.strip().str.upper().fillna('')
        out = data.copy()
        out['gender_M'] = s.isin(self.male_vals).astype(int)
        out['gender_F'] = s.isin(self.female_vals).astype(int)
        self.feature_names = ['gender_M', 'gender_F']
        return out

    def get_feature_names(self) -> list:
        return self.feature_names