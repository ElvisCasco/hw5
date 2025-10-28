from abc import ABC, abstractmethod
import pandas as pd

class FeatureTransformer(ABC):
    """Abstract base class for feature transformers"""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_feature_names(self) -> list:
        raise NotImplementedError


class BMICalculator(FeatureTransformer):
    """Calculate BMI from height and weight (assumes height in meters)"""

    def __init__(self):
        self._feature_names = ["bmi"]

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        h = out["height"].replace(0, pd.NA)
        out["bmi"] = out["weight"] / (h ** 2)
        return out

    def get_feature_names(self) -> list:
        return self._feature_names


class EthnicityEncoder(FeatureTransformer):
    """One-hot encode ethnicity into ethnicity_* columns"""

    def __init__(self):
        self._feature_names = []

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        dummies = pd.get_dummies(out["ethnicity"], prefix="ethnicity", dtype=int)
        self._feature_names = dummies.columns.tolist()
        return pd.concat([out, dummies], axis=1)

    def get_feature_names(self) -> list:
        return self._feature_names


class GenderBinaryEncoder(FeatureTransformer):
    """Binary encode gender into gender_M and gender_F"""

    def __init__(self, male_vals=None, female_vals=None):
        self._feature_names = ["gender_M", "gender_F"]
        self.male_vals = {v.upper() for v in (male_vals or ["M", "MALE"])}
        self.female_vals = {v.upper() for v in (female_vals or ["F", "FEMALE"])}

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        s = out["gender"].astype(str).str.strip().str.upper()
        out["gender_M"] = s.isin(self.male_vals).astype(int)
        out["gender_F"] = s.isin(self.female_vals).astype(int)  # fixed: astype(int)
        return out

    def get_feature_names(self) -> list:
        return self._feature_names