from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class DiabetesModel:
    """Model class for diabetes prediction"""
    
    def __init__(
        self, 
        feature_columns: List[str],
        target_column: str,
        hyperparameters: Optional[Dict] = None
    ):
        """
        Initialize the model with feature columns and hyperparameters
        
        Args:
            feature_columns: List of column names to use as features
            target_column: Name of the target column
            hyperparameters: Dictionary of model hyperparameters (optional)
        """
        # Private attributes
        self._feature_columns = feature_columns
        self._target_column = target_column
        self._hyperparameters = hyperparameters or {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        }
        
        # Public model attribute
        self.model = RandomForestClassifier(**self._hyperparameters)
        
    def train(self, train_data: pd.DataFrame) -> None:
        """
        Train the model using the provided training data
        
        Args:
            train_data: DataFrame containing training data with features and target
        """
        if not all(col in train_data.columns for col in self._feature_columns):
            raise ValueError(f"Missing feature columns. Required: {self._feature_columns}")
            
        if self._target_column not in train_data.columns:
            raise ValueError(f"Missing target column: {self._target_column}")
            
        X = train_data[self._feature_columns]
        y = train_data[self._target_column]
        
        self.model.fit(X, y)
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions on new data
        
        Args:
            data: DataFrame containing features for prediction
            
        Returns:
            Array of predicted probabilities for each class
        """
        if not all(col in data.columns for col in self._feature_columns):
            raise ValueError(f"Missing feature columns. Required: {self._feature_columns}")
            
        X = data[self._feature_columns]
        return self.model.predict_proba(X)

    def pred_auc_score(self, train_data: pd.DataFrame, test_data: pd.DataFrame, verbose: bool = True) -> Dict[str, float]:
        """
        Compute ROC AUC on train and test DataFrames using the model's predict_proba.
        Returns a dict: {'train_auc': float, 'test_auc': float}
        """
        # Validate columns
        for df, name in ((train_data, "train"), (test_data, "test")):
            if not all(col in df.columns for col in self._feature_columns):
                raise ValueError(f"Missing feature columns in {name} data. Required: {self._feature_columns}")
            if self._target_column not in df.columns:
                raise ValueError(f"Missing target column '{self._target_column}' in {name} data.")
        
        # Get probabilities for positive class (assumes binary classification)
        train_probs = self.model.predict_proba(train_data[self._feature_columns])[:, 1]
        test_probs = self.model.predict_proba(test_data[self._feature_columns])[:, 1]
        
        y_train = train_data[self._target_column]
        y_test = test_data[self._target_column]
        
        train_auc = roc_auc_score(y_train, train_probs)
        test_auc = roc_auc_score(y_test, test_probs)
        
        #if verbose:
        #    print(f"\nTrain AUC: {train_auc:.4f}\n")
        #    print(f"\nTest AUC: {test_auc:.4f}\n")
        print(f"\nPrediction AUC score:")
        return {"train_auc": float(train_auc), "test_auc": float(test_auc)}
