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
        hyperparameters: Optional[Dict] = None,
    ):
        # private attributes
        self._feature_columns = feature_columns
        self._target_column = target_column
        self._hyperparameters = hyperparameters or {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42,
        }
        # public model
        self.model = RandomForestClassifier(**self._hyperparameters)

    def train(self, train_data: pd.DataFrame) -> None:
        missing = [c for c in self._feature_columns if c not in train_data.columns]
        if missing:
            raise ValueError(f"Missing feature columns in train data: {missing}")
        if self._target_column not in train_data.columns:
            raise ValueError(f"Missing target column: {self._target_column}")
        X = train_data[self._feature_columns]
        y = train_data[self._target_column]
        self.model.fit(X, y)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        # Align input columns to training features (add missing with zeros)
        X = data.copy()
        for col in self._feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self._feature_columns]
        return self.model.predict_proba(X)

    def pred_auc_score(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, verbose: bool = True
    ) -> Dict[str, float]:
        for df, name in ((train_data, "train"), (test_data, "test")):
            missing = [c for c in self._feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f"Missing feature columns in {name} data: {missing}")
            if self._target_column not in df.columns:
                raise ValueError(
                    f"Missing target column '{self._target_column}' in {name} data."
                )
        train_probs = self.model.predict_proba(train_data[self._feature_columns])[:, 1]
        test_probs = self.model.predict_proba(test_data[self._feature_columns])[:, 1]
        y_train = train_data[self._target_column]
        y_test = test_data[self._target_column]
        train_auc = float(roc_auc_score(y_train, train_probs))
        test_auc = float(roc_auc_score(y_test, test_probs))
        if verbose:
            print(f"Train AUC: {train_auc:.4f}")
            print(f"Test  AUC: {test_auc:.4f}")
        return {"train_auc": train_auc, "test_auc": test_auc}