import pytest
import pandas as pd
import numpy as np
from process_data.model.predictor import DiabetesModel

class TestDiabetesModel:
    """Test DiabetesModel class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training and test data"""
        np.random.seed(42)
        train = pd.DataFrame({
            'age': np.random.randint(20, 70, 100),
            'bmi': np.random.uniform(18, 35, 100),
            'gender_M': np.random.randint(0, 2, 100),
            'gender_F': np.random.randint(0, 2, 100),
            'diabetes_mellitus': np.random.randint(0, 2, 100)
        })
        test = pd.DataFrame({
            'age': np.random.randint(20, 70, 30),
            'bmi': np.random.uniform(18, 35, 30),
            'gender_M': np.random.randint(0, 2, 30),
            'gender_F': np.random.randint(0, 2, 30),
            'diabetes_mellitus': np.random.randint(0, 2, 30)
        })
        return train, test
    
    def test_model_initialization(self):
        """Test model is initialized correctly"""
        model = DiabetesModel(
            feature_columns=['age', 'bmi'],
            target_column='diabetes_mellitus'
        )
        
        assert model._feature_columns == ['age', 'bmi']
        assert model._target_column == 'diabetes_mellitus'
        assert model.model is not None
    
    def test_custom_hyperparameters(self):
        """Test model accepts custom hyperparameters"""
        hyperparams = {'n_estimators': 50, 'max_depth': 5}
        model = DiabetesModel(
            feature_columns=['age'],
            target_column='diabetes_mellitus',
            hyperparameters=hyperparams
        )
        
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 5
    
    def test_train(self, sample_data):
        """Test model training"""
        train, _ = sample_data
        model = DiabetesModel(
            feature_columns=['age', 'bmi', 'gender_M', 'gender_F'],
            target_column='diabetes_mellitus',
            hyperparameters={'n_estimators': 10, 'random_state': 42}
        )
        
        model.train(train)
        assert hasattr(model.model, 'estimators_')  # Check model is fitted
    
    def test_predict(self, sample_data):
        """Test prediction returns probabilities"""
        train, test = sample_data
        model = DiabetesModel(
            feature_columns=['age', 'bmi', 'gender_M', 'gender_F'],
            target_column='diabetes_mellitus',
            hyperparameters={'n_estimators': 10, 'random_state': 42}
        )
        
        model.train(train)
        predictions = model.predict(test)
        
        assert predictions.shape == (30, 2)  # 30 samples, 2 classes
        assert np.all((predictions >= 0) & (predictions <= 1))  # Probabilities
        assert np.allclose(predictions.sum(axis=1), 1)  # Sum to 1
    
    def test_predict_missing_columns(self, sample_data):
        """Test predict handles missing feature columns"""
        train, test = sample_data
        model = DiabetesModel(
            feature_columns=['age', 'bmi', 'gender_M', 'gender_F', 'ethnicity_Asian'],
            target_column='diabetes_mellitus',
            hyperparameters={'n_estimators': 10, 'random_state': 42}
        )
        
        # Add ethnicity_Asian to train but not test
        train['ethnicity_Asian'] = 0
        model.train(train)
        
        # Should not raise error (adds missing column with zeros)
        predictions = model.predict(test)
        assert predictions.shape[0] == len(test)
    
    def test_pred_auc_score(self, sample_data):
        """Test AUC score calculation"""
        train, test = sample_data
        model = DiabetesModel(
            feature_columns=['age', 'bmi', 'gender_M', 'gender_F'],
            target_column='diabetes_mellitus',
            hyperparameters={'n_estimators': 10, 'random_state': 42}
        )
        
        model.train(train)
        scores = model.pred_auc_score(train, test, verbose=False)
        
        assert 'train_auc' in scores
        assert 'test_auc' in scores
        assert 0 <= scores['train_auc'] <= 1
        assert 0 <= scores['test_auc'] <= 1
    
    def test_missing_target_column(self, sample_data):
        """Test error is raised if target column is missing"""
        train, _ = sample_data
        train_no_target = train.drop('diabetes_mellitus', axis=1)
        
        model = DiabetesModel(
            feature_columns=['age', 'bmi'],
            target_column='diabetes_mellitus'
        )
        
        with pytest.raises(ValueError, match="Missing target column"):
            model.train(train_no_target)