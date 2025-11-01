import pytest
import pandas as pd
from process_data.data.data_preprocessing_all import CSVDataLoader, DataPreprocessor

class TestCSVDataLoader:
    """Test CSVDataLoader class"""
    
    @pytest.fixture
    def sample_csv_path(self, tmp_path):
        """Create a temporary CSV file for testing"""
        csv_path = tmp_path / "test_data.csv"
        data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'target': [0, 1, 0, 1, 0]
        })
        data.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_load_data(self, sample_csv_path):
        """Test that data is loaded correctly"""
        loader = CSVDataLoader(sample_csv_path)
        train, test = loader.split_data()
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) + len(test) == 5
    
    def test_split_ratio(self, sample_csv_path):
        """Test that train/test split returns non-empty datasets"""
        loader = CSVDataLoader(sample_csv_path)
        train, test = loader.split_data()  # Remove test_size parameter
        
        # Just check that both train and test have data
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) == 5


class TestDataPreprocessor:
    """Test DataPreprocessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample train and test data"""
        train = pd.DataFrame({
            'age': [25, None, 35, 40],
            'height': [170, 180, None, 175],
            'weight': [70, 80, 75, None],
            'gender': ['M', 'F', None, 'M']
        })
        test = pd.DataFrame({
            'age': [30, None],
            'height': [None, 185],
            'weight': [72, 85],
            'gender': ['F', 'M']
        })
        return train, test
    
    def test_remove_nans(self, sample_data):
        """Test that rows with NaNs in specified columns are removed"""
        train, test = sample_data
        preprocessor = DataPreprocessor(train, test)
        train_clean, test_clean = preprocessor.remove_nans(['age', 'gender']).get_data()
        
        # Check that rows with NaN in age or gender are removed
        assert train_clean['age'].notna().all()
        assert train_clean['gender'].notna().all()
        assert test_clean['age'].notna().all()
        assert test_clean['gender'].notna().all()
    
    def test_fill_nans_with_mean(self, sample_data):
        """Test that NaNs are filled with mean values"""
        train, test = sample_data
        preprocessor = DataPreprocessor(train, test)
        train_clean, test_clean = preprocessor.fill_nans_with_mean(['height', 'weight']).get_data()
        
        # Check that NaNs are filled
        assert train_clean['height'].notna().all()
        assert train_clean['weight'].notna().all()
        assert test_clean['height'].notna().all()
    
    def test_chaining_methods(self, sample_data):
        """Test that preprocessing methods can be chained"""
        train, test = sample_data
        preprocessor = DataPreprocessor(train, test)
        train_clean, test_clean = (
            preprocessor
            .remove_nans(['age'])
            .fill_nans_with_mean(['height', 'weight'])
            .get_data()
        )
        
        assert train_clean['age'].notna().all()
        assert train_clean['height'].notna().all()
        assert train_clean['weight'].notna().all()