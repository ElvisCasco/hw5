import pandas as pd
from sklearn.model_selection import train_test_split

class CSVDataLoader:
    def __init__(self, file_name: str, test_size: float = 0.2, random_state: int = 42):
        self.file_name = file_name
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.file_name)
        print(f"\nData loaded. Shape: {data.shape}\n")
        return data

    def split_data(self):
        data = self.load_data()
        train, test = train_test_split(data, test_size=self.test_size, random_state=self.random_state)
        print(f"Data split: \nTrain ({train.shape}), \nTest ({test.shape})\n")
        return train, test

class DataPreprocessor:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.train = train_df.copy()
        self.test = test_df.copy()

    def remove_nans(self, subset=['age', 'gender', 'ethnicity']):
        self.train = self.train.dropna(subset=subset)
        self.test = self.test.dropna(subset=subset)
        print(f"Data after removing rows with NaNs in {subset}: \nTrain ({self.train.shape}), \nTest ({self.test.shape})\n")
        return self

    def fill_nans_with_mean(self, columns=['height', 'weight']):
        for col in columns:
            if col in self.train.columns:
                self.train[col] = self.train[col].fillna(self.train[col].mean())
            if col in self.test.columns:
                self.test[col] = self.test[col].fillna(self.test[col].mean())
        print(f"Filled NaNs in {columns} with mean.\n")
        return self

    def get_data(self):
        return self.train, self.test
    
