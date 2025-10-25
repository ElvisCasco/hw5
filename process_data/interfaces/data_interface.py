from abc import ABC, abstractmethod
import pandas as pd

class DataLoaderInterface(ABC):
    @abstractmethod
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass