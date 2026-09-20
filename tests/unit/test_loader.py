import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.loader import FakeNewsDataLoader, DataSplitter

class TestDataSplitter:
    @pytest.fixture
    def sample_dataframe(self):
        data = {
            'content': [f'Article {i}' for i in range(100)],
            'label': [i % 2 for i in range(100)]
        }
        return pd.DataFrame(data)
    
    def test_split_data(self, sample_dataframe):
        splitter = DataSplitter(test_size=0.15, val_size=0.15)
        train, val, test = splitter.split_data(sample_dataframe)
        
        assert len(train) + len(val) + len(test) == len(sample_dataframe)
        assert len(test) == 15

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
