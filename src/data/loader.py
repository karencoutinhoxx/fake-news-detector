import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsDataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_isot_dataset(self) -> pd.DataFrame:
        fake_path = self.data_dir / "fake.csv"
        real_path = self.data_dir / "true.csv"
        
        if not fake_path.exists() or not real_path.exists():
            logger.warning("ISOT dataset not found")
            return None
        
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        
        fake_df['label'] = 1
        real_df['label'] = 0
        
        df = pd.concat([fake_df, real_df], ignore_index=True)
        logger.info(f"Loaded {len(df)} articles")
        return df
    
    def load_custom_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows")
        return df

class DataSplitter:
    def __init__(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split_data(self, df: pd.DataFrame, stratify_column: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split
        
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[stratify_column]
        )
        
        adjusted_val_size = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val_size,
            random_state=self.random_state,
            stratify=train_val[stratify_column]
        )
        
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test
    
    def save_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, 
                   output_dir: str = "data/splits"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train.to_csv(output_dir / "train.csv", index=False)
        val.to_csv(output_dir / "val.csv", index=False)
        test.to_csv(output_dir / "test.csv", index=False)
        
        logger.info(f"Saved splits to {output_dir}")
