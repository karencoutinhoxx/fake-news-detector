import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import FakeNewsDataLoader, DataSplitter
from features.engineer import FeatureEngineer
from models.ensemble import FakeNewsEnsemble
from evaluation.metrics import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    logger.info("Starting Fake News Detection Ensemble Training")
    
    loader = FakeNewsDataLoader(data_dir=args.data_dir)
    
    if args.dataset == "isot":
        df = loader.load_isot_dataset()
    else:
        df = loader.load_custom_csv(args.dataset)
    
    if df is None:
        logger.error("Failed to load dataset. Exiting.")
        return 1
    
    logger.info(f"Loaded {len(df)} articles")
    
    splitter = DataSplitter(test_size=0.15, val_size=0.15, random_state=42)
    train_df, val_df, test_df = splitter.split_data(df)
    
    splitter.save_splits(train_df, val_df, test_df, output_dir=args.splits_dir)
    
    engineer = FeatureEngineer()
    
    logger.info("Extracting features...")
    train_features = engineer.combine_all_features(train_df)
    train_features = train_features.fillna(0)
    
    val_features = engineer.combine_all_features(val_df)
    val_features = val_features.fillna(0)
    
    test_features = engineer.combine_all_features(test_df)
    test_features = test_features.fillna(0)
    
    logger.info(f"Training ensemble...")
    ensemble = FakeNewsEnsemble()
    y_train = train_df['label'].values
    ensemble.fit(train_features, y_train)
    
    logger.info("Evaluating...")
    evaluator = ModelEvaluator()
    
    val_preds = ensemble.predict(val_features)
    val_probs = ensemble.predict_proba(val_features)
    
    val_metrics = evaluator.evaluate(val_df['label'].values, val_preds, val_probs[:, 1])
    logger.info(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    
    test_preds = ensemble.predict(test_features)
    test_probs = ensemble.predict_proba(test_features)
    
    test_metrics = evaluator.evaluate(test_df['label'].values, test_preds, test_probs[:, 1])
    logger.info(f"Test - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    Path("models").mkdir(exist_ok=True)
    ensemble.save("models/ensemble_final.pkl")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'train_size': len(train_df),
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in test_metrics.items()}
    }
    
    with open("models/ensemble_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Training complete!")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fake news detection ensemble")
    parser.add_argument('--dataset', default='isot', help='Dataset to use')
    parser.add_argument('--data-dir', default='data/raw', help='Data directory')
    parser.add_argument('--splits-dir', default='data/splits', help='Splits directory')
    
    args = parser.parse_args()
    exit(main(args))
