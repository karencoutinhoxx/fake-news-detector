import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsEnsemble:
    def __init__(self):
        self.gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.meta_classifier = LogisticRegression()
        self.scaler = StandardScaler()
    
    def fit(self, X_features: pd.DataFrame, y: np.ndarray):
        logger.info("Training Gradient Boosting...")
        X_features_scaled = self.scaler.fit_transform(X_features)
        self.gb_classifier.fit(X_features_scaled, y)
        
        logger.info("Training Random Forest...")
        self.rf_classifier.fit(X_features_scaled, y)
        
        logger.info("Training Meta Classifier...")
        gb_probs = self.gb_classifier.predict_proba(X_features_scaled)[:, 1]
        rf_probs = self.rf_classifier.predict_proba(X_features_scaled)[:, 1]
        
        meta_features = np.column_stack([gb_probs, rf_probs])
        self.meta_classifier.fit(meta_features, y)
        
        logger.info("Ensemble training complete")
    
    def predict_proba(self, X_features: pd.DataFrame) -> np.ndarray:
        X_features_scaled = self.scaler.transform(X_features)
        
        gb_probs = self.gb_classifier.predict_proba(X_features_scaled)[:, 1]
        rf_probs = self.rf_classifier.predict_proba(X_features_scaled)[:, 1]
        
        meta_features = np.column_stack([gb_probs, rf_probs])
        ensemble_probs = self.meta_classifier.predict_proba(meta_features)
        
        return ensemble_probs
    
    def predict(self, X_features: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X_features)
        return np.argmax(probs, axis=1)
    
    def save(self, filepath: str):
        import joblib
        joblib.dump({
            'gb_classifier': self.gb_classifier,
            'rf_classifier': self.rf_classifier,
            'meta_classifier': self.meta_classifier,
            'scaler': self.scaler
        }, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        import joblib
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found at {filepath}")
        
        ensemble = FakeNewsEnsemble()
        components = joblib.load(filepath)
        
        ensemble.gb_classifier = components['gb_classifier']
        ensemble.rf_classifier = components['rf_classifier']
        ensemble.meta_classifier = components['meta_classifier']
        ensemble.scaler = components['scaler']
        
        logger.info(f"Ensemble loaded from {filepath}")
        return ensemble
