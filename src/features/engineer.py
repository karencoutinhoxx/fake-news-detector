import pandas as pd
import numpy as np
from textblob import TextBlob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        pass
    
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        features['text_length'] = df['content'].str.len()
        features['word_count'] = df['content'].str.split().str.len()
        features['avg_word_length'] = features['text_length'] / features['word_count']
        features['uppercase_ratio'] = df['content'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        features['digit_ratio'] = df['content'].apply(
            lambda x: sum(1 for c in x if c.isdigit()) / len(x) if len(x) > 0 else 0
        )
        
        logger.info("Extracted basic features")
        return features
    
    def extract_linguistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        features['polarity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        features['subjectivity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        features['exclamation_count'] = df['content'].str.count('!')
        features['question_count'] = df['content'].str.count(r'\?')
        
        logger.info("Extracted linguistic features")
        return features
    
    def extract_clickbait_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        features['all_caps_words'] = df['content'].apply(
            lambda x: sum(1 for word in str(x).split() if word.isupper() and len(word) > 1)
        )
        
        urgency_words = ['must', 'urgent', 'shocking', 'breaking', 'exposed']
        features['urgency_markers'] = df['content'].apply(
            lambda x: sum(str(x).lower().count(word) for word in urgency_words)
        )
        
        logger.info("Extracted clickbait features")
        return features
    
    def combine_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        basic = self.extract_basic_features(df)
        linguistic = self.extract_linguistic_features(df)
        clickbait = self.extract_clickbait_features(df)
        
        combined = pd.concat([basic, linguistic, clickbait], axis=1)
        logger.info(f"Combined features: {combined.shape[1]} features total")
        
        return combined
