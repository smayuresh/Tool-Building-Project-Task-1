import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats
import re
from tqdm import tqdm

class EnhancedBugReportClassifier(BaseEstimator, ClassifierMixin):
    """
    An ensemble classifier for bug report classification that combines:
    1. TF-IDF + Naive Bayes (baseline approach)
    2. TF-IDF + Random Forest (better with high-dimensional data)
    3. Custom features + XGBoost (good with engineered features)
    4. Custom features + LightGBM (efficient with large datasets)
    """
    
    def __init__(self):
        # Text vectorizer with improved settings
        self.tfidf = TfidfVectorizer(
            max_features=10000,  # Increased from 5000
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,  # Remove very rare terms
            max_df=0.95  # Remove very common terms
        )
        
        # Base classifiers with class weight adjustments
        self.naive_bayes = MultinomialNB(
            class_prior=[0.3, 0.7]  # Prior probabilities for classes
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=15,  # Increased from 10
            class_weight='balanced',
            n_jobs=-1
        )
        self.xgboost = xgb.XGBClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=6,
            learning_rate=0.05,  # Reduced from 0.1
            scale_pos_weight=8,  # Handle class imbalance
            n_jobs=-1
        )
        self.lightgbm = lgb.LGBMClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=6,
            learning_rate=0.05,  # Reduced from 0.1
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Meta classifier with class weight
        self.meta_classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=1000
        )
        
        # Enhanced performance patterns
        self.performance_patterns = {
            'memory': r'\b(memory|ram|gpu|cuda|leak|oom|allocation|heap|stack|buffer|memory_usage|memory_limit)\b',
            'speed': r'\b(slow|fast|speed|latency|throughput|performance|bottleneck|optimization|efficient|inefficient)\b',
            'resource': r'\b(cpu|gpu|disk|io|utilization|resource|consumption|usage|load|bandwidth)\b',
            'error': r'\b(error|exception|crash|fail|bug|issue|problem|invalid|incorrect)\b',
            'timing': r'\b(time|duration|delay|timeout|wait|synchronization)\b',
            'scale': r'\b(scale|large|big|huge|massive|memory-intensive|cpu-intensive)\b'
        }
    
    def _extract_features(self, text):
        """Extract enhanced custom features from text"""
        if pd.isna(text):
            text = ""
            
        text_lower = text.lower()
        features = {}
        
        # Count performance-related terms with context
        for category, pattern in self.performance_patterns.items():
            matches = re.findall(pattern, text_lower)
            features[f'{category}_count'] = len(matches)
            features[f'{category}_density'] = len(matches) / (len(text_lower.split()) + 1)
        
        # Add enhanced text features
        code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
        features.update({
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text else 0,
            'code_block_count': len(code_blocks),
            'code_lines': sum(block.count('\n') for block in code_blocks),
            'has_numbers': bool(re.search(r'\d+', text)),
            'number_count': len(re.findall(r'\d+', text)),
            'has_urls': bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'has_stacktrace': bool(re.search(r'(?:Exception|Error|Stack trace|Traceback)', text, re.IGNORECASE)),
            'has_metrics': bool(re.search(r'\d+\s*(?:ms|sec|min|gb|mb|kb)', text, re.IGNORECASE))
        })
        
        return features
    
    def _prepare_data(self, X):
        """Prepare both TF-IDF and custom features"""
        # Combine text fields with weights
        combined_text = (
            X['Title'].fillna('') + ' ' +  # Title is important
            X['Title'].fillna('') + ' ' +  # Repeat title for emphasis
            X['Body'].fillna('') + ' ' + 
            X['Comments'].fillna('')
        )
        
        # TF-IDF features
        if not hasattr(self.tfidf, 'vocabulary_'):
            tfidf_features = self.tfidf.fit_transform(combined_text)
        else:
            tfidf_features = self.tfidf.transform(combined_text)
        
        # Custom features
        custom_features = []
        for idx, row in X.iterrows():
            # Extract features from each field separately
            title_features = self._extract_features(row['Title'])
            body_features = self._extract_features(row['Body'])
            comments_features = self._extract_features(row['Comments'])
            
            # Combine features
            combined_features = {}
            for key in title_features:
                # Weight title features more heavily
                combined_features[f'title_{key}'] = title_features[key] * 2
                combined_features[f'body_{key}'] = body_features[key]
                combined_features[f'comments_{key}'] = comments_features[key]
            
            custom_features.append(combined_features)
        
        custom_features_df = pd.DataFrame(custom_features)
        
        return tfidf_features, custom_features_df
    
    def fit(self, X, y):
        """Train the ensemble classifier"""
        # Prepare features
        tfidf_features, custom_features = self._prepare_data(X)
        
        # Train base classifiers
        self.naive_bayes.fit(tfidf_features, y)
        self.random_forest.fit(tfidf_features, y)
        self.xgboost.fit(custom_features, y)
        self.lightgbm.fit(custom_features, y)
        
        # Get predictions for meta-classifier
        nb_pred = self.naive_bayes.predict_proba(tfidf_features)[:, 1]
        rf_pred = self.random_forest.predict_proba(tfidf_features)[:, 1]
        xgb_pred = self.xgboost.predict_proba(custom_features)[:, 1]
        lgb_pred = self.lightgbm.predict_proba(custom_features)[:, 1]
        
        # Add confidence scores as features
        meta_features = np.column_stack([
            nb_pred, rf_pred, xgb_pred, lgb_pred,
            nb_pred * rf_pred,  # Interaction terms
            xgb_pred * lgb_pred
        ])
        
        # Train meta-classifier
        self.meta_classifier.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.3).astype(int)  # Lower threshold for positive class
    
    def predict_proba(self, X):
        """Get probability predictions"""
        # Prepare features
        tfidf_features, custom_features = self._prepare_data(X)
        
        # Get predictions from base classifiers
        nb_pred = self.naive_bayes.predict_proba(tfidf_features)[:, 1]
        rf_pred = self.random_forest.predict_proba(tfidf_features)[:, 1]
        xgb_pred = self.xgboost.predict_proba(custom_features)[:, 1]
        lgb_pred = self.lightgbm.predict_proba(custom_features)[:, 1]
        
        # Add confidence scores as features
        meta_features = np.column_stack([
            nb_pred, rf_pred, xgb_pred, lgb_pred,
            nb_pred * rf_pred,  # Interaction terms
            xgb_pred * lgb_pred
        ])
        
        # Get probabilities from meta-classifier
        return self.meta_classifier.predict_proba(meta_features) 