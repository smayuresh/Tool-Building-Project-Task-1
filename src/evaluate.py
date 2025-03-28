import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats
from tqdm import tqdm
import os
from ensemble_classifier import EnhancedBugReportClassifier

def load_data(project_name):
    """Load dataset for a specific project"""
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            'lab1', 'datasets', f'{project_name}.csv')
    return pd.read_csv(file_path)

def evaluate_classifier(classifier, X, y, n_iterations=30):
    """Evaluate classifier with multiple iterations"""
    metrics = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for i in tqdm(range(n_iterations), desc="Evaluating"):
        # Split data 70-30
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            stratify=y,
            random_state=i
        )
        
        # Train and predict
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        # Calculate metrics
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
    
    return {
        'precision_mean': np.mean(metrics['precision']),
        'precision_std': np.std(metrics['precision']),
        'recall_mean': np.mean(metrics['recall']),
        'recall_std': np.std(metrics['recall']),
        'f1_mean': np.mean(metrics['f1']),
        'f1_std': np.std(metrics['f1']),
        'raw_metrics': metrics
    }

def create_baseline():
    """Create baseline classifier (Naive Bayes + TF-IDF)"""
    class BaselineClassifier:
        def __init__(self):
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=2,  # Remove very rare terms
                max_df=0.95  # Remove very common terms
            )
            self.classifier = MultinomialNB(
                class_prior=[0.3, 0.7]  # Prior probabilities for classes
            )
            
        def fit(self, X, y):
            # Combine text fields with title emphasis
            text = (
                X['Title'].fillna('') + ' ' +  # Original title
                X['Title'].fillna('') + ' ' +  # Repeat title for emphasis
                X['Body'].fillna('') + ' ' + 
                X['Comments'].fillna('')
            )
            X_transformed = self.vectorizer.fit_transform(text)
            self.classifier.fit(X_transformed, y)
            return self
            
        def predict(self, X):
            # Combine text fields with title emphasis
            text = (
                X['Title'].fillna('') + ' ' +  # Original title
                X['Title'].fillna('') + ' ' +  # Repeat title for emphasis
                X['Body'].fillna('') + ' ' + 
                X['Comments'].fillna('')
            )
            X_transformed = self.vectorizer.transform(text)
            return self.classifier.predict(X_transformed)
    
    return BaselineClassifier()

def main():
    # Projects to evaluate
    projects = ['tensorflow', 'pytorch', 'keras', 'incubator-mxnet', 'caffe']
    
    # Store aggregate metrics across all projects
    aggregate_metrics = {
        'baseline': {
            'precision': [], 'recall': [], 'f1': []
        },
        'enhanced': {
            'precision': [], 'recall': [], 'f1': []
        }
    }
    
    results = {}
    for project in projects:
        print(f"\nEvaluating on {project} dataset:")
        
        # Load data
        data = load_data(project)
        print(f"Dataset size: {len(data)} samples ({sum(data['class'] == 1)} positive, {sum(data['class'] == 0)} negative)")
        
        # Create classifiers
        baseline = create_baseline()
        enhanced = EnhancedBugReportClassifier()
        
        # Evaluate both classifiers
        print("\nBaseline performance:")
        baseline_results = evaluate_classifier(baseline, data, data['class'])
        
        print("\nEnhanced classifier performance:")
        enhanced_results = evaluate_classifier(enhanced, data, data['class'])
        
        # Store results
        results[project] = {
            'baseline': baseline_results,
            'enhanced': enhanced_results
        }
        
        # Add to aggregate metrics
        aggregate_metrics['baseline']['precision'].append(baseline_results['precision_mean'])
        aggregate_metrics['baseline']['recall'].append(baseline_results['recall_mean'])
        aggregate_metrics['baseline']['f1'].append(baseline_results['f1_mean'])
        
        aggregate_metrics['enhanced']['precision'].append(enhanced_results['precision_mean'])
        aggregate_metrics['enhanced']['recall'].append(enhanced_results['recall_mean'])
        aggregate_metrics['enhanced']['f1'].append(enhanced_results['f1_mean'])
        
        # Print results
        print(f"\nResults for {project}:")
        print("\nBaseline:")
        print(f"Precision: {baseline_results['precision_mean']:.3f} (±{baseline_results['precision_std']:.3f})")
        print(f"Recall: {baseline_results['recall_mean']:.3f} (±{baseline_results['recall_std']:.3f})")
        print(f"F1 Score: {baseline_results['f1_mean']:.3f} (±{baseline_results['f1_std']:.3f})")
        
        print("\nEnhanced:")
        print(f"Precision: {enhanced_results['precision_mean']:.3f} (±{enhanced_results['precision_std']:.3f})")
        print(f"Recall: {enhanced_results['recall_mean']:.3f} (±{enhanced_results['recall_std']:.3f})")
        print(f"F1 Score: {enhanced_results['f1_mean']:.3f} (±{enhanced_results['f1_std']:.3f})")
        
        # Perform statistical tests
        for metric in ['precision', 'recall', 'f1']:
            stat, p_value = stats.wilcoxon(
                enhanced_results['raw_metrics'][metric],
                baseline_results['raw_metrics'][metric]
            )
            print(f"\n{metric.capitalize()} - Wilcoxon test p-value: {p_value:.4f}")
    
    # Print aggregate results
    print("\n=== Aggregate Results Across All Projects ===")
    print("\nBaseline:")
    print(f"Average Precision: {np.mean(aggregate_metrics['baseline']['precision']):.3f} (±{np.std(aggregate_metrics['baseline']['precision']):.3f})")
    print(f"Average Recall: {np.mean(aggregate_metrics['baseline']['recall']):.3f} (±{np.std(aggregate_metrics['baseline']['recall']):.3f})")
    print(f"Average F1 Score: {np.mean(aggregate_metrics['baseline']['f1']):.3f} (±{np.std(aggregate_metrics['baseline']['f1']):.3f})")
    
    print("\nEnhanced:")
    print(f"Average Precision: {np.mean(aggregate_metrics['enhanced']['precision']):.3f} (±{np.std(aggregate_metrics['enhanced']['precision']):.3f})")
    print(f"Average Recall: {np.mean(aggregate_metrics['enhanced']['recall']):.3f} (±{np.std(aggregate_metrics['enhanced']['recall']):.3f})")
    print(f"Average F1 Score: {np.mean(aggregate_metrics['enhanced']['f1']):.3f} (±{np.std(aggregate_metrics['enhanced']['f1']):.3f})")
    
    # Perform statistical tests on aggregate results
    for metric in ['precision', 'recall', 'f1']:
        stat, p_value = stats.wilcoxon(
            aggregate_metrics['enhanced'][metric],
            aggregate_metrics['baseline'][metric]
        )
        print(f"\nAggregate {metric.capitalize()} - Wilcoxon test p-value: {p_value:.4f}")

if __name__ == "__main__":
    main() 