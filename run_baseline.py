import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluate import load_data, evaluate_classifier, create_baseline
from datetime import datetime

def preprocess_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower()

def run_baseline_evaluation():
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'baseline_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'baseline_results_{timestamp}.txt')
    
    # Projects to evaluate
    projects = ['tensorflow', 'pytorch', 'keras', 'incubator-mxnet', 'caffe']
    
    with open(results_file, 'w') as f:
        f.write("Baseline Evaluation Results\n")
        f.write("=========================\n\n")
        
        for project in projects:
            f.write(f"\nProject: {project}\n")
            f.write("-" * 50 + "\n")
            
            # Load data
            data = load_data(project)
            
            # Preprocess text
            data['Title'] = data['Title'].apply(preprocess_text)
            data['Body'] = data['Body'].apply(preprocess_text)
            data['Comments'] = data['Comments'].apply(preprocess_text)
            
            # Combine text fields
            data['processed_text'] = data['Title'] + ' ' + data['Body'] + ' ' + data['Comments']
            
            X = data
            y = data['class']
            
            f.write(f"Dataset size: {len(data)} samples\n")
            f.write(f"Positive samples: {sum(y == 1)}\n")
            f.write(f"Negative samples: {sum(y == 0)}\n\n")
            
            # Create and evaluate baseline
            baseline = create_baseline()
            metrics = evaluate_classifier(baseline, X, y)
            
            # Write results
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Write detailed results for each iteration
            f.write("Detailed Results:\n")
            for i, (precision, recall, f1) in enumerate(zip(metrics['raw_metrics']['precision'], 
                                                          metrics['raw_metrics']['recall'], 
                                                          metrics['raw_metrics']['f1'])):
                f.write(f"Iteration {i+1}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n")
            f.write("\n")
            
        # Write summary statistics
        f.write("\nSummary Statistics\n")
        f.write("==================\n")
        for metric in ['precision', 'recall', 'f1']:
            f.write(f"\n{metric.capitalize()}:\n")
            f.write(f"Mean: {metrics[f'{metric}_mean']:.4f}\n")
            f.write(f"Std: {metrics[f'{metric}_std']:.4f}\n")

if __name__ == "__main__":
    run_baseline_evaluation() 