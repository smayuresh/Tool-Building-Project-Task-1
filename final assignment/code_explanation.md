# Code Explanation: Enhanced Bug Report Classifier

This document provides a detailed explanation of the code structure and implementation details for the Enhanced Bug Report Classifier.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Main Script (main.py)](#main-script-mainpy)
3. [Ensemble Classifier (ensemble_classifier.py)](#ensemble-classifier-ensemble_classifierpy)
4. [Evaluation Framework (evaluate.py)](#evaluation-framework-evaluatepy)
5. [Visualization (visualization.py)](#visualization-visualizationpy)
6. [Workflow and Dataflow](#workflow-and-dataflow)

## Project Structure

The code is organized into four main Python modules, each with a specific role in the bug report classification system:

```
src/
├── main.py                 # Main execution script
├── ensemble_classifier.py  # Enhanced classifier implementation
├── evaluate.py             # Evaluation framework
└── visualization.py        # Result visualization tools
```

## Main Script (main.py)

The `main.py` script serves as the entry point for the application, orchestrating the entire workflow:

```python
def main():
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run evaluation comparing baseline and enhanced classifiers
    results, json_file, framework_names = run_evaluation()
    
    # Create visualizations
    visualize_results(results, framework_names)
```

### Key Components:

1. **Data Loading (`load_data`)**:
   - Attempts to load datasets from multiple possible locations
   - Supports flexible dataset placement
   - Handles file not found errors gracefully

2. **Baseline Results Loading (`load_baseline_results`)**:
   - Loads baseline results from Lab 1 experiments
   - Extracts metrics from result files
   - Falls back to default values if results are not found

3. **Text Preprocessing (`preprocess_text`)**:
   - Converts text to lowercase
   - Handles missing values (NaN)
   - Ensures all text is in string format

4. **Evaluation Pipeline (`run_evaluation`)**:
   - Loads datasets for all frameworks
   - Compares enhanced classifier against baseline
   - Generates detailed results files

## Ensemble Classifier (ensemble_classifier.py)

The `EnhancedBugReportClassifier` class implements the core classification logic:

```python
class EnhancedBugReportClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # Initialize vectorizers and classifiers
        
    def _extract_features(self, text):
        # Custom feature extraction
        
    def fit(self, X, y):
        # Train the ensemble model
        
    def predict(self, X):
        # Make predictions
        
    def predict_proba(self, X):
        # Get probability scores
```

### Key Innovations:

1. **Multi-Stage Classification**:
   - TF-IDF vectorization for text processing
   - Multiple base classifiers (Naive Bayes, Random Forest)
   - Meta-classifier to combine predictions

2. **Pattern-Based Feature Extraction**:
   - Custom regular expressions for performance-related terms
   - Categorized patterns (memory, speed, resource, error, timing)
   - Density and count features for each category

3. **Text Field Weighting**:
   - Emphasizes title fields by repeating them
   - Combines title, body, and comments with appropriate weights
   - Maintains structure of bug report information

4. **Enhanced Preprocessing**:
   - Handles missing values gracefully
   - Standardizes text formatting
   - Extracts relevant n-grams (unigrams, bigrams, trigrams)

## Evaluation Framework (evaluate.py)

The evaluation module provides robust testing of classifier performance:

```python
def evaluate_classifier(classifier, X, y, n_iterations=10, name="Classifier"):
    # Evaluate over multiple iterations with different train/test splits
    
    # Calculate and return metrics (precision, recall, F1)
```

### Key Features:

1. **Multiple Iterations**:
   - Runs 10 evaluation iterations with different random seeds
   - Ensures statistically valid results
   - Computes mean and standard deviation for metrics

2. **Stratified Sampling**:
   - Maintains class distribution in train/test splits
   - Ensures fair evaluation on imbalanced datasets
   - Uses 70/30 train/test split ratio

3. **Comprehensive Metrics**:
   - Precision: Accuracy of positive predictions
   - Recall: Ability to find all positive instances
   - F1 Score: Harmonic mean of precision and recall

4. **Statistical Analysis**:
   - Wilcoxon signed-rank tests for statistical significance
   - Compares enhanced classifier against baseline
   - Tests significance for each metric and framework

## Visualization (visualization.py)

The visualization module creates informative plots to compare performance:

```python
def visualize_results(results, framework_names):
    # Generate multiple visualization plots
    
    # Save them to the results directory
```

### Key Visualizations:

1. **F1 Score Comparison**:
   - Bar chart comparing baseline and enhanced F1 scores
   - Shows performance across all frameworks
   - Includes error bars for standard deviation

2. **Performance Change**:
   - Percentage improvement/decline for each framework
   - Color-coded bars (green for improvement, red for decline)
   - Horizontal reference line at 0%

3. **Precision-Recall Comparison**:
   - Scatter plots showing precision-recall trade-offs
   - Arrows indicating direction of change
   - Separate plot for each framework

4. **Summary Metrics**:
   - Average performance across all frameworks
   - Comparison of precision, recall, and F1
   - Percentage change annotations

## Workflow and Dataflow

The application follows a clear workflow:

1. **Data Loading**:
   - Load bug report datasets for each framework
   - Preprocess text fields (title, body, comments)
   - Load baseline results from Lab 1

2. **Model Training and Evaluation**:
   - Train the enhanced classifier on multiple train/test splits
   - Evaluate performance using precision, recall, and F1 score
   - Compare against baseline performance

3. **Results Generation**:
   - Generate detailed text reports with metrics
   - Create JSON file with results for programmatic access
   - Generate visualization plots for analysis

4. **Visualization**:
   - Create comparison plots for visual analysis
   - Save plots to results directory
   - Display aggregate performance metrics

The data flows through the system as follows:

```
Bug Report CSV → Preprocessed Data → Feature Extraction → Model Training → Evaluation → Results & Visualizations
```

Each stage transforms the data to extract more meaningful insights from the bug reports, ultimately leading to improved classification of performance-related bugs in deep learning frameworks. 