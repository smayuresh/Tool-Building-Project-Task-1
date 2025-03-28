# Enhanced Bug Report Classifier - User Manual

This manual provides detailed instructions on how to set up and use the Enhanced Bug Report Classifier for identifying performance-related bug reports in deep learning frameworks.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Data Preparation](#data-preparation)
4. [Basic Usage](#basic-usage)
5. [Advanced Usage](#advanced-usage)
6. [Command-Line Interface](#command-line-interface)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/bug-report-classifier.git
cd bug-report-classifier
```

### Step 2: Set Up Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Configuration

The classifier can be configured through various parameters:

- **Preprocessing Options**:
  - `use_stemming`: Whether to apply stemming (default: True)
  - `use_lemmatization`: Whether to apply lemmatization (default: False)

- **Feature Extraction Options**:
  - `tfidf_ngram_range`: Range of n-grams for TF-IDF (default: (1, 3))
  - `tfidf_max_features`: Maximum number of features for TF-IDF (default: 10000)
  - `min_df`: Minimum document frequency (default: 2)
  - `max_df`: Maximum document frequency (default: 0.95)

- **Classifier Options**:
  - `class_prior`: Prior probabilities for Naive Bayes (default: [0.3, 0.7])
  - `n_estimators`: Number of trees for Random Forest (default: 200)
  - `max_depth`: Maximum depth for Random Forest (default: 15)
  - `scale_pos_weight`: Class weight for XGBoost (default: 8)
  - `class_weight`: Class weight for LightGBM (default: 'balanced')

Each of these parameters has a significant impact on the classifier's performance:

| Parameter | Impact |
|-----------|--------|
| `tfidf_ngram_range` | Higher ranges capture more context but increase feature space |
| `tfidf_max_features` | Higher values capture more unique terms but increase dimensionality |
| `class_prior` | Helps handle class imbalance in the dataset |
| `n_estimators` | More trees generally improve performance but increase training time |
| `scale_pos_weight` | Helps balance positive and negative classes |

## Data Preparation

The classifier expects input data in CSV format with the following columns:

- **Title**: The title of the bug report
- **Body**: The description of the bug report
- **Comments**: Additional comments on the bug report
- **class**: The binary label (1 for performance-related, 0 for non-performance-related)

Example dataset format:

```
Title,Body,Comments,class
"GPU memory usage is very high","At the beginning of training, the GPU memory usage is high...","This issue occurs with large batch sizes",1
"Error in training module","The training module fails when processing large datasets...","",0
```

## Basic Usage

### Using as a Module

```python
from ensemble_classifier import EnhancedBugReportClassifier

# Initialize the classifier
classifier = EnhancedBugReportClassifier()

# Load and preprocess your data
import pandas as pd
data = pd.read_csv("your_dataset.csv")

# Train the classifier
classifier.train(data)

# Evaluate the classifier
metrics = classifier.evaluate(data)

# Print metrics
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

### Using the Evaluation Script

We provide an evaluation script that runs experiments across multiple frameworks:

```bash
python evaluate.py
```

This script will:
1. Load datasets from all frameworks (TensorFlow, PyTorch, Keras, MXNet, Caffe)
2. Train and evaluate both baseline and enhanced classifiers
3. Print detailed results and statistical comparisons

## Advanced Usage

### Custom Feature Engineering

You can enhance the feature engineering process by modifying the feature weights:

```python
classifier = EnhancedBugReportClassifier(
    title_weight=2.0,  # Emphasize title importance
    body_weight=1.0,   # Standard weight for body
    comments_weight=0.5  # Reduce impact of comments
)
```

### Ensemble Configuration

The classifier uses an ensemble of multiple classifiers. You can configure individual classifiers:

```python
classifier = EnhancedBugReportClassifier(
    use_naive_bayes=True,
    use_random_forest=True,
    use_xgboost=True,
    use_lightgbm=True
)
```

### Saving and Loading Models

To save and load trained models:

```python
# Save the trained model
classifier.save_model('trained_model.pkl')

# Load the model later
loaded_classifier = EnhancedBugReportClassifier.load_model('trained_model.pkl')
```

## Command-Line Interface

The `evaluate.py` script provides a command-line interface for running experiments:

```bash
python evaluate.py [--project PROJECT] [--n_iterations N] [--output_dir DIR]
```

Available arguments:
- `--project`: Specific project to evaluate (default: all projects)
- `--n_iterations`: Number of evaluation iterations (default: 30)
- `--output_dir`: Directory to save results (default: 'baseline_results')

Example:
```bash
python evaluate.py --project tensorflow --n_iterations 50
```

## Interpreting Results

### Performance Metrics

The classifier reports the following metrics:

- **Precision**: Proportion of true positives among all positive predictions
- **Recall**: Proportion of true positives among all actual positives
- **F1 Score**: Harmonic mean of precision and recall

### Understanding the Results

A typical results output looks like this:

```
=== Results for tensorflow ===
Baseline:
- Precision: 0.5310 (±0.0373)
- Recall: 0.5905 (±0.0569)
- F1 Score: 0.5580 (±0.0390)

Enhanced:
- Precision: 0.5680 (±0.1720)
- Recall: 0.3370 (±0.1300)
- F1 Score: 0.4060 (±0.1210)
```

### Statistical Significance

The results include p-values from Wilcoxon tests to determine statistical significance:

```
Wilcoxon test p-values:
- Precision: 0.0000
- Recall: 0.0001
- F1 Score: 0.0000
```

## Troubleshooting

### Common Issues

1. **LightGBM Warnings**:
   - Warning: "No further splits with positive gain"
   - This is expected and doesn't affect performance

2. **Memory Issues**:
   - Solution: Reduce batch size or use smaller feature set
   - Adjust `tfidf_max_features` parameter

3. **Performance Issues**:
   - Solution: Use parallel processing
   - Optimize feature extraction
   - Reduce model complexity

4. **Dataset Format Issues**:
   - Ensure your dataset has required columns
   - Check for missing values
   - Verify data types

### Performance Tips

1. **Balancing Speed and Accuracy**:
   - For faster processing: Reduce feature set size
   - For better accuracy: Increase feature set size

2. **Memory Optimization**:
   - Process datasets in chunks
   - Use smaller feature sets
   - Enable parallel processing

3. **Improving Results**:
   - Adjust class weights
   - Modify feature weights
   - Tune classifier parameters

### Getting Help

If you encounter issues not covered in this manual, please check the GitHub repository for updates or open an issue with the details of your problem. 