# User Manual - Enhanced Bug Report Classifier

## Introduction

The Enhanced Bug Report Classifier is a machine learning system designed to automatically identify performance-related bug reports in deep learning frameworks. This manual provides comprehensive instructions for using the tool effectively.

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git (for version control)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/smayuresh/Tool-Building-Project-Task-1.git
cd Tool-Building-Project-Task-1
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Basic Usage

### As a Module

```python
from src.ensemble_classifier import EnhancedBugReportClassifier

# Initialize classifier
classifier = EnhancedBugReportClassifier()

# Train and evaluate
classifier.train(data)
metrics = classifier.evaluate(test_data)
```

### Command Line Interface

```bash
python src/evaluate.py [--project PROJECT] [--n_iterations N]
```

Available arguments:
- `--project`: Specific project to evaluate (default: all projects)
- `--n_iterations`: Number of evaluation iterations (default: 30)

Example:
```bash
python src/evaluate.py --project tensorflow --n_iterations 50
```

## Advanced Usage

### Custom Feature Engineering

The classifier supports custom feature engineering through the following methods:

```python
# Add custom features
classifier.add_custom_features(feature_function)

# Modify feature weights
classifier.set_feature_weights(weights)
```

### Ensemble Configuration

You can configure the ensemble of classifiers:

```python
# Set base classifiers
classifier.set_base_classifiers([
    'naive_bayes',
    'random_forest',
    'xgboost',
    'lightgbm'
])

# Configure meta-classifier
classifier.set_meta_classifier('logistic_regression')
```

### Saving and Loading Models

```python
# Save trained model
classifier.save_model('model.pkl')

# Load saved model
classifier.load_model('model.pkl')
```

## Configuration

### Preprocessing Options

```python
# Configure text preprocessing
classifier.set_preprocessing_options({
    'remove_urls': True,
    'remove_numbers': False,
    'lowercase': True
})
```

### Feature Extraction Settings

```python
# Set feature extraction parameters
classifier.set_feature_params({
    'max_features': 1000,
    'ngram_range': (1, 2),
    'min_df': 2
})
```

### Classifier Settings

```python
# Configure individual classifiers
classifier.set_classifier_params('xgboost', {
    'n_estimators': 200,
    'learning_rate': 0.1
})
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Process smaller chunks of data
   - Use smaller feature set

2. **Performance Issues**
   - Enable parallel processing
   - Use GPU acceleration
   - Optimize feature extraction

3. **Inconsistent Results**
   - Check random seed settings
   - Verify data preprocessing
   - Ensure consistent environment

### Getting Help

- Check the [GitHub Issues](https://github.com/smayuresh/Tool-Building-Project-Task-1/issues)
- Review the [Replication Guide](replication.md)
- Contact the maintainers

## Best Practices

1. **Data Preparation**
   - Clean and preprocess data thoroughly
   - Validate input format
   - Handle missing values appropriately

2. **Model Training**
   - Use cross-validation
   - Monitor training progress
   - Save checkpoints regularly

3. **Evaluation**
   - Use appropriate metrics
   - Compare with baselines
   - Document results

## Additional Resources

- [Requirements](requirements.md)
- [Replication Guide](replication.md)
- [GitHub Repository](https://github.com/smayuresh/Tool-Building-Project-Task-1) 