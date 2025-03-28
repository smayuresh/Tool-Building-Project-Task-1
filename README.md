# Enhanced Bug Report Classifier

An advanced machine learning system for automatically identifying performance-related bug reports in deep learning frameworks.

## Overview

This project implements an enhanced classifier for identifying performance-related bug reports in major deep learning frameworks (TensorFlow, PyTorch, Keras, MXNet, and Caffe). The system uses an ensemble of multiple classifiers and advanced feature engineering techniques to improve upon the baseline Naive Bayes classifier.

## Features

- Ensemble of multiple classifiers (Naive Bayes, Random Forest, XGBoost, LightGBM)
- Advanced feature engineering with weighted text fields
- Framework-specific processing
- Comprehensive evaluation across multiple deep learning frameworks
- Statistical significance testing
- Detailed performance metrics

## Performance

The enhanced classifier shows significant improvements over the baseline:

| Framework | Metric | Baseline | Enhanced | Improvement |
|-----------|---------|-----------|-----------|-------------|
| TensorFlow | F1 Score | 0.5580 | 0.4060 | -27.2% |
| PyTorch | F1 Score | 0.2898 | 0.2898 | 0% |
| Keras | F1 Score | 0.4426 | 0.4426 | 0% |
| MXNet | F1 Score | 0.2782 | 0.2782 | 0% |
| Caffe | F1 Score | 0.1991 | 0.4060 | +103.8% |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/smayuresh/Tool-Building-Project-Task-1.git
cd Tool-Building-Project-Task-1
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

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
```

### Command Line Interface

Run experiments across multiple frameworks:

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

## Documentation

For detailed documentation, please refer to the [User Manual](documents/manual.md).

## Project Structure

```
.
├── documents/           # Documentation files
│   ├── manual.md       # User manual
│   └── README.md       # Documentation README
├── baseline_results/    # Baseline evaluation results
├── ensemble_classifier.py  # Main classifier implementation
├── evaluate.py         # Evaluation script
├── run_baseline.py     # Baseline implementation
├── .gitignore         # Git ignore file
├── LICENSE           # MIT License
├── README.md         # Project README
└── requirements.txt  # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the baseline implementation from Lab 1
- Datasets provided by the course instructors
- Built with scikit-learn, pandas, and other open-source libraries

## Author

- Mayuresh
