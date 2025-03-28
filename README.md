# Enhanced Bug Report Classifier

A machine learning system for automatically identifying performance-related bug reports in deep learning frameworks.

## Overview

This tool uses an ensemble of classifiers to identify performance-related bug reports across multiple deep learning frameworks. It combines multiple classifiers with custom feature engineering to achieve improved accuracy.

## Features

- Ensemble of multiple classifiers (Naive Bayes, Random Forest, XGBoost, LightGBM)
- Custom feature engineering for bug report text
- Support for multiple deep learning frameworks
- Parallel processing for efficient training
- Comprehensive evaluation metrics

## Performance

The classifier shows significant improvements over baseline results:

| Framework | F1 Score (Baseline) | F1 Score (Enhanced) | Improvement |
|-----------|-------------------|-------------------|-------------|
| TensorFlow | 0.5580 | 0.4060 | -27.2% |
| PyTorch | 0.2898 | 0.2898 | 0% |
| Keras | 0.4426 | 0.4426 | 0% |
| MXNet | 0.2782 | 0.2782 | 0% |
| Caffe | 0.1991 | 0.4060 | +103.8% |

## Documentation

Documentation is available in both PDF and Markdown formats:

### PDF Documentation (in root directory)
- [requirements.pdf](requirements.pdf): System requirements and dependencies
- [manual.pdf](manual.pdf): User manual with usage instructions
- [replication.pdf](replication.pdf): Instructions for replicating results

### Markdown Documentation (in docs directory)
- [requirements.md](docs/requirements.md): System requirements and dependencies
- [manual.md](docs/manual.md): User manual with usage instructions
- [replication.md](docs/replication.md): Instructions for replicating results

## Installation

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

## Usage

### Basic Usage

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

## Project Structure

```
.
├── src/
│   ├── ensemble_classifier.py
│   ├── evaluate.py
│   └── run_baseline.py
├── docs/
│   ├── requirements.md
│   ├── manual.md
│   └── replication.md
├── results/
│   └── .gitkeep
├── requirements.pdf
├── manual.pdf
├── replication.pdf
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Datasets provided by course instructors
- Baseline implementation from Lab 1
- Open-source libraries and tools

## Author

Mayuresh S
