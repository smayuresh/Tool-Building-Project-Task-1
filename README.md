# Enhanced Bug Report Classification Tool

## 📋 Project Overview

This project implements an enhanced bug report classification system that aims to improve upon baseline classifiers across multiple deep learning frameworks. The solution leverages advanced text preprocessing techniques and feature engineering to classify software bug reports effectively.

## 🌟 Key Results

Performance comparison across different frameworks:

| Framework   | Baseline F1 | Enhanced F1 | Change    |
|------------|-------------|-------------|-----------|
| TensorFlow | 0.5580      | 0.4060      | -27.2%    |
| PyTorch    | 0.2898      | 0.2898      | 0%        |
| Keras      | 0.4426      | 0.4426      | 0%        |
| MXNet      | 0.2782      | 0.2782      | 0%        |
| Caffe      | 0.1991      | 0.4060      | +103.8%   |

### Key Findings

1. **Framework-Specific Performance**: 
   - Significant improvement in Caffe (+103.8%)
   - Stable performance in PyTorch, Keras, and MXNet
   - Performance decrease in TensorFlow that requires investigation

2. **Performance Analysis**:
   - Best baseline performance: TensorFlow (F1: 0.5580)
   - Best enhanced performance: TensorFlow/Caffe (F1: 0.4060)
   - Most significant improvement: Caffe (+103.8%)

### Implementation Details

The project includes:
- Detailed results for each framework in the `results` directory
- Framework-specific performance metrics and analysis
- Statistical significance testing
- Runtime performance measurements

## 📊 Results Structure

All results are organized in the `results` directory with framework-specific files:
- `tensorflow_results.txt`
- `pytorch_results.txt`
- `keras_results.txt`
- `mxnet_results.txt`
- `caffe_results.txt`

Each file contains:
1. Performance Metrics
2. Detailed Metrics (Precision, Recall, F1)
3. Statistical Analysis
4. Feature Importance Analysis
5. Runtime Statistics

## Features

- Ensemble of multiple classifiers (Naive Bayes, Random Forest, XGBoost, LightGBM)
- Custom feature engineering for bug report text
- Support for multiple deep learning frameworks
- Parallel processing for efficient training
- Comprehensive evaluation metrics

## Documentation

Comprehensive documentation is available in PDF format in the root directory:

- [requirements.pdf](requirements.pdf): Detailed system requirements, dependencies, and installation instructions
- [manual.pdf](manual.pdf): Complete user manual with basic and advanced usage instructions
- [replication.pdf](replication.pdf): Step-by-step guide for replicating the reported results

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
├── baseline_results_20250328_072121.txt
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
