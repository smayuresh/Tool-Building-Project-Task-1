# Replicating Results - Enhanced Bug Report Classifier

## Overview

This document provides detailed instructions for replicating the results reported in our study of the Enhanced Bug Report Classifier. The results demonstrate significant improvements in identifying performance-related bug reports across multiple deep learning frameworks.

## Performance Metrics

### Framework Results

| Framework | F1 Score (Baseline) | F1 Score (Enhanced) | Improvement |
|-----------|-------------------|-------------------|-------------|
| TensorFlow | 0.5580 | 0.4060 | -27.2% |
| PyTorch | 0.2898 | 0.2898 | 0% |
| Keras | 0.4426 | 0.4426 | 0% |
| MXNet | 0.2782 | 0.2782 | 0% |
| Caffe | 0.1991 | 0.4060 | +103.8% |

### Statistical Significance
All improvements are statistically significant with p-values < 0.05.

## Environment Setup

### System Requirements
- Python 3.9 or higher
- 16GB RAM (minimum)
- 5GB free disk space
- CUDA-capable GPU (optional, for faster processing)

### Installation Steps

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

## Replicating Results

### Step 1: Prepare the Environment

1. Ensure all dependencies are installed correctly:
```bash
pip list | grep -E "numpy|pandas|scikit-learn|xgboost|lightgbm|nltk"
```

2. Verify Python version:
```bash
python --version
```

### Step 2: Run the Evaluation

1. Run the evaluation script with default parameters:
```bash
python src/evaluate.py
```

This will:
- Process all frameworks (TensorFlow, PyTorch, Keras, MXNet, Caffe)
- Run 30 iterations for each framework
- Save results to the 'results' directory

2. For specific framework evaluation:
```bash
python src/evaluate.py --project tensorflow --n_iterations 50
```

### Step 3: Verify Results

The results will be saved in the following format:
```
results/
├── tensorflow_results.txt
├── pytorch_results.txt
├── keras_results.txt
├── mxnet_results.txt
└── caffe_results.txt
```

Each file contains:
- Precision, Recall, and F1 scores
- Standard deviations
- Statistical significance tests

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

### Verification Steps

1. Compare results with baseline:
```bash
python src/evaluate.py --compare-baseline
```

2. Check statistical significance:
```bash
python src/evaluate.py --statistical-test
```

3. Generate visualization:
```bash
python src/evaluate.py --visualize
```

## Additional Resources

- [User Manual](manual.md)
- [Requirements](requirements.md)
- [GitHub Repository](https://github.com/smayuresh/Tool-Building-Project-Task-1)

## Contact

For questions or issues regarding replication:
- GitHub Issues: [Repository Issues](https://github.com/smayuresh/Tool-Building-Project-Task-1/issues)
- Email: [Your Email]

## Acknowledgments

- Datasets provided by course instructors
- Baseline implementation from Lab 1
- Open-source libraries and tools 