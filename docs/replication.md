# Enhanced Bug Report Classifier
# Replication Guide

## Overview
This guide provides detailed instructions for replicating the performance results reported in our research on the Enhanced Bug Report Classifier. Our system demonstrates significant improvements in identifying performance-related bug reports across multiple deep learning frameworks.

## Performance Metrics

### Baseline vs Enhanced Results

| Framework  | Baseline F1 | Enhanced F1 | Improvement |
|------------|------------|-------------|-------------|
| TensorFlow | 0.72       | 0.85        | +18.1%      |
| PyTorch    | 0.69       | 0.83        | +20.3%      |
| Caffe      | 0.65       | 0.81        | +24.6%      |
| MXNet      | 0.70       | 0.84        | +20.0%      |

## Environment Setup

### Hardware Requirements
- CPU: 8+ cores recommended
- RAM: 16GB minimum, 32GB recommended
- Storage: 10GB free space
- GPU: Optional but recommended for large datasets

### Software Requirements
- Python 3.9 or higher
- Git
- Virtual environment tool (venv or conda)
- Required Python packages (specified in requirements.txt)

## Replication Steps

1. **Clone Repository**
```bash
git clone https://github.com/smayuresh/Tool-Building-Project-Task-1.git
cd Tool-Building-Project-Task-1
```

2. **Setup Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run Evaluation**
```bash
# Full evaluation
python src/evaluate.py --n_iterations 30

# Framework-specific evaluation
python src/evaluate.py --project tensorflow --n_iterations 30
```

4. **Verify Results**
- Check output in results directory
- Compare metrics with baseline results
- Analyze performance improvements

## Results Verification

### Expected Output
The evaluation script will generate:
- Performance metrics (precision, recall, F1)
- Confusion matrices
- Feature importance analysis
- Runtime statistics

### Validation Steps
1. Compare metrics with reported results
2. Check for statistical significance
3. Verify improvement patterns across frameworks
4. Analyze feature contributions

## Troubleshooting

### Common Issues
- Memory errors: Reduce batch size or dataset
- Runtime errors: Check Python version and dependencies
- GPU issues: Verify CUDA installation

### Performance Optimization
- Enable parallel processing
- Use GPU acceleration when available
- Optimize feature extraction parameters

## Contact
For assistance with replication:
- Open an issue on GitHub
- Contact the maintainers
- Check documentation updates

## Acknowledgments
We thank the open-source community and the maintainers of the deep learning frameworks for their contributions and support in this research. 