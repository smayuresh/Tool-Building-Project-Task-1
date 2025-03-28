# Enhanced Bug Report Classifier for Deep Learning Frameworks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-1.3+-green.svg?logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.5+-red.svg?logo=matplotlib&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

<img src="https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png" width="200px" alt="scikit-learn logo">
<br/>
<br/>

</div>

This project implements an advanced ensemble-based classifier for identifying performance-related bugs in deep learning framework bug reports. It builds upon the baseline approach (Naive Bayes + TF-IDF) and demonstrates significant improvements across multiple frameworks.

## 📋 Project Overview

Bug report classification is a critical task for software maintenance, particularly for complex systems like deep learning frameworks. Performance-related bugs can be especially challenging to identify, as they often involve nuanced descriptions and technical terminology.

This project addresses this challenge by:

1. 🔄 Implementing an enhanced ensemble classifier that combines multiple classification approaches
2. 🔍 Adding custom feature extraction specifically targeting performance-related terminology
3. 🧠 Leveraging meta-learning to combine predictions from individual classifiers
4. 📊 Demonstrating improved results across multiple deep learning frameworks

## 📂 Project Structure

The project is organized as follows:

```
final assignment/
│
├── src/                        # Source code
│   ├── main.py                 # Main execution script
│   ├── ensemble_classifier.py  # Enhanced classifier implementation
│   ├── evaluate.py             # Evaluation framework
│   ├── visualization.py        # Result visualization tools
│   └── results/                # Output directory for results
│
├── datasets/                   # Input datasets
│   ├── tensorflow.csv
│   ├── pytorch.csv
│   ├── keras.csv
│   ├── incubator-mxnet.csv
│   └── caffe.csv
│
├── baseline_results/           # Original baseline results from Lab 1
│
├── README.md                   # This file
├── manual.pdf                  # Usage instructions
├── requirements.pdf            # Dependencies and requirements
└── replication.pdf             # Instructions for replicating results
```

## ✨ Key Features

### Enhanced Bug Report Classifier

<div align="center">
  <img src="https://mermaid.ink/img/pako:eNp1kctqwzAQRX9lmJUDTkMXJRSHNoGQ0iZ0k1XQo4miGoItGUkuDcb_XiuprSZpl-LO0Tl3hLSELOYECbxmPqgXYQJDV7rz3hoPxgb-WudQ5UJCclQmYKyZklJrOD-n4R9Zr9eRlM63QkeGLXvtVQ3HK9hZB4YBVZ3BBqNP3KL3VxzGtBdGt-kN0MbzgI_v02k-jI6_XuhWaxP8ug9vRLXx3-7LnfUFcDa8wTXjUaY1KG1BjFhUbJL69B9DJXo8pZb1s-FJ6YnmWumAFqyB1sqWC6iFr8D5Ak-2KXET4DcRbK5XCrLgO-TIuDKaauDkbCvb42_9qGCR5aIwVSW4F7YBHnuKAUl9PMGLZPa2yVZIZXHXjZM0KxbJLlskdJdGu3lCA_yCvIcka-E-DjSQeLgr-QPPhLHA?type=png" width="600px">
</div>

The enhanced classifier extends the baseline approach with:

1. **🔀 Multi-classifier Ensemble**: Combines Naive Bayes and Random Forest classifiers trained on TF-IDF features
2. **📝 Pattern-based Feature Extraction**: Custom extraction of performance-related terms categorized by type:
   - Memory issues: memory, ram, gpu, cuda, leak, oom, allocation, etc.
   - Speed issues: slow, fast, speed, latency, throughput, performance, etc.
   - Resource issues: cpu, gpu, disk, io, utilization, resource, etc.
   - Error indicators: error, exception, crash, fail, bug, issue, etc.
   - Timing issues: time, duration, delay, timeout, wait, etc.
3. **🎯 Meta-classification**: Uses Logistic Regression to combine predictions from base classifiers
4. **⚖️ Text Field Weighting**: Applies higher emphasis to title fields in bug reports
5. **🧹 Improved Preprocessing**: Better handling of text fields with comprehensive cleaning

### Evaluation Framework

<div align="center">
  <img src="https://mermaid.ink/img/pako:eNptksGKwjAQhl8lzGkFe-hBoaWou4IH8aJ48JKlna2BNglJKrXYd9-kqdXd3Wb-mfnmJ0OukMaUQwC7VJnkm9G2MTyV35XRph5rlSnpbVs1XBq9gWAjdQulZvRQWH8JCVebwBuHv9q2FQv9DsYo_sKyaEwDR0pTQUZ1gfP5LHjBpZTN6V5wnmODnf5Q1z5-DsehUHHBueKVFRWXPHrEfYdzwnbWVwOhhZUVJgpXuPVJPXSXfV3b5fU5_CJpmhpZG9UKdqrHoqpkuK-_MWitB0IEtCWF262EIGqQFjRdJsOUJbJMTyZnnmxgS1nCLNGm66X36JSARfpPH07G-fzNJvKSRYgDn-crhH3-w0OW0vYf6JeQvQ?type=png" width="600px">
</div>

The evaluation process includes:

1. **🔄 Multiple Iterations**: Each classifier is evaluated across 10 separate train/test splits
2. **📊 Stratified Sampling**: Preserves class distribution in each split
3. **📏 Multiple Metrics**: Precision, recall, and F1 score for comprehensive evaluation
4. **📉 Statistical Reporting**: Mean and standard deviation for each metric
5. **🌐 Cross-Framework Analysis**: Evaluation across five deep learning frameworks
6. **⚖️ Baseline Comparison**: Direct comparison with baseline results from Lab 1

## 📊 Results Summary

<div align="center">
  <table>
    <tr>
      <th>Framework</th>
      <th>Baseline F1</th>
      <th>Enhanced F1</th>
      <th>Improvement</th>
    </tr>
    <tr>
      <td><img src="https://www.tensorflow.org/images/tf_logo_social.png" width="20"/> TensorFlow</td>
      <td>0.5406</td>
      <td>0.5442 ± 0.0768</td>
      <td>+0.67% 📈</td>
    </tr>
    <tr>
      <td><img src="https://pytorch.org/assets/images/pytorch-logo.png" width="20"/> PyTorch</td>
      <td>0.5519</td>
      <td>0.5610 ± 0.0400</td>
      <td>+1.66% 📈</td>
    </tr>
    <tr>
      <td><img src="https://keras.io/img/logo.png" width="20"/> Keras</td>
      <td>0.5369</td>
      <td>0.6014 ± 0.0480</td>
      <td>+12.01% 📈</td>
    </tr>
    <tr>
      <td><img src="https://mxnet.apache.org/versions/1.9.1/assets/img/mxnet_logo.png" width="20"/> MXNet</td>
      <td>0.5479</td>
      <td>0.5161 ± 0.0689</td>
      <td>-5.81% 📉</td>
    </tr>
    <tr>
      <td><img src="https://caffe.berkeleyvision.org/tutorial/fig/caffe-logo.png" width="20"/> Caffe</td>
      <td>0.4428</td>
      <td>0.4692 ± 0.1238</td>
      <td>+5.97% 📈</td>
    </tr>
    <tr>
      <td><strong>Average</strong></td>
      <td><strong>0.5240</strong></td>
      <td><strong>0.5384</strong></td>
      <td><strong>+2.74% 📈</strong></td>
    </tr>
  </table>
</div>

The enhanced classifier demonstrated improvement on 4 out of 5 frameworks, with an overall F1 score improvement of 2.74%. The most significant improvement was observed on the Keras dataset (+12.01%).

## 📈 Visualizations

The evaluation generates several visualizations to aid in understanding the results:

<div align="center">
  <div style="display: flex; flex-wrap: wrap; justify-content: center;">
    <div style="flex: 1; min-width: 300px; max-width: 500px; margin: 10px;">
      <h3>1. F1 Score Comparison</h3>
      <img src="src/results/f1_comparison.png" width="100%" alt="F1 Score Comparison">
    </div>
    <div style="flex: 1; min-width: 300px; max-width: 500px; margin: 10px;">
      <h3>2. Performance Change</h3>
      <img src="src/results/performance_change.png" width="100%" alt="Performance Change">
    </div>
  </div>
  <div style="display: flex; flex-wrap: wrap; justify-content: center;">
    <div style="flex: 1; min-width: 300px; max-width: 500px; margin: 10px;">
      <h3>3. Precision-Recall Comparison</h3>
      <img src="src/results/precision_recall_comparison.png" width="100%" alt="Precision-Recall Comparison">
    </div>
    <div style="flex: 1; min-width: 300px; max-width: 500px; margin: 10px;">
      <h3>4. Summary Metrics</h3>
      <img src="src/results/summary_metrics.png" width="100%" alt="Summary Metrics">
    </div>
  </div>
</div>

## 🚀 Quick Start

To run the classifier and evaluation:

```bash
# Navigate to the source directory
cd final_assignment/src

# Run the main evaluation script
python main.py

# Generate visualizations only
python visualization.py
```

For detailed usage instructions, see [manual.pdf](manual.pdf).

## 📦 Dependencies

<div align="center">
  <table>
    <tr>
      <th>Library</th>
      <th>Version</th>
      <th>Purpose</th>
    </tr>
    <tr>
      <td><img src="https://numpy.org/images/logo.svg" width="20"/> NumPy</td>
      <td>≥ 1.20.0</td>
      <td>Numerical computing</td>
    </tr>
    <tr>
      <td><img src="https://pandas.pydata.org/static/img/pandas_mark.svg" width="20"/> pandas</td>
      <td>≥ 1.3.0</td>
      <td>Data manipulation</td>
    </tr>
    <tr>
      <td><img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="20"/> scikit-learn</td>
      <td>≥ 1.0.0</td>
      <td>Machine learning</td>
    </tr>
    <tr>
      <td><img src="https://matplotlib.org/_static/images/logo2.svg" width="20"/> matplotlib</td>
      <td>≥ 3.5.0</td>
      <td>Visualization</td>
    </tr>
    <tr>
      <td><img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width="20"/> seaborn</td>
      <td>≥ 0.11.0</td>
      <td>Statistical visualization</td>
    </tr>
  </table>
</div>

See [requirements.pdf](requirements.pdf) for a complete list of required dependencies.

## 🔄 Replication

For detailed instructions on how to replicate the results, see [replication.pdf](replication.pdf).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

<div align="center">
  <br>
  <img src="https://forthebadge.com/images/badges/built-with-science.svg"/>
  <img src="https://forthebadge.com/images/badges/made-with-python.svg"/>
  <br>
  <br>
</div> 