import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Data for visualization
frameworks = ['TensorFlow', 'PyTorch', 'Keras', 'MXNet', 'Caffe']
baseline_f1 = [0.5580, 0.2898, 0.4426, 0.2782, 0.1991]
enhanced_f1 = [0.4060, 0.2898, 0.4426, 0.2782, 0.4060]
changes = [-27.2, 0, 0, 0, 103.8]

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")

# 1. F1 Score Comparison Plot
plt.figure(figsize=(12, 6))
x = np.arange(len(frameworks))
width = 0.35

plt.bar(x - width/2, baseline_f1, width, label='Baseline', color='skyblue')
plt.bar(x + width/2, enhanced_f1, width, label='Enhanced', color='lightcoral')

plt.xlabel('Framework')
plt.ylabel('F1 Score')
plt.title('Baseline vs Enhanced F1 Scores by Framework')
plt.xticks(x, frameworks, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('results/f1_comparison.png')
plt.close()

# 2. Performance Change Plot
plt.figure(figsize=(10, 6))
plt.bar(frameworks, changes, color='lightgreen')
plt.xlabel('Framework')
plt.ylabel('Change (%)')
plt.title('Performance Change by Framework')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/performance_change.png')
plt.close()

# 3. Feature Importance Heatmap
feature_importance = {
    'Memory Usage': [0.245, 0.234, 0.167, 0.178, 0.278],
    'Training Time': [0.198, 0.278, 0.198, 0.289, 0.234],
    'Model Size': [0.134, 0.167, 0.256, 0.234, 0.167],
    'Batch Size': [0.156, 0.145, 0.145, 0.156, 0.145],
    'GPU Utilization': [0.112, 0.098, 0.134, 0.123, 0.123]
}

df_features = pd.DataFrame(feature_importance, index=frameworks)

plt.figure(figsize=(12, 8))
sns.heatmap(df_features, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Feature Importance Across Frameworks')
plt.tight_layout()
plt.savefig('results/feature_importance.png')
plt.close()

# 4. Runtime Performance Plot
runtime_data = {
    'Processing Time (s)': [2.3, 2.1, 2.4, 2.2, 2.5],
    'Memory Usage (GB)': [1.2, 1.1, 1.3, 1.2, 1.3]
}

df_runtime = pd.DataFrame(runtime_data, index=frameworks)

plt.figure(figsize=(12, 6))
df_runtime.plot(kind='bar')
plt.xlabel('Framework')
plt.ylabel('Value')
plt.title('Runtime Performance Metrics')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('results/runtime_performance.png')
plt.close()

# 5. Precision-Recall Comparison
precision = [0.4500, 0.3200, 0.4800, 0.3100, 0.4500]
recall = [0.3700, 0.2650, 0.4100, 0.2500, 0.3700]

plt.figure(figsize=(10, 6))
plt.scatter(recall, precision)
for i, framework in enumerate(frameworks):
    plt.annotate(framework, (recall[i], precision[i]))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Comparison')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/precision_recall.png')
plt.close()

print("All visualization plots have been generated in the results directory.") 