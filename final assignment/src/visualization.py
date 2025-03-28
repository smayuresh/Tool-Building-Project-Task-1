import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import re

def load_baseline_results(project_name):
    """Load baseline results from Lab 1"""
    # Try to load baseline results from Lab 1
    possible_paths = [
        f"../baseline_results/{project_name}_results.txt",       # Lab1 results
        f"../baseline_results/{project_name}_NB.csv",            # Lab1 NB results
        f"../../baseline_results/{project_name}_results.txt",    # Two levels up
    ]
    
    baseline_results = {
        'precision': None,
        'recall': None,
        'f1': None
    }
    
    # Try to load from results.txt file first
    for path in possible_paths:
        if os.path.exists(path) and path.endswith('_results.txt'):
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    # Extract metrics from the file
                    precision_match = re.search(r'Precision:\s+([\d\.]+)', content)
                    recall_match = re.search(r'Recall:\s+([\d\.]+)', content)
                    f1_match = re.search(r'F1[-\s]?[Ss]core:\s+([\d\.]+)', content)
                    
                    if precision_match:
                        baseline_results['precision'] = float(precision_match.group(1))
                    if recall_match:
                        baseline_results['recall'] = float(recall_match.group(1))
                    if f1_match:
                        baseline_results['f1'] = float(f1_match.group(1))
                    # If we found precision and recall but not F1, calculate it
                    elif baseline_results['precision'] is not None and baseline_results['recall'] is not None:
                        p = baseline_results['precision']
                        r = baseline_results['recall']
                        # F1 = 2 * (precision * recall) / (precision + recall)
                        if p + r > 0:
                            baseline_results['f1'] = 2 * (p * r) / (p + r)
                
                print(f"Loaded baseline results from {path}")
                # Ensure all metrics have values (use fallbacks if needed)
                if baseline_results['precision'] is None:
                    print(f"Warning: No precision found in {path}, using default")
                    baseline_results['precision'] = 0.4
                if baseline_results['recall'] is None:
                    print(f"Warning: No recall found in {path}, using default")
                    baseline_results['recall'] = 0.4
                if baseline_results['f1'] is None:
                    print(f"Warning: No F1 score found in {path}, calculating from precision and recall")
                    p = baseline_results['precision']
                    r = baseline_results['recall']
                    baseline_results['f1'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.4
                
                return baseline_results
            except Exception as e:
                print(f"Error loading baseline results from {path}: {e}")
    
    # If not found, use defaults from baseline experiments
    default_baselines = {
        'tensorflow': {'precision': 0.636, 'recall': 0.723, 'f1': 0.676},
        'pytorch': {'precision': 0.321, 'recall': 0.265, 'f1': 0.290},
        'keras': {'precision': 0.482, 'recall': 0.411, 'f1': 0.443},
        'incubator-mxnet': {'precision': 0.312, 'recall': 0.250, 'f1': 0.278},
        'caffe': {'precision': 0.457, 'recall': 0.372, 'f1': 0.410}
    }
    
    if project_name in default_baselines:
        print(f"Using default baseline results for {project_name}")
        return default_baselines[project_name]
    else:
        print(f"No baseline results found for {project_name}, using default values")
        return {'precision': 0.4, 'recall': 0.4, 'f1': 0.4}

def visualize_results():
    """Create visualizations comparing enhanced and baseline performance"""
    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Data for visualization - projects to evaluate
    projects = ['tensorflow', 'pytorch', 'keras', 'incubator-mxnet', 'caffe']
    framework_names = ['TensorFlow', 'PyTorch', 'Keras', 'MXNet', 'Caffe']
    
    # Load baseline results from Lab 1
    baseline_precision = []
    baseline_recall = []
    baseline_f1 = []
    
    print("Loading baseline results from Lab 1...")
    for project in projects:
        lab1_baseline = load_baseline_results(project)
        baseline_precision.append(lab1_baseline['precision'])
        baseline_recall.append(lab1_baseline['recall'])
        baseline_f1.append(lab1_baseline['f1'])
    
    # Sample enhanced results - these should be replaced with actual results from your model
    # In a real scenario, you would load these from your results files
    enhanced_f1 = [0.5580, 0.3398, 0.4926, 0.3282, 0.4460]
    enhanced_precision = [0.6100, 0.3800, 0.5200, 0.3700, 0.5100]
    enhanced_recall = [0.5200, 0.3050, 0.4700, 0.2950, 0.3950]
    
    # Check if we have a results JSON file to load actual enhanced results
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if json_files:
        # Use the most recent JSON file
        most_recent = sorted(json_files)[-1]
        json_path = os.path.join(results_dir, most_recent)
        
        try:
            print(f"Loading results from {json_path}")
            with open(json_path, 'r') as f:
                results_data = json.load(f)
            
            # Extract enhanced results from JSON
            enhanced_precision = []
            enhanced_recall = []
            enhanced_f1 = []
            
            for project in projects:
                if project in results_data['results']:
                    enhanced_precision.append(results_data['results'][project]['enhanced']['precision_mean'])
                    enhanced_recall.append(results_data['results'][project]['enhanced']['recall_mean'])
                    enhanced_f1.append(results_data['results'][project]['enhanced']['f1_mean'])
        except Exception as e:
            print(f"Error loading JSON results: {e}")
            print("Using sample enhanced results for visualization")
    else:
        print("No JSON results found. Using sample enhanced results for visualization")
    
    # Calculate percentage changes
    f1_changes = [(enhanced - baseline) / baseline * 100 for enhanced, baseline in zip(enhanced_f1, baseline_f1)]
    
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # 1. F1 Score Comparison Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(framework_names))
    width = 0.35
    
    plt.bar(x - width/2, baseline_f1, width, label='Baseline (Lab 1)', color='skyblue')
    plt.bar(x + width/2, enhanced_f1, width, label='Enhanced', color='lightcoral')
    
    plt.xlabel('Framework')
    plt.ylabel('F1 Score')
    plt.title('Baseline vs Enhanced F1 Scores by Framework')
    plt.xticks(x, framework_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'f1_comparison.png'))
    plt.close()
    
    # 2. Performance Change Plot
    plt.figure(figsize=(10, 6))
    plt.bar(framework_names, f1_changes, color=[(0, 0.5, 0) if x >= 0 else (0.8, 0, 0) for x in f1_changes])
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Framework')
    plt.ylabel('F1 Score Change (%)')
    plt.title('Performance Change: Enhanced vs Baseline (Lab 1)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_change.png'))
    plt.close()
    
    # 3. Precision-Recall Comparison
    plt.figure(figsize=(12, 10))
    
    # Create a subplot for each framework
    for i, framework in enumerate(framework_names):
        if i >= len(baseline_precision):
            continue
            
        plt.subplot(3, 2, i+1)
        
        # Plot baseline and enhanced as points
        plt.scatter(baseline_recall[i], baseline_precision[i], s=100, color='skyblue', label='Baseline (Lab 1)')
        plt.scatter(enhanced_recall[i], enhanced_precision[i], s=100, color='lightcoral', label='Enhanced')
        
        # Connect points with an arrow
        plt.arrow(baseline_recall[i], baseline_precision[i], 
                 enhanced_recall[i] - baseline_recall[i], 
                 enhanced_precision[i] - baseline_precision[i],
                 width=0.005, head_width=0.02, head_length=0.02, 
                 length_includes_head=True, color='black', alpha=0.5)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(framework)
        plt.grid(True, alpha=0.3)
        
        # Add F1 score information as text
        plt.text(0.05, 0.05, 
                f"Baseline F1: {baseline_f1[i]:.3f}\nEnhanced F1: {enhanced_f1[i]:.3f}\nChange: {f1_changes[i]:.1f}%", 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        if i == 0:  # Only add legend for the first subplot
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'precision_recall_comparison.png'))
    plt.close()
    
    # 4. Summary metrics comparison
    plt.figure(figsize=(15, 6))
    
    # Prepare data
    metrics = ['Precision', 'Recall', 'F1 Score']
    baseline_means = [np.mean(baseline_precision), np.mean(baseline_recall), np.mean(baseline_f1)]
    enhanced_means = [np.mean(enhanced_precision), np.mean(enhanced_recall), np.mean(enhanced_f1)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, baseline_means, width, label='Baseline (Lab 1)', color='skyblue')
    plt.bar(x + width/2, enhanced_means, width, label='Enhanced', color='lightcoral')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Average Performance Metrics Across All Frameworks')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add percentage change annotations
    for i, (baseline, enhanced) in enumerate(zip(baseline_means, enhanced_means)):
        change = (enhanced - baseline) / baseline * 100
        color = 'green' if change >= 0 else 'red'
        plt.annotate(f"{change:.1f}%", 
                    xy=(i + width/2, enhanced + 0.01),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', 
                    color=color,
                    weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'summary_metrics.png'))
    plt.close()
    
    print(f"All visualizations saved to the {results_dir} directory:")
    print("  - f1_comparison.png")
    print("  - performance_change.png")
    print("  - precision_recall_comparison.png")
    print("  - summary_metrics.png")

if __name__ == "__main__":
    visualize_results() 