# System Requirements and Dependencies

## 1. Core Dependencies

### Python Packages
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- xgboost==2.0.1
- lightgbm==4.1.0
- nltk==3.8.1
- joblib==1.3.2
- tqdm==4.65.0

### Text Processing
- nltk==3.8.1
- scikit-learn==1.3.0

### Visualization
- matplotlib==3.7.2
- seaborn==0.12.2

### Utilities
- joblib==1.3.2
- tqdm==4.65.0

## 2. System Requirements

### Hardware
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 16GB minimum
- Storage: 5GB free space
- GPU: CUDA-capable GPU (optional, for faster processing)

### Software
- Python 3.9 or higher
- pip (Python package manager)
- Git (for version control)

## 3. Installation Instructions

### Step 1: Clone the repository
```bash
git clone https://github.com/smayuresh/Tool-Building-Project-Task-1.git
cd Tool-Building-Project-Task-1
```

### Step 2: Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Version Compatibility

### Python Version
- Minimum: Python 3.9
- Recommended: Python 3.9-3.11

### Package Versions
All package versions are specified in requirements.txt for compatibility.

## 5. Additional Notes

### Virtual Environment
- Using virtual environment is recommended
- Helps isolate project dependencies
- Prevents conflicts with system packages

### GPU Support
- Optional for faster processing
- Requires CUDA toolkit
- Compatible with NVIDIA GPUs

### Troubleshooting
- If installation fails, try updating pip:
  ```bash
  python -m pip install --upgrade pip
  ```
- For GPU issues, verify CUDA installation
- Check system requirements if performance is poor 