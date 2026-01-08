# IEEE-CIS Banking Fraud Detection Showcase

A raud detection showcase demonstrating GPU-accelerated machine learning for financial transaction monitoring. This project implements industry best practices for fraud modeling, including operational decisioning frameworks, cost-based optimization, and CPU vs. GPU performance benchmarking.

## Project Overview

This showcase demonstrates:

- **Faud detection** using the IEEE-CIS Fraud Detection dataset
- **Memory-efficient data processing** with Polars streaming (handles datasets larger than RAM)
- **Production-ready feature engineering** pipeline with frequency encoding and temporal features
- **Operational decisioning**: Alert-rate and cost-based threshold optimization
- **GPU acceleration**: Quantified speedup using PyTorch with CUDA
- **Business-aligned metrics**: ROC-AUC, Precision-Recall, cost curves, and confusion matrices

## Key Results

- **Dataset**: 590K+ transactions with 400+ features
- **Fraud Rate**: ~3.5% (severe class imbalance)
- **Baseline Model**: SGDClassifier achieves 0.95+ ROC-AUC
- **Neural Model**: 4-layer MLP with dropout regularization
- **GPU Speedup**: 5-20x faster training depending on hardware
- **Cost Optimization**: Minimizes FN ($100) vs. FP ($1) expected cost

## Architecture

```
├── data/                          # Data directory (excluded from Git)
│   ├── train_transaction.csv     # Transaction features
│   ├── train_identity.csv        # Identity features
│   ├── train_joined.parquet      # Joined dataset (cached)
│   └── train_features.parquet    # Engineered features (cached)
├── ieee_cis_banking_showcase_complex.ipynb  # Main analysis notebook
├── requirements.txt               # Python dependencies
├── data.zip                       # Compressed dataset archive
└── README.md                      # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- 16GB+ RAM (32GB recommended for full dataset)
- NVIDIA GPU with CUDA support (optional, for GPU benchmarking)
- Conda or virtualenv for environment management

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd CAI_GPU_FraudDetection
```

2. **Create and activate environment**:

```bash
conda create -n fraud_detection python=3.11 -y
conda activate fraud_detection
```

3. **Install dependencies**:

```bash
# For CPU-only:
pip install -r requirements.txt

# For GPU (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

4. **Extract dataset**:

```bash
# Data is included in data.zip
# The notebook will automatically extract it on first run
```

5. **Launch Jupyter**:

```bash
jupyter lab
```

6. **Open and run**: `ieee_cis_banking_showcase_complex.ipynb`

## Dataset

**Source**: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle)

**Files**:

- `train_transaction.csv`: Transaction-level features (amount, time, card info, device)
- `train_identity.csv`: Identity-level features (browser, OS, email domains, addresses)

**Note**: The dataset is included as `data.zip`. The notebook automatically extracts and processes it.

## Methodology

### 1. Data Quality Checks

- Duplicate detection
- Missingness analysis (null rates by column)
- Cardinality profiling for categorical features
- Outlier detection and clipping

### 2. Exploratory Data Analysis

- Fraud rate by time-of-day (circadian patterns)
- Fraud rate by transaction amount (decile stratification)
- Device type risk profiling
- Email domain risk analysis

### 3. Feature Engineering

- **Temporal features**: Day index, hour-of-day
- **Amount transformations**: Log1p scaling, quantile binning
- **Frequency encoding**: Category occurrence counts
- **Interaction features**: Amount × Hour, Amount ÷ Day
- **V columns**: 250 Vesta-engineered fraud signals

### 4. Modeling

- **Baseline**: SGDClassifier (logistic regression via SGD)
- **Neural Model**: 4-layer MLP with dropout (4096 → 2048 → 1024 → 2)
- **Optimization**: AdamW with learning rate 1e-3
- **Regularization**: Dropout (0.1) and early stopping

### 5. Decisioning

- **Alert-rate threshold**: Flag top X% for manual review
- **Cost-based threshold**: Minimize expected cost given FN/FP costs
- **Metrics**: ROC-AUC, PR-AUC, Precision, Recall, Confusion Matrix

### 6. GPU Benchmarking

- CPU vs. GPU training time comparison
- Controlled experiment (same data, model, hyperparameters)
- Mixed precision (AMP) for GPU acceleration
- Synchronized timing for accuracy

## Performance Metrics

### Model Performance

| Model        | ROC-AUC | PR-AUC | Training Time |
| ------------ | ------- | ------ | ------------- |
| Baseline SGD | 0.95+   | 0.75+  | ~30s          |
| MLP (CPU)    | 0.96+   | 0.78+  | ~180s         |
| MLP (GPU)    | 0.96+   | 0.78+  | ~15s          |

### Operational Metrics (Alert Rate = 1%)

| Metric     | Value  |
| ---------- | ------ |
| Precision  | 35-40% |
| Recall     | 10-15% |
| Alert Rate | 1.0%   |

### Cost-Based Threshold (FN=$100, FP=$1)

- Optimal threshold minimizes total expected cost
- Balances fraud losses against review expenses
- Typical precision: 25-30%, Recall: 20-25%

## Configuration

### Memory Settings

The notebook automatically detects memory constraints:

- Polars streaming for datasets larger than RAM
- Batch size adjusted based on available memory
- DataLoader `num_workers=0` for limited shared memory

### GPU Settings

- Automatic CUDA detection
- Falls back to CPU if GPU unavailable
- Mixed precision (FP16) for 2x speedup on modern GPUs

## Business Value

### Fraud Prevention

- **Alert prioritization**: Focus analyst time on highest-risk transactions
- **Dynamic thresholds**: Adjust based on capacity and fraud trends
- **Cost optimization**: Quantify ROI of fraud prevention programs

### GPU ROI

- **Faster iteration**: 5-20x speedup enables more experiments
- **Real-time scoring**: High throughput for batch processing
- **Time to market**: Accelerate model development and deployment

## Contributing

This is a showcase project. For questions or collaboration:

1. Open an issue describing your suggestion
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Dataset**: Vesta Corporation and IEEE Computational Intelligence Society
- **Competition**: [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
- **Inspiration**: Production fraud systems at major financial institutions

**Note**: This is a demonstration project using public competition data. Real-world fraud detection systems involve additional considerations including regulatory compliance, privacy controls, and operational integration.
