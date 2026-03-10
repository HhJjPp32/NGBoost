# XGBoost ML Pipeline for CFDST Column Strength Prediction

This project provides a complete, modular machine learning pipeline for predicting the ultimate bearing capacity (Nexp) of Concrete-Filled Double Skin Tubular (CFDST) columns using XGBoost.

## Features

- **Data Loading & Preprocessing**: Automated CSV loading, column cleaning, missing value imputation
- **XGBoost Model Training**: With optional Optuna hyperparameter optimization
- **Comprehensive Evaluation**: R², RMSE, MAE, MAPE, and **COV (Coefficient of Variation)**
- **Visualization**: Prediction scatter plots, feature importance, residual analysis
- **Feature Selection**: Recursive Feature Elimination (RFE) pipeline
- **Modular Design**: Clean separation of concerns with type hints and logging

## Project Structure

```
xgboost_project/
├── config/
│   └── config.yaml          # Core configuration file
├── data/                    # Data directory (place your CSV here)
├── logs/                    # Optuna database and best_params.json
├── output/                  # Models, plots, and reports
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # CSV loading and validation
│   ├── preprocessor.py      # Data preprocessing pipeline
│   ├── model_trainer.py     # XGBoost + Optuna training
│   ├── evaluator.py         # Evaluation metrics (including COV)
│   ├── visualizer.py        # Plotting functions
│   ├── predictor.py         # Inference/prediction logic
│   └── utils.py             # Utilities and logging
├── train.py                 # Main training script
├── predict.py               # Main prediction script
├── feature_selection.py     # RFE feature selection script
└── requirements.txt         # Python dependencies
```

## Installation

```bash
# Clone or download the project
cd xgboost_project

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Place your CSV data file in the `data/` directory. The CSV should contain:
- Features: Dimensionless parameters (e.g., ratios, normalized values)
- Target: `Nexp` column (ultimate bearing capacity)

Columns to exclude (dimensional parameters) can be configured in `config/config.yaml` under `drop_columns`.

### 2. Training

#### Basic Training (without Optuna optimization)
```bash
python train.py --no-optimize
```

#### Training with Optuna Hyperparameter Optimization
```bash
# Run with default number of trials (100)
python train.py --optimize

# Specify custom number of trials
python train.py --optimize --n-trials 200

# Use custom data file
python train.py --optimize --data data/my_data.csv
```

#### Training Outputs
- Model: `output/xgboost_model.pkl`
- Best Parameters: `logs/best_params.json`
- Optuna Study: `logs/optuna_study.db`
- Plots: `output/*.png`
- Metrics: `output/metrics.txt`
- Feature List: `output/feature_names.txt`

### 3. Feature Selection (RFE)

Run recursive feature elimination to find the optimal feature subset:

```bash
# Run with default settings
python feature_selection.py

# Custom minimum features
python feature_selection.py --min-features 5

# Custom output directory
python feature_selection.py --output output/my_selection
```

Outputs:
- `feature_selection_results.json`: Detailed results for each iteration
- `best_features_cov.txt`: Best feature set based on COV
- `best_features_r2.txt`: Best feature set based on R²
- `feature_selection_report.txt`: Summary report
- `feature_selection_results.png`: Visualization

### 4. Making Predictions

```bash
# Predict on new data
python predict.py --input data/new_data.csv --output predictions.csv

# With custom config
python predict.py --input data/new_data.csv --config config/config.yaml
```

## Configuration

Edit `config/config.yaml` to customize the pipeline:

### Key Settings

```yaml
# Target column
target:
  name: "Nexp"

# Columns to drop (dimensional parameters)
drop_columns:
  - "b"
  - "h"
  - "r0"
  - "t"
  - "L"
  - "e"

# Preprocessing
preprocessing:
  imputation_strategy: "median"  # median, mean, most_frequent
  scale_features: false          # XGBoost doesn't require scaling
  test_size: 0.2

# Optuna optimization
optuna:
  use_optuna: true
  n_trials: 100
  timeout: 3600

# Preset XGBoost parameters
xgboost_params:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.1
  # ... more parameters

# Cross-validation
cross_validation:
  n_splits: 5
```

## Core Metric: COV (Coefficient of Variation)

COV is the key engineering metric for this project:

$$
COV = \frac{\sigma_{\xi}}{\mu_{\xi}}, \quad \text{where } \xi = \frac{y_{pred}}{y_{true}}
$$

- **Lower COV is better** (ideally < 0.1)
- Measures the consistency of predictions relative to actual values
- Critical for engineering applications

## Module Reference

### `src/data_loader.py`
- `DataLoader`: Handles CSV loading, column cleaning, validation

### `src/preprocessor.py`
- `DataPreprocessor`: Missing value imputation, train/test splitting

### `src/model_trainer.py`
- `XGBoostTrainer`: Training with/without Optuna, model persistence

### `src/evaluator.py`
- `ModelEvaluator`: Metrics calculation (R², RMSE, MAE, MAPE, COV)

### `src/visualizer.py`
- `ModelVisualizer`: Prediction plots, feature importance, residuals

### `src/predictor.py`
- `Predictor`: Inference on new data

### `src/utils.py`
- Logging setup, config loading, utility functions

## Example Usage in Python

```python
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import XGBoostTrainer
from src.evaluator import ModelEvaluator
from src.utils import load_config

# Load config
config = load_config("config/config.yaml")

# Load and preprocess data
data_loader = DataLoader(config)
df, feature_cols = data_loader.load_and_prepare()

preprocessor = DataPreprocessor(config)
df_processed, feature_cols = preprocessor.preprocess(df, feature_cols)
X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed, feature_cols)

# Train model
trainer = XGBoostTrainer(config)
model = trainer.train(X_train, y_train, X_test, y_test)

# Evaluate
evaluator = ModelEvaluator()
y_pred = trainer.predict(X_test)
metrics = evaluator.evaluate(y_test, y_pred, "Test")

print(f"COV: {metrics['COV']:.6f}")
```

## Requirements

- Python 3.8+
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- optuna >= 3.4.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pyyaml >= 6.0

## Troubleshooting

### Issue: Model file not found during prediction
**Solution**: Run `python train.py` first to train and save the model.

### Issue: Missing columns error
**Solution**: Check that your CSV contains all required columns. Update `drop_columns` in config.yaml if needed.

### Issue: Out of memory during Optuna optimization
**Solution**: Reduce `n_trials` in config or use command line: `python train.py --optimize --n-trials 50`

## License

This project is provided for research and educational purposes.

## Citation

If you use this code in your research, please cite:

```
XGBoost ML Pipeline for CFDST Column Strength Prediction
Based on: Zhenlin Li et al. (2022)
```
