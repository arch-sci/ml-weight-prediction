# Predictive Modeling of Adult Weight — NHIS Dataset  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arch-sci/ml-weight-prediction/blob/main/ml-weight-prediction.ipynb)

**Machine Learning course project | EDHEC Business School**


Predicting adult body weight from 600+ health and demographic survey features 
(NHIS dataset), with a focus on robust preprocessing and systematic model comparison.

## Problem

Survey data is messy by design : the NHIS dataset encodes missing or refused 
answers as specific integer codes (7, 8, 9, 97, 99...) that corrupt numerical 
analysis if treated as valid values. The challenge was as much about building a 
reliable preprocessing pipeline as about tuning models.

## Approach

### Custom Preprocessing
- **GapJunkTransformer** : custom sklearn transformer that detects survey junk 
  codes by identifying numerical gaps in feature distributions, then replaces 
  them with NaN — fitted on train folds only to prevent data leakage
- **LowNullColumnSelector** : drops features with >50% missing values per fold
- Median/mode imputation, target encoding for high-cardinality categoricals, 
  OneHotEncoding with min_frequency=1% to control dimensionality

### Model Benchmark
10 models compared within a cross-validation pipeline :

| Model | Test MSE |
|---|---|
| GradientBoosting | 213.00 |
| **XGBoost** | **213.26** |
| RandomForest | 216.80 |
| Stacking (Lasso + XGB) | 220.40 |
| Physics Benchmark (Height, BMI, Age, Sex) | 239.72 |
| Ridge / Lasso | ~308 |

### Key findings
- The relationship between survey features and weight is fundamentally 
  non-linear — linear models plateaued ~17.5 lbs RMSE despite regularization
- The physics benchmark (4 features, polynomial) outperformed all linear models,
  suggesting the dataset has a noise floor driven by BMI and Height
- Stacking did not improve over standalone XGBoost, confirming the linear 
  component was redundant

## Stack
Python · pandas · scikit-learn · XGBoost · numpy · matplotlib

## Results
XGBoost selected for production : Test MSE ~213 (RMSE ~14.6 lbs), 
stable across random seeds.
