# house-price-prediction
This project predicts house sale prices using the Kaggle House Prices dataset. The goal is to build machine learning models that accurately estimate the price of a house.

# 🏠 House Prices Prediction

> Predict sales prices and practice feature engineering, RFs, and gradient boosting.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF)

---

## 📋 Pipeline

| Step | Description |
|------|-------------|
| 01 | Load & clean data — removed ID columns and dropped features with >15% missing values |
| 02 | Handle missing values — used KNN Imputer to fill remaining nulls |
| 03 | Feature encoding — applied One-Hot Encoding to all categorical columns |
| 04 | Feature selection — removed multicollinear features using VIF > 10 |
| 05 | Model training — trained Random Forest and Gradient Boosting regressors |
| 06 | Submission — generated CSV predictions for Kaggle leaderboard |

---

## 🤖 Models

### 🌲 Random Forest
- Ensemble of decision trees built independently in parallel
- Fast and robust to overfitting
- `n_estimators=100, random_state=42`

### 📈 Gradient Boosting
- Trees built sequentially, each correcting the previous
- High accuracy with careful tuning
- `n_estimators=50, learning_rate=0.1, max_depth=3`

---

## 📦 Libraries

- `pandas` — data manipulation
- `numpy` — numerical computing
- `scikit-learn` — machine learning models
- `statsmodels` — VIF calculation
- `KNNImputer` — missing value handling

---

## 📄 Output Files

- `submission_rf.csv` — Random Forest predictions
- `submission_gb.csv` — Gradient Boosting predictions

---

## 📊 Dataset

[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
