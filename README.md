# house-price-prediction
This project predicts house sale prices using the Kaggle House Prices dataset. The goal is to build machine learning models that accurately estimate the price of a house.

<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=009688&height=200&section=header&text=House%20Price%20Prediction%20🏠&fontSize=50" width="100%" />

  <p>
    <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" />
    <img src="https://img.shields.io/badge/Scikit--Learn-Latest-orange?style=flat-square&logo=scikitlearn" />
    <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-red?style=flat-square&logo=pandas" />
    <img src="https://img.shields.io/badge/Status-Completed-success?style=flat-square" />
  </p>

  <h3>Predicting real estate prices using Advanced Regression and Feature Engineering</h3>
</div>

---

## 📌 Project Overview
This project focuses on predicting house prices based on a variety of features (geographical, structural, and quality-based). The workflow involves rigorous data cleaning, handling multicollinearity via **VIF**, and implementing ensemble learning models.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-Learn, Statsmodels
*   **Models:** Random Forest Regressor, Gradient Boosting Regressor
*   **Validation:** Root Mean Squared Error (RMSE)

---

## 🚀 Key Features & Workflow

### 1. Data Preprocessing
*   **Feature Removal:** Dropped high-cardinality/missing-value columns like `Alley`, `PoolQC`, and `Fence`.
*   **Handling Multicollinearity:** Used **Variance Inflation Factor (VIF)** to identify and remove features with high redundancy (VIF > 10).
*   **Imputation:** Implemented **KNNImputer** to fill missing values based on nearest-neighbor logic rather than simple means.

### 2. Modeling & Evaluation
I compared two powerful ensemble methods to find the best fit for the dataset:

| Model | RMSE |
| :--- | :--- |
| **Gradient Boosting** | **33,019.79** |
| **Random Forest** | 33,464.79 |

> [!TIP]
> **Gradient Boosting** achieved the lowest error rate in this specific implementation, making it the primary model for final predictions.

---

## 📂 Project Structure
```bash
├── train.csv                # Training dataset
├── test.csv                 # Test dataset for predictions
├── house_price_model.ipynb  # Main Jupyter Notebook
├── submission_gb.csv        # Final Gradient Boosting predictions
└── submission_rf.csv        # Final Random Forest predictions
