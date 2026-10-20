# Airline Passenger Satisfaction Challenge (Active Competition)

## ⚠️ Important Compliance Notice
This repository belongs to an active Kaggle recruitment/challenge window. To strictly comply with Kaggle's anti-cheating guidelines and prevent immediate disqualification, the full executable `.ipynb` codebase has been securely backed up locally and is temporarily restricted from public hosting. 

The full repository will be made publicly available immediately upon the official closure of the competition leaderboard.

---

## 📊 Pipeline Architecture & Performance Summary

### 1. Data Preprocessing & Feature Engineering
- **Multicollinearity Mitigation:** Explicitly isolated and dropped the highly correlated component columns (`Arrival Delay` and `Departure Delay`), successfully reducing internal VIF variance inflation.
- **Categorical Synchronization:** Utilized `CategoricalDtype` mapping to dynamically unify category levels between training and test sets, ensuring zero boundary mismatch for tree splits (XGBoost, LightGBM).

### 2. Validation Strategy & Compute Partition
- **Stratified 5-Fold Cross-Validation:** Implemented global stratified splits to rigidly preserve class balances (44.4% satisfied / 55.6% dissatisfied) and ensure leak-free validation.
- **Hybrid Hardware Scaling:** To bypass cloud environment bottlenecks, structured a parallel thread-safe CPU layout for LightGBM alongside native GPU-CUDA memory allocation for XGBoost and CatBoost.

### 3. Model Performance (Out-of-Fold AUC)
- **LightGBM Model:** 0.95855
- **XGBoost Model:** 0.95875
- **CatBoost Model:** 0.95831
- **★ Final Weighted Ensemble Blend:** **0.95893** (Weights: 40% LGB / 54% XGB / 6% CAT)

*The optimal blending weights were derived via an automated out-of-fold probability grid search, ensuring maximum curve generalization on unseen data.*
