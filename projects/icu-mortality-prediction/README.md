# 🏥 ICU Mortality Prediction Engine & Explainable AI

## 📖 Project Overview
This project focuses on predicting patient mortality risk within the first 24 hours of Intensive Care Unit (ICU) admission using physiological vitals and lab tests. Built as a part of a predictive healthcare initiative, it demonstrates how Machine Learning can be applied to extremely messy, highly imbalanced medical data while maintaining absolute transparency for clinical staff.

## 🎯 STAR Methodology
* Situation: Hospital networks generate massive amounts of ICU data but struggle to proactively identify high-risk patients early due to the sheer volume of vitals and lab results. Furthermore, the data is highly imbalanced (most patients survive) and contains many missing values.
* Task: Develop a robust classification engine capable of handling messy, tabular medical data while strictly preventing data leakage. Crucially, the model's decisions must be fully explainable to doctors to ensure clinical trust.
* Action: * Implemented a robust scikit-learn Pipeline with ColumnTransformer to handle imputation and One-Hot Encoding strictly post-split, effectively preventing Data Leakage.
  * Trained an XGBoost Classifier, utilizing the scale_pos_weight parameter to penalize false negatives and naturally handle the severe class imbalance.
  * Integrated SHAP (Shapley Additive exPlanations) to break the "black-box" effect, generating both global beeswarm plots (for general feature importance) and local waterfall plots (for patient-specific diagnosis).
* Result: Successfully built a reproducible, production-ready pipeline. The explainability layer allows Chief Medical Officers to see exactly which vitals (e.g., GCS motor response, Apache score) contribute to a high-risk prediction for individual patients. Evaluated strictly on ROC-AUC.

## 🛠️ Tech Stack
* Language: Python
* Data Manipulation: Pandas, NumPy
* Machine Learning: Scikit-Learn (Pipelines, Imputers), XGBoost
* Explainable AI: SHAP
* Serialization: Joblib

## 🚀 How to Run Locally

1. Clone the repository:
   `bash
   git clone [https://github.com/ssargsyann/ICU-Mortality-Prediction.git]
   cd ICU-Mortality-Prediction
