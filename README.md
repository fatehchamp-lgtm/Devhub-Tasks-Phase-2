# 🔮 Task 2: Telco Customer Churn Prediction Pipeline

An end-to-end professional Machine Learning production pipeline built using **Scikit-learn** for data processing/modeling and **Streamlit** for the interactive user interface. This project was developed as part of the Phase 2 tasks for the DevelopersHub Corporation Virtual Learning Program.

---

## 🚀 Project Overview
Customer churn is one of the most vital metrics for any subscription-based enterprise. This project automates the workflow of reading raw subscriber profiles, performing robust scaling and dynamic one-hot categorical encoding via Scikit-learn's `ColumnTransformer`, optimizing hyperparameters utilizing a stratified 3-Fold grid search (`GridSearchCV`), and serving the top-performing pipeline artifact directly inside a highly intuitive web layout.

### Key Highlights
- **Automated Data Intake**: Dynamic dataset ingestion from IBM's official Telco data storage with fallback to automated local CSV caching.
- **Robust Feature Engineering**: Pipeline architectures featuring `SimpleImputer` (median/most-frequent strategies), `StandardScaler`, and `OneHotEncoder(handle_unknown='ignore')` wrapped into a clean unified block.
- **Model Tuning Stack**: Parallel grid search evaluation over multiple classifier backends including **Logistic Regression** and **Random Forest Classifiers**.
- **Bulletproof Serialized Deployment**: Dynamic absolute path loading configuration inside the web framework ensuring seamless operation across directory structures.

---

## 📁 Repository Directory Architecture
```text
Task 2 - Customer Churn/
│
├── data/
│   └── Telco-Customer-Churn.csv         # Automatically cached local data repository
│
├── saved_pipeline/
│   └── customer_churn_pipeline.joblib   # Highly-optimized production pipeline weights
│
├── task2_pipeline.py                    # Complete core training, grid tuning, and export backend script
├── app.py                               # Live responsive Streamlit web layout framework
└── README.md                            # Comprehensive execution documentation manual