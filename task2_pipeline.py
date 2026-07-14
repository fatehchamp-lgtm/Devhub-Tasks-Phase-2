import pandas as pd
import numpy as np
import os

# Scikit-learn Pipeline & Preprocessing Modules
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Model Selection, Tuning & Evaluation Metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Model Exporting
import joblib

# ==========================================================
# STRICT SYSTEM PATH RESOLUTION (No more duplicate folders!)
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(current_dir, "data")
export_directory = os.path.join(current_dir, "saved_pipeline")

os.makedirs(data_directory, exist_ok=True)
os.makedirs(export_directory, exist_ok=True)

local_data_path = os.path.join(data_directory, "Telco-Customer-Churn.csv")
export_file_path = os.path.join(export_directory, "customer_churn_pipeline.joblib")

print("=== Step 1: Loading Telco Customer Churn Dataset ===")
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(DATA_URL)
    print(f"[SUCCESS] Dataset loaded online. Shape: {df.shape}")
    df.to_csv(local_data_path, index=False)
    print(f"[INFO] Local backup cached successfully at: {local_data_path}")
except Exception as e:
    print(f"[WARN] Online fetch failed. Reading from local backup...")
    df = pd.read_csv(local_data_path)

# ==========================================================
# 2. DATA CLEANING & TYPE CASTING
# ==========================================================
print("\n=== Step 2: Cleaning and Preprocessing Data Structures ===")
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

X = df.drop(columns=['Churn'])
y = df['Churn']

numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================================
# 3. CONSTRUCTING THE PREPROCESSING PIPELINE
# ==========================================================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ==========================================================
# 4. MODEL HYPERPARAMETER TUNING
# ==========================================================
print("\n=== Step 3: Running Model Selection & Grid Search ===")
master_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=1000, random_state=42)],
        'classifier__C': [1.0]
    },
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [100],
        'classifier__max_depth': [5, 10]
    }
]

grid_search = GridSearchCV(estimator=master_pipeline, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_production_pipeline = grid_search.best_estimator_

# ==========================================================
# 5. MODEL EVALUATION & EXPORT
# ==========================================================
y_pred = best_production_pipeline.predict(X_test)
print(f"🎯 Model Accuracy Score achieved: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\n=== Step 4: Exporting Production Pipeline ===")
joblib.dump(best_production_pipeline, export_file_path)
print(f"[SUCCESS] Saved Target Path Location: {export_file_path}")