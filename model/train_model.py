"""
Breast Cancer Prediction System - Model Training Script
This script trains a Logistic Regression model on the Breast Cancer Wisconsin dataset
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BREAST CANCER PREDICTION SYSTEM - MODEL TRAINING")
print("="*60)

# 1. Load the Breast Cancer Wisconsin Dataset
print("\n1. Loading dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

print(f"   Dataset loaded successfully!")
print(f"   Shape: {df.shape}")
print(f"   Features: {df.shape[1] - 1}")
print(f"   Samples: {df.shape[0]}")

# 2. Check for missing values
print("\n2. Checking for missing values...")
missing_values = df.isnull().sum().sum()
print(f"   Total missing values: {missing_values}")

# 3. Feature Selection (5 features as required)
print("\n3. Selecting features...")
selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean concavity'
]

X = df[selected_features]
y = df['diagnosis']

print(f"   Selected features:")
for i, feature in enumerate(selected_features, 1):
    print(f"      {i}. {feature}")

# 4. Check target distribution
print("\n4. Target distribution:")
print(f"   Malignant (0): {(y == 0).sum()}")
print(f"   Benign (1):    {(y == 1).sum()}")

# 5. Train-Test Split
print("\n5. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples:  {X_test.shape[0]}")

# 6. Feature Scaling
print("\n6. Applying feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   Feature scaling completed!")

# 7. Model Training
print("\n7. Training Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
print("   Model training completed!")

# 8. Model Evaluation
print("\n8. Evaluating model...")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print("="*60)

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\n[TN  FP]")
print("[FN  TP]")
print(f"\nTrue Negatives (TN):  {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP):  {cm[1,1]}")

# 9. Save Model and Components
print("\n9. Saving model and components...")
joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'feature_names.pkl')
print("   ✓ breast_cancer_model.pkl")
print("   ✓ scaler.pkl")
print("   ✓ feature_names.pkl")

# 10. Demonstrate Model Loading
print("\n10. Demonstrating model reload...")
loaded_model = joblib.load('breast_cancer_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_features = joblib.load('feature_names.pkl')
print("   Model loaded successfully!")

# Test prediction
sample_data = X_test.iloc[0:1]
actual = y_test.iloc[0]
sample_scaled = loaded_scaler.transform(sample_data)
prediction = loaded_model.predict(sample_scaled)[0]
prediction_proba = loaded_model.predict_proba(sample_scaled)[0]

print("\n" + "="*60)
print("SAMPLE PREDICTION TEST")
print("="*60)
print(f"Actual Diagnosis:    {'Benign' if actual == 1 else 'Malignant'}")
print(f"Predicted Diagnosis: {'Benign' if prediction == 1 else 'Malignant'}")
print(f"\nPrediction Probabilities:")
print(f"  Malignant: {prediction_proba[0]:.4f} ({prediction_proba[0]*100:.2f}%)")
print(f"  Benign:    {prediction_proba[1]:.4f} ({prediction_proba[1]*100:.2f}%)")
print(f"\nMatch: {'✓ Correct' if actual == prediction else '✗ Incorrect'}")
print("="*60)

print("\n✓ Model training and saving completed successfully!")
print("\nNote: This system is strictly for educational purposes")
print("      and must not be presented as a medical diagnostic tool.")