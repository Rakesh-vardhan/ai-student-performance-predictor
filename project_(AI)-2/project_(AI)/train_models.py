import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    attendance = np.random.randint(50, 100, n_samples)
    cgpa = np.random.uniform(5, 10, n_samples)
    
    num_courses = np.random.randint(3, 8, n_samples)
    avg_grades = np.array([np.mean(np.random.uniform(2, 10, n)) for n in num_courses])
    
    # Generate synthetic data for certificates and internships
    certificates = np.random.uniform(0, 10, n_samples)  # Value from 0 to 10
    internships = np.random.choice([0, 4, 6, 8, 10], n_samples)  # Values from the frontend options

    features = pd.DataFrame({
        'attendance': attendance,
        'cgpa': cgpa,
        'avg_current_grade': avg_grades,
        'certificates': certificates,
        'internships': internships
    })

    # Adjust formula to include certificates and internships
    final_grade = 0.4 * avg_grades + 0.25 * cgpa + 0.15 * (attendance / 10) + 0.1 * (certificates / 10) + 0.1 * (internships / 10)
    final_grade = np.clip(final_grade + np.random.normal(0, 0.5, n_samples), 0, 10)
    
    risk = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score = 0
        # Attendance factor
        if attendance[i] < 75:
            score += 3
        elif attendance[i] < 85:
            score += 1
            
        # CGPA factor
        if cgpa[i] < 6:
            score += 3
        elif cgpa[i] < 7:
            score += 2
        elif cgpa[i] < 8:
            score += 1
            
        # Current grades factor
        if avg_grades[i] < 5:
            score += 3
        elif avg_grades[i] < 6:
            score += 2
        elif avg_grades[i] < 7:
            score += 1
            
        # Certificate factor
        if certificates[i] < 2:
            score += 1
        
        # Internship factor
        if internships[i] == 0:
            score += 2
        elif internships[i] < 6:
            score += 1
            
        # Determine risk level
        risk[i] = 2 if score >= 6 else (1 if score >= 3 else 0)
        
    return features, final_grade, risk

def train_and_save_models():
    print("Training AI models for Student Performance Prediction...")
    
    features, final_grade, risk = generate_synthetic_data(2000)
    
    X_train, X_test, y_grade_train, y_grade_test, y_risk_train, y_risk_test = train_test_split(
        features, final_grade, risk, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_grade_cat_train = pd.cut(y_grade_train, bins=[0, 5, 6, 7, 8, 9, 10], labels=[0, 1, 2, 3, 4, 5])
    rf_model.fit(X_train_scaled, y_grade_cat_train)
    
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train_scaled, y_risk_train)
    
    # Print feature importances
    print("Feature Importances for Grade Prediction:")
    for feature, importance in zip(features.columns, rf_model.feature_importances_):
        print(f"{feature}: {importance:.4f}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    y_grade_cat_test = pd.cut(y_grade_test, bins=[0, 5, 6, 7, 8, 9, 10], labels=[0, 1, 2, 3, 4, 5])
    rf_accuracy = rf_model.score(X_test_scaled, y_grade_cat_test)
    lr_accuracy = lr_model.score(X_test_scaled, y_risk_test)
    
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

if __name__ == "__main__":
    train_and_save_models()