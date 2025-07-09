from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import pickle
import os
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_folder='.')

# Enable CORS for all routes (helps with local development)
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Handle OPTIONS requests for CORS preflight
@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    return '', 204

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Check if models exist, if not train them
if not (os.path.exists('models/random_forest_model.pkl') and 
        os.path.exists('models/logistic_regression_model.pkl')):
    from train_models import train_and_save_models
    train_and_save_models()

# Load the trained models
try:
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
        
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
        
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    rf_model = RandomForestClassifier()
    lr_model = LogisticRegression()
    scaler = StandardScaler()

@app.route('/')
def index():
    return send_from_directory('.', 'page.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received prediction request")
        data = request.json
        print(f"Request data: {data}")
        
        # Extract features
        courses = data.get('courses', [])
        attendance = data.get('attendance', '')
        cgpa = data.get('cgpa', '')
        
        # Get internship value with a default of 0
        internship_value = float(data.get('internship', 0))
        internship_type = data.get('internship', '0')  # Also keep the string value for recommendation
        
        # Get certificates and extract values
        certificate_values = []
        certificates = data.get('certificates', [])
        if isinstance(certificates, list):
            for cert in certificates:
                try:
                    certificate_values.append(float(cert))
                except (ValueError, TypeError):
                    print(f"Invalid certificate value: {cert}")
        
        certificate_value = sum(certificate_values) if certificate_values else 0
        
        # Validate input
        if not courses or not attendance or not cgpa:
            return jsonify({"error": "Missing required data"}), 400
        
        # Process course data
        total_grade_points = 0
        total_credits = 0
        
        for course in courses:
            grade_value = 0
            grade_range = course.get('grade', '')
            credit = int(course.get('credits', 0))
            
            if grade_range == "91-100": grade_value = 10    # O
            elif grade_range == "81-90": grade_value = 9    # A+
            elif grade_range == "71-80": grade_value = 8    # A
            elif grade_range == "61-70": grade_value = 7    # B+
            elif grade_range == "56-60": grade_value = 6    # B
            elif grade_range == "51-55": grade_value = 5    # C
            elif grade_range == "0-50": grade_value = 2     # F
            
            total_grade_points += grade_value * credit
            total_credits += credit
        
        avg_grade = total_grade_points / max(total_credits, 1)
        
        # Process attendance
        attendance_value = 0
        if attendance == ">95": attendance_value = 95
        elif attendance == ">85": attendance_value = 85
        elif attendance == ">75": attendance_value = 75
        elif attendance == ">65": attendance_value = 65
        elif attendance == ">50": attendance_value = 50
        
        # Process CGPA
        cgpa_value = 0
        if cgpa == "9~10": cgpa_value = 9.5
        elif cgpa == "8~9": cgpa_value = 8.5
        elif cgpa == "7~8": cgpa_value = 7.5
        elif cgpa == "6~7": cgpa_value = 6.5
        elif cgpa == "5~6": cgpa_value = 5.5
        
        # Check if the original models have the right feature dimensions
        # If not, fall back to the required features for the existing models
        if hasattr(rf_model, 'n_features_in_') and rf_model.n_features_in_ == 3:
            # Original model only uses 3 features
            features = np.array([[attendance_value, cgpa_value, avg_grade]])
            print("Using original 3-feature model")
        else:
            # Use all features including certificates and internships
            features = np.array([[
                attendance_value, 
                cgpa_value, 
                avg_grade,
                certificate_value,
                internship_value
            ]])
            print("Using extended 5-feature model")
            
        print(f"Processed features: {features}")
        
        # Scale features - handle the case where scaler expects different dimensions
        try:
            features_scaled = scaler.transform(features)
        except ValueError as e:
            print(f"Scaling error: {e}. Falling back to original features.")
            # If there's a dimension mismatch, use the first 3 features as a fallback
            features = np.array([[attendance_value, cgpa_value, avg_grade]])
            features_scaled = scaler.transform(features)
        
        # Make predictions
        grade_prediction = rf_model.predict(features_scaled)[0]
        risk_prediction = lr_model.predict(features_scaled)[0]
        
        # Convert grade prediction to grade range
        grade_ranges = ["F (Below 50%)", "C to B (50-60%)", "B to B+ (60-70%)", 
                       "B+ to A (70-80%)", "A to A+ (80-90%)", "A+ to O (90-100%)"]
        
        grade_prediction_index = min(max(int(grade_prediction), 0), len(grade_ranges) - 1)
        predicted_grade = grade_ranges[grade_prediction_index]
        
        # Convert risk prediction to risk level and performance category
        risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
        performance_categories = ["Performing Well", "Needs Improvement", "Needs Immediate Intervention"]
        
        risk_index = min(max(int(risk_prediction), 0), len(risk_levels) - 1)
        risk_level = risk_levels[risk_index]
        performance_category = performance_categories[risk_index]
        
        # Generate recommendations
        recommendations = []
        
        if attendance_value < 75:
            recommendations.append("Improve class attendance to at least 75%")
        
        if avg_grade < 6:
            recommendations.append("Focus on improving grades in current courses")
        
        if cgpa_value < 7:
            recommendations.append("Develop better study habits to improve overall CGPA")

        # FIXED: Better recommendations for internships based on selection
        if internship_value == 0:
            recommendations.append("Apply for an internship to enhance practical skills and improve employment prospects")
        elif internship_type == "4":  # Paid Internships
            recommendations.append("Aim for an internship at Fortune 500 companies or through institutional connections (SRM/IIT/NIT)")
        elif internship_type == "6":  # Small Companies
            recommendations.append("Consider upgrading your internship experience to Fortune 500 companies or through SRM/IIT/NIT programs")
            
        # FIXED: Certificate recommendations that actually show up
        if certificate_value == 0:
            recommendations.append("Obtain industry-recognized certifications to enhance your skills and resume")
        elif certificate_value < 3:
            recommendations.append("Consider higher-value certifications like CISCO, CCNA, or specialized programs through NPTEL")
        
        # Don't include certificate recommendations if user already has high-value certifications
        if certificate_value >= 5:
            recommendations = [rec for rec in recommendations if "certification" not in rec.lower()]
        
        if risk_index >= 1:
            recommendations.append("Consider seeking academic counseling or tutoring")
        
        if len(recommendations) == 0:
            recommendations.append("Continue with current academic performance")
        
        # Return prediction results
        result = {
            "predictedGrade": predicted_grade,
            "performanceCategory": performance_category,
            "riskLevel": risk_level,
            "recommendations": recommendations,
            "predictedGradeValue": float(avg_grade) * 10
        }
        
        print(f"Returning result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
