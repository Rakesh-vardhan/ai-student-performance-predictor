# Student Performance Predictor

A full-stack web application that predicts a student’s academic performance and provides personalized recommendations based on course grades, attendance, CGPA, internships, and certifications. The backend uses machine learning models deployed with Flask, and the frontend is an interactive HTML/JavaScript interface.

---

Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [API Endpoints](#api-endpoints)  
- [Project Structure](#project-structure)  
- [Future Improvements](#future-improvements)  
- [Contributing](#contributing)  
- [License](#license)  

---

Overview

This project uses pre-trained Random Forest and Logistic Regression models to predict a student’s final grade range and academic risk level. It accepts inputs like grades and credits per course, attendance percentage, CGPA range, internship experience, and completed certifications. Based on the prediction, it provides actionable recommendations to help students improve.

---

Features

- Dynamic addition of multiple courses and certifications in the frontend form  
- Input validation and encoding of categorical data into numerical features  
- Scaled feature inputs for accurate machine learning prediction  
- Predicts final grade range and risk level using Random Forest and Logistic Regression  
- Provides personalized recommendations based on attendance, grades, certifications, and internships  
- REST API built with Flask serving predictions and the frontend UI  
- CORS enabled for cross-origin resource sharing during development  

---

Technologies Used

| Component         | Technology / Library             |
|-------------------|--------------------------------|
| Backend           | Python, Flask                  |
| Machine Learning  | scikit-learn (RandomForestClassifier, LogisticRegression, StandardScaler) |
| Data Handling     | pandas, numpy                  |
| Serialization     | pickle                        |
| Frontend          | HTML5, CSS3, JavaScript (Vanilla) |
| API Communication | Fetch API                     |

---

 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor

2. **Create and activate a virtual environment (optional but recommended)**

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3.Install dependencies

pip install -r requirements.txt

4.Train models (if not already trained)

The app will automatically train models if not found in the models/ directory, but you can also run:

python train_models.py

#Usage

1.Run the Flask application

python app.py

2.Open your browser and go to

http://localhost:5000/

3.Fill in your courses, attendance, CGPA, internship details, and certificates

4.Click "My Performance" to get predictions and personalized recommendations

##API Endpoints

*GET /
Serves the main frontend page.

*POST /predict
Accepts JSON data with student details and returns predicted grade range, risk level, performance category, and recommendations.

Sample /predict request JSON

{
  "courses": [
    {"grade": "81-90", "credits": 3},
    {"grade": "71-80", "credits": 4}
  ],
  "attendance": ">85",
  "cgpa": "7~8",
  "internship": "4",
  "certificates": ["5", "3"]
}
##Project Structure


├── app.py                    # Flask backend app
├── train_models.py           # Script to train and save ML models
├── models/                   # Directory containing saved ML models and scaler
├── page.html                 # Frontend HTML page
├── page.js                   # Frontend JavaScript for interaction
├── page.css                  # (Optional) CSS file for styling frontend
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation


##Future Improvements

*Add user authentication and profile management

*Store and track user predictions over time

*Visualize results with charts and graphs

*Add more ML features (attendance history, project grades)

*Improve UI responsiveness for mobile devices

*Deploy on cloud platforms (Heroku, AWS, PythonAnywhere)

Contributing
Feel free to fork the repo, make changes, and open pull requests. For major changes, please open an issue first to discuss what you would like to change.


