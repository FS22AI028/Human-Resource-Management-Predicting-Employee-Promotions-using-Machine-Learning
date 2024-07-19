from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load the model
model = joblib.load('project/Flask/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

def preprocess_department(department):
    # Implement LabelEncoder logic
    le = LabelEncoder()
    department_encoded = le.fit_transform([department])
    return department_encoded[0]

def preprocess_education(education):
    # Implement LabelEncoder logic
    le = LabelEncoder()
    education_encoded = le.fit_transform([education])
    return education_encoded[0]



@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        # Retrieve form data and preprocess
        department = preprocess_department(request.form['department'])
        education = preprocess_education(request.form['education'])
        no_of_trainings = int(request.form['no_of_trainings'])
        age = int(request.form['age'])
        previous_year_rating = float(request.form['previous_year_rating'])
        length_of_service = int(request.form['length_of_service'])
        KPIs_met_above_80 = int(request.form['KPIs_met_above_80'])
        awards_won = int(request.form['awards_won'])
        avg_training_score = int(request.form['avg_training_score'])

        # Convert inputs to model input format
        input_data = np.array([[department, education, no_of_trainings, age, previous_year_rating,
                                length_of_service, KPIs_met_above_80, awards_won, avg_training_score]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            prediction_text = "The employee is likely to be promoted."
        else:
            prediction_text = "The employee is not likely to be promoted."

        return render_template('result.html', prediction_text=prediction_text)
    
    # Handle GET request if needed (for direct access to /result)
    return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True)
