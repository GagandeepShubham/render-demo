from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

#Load the trained model
model_path = 'JS_model.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        IQ_Score = int(request.form['IQ_Score'])
        Work_Experience_Years = float(request.form['Work_Experience_Years'])
        Has_ML_Skills = int(request.form['Has_ML_Skills'])
        Has_Achievements = int(request.form['Has_Achievements'])
        Upskilling_Hours_Per_Day = float(request.form['Upskilling_Hours_Per_Day'])
        Consistency_Score = int(request.form['Consistency_Score'])

        # Validation rules
        if not (70 <= IQ_Score <= 160):
            return render_template('index.html', prediction_text="Invalid IQ value!")

        if not (0 <= Work_Experience_Years <= 50):
            return render_template('index.html', prediction_text="Invalid work experience!")

        if Has_ML_Skills not in (0, 1):
            return render_template('index.html', prediction_text="Invalid ML skills value!")

        if Has_Achievements not in (0, 1):
            return render_template('index.html', prediction_text="Invalid achievements value!")

        if not (0 <= Upskilling_Hours_Per_Day <= 24):
            return render_template('index.html', prediction_text="Invalid upskilling hours!")

        if not (1 <= Consistency_Score <= 10):
            return render_template('index.html', prediction_text="Invalid consistency score!")

        # Predict
        final_features = [[
            IQ_Score,
            Work_Experience_Years,
            Has_ML_Skills,
            Has_Achievements,
            Upskilling_Hours_Per_Day,
            Consistency_Score
        ]]

        df = pd.DataFrame(final_features, columns=[
            'IQ_Score',
            'Work_Experience_Years',
            'Has_ML_Skills',
            'Has_Achievements',
            'Upskilling_Hours_Per_Day',
            'Consistency_Score'
        ])

        prediction = model.predict(df)[0]
        output = "Yes" if prediction == 1 else "No"

        return render_template('index.html', prediction_text=f"Prediction: {output}")

    except ValueError:
        return render_template('index.html', prediction_text="Invalid input format!")

if __name__ == "__main__":
    app.run(debug=True)