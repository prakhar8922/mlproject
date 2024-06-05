from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize the Flask application
application = Flask(__name__)

# Assign app as an alias for application
app = application

## Route for the home page
@app.route('/')
def index():
    # Render the index.html template for the home page
    return render_template('index.html')

## Route for the prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Render the home.html template when a GET request is made
        return render_template('home.html')
    else:
        # Create an instance of CustomData with form data when a POST request is made
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert the custom data to a DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        # Get predictions for the input data
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Render the home.html template with the prediction results
        return render_template('home.html', results=results[0])

# Run the Flask application
if __name__ == "__main__":
    # The application will run on the host '0.0.0.0'
    app.run(host="0.0.0.0")
