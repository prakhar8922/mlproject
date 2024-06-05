import sys
import os  # Add this import statement for using os.path.join
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Define paths for model and preprocessor files
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before Loading")
            
            # Load the model and preprocessor objects from their respective files
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("After Loading")
            
            # Transform the features using the preprocessor
            data_scaled = preprocessor.transform(features)
            
            # Predict the outcomes using the model
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            # Raise a custom exception in case of any errors
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, 
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        # Initialize the attributes with provided values
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary from the attributes
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary to a DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise a custom exception in case of any errors
            raise CustomException(e, sys)
