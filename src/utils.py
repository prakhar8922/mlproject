import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        # Extract directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create directories if they don't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode and serialize the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if an error occurs during the process
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}  # Initialize an empty dictionary to store model performance scores

        for i in range(len(list(models))):  # Iterate through each model in the models dictionary
            model = list(models.values())[i]  # Get the model object

            # Commented code for hyperparameter tuning
            para = param[list(models.keys())[i]]  # Get hyperparameters for the model (if defined)
            gs = GridSearchCV(model, para, cv=3)  # Initialize GridSearchCV for hyperparameter tuning
            gs.fit(X_train, y_train)  # Fit GridSearchCV on the training data
            model.set_params(**gs.best_params_)  # Set the best parameters found by GridSearchCV on the model

            model.fit(X_train, y_train)  # Train the model on the training data

            y_train_pred = model.predict(X_train)  # Make predictions on the training data
            y_test_pred = model.predict(X_test)  # Make predictions on the test data

            train_model_score = r2_score(y_train, y_train_pred)  # Calculate R-squared score for training data
            test_model_score = r2_score(y_test, y_test_pred)  # Calculate R-squared score for test data

            report[list(models.keys())[i]] = test_model_score  # Store the test score in the report dictionary

        return report  # Return the dictionary containing test scores for each model

    except Exception as e:
        raise CustomException(e, sys)  # Raise a custom exception in case of errors


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)