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
