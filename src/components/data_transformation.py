import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    A data class for storing configuration settings related to data transformation.
    """
    # Define a default value for the preprocessor object file path
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
         """
        Method to get the data transformation pipeline.
        Constructs and returns a preprocessing pipeline for transforming the data.
        """
         try:
             numerical_columns = ["writing_score", "reading_score"]
             categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
             # Define pipeline for numerical features
             num_pipeline=Pipeline(
                 steps=[
                     ("imputer", SimpleImputer(strategy='median')),  # Impute missing values with median
                    ("scaler", StandardScaler())  # Standardize numerical features
                 ]
             )
              # Define pipeline for categorical features
             cat_pipeline=Pipeline(

                steps=[
                 ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values with most frequent
                    ("one_hot_encoder", OneHotEncoder()),  # One-hot encode categorical features
                    ("scaler", StandardScaler(with_mean=False))  # Standardize categorical features
                
                ]

            )
             # Log information about categorical and numerical columns
             logging.info(f"Categorical columns: {categorical_columns}")
             logging.info(f"Numerical columns: {numerical_columns}")

             # Combine numerical and categorical pipelines using ColumnTransformer
             preprocessor=  ColumnTransformer(
                 [
                     ("num_pipeline",num_pipeline,numerical_columns),# Apply num_pipeline to numerical columns
                     ("cat_pipelines",cat_pipeline,categorical_columns) # Apply cat_pipeline to categorical columns
                
                 ]
             )
             return preprocessor
         except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the train and test data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Log that reading train and test data is completed
            logging.info("Read train and test data completed")

            # Log obtaining preprocessing object
            logging.info("Obtaining preprocessing object")

            # Obtain the preprocessing object using the method get_data_transformer_object
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column name and numerical columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target features for train and test datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Log applying preprocessing object on training and testing dataframes
            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Apply preprocessing object to transform input features of train and test datasets
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate input features with target features as arrays for train and test datasets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Log that preprocessing object is saved
            logging.info("Saved preprocessing object.")

            # Save preprocessing object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return transformed train and test arrays along with preprocessing object file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Raise a custom exception if an error occurs during the process
            raise CustomException(e, sys)
