import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        """
        Method to handle the data ingestion process.
        Reads the dataset, saves it to specified paths, and splits the data into training and test sets.
        """
        logging.info("Entered the data ingestion method or component")
        try:
              # Read the dataset from the specified CSV file into a DataFrame
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info('Read the dataset as dataframe')

            # Create directories for the training data path if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Save the raw data to the specified raw data path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            # Split the data into training and test sets (80% train, 20% test)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            # Save the training set to the specified training data path
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            # Save the test set to the specified test data path
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")
            
            # Return the paths to the training and test data files
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()