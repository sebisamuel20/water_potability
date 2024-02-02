import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split 
from dataclasses import dataclass 

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path : str = os.path.join("artifacts", "test.csv")
    raw_data_path : str = os.path.join("artifacts", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    

    def initiate_data_ingestion(self):
        logging.info("Starting the data ingestion process")

        try:
            df = pd.read_csv('notebook/data/water_potability_cleaned.csv')
            logging.info("File has been read as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            logging.info("Loading data into the raw_data.csv")
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Loaded data into the raw_data.csv")

            logging.info("Initiating train/test split")
            train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 69)
            logging.info("Train/test split completed")

            logging.info("Loading the train and test datasets to their respective csv files")
            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Loading the train and test datasets complete")

            logging.info("Data Ingestion is complete")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            return CustomException(e, sys)






