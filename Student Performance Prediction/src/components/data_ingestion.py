import os
import sys
from ..exception import CustomException
from ..logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from .data_transformation import DataTransformation
from .model_trainer import ModelTrainerConfig 
from .model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv') ##read from csv , you can change it to your path, can be mongodb/snowflake, API etc
            logging.info("Read CSV file in datframe")


            #os.mkdir(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            ##creating artifacts folder
            logging.info("Created artifacts folder if not exists")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            ##save raw data into raw_data_path
            logging.info("Saved Raw data")


            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)


            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(train_data, test_data)

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    
