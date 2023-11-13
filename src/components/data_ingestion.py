#Aim is to read the data from a specific data source i.e. cloud, URL, archive, local drive etc.

import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #used to create class variables

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifact","train.csv") #folder name = artifact, file name=trian.csv
    test_data_path: str=os.path.join("artifact","test.csv")
    raw_data_path: str=os.path.join("artifact","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # All the 3 paths will be saved as variable under ingestion_config
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 
            #here we getting the directory name with respect to train_data_path which is 'artifact' and creating it
            #exist_ok =True will keep the folder saved

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #save the read data into raw_data path

            logging.info("Ingested raw data stored in {}".format(self.ingestion_config.raw_data_path))
            logging.info("Train Test split initiated")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) #save the train data into train_data_path
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) #save the test data into test_data_path

            logging.info("Data Ingestion and data split is completed!")
            logging.info("Training data stored in {}".format(self.ingestion_config.train_data_path))
            logging.info("Test data stored in {}".format(self.ingestion_config.test_data_path))

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)

#test the script

if __name__ == '__main__':
    data_intake = DataIngestion()
    data_intake.initiate_data_ingestion()
