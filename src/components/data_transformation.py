import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer #For missing values in data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTranformationConfig:
    pre_processor_obj_file_path = os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTranformationConfig()
    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation for numerical and categorical datas
        '''
        try:
            numerical_features = ['reading_score', 'writing_score'] #We already have these from the EDA file
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')), #first it will fill the missing values
                    ("scaler",StandardScaler(with_mean=False)) #then scale it
                    #It is possible to disable either centering 
                    #or scaling by either passing with_mean=False or with_std=False to the constructor of StandardScaler
                    #Did this as there was error saying cannot center sparse matrices
                ]
            )

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical column encoding and standard scaling completed")

            #Combine the above 2 pipelines

            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_features),
                    ("cat_pipeline",categorical_pipeline,categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed for proceeding to transformation")
            logging.info("reading preprocessor object")

            pre_processing_object =self.get_data_transformer_obj()
            target_column_name ="math_score"
            numerical_features = [ 'reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            input_feature_train_df = train_df.drop([target_column_name],axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df = test_df.drop([target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test data")

            input_feature_train_arr=pre_processing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=pre_processing_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] #concatenating processed input with target
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Pre-processing is complete")
            logging.info("Saving the data into pickle file in {}".format(self.data_transformation_config.pre_processor_obj_file_path))

            #We will save the pickle file using the below function which will be defined in utils.py
            save_object(
                file_path = self.data_transformation_config.pre_processor_obj_file_path,
                obj=pre_processing_object
            )

            return(
                train_arr,test_arr,self.data_transformation_config.pre_processor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)





