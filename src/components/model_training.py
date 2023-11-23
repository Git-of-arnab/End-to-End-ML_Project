import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.exception import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],  #take all the rows and all the columns except the last column which is the target
                train_arr[:,-1],   #take all the rows and the last column
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
            }

            #'evaluate_mode' is a function defined under 'utils.py'
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            #Get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            ## Get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] #returns the index of the key having best model score

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Suitable model found")
        
            logging.info("Best model found on both test and train dataset")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2score = r2_score(y_test,predicted)
            return r2score,best_model
    
        except Exception as e:
            raise CustomException(e,sys)
    


        