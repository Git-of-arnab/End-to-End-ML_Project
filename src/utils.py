import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
import dill

#dill can be used to store Python objects to a file, 
#but the primary usage is to send Python objects across the network as a byte stream
#https://dill.readthedocs.io/#:~:text=dill%20can%20be%20used%20to,network%20as%20a%20byte%20stream.

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:  #Open the file in a binary write mode and assign it to a variable 'file_obj'
            dill.dump(obj, file_obj)  #Serialize the 'pre_processing_object' and store it in 'file_obj' which is nothing but in the file at file_path

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    try:
        report ={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_params=param[list(models.keys())[i]]

            grid_cv = GridSearchCV(model,model_params,cv=3)

            grid_cv.fit(x_train,y_train)
            
            model.set_params(**grid_cv.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)