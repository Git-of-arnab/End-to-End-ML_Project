#Pipelines will be used by the application to interact with the base code

import sys
import os
from src.exception import CustomException
from src.utils import load_object

import pandas as pd

class PredictPipeline:
	def __init__(self):
		pass
	
	def predict(self,features):
		try:
			model_path = 'artifact\model.pkl'                   #Define the path of the pickle file of the model
			preprocessor_path = 'artifact\preprocessor.pkl'     #Define the path of the pickle file of the preprocessor
			model = load_object(file_path=model_path)           #Load the model.pkl file into a variable
			preprocessor = load_object(file_path=preprocessor_path) #Load the preprocessor.pkl file into a variable
			data_scaled = preprocessor.transform(features)      #Call the preprocessor object to do feature engineering
			preds = model.predict(data_scaled)                  #Call the model object to predict on the preprocessed features
			return preds                                       #return the predictions
		
		except Exception as e:
			raise CustomException(e,sys)

class customData: #This class will be responsible for taking all the data from the html page and giving it to the back end
	def __init__(self,
				 gender:str,
				 race_ethnicity:int,
				 parental_level_of_education,
				 lunch:str,
				 test_preparation_course:str,
				 reading_score:int,
				 writing_score:int):
		self.gender = gender
		self.race_ethnicity=race_ethnicity
		self.parental_level_of_education=parental_level_of_education
		self.lunch=lunch
		self.test_preparation_course=test_preparation_course
		self.reading_score=reading_score
		self.writing_score=writing_score

	def get_data_as_data_frame(self): #this function will return the data received from our website as dataframe as we have use dataframe during training
		try:
			custom_data_input_dict={
				"gender":[self.gender],
				"race_ethnicity":[self.race_ethnicity],
				"parental_level_of_education":[self.parental_level_of_education],
				"lunch":[self.lunch],
				"test_preparation_course":[self.test_preparation_course],
				"reading_score":[self.reading_score],
				"writing_score":[self.writing_score]
			}

			return pd.DataFrame(custom_data_input_dict) #returning Dataframe
		
		except Exception as e:
			raise CustomException(e,sys)