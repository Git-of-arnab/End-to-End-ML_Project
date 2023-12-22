#We will be using Flask
import flask
from flask import Flask,request,render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import customData,PredictPipeline

application = Flask(__name__)

app = application

##ROute for our homepage

@app.route('/')
def index():
    return render_template('index.html')

#when we use 'render_template' it will go and search for a 'templates' folder

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html') #home.html will have the input entry which we need to give as parameter for the prediction
    else:
        data = customData(
            gender = request.form.get('gender'),
            race_ethnicity= request.form.get('ethnicity'),
            parental_level_of_education= request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')

        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df) #The output will be returned in list format

        return render_template('home.html',results=results[0]) #The results value needs to be present in home.html for rendering
    
##Test-run

if __name__=='__main__':
    #app.run(host="0.0.0.0",debug=True)
    app.run(host="0.0.0.0")  #Removing debug=True for deploying on AWS