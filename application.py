from flask import Flask,render_template,request
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline 

application = Flask(__name__)
app = application

#Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race/ethnicity'),
            lunch = request.form.get('lunch'),
            parental_level_of_education = request.form.get('parental level of education'),
            test_preparation_course = request.form.get('test preparation course'),
            reading_score = request.form.get('reading score'),
            writing_score = request.form.get('writing score'),

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results = results[0])
    
if __name__ == '__main__':
    app.run(host="0.0.0.0")