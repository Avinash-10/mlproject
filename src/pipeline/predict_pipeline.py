## It is used to predict the model

import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):  
        try:
                                                                 #this function will help to predict our whole model
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'     #resposible to handle categorical feature,featur scaling etc
            model = load_object(file_path = model_path)          #it will just load the pickle file
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
        

#it is responible to map the web app or front end data to backend 

class CustomData:
    def __init__(self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education:str,
        lunch:str,
        test_preparation_course:str,
        reading_score:str,
        writing_score:str):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self): #it will give the data in dataframe
        try:
            custom_data_input_dict =  {
                "gender" : [self.gender],
                "race/ethnicity" : [self.race_ethnicity],
                "parental level of education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test preparation course" : [self.test_preparation_course],
                "reading score" : [self.reading_score],
                "writing score" : [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)


        
        

