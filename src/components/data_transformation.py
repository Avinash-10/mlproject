## It will be used to transform the data into one hot encoding or categorical to numerical or label encodeing etc.

import os
import sys
import numpy as np
import pandas as pd 

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object
import pickle

@dataclass
class DataTransformantionConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl') #to convert this data tranformation into a pickle file

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformantionConfig()

    def get_datatransformer_obj(self): #it is used to convert categorical to numerical or perform onehotencoder etc,etc.
        try:
            numerical_columns = ['writing score','reading score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            numerical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler(with_mean=False))
                ]
            )

            catergorical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical columns standard scaling completed')
        
            logging.info('Catergorical column encoding completed')

            preprocessor = ColumnTransformer(
                [
                ('numerical pipeline', numerical_pipeline, numerical_columns),
                ('categorical columns', catergorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info('Train Test data has been read')

            logging.info('Obtaining preprocessng data')

            preprocessing_obj = self.get_datatransformer_obj()

            target_column = 'math score'
            numerical_columns = ['writing score','reading score']

            input_feature_train_data = train_data.drop(columns=[target_column],axis=1)
            target_feature_train_data = train_data[target_column]

            input_feature_test_data = test_data.drop(columns=[target_column],axis=1)
            target_feature_test_data = test_data[target_column]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            
            input_feature_train_data_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_data_arr = preprocessing_obj.transform(input_feature_test_data)

            train_arr = np.c_[input_feature_train_data_arr, np.array(target_feature_train_data)]

            test_arr = np.c_[input_feature_test_data_arr, np.array(target_feature_test_data)]

            logging.info("Saved preprocessing object.")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)


