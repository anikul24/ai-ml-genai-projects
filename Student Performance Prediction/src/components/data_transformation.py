from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from ..exception import CustomException
from ..logger import logging
from ..utils import save_object
import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_train_path = os.path.join('artifacts', 'train.npy')
    transformed_test_path = os.path.join('artifacts', 'test.npy')
    target_encoder_path = os.path.join('artifacts', 'target_encoder.pkl')
    scaler_path = os.path.join('artifacts', 'scaler.pkl')
    ohe_path = os.path.join('artifacts', 'ohe.pkl')
    columns_path = os.path.join('artifacts', 'columns.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            Numerical_features = ['reading_score', 'writing_score']
            Categorial_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), ##for missing values
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), ##for missing values
                ('ohe', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info("Numerical and Categorical pipeline completed")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, Numerical_features), 
                ('cat_pipeline', cat_pipeline, Categorial_features)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['reading_score', 'writing_score']
            Categorial_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']


            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1) #drop target column for input features train dataset
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1) #drop target column for input features test dataset
            target_feature_test_df = test_df[target_column_name]

            logging.info("Apply preprocessing object on train and test df")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)          

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")
            logging.info("Saved transformed train and test array")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                        obj=preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)