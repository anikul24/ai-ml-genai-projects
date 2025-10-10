import os
import sys
from dataclasses import dataclass
from ..exception import CustomException
from ..logger import logging
from ..utils import save_object, evaluate_models


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression,Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Entered the data initiate_model_trainer component")
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("Models are being defined")
            models = {
                # "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                # "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "KNeighbors Regressor": KNeighborsRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_index = list(model_report.values()).index(best_model_score)
            best_model_name = list(model_report.keys())[best_model_index]

            logging.info(f"Best model found, Model name: {best_model_name}, R2 score: {best_model_score}")  


            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise CustomException("No best model found")

            logging.info("Best model is being saved")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_predicted = best_model.predict(X_test)

            r2_score_val = r2_score(y_test, y_predicted)

            return r2_score_val

        except Exception as e:
            raise CustomException(e, sys)


    