import numpy as np
import os
import sys
import time
import dill
from sklearn.metrics import r2_score

from .exception import CustomException
from .logger import logging
from sklearn.model_selection import  RandomizedSearchCV



def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)        
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}
        dt_params = {"criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
             "max_depth": [3, 5, 10, 15, 20, None],
             "min_samples_split": [2, 5, 10, 15],
             "min_samples_leaf": [1, 2, 5, 10]}
        gb_params = {"learning_rate": [0.01, 0.1, 0.2, 0.3],
             "max_depth": [3, 5, 10, 15],   
                "n_estimators": [100, 200, 300, 500],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
        lr_params = {"fit_intercept": [True, False],
                "n_jobs": [1,5,10,15,None],
                "copy_X": [True, False],
                'positive': [True,False]}
        xgb_params = {'learning_rate': [0.1, 0.01, 0.05, 0.02],
              'max_depth': [3, 5, 10, 15],
                'n_estimators': [100, 200, 300, 500],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
        catboost_params = {'depth': [3, 5, 10, 15],
                  'learning_rate': [0.1, 0.01, 0.05, 0.02],
                  'iterations': [100, 200, 300, 500],
                  'loss_function': ['RMSE', 'MAE', 'Quantile', 'LogLinQuantile', 'Poisson']}
        ada_params = {'learning_rate': [0.1, 0.01, 0.05, 0.02],
              'n_estimators': [100, 200, 300, 500],
                'loss': ['linear', 'square', 'exponential']}
        lasso_params = {'alpha': [0.1, 0.01, 0.05, 0.02],
                'selection': ['cyclic', 'random'],
                "copy_X": [True, False],
                'positive': [True,False]}
        ridge_params = {'alpha': [0.1, 0.01, 0.05, 0.02],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                "copy_X": [True, False],
                'positive': [True,False]
                }
        knn_params = {'n_neighbors': [3, 5, 7, 9, 11],
              'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        
        params = {
            # "Random Forest": rf_params,
            "Decision Tree": dt_params,
            "Gradient Boosting": gb_params,
            "Linear Regression": lr_params,
            # "XGBRegressor": xgb_params,
            # "CatBoosting Regressor": catboost_params,
            "AdaBoost Regressor": ada_params,
            "Lasso": lasso_params,
            "Ridge": ridge_params,
            "KNeighbors Regressor": knn_params
        }

        model_list = {}
        for model_name,model in models.items():
            #model = models.get(i)
            start_time = time.perf_counter()
            logging.info(f"=="*25)
            logging.info("model_name: {}".format(model_name))
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            
            logging.info(f"{model_name} Train score: {train_model_score} Test Score: {test_model_score} .. Before Hyperparameter tuning")

            logging.info(f"-"*25)
            logging.info(f"{model_name} hyper param tuning started")
            model_hyper = models[model_name]
            param = params[model_name]
            rs = RandomizedSearchCV(estimator=model_hyper,
                                    param_distributions=param,
                                    n_iter=100,
                                    n_jobs=-1,
                                    cv=3,
                                    verbose=2)
            
            rs.fit(X_train, y_train)
            model_hyper.set_params(**rs.best_params_)
            model_hyper.fit(X_train, y_train)
            y_test_pred_hyper = model_hyper.predict(X_test)
            hyper_test_model_score = r2_score(y_test, y_test_pred_hyper)
            logging.info(f"{model_name} hyper param tuning completed")
            logging.info(f"{model_name} Test Score after hyper parm tuning: {hyper_test_model_score}")
            
            logging.info(f"{model_name} R2 score: {hyper_test_model_score}")
            model_list[model_name] = hyper_test_model_score
            end_time = time.perf_counter()

            elapsed_time = (end_time - start_time)/60

            logging.info(f"Function executed in: {elapsed_time:.2f} minutes")

            logging.info(f"=="*25)

        return model_list

    except Exception as e:
        raise CustomException(e, sys)
    

def hyperparam_tuning(X_train, y_train,X_test, y_test,models):
    try:
        rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}
        dt_params = {"criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
             "max_depth": [3, 5, 10, 15, 20, None],
             "min_samples_split": [2, 5, 10, 15],
             "min_samples_leaf": [1, 2, 5, 10]}
        gb_params = {"learning_rate": [0.01, 0.1, 0.2, 0.3],
             "max_depth": [3, 5, 10, 15],   
                "n_estimators": [100, 200, 300, 500],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
        lr_params = {"fit_intercept": [True, False],
                "normalize": [True, False],
                "copy_X": [True, False]}
        xgb_params = {'learning_rate': [0.1, 0.01, 0.05, 0.02],
              'max_depth': [3, 5, 10, 15],
                'n_estimators': [100, 200, 300, 500],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
        catboost_params = {'depth': [3, 5, 10, 15],
                  'learning_rate': [0.1, 0.01, 0.05, 0.02],
                  'iterations': [100, 200, 300, 500]}
        ada_params = {'learning_rate': [0.1, 0.01, 0.05, 0.02],
              'n_estimators': [100, 200, 300, 500],
                'loss': ['linear', 'square', 'exponential']}
        lasso_params = {'alpha': [0.1, 0.01, 0.05, 0.02],
                'selection': ['cyclic', 'random']}
        ridge_params = {'alpha': [0.1, 0.01, 0.05, 0.02],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
        knn_params = {'n_neighbors': [3, 5, 7, 9, 11],
              'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        
        params = {
            "Random Forest": rf_params,
            "Decision Tree": dt_params,
            "Gradient Boosting": gb_params,
            "Linear Regression": lr_params,
            # "XGBRegressor": xgb_params,
            # "CatBoosting Regressor": catboost_params,
            "AdaBoost Regressor": ada_params,
            "Lasso": lasso_params,
            "Ridge": ridge_params,
            "KNeighbors Regressor": knn_params
        }
        report = {}

        for model_name,model in models.items():

            logging.info(f"{model_name} hyper parm tuning started")
            param = params[model_name]
            rs = RandomizedSearchCV(estimator=model,
                                    param_distributions=param,
                                    n_iter=100,
                                    n_jobs=-1,
                                    cv=3,
                                    verbose=2)
            
            rs.fit(X_train, y_train)
            logging.info(f"{model_name} best params: {rs.best_params_}")
            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            logging.info(f"{model_name} hyper parm tuning completed")
            logging.info(f"{model_name} Test Score after hyper parm tuning: {test_model_score}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)    

