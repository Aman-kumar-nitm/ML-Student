# Basic Import
import os
import sys
import numpy as np
import pandas as pd

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.utils import evaluate_model,save_object
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Model training started")
            X_train,Y_train,X_test,Y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "KNN":KNeighborsRegressor(),
                "Adaboost":AdaBoostRegressor(),
                "SVM":SVR(),
                "Linear Regression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso()
                
            }
            param_distributions = {

            "Random Forest": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"]
            },

            "Decision Tree": {
                "criterion": ["squared_error", "friedman_mse"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },

            "KNN": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"]
            },

            "Adaboost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0],
                "loss": ["linear", "square", "exponential"]
            },

            "SVM": {
                "kernel": ["linear", "rbf", "poly"],
                "C": [0.1, 1, 10, 100],
                "epsilon": [0.01, 0.1, 0.2],
                "gamma": ["scale", "auto"]
            },

            "Linear Regression": {
             # No major hyperparameters
            "fit_intercept": [True, False]
            },

            "Ridge": {
                "alpha": [0.01, 0.1, 1.0, 10, 100],
                "solver": ["auto", "svd", "cholesky", "lsqr"]
            },

            "Lasso": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10],
                "selection": ["cyclic", "random"]
            }
            }

            model_report, trained_models = evaluate_model(
            X_train, Y_train, X_test, Y_test, models, params=param_distributions
            )
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = trained_models[best_model_name]

            
            if best_model_score<0.6:
                raise CustomException("No best Model found",sys)
            
            logging.info("Model Training Done")

            save_object(self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted=best_model.predict(X_test)
            accuracy=r2_score(Y_test,predicted)
            return accuracy
            
        except Exception as e:
            raise CustomException(e,sys)
            

